"""Post-hoc, GT-local evaluation for the two blind MLLM protocols.

This module never participates in an MLLM request.  It joins aggregate result
rows to the review export only after inference has finished, then writes
protocol-specific per-image tables and summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GroundTruth:
    id: str
    task_id: str
    label: str  # edited | not_edited
    edit_region_xyxy: list[int] | None


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_dump(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def review_ground_truth(path: Path, status: str = "good", include_source_pairs: bool = True) -> list[GroundTruth]:
    """Mirror ``inputs.from_review_export`` ordering and real-image de-duplication."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("review export must contain a records array")
    output: list[GroundTruth] = []
    seen_sources: set[str] = set()
    for row in records:
        if row.get("status") != status:
            continue
        task_id = str(row["task_id"])
        box = row.get("edit_region_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
            raise ValueError(f"{task_id}: missing or invalid edit_region_xyxy")
        output.append(GroundTruth(f"{task_id}__forged", task_id, "edited", box))
        source = str(row.get("source_image", ""))
        if include_source_pairs and source not in seen_sources:
            output.append(GroundTruth(f"{task_id}__real", task_id, "not_edited", None))
            seen_sources.add(source)
    if not output:
        raise ValueError(f"no review records with status={status!r}")
    return output


def _load_results(path: Path, protocol_version: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the last valid result for every (image, protocol) key.

    Incomplete rows are retained in the source JSONL for coverage auditing but
    never replace a valid three-replicate aggregate.
    """
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        protocol = row.get("protocol_key") or row.get("protocol_id")
        if protocol not in {"detection", "localization"}:
            continue
        if protocol_version is not None and row.get("protocol_version") != protocol_version:
            continue
        key = (str(row.get("id")), protocol)
        row["_result_line"] = line_number
        if row.get("status") == "ok" and row.get("valid_for_metrics") is True:
            rows[key] = row
        elif key not in rows:
            rows[key] = row
    return rows


def _run_metadata(results_path: Path, results: dict[tuple[str, str], dict[str, Any]], run_manifest_path: Path | None) -> dict[str, Any]:
    """Flatten run metadata for human-readable JSON/CSV summaries without secrets."""
    manifest_path = run_manifest_path or results_path.with_suffix(".run_manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exemplar = next(iter(results.values()), {})
    model = manifest.get("model", {})
    source = manifest.get("input", {})
    api = manifest.get("api", {})
    retry = manifest.get("retry", {})
    image = manifest.get("image", {})
    return {
        "run_id": manifest.get("run_id") or exemplar.get("run_id") or results_path.stem,
        "condition": manifest.get("condition") or exemplar.get("condition"),
        "model_id": model.get("id") or exemplar.get("model"),
        "model_slug": model.get("slug") or exemplar.get("model_slug"),
        "protocol_version": manifest.get("protocol", {}).get("version") or exemplar.get("protocol_version"),
        "input_source": source.get("source"),
        "input_manifest_sha256": source.get("manifest_sha256") or exemplar.get("input_manifest_sha256"),
        "input_images": source.get("images"),
        "review_export": source.get("review_export"),
        "review_status": source.get("review_status"),
        "include_source_pairs": source.get("include_source_pairs"),
        "max_tokens": model.get("max_tokens"),
        "temperature": model.get("temperature"),
        "concurrency": model.get("concurrency"),
        "request_format": model.get("request_format"),
        "api_timeout_seconds": api.get("timeout_seconds"),
        "retry_max_retries_per_replicate": retry.get("max_retries_per_replicate"),
        "retry_base_backoff_seconds": json.dumps(retry.get("base_backoff_seconds"), ensure_ascii=False) if retry.get("base_backoff_seconds") is not None else None,
        "image_transport": image.get("transport") or exemplar.get("image_transport"),
        "image_detail": image.get("detail"),
        "image_max_long_side": image.get("maxLongSide"),
        "config_fingerprint_sha256": manifest.get("config_fingerprint_sha256") or exemplar.get("config_fingerprint_sha256"),
        "run_manifest_path": str(manifest_path) if manifest_path.is_file() else exemplar.get("run_manifest_path"),
        "aggregate_results_path": str(results_path),
    }


def _auc_roc(rows: list[dict[str, Any]]) -> float | None:
    positives = sum(row["gt_edited"] for row in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(rows, key=lambda row: row["model_score"], reverse=True)
    points = [(0.0, 0.0)]
    tp = fp = 0
    index = 0
    while index < len(ranked):
        score = ranked[index]["model_score"]
        group: list[dict[str, Any]] = []
        while index < len(ranked) and ranked[index]["model_score"] == score:
            group.append(ranked[index]); index += 1
        tp += sum(row["gt_edited"] for row in group)
        fp += len(group) - sum(row["gt_edited"] for row in group)
        points.append((fp / negatives, tp / positives))
    return sum((points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]) / 2 for i in range(1, len(points)))


def _average_precision(rows: list[dict[str, Any]]) -> float | None:
    positives = sum(row["gt_edited"] for row in rows)
    if not positives:
        return None
    ranked = sorted(rows, key=lambda row: row["model_score"], reverse=True)
    tp = fp = 0
    ap = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index]["model_score"]
        group: list[dict[str, Any]] = []
        while index < len(ranked) and ranked[index]["model_score"] == score:
            group.append(ranked[index]); index += 1
        group_tp = sum(row["gt_edited"] for row in group)
        tp += group_tp
        fp += len(group) - group_tp
        if group_tp:
            ap += (group_tp / positives) * (tp / (tp + fp))
    return ap


def _boxes(row: dict[str, Any] | None) -> list[list[float]]:
    if not row:
        return []
    values = row.get("regions_px") or []
    boxes: list[list[float]] = []
    for value in values:
        if isinstance(value, list) and len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
            x1, y1, x2, y2 = map(float, value)
            if x1 < x2 and y1 < y2:
                boxes.append([x1, y1, x2, y2])
    return boxes


def _overlaps(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _inside(inner: list[float], outer: list[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "ok" and row.get("valid_for_metrics") is True)


def evaluate_review_export(
    results_path: Path,
    review_export: Path,
    output_dir: Path,
    *,
    status: str = "good",
    include_source_pairs: bool = True,
    protocol_version: str | None = None,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write separate detection/localization per-image tables and summaries."""
    gt_rows = review_ground_truth(review_export, status, include_source_pairs)
    results = _load_results(results_path, protocol_version)
    run_metadata = _run_metadata(results_path, results, run_manifest_path)

    detection_rows: list[dict[str, Any]] = []
    for gt in gt_rows:
        result = results.get((gt.id, "detection"))
        is_valid = _valid(result)
        predicted = result.get("decision") if is_valid else None
        detection_rows.append({
            "run_id": run_metadata["run_id"],
            "condition": run_metadata["condition"],
            "model_id": run_metadata["model_id"],
            "model_slug": run_metadata["model_slug"],
            "id": gt.id,
            "task_id": gt.task_id,
            "gt_label": gt.label,
            "gt_edited": gt.label == "edited",
            "result_status": result.get("status") if result else "missing_result",
            "valid_for_metrics": is_valid,
            "model_decision": predicted,
            "model_score": result.get("score") if is_valid else None,
            "model_p_ai_edited": result.get("p_ai_edited") if is_valid else None,
            "model_aggregate": result.get("result") if is_valid else None,
            "is_correct": (predicted == gt.label) if is_valid else None,
            "result_line": result.get("_result_line") if result else None,
        })
    valid_detection = [row for row in detection_rows if row["valid_for_metrics"]]
    tp = sum(row["gt_edited"] and row["model_decision"] == "edited" for row in valid_detection)
    tn = sum(not row["gt_edited"] and row["model_decision"] == "not_edited" for row in valid_detection)
    fp = sum(not row["gt_edited"] and row["model_decision"] == "edited" for row in valid_detection)
    fn = sum(row["gt_edited"] and row["model_decision"] == "not_edited" for row in valid_detection)
    detection_summary: dict[str, Any] = {
        **run_metadata,
        "protocol": "detection",
        "expected_images": len(detection_rows),
        "valid_images": len(valid_detection),
        "coverage": _safe_div(len(valid_detection), len(detection_rows)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, len(valid_detection)),
        "precision": _safe_div(tp, tp + fp),
        "recall_tpr": _safe_div(tp, tp + fn),
        "specificity_tnr": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": None,
        "auroc": _auc_roc(valid_detection),
        "average_precision": _average_precision(valid_detection),
    }
    if detection_summary["recall_tpr"] is not None and detection_summary["specificity_tnr"] is not None:
        detection_summary["balanced_accuracy"] = (detection_summary["recall_tpr"] + detection_summary["specificity_tnr"]) / 2

    localization_rows: list[dict[str, Any]] = []
    for gt in gt_rows:
        result = results.get((gt.id, "localization"))
        is_valid = _valid(result)
        boxes = _boxes(result) if is_valid else []
        if gt.edit_region_xyxy is not None:
            gt_box = list(map(float, gt.edit_region_xyxy))
            any_overlap = any(_overlaps(box, gt_box) for box in boxes)
            all_inside = bool(boxes) and all(_inside(box, gt_box) for box in boxes)
            real_rejection_correct = None
        else:
            gt_box = None
            any_overlap = None
            all_inside = None
            real_rejection_correct = is_valid and result.get("decision") == "no_localized_edit" and not boxes
        localization_rows.append({
            "run_id": run_metadata["run_id"],
            "condition": run_metadata["condition"],
            "model_id": run_metadata["model_id"],
            "model_slug": run_metadata["model_slug"],
            "id": gt.id,
            "task_id": gt.task_id,
            "gt_label": gt.label,
            "gt_edit_region_xyxy": gt.edit_region_xyxy,
            "result_status": result.get("status") if result else "missing_result",
            "valid_for_metrics": is_valid,
            "model_decision": result.get("decision") if is_valid else None,
            "predicted_regions_xyxy": boxes,
            "model_aggregate": result.get("result") if is_valid else None,
            "has_predicted_region": bool(boxes) if is_valid else None,
            "any_box_overlaps_gt": any_overlap,
            "all_predicted_boxes_inside_gt": all_inside,
            "real_no_edit_correct": real_rejection_correct,
            "result_line": result.get("_result_line") if result else None,
        })
    forged_localization = [row for row in localization_rows if row["gt_label"] == "edited"]
    valid_forged = [row for row in forged_localization if row["valid_for_metrics"]]
    real_localization = [row for row in localization_rows if row["gt_label"] == "not_edited"]
    valid_real = [row for row in real_localization if row["valid_for_metrics"]]
    overlap_success = sum(bool(row["any_box_overlaps_gt"]) for row in valid_forged)
    contained_success = sum(bool(row["all_predicted_boxes_inside_gt"]) for row in valid_forged)
    real_rejection_success = sum(bool(row["real_no_edit_correct"]) for row in valid_real)
    localization_summary: dict[str, Any] = {
        **run_metadata,
        "protocol": "localization",
        "expected_images": len(localization_rows),
        "valid_images": len([row for row in localization_rows if row["valid_for_metrics"]]),
        "coverage": _safe_div(sum(row["valid_for_metrics"] for row in localization_rows), len(localization_rows)),
        "forged_expected": len(forged_localization),
        "forged_valid": len(valid_forged),
        "forged_coverage": _safe_div(len(valid_forged), len(forged_localization)),
        "box_overlap_successes": overlap_success,
        "box_overlap_accuracy": _safe_div(overlap_success, len(valid_forged)),
        "all_boxes_inside_gt_successes": contained_success,
        "all_boxes_inside_gt_accuracy": _safe_div(contained_success, len(valid_forged)),
        "real_expected": len(real_localization),
        "real_valid": len(valid_real),
        "real_coverage": _safe_div(len(valid_real), len(real_localization)),
        "real_no_edit_successes": real_rejection_success,
        "real_no_edit_accuracy": _safe_div(real_rejection_success, len(valid_real)),
    }

    _jsonl_dump(output_dir / "detection_per_image.jsonl", detection_rows)
    _json_dump(output_dir / "detection_metrics.json", detection_summary)
    _csv_dump(output_dir / "detection_metrics.csv", detection_summary)
    _jsonl_dump(output_dir / "localization_per_image.jsonl", localization_rows)
    _json_dump(output_dir / "localization_metrics.json", localization_summary)
    _csv_dump(output_dir / "localization_metrics.csv", localization_summary)
    return {"detection": detection_summary, "localization": localization_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blind MLLM results against the local review export")
    parser.add_argument("--results", type=Path, required=True, help="aggregate <condition>.jsonl from one model")
    parser.add_argument("--review-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-status", default="good")
    parser.add_argument("--no-source-pairs", action="store_true")
    parser.add_argument("--protocol-version")
    parser.add_argument("--run-manifest", type=Path, help="optional secret-free <run_id>.run_manifest.json")
    args = parser.parse_args()
    summary = evaluate_review_export(
        args.results, args.review_export, args.output_dir,
        status=args.review_status,
        include_source_pairs=not args.no_source_pairs,
        protocol_version=args.protocol_version,
        run_manifest_path=args.run_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
