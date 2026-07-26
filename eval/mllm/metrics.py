"""Post-hoc, GT-local evaluation for the two blind MLLM protocols.

This module never participates in an MLLM request.  It joins aggregate result
rows to the review export only after inference has finished, then writes
protocol-specific per-image tables and summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .results import ProtocolVersionSelector, protocol_version_matches


@dataclass(frozen=True)
class GroundTruth:
    id: str
    task_id: str
    label: str  # edited | not_edited
    edit_region_xyxy: list[int] | None
    source_image: Path
    spliced_image: Path | None


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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def _resolved_review_path(repo_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"missing {label}")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def review_ground_truth(
    path: Path,
    status: str = "good",
    include_source_pairs: bool = True,
    repo_root: Path | None = None,
) -> list[GroundTruth]:
    """Mirror ``inputs.from_review_export`` ordering and real-image de-duplication."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("review export must contain a records array")
    root = (repo_root or Path.cwd()).resolve()
    output: list[GroundTruth] = []
    seen_sources: set[Path] = set()
    for row in records:
        if row.get("status") != status:
            continue
        task_id = str(row["task_id"])
        box = row.get("edit_region_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
            raise ValueError(f"{task_id}: missing or invalid edit_region_xyxy")
        source = _resolved_review_path(root, row.get("source_image"), "source_image")
        spliced = _resolved_review_path(root, row.get("spliced_image"), "spliced_image")
        output.append(
            GroundTruth(
                f"{task_id}__forged",
                task_id,
                "edited",
                box,
                source,
                spliced,
            )
        )
        if include_source_pairs and source not in seen_sources:
            output.append(
                GroundTruth(
                    f"{task_id}__real",
                    task_id,
                    "not_edited",
                    None,
                    source,
                    None,
                )
            )
            seen_sources.add(source)
    if not output:
        raise ValueError(f"no review records with status={status!r}")
    return output


def _load_results(
    path: Path,
    protocol_version: ProtocolVersionSelector = None,
) -> dict[tuple[str, str], dict[str, Any]]:
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
        if not protocol_version_matches(row, protocol_version):
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


def _iou(a: list[float], b: list[float]) -> float:
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _inside(inner: list[float], outer: list[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _load_rgb(path: Path) -> Image.Image:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "MLLM exact-diff pixel metrics require Pillow"
        ) from exc
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _exact_diff_mask(source_path: Path, spliced_path: Path) -> Image.Image:
    """Return the canonical nonzero RGB-difference GT before JPEG re-encoding."""
    from PIL import ImageChops

    source = _load_rgb(source_path)
    spliced = _load_rgb(spliced_path)
    if source.size != spliced.size:
        raise ValueError(
            f"source/spliced size mismatch: {source.size} != {spliced.size}"
        )
    red, green, blue = ImageChops.difference(source, spliced).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    target = maximum.point(lambda value: 255 if value > 0 else 0, mode="L")
    if target.getbbox() is None:
        raise ValueError(f"exact-difference mask is empty: {spliced_path}")
    return target


def _target_mask(gt: GroundTruth) -> tuple[Image.Image, str]:
    if gt.spliced_image is not None:
        return _exact_diff_mask(gt.source_image, gt.spliced_image), "exact_diff"
    from PIL import Image

    source = _load_rgb(gt.source_image)
    return Image.new("L", source.size, 0), "all_zero"


def _rasterize_boxes(
    size: tuple[int, int],
    boxes: list[list[float]],
) -> Image.Image:
    from PIL import Image, ImageDraw

    width, height = size
    prediction = Image.new("L", size, 0)
    draw = ImageDraw.Draw(prediction)
    for box in boxes:
        x1, y1, x2, y2 = (
            max(0, min(width, round(box[0]))),
            max(0, min(height, round(box[1]))),
            max(0, min(width, round(box[2]))),
            max(0, min(height, round(box[3]))),
        )
        if x1 < x2 and y1 < y2:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    return prediction


def _box_mask(
    size: tuple[int, int],
    box: list[float],
) -> Image.Image:
    return _rasterize_boxes(size, [box])


def _binary_mask_metrics(
    prediction: Image.Image,
    target: Image.Image,
) -> dict[str, Any]:
    """Pixel metrics for the disclosed binary MLLM bbox-union adapter."""
    from PIL import ImageChops

    predicted = prediction.convert("L")
    truth = target.convert("L")
    if predicted.size != truth.size:
        raise ValueError(
            f"prediction/target size mismatch: {predicted.size} != {truth.size}"
        )
    pixels = predicted.width * predicted.height
    predicted_positive = int(predicted.histogram()[255])
    target_positive = int(truth.histogram()[255])
    tp = int(ImageChops.multiply(predicted, truth).histogram()[255])
    fp = predicted_positive - tp
    fn = target_positive - tp
    tn = pixels - tp - fp - fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else None
    return {
        "pixels": pixels,
        "target_positive_pixels": target_positive,
        "predicted_positive_pixels": predicted_positive,
        "predicted_positive_fraction": _safe_div(predicted_positive, pixels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]
    return statistics.fmean(values) if values else None


def _valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "ok" and row.get("valid_for_metrics") is True)


def evaluate_review_export(
    results_path: Path,
    review_export: Path,
    output_dir: Path,
    *,
    status: str = "good",
    include_source_pairs: bool = True,
    protocol_version: ProtocolVersionSelector = None,
    run_manifest_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write separate detection/localization per-image tables and summaries."""
    gt_rows = review_ground_truth(
        review_export,
        status,
        include_source_pairs,
        repo_root=repo_root,
    )
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
        target = None
        gt_mask_kind = "exact_diff" if gt.spliced_image is not None else "all_zero"
        gt_mask_sha256 = None
        pixel_metrics = None
        union_box_iou = None
        box_hit_at_0_3 = None
        if is_valid:
            target, gt_mask_kind = _target_mask(gt)
            gt_mask_sha256 = hashlib.sha256(target.tobytes()).hexdigest()
            prediction = _rasterize_boxes(target.size, boxes)
            reported_size = result.get("image_size")
            if (
                isinstance(reported_size, list)
                and len(reported_size) == 2
                and tuple(int(value) for value in reported_size) != target.size
            ):
                raise ValueError(
                    f"{gt.id}: result image_size {reported_size} != GT {target.size}"
                )
            pixel_metrics = _binary_mask_metrics(prediction, target)
        if gt.edit_region_xyxy is not None:
            gt_box = list(map(float, gt.edit_region_xyxy))
            best_iou = max((_iou(box, gt_box) for box in boxes), default=0.0) if is_valid else None
            any_overlap = any(_overlaps(box, gt_box) for box in boxes) if is_valid else None
            iou_at_0_1 = best_iou >= 0.10 if best_iou is not None else None
            iou_at_0_25 = best_iou >= 0.25 if best_iou is not None else None
            iou_at_0_5 = best_iou >= 0.50 if best_iou is not None else None
            all_inside = bool(boxes) and all(_inside(box, gt_box) for box in boxes) if is_valid else None
            if is_valid:
                union_box_metrics = _binary_mask_metrics(
                    prediction,
                    _box_mask(target.size, gt_box),
                )
                union_box_iou = union_box_metrics["iou"]
                box_hit_at_0_3 = (
                    union_box_iou > 0.3
                    if union_box_iou is not None
                    else False
                )
            real_rejection_correct = None
        else:
            gt_box = None
            any_overlap = None
            best_iou = None
            iou_at_0_1 = None
            iou_at_0_25 = None
            iou_at_0_5 = None
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
            "gt_mask_kind": gt_mask_kind,
            "gt_mask_sha256": gt_mask_sha256,
            "gt_positive_pixels": (
                pixel_metrics["target_positive_pixels"]
                if pixel_metrics is not None
                else None
            ),
            "result_status": result.get("status") if result else "missing_result",
            "valid_for_metrics": is_valid,
            "model_decision": result.get("decision") if is_valid else None,
            "predicted_regions_xyxy": boxes,
            "predicted_mask_path": result.get("mask_path") if is_valid else None,
            "prediction_adapter": "aggregated_bbox_union_binary_mask",
            "model_aggregate": result.get("result") if is_valid else None,
            "has_predicted_region": bool(boxes) if is_valid else None,
            "pixel_metrics": pixel_metrics,
            "pixel_precision": (
                pixel_metrics["precision"] if pixel_metrics is not None else None
            ),
            "pixel_recall": (
                pixel_metrics["recall"] if pixel_metrics is not None else None
            ),
            "pixel_f1": pixel_metrics["f1"] if pixel_metrics is not None else None,
            "pixel_iou": pixel_metrics["iou"] if pixel_metrics is not None else None,
            "pixel_mcc": pixel_metrics["mcc"] if pixel_metrics is not None else None,
            "any_box_overlaps_gt": any_overlap,
            "best_box_iou": best_iou,
            "box_iou_at_0_1": iou_at_0_1,
            "box_iou_at_0_25": iou_at_0_25,
            "box_iou_at_0_5": iou_at_0_5,
            "union_mask_edit_box_iou": union_box_iou,
            "box_hit_at_0_3": box_hit_at_0_3,
            "all_predicted_boxes_inside_gt": all_inside,
            "real_no_edit_correct": real_rejection_correct,
            "result_line": result.get("_result_line") if result else None,
        })
    forged_localization = [row for row in localization_rows if row["gt_label"] == "edited"]
    valid_forged = [row for row in forged_localization if row["valid_for_metrics"]]
    real_localization = [row for row in localization_rows if row["gt_label"] == "not_edited"]
    valid_real = [row for row in real_localization if row["valid_for_metrics"]]
    overlap_success = sum(bool(row["any_box_overlaps_gt"]) for row in valid_forged)
    iou_at_0_1_success = sum(bool(row["box_iou_at_0_1"]) for row in valid_forged)
    iou_at_0_25_success = sum(bool(row["box_iou_at_0_25"]) for row in valid_forged)
    iou_at_0_5_success = sum(bool(row["box_iou_at_0_5"]) for row in valid_forged)
    box_hit_at_0_3_success = sum(
        bool(row["box_hit_at_0_3"]) for row in valid_forged
    )
    contained_success = sum(bool(row["all_predicted_boxes_inside_gt"]) for row in valid_forged)
    real_rejection_success = sum(bool(row["real_no_edit_correct"]) for row in valid_real)
    pixel_counts = {
        name: sum(
            int(row["pixel_metrics"][name])
            for row in valid_forged
            if isinstance(row.get("pixel_metrics"), dict)
        )
        for name in ("tp", "fp", "fn", "tn")
    }
    pixel_micro_precision = _safe_div(
        pixel_counts["tp"],
        pixel_counts["tp"] + pixel_counts["fp"],
    )
    pixel_micro_recall = _safe_div(
        pixel_counts["tp"],
        pixel_counts["tp"] + pixel_counts["fn"],
    )
    pixel_micro_f1 = _safe_div(
        2 * pixel_counts["tp"],
        2 * pixel_counts["tp"] + pixel_counts["fp"] + pixel_counts["fn"],
    )
    pixel_micro_iou = _safe_div(
        pixel_counts["tp"],
        pixel_counts["tp"] + pixel_counts["fp"] + pixel_counts["fn"],
    )
    real_predicted_pixels = sum(
        int(row["pixel_metrics"]["predicted_positive_pixels"])
        for row in valid_real
        if isinstance(row.get("pixel_metrics"), dict)
    )
    real_total_pixels = sum(
        int(row["pixel_metrics"]["pixels"])
        for row in valid_real
        if isinstance(row.get("pixel_metrics"), dict)
    )
    primary_pixel_iou = _mean_metric(valid_forged, "pixel_iou")
    localization_summary: dict[str, Any] = {
        **run_metadata,
        "protocol": "localization",
        "primary_t2_metric": "forged_macro_pixel_iou_exact_diff",
        "primary_t2_value": primary_pixel_iou,
        "prediction_adapter": "aggregated_mllm_bbox_union_binary_mask",
        "pixel_ground_truth": (
            "nonzero_rgb_difference_between_decoded_source_and_spliced_png_"
            "before_canonical_jpeg_encoding"
        ),
        "pixel_average_precision": None,
        "pixel_average_precision_status": (
            "not_applicable_binary_bbox_adapter_has_no_continuous_pixel_scores"
        ),
        "expected_images": len(localization_rows),
        "valid_images": len([row for row in localization_rows if row["valid_for_metrics"]]),
        "coverage": _safe_div(sum(row["valid_for_metrics"] for row in localization_rows), len(localization_rows)),
        "forged_expected": len(forged_localization),
        "forged_valid": len(valid_forged),
        "forged_coverage": _safe_div(len(valid_forged), len(forged_localization)),
        "forged_pixel_macro_precision": _mean_metric(
            valid_forged,
            "pixel_precision",
        ),
        "forged_pixel_macro_recall": _mean_metric(valid_forged, "pixel_recall"),
        "forged_pixel_macro_f1": _mean_metric(valid_forged, "pixel_f1"),
        "forged_pixel_macro_iou": primary_pixel_iou,
        "forged_pixel_macro_mcc": _mean_metric(valid_forged, "pixel_mcc"),
        "forged_pixel_micro_tp": pixel_counts["tp"],
        "forged_pixel_micro_fp": pixel_counts["fp"],
        "forged_pixel_micro_fn": pixel_counts["fn"],
        "forged_pixel_micro_tn": pixel_counts["tn"],
        "forged_pixel_micro_precision": pixel_micro_precision,
        "forged_pixel_micro_recall": pixel_micro_recall,
        "forged_pixel_micro_f1": pixel_micro_f1,
        "forged_pixel_micro_iou": pixel_micro_iou,
        "auxiliary_box_metric": (
            "iou_of_aggregated_bbox_union_mask_with_edit_region_xyxy"
        ),
        "auxiliary_box_hit_threshold": "strict_greater_than_0.3",
        "auxiliary_box_hit_successes": box_hit_at_0_3_success,
        "auxiliary_box_hit_accuracy": _safe_div(
            box_hit_at_0_3_success,
            len(valid_forged),
        ),
        "box_overlap_successes": overlap_success,
        "box_overlap_accuracy": _safe_div(overlap_success, len(valid_forged)),
        "box_iou_at_0_1_successes": iou_at_0_1_success,
        "box_iou_at_0_1_accuracy": _safe_div(iou_at_0_1_success, len(valid_forged)),
        "box_iou_at_0_25_successes": iou_at_0_25_success,
        "box_iou_at_0_25_accuracy": _safe_div(iou_at_0_25_success, len(valid_forged)),
        "box_iou_at_0_5_successes": iou_at_0_5_success,
        "box_iou_at_0_5_accuracy": _safe_div(iou_at_0_5_success, len(valid_forged)),
        "all_boxes_inside_gt_successes": contained_success,
        "all_boxes_inside_gt_accuracy": _safe_div(contained_success, len(valid_forged)),
        "real_expected": len(real_localization),
        "real_valid": len(valid_real),
        "real_coverage": _safe_div(len(valid_real), len(real_localization)),
        "real_no_edit_successes": real_rejection_success,
        "real_no_edit_accuracy": _safe_div(real_rejection_success, len(valid_real)),
        "real_predicted_positive_fraction_macro": _mean_metric(
            [
                {
                    "predicted_positive_fraction": (
                        row["pixel_metrics"]["predicted_positive_fraction"]
                    )
                }
                for row in valid_real
                if isinstance(row.get("pixel_metrics"), dict)
            ],
            "predicted_positive_fraction",
        ),
        "real_predicted_positive_fraction_micro": _safe_div(
            real_predicted_pixels,
            real_total_pixels,
        ),
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
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = evaluate_review_export(
        args.results, args.review_export, args.output_dir,
        status=args.review_status,
        include_source_pairs=not args.no_source_pairs,
        protocol_version=args.protocol_version,
        run_manifest_path=args.run_manifest,
        repo_root=args.repo_root.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
