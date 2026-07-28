#!/usr/bin/env python3
"""Aggregate the four MLLMs on the authoritative Balanced250 MLLM scope.

Detection is scored on local750 + real250.  Localization is scored on every
one of the 750 forged local-splice images.  A missing/invalid aggregate or an
aggregate without any in-bounds bbox is an explicit localization miss.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mllm.metrics import (
    _auc_roc,
    _average_precision,
    _binary_mask_metrics,
    _exact_diff_mask,
    _inside,
    _iou,
    _overlaps,
    _rasterize_boxes,
)


SCOPE_ID = "claimforge-mllm-balanced250-local750-real250-v2"
DEFAULT_LEDGER = Path("annotations/claimforge_mllm_benchmark1000_v2.jsonl")
DEFAULT_OUTPUT = Path("results/mllm/balanced250_local1000_v2")
MODEL_FILES: dict[str, dict[str, Any]] = {
    "gpt_5_6_luna": {
        "display_name": "GPT-5.6 Luna",
        "paths": [
            "results/mllm/gpt/gpt56luna_pilot_good275_c15_v3_20260715T153257_0800.jsonl",
            "results/mllm/gpt/final_cat251_trash250_total501_suite0724_20260725.jsonl",
            "results/mllm/gpt/fullai_orangebox_all807_detectionv3_20260725.jsonl",
            "results/mllm/gpt/gpt_cat_detection_recovery_intent_tokens2000_20260728.jsonl",
            "results/mllm/gpt/gpt_cat_localization_recovery_combined_20260728.jsonl",
        ],
    },
    "qwen_3_7_plus": {
        "display_name": "Qwen 3.7 Plus",
        "paths": [
            "results/mllm/qwen3_7_plus/qwen37plus_pilot_good275_c15_v3_20260715T153257_0800.jsonl",
            "results/mllm/qwen3_7_plus/final_cat251_trash250_total501_suite0724_20260725.jsonl",
            "results/mllm/qwen3_7_plus/fullai_orangebox_all807_detectionv3_20260725.jsonl",
        ],
    },
    "claude_opus_4_8": {
        "display_name": "Claude Opus 4.8",
        "paths": [
            "results/mllm/claude_opus_4_8/mouse_good275_total550_v3_20260727.jsonl",
            "results/mllm/claude_opus_4_8/mouse6_oversize_anthropic_native_v3_20260727.jsonl",
            "results/mllm/claude_opus_4_8/final_cat251_trash250_total501_suite0724_20260725.jsonl",
            "results/mllm/claude_opus_4_8/final501_oversize9_anthropic_native_suite0724_20260727.jsonl",
            "results/mllm/claude_opus_4_8/fullai_orangebox_all807_detectionv3_20260725.jsonl",
            "results/mllm/claude_opus_4_8/claude_mouse_localization_recovery_20260728.jsonl",
        ],
    },
    "doubao_seed_2_1_pro_260628": {
        "display_name": "Doubao Seed 2.1 Pro 260628",
        "paths": [
            "results/mllm/doubao_seed_2_1_pro_260628/doubao_main_local776_suite0724_20260726.jsonl",
            "results/mllm/doubao_seed_2_1_pro_260628/doubao_real_source_union270_detectionv3_20260726.jsonl",
            "results/mllm/doubao_seed_2_1_pro_260628/doubao_fullai_orangebox_all807_detectionv3_20260726.jsonl",
            "results/mllm/doubao_seed_2_1_pro_260628/doubao_main_local776_localization_bbox1000_v5_20260727.jsonl",
        ],
    },
}


def _safe_div(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]
    return statistics.fmean(values) if values else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _is_valid(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("status") == "ok"
        and row.get("valid_for_metrics") is True
    )


def _load_model_results(
    repo_root: Path, paths: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Use the last valid aggregate in the declared collection order."""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in paths:
        path = repo_root / raw_path
        if not path.is_file():
            continue
        for line_number, row in enumerate(_load_jsonl(path), 1):
            protocol = row.get("protocol_key") or row.get("protocol_id")
            if protocol not in {"detection", "localization"}:
                continue
            image_sha256 = row.get("image_sha256")
            if not isinstance(image_sha256, str):
                continue
            item = dict(row)
            item["_source_result_path"] = raw_path
            item["_source_result_line"] = line_number
            key = (image_sha256, str(protocol))
            if _is_valid(item) or key not in indexed:
                indexed[key] = item
    return indexed


def _valid_in_bounds_boxes(
    row: dict[str, Any] | None, width: int, height: int
) -> tuple[list[list[float]], int]:
    if not _is_valid(row):
        return [], 0
    raw_boxes = row.get("regions_px")
    if not isinstance(raw_boxes, list):
        return [], 0
    boxes: list[list[float]] = []
    invalid = 0
    for raw in raw_boxes:
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or not all(isinstance(value, (int, float)) for value in raw)
        ):
            invalid += 1
            continue
        box = [float(value) for value in raw]
        if (
            0 <= box[0] < box[2] <= width
            and 0 <= box[1] < box[3] <= height
        ):
            boxes.append(box)
        else:
            invalid += 1
    return boxes, invalid


def _detection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_for_metrics"]]
    tp = sum(row["gt_edited"] and row["model_decision"] == "edited" for row in valid)
    tn = sum(not row["gt_edited"] and row["model_decision"] == "not_edited" for row in valid)
    fp = sum(not row["gt_edited"] and row["model_decision"] == "edited" for row in valid)
    fn = sum(row["gt_edited"] and row["model_decision"] == "not_edited" for row in valid)
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    return {
        "expected_images": len(rows),
        "valid_images": len(valid),
        "coverage": _safe_div(len(valid), len(rows)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, len(valid)),
        "precision": _safe_div(tp, tp + fp),
        "recall_tpr": tpr,
        "specificity_tnr": tnr,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (
            (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
        ),
        "auroc": _auc_roc(valid),
        "average_precision": _average_precision(valid),
    }


def _condition_detection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_for_metrics"]]
    edited = sum(row["model_decision"] == "edited" for row in valid)
    correct = sum(row["is_correct"] for row in valid)
    return {
        "expected": len(rows),
        "valid": len(valid),
        "coverage": _safe_div(len(valid), len(rows)),
        "edited_predictions": edited,
        "edited_rate": _safe_div(edited, len(valid)),
        "accuracy": _safe_div(correct, len(valid)),
    }


def _localization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(rows)
    result_valid = sum(row["result_valid_for_metrics"] for row in rows)
    with_bbox = sum(row["has_valid_bbox"] for row in rows)
    overlap = sum(row["any_box_overlaps_gt"] for row in rows)
    iou01 = sum(row["box_iou_at_0_1"] for row in rows)
    iou025 = sum(row["box_iou_at_0_25"] for row in rows)
    iou05 = sum(row["box_iou_at_0_5"] for row in rows)
    contained = sum(row["all_predicted_boxes_inside_gt"] for row in rows)
    pixel_counts = {
        key: sum(int(row["pixel_metrics"][key]) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    return {
        "expected_forged": expected,
        "scored_forged": expected,
        "result_valid": result_valid,
        "result_coverage": _safe_div(result_valid, expected),
        "with_valid_bbox": with_bbox,
        "strict_localization_misses": expected - with_bbox,
        "invalid_or_missing_result_misses": sum(
            not row["result_valid_for_metrics"] for row in rows
        ),
        "valid_result_but_no_valid_bbox_misses": sum(
            row["result_valid_for_metrics"] and not row["has_valid_bbox"]
            for row in rows
        ),
        "box_overlap_successes": overlap,
        "box_overlap_accuracy": _safe_div(overlap, expected),
        "box_iou_at_0_1_successes": iou01,
        "box_iou_at_0_1_accuracy": _safe_div(iou01, expected),
        "box_iou_at_0_25_successes": iou025,
        "box_iou_at_0_25_accuracy": _safe_div(iou025, expected),
        "box_iou_at_0_5_successes": iou05,
        "box_iou_at_0_5_accuracy": _safe_div(iou05, expected),
        "all_boxes_inside_gt_successes": contained,
        "all_boxes_inside_gt_accuracy": _safe_div(contained, expected),
        "forged_pixel_macro_precision": _mean(rows, "pixel_precision"),
        "forged_pixel_macro_recall": _mean(rows, "pixel_recall"),
        "forged_pixel_macro_f1": _mean(rows, "pixel_f1"),
        "forged_pixel_macro_iou": _mean(rows, "pixel_iou"),
        "forged_pixel_macro_mcc": _mean(rows, "pixel_mcc"),
        "forged_pixel_micro_tp": pixel_counts["tp"],
        "forged_pixel_micro_fp": pixel_counts["fp"],
        "forged_pixel_micro_fn": pixel_counts["fn"],
        "forged_pixel_micro_tn": pixel_counts["tn"],
        "forged_pixel_micro_precision": _safe_div(
            pixel_counts["tp"], pixel_counts["tp"] + pixel_counts["fp"]
        ),
        "forged_pixel_micro_recall": _safe_div(
            pixel_counts["tp"], pixel_counts["tp"] + pixel_counts["fn"]
        ),
        "forged_pixel_micro_f1": _safe_div(
            2 * pixel_counts["tp"],
            2 * pixel_counts["tp"] + pixel_counts["fp"] + pixel_counts["fn"],
        ),
        "forged_pixel_micro_iou": _safe_div(
            pixel_counts["tp"],
            pixel_counts["tp"] + pixel_counts["fp"] + pixel_counts["fn"],
        ),
        "denominator_policy": (
            "all expected forged images; missing, invalid, empty, or entirely "
            "out-of-bounds bbox output is a miss"
        ),
    }


def aggregate(
    repo_root: Path, ledger_path: Path, output_dir: Path
) -> dict[str, Any]:
    ledger = ledger_path if ledger_path.is_absolute() else repo_root / ledger_path
    scope = _load_jsonl(ledger)
    if len(scope) != 1000:
        raise ValueError(f"scope has {len(scope)} rows, expected 1000")
    counts = collections.Counter(row["candidate"] for row in scope)
    if counts != {"real": 250, "mouse": 250, "cat": 250, "trash_can": 250}:
        raise ValueError(f"unexpected scope counts: {dict(counts)}")

    output = output_dir if output_dir.is_absolute() else repo_root / output_dir
    indexes = {
        model: _load_model_results(repo_root, config["paths"])
        for model, config in MODEL_FILES.items()
    }
    detection_rows: dict[str, list[dict[str, Any]]] = {
        model: [] for model in MODEL_FILES
    }
    localization_rows: dict[str, list[dict[str, Any]]] = {
        model: [] for model in MODEL_FILES
    }

    for item in scope:
        image_sha256 = str(item["image_sha256"])
        gt_edited = item["label"] == "forged"
        condition = str(item["condition"])
        for model, index in indexes.items():
            result = index.get((image_sha256, "detection"))
            valid = _is_valid(result)
            decision = result.get("decision") if valid else None
            raw_score = result.get("score") if valid else None
            if not isinstance(raw_score, (int, float)) and valid:
                probability = result.get("p_ai_edited")
                raw_score = (
                    float(probability) / 100
                    if isinstance(probability, (int, float))
                    else (1.0 if decision == "edited" else 0.0)
                )
            detection_rows[model].append(
                {
                    "scope_id": SCOPE_ID,
                    "model": MODEL_FILES[model]["display_name"],
                    "model_slug": model,
                    "id": item["id"],
                    "task_id": item["task_id"],
                    "condition": condition,
                    "candidate": item["candidate"],
                    "image_path": item["image_path"],
                    "image_sha256": image_sha256,
                    "gt_label": item["label"],
                    "gt_edited": gt_edited,
                    "result_status": (
                        result.get("status") if result else "missing_result"
                    ),
                    "valid_for_metrics": valid,
                    "model_decision": decision,
                    "model_score": raw_score,
                    "is_correct": (
                        decision == ("edited" if gt_edited else "not_edited")
                        if valid
                        else False
                    ),
                    "source_result_path": (
                        result.get("_source_result_path") if result else None
                    ),
                    "source_result_line": (
                        result.get("_source_result_line") if result else None
                    ),
                }
            )

    forged_scope = [row for row in scope if row["label"] == "forged"]
    for position, item in enumerate(forged_scope, 1):
        width = int(item["image_size"]["width"])
        height = int(item["image_size"]["height"])
        forged_path = Path(str(item["image_path"]))
        source_path = Path(str(item["source_image"]))
        forged = forged_path if forged_path.is_absolute() else repo_root / forged_path
        source = source_path if source_path.is_absolute() else repo_root / source_path
        target = _exact_diff_mask(source, forged)
        if target.size != (width, height):
            raise ValueError(
                f"{item['id']}: ledger size {(width, height)} != {target.size}"
            )
        target_sha256 = hashlib.sha256(target.tobytes()).hexdigest()
        gt_box = [float(value) for value in item["edit_region_xyxy"]]
        for model, index in indexes.items():
            result = index.get((str(item["image_sha256"]), "localization"))
            result_valid = _is_valid(result)
            boxes, invalid_box_count = _valid_in_bounds_boxes(
                result, width, height
            )
            prediction = _rasterize_boxes(target.size, boxes)
            pixel = _binary_mask_metrics(prediction, target)
            best_iou = max((_iou(box, gt_box) for box in boxes), default=0.0)
            row = {
                "scope_id": SCOPE_ID,
                "model": MODEL_FILES[model]["display_name"],
                "model_slug": model,
                "id": item["id"],
                "task_id": item["task_id"],
                "condition": item["condition"],
                "candidate": item["candidate"],
                "image_path": item["image_path"],
                "image_sha256": item["image_sha256"],
                "gt_edit_region_xyxy": item["edit_region_xyxy"],
                "gt_mask_kind": "exact_diff",
                "gt_mask_sha256": target_sha256,
                "gt_positive_pixels": pixel["target_positive_pixels"],
                "result_status": (
                    result.get("status") if result else "missing_result"
                ),
                "result_valid_for_metrics": result_valid,
                "model_decision": (
                    result.get("decision") if result_valid else None
                ),
                "predicted_regions_xyxy": boxes,
                "invalid_or_out_of_bounds_box_count": invalid_box_count,
                "has_valid_bbox": bool(boxes),
                "strict_localization_miss": not boxes,
                "pixel_metrics": pixel,
                "pixel_precision": pixel["precision"],
                "pixel_recall": pixel["recall"],
                "pixel_f1": pixel["f1"],
                "pixel_iou": pixel["iou"],
                "pixel_mcc": pixel["mcc"],
                "any_box_overlaps_gt": any(
                    _overlaps(box, gt_box) for box in boxes
                ),
                "best_box_iou": best_iou,
                "box_iou_at_0_1": best_iou >= 0.10,
                "box_iou_at_0_25": best_iou >= 0.25,
                "box_iou_at_0_5": best_iou >= 0.50,
                "all_predicted_boxes_inside_gt": bool(boxes)
                and all(_inside(box, gt_box) for box in boxes),
                "source_result_path": (
                    result.get("_source_result_path") if result else None
                ),
                "source_result_line": (
                    result.get("_source_result_line") if result else None
                ),
            }
            localization_rows[model].append(row)
        if position % 50 == 0:
            print(f"localization GT {position}/{len(forged_scope)}", flush=True)

    combined: dict[str, Any] = {
        "schema_version": "claimforge_mllm_balanced250_aggregate_v2",
        "scope_id": SCOPE_ID,
        "scope_ledger": str(ledger_path),
        "scope_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "counts": {
            "total": 1000,
            "real": 250,
            "local_forged": 750,
            "mouse": 250,
            "cat": 250,
            "trash_can": 250,
        },
        "localization_policy": (
            "Strict forged denominator: missing/invalid aggregate, empty bbox, "
            "or no in-bounds bbox is a localization miss."
        ),
        "models": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for model, config in MODEL_FILES.items():
        model_dir = output / model
        detection = _detection_summary(detection_rows[model])
        detection["by_condition"] = {
            condition: _condition_detection(
                [
                    row
                    for row in detection_rows[model]
                    if row["condition"] == condition
                ]
            )
            for condition in ("real", "local_mouse", "local_cat", "local_trash_can")
        }
        localization = _localization_summary(localization_rows[model])
        localization["by_condition"] = {
            condition: _localization_summary(
                [
                    row
                    for row in localization_rows[model]
                    if row["condition"] == condition
                ]
            )
            for condition in ("local_mouse", "local_cat", "local_trash_can")
        }
        _dump_jsonl(model_dir / "detection_per_image.jsonl", detection_rows[model])
        _dump_json(model_dir / "detection_metrics.json", detection)
        _dump_csv(
            model_dir / "detection_metrics.csv",
            [{key: value for key, value in detection.items() if key != "by_condition"}],
        )
        _dump_jsonl(
            model_dir / "localization_per_image.jsonl", localization_rows[model]
        )
        _dump_json(model_dir / "localization_metrics.json", localization)
        _dump_csv(
            model_dir / "localization_metrics.csv",
            [
                {
                    key: value
                    for key, value in localization.items()
                    if key != "by_condition"
                }
            ],
        )
        combined["models"][model] = {
            "display_name": config["display_name"],
            "source_result_files": config["paths"],
            "detection": detection,
            "localization": localization,
        }
        summary_rows.append(
            {
                "model": config["display_name"],
                "model_slug": model,
                "detection_accuracy": detection["accuracy"],
                "detection_precision": detection["precision"],
                "detection_recall_tpr": detection["recall_tpr"],
                "detection_specificity_tnr": detection["specificity_tnr"],
                "detection_f1": detection["f1"],
                "detection_balanced_accuracy": detection["balanced_accuracy"],
                "detection_auroc": detection["auroc"],
                "detection_average_precision": detection["average_precision"],
                "localization_expected_forged": localization["expected_forged"],
                "localization_result_valid": localization["result_valid"],
                "localization_strict_misses": localization[
                    "strict_localization_misses"
                ],
                "localization_overlap_accuracy": localization[
                    "box_overlap_accuracy"
                ],
                "localization_iou_at_0_1": localization[
                    "box_iou_at_0_1_accuracy"
                ],
                "localization_iou_at_0_25": localization[
                    "box_iou_at_0_25_accuracy"
                ],
                "localization_iou_at_0_5": localization[
                    "box_iou_at_0_5_accuracy"
                ],
                "localization_pixel_macro_iou": localization[
                    "forged_pixel_macro_iou"
                ],
                "localization_pixel_micro_iou": localization[
                    "forged_pixel_micro_iou"
                ],
            }
        )
    _dump_json(output / "summary.json", combined)
    _dump_csv(output / "summary.csv", summary_rows)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate four MLLMs on Balanced250 local750 + real250"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = aggregate(
        args.repo_root.resolve(), args.ledger, args.output_dir
    )
    compact = {
        model: {
            "detection_accuracy": value["detection"]["accuracy"],
            "localization_overlap_accuracy": value["localization"][
                "box_overlap_accuracy"
            ],
            "localization_strict_misses": value["localization"][
                "strict_localization_misses"
            ],
        }
        for model, value in result["models"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
