from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval.opensource.maskclip_metrics import (
    descriptive,
    finite_float,
    image_detection_metrics,
    safe_div,
)


def summarize_trufor_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    classification_threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    latest = {str(row["id"]): row for row in rows if isinstance(row.get("id"), str)}
    expected_ids = [str(row["sample_id"]) for row in expected_rows]
    selected = [latest[row_id] for row_id in expected_ids if row_id in latest]
    valid = [row for row in selected if row.get("status") == "ok"]

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_kind[str(row.get("kind"))].append(row)
    score_by_kind = {
        kind: descriptive(
            float(row["score"])
            for row in kind_rows
            if finite_float(row.get("score")) is not None
        )
        for kind, kind_rows in sorted(by_kind.items())
    }

    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in valid:
        score = finite_float(row.get("score"))
        if score is not None:
            paired[str(row["task_id"])][str(row["kind"])] = score
    paired_deltas = [
        values["forged"] - values["real"]
        for values in paired.values()
        if "real" in values and "forged" in values
    ]

    forged_localization = [
        row["localization"]["native"]
        for row in valid
        if row.get("kind") == "forged"
        and isinstance(row.get("localization"), dict)
        and isinstance(row["localization"].get("native"), dict)
    ]
    localization = {
        "images": len(forged_localization),
        **{
            metric: descriptive(
                float(row[metric])
                for row in forged_localization
                if finite_float(row.get(metric)) is not None
            )
            for metric in (
                "pixel_ap",
                "precision",
                "recall",
                "f1",
                "iou",
                "mcc",
                "predicted_positive_fraction",
                "score_mean",
                "score_max",
            )
        },
    }
    counts = {
        name: sum(int(row.get(name, 0)) for row in forged_localization)
        for name in ("tp", "fp", "fn", "tn")
    }
    localization["micro_at_threshold"] = {
        "threshold": mask_threshold,
        **counts,
        "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
        "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
        "f1": safe_div(
            2 * counts["tp"],
            2 * counts["tp"] + counts["fp"] + counts["fn"],
        ),
        "iou": safe_div(
            counts["tp"],
            counts["tp"] + counts["fp"] + counts["fn"],
        ),
    }

    real_localization = [
        row["localization"]["native"]
        for row in valid
        if row.get("kind") == "real"
        and isinstance(row.get("localization"), dict)
        and isinstance(row["localization"].get("native"), dict)
    ]
    reliability_rows = [
        row["reliability"]
        for row in valid
        if isinstance(row.get("reliability"), dict)
    ]

    return {
        "schema_version": "opensource_summary_v1",
        "coverage": {
            "expected_images": len(expected_rows),
            "result_images": len(selected),
            "valid_images": len(valid),
            "error_images": len(selected) - len(valid),
            "missing_images": len(expected_rows) - len(selected),
        },
        "score_by_kind": score_by_kind,
        "paired_score_delta": descriptive(paired_deltas),
        "paired_ranking_accuracy": (
            sum(delta > 0 for delta in paired_deltas) / len(paired_deltas)
            if paired_deltas
            else None
        ),
        "detection": image_detection_metrics(valid, classification_threshold),
        "localization_forged": {"native": localization},
        "localization_real": {
            "images": len(real_localization),
            "predicted_positive_fraction": descriptive(
                float(row["predicted_positive_fraction"])
                for row in real_localization
            ),
            "score_mean": descriptive(
                float(row["score_mean"]) for row in real_localization
            ),
            "score_max": descriptive(
                float(row["score_max"]) for row in real_localization
            ),
        },
        "reliability": {
            "images": len(reliability_rows),
            **{
                metric: descriptive(
                    float(row[metric])
                    for row in reliability_rows
                    if finite_float(row.get(metric)) is not None
                )
                for metric in ("min", "mean", "median", "p05", "p95", "max")
            },
        },
        "latency_ms": descriptive(
            float(row["latency_ms"])
            for row in valid
            if finite_float(row.get("latency_ms")) is not None
        ),
        "peak_cuda_memory_bytes": descriptive(
            float(row["peak_cuda_memory_bytes"])
            for row in valid
            if finite_float(row.get("peak_cuda_memory_bytes")) is not None
        ),
    }
