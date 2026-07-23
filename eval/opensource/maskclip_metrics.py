from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def descriptive(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if not numbers:
        return {"count": 0}
    ordered = sorted(numbers)

    def quantile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p05": quantile(0.05),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def binary_pixel_metrics(
    score_map: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(f"score/target shape mismatch: {scores.shape} != {truth.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if scores.size == 0:
        raise ValueError("score map is empty")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("score map falls outside [0, 1]")

    prediction = scores >= threshold
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else None
    ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        ap = float(average_precision_score(truth.reshape(-1), scores.reshape(-1)))
    return {
        "threshold": threshold,
        "pixels": int(scores.size),
        "target_positive_pixels": int(np.count_nonzero(truth)),
        "predicted_positive_pixels": int(np.count_nonzero(prediction)),
        "predicted_positive_fraction": float(np.mean(prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
        "pixel_ap": ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


def image_detection_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("label") in (0, 1)
        and finite_float(row.get("score")) is not None
    ]
    labels = np.asarray([int(row["label"]) for row in valid], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in valid], dtype=np.float64)
    result: dict[str, Any] = {
        "valid_images": len(valid),
        "threshold": threshold,
    }
    if len(set(labels.tolist())) < 2:
        return result
    predictions = (scores >= threshold).astype(np.int64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    eligible = np.where(false_positive_rate <= 0.05)[0]
    best_index = int(eligible[np.argmax(true_positive_rate[eligible])])
    result.update(
        {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "accuracy": float(np.mean(predictions == labels)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "tpr_at_fpr_5_percent": float(true_positive_rate[best_index]),
            "tpr_at_fpr_5_percent_threshold": finite_float(
                thresholds[best_index]
            ),
        }
    )
    return result


def summarize_results(
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

    localization: dict[str, Any] = {}
    for space in ("model_512", "native"):
        metric_rows = [
            row["localization"][space]
            for row in valid
            if row.get("kind") == "forged"
            and isinstance(row.get("localization"), dict)
            and isinstance(row["localization"].get(space), dict)
        ]
        localization[space] = {
            "images": len(metric_rows),
            **{
                metric: descriptive(
                    float(row[metric])
                    for row in metric_rows
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
            name: sum(int(row.get(name, 0)) for row in metric_rows)
            for name in ("tp", "fp", "fn", "tn")
        }
        localization[space]["micro_at_threshold"] = {
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
        row["localization"]["model_512"]
        for row in valid
        if row.get("kind") == "real"
        and isinstance(row.get("localization"), dict)
        and isinstance(row["localization"].get("model_512"), dict)
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
        "localization_forged": localization,
        "localization_real": {
            "images": len(real_localization),
            "predicted_positive_fraction": descriptive(
                float(row["predicted_positive_fraction"])
                for row in real_localization
            ),
            "score_mean": descriptive(float(row["score_mean"]) for row in real_localization),
            "score_max": descriptive(float(row["score_max"]) for row in real_localization),
        },
        "latency_ms": descriptive(
            float(row["latency_ms"])
            for row in valid
            if finite_float(row.get("latency_ms")) is not None
        ),
    }
