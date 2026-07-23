from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

import numpy as np

from eval.opensource.maskclip_metrics import (
    binary_pixel_metrics,
    descriptive,
    finite_float,
    safe_div,
)


FIXED_MASK_THRESHOLD = 0.5
LOCALIZATION_METRICS = (
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
THRESHOLDED_MACRO_METRICS = (
    "precision",
    "recall",
    "f1",
    "iou",
    "mcc",
)
BOOTSTRAP_METRICS = (
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_predicted_positive_fraction_macro_at_0_5",
    "real_predicted_positive_fraction_micro_at_0_5",
)


def _require_fixed_threshold(threshold: float) -> float:
    value = float(threshold)
    if not math.isclose(value, FIXED_MASK_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "CAT-Net v2 primary localization metrics use the fixed "
            f"{FIXED_MASK_THRESHOLD} threshold, not {value}"
        )
    return FIXED_MASK_THRESHOLD


def _mean_finite(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := finite_float(row.get(metric))) is not None
    ]
    return statistics.fmean(values) if values else None


def _mcc_from_counts(
    *,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def _aggregate_localization(
    metric_rows: list[dict[str, Any]],
    *,
    mask_threshold: float,
) -> dict[str, Any]:
    threshold = _require_fixed_threshold(mask_threshold)
    for index, row in enumerate(metric_rows):
        row_threshold = finite_float(row.get("threshold"))
        if row_threshold is not None and not math.isclose(
            row_threshold,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "localization row "
                f"{index} threshold {row_threshold} != fixed threshold {threshold}"
            )

    counts = {
        name: sum(int(row.get(name, 0)) for row in metric_rows)
        for name in ("tp", "fp", "fn", "tn")
    }
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    return {
        "images": len(metric_rows),
        **{
            metric: descriptive(
                value
                for row in metric_rows
                if (value := finite_float(row.get(metric))) is not None
            )
            for metric in LOCALIZATION_METRICS
        },
        "macro_at_threshold": {
            "threshold": threshold,
            **{
                metric: _mean_finite(metric_rows, metric)
                for metric in THRESHOLDED_MACRO_METRICS
            },
        },
        "micro_at_threshold": {
            "threshold": threshold,
            **counts,
            "pixels": tp + fp + fn + tn,
            "target_positive_pixels": tp + fn,
            "predicted_positive_pixels": tp + fp,
            "predicted_positive_fraction": safe_div(tp + fp, tp + fp + fn + tn),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "mcc": _mcc_from_counts(tp=tp, fp=fp, fn=fn, tn=tn),
        },
    }


def _localization_rows(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    localization_space: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        localization = row.get("localization")
        if row.get("kind") != kind or not isinstance(localization, dict):
            continue
        metrics = localization.get(localization_space)
        if isinstance(metrics, dict):
            selected.append(metrics)
    return selected


def summarize_catnet_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
) -> dict[str, Any]:
    """Summarize CAT-Net's native-resolution T2 localization output.

    CAT-Net v2 does not expose a calibrated image-level detection head in this
    adapter. This summary therefore intentionally derives no T1 score,
    decision, ranking statistic, AP, or AUROC from its localization map.
    """

    threshold = _require_fixed_threshold(mask_threshold)
    latest = {
        str(row["id"]): row
        for row in rows
        if isinstance(row.get("id"), str)
    }
    expected_ids = [str(row["sample_id"]) for row in expected_rows]
    selected = [latest[row_id] for row_id in expected_ids if row_id in latest]
    valid = [row for row in selected if row.get("status") == "ok"]
    forged_rows = _localization_rows(
        valid,
        kind="forged",
        localization_space="native",
    )
    real_rows = _localization_rows(
        valid,
        kind="real",
        localization_space="native",
    )

    return {
        "schema_version": "opensource_summary_v1",
        "task_scope": {
            "primary_task": "T2_localization",
            "mask_threshold": threshold,
        },
        "coverage": {
            "expected_images": len(expected_rows),
            "result_images": len(selected),
            "valid_images": len(valid),
            "error_images": len(selected) - len(valid),
            "missing_images": len(expected_rows) - len(selected),
        },
        "localization": {
            "native": _aggregate_localization(
                forged_rows,
                mask_threshold=threshold,
            )
        },
        "real_localization": {
            "native": _aggregate_localization(
                real_rows,
                mask_threshold=threshold,
            )
        },
        "latency_ms": descriptive(
            value
            for row in valid
            if (value := finite_float(row.get("latency_ms"))) is not None
        ),
        "peak_cuda_memory_bytes": descriptive(
            value
            for row in valid
            if (
                value := finite_float(row.get("peak_cuda_memory_bytes"))
            )
            is not None
        ),
    }


def _pair_row(pair: Any, kind: str) -> dict[str, Any]:
    if isinstance(pair, Mapping):
        row = pair.get(kind)
    else:
        row = getattr(pair, kind, None)
    if not isinstance(row, dict):
        raise ValueError(f"pair has no {kind!r} result row")
    return row


def _pair_metrics(
    pair: Any,
    *,
    kind: str,
    localization_space: str,
) -> dict[str, Any]:
    row = _pair_row(pair, kind)
    localization = row.get("localization")
    if not isinstance(localization, dict):
        raise ValueError(f"{kind} result has no localization object")
    metrics = localization.get(localization_space)
    if not isinstance(metrics, dict):
        raise ValueError(
            f"{kind} result has no {localization_space!r} localization metrics"
        )
    return metrics


def _required_float(value: Any, label: str) -> float:
    result = finite_float(value)
    if result is None:
        raise ValueError(f"{label} is not finite")
    return result


def _pair_slice_arrays(
    pairs: list[Any],
    *,
    localization_space: str,
) -> dict[str, np.ndarray]:
    forged = [
        _pair_metrics(
            pair,
            kind="forged",
            localization_space=localization_space,
        )
        for pair in pairs
    ]
    real = [
        _pair_metrics(
            pair,
            kind="real",
            localization_space=localization_space,
        )
        for pair in pairs
    ]
    for kind, metric_rows in (("forged", forged), ("real", real)):
        for index, row in enumerate(metric_rows):
            row_threshold = finite_float(row.get("threshold"))
            if row_threshold is not None:
                _require_fixed_threshold(row_threshold)
            if int(row.get("pixels", 0)) <= 0:
                raise ValueError(f"{kind} pair {index} has no evaluated pixels")

    return {
        "pixel_ap": np.asarray(
            [
                _required_float(row.get("pixel_ap"), "forged pixel AP")
                for row in forged
            ],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [
                _required_float(row.get("f1"), "forged pixel F1")
                for row in forged
            ],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [
                _required_float(row.get("iou"), "forged pixel IoU")
                for row in forged
            ],
            dtype=np.float64,
        ),
        "tp": np.asarray([int(row["tp"]) for row in forged], dtype=np.int64),
        "fp": np.asarray([int(row["fp"]) for row in forged], dtype=np.int64),
        "fn": np.asarray([int(row["fn"]) for row in forged], dtype=np.int64),
        "forged_pixels": np.asarray(
            [int(row["pixels"]) for row in forged],
            dtype=np.int64,
        ),
        "forged_target_positive_pixels": np.asarray(
            [int(row["target_positive_pixels"]) for row in forged],
            dtype=np.int64,
        ),
        "real_predicted_positive_pixels": np.asarray(
            [
                int(
                    row.get(
                        "predicted_positive_pixels",
                        int(row.get("tp", 0)) + int(row.get("fp", 0)),
                    )
                )
                for row in real
            ],
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(
            [int(row["pixels"]) for row in real],
            dtype=np.int64,
        ),
        "real_predicted_positive_fraction": np.asarray(
            [
                _required_float(
                    row.get("predicted_positive_fraction"),
                    "real predicted-positive fraction",
                )
                for row in real
            ],
            dtype=np.float64,
        ),
    }


def _pair_slice_point_metrics(
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    real_predicted_positive = int(
        np.sum(arrays["real_predicted_positive_pixels"])
    )
    real_pixels = int(np.sum(arrays["real_pixels"]))
    pixel_f1_denominator = 2 * tp + fp + fn
    pixel_iou_denominator = tp + fp + fn
    if pixel_f1_denominator <= 0 or pixel_iou_denominator <= 0:
        raise ValueError("forged localization slice has no positive target pixels")
    if real_pixels <= 0:
        raise ValueError("real localization slice has no evaluated pixels")
    return {
        "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
        "pixel_f1_macro_at_0_5": float(np.mean(arrays["pixel_f1"])),
        "pixel_iou_macro_at_0_5": float(np.mean(arrays["pixel_iou"])),
        "pixel_f1_micro_at_0_5": 2.0 * tp / pixel_f1_denominator,
        "pixel_iou_micro_at_0_5": tp / pixel_iou_denominator,
        "real_predicted_positive_fraction_macro_at_0_5": float(
            np.mean(arrays["real_predicted_positive_fraction"])
        ),
        "real_predicted_positive_fraction_micro_at_0_5": (
            real_predicted_positive / real_pixels
        ),
    }


def _percentile_ci(values: list[float]) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def summarize_catnet_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    """Pair-bootstrap localization estimates without deriving T1 metrics."""

    if not pairs:
        raise ValueError("pair slice is empty")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not localization_space:
        raise ValueError("localization space must be non-empty")

    arrays = _pair_slice_arrays(
        pairs,
        localization_space=localization_space,
    )
    point = _pair_slice_point_metrics(arrays)
    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    for _ in range(iterations):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sampled = {name: values[indices] for name, values in arrays.items()}
        values = _pair_slice_point_metrics(sampled)
        for name in BOOTSTRAP_METRICS:
            replicates[name].append(float(values[name]))

    edit_fractions = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    return {
        "pairs": len(pairs),
        "images": len(pairs) * 2,
        "localization_space": localization_space,
        "mask_threshold": FIXED_MASK_THRESHOLD,
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "edit_fraction": {
            "min": float(np.min(edit_fractions)),
            "median": float(np.median(edit_fractions)),
            "mean": float(np.mean(edit_fractions)),
            "max": float(np.max(edit_fractions)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }


__all__ = [
    "BOOTSTRAP_METRICS",
    "FIXED_MASK_THRESHOLD",
    "binary_pixel_metrics",
    "summarize_catnet_pair_slice",
    "summarize_catnet_results",
]
