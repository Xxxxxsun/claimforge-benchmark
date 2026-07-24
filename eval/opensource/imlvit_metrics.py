from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from eval.opensource.maskclip_metrics import descriptive, finite_float, safe_div


FIXED_MASK_THRESHOLD = 0.5
LOCALIZATION_SPACES = ("model_1024", "native")
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
THRESHOLDED_MACRO_METRICS = ("precision", "recall", "f1", "iou", "mcc")
BOOTSTRAP_METRICS = (
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_false_positive_area_fraction_macro_at_0_5",
    "real_false_positive_area_fraction_micro_at_0_5",
)


def _require_fixed_threshold(threshold: float) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("IML-ViT localization threshold is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError("IML-ViT localization threshold is not finite")
    if not math.isclose(
        value,
        FIXED_MASK_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "IML-ViT primary localization metrics use the fixed "
            f"{FIXED_MASK_THRESHOLD} threshold, not {value}"
        )
    return FIXED_MASK_THRESHOLD


def _binary_target(target: np.ndarray) -> np.ndarray:
    raw = np.asarray(target)
    if raw.dtype == np.bool_:
        return raw
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target is not a binary array") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("target contains non-finite values")
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("target contains values other than 0 and 1")
    return numeric.astype(bool)


def _mcc_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def binary_pixel_metrics_strict(
    probability_map: np.ndarray,
    target: np.ndarray,
    threshold: float = FIXED_MASK_THRESHOLD,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    """Measure IML-ViT probabilities using strict float32 ``p > 0.5``.

    IML-ViT inference emits a sigmoid probability map.  Converting to float32
    here before validation, thresholding, and AP calculation matches the
    persisted probability artifact and prevents hidden float64 boundary drift.
    AP is only defined when requested and when both target classes are present.
    Thus an authentic image's all-negative target always has ``pixel_ap=None``.
    """

    threshold_value = _require_fixed_threshold(threshold)
    try:
        scores = np.asarray(probability_map, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability map is not numeric") from exc
    truth = _binary_target(target)
    if scores.shape != truth.shape:
        raise ValueError(
            f"probability/target shape mismatch: {scores.shape} != {truth.shape}"
        )
    if scores.size == 0:
        raise ValueError("probability map is empty")
    if not np.isfinite(scores).all():
        raise ValueError("probability map contains non-finite values")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("probability map falls outside [0, 1]")

    prediction = scores > threshold_value
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    pixel_ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )

    return {
        "threshold": threshold_value,
        "threshold_operator": ">",
        "probability_dtype": "float32",
        "pixels": int(scores.size),
        "target_positive_pixels": int(np.count_nonzero(truth)),
        "predicted_positive_pixels": int(np.count_nonzero(prediction)),
        "predicted_positive_fraction": float(np.mean(prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "mcc": _mcc_from_counts(tp=tp, fp=fp, fn=fn, tn=tn),
        "pixel_ap": pixel_ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


# The short alias keeps runner call sites compact while preserving the strict
# contract in the implementation name.
binary_pixel_metrics = binary_pixel_metrics_strict


def _required_count(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is not a non-negative integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a non-negative integer") from exc
    if not math.isfinite(numeric) or numeric != integer or integer < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return integer


def _validate_localization_row(
    row: dict[str, Any],
    *,
    index: int,
    kind: str,
    space: str,
) -> None:
    threshold = row.get("threshold")
    if finite_float(threshold) is None:
        raise ValueError(
            f"{kind} {space} localization row {index} has no threshold"
        )
    _require_fixed_threshold(threshold)
    if row.get("threshold_operator") != ">":
        raise ValueError(
            f"{kind} {space} localization row {index} uses "
            f"{row.get('threshold_operator')!r}, expected '>'"
        )
    dtype = row.get("probability_dtype")
    if dtype is not None and dtype != "float32":
        raise ValueError(
            f"{kind} {space} localization row {index} uses probability "
            f"dtype {dtype!r}, expected 'float32'"
        )

    counts = {
        name: _required_count(
            row.get(name),
            label=f"{kind} {space} localization row {index} {name}",
        )
        for name in (
            "pixels",
            "target_positive_pixels",
            "predicted_positive_pixels",
            "tp",
            "fp",
            "fn",
            "tn",
        )
    }
    if counts["pixels"] <= 0:
        raise ValueError(
            f"{kind} {space} localization row {index} has no evaluated pixels"
        )
    if sum(counts[name] for name in ("tp", "fp", "fn", "tn")) != counts[
        "pixels"
    ]:
        raise ValueError(
            f"{kind} {space} localization row {index} confusion counts "
            "do not sum to pixels"
        )
    if counts["tp"] + counts["fn"] != counts["target_positive_pixels"]:
        raise ValueError(
            f"{kind} {space} localization row {index} target-positive "
            "count is inconsistent"
        )
    if counts["tp"] + counts["fp"] != counts["predicted_positive_pixels"]:
        raise ValueError(
            f"{kind} {space} localization row {index} predicted-positive "
            "count is inconsistent"
        )

    predicted_fraction = finite_float(row.get("predicted_positive_fraction"))
    expected_fraction = counts["predicted_positive_pixels"] / counts["pixels"]
    if predicted_fraction is None or not math.isclose(
        predicted_fraction,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{kind} {space} localization row {index} predicted-positive "
            "fraction is inconsistent"
        )
    for metric in ("precision", "recall", "f1", "iou", "mcc"):
        value = row.get(metric)
        if value is not None and finite_float(value) is None:
            raise ValueError(
                f"{kind} {space} localization row {index} {metric} "
                "is not finite"
            )
    for metric in ("score_mean", "score_max"):
        value = finite_float(row.get(metric))
        if value is None or value < 0.0 or value > 1.0:
            raise ValueError(
                f"{kind} {space} localization row {index} {metric} "
                "falls outside [0, 1]"
            )

    pixel_ap = row.get("pixel_ap")
    if kind == "real":
        if counts["target_positive_pixels"] != 0:
            raise ValueError(
                f"real {space} localization row {index} has positive target "
                "pixels"
            )
        if pixel_ap is not None:
            raise ValueError(
                f"real {space} localization row {index} pixel AP must be null"
            )
    else:
        if counts["target_positive_pixels"] <= 0:
            raise ValueError(
                f"forged {space} localization row {index} has no positive "
                "target pixels"
            )
        ap_value = finite_float(pixel_ap)
        if ap_value is None or ap_value < 0.0 or ap_value > 1.0:
            raise ValueError(
                f"forged {space} localization row {index} pixel AP is invalid"
            )


def _mean_finite(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := finite_float(row.get(metric))) is not None
    ]
    return statistics.fmean(values) if values else None


def _aggregate_localization(
    metric_rows: list[dict[str, Any]],
    *,
    kind: str,
    space: str,
) -> dict[str, Any]:
    for index, row in enumerate(metric_rows):
        _validate_localization_row(
            row,
            index=index,
            kind=kind,
            space=space,
        )
    counts = {
        name: sum(int(row[name]) for row in metric_rows)
        for name in ("tp", "fp", "fn", "tn")
    }
    tp, fp, fn, tn = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )
    pixels = tp + fp + fn + tn
    predicted_positive = tp + fp
    aggregate = {
        "images": len(metric_rows),
        **{
            metric: (
                None
                if kind == "real" and metric == "pixel_ap"
                else descriptive(
                    value
                    for row in metric_rows
                    if (value := finite_float(row.get(metric))) is not None
                )
            )
            for metric in LOCALIZATION_METRICS
        },
        "macro_at_threshold": {
            "threshold": FIXED_MASK_THRESHOLD,
            "threshold_operator": ">",
            **{
                metric: _mean_finite(metric_rows, metric)
                for metric in THRESHOLDED_MACRO_METRICS
            },
            "predicted_positive_fraction": _mean_finite(
                metric_rows,
                "predicted_positive_fraction",
            ),
        },
        "micro_at_threshold": {
            "threshold": FIXED_MASK_THRESHOLD,
            "threshold_operator": ">",
            **counts,
            "pixels": pixels,
            "target_positive_pixels": tp + fn,
            "predicted_positive_pixels": predicted_positive,
            "predicted_positive_fraction": safe_div(
                predicted_positive,
                pixels,
            ),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "mcc": _mcc_from_counts(tp=tp, fp=fp, fn=fn, tn=tn),
        },
    }
    if kind == "real":
        aggregate["macro_at_threshold"]["false_positive_area_fraction"] = (
            aggregate["macro_at_threshold"]["predicted_positive_fraction"]
        )
        aggregate["micro_at_threshold"]["false_positive_area_fraction"] = (
            aggregate["micro_at_threshold"]["predicted_positive_fraction"]
        )
    return aggregate


def _localization_metrics(
    row: dict[str, Any],
    *,
    kind: str,
    space: str,
    index: int,
) -> dict[str, Any]:
    localization = row.get("localization")
    if not isinstance(localization, dict):
        raise ValueError(f"{kind} result row {index} has no localization object")
    metrics = localization.get(space)
    if not isinstance(metrics, dict):
        raise ValueError(
            f"{kind} result row {index} has no {space!r} localization metrics"
        )
    return metrics


def _pair_row(pair: Any, kind: str) -> dict[str, Any]:
    row = pair.get(kind) if isinstance(pair, Mapping) else getattr(pair, kind, None)
    if not isinstance(row, dict):
        raise ValueError(f"pair has no {kind!r} result row")
    return row


def _required_probability(value: Any, *, label: str) -> float:
    result = finite_float(value)
    if result is None or result < 0.0 or result > 1.0:
        raise ValueError(f"{label} is not a probability")
    return result


def _pair_slice_arrays(
    pairs: list[Any],
    *,
    localization_space: str,
) -> dict[str, np.ndarray]:
    forged: list[dict[str, Any]] = []
    real: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        forged_row = _pair_row(pair, "forged")
        real_row = _pair_row(pair, "real")
        forged_metrics = _localization_metrics(
            forged_row,
            kind="forged",
            space=localization_space,
            index=index,
        )
        real_metrics = _localization_metrics(
            real_row,
            kind="real",
            space=localization_space,
            index=index,
        )
        _validate_localization_row(
            forged_metrics,
            index=index,
            kind="forged",
            space=localization_space,
        )
        _validate_localization_row(
            real_metrics,
            index=index,
            kind="real",
            space=localization_space,
        )
        forged.append(forged_metrics)
        real.append(real_metrics)

    return {
        "pixel_ap": np.asarray(
            [
                _required_probability(
                    row.get("pixel_ap"),
                    label="forged pixel AP",
                )
                for row in forged
            ],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [
                _required_probability(
                    row.get("f1"),
                    label="forged pixel F1",
                )
                for row in forged
            ],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [
                _required_probability(
                    row.get("iou"),
                    label="forged pixel IoU",
                )
                for row in forged
            ],
            dtype=np.float64,
        ),
        "tp": np.asarray([int(row["tp"]) for row in forged], dtype=np.int64),
        "fp": np.asarray([int(row["fp"]) for row in forged], dtype=np.int64),
        "fn": np.asarray([int(row["fn"]) for row in forged], dtype=np.int64),
        "tn": np.asarray([int(row["tn"]) for row in forged], dtype=np.int64),
        "forged_pixels": np.asarray(
            [int(row["pixels"]) for row in forged],
            dtype=np.int64,
        ),
        "forged_target_positive_pixels": np.asarray(
            [int(row["target_positive_pixels"]) for row in forged],
            dtype=np.int64,
        ),
        "real_predicted_positive_pixels": np.asarray(
            [int(row["predicted_positive_pixels"]) for row in real],
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(
            [int(row["pixels"]) for row in real],
            dtype=np.int64,
        ),
        "real_false_positive_area_fraction": np.asarray(
            [
                _required_probability(
                    row.get("predicted_positive_fraction"),
                    label="real false-positive area fraction",
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
    real_positive = int(np.sum(arrays["real_predicted_positive_pixels"]))
    real_pixels = int(np.sum(arrays["real_pixels"]))
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    if f1_denominator <= 0 or iou_denominator <= 0:
        raise ValueError("forged localization slice has no positive target pixels")
    if real_pixels <= 0:
        raise ValueError("real localization slice has no evaluated pixels")
    return {
        "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
        "pixel_f1_macro_at_0_5": float(np.mean(arrays["pixel_f1"])),
        "pixel_iou_macro_at_0_5": float(np.mean(arrays["pixel_iou"])),
        "pixel_f1_micro_at_0_5": 2.0 * tp / f1_denominator,
        "pixel_iou_micro_at_0_5": tp / iou_denominator,
        "real_false_positive_area_fraction_macro_at_0_5": float(
            np.mean(arrays["real_false_positive_area_fraction"])
        ),
        "real_false_positive_area_fraction_micro_at_0_5": (
            real_positive / real_pixels
        ),
    }


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot calculate a confidence interval from no values")
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def summarize_imlvit_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    """Pair-bootstrap IML-ViT T2 estimates without deriving a T1 score."""

    if not pairs:
        raise ValueError("pair slice is empty")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if localization_space not in LOCALIZATION_SPACES:
        raise ValueError(
            f"unsupported localization space {localization_space!r}"
        )

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

    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    tn = int(np.sum(arrays["tn"]))
    edit_fractions = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    return {
        "pairs": len(pairs),
        "images": len(pairs) * 2,
        "localization_space": localization_space,
        "bootstrap_samples": iterations,
        "seed": int(seed),
        "mask_threshold": FIXED_MASK_THRESHOLD,
        "threshold_operator": ">",
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "forged_micro_at_threshold": {
            "threshold": FIXED_MASK_THRESHOLD,
            "threshold_operator": ">",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pixels": tp + fp + fn + tn,
            "target_positive_pixels": tp + fn,
            "predicted_positive_pixels": tp + fp,
            "predicted_positive_fraction": safe_div(
                tp + fp,
                tp + fp + fn + tn,
            ),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "mcc": _mcc_from_counts(tp=tp, fp=fp, fn=fn, tn=tn),
        },
        "edit_fraction": {
            "min": float(np.min(edit_fractions)),
            "median": float(np.median(edit_fractions)),
            "mean": float(np.mean(edit_fractions)),
            "max": float(np.max(edit_fractions)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }


def _select_rows(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    latest = {
        str(row["id"]): row
        for row in rows
        if isinstance(row.get("id"), str)
    }
    if expected_rows is None:
        expected_ids = list(latest)
    else:
        expected_ids = [str(row["sample_id"]) for row in expected_rows]
    return (
        [latest[row_id] for row_id in expected_ids if row_id in latest],
        len(expected_ids),
    )


def _complete_pairs(
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid_rows:
        kind = row.get("kind")
        task_id = row.get("task_id")
        if kind not in ("real", "forged"):
            raise ValueError(f"valid result has unsupported kind {kind!r}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("valid result has no non-empty task_id")
        by_task[task_id][str(kind)] = row
    return [
        {
            "real": kinds["real"],
            "forged": kinds["forged"],
        }
        for _, kinds in sorted(by_task.items())
        if "real" in kinds and "forged" in kinds
    ]


def summarize_imlvit_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None = None,
    *,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
    bootstrap_samples: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Strict T2-only summary for IML-ViT model/native probability maps.

    The function deliberately never reads a top-level image score.  Bootstrap
    units are complete real/forged task pairs, so paired benchmark structure is
    preserved without turning a localization map into an unofficial T1 score.
    """

    threshold = _require_fixed_threshold(mask_threshold)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    selected, expected_count = _select_rows(rows, expected_rows)
    valid = [row for row in selected if row.get("status") == "ok"]

    localization_forged: dict[str, Any] = {}
    localization_real: dict[str, Any] = {}
    for space in LOCALIZATION_SPACES:
        forged_metrics = [
            _localization_metrics(
                row,
                kind="forged",
                space=space,
                index=index,
            )
            for index, row in enumerate(valid)
            if row.get("kind") == "forged"
        ]
        real_metrics = [
            _localization_metrics(
                row,
                kind="real",
                space=space,
                index=index,
            )
            for index, row in enumerate(valid)
            if row.get("kind") == "real"
        ]
        localization_forged[space] = _aggregate_localization(
            forged_metrics,
            kind="forged",
            space=space,
        )
        localization_real[space] = _aggregate_localization(
            real_metrics,
            kind="real",
            space=space,
        )

    pairs = _complete_pairs(valid)
    pair_bootstrap = {
        "bootstrap_samples": bootstrap_samples,
        "seed": int(seed),
        **{
            space: (
                summarize_imlvit_pair_slice(
                    pairs,
                    iterations=bootstrap_samples,
                    seed=seed,
                    localization_space=space,
                )
                if pairs
                else None
            )
            for space in LOCALIZATION_SPACES
        },
    }
    paired_ids = {
        id(row)
        for pair in pairs
        for row in (pair["real"], pair["forged"])
    }
    return {
        "schema_version": "opensource_summary_v1",
        "task_scope": {
            "primary_task": "T2_localization",
            "valid_for_t1": False,
            "valid_for_t2": True,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_1024",
            "localization_semantics": (
                "imlvit_sigmoid_manipulation_probability_float32"
            ),
            "probability_dtype": "float32",
            "mask_threshold": threshold,
            "threshold_operator": ">",
        },
        "coverage": {
            "expected_images": expected_count,
            "result_images": len(selected),
            "valid_images": len(valid),
            "error_images": len(selected) - len(valid),
            "missing_images": expected_count - len(selected),
        },
        "paired_coverage": {
            "complete_pairs": len(pairs),
            "paired_images": len(paired_ids),
            "unpaired_valid_images": len(valid) - len(paired_ids),
        },
        "localization_forged": localization_forged,
        "localization_real": localization_real,
        "pair_bootstrap": pair_bootstrap,
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


__all__ = [
    "BOOTSTRAP_METRICS",
    "FIXED_MASK_THRESHOLD",
    "LOCALIZATION_SPACES",
    "binary_pixel_metrics",
    "binary_pixel_metrics_strict",
    "summarize_imlvit_pair_slice",
    "summarize_imlvit_results",
]
