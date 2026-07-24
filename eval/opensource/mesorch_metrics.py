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
THRESHOLD_OPERATOR = ">"
LOCALIZATION_SPACES = ("model_512", "native")
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


def _require_fixed_threshold(threshold: Any) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mesorch localization threshold is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError("Mesorch localization threshold is not finite")
    if not math.isclose(
        value,
        FIXED_MASK_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Mesorch localization uses the fixed "
            f"{FIXED_MASK_THRESHOLD} threshold, not {value}"
        )
    return FIXED_MASK_THRESHOLD


def _binary_target(target: np.ndarray) -> np.ndarray:
    raw = np.asarray(target)
    if raw.dtype == np.bool_:
        truth = raw
    else:
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("target is not a binary array") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("target contains non-finite values")
        if not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError("target contains values other than 0 and 1")
        truth = numeric.astype(bool)
    if truth.ndim != 2:
        raise ValueError(f"target must be two-dimensional, got {truth.shape}")
    return truth


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
    """Evaluate a Mesorch probability map with strict float32 ``p > 0.5``.

    Mesorch's sigmoid output is converted to float32 before range checking,
    thresholding, and AP calculation.  This makes the metric path identical to
    the persisted float32 artifact, including values that round onto 0.5.  AP
    is defined for forged masks only; an all-negative authentic target yields
    ``pixel_ap=None``.
    """

    threshold_value = _require_fixed_threshold(threshold)
    try:
        scores = np.asarray(probability_map, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability map is not numeric") from exc
    if scores.ndim != 2:
        raise ValueError(
            f"probability map must be two-dimensional, got {scores.shape}"
        )
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
    if include_ap and truth.any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )

    return {
        "threshold": threshold_value,
        "threshold_operator": THRESHOLD_OPERATOR,
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


def _validate_metric(
    value: Any,
    expected: float | None,
    *,
    label: str,
    lower: float = 0.0,
) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"{label} must be null")
        return
    numeric = finite_float(value)
    if (
        numeric is None
        or numeric < lower
        or numeric > 1.0
        or not math.isclose(
            numeric,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{label} is inconsistent")


def _validate_localization_row(
    row: dict[str, Any],
    *,
    index: int,
    kind: str,
    space: str,
) -> None:
    _require_fixed_threshold(row.get("threshold"))
    if row.get("threshold_operator") != THRESHOLD_OPERATOR:
        raise ValueError(
            f"{kind} {space} row {index} uses "
            f"{row.get('threshold_operator')!r}, expected '>'"
        )
    if row.get("probability_dtype") != "float32":
        raise ValueError(
            f"{kind} {space} row {index} has wrong probability dtype"
        )

    counts = {
        name: _required_count(
            row.get(name),
            label=f"{kind} {space} row {index} {name}",
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
        raise ValueError(f"{kind} {space} row {index} has no pixels")
    if sum(counts[name] for name in ("tp", "fp", "fn", "tn")) != counts[
        "pixels"
    ]:
        raise ValueError(f"{kind} {space} row {index} counts do not sum")
    if counts["tp"] + counts["fn"] != counts["target_positive_pixels"]:
        raise ValueError(f"{kind} {space} row {index} target count mismatch")
    if counts["tp"] + counts["fp"] != counts["predicted_positive_pixels"]:
        raise ValueError(f"{kind} {space} row {index} prediction count mismatch")

    expected_fraction = counts["predicted_positive_pixels"] / counts["pixels"]
    fraction = finite_float(row.get("predicted_positive_fraction"))
    if (
        fraction is None
        or fraction < 0.0
        or fraction > 1.0
        or not math.isclose(
            fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{kind} {space} row {index} fraction mismatch")

    expected_metrics = {
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
        "mcc": _mcc_from_counts(
            tp=counts["tp"],
            fp=counts["fp"],
            fn=counts["fn"],
            tn=counts["tn"],
        ),
    }
    for name, expected in expected_metrics.items():
        _validate_metric(
            row.get(name),
            expected,
            label=f"{kind} {space} row {index} {name}",
            lower=-1.0 if name == "mcc" else 0.0,
        )

    score_mean = finite_float(row.get("score_mean"))
    score_max = finite_float(row.get("score_max"))
    if (
        score_mean is None
        or score_max is None
        or score_mean < 0.0
        or score_mean > 1.0
        or score_max < 0.0
        or score_max > 1.0
        or score_mean > score_max
    ):
        raise ValueError(
            f"{kind} {space} row {index} probability summary is invalid"
        )

    pixel_ap = row.get("pixel_ap")
    if kind == "real":
        if counts["target_positive_pixels"] != 0:
            raise ValueError(
                f"real {space} row {index} has positive target pixels"
            )
        if pixel_ap is not None:
            raise ValueError(
                f"real {space} row {index} pixel AP must be null"
            )
    else:
        ap = finite_float(pixel_ap)
        if counts["target_positive_pixels"] <= 0:
            raise ValueError(
                f"forged {space} row {index} has no positive target pixels"
            )
        if ap is None or ap < 0.0 or ap > 1.0:
            raise ValueError(
                f"forged {space} row {index} pixel AP is invalid"
            )


def _validate_nonnegative_optional(value: Any, *, label: str) -> None:
    if value is None:
        return
    numeric = finite_float(value)
    if numeric is None or numeric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def _validate_valid_result_row(
    row: dict[str, Any],
    *,
    index: int,
) -> None:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"valid row {index} has no non-empty id")
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"valid row {index} has no non-empty task_id")
    kind = row.get("kind")
    if kind not in ("real", "forged"):
        raise ValueError(f"valid row {index} has unsupported kind {kind!r}")
    if row.get("label") != int(kind == "forged"):
        raise ValueError(f"valid row {index} kind/label mismatch")
    _validate_nonnegative_optional(
        row.get("latency_ms"),
        label=f"valid row {index} latency_ms",
    )
    _validate_nonnegative_optional(
        row.get("peak_cuda_memory_bytes"),
        label=f"valid row {index} peak_cuda_memory_bytes",
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
            "threshold_operator": THRESHOLD_OPERATOR,
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
            "threshold_operator": THRESHOLD_OPERATOR,
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
        raise ValueError(f"{kind} row {index} has no localization object")
    metrics = localization.get(space)
    if not isinstance(metrics, dict):
        raise ValueError(f"{kind} row {index} has no {space} metrics")
    return metrics


def _row_id(row: dict[str, Any], *, index: int) -> str:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"result row {index} has no non-empty id")
    return row_id


def _expected_id(row: dict[str, Any], *, index: int) -> str:
    row_id = row.get("sample_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"expected row {index} has no non-empty sample_id")
    return row_id


def _validate_expected_identity(
    result: dict[str, Any],
    expected: dict[str, Any],
    *,
    sample_id: str,
) -> None:
    comparisons = (
        ("task_id", "task_id"),
        ("kind", "kind"),
        ("label", "label"),
        ("domain", "domain"),
    )
    for result_key, expected_key in comparisons:
        if (
            expected_key in expected
            and result.get(result_key) != expected.get(expected_key)
        ):
            raise ValueError(
                f"result {sample_id} {result_key} does not match expected row"
            )


def _select_rows(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"result row {index} is not an object")
        latest[_row_id(row, index=index)] = row

    if expected_rows is None:
        return list(latest.values()), len(latest)

    expected_by_id: dict[str, dict[str, Any]] = {}
    for index, expected in enumerate(expected_rows):
        if not isinstance(expected, dict):
            raise ValueError(f"expected row {index} is not an object")
        sample_id = _expected_id(expected, index=index)
        if sample_id in expected_by_id:
            raise ValueError(f"duplicate expected sample_id {sample_id}")
        expected_by_id[sample_id] = expected

    selected: list[dict[str, Any]] = []
    for sample_id, expected in expected_by_id.items():
        result = latest.get(sample_id)
        if result is None:
            continue
        _validate_expected_identity(result, expected, sample_id=sample_id)
        selected.append(result)
    return selected, len(expected_by_id)


def _complete_pairs(
    valid_rows: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid_rows:
        kind = str(row["kind"])
        task_id = str(row["task_id"])
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate {kind} row within task {task_id}")
        by_task[task_id][kind] = row

    pairs: list[dict[str, dict[str, Any]]] = []
    for task_id, kinds in sorted(by_task.items()):
        if set(kinds) != {"real", "forged"}:
            continue
        real = kinds["real"]
        forged = kinds["forged"]
        if real["id"] == forged["id"]:
            raise ValueError(f"pair {task_id} reuses one sample id")
        if (
            "domain" in real
            and "domain" in forged
            and real.get("domain") != forged.get("domain")
        ):
            raise ValueError(f"pair {task_id} has mismatched domains")
        pairs.append({"real": real, "forged": forged})
    return pairs


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
    real_metrics: list[dict[str, Any]] = []
    forged_metrics: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for index, pair in enumerate(pairs):
        real = _pair_row(pair, "real")
        forged = _pair_row(pair, "forged")
        for kind, row in (("real", real), ("forged", forged)):
            if row.get("status") != "ok":
                raise ValueError(f"{kind} pair {index} is not status ok")
            if row.get("kind") != kind:
                raise ValueError(f"{kind} pair {index} has wrong kind")
            if row.get("label") != int(kind == "forged"):
                raise ValueError(f"{kind} pair {index} has wrong label")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{kind} pair {index} has no sample id")
            if row_id in seen_sample_ids:
                raise ValueError(f"duplicate sample id {row_id} in pair slice")
            seen_sample_ids.add(row_id)

        real_task_id = real.get("task_id")
        forged_task_id = forged.get("task_id")
        if (
            not isinstance(real_task_id, str)
            or not real_task_id
            or not isinstance(forged_task_id, str)
            or not forged_task_id
            or real_task_id != forged_task_id
        ):
            raise ValueError(f"pair {index} has mismatched task IDs")
        if real_task_id in seen_task_ids:
            raise ValueError(f"duplicate task pair {real_task_id}")
        seen_task_ids.add(real_task_id)
        if (
            "domain" in real
            and "domain" in forged
            and real.get("domain") != forged.get("domain")
        ):
            raise ValueError(f"pair {index} has mismatched domains")

        real_metric = _localization_metrics(
            real,
            kind="real",
            space=localization_space,
            index=index,
        )
        forged_metric = _localization_metrics(
            forged,
            kind="forged",
            space=localization_space,
            index=index,
        )
        _validate_localization_row(
            real_metric,
            index=index,
            kind="real",
            space=localization_space,
        )
        _validate_localization_row(
            forged_metric,
            index=index,
            kind="forged",
            space=localization_space,
        )
        real_metrics.append(real_metric)
        forged_metrics.append(forged_metric)

    return {
        "pixel_ap": np.asarray(
            [
                _required_probability(
                    row.get("pixel_ap"),
                    label="forged pixel AP",
                )
                for row in forged_metrics
            ],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [
                _required_probability(row.get("f1"), label="forged pixel F1")
                for row in forged_metrics
            ],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [
                _required_probability(row.get("iou"), label="forged pixel IoU")
                for row in forged_metrics
            ],
            dtype=np.float64,
        ),
        "tp": np.asarray(
            [int(row["tp"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "fp": np.asarray(
            [int(row["fp"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "fn": np.asarray(
            [int(row["fn"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "tn": np.asarray(
            [int(row["tn"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "forged_pixels": np.asarray(
            [int(row["pixels"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "forged_target_positive_pixels": np.asarray(
            [int(row["target_positive_pixels"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "real_predicted_positive_pixels": np.asarray(
            [int(row["predicted_positive_pixels"]) for row in real_metrics],
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(
            [int(row["pixels"]) for row in real_metrics],
            dtype=np.int64,
        ),
        "real_false_positive_area_fraction": np.asarray(
            [
                _required_probability(
                    row.get("predicted_positive_fraction"),
                    label="real false-positive area fraction",
                )
                for row in real_metrics
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


def _require_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{label} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{label} must be positive")
    return integer


def _require_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def summarize_mesorch_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    """Pair-bootstrap Mesorch T2 estimates without deriving a T1 score."""

    if not pairs:
        raise ValueError("pair slice is empty")
    iterations_value = _require_positive_integer(
        iterations,
        label="bootstrap iterations",
    )
    seed_value = _require_integer(seed, label="bootstrap seed")
    if localization_space not in LOCALIZATION_SPACES:
        raise ValueError(
            f"unsupported localization space {localization_space!r}"
        )

    arrays = _pair_slice_arrays(
        pairs,
        localization_space=localization_space,
    )
    point = _pair_slice_point_metrics(arrays)
    rng = np.random.default_rng(seed_value)
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    for _ in range(iterations_value):
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
        "bootstrap_samples": iterations_value,
        "seed": seed_value,
        "mask_threshold": FIXED_MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "forged_micro_at_threshold": {
            "threshold": FIXED_MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
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


def summarize_mesorch_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None = None,
    *,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
    bootstrap_samples: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Build a strict T2-only Mesorch summary in model and native spaces.

    Top-level image scores are deliberately never read.  Bootstrap units are
    complete authentic/forged task pairs, preserving benchmark dependence
    without constructing an unofficial image-level detector from a mask.
    """

    threshold = _require_fixed_threshold(mask_threshold)
    bootstrap_samples_value = _require_positive_integer(
        bootstrap_samples,
        label="bootstrap_samples",
    )
    seed_value = _require_integer(seed, label="bootstrap seed")
    selected, expected_count = _select_rows(rows, expected_rows)
    valid = [row for row in selected if row.get("status") == "ok"]
    for index, row in enumerate(valid):
        _validate_valid_result_row(row, index=index)

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
        "bootstrap_samples": bootstrap_samples_value,
        "seed": seed_value,
        **{
            space: (
                summarize_mesorch_pair_slice(
                    pairs,
                    iterations=bootstrap_samples_value,
                    seed=seed_value,
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
            "auxiliary_localization_space": "model_512",
            "localization_semantics": (
                "mesorch_sigmoid_manipulation_probability_float32"
            ),
            "probability_dtype": "float32",
            "mask_threshold": threshold,
            "threshold_operator": THRESHOLD_OPERATOR,
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
            float(row["latency_ms"])
            for row in valid
            if row.get("latency_ms") is not None
        ),
        "peak_cuda_memory_bytes": descriptive(
            float(row["peak_cuda_memory_bytes"])
            for row in valid
            if row.get("peak_cuda_memory_bytes") is not None
        ),
    }


__all__ = [
    "BOOTSTRAP_METRICS",
    "FIXED_MASK_THRESHOLD",
    "LOCALIZATION_SPACES",
    "THRESHOLD_OPERATOR",
    "binary_pixel_metrics",
    "binary_pixel_metrics_strict",
    "summarize_mesorch_pair_slice",
    "summarize_mesorch_results",
]
