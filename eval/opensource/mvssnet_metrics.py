from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from eval.opensource.maskclip_metrics import descriptive, finite_float, safe_div


FIXED_THRESHOLD = 0.5
LOCALIZATION_SPACES = ("native", "model_512")
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
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "accuracy_at_0_5",
    "balanced_accuracy_at_0_5",
    "image_f1_at_0_5",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "official_png_auroc",
    "official_png_average_precision",
    "official_png_tpr_at_fpr_5_percent",
    "official_png_accuracy_at_0_5",
    "official_png_balanced_accuracy_at_0_5",
    "official_png_image_f1_at_0_5",
    "official_png_paired_ranking_accuracy",
    "official_png_paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_false_positive_area_fraction_macro_at_0_5",
    "real_false_positive_area_fraction_micro_at_0_5",
)


def _require_fixed_threshold(threshold: float, *, label: str) -> float:
    value = float(threshold)
    if not math.isclose(value, FIXED_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"MVSS-Net {label} uses the fixed {FIXED_THRESHOLD} threshold, "
            f"not {value}"
        )
    return FIXED_THRESHOLD


def _mcc_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def binary_pixel_metrics_strict(
    score_map: np.ndarray,
    target: np.ndarray,
    threshold: float = FIXED_THRESHOLD,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    """Calculate pixel metrics with MVSS-Net's strict ``score > threshold``.

    The official MVSS-Net evaluator uses a strict comparison.  In particular,
    an exact probability of 0.5 is negative.  This differs from the shared
    benchmark helper, which uses ``>=``.
    """

    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(
            f"score/target shape mismatch: {scores.shape} != {truth.shape}"
        )
    if scores.size == 0:
        raise ValueError("score map is empty")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("score map falls outside [0, 1]")

    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold is not finite")
    prediction = scores > threshold_value
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )

    return {
        "threshold": threshold_value,
        "threshold_operator": ">",
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
        "pixel_ap": ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


def _tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, float | None]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    eligible = np.flatnonzero(false_positive_rate <= target_fpr)
    best_tpr = float(np.max(true_positive_rate[eligible]))
    tied = eligible[true_positive_rate[eligible] == best_tpr]
    # Prefer the lowest finite threshold among equally good operating points.
    # It is the least conservative threshold that still respects the FPR cap.
    finite_tied = [
        int(index)
        for index in tied
        if finite_float(thresholds[int(index)]) is not None
    ]
    best_index = finite_tied[-1] if finite_tied else int(tied[-1])
    return best_tpr, finite_float(thresholds[best_index])


def _detection_from_arrays(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("detection labels and scores must be one-dimensional")
    if labels_array.shape != scores_array.shape:
        raise ValueError("detection label/score length mismatch")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("detection labels must be binary")
    if not np.isfinite(scores_array).all():
        raise ValueError("detection scores contain non-finite values")
    if (
        scores_array.size
        and (
            float(np.min(scores_array)) < 0.0
            or float(np.max(scores_array)) > 1.0
        )
    ):
        raise ValueError("detection scores fall outside [0, 1]")

    predictions = scores_array > threshold
    positive = labels_array == 1
    negative = ~positive
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & negative))
    fn = int(np.count_nonzero(~predictions & positive))
    tn = int(np.count_nonzero(~predictions & negative))
    accuracy = (
        float(np.mean(predictions == positive))
        if labels_array.size
        else None
    )
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    result: dict[str, Any] = {
        "valid_images": int(labels_array.size),
        "threshold": threshold,
        "threshold_operator": ">",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn) or 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    if positive.any() and negative.any():
        tpr, tpr_threshold = _tpr_at_fpr(
            labels_array,
            scores_array,
            0.05,
        )
        result.update(
            {
                "auroc": float(roc_auc_score(labels_array, scores_array)),
                "average_precision": float(
                    average_precision_score(labels_array, scores_array)
                ),
                "tpr_at_fpr_5_percent": tpr,
                "tpr_at_fpr_5_percent_threshold": tpr_threshold,
            }
        )
    return result


def image_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    *,
    score_key: str = "score",
) -> dict[str, Any]:
    """Summarize T1 detection using a selected top-level row score."""

    threshold_value = _require_fixed_threshold(
        threshold,
        label="T1 detection",
    )
    valid: list[tuple[int, float]] = []
    for row in rows:
        score = finite_float(row.get(score_key))
        if (
            row.get("status") == "ok"
            and row.get("label") in (0, 1)
            and score is not None
        ):
            valid.append((int(row["label"]), score))
    labels = np.asarray([label for label, _ in valid], dtype=np.int64)
    scores = np.asarray([score for _, score in valid], dtype=np.float64)
    result = _detection_from_arrays(
        labels,
        scores,
        threshold=threshold_value,
    )
    result["score_key"] = score_key
    return result


# Short alias for callers that do not need to emphasize the strict comparator.
image_detection_metrics = image_detection_metrics_strict


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
    mask_threshold: float,
) -> dict[str, Any]:
    threshold = _require_fixed_threshold(
        mask_threshold,
        label="T2 localization",
    )
    for index, row in enumerate(metric_rows):
        row_threshold = finite_float(row.get("threshold"))
        if row_threshold is not None and not math.isclose(
            row_threshold,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"localization row {index} threshold {row_threshold} "
                f"!= fixed threshold {threshold}"
            )
        operator = row.get("threshold_operator")
        if operator is not None and operator != ">":
            raise ValueError(
                f"localization row {index} uses {operator!r}, expected '>'"
            )

    counts = {
        name: sum(int(row.get(name, 0)) for row in metric_rows)
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
    predicted_fraction_micro = safe_div(predicted_positive, pixels)
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
            "threshold_operator": ">",
            **{
                metric: _mean_finite(metric_rows, metric)
                for metric in THRESHOLDED_MACRO_METRICS
            },
            "false_positive_area_fraction": _mean_finite(
                metric_rows,
                "predicted_positive_fraction",
            ),
        },
        "micro_at_threshold": {
            "threshold": threshold,
            "threshold_operator": ">",
            **counts,
            "pixels": pixels,
            "target_positive_pixels": tp + fn,
            "predicted_positive_pixels": predicted_positive,
            "predicted_positive_fraction": predicted_fraction_micro,
            "false_positive_area_fraction": predicted_fraction_micro,
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


def _score_summary(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(score_key))
        if value is not None:
            by_kind[str(row.get("kind"))].append(value)
    return {
        kind: descriptive(values)
        for kind, values in sorted(by_kind.items())
    }


def _paired_scores(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
) -> tuple[list[float], float | None]:
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = finite_float(row.get(score_key))
        if value is not None and row.get("kind") in ("real", "forged"):
            by_task[str(row.get("task_id"))][str(row["kind"])] = value
    deltas = [
        values["forged"] - values["real"]
        for values in by_task.values()
        if "real" in values and "forged" in values
    ]
    ranking = (
        sum(delta > 0 for delta in deltas) / len(deltas)
        if deltas
        else None
    )
    return deltas, ranking


def summarize_mvssnet_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    classification_threshold: float = FIXED_THRESHOLD,
    mask_threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Summarize MVSS-Net T1/T2 results using the official strict threshold."""

    classification_threshold = _require_fixed_threshold(
        classification_threshold,
        label="T1 detection",
    )
    mask_threshold = _require_fixed_threshold(
        mask_threshold,
        label="T2 localization",
    )
    latest = {
        str(row["id"]): row
        for row in rows
        if isinstance(row.get("id"), str)
    }
    expected_ids = [str(row["sample_id"]) for row in expected_rows]
    selected = [latest[row_id] for row_id in expected_ids if row_id in latest]
    valid = [row for row in selected if row.get("status") == "ok"]

    score_deltas, ranking = _paired_scores(valid, score_key="score")
    official_deltas, official_ranking = _paired_scores(
        valid,
        score_key="official_png_score",
    )
    forged: dict[str, Any] = {}
    real: dict[str, Any] = {}
    for space in LOCALIZATION_SPACES:
        forged[space] = _aggregate_localization(
            _localization_rows(
                valid,
                kind="forged",
                localization_space=space,
            ),
            mask_threshold=mask_threshold,
        )
        real[space] = _aggregate_localization(
            _localization_rows(
                valid,
                kind="real",
                localization_space=space,
            ),
            mask_threshold=mask_threshold,
        )

    return {
        "schema_version": "opensource_summary_v1",
        "task_scope": {
            "primary_detection_score": "score",
            "primary_detection_semantics": "raw_GMP_model_512",
            "secondary_detection_score": "official_png_score",
            "primary_localization_space": "native",
            "primary_localization_semantics": "official_uint8_div_255",
            "secondary_localization_space": "model_512",
            "classification_threshold": classification_threshold,
            "mask_threshold": mask_threshold,
            "threshold_operator": ">",
        },
        "coverage": {
            "expected_images": len(expected_rows),
            "result_images": len(selected),
            "valid_images": len(valid),
            "error_images": len(selected) - len(valid),
            "missing_images": len(expected_rows) - len(selected),
        },
        "score_by_kind": _score_summary(valid, score_key="score"),
        "official_png_score_by_kind": _score_summary(
            valid,
            score_key="official_png_score",
        ),
        "paired_score_delta": descriptive(score_deltas),
        "paired_ranking_accuracy": ranking,
        "official_png_paired_score_delta": descriptive(official_deltas),
        "official_png_paired_ranking_accuracy": official_ranking,
        "detection": image_detection_metrics_strict(
            valid,
            classification_threshold,
            score_key="score",
        ),
        "official_png_detection": image_detection_metrics_strict(
            valid,
            classification_threshold,
            score_key="official_png_score",
        ),
        "localization_forged": forged,
        "localization_real": real,
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
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} falls outside [0, 1]")
    return result


def _pair_slice_arrays(
    pairs: list[Any],
    *,
    localization_space: str,
) -> dict[str, np.ndarray]:
    real_rows = [_pair_row(pair, "real") for pair in pairs]
    forged_rows = [_pair_row(pair, "forged") for pair in pairs]
    real_metrics = [
        _pair_metrics(
            pair,
            kind="real",
            localization_space=localization_space,
        )
        for pair in pairs
    ]
    forged_metrics = [
        _pair_metrics(
            pair,
            kind="forged",
            localization_space=localization_space,
        )
        for pair in pairs
    ]
    for kind, metric_rows in (
        ("real", real_metrics),
        ("forged", forged_metrics),
    ):
        for index, row in enumerate(metric_rows):
            threshold = finite_float(row.get("threshold"))
            if threshold is not None:
                _require_fixed_threshold(
                    threshold,
                    label=f"{kind} T2 localization",
                )
            operator = row.get("threshold_operator")
            if operator is not None and operator != ">":
                raise ValueError(
                    f"{kind} pair {index} uses {operator!r}, expected '>'"
                )
            if int(row.get("pixels", 0)) <= 0:
                raise ValueError(f"{kind} pair {index} has no evaluated pixels")

    return {
        "real_score": np.asarray(
            [
                _required_float(row.get("score"), "real raw GMP score")
                for row in real_rows
            ],
            dtype=np.float64,
        ),
        "forged_score": np.asarray(
            [
                _required_float(row.get("score"), "forged raw GMP score")
                for row in forged_rows
            ],
            dtype=np.float64,
        ),
        "real_official_png_score": np.asarray(
            [
                _required_float(
                    row.get("official_png_score"),
                    "real official PNG score",
                )
                for row in real_rows
            ],
            dtype=np.float64,
        ),
        "forged_official_png_score": np.asarray(
            [
                _required_float(
                    row.get("official_png_score"),
                    "forged official PNG score",
                )
                for row in forged_rows
            ],
            dtype=np.float64,
        ),
        "pixel_ap": np.asarray(
            [
                _required_float(
                    row.get("pixel_ap"),
                    "forged pixel AP",
                )
                for row in forged_metrics
            ],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [
                _required_float(row.get("f1"), "forged pixel F1")
                for row in forged_metrics
            ],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [
                _required_float(row.get("iou"), "forged pixel IoU")
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
        "forged_pixels": np.asarray(
            [int(row["pixels"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "forged_target_positive_pixels": np.asarray(
            [
                int(row["target_positive_pixels"])
                for row in forged_metrics
            ],
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
                for row in real_metrics
            ],
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(
            [int(row["pixels"]) for row in real_metrics],
            dtype=np.int64,
        ),
        "real_predicted_positive_fraction": np.asarray(
            [
                _required_float(
                    row.get("predicted_positive_fraction"),
                    "real false-positive area fraction",
                )
                for row in real_metrics
            ],
            dtype=np.float64,
        ),
    }


def _paired_detection_point(
    real: np.ndarray,
    forged: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(real.size, dtype=np.int64),
            np.ones(forged.size, dtype=np.int64),
        ]
    )
    scores = np.concatenate([real, forged])
    detection = _detection_from_arrays(
        labels,
        scores,
        threshold=FIXED_THRESHOLD,
    )
    return {
        f"{prefix}auroc": float(detection["auroc"]),
        f"{prefix}average_precision": float(
            detection["average_precision"]
        ),
        f"{prefix}tpr_at_fpr_5_percent": float(
            detection["tpr_at_fpr_5_percent"]
        ),
        f"{prefix}accuracy_at_0_5": float(detection["accuracy"]),
        f"{prefix}balanced_accuracy_at_0_5": float(
            detection["balanced_accuracy"]
        ),
        f"{prefix}image_f1_at_0_5": float(detection["f1"]),
        f"{prefix}paired_ranking_accuracy": float(
            np.mean(forged > real)
        ),
        f"{prefix}paired_score_delta_mean": float(
            np.mean(forged - real)
        ),
    }


def _pair_slice_point_metrics(
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    point = _paired_detection_point(
        arrays["real_score"],
        arrays["forged_score"],
    )
    point.update(
        _paired_detection_point(
            arrays["real_official_png_score"],
            arrays["forged_official_png_score"],
            prefix="official_png_",
        )
    )
    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    real_positive = int(
        np.sum(arrays["real_predicted_positive_pixels"])
    )
    real_pixels = int(np.sum(arrays["real_pixels"]))
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    if f1_denominator <= 0 or iou_denominator <= 0:
        raise ValueError("forged localization slice has no positive target pixels")
    if real_pixels <= 0:
        raise ValueError("real localization slice has no evaluated pixels")
    point.update(
        {
            "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
            "pixel_f1_macro_at_0_5": float(
                np.mean(arrays["pixel_f1"])
            ),
            "pixel_iou_macro_at_0_5": float(
                np.mean(arrays["pixel_iou"])
            ),
            "pixel_f1_micro_at_0_5": 2.0 * tp / f1_denominator,
            "pixel_iou_micro_at_0_5": tp / iou_denominator,
            "real_false_positive_area_fraction_macro_at_0_5": float(
                np.mean(arrays["real_predicted_positive_fraction"])
            ),
            "real_false_positive_area_fraction_micro_at_0_5": (
                real_positive / real_pixels
            ),
        }
    )
    return point


def _percentile_ci(values: Iterable[float]) -> list[float]:
    values_list = list(values)
    if not values_list:
        raise ValueError("cannot calculate a confidence interval from no values")
    return [
        float(np.percentile(values_list, 2.5)),
        float(np.percentile(values_list, 97.5)),
    ]


def _sign_test(delta: np.ndarray) -> dict[str, Any]:
    wins = int(np.count_nonzero(delta > 0))
    losses = int(np.count_nonzero(delta < 0))
    ties = int(np.count_nonzero(delta == 0))
    non_ties = wins + losses
    if non_ties:
        lower_tail = min(wins, losses)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(non_ties, k) for k in range(lower_tail + 1))
            / (2**non_ties),
        )
    else:
        p_value = 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "two_sided_exact_p": p_value,
    }


def summarize_mvssnet_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    """Pair-bootstrap T1/T2 estimates while preserving real/forged pairing."""

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

    delta = arrays["forged_score"] - arrays["real_score"]
    official_delta = (
        arrays["forged_official_png_score"]
        - arrays["real_official_png_score"]
    )
    edit_fractions = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    primary_predictions = np.concatenate(
        [
            arrays["real_score"] > FIXED_THRESHOLD,
            arrays["forged_score"] > FIXED_THRESHOLD,
        ]
    )
    official_predictions = np.concatenate(
        [
            arrays["real_official_png_score"] > FIXED_THRESHOLD,
            arrays["forged_official_png_score"] > FIXED_THRESHOLD,
        ]
    )
    labels = np.concatenate(
        [
            np.zeros(len(pairs), dtype=bool),
            np.ones(len(pairs), dtype=bool),
        ]
    )

    return {
        "pairs": len(pairs),
        "images": len(pairs) * 2,
        "localization_space": localization_space,
        "classification_threshold": FIXED_THRESHOLD,
        "mask_threshold": FIXED_THRESHOLD,
        "threshold_operator": ">",
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_0_5": {
            "tp": int(np.count_nonzero(primary_predictions & labels)),
            "fp": int(np.count_nonzero(primary_predictions & ~labels)),
            "fn": int(np.count_nonzero(~primary_predictions & labels)),
            "tn": int(np.count_nonzero(~primary_predictions & ~labels)),
        },
        "official_png_image_confusion_at_0_5": {
            "tp": int(np.count_nonzero(official_predictions & labels)),
            "fp": int(np.count_nonzero(official_predictions & ~labels)),
            "fn": int(np.count_nonzero(~official_predictions & labels)),
            "tn": int(np.count_nonzero(~official_predictions & ~labels)),
        },
        "paired_sign_test": _sign_test(delta),
        "official_png_paired_sign_test": _sign_test(official_delta),
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
    "FIXED_THRESHOLD",
    "LOCALIZATION_SPACES",
    "binary_pixel_metrics_strict",
    "image_detection_metrics",
    "image_detection_metrics_strict",
    "summarize_mvssnet_pair_slice",
    "summarize_mvssnet_results",
]
