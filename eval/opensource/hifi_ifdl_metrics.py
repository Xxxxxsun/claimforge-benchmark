from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from eval.opensource.maskclip_metrics import descriptive, finite_float, safe_div


FIXED_CLASSIFICATION_THRESHOLD = 0.5
FIXED_MASK_THRESHOLD = 2.3
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD_OPERATOR = ">="
LOCALIZATION_SPACES = ("model_256", "native")
FINE_CLASS_NAMES = (
    "authentic",
    "splice",
    "inpainting",
    "copy_move",
    "faceshifter",
    "stgan",
    "star2",
    "hisd",
    "stylegan2",
    "stylegan3",
    "ddpm",
    "ddim",
    "d_latent",
    "glide",
)
HIERARCHY_LOGIT_SIZES = {
    "out0_coarse_3class": 3,
    "out1_5class": 5,
    "out2_7class": 7,
    "out3_fine_14class": 14,
}
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
    "official_argmax_accuracy",
    "official_argmax_balanced_accuracy",
    "official_argmax_image_f1",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_2_3",
    "pixel_iou_macro_at_2_3",
    "pixel_f1_micro_at_2_3",
    "pixel_iou_micro_at_2_3",
    "real_false_positive_area_fraction_macro_at_2_3",
    "real_false_positive_area_fraction_micro_at_2_3",
)


def _require_fixed_threshold(
    threshold: Any,
    *,
    expected: float,
    label: str,
) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HiFi-IFDL {label} threshold is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"HiFi-IFDL {label} uses fixed threshold {expected}, not {value}"
        )
    return expected


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


def binary_distance_metrics_strict(
    distance_map: np.ndarray,
    target: np.ndarray,
    threshold: float = FIXED_MASK_THRESHOLD,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    """Measure native HiFi distance scores using official ``d >= 2.3``."""

    threshold_value = _require_fixed_threshold(
        threshold,
        expected=FIXED_MASK_THRESHOLD,
        label="T2 localization",
    )
    try:
        raw_scores = np.asarray(distance_map)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance map is not numeric") from exc
    if raw_scores.dtype != np.float32:
        raise ValueError(
            f"distance map dtype must be float32, got {raw_scores.dtype}"
        )
    if raw_scores.ndim != 2:
        raise ValueError(
            f"distance map must be two-dimensional, got {raw_scores.shape}"
        )
    scores = np.asarray(raw_scores, dtype=np.float32)
    truth = _binary_target(target)
    if scores.shape != truth.shape:
        raise ValueError(
            f"distance/target shape mismatch: {scores.shape} != {truth.shape}"
        )
    if scores.size == 0:
        raise ValueError("distance map is empty")
    if not np.isfinite(scores).all():
        raise ValueError("distance map contains non-finite values")
    if float(scores.min()) < 0.0:
        raise ValueError("distance map contains negative values")

    prediction = scores >= threshold_value
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
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "score_semantics": "hifi_hypersphere_euclidean_distance",
        "score_dtype": "float32",
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


binary_pixel_metrics = binary_distance_metrics_strict


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
    finite_tied = [
        int(index)
        for index in tied
        if finite_float(thresholds[int(index)]) is not None
    ]
    best_index = finite_tied[-1] if finite_tied else int(tied[-1])
    return best_tpr, finite_float(thresholds[best_index])


def _binary_detection_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=bool)
    if labels_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("detection arrays must be one-dimensional")
    if labels_array.shape != predictions_array.shape:
        raise ValueError("detection array length mismatch")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("detection labels must be binary")
    positive = labels_array == 1
    negative = ~positive
    tp = int(np.count_nonzero(predictions_array & positive))
    fp = int(np.count_nonzero(predictions_array & negative))
    fn = int(np.count_nonzero(~predictions_array & positive))
    tn = int(np.count_nonzero(~predictions_array & negative))
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "valid_images": int(labels_array.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (
            float(np.mean(predictions_array == positive))
            if labels_array.size
            else None
        ),
        "balanced_accuracy": balanced_accuracy,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn) or 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


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
    if scores_array.size and (
        float(scores_array.min()) < 0.0 or float(scores_array.max()) > 1.0
    ):
        raise ValueError("detection scores fall outside [0, 1]")

    binary = _binary_detection_from_arrays(
        labels_array,
        scores_array > threshold,
    )
    result = {
        **binary,
        "threshold": threshold,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
    }
    positive = labels_array == 1
    negative = ~positive
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
    threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    *,
    score_key: str = "score",
) -> dict[str, Any]:
    threshold_value = _require_fixed_threshold(
        threshold,
        expected=FIXED_CLASSIFICATION_THRESHOLD,
        label="T1 detection",
    )
    valid: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        if row.get("label") not in (0, 1):
            raise ValueError(f"valid T1 row {index} has invalid label")
        score = finite_float(row.get(score_key))
        if score is None:
            raise ValueError(f"valid T1 row {index} has no finite {score_key}")
        valid.append((int(row["label"]), score))
    result = _detection_from_arrays(
        np.asarray([label for label, _ in valid], dtype=np.int64),
        np.asarray([score for _, score in valid], dtype=np.float64),
        threshold=threshold_value,
    )
    result["score_key"] = score_key
    result["score_semantics"] = (
        "one_minus_softmax_probability_of_hifi_fine_class_0_authentic"
    )
    return result


image_detection_metrics = image_detection_metrics_strict


def official_argmax_detection_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[bool] = []
    for index, row in enumerate(rows):
        if row.get("status") != "ok" or row.get("label") not in (0, 1):
            continue
        decision = row.get("official_binary_decision")
        if not isinstance(decision, bool):
            raise ValueError(
                f"valid row {index} has no boolean official argmax decision"
            )
        labels.append(int(row["label"]))
        predictions.append(decision)
    return {
        **_binary_detection_from_arrays(
            np.asarray(labels, dtype=np.int64),
            np.asarray(predictions, dtype=bool),
        ),
        "decision_rule": "argmax_fine_14_class_index_not_equal_to_0",
        "eligible_for_threshold_free_metrics": False,
    }


def _required_float_vector(
    value: Any,
    *,
    length: int,
    label: str,
) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if vector.shape != (length,):
        raise ValueError(f"{label} shape is {vector.shape}, expected {(length,)}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _validate_t1_contract_row(
    row: dict[str, Any],
    *,
    index: int,
) -> None:
    score = finite_float(row.get("score"))
    if score is None or score < 0.0 or score > 1.0:
        raise ValueError(f"valid row {index} has invalid T1 score")
    if row.get("score_source") != "native_out3_fine_14class_head":
        raise ValueError(f"valid row {index} has wrong T1 score source")
    if (
        row.get("score_semantics")
        != "one_minus_softmax_probability_fine_class_0_authentic"
    ):
        raise ValueError(f"valid row {index} has wrong T1 score semantics")
    _require_fixed_threshold(
        row.get("classification_threshold"),
        expected=FIXED_CLASSIFICATION_THRESHOLD,
        label="T1 detection",
    )
    if (
        row.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
    ):
        raise ValueError(
            f"valid row {index} has wrong classification threshold operator"
        )

    hierarchy = row.get("classification_hierarchy_logits")
    if not isinstance(hierarchy, Mapping):
        raise ValueError(f"valid row {index} has no hierarchy logits")
    if set(hierarchy) != set(HIERARCHY_LOGIT_SIZES):
        raise ValueError(f"valid row {index} hierarchy schema mismatch")
    logits_by_name = {
        name: _required_float_vector(
            hierarchy[name],
            length=size,
            label=f"valid row {index} {name} logits",
        )
        for name, size in HIERARCHY_LOGIT_SIZES.items()
    }
    probabilities = _required_float_vector(
        row.get("classification_probabilities"),
        length=len(FINE_CLASS_NAMES),
        label=f"valid row {index} fine probabilities",
    )
    if (
        float(probabilities.min()) < 0.0
        or float(probabilities.max()) > 1.0
        or not math.isclose(
            float(probabilities.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    ):
        raise ValueError(f"valid row {index} has invalid fine probabilities")
    fine_logits = logits_by_name["out3_fine_14class"]
    shifted = fine_logits - float(np.max(fine_logits))
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= float(expected_probabilities.sum())
    if not np.allclose(
        probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"valid row {index} probabilities do not match fine logits"
        )
    expected_score = 1.0 - float(probabilities[0])
    if not math.isclose(
        score,
        expected_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"valid row {index} score does not equal one minus P(authentic)"
        )

    fine_index_value = row.get("official_fine_class_index")
    if (
        isinstance(fine_index_value, bool)
        or not isinstance(fine_index_value, (int, np.integer))
    ):
        raise ValueError(f"valid row {index} has invalid fine-class index")
    fine_index = int(fine_index_value)
    expected_index = int(np.argmax(fine_logits))
    if fine_index != expected_index:
        raise ValueError(
            f"valid row {index} fine-class index does not match argmax"
        )
    if row.get("official_fine_class_name") != FINE_CLASS_NAMES[fine_index]:
        raise ValueError(f"valid row {index} has wrong fine-class name")
    official = row.get("official_binary_decision")
    if not isinstance(official, bool) or official != (fine_index != 0):
        raise ValueError(f"valid row {index} has wrong official decision")
    benchmark = row.get("benchmark_binary_decision")
    expected_benchmark = score > FIXED_CLASSIFICATION_THRESHOLD
    if not isinstance(benchmark, bool) or benchmark != expected_benchmark:
        raise ValueError(f"valid row {index} has wrong benchmark decision")
    expected_decision = "forged" if expected_benchmark else "authentic"
    if row.get("decision") != expected_decision:
        raise ValueError(f"valid row {index} has wrong benchmark decision label")


def _validate_valid_result_row(
    row: dict[str, Any],
    *,
    index: int,
) -> None:
    kind = row.get("kind")
    if kind not in ("real", "forged"):
        raise ValueError(f"valid row {index} has unsupported kind {kind!r}")
    expected_label = int(kind == "forged")
    if row.get("label") != expected_label:
        raise ValueError(f"valid row {index} kind/label mismatch")
    _validate_t1_contract_row(row, index=index)


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
    threshold = _require_fixed_threshold(
        row.get("threshold"),
        expected=FIXED_MASK_THRESHOLD,
        label="T2 localization",
    )
    if threshold != FIXED_MASK_THRESHOLD:
        raise AssertionError("unreachable")
    if row.get("threshold_operator") != MASK_THRESHOLD_OPERATOR:
        raise ValueError(
            f"{kind} {space} row {index} uses threshold operator "
            f"{row.get('threshold_operator')!r}"
        )
    if (
        row.get("score_semantics")
        != "hifi_hypersphere_euclidean_distance"
    ):
        raise ValueError(f"{kind} {space} row {index} has wrong score semantics")
    if row.get("score_dtype") != "float32":
        raise ValueError(f"{kind} {space} row {index} has wrong score dtype")

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

    fraction = finite_float(row.get("predicted_positive_fraction"))
    expected_fraction = counts["predicted_positive_pixels"] / counts["pixels"]
    if fraction is None or not math.isclose(
        fraction,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
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
        value = row.get(name)
        if expected is None:
            if value is not None:
                raise ValueError(
                    f"{kind} {space} row {index} {name} must be null"
                )
            continue
        numeric = finite_float(value)
        lower_bound = -1.0 if name == "mcc" else 0.0
        if (
            numeric is None
            or numeric < lower_bound
            or numeric > 1.0
            or not math.isclose(
                numeric,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{kind} {space} row {index} inconsistent {name}"
            )
    for name in ("score_mean", "score_max"):
        value = finite_float(row.get(name))
        if value is None or value < 0.0:
            raise ValueError(f"{kind} {space} row {index} invalid {name}")

    pixel_ap = row.get("pixel_ap")
    if kind == "real":
        if counts["target_positive_pixels"] != 0 or pixel_ap is not None:
            raise ValueError(f"real {space} row {index} has invalid target/AP")
    else:
        value = finite_float(pixel_ap)
        if (
            counts["target_positive_pixels"] <= 0
            or value is None
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(f"forged {space} row {index} has invalid target/AP")


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
            "threshold_operator": MASK_THRESHOLD_OPERATOR,
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
            "threshold_operator": MASK_THRESHOLD_OPERATOR,
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


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = finite_float(row.get("score"))
        if score is not None:
            by_kind[str(row.get("kind"))].append(score)
    return {
        kind: descriptive(values)
        for kind, values in sorted(by_kind.items())
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
    expected_ids = (
        list(latest)
        if expected_rows is None
        else [str(row["sample_id"]) for row in expected_rows]
    )
    return (
        [latest[row_id] for row_id in expected_ids if row_id in latest],
        len(expected_ids),
    )


def _complete_pairs(
    rows: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        kind = row.get("kind")
        task_id = row.get("task_id")
        if kind not in ("real", "forged"):
            raise ValueError(f"valid result has unsupported kind {kind!r}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("valid result has no task_id")
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate {kind} row within task {task_id}")
        by_task[task_id][str(kind)] = row
    return [
        {"real": values["real"], "forged": values["forged"]}
        for _, values in sorted(by_task.items())
        if set(values) == {"real", "forged"}
    ]


def summarize_hifi_ifdl_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None = None,
    *,
    classification_threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
    bootstrap_samples: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    classification_threshold = _require_fixed_threshold(
        classification_threshold,
        expected=FIXED_CLASSIFICATION_THRESHOLD,
        label="T1 detection",
    )
    mask_threshold = _require_fixed_threshold(
        mask_threshold,
        expected=FIXED_MASK_THRESHOLD,
        label="T2 localization",
    )
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, (int, np.integer))
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    selected, expected_count = _select_rows(rows, expected_rows)
    valid = [row for row in selected if row.get("status") == "ok"]
    for index, row in enumerate(valid):
        _validate_valid_result_row(row, index=index)

    forged: dict[str, Any] = {}
    real: dict[str, Any] = {}
    for space in LOCALIZATION_SPACES:
        forged_rows = [
            _localization_metrics(
                row,
                kind="forged",
                space=space,
                index=index,
            )
            for index, row in enumerate(valid)
            if row.get("kind") == "forged"
        ]
        real_rows = [
            _localization_metrics(
                row,
                kind="real",
                space=space,
                index=index,
            )
            for index, row in enumerate(valid)
            if row.get("kind") == "real"
        ]
        forged[space] = _aggregate_localization(
            forged_rows,
            kind="forged",
            space=space,
        )
        real[space] = _aggregate_localization(
            real_rows,
            kind="real",
            space=space,
        )

    pairs = _complete_pairs(valid)
    pair_bootstrap = {
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        **{
            space: (
                summarize_hifi_ifdl_pair_slice(
                    pairs,
                    iterations=int(bootstrap_samples),
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
    paired_deltas = [
        float(pair["forged"]["score"]) - float(pair["real"]["score"])
        for pair in pairs
    ]
    return {
        "schema_version": "opensource_summary_v1",
        "task_scope": {
            "primary_task": "T1_detection_and_T2_localization",
            "valid_for_t1": True,
            "valid_for_t2": True,
            "primary_detection_score": "score",
            "primary_detection_semantics": (
                "one_minus_softmax_probability_of_hifi_fine_class_0_authentic"
            ),
            "benchmark_classification_threshold": classification_threshold,
            "benchmark_classification_threshold_operator": (
                CLASSIFICATION_THRESHOLD_OPERATOR
            ),
            "official_binary_decision": (
                "argmax_fine_14_class_index_not_equal_to_0"
            ),
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_256",
            "localization_semantics": (
                "hifi_hypersphere_euclidean_distance"
            ),
            "mask_threshold": mask_threshold,
            "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
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
        "score_by_kind": _score_summary(valid),
        "paired_score_delta": descriptive(paired_deltas),
        "paired_ranking_accuracy": (
            sum(delta > 0.0 for delta in paired_deltas) / len(paired_deltas)
            if paired_deltas
            else None
        ),
        "detection": image_detection_metrics_strict(
            valid,
            classification_threshold,
        ),
        "official_argmax_detection": official_argmax_detection_metrics(valid),
        "localization_forged": forged,
        "localization_real": real,
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
    real_rows: list[dict[str, Any]] = []
    forged_rows: list[dict[str, Any]] = []
    real_metrics: list[dict[str, Any]] = []
    forged_metrics: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
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
        real_rows.append(real)
        forged_rows.append(forged)
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

    real_fractions = np.asarray(
        [
            _required_probability(
                row.get("predicted_positive_fraction"),
                label="real false-positive area fraction",
            )
            for row in real_metrics
        ],
        dtype=np.float64,
    )
    return {
        "real_score": np.asarray(
            [
                _required_probability(row.get("score"), label="real T1 score")
                for row in real_rows
            ],
            dtype=np.float64,
        ),
        "forged_score": np.asarray(
            [
                _required_probability(
                    row.get("score"),
                    label="forged T1 score",
                )
                for row in forged_rows
            ],
            dtype=np.float64,
        ),
        "real_official": np.asarray(
            [
                _required_bool(
                    row.get("official_binary_decision"),
                    label="real official decision",
                )
                for row in real_rows
            ],
            dtype=bool,
        ),
        "forged_official": np.asarray(
            [
                _required_bool(
                    row.get("official_binary_decision"),
                    label="forged official decision",
                )
                for row in forged_rows
            ],
            dtype=bool,
        ),
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
        "tp": np.asarray([int(row["tp"]) for row in forged_metrics], dtype=np.int64),
        "fp": np.asarray([int(row["fp"]) for row in forged_metrics], dtype=np.int64),
        "fn": np.asarray([int(row["fn"]) for row in forged_metrics], dtype=np.int64),
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
        "real_predicted_positive_fraction": real_fractions,
    }


def _required_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} is not boolean")
    return value


def _paired_detection_point(
    real: np.ndarray,
    forged: np.ndarray,
    real_official: np.ndarray,
    forged_official: np.ndarray,
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
        threshold=FIXED_CLASSIFICATION_THRESHOLD,
    )
    official = _binary_detection_from_arrays(
        labels,
        np.concatenate([real_official, forged_official]),
    )
    return {
        "auroc": float(detection["auroc"]),
        "average_precision": float(detection["average_precision"]),
        "tpr_at_fpr_5_percent": float(detection["tpr_at_fpr_5_percent"]),
        "accuracy_at_0_5": float(detection["accuracy"]),
        "balanced_accuracy_at_0_5": float(detection["balanced_accuracy"]),
        "image_f1_at_0_5": float(detection["f1"]),
        "official_argmax_accuracy": float(official["accuracy"]),
        "official_argmax_balanced_accuracy": float(
            official["balanced_accuracy"]
        ),
        "official_argmax_image_f1": float(official["f1"]),
        "paired_ranking_accuracy": float(np.mean(forged > real)),
        "paired_score_delta_mean": float(np.mean(forged - real)),
    }


def _pair_slice_point_metrics(
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    point = _paired_detection_point(
        arrays["real_score"],
        arrays["forged_score"],
        arrays["real_official"],
        arrays["forged_official"],
    )
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
    point.update(
        {
            "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
            "pixel_f1_macro_at_2_3": float(np.mean(arrays["pixel_f1"])),
            "pixel_iou_macro_at_2_3": float(np.mean(arrays["pixel_iou"])),
            "pixel_f1_micro_at_2_3": 2.0 * tp / f1_denominator,
            "pixel_iou_micro_at_2_3": tp / iou_denominator,
            "real_false_positive_area_fraction_macro_at_2_3": float(
                np.mean(arrays["real_predicted_positive_fraction"])
            ),
            "real_false_positive_area_fraction_micro_at_2_3": (
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


def summarize_hifi_ifdl_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pair slice is empty")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations <= 0
    ):
        raise ValueError("bootstrap iterations must be a positive integer")
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
    edit_fractions = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    benchmark_predictions = np.concatenate(
        [
            arrays["real_score"] > FIXED_CLASSIFICATION_THRESHOLD,
            arrays["forged_score"] > FIXED_CLASSIFICATION_THRESHOLD,
        ]
    )
    official_predictions = np.concatenate(
        [arrays["real_official"], arrays["forged_official"]]
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
        "bootstrap_samples": int(iterations),
        "seed": int(seed),
        "classification_threshold": FIXED_CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "mask_threshold": FIXED_MASK_THRESHOLD,
        "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_0_5": {
            "tp": int(np.count_nonzero(benchmark_predictions & labels)),
            "fp": int(np.count_nonzero(benchmark_predictions & ~labels)),
            "fn": int(np.count_nonzero(~benchmark_predictions & labels)),
            "tn": int(np.count_nonzero(~benchmark_predictions & ~labels)),
        },
        "official_argmax_confusion": {
            "tp": int(np.count_nonzero(official_predictions & labels)),
            "fp": int(np.count_nonzero(official_predictions & ~labels)),
            "fn": int(np.count_nonzero(~official_predictions & labels)),
            "tn": int(np.count_nonzero(~official_predictions & ~labels)),
        },
        "paired_sign_test": _sign_test(delta),
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
    "CLASSIFICATION_THRESHOLD_OPERATOR",
    "FINE_CLASS_NAMES",
    "FIXED_CLASSIFICATION_THRESHOLD",
    "FIXED_MASK_THRESHOLD",
    "HIERARCHY_LOGIT_SIZES",
    "LOCALIZATION_SPACES",
    "MASK_THRESHOLD_OPERATOR",
    "binary_distance_metrics_strict",
    "binary_pixel_metrics",
    "image_detection_metrics",
    "image_detection_metrics_strict",
    "official_argmax_detection_metrics",
    "summarize_hifi_ifdl_pair_slice",
    "summarize_hifi_ifdl_results",
]
