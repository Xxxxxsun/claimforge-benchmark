"""Strict paired T1/T2 metrics for the official NFA-ViT BR-Gen checkpoint.

NFA-ViT exposes two independent native heads:

* ``pred_label = sigmoid(cls_decoder(...))`` is the image-level manipulation
  probability used for T1.  Higher means manipulated and the official binary
  rule is the strict comparison ``score > 0.5``.
* ``pred_mask = sigmoid(interpolate(seg_decoder(...)))`` is the continuous
  512x512 manipulation map used for T2.  CLAIMFORGE additionally restores that
  probability map to native geometry before applying the same strict rule.

The T2 numerical implementation is shared with the audited Mesorch/DINOv3-IML
512/native protocol.  This module adds strict native-head validation, paired
T1 bootstrap statistics, and the pre-registered ``S_joint`` diagnostic without
ever deriving an image score from the dense map.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from eval.opensource.maskclip_metrics import descriptive, finite_float, safe_div
from eval.opensource.mesorch_metrics import (
    FIXED_MASK_THRESHOLD as _FIXED_MASK_THRESHOLD,
    LOCALIZATION_SPACES as _LOCALIZATION_SPACES,
    THRESHOLD_OPERATOR as _MASK_THRESHOLD_OPERATOR,
    binary_pixel_metrics_strict as _binary_pixel_metrics_strict,
    summarize_mesorch_results as _summarize_localization_results,
)


FIXED_CLASSIFICATION_THRESHOLD = 0.5
FIXED_MASK_THRESHOLD = _FIXED_MASK_THRESHOLD
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD_OPERATOR = _MASK_THRESHOLD_OPERATOR
LOCALIZATION_SPACES = _LOCALIZATION_SPACES

BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "accuracy_at_0_5",
    "balanced_accuracy_at_0_5",
    "image_precision_at_0_5",
    "image_recall_at_0_5",
    "image_f1_at_0_5",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_false_positive_area_fraction_macro_at_0_5",
    "real_false_positive_area_fraction_micro_at_0_5",
    "s_joint_macro_at_0_5",
    "s_joint_micro_at_0_5",
)
OPTIONAL_BOOTSTRAP_METRICS = (
    "pixel_mcc_macro_at_0_5",
    "pixel_mcc_micro_at_0_5",
)

if LOCALIZATION_SPACES != ("model_512", "native"):
    raise RuntimeError("shared localization metric spaces changed unexpectedly")
if FIXED_MASK_THRESHOLD != 0.5 or MASK_THRESHOLD_OPERATOR != ">":
    raise RuntimeError("shared strict localization threshold contract changed")


def _rebrand_error(error: ValueError) -> ValueError:
    return ValueError(str(error).replace("Mesorch", "NFA-ViT"))


def _require_fixed_threshold(
    threshold: Any,
    *,
    label: str,
) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NFA-ViT {label} threshold is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value,
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"NFA-ViT {label} uses fixed threshold 0.5, not {value}"
        )
    return 0.5


def _require_probability(value: Any, *, label: str) -> float:
    result = finite_float(value)
    if result is None or result < 0.0 or result > 1.0:
        raise ValueError(f"{label} is not a probability")
    return result


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is not boolean")
    return bool(value)


def _released_float32_sigmoid(value: float) -> float:
    """Replay the released ``torch.sigmoid`` classifier output on float32."""

    try:
        logit = torch.tensor(value, dtype=torch.float32, device="cpu")
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "classification logit cannot be represented as torch.float32"
        ) from exc
    if not bool(torch.isfinite(logit).item()):
        raise ValueError("classification logit is not finite in torch.float32")
    return float(torch.sigmoid(logit).item())


def binary_pixel_metrics_strict(
    probability_map: np.ndarray,
    target: np.ndarray,
    threshold: float = FIXED_MASK_THRESHOLD,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    """Evaluate the persisted float32 map using strict ``p > 0.5``."""

    try:
        return _binary_pixel_metrics_strict(
            probability_map,
            target,
            threshold,
            include_ap=include_ap,
        )
    except ValueError as exc:
        raise _rebrand_error(exc) from exc


binary_pixel_metrics = binary_pixel_metrics_strict


def _classification_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("classification")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("NFA-ViT classification field is not an object")
    return value


def _first_present(
    row: Mapping[str, Any],
    classification: Mapping[str, Any],
    *,
    classification_keys: tuple[str, ...],
    row_keys: tuple[str, ...],
) -> Any:
    for key in classification_keys:
        if key in classification:
            return classification[key]
    for key in row_keys:
        if key in row:
            return row[key]
    return None


def _classification_contract(
    row: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Normalize the frozen row schema and verify the native classifier head.

    The canonical form is ``row.score`` plus a ``classification`` object.  The
    explicitly listed top-level aliases are accepted so that an early runner
    using the same semantics remains auditable; no dense-map-derived fallback
    is accepted.
    """

    score = _require_probability(
        row.get("score"),
        label=f"valid row {index} native T1 score",
    )
    classification = _classification_mapping(row)
    raw_logit = finite_float(
        _first_present(
            row,
            classification,
            classification_keys=("raw_logit", "logit"),
            row_keys=("classification_logit", "classification_raw_logit"),
        )
    )
    if raw_logit is None:
        raise ValueError(f"valid row {index} has no finite classification logit")

    recorded_probability = _first_present(
        row,
        classification,
        classification_keys=("probability", "score"),
        row_keys=("classification_probability", "classification_score"),
    )
    if recorded_probability is not None:
        probability = _require_probability(
            recorded_probability,
            label=f"valid row {index} classification probability",
        )
        if not math.isclose(
            probability,
            score,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"valid row {index} classification probability/score mismatch"
            )

    expected_score = _released_float32_sigmoid(raw_logit)
    if not math.isclose(
        score,
        expected_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"valid row {index} score does not equal sigmoid classifier logit"
        )
    expected_decision = score > FIXED_CLASSIFICATION_THRESHOLD
    replayed_decision = expected_score > FIXED_CLASSIFICATION_THRESHOLD
    if expected_decision != replayed_decision:
        raise ValueError(
            f"valid row {index} score and float32 sigmoid classifier logit "
            "cross the 0.5 threshold differently"
        )

    threshold = _first_present(
        row,
        classification,
        classification_keys=("threshold",),
        row_keys=("classification_threshold",),
    )
    _require_fixed_threshold(
        threshold,
        label="T1 detection",
    )
    operator = _first_present(
        row,
        classification,
        classification_keys=("threshold_operator", "operator"),
        row_keys=("classification_threshold_operator",),
    )
    if operator != CLASSIFICATION_THRESHOLD_OPERATOR:
        raise ValueError(
            f"valid row {index} classification operator is not strict '>'"
        )

    decision_values: list[tuple[str, Any]] = []
    if "decision" in classification:
        decision_values.append(
            ("classification decision", classification["decision"])
        )
    for key in (
        "classification_decision_strict_gt_0_5",
        "classification_decision",
        "benchmark_binary_decision",
    ):
        if key in row:
            decision_values.append((key, row[key]))
    if not decision_values:
        raise ValueError(f"valid row {index} has no classification decision")
    for decision_label, decision_value in decision_values:
        if isinstance(decision_value, str):
            if decision_value not in ("authentic", "forged"):
                raise ValueError(f"valid row {index} has invalid decision label")
            decision = decision_value == "forged"
        else:
            decision = _require_bool(
                decision_value,
                label=f"valid row {index} {decision_label}",
            )
        if decision != expected_decision:
            raise ValueError(
                f"valid row {index} classification decision/score mismatch"
            )

    top_level_label = row.get("decision")
    if top_level_label is not None:
        expected_label = "forged" if expected_decision else "authentic"
        if top_level_label != expected_label:
            raise ValueError(f"valid row {index} has wrong decision label")

    if row.get("valid_for_t1") is not None and row.get("valid_for_t1") is not True:
        raise ValueError(f"valid row {index} incorrectly disables native T1")
    if row.get("valid_for_t2") is not None and row.get("valid_for_t2") is not True:
        raise ValueError(f"valid row {index} incorrectly disables native T2")

    return {
        "raw_logit": raw_logit,
        "score": score,
        "decision": expected_decision,
    }


def _binary_detection(
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
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
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
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
        "precision": safe_div(tp, tp + fp),
        "recall": recall,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn) or 0.0,
        "sensitivity": recall,
        "specificity": specificity,
    }


def _tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.05,
) -> tuple[float, float | None]:
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    eligible = np.flatnonzero(fpr <= target_fpr)
    best_tpr = float(np.max(tpr[eligible]))
    tied = eligible[tpr[eligible] == best_tpr]
    finite = [
        int(position)
        for position in tied
        if finite_float(thresholds[int(position)]) is not None
    ]
    position = finite[-1] if finite else int(tied[-1])
    return best_tpr, finite_float(thresholds[position])


def _detection_from_arrays(
    labels: np.ndarray,
    scores: np.ndarray,
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
        float(scores_array.min()) < 0.0
        or float(scores_array.max()) > 1.0
    ):
        raise ValueError("detection scores fall outside [0, 1]")

    result = {
        **_binary_detection(
            labels_array,
            scores_array > FIXED_CLASSIFICATION_THRESHOLD,
        ),
        "threshold": FIXED_CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "score_semantics": "native_sigmoid_cls_decoder_manipulation_probability",
    }
    positive = labels_array == 1
    negative = ~positive
    if positive.any() and negative.any():
        tpr, threshold = _tpr_at_fpr(labels_array, scores_array)
        result.update(
            {
                "auroc": float(roc_auc_score(labels_array, scores_array)),
                "average_precision": float(
                    average_precision_score(labels_array, scores_array)
                ),
                "tpr_at_fpr_5_percent": tpr,
                "tpr_at_fpr_5_percent_threshold": threshold,
            }
        )
    return result


def image_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate the official native T1 head; dense maps are never inspected."""

    _require_fixed_threshold(threshold, label="T1 detection")
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        kind = row.get("kind")
        if kind not in ("real", "forged"):
            raise ValueError(f"valid row {index} has unsupported kind {kind!r}")
        expected_label = int(kind == "forged")
        if row.get("label") != expected_label:
            raise ValueError(f"valid row {index} kind/label mismatch")
        classification = _classification_contract(row, index=index)
        labels.append(expected_label)
        scores.append(float(classification["score"]))
    return _detection_from_arrays(
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


image_detection_metrics = image_detection_metrics_strict


def _row_id(row: Mapping[str, Any], *, index: int) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"result row {index} has no non-empty id")
    return value


def _expected_id(row: Mapping[str, Any], *, index: int) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected row {index} has no non-empty sample_id")
    return value


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

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, expected in enumerate(expected_rows):
        if not isinstance(expected, dict):
            raise ValueError(f"expected row {index} is not an object")
        sample_id = _expected_id(expected, index=index)
        if sample_id in seen:
            raise ValueError(f"duplicate expected sample_id {sample_id}")
        seen.add(sample_id)
        result = latest.get(sample_id)
        if result is None:
            continue
        for key in ("task_id", "kind", "label", "domain"):
            if key in expected and result.get(key) != expected.get(key):
                raise ValueError(
                    f"result {sample_id} {key} does not match expected row"
                )
        selected.append(result)
    return selected, len(seen)


def _complete_pairs(
    rows: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        task_id = row.get("task_id")
        kind = row.get("kind")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("valid result has no task_id")
        if kind not in ("real", "forged"):
            raise ValueError(f"valid result has unsupported kind {kind!r}")
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate {kind} row within task {task_id}")
        by_task[task_id][kind] = row
    return [
        {"real": values["real"], "forged": values["forged"]}
        for _, values in sorted(by_task.items())
        if set(values) == {"real", "forged"}
    ]


def _localization_metrics(
    row: Mapping[str, Any],
    *,
    space: str,
    label: str,
) -> Mapping[str, Any]:
    localization = row.get("localization")
    if not isinstance(localization, Mapping):
        raise ValueError(f"{label} has no localization object")
    metrics = localization.get(space)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} has no {space} localization metrics")
    return metrics


def _pair_arrays(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    localization_space: str,
) -> dict[str, np.ndarray]:
    real_scores: list[float] = []
    forged_scores: list[float] = []
    pixel_ap: list[float] = []
    pixel_f1: list[float] = []
    pixel_iou: list[float] = []
    pixel_mcc: list[float] = []
    tp: list[int] = []
    fp: list[int] = []
    fn: list[int] = []
    tn: list[int] = []
    forged_pixels: list[int] = []
    target_positive: list[int] = []
    real_positive: list[int] = []
    real_pixels: list[int] = []
    real_fraction: list[float] = []
    for index, pair in enumerate(pairs):
        real = pair["real"]
        forged = pair["forged"]
        if real.get("status") != "ok" or forged.get("status") != "ok":
            raise ValueError(f"pair {index} is not fully valid")
        if real.get("task_id") != forged.get("task_id"):
            raise ValueError(f"pair {index} has mismatched task IDs")
        real_contract = _classification_contract(real, index=2 * index)
        forged_contract = _classification_contract(forged, index=2 * index + 1)
        real_scores.append(float(real_contract["score"]))
        forged_scores.append(float(forged_contract["score"]))

        real_metric = _localization_metrics(
            real,
            space=localization_space,
            label=f"real pair {index}",
        )
        forged_metric = _localization_metrics(
            forged,
            space=localization_space,
            label=f"forged pair {index}",
        )
        pixel_ap.append(
            _require_probability(
                forged_metric.get("pixel_ap"),
                label=f"forged pair {index} pixel AP",
            )
        )
        pixel_f1.append(
            _require_probability(
                forged_metric.get("f1"),
                label=f"forged pair {index} pixel F1",
            )
        )
        pixel_iou.append(
            _require_probability(
                forged_metric.get("iou"),
                label=f"forged pair {index} pixel IoU",
            )
        )
        mcc = finite_float(forged_metric.get("mcc"))
        pixel_mcc.append(float("nan") if mcc is None else mcc)
        for source, destination, key in (
            (forged_metric, tp, "tp"),
            (forged_metric, fp, "fp"),
            (forged_metric, fn, "fn"),
            (forged_metric, tn, "tn"),
            (forged_metric, forged_pixels, "pixels"),
            (forged_metric, target_positive, "target_positive_pixels"),
            (real_metric, real_positive, "predicted_positive_pixels"),
            (real_metric, real_pixels, "pixels"),
        ):
            value = source.get(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise ValueError(f"pair {index} {key} is not an integer")
            destination.append(int(value))
        real_fraction.append(
            _require_probability(
                real_metric.get("predicted_positive_fraction"),
                label=f"real pair {index} false-positive area fraction",
            )
        )
    return {
        "real_score": np.asarray(real_scores, dtype=np.float64),
        "forged_score": np.asarray(forged_scores, dtype=np.float64),
        "pixel_ap": np.asarray(pixel_ap, dtype=np.float64),
        "pixel_f1": np.asarray(pixel_f1, dtype=np.float64),
        "pixel_iou": np.asarray(pixel_iou, dtype=np.float64),
        "pixel_mcc": np.asarray(pixel_mcc, dtype=np.float64),
        "tp": np.asarray(tp, dtype=np.int64),
        "fp": np.asarray(fp, dtype=np.int64),
        "fn": np.asarray(fn, dtype=np.int64),
        "tn": np.asarray(tn, dtype=np.int64),
        "forged_pixels": np.asarray(forged_pixels, dtype=np.int64),
        "forged_target_positive_pixels": np.asarray(
            target_positive,
            dtype=np.int64,
        ),
        "real_predicted_positive_pixels": np.asarray(
            real_positive,
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(real_pixels, dtype=np.int64),
        "real_predicted_positive_fraction": np.asarray(
            real_fraction,
            dtype=np.float64,
        ),
    }


def _mcc_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def _pair_point(arrays: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    real = arrays["real_score"]
    forged = arrays["forged_score"]
    labels = np.concatenate(
        [
            np.zeros(real.size, dtype=np.int64),
            np.ones(forged.size, dtype=np.int64),
        ]
    )
    detection = _detection_from_arrays(labels, np.concatenate([real, forged]))
    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    tn = int(np.sum(arrays["tn"]))
    real_positive = int(np.sum(arrays["real_predicted_positive_pixels"]))
    real_pixels = int(np.sum(arrays["real_pixels"]))
    macro_mcc_values = arrays["pixel_mcc"][
        np.isfinite(arrays["pixel_mcc"])
    ]
    pixel_f1_macro = float(np.mean(arrays["pixel_f1"]))
    pixel_f1_micro = safe_div(2 * tp, 2 * tp + fp + fn)
    if pixel_f1_micro is None:
        raise ValueError("forged localization has no evaluable pixels")
    image_f1 = float(detection["f1"])
    return {
        "auroc": float(detection["auroc"]),
        "average_precision": float(detection["average_precision"]),
        "tpr_at_fpr_5_percent": float(detection["tpr_at_fpr_5_percent"]),
        "accuracy_at_0_5": float(detection["accuracy"]),
        "balanced_accuracy_at_0_5": float(detection["balanced_accuracy"]),
        "image_precision_at_0_5": (
            None
            if detection["precision"] is None
            else float(detection["precision"])
        ),
        "image_recall_at_0_5": float(detection["recall"]),
        "image_f1_at_0_5": image_f1,
        "paired_ranking_accuracy": float(np.mean(forged > real)),
        "paired_score_delta_mean": float(np.mean(forged - real)),
        "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
        "pixel_f1_macro_at_0_5": pixel_f1_macro,
        "pixel_iou_macro_at_0_5": float(np.mean(arrays["pixel_iou"])),
        "pixel_mcc_macro_at_0_5": (
            float(np.mean(macro_mcc_values))
            if macro_mcc_values.size
            else None
        ),
        "pixel_f1_micro_at_0_5": pixel_f1_micro,
        "pixel_iou_micro_at_0_5": safe_div(tp, tp + fp + fn),
        "pixel_mcc_micro_at_0_5": _mcc_from_counts(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        ),
        "real_false_positive_area_fraction_macro_at_0_5": float(
            np.mean(arrays["real_predicted_positive_fraction"])
        ),
        "real_false_positive_area_fraction_micro_at_0_5": safe_div(
            real_positive,
            real_pixels,
        ),
        "s_joint_macro_at_0_5": image_f1 * pixel_f1_macro,
        "s_joint_micro_at_0_5": image_f1 * pixel_f1_micro,
    }


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot calculate confidence interval from no values")
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _sign_test(delta: np.ndarray) -> dict[str, Any]:
    wins = int(np.count_nonzero(delta > 0.0))
    losses = int(np.count_nonzero(delta < 0.0))
    ties = int(np.count_nonzero(delta == 0.0))
    non_ties = wins + losses
    if non_ties:
        tail = min(wins, losses)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(non_ties, k) for k in range(tail + 1))
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


def summarize_nfa_vit_pair_slice(
    pairs: list[dict[str, dict[str, Any]]],
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
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError("bootstrap seed must be an integer")
    if localization_space not in LOCALIZATION_SPACES:
        raise ValueError(
            f"unsupported localization space {localization_space!r}"
        )

    arrays = _pair_arrays(pairs, localization_space=localization_space)
    point = _pair_point(arrays)
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS + OPTIONAL_BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(int(seed))
    for _ in range(int(iterations)):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sampled = {name: values[indices] for name, values in arrays.items()}
        values = _pair_point(sampled)
        for name, replicate_values in replicates.items():
            value = values[name]
            if value is not None and math.isfinite(float(value)):
                replicate_values.append(float(value))

    delta = arrays["forged_score"] - arrays["real_score"]
    predictions = np.concatenate(
        [
            arrays["real_score"] > FIXED_CLASSIFICATION_THRESHOLD,
            arrays["forged_score"] > FIXED_CLASSIFICATION_THRESHOLD,
        ]
    )
    labels = np.concatenate(
        [
            np.zeros(len(pairs), dtype=bool),
            np.ones(len(pairs), dtype=bool),
        ]
    )
    edit_fraction = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    result: dict[str, Any] = {
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
            if point[name] is not None
        },
        "image_confusion_at_0_5": {
            "tp": int(np.count_nonzero(predictions & labels)),
            "fp": int(np.count_nonzero(predictions & ~labels)),
            "fn": int(np.count_nonzero(~predictions & labels)),
            "tn": int(np.count_nonzero(~predictions & ~labels)),
        },
        "paired_sign_test": _sign_test(delta),
        "edit_fraction": {
            "min": float(np.min(edit_fraction)),
            "median": float(np.median(edit_fraction)),
            "mean": float(np.mean(edit_fraction)),
            "max": float(np.max(edit_fraction)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }
    for name in OPTIONAL_BOOTSTRAP_METRICS:
        estimate = point[name]
        result[name] = (
            None
            if estimate is None
            else {
                "estimate": float(estimate),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
        )
    return result


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        contract = _classification_contract(row, index=index)
        by_kind[str(row["kind"])].append(float(contract["score"]))
    return {
        kind: descriptive(values)
        for kind, values in sorted(by_kind.items())
    }


def _native_t1_saturation_diagnostic(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        contract = _classification_contract(row, index=index)
        by_kind[str(row["kind"])].append(float(contract["score"]))

    kind_summary: dict[str, Any] = {}
    all_scores: list[float] = []
    for kind, values in sorted(by_kind.items()):
        array = np.asarray(values, dtype=np.float64)
        all_scores.extend(values)
        kind_summary[kind] = {
            "images": int(array.size),
            "minimum": float(np.min(array)) if array.size else None,
            "maximum": float(np.max(array)) if array.size else None,
            "mean": float(np.mean(array)) if array.size else None,
            "standard_deviation": (
                float(np.std(array)) if array.size else None
            ),
            "exact_zero": int(np.count_nonzero(array == 0.0)),
            "exact_one": int(np.count_nonzero(array == 1.0)),
            "at_most_0_001": int(np.count_nonzero(array <= 0.001)),
            "at_least_0_999": int(np.count_nonzero(array >= 0.999)),
            "unique_float64_values": int(np.unique(array).size),
        }
    deltas = np.asarray(
        [
            float(pair["forged"]["score"]) - float(pair["real"]["score"])
            for pair in pairs
        ],
        dtype=np.float64,
    )
    all_array = np.asarray(all_scores, dtype=np.float64)
    all_near_one = bool(all_array.size and np.all(all_array >= 0.999))
    return {
        "status": "native_head_release_risk_diagnostic",
        "native_t1_metrics_reported": True,
        "native_t1_metrics_protocol_role": "primary_frozen_protocol_output",
        "score_source": "native_cls_decoder_sigmoid",
        "dense_map_fallback_forbidden": True,
        "public_release_evaluator_uses_native_t1_for_checkpoint_selection": (
            False
        ),
        "exact_checkpoint_selection_used_native_t1": None,
        "exact_checkpoint_training_and_selection_protocol": "unpublished",
        "near_zero_threshold": 0.001,
        "near_one_threshold": 0.999,
        "by_kind": kind_summary,
        "all_images_at_least_0_999": all_near_one,
        "saturation_flag": all_near_one,
        "paired_equal_within_1e_7": int(
            np.count_nonzero(np.abs(deltas) <= 1e-7)
        ),
        "paired_tasks": int(deltas.size),
    }


def _raw_logit_secondary_diagnostic(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Describe ordering left in logits when float32 sigmoid scores saturate.

    This is deliberately not a replacement T1 score.  The frozen primary
    interface remains the released probability and strict ``p > 0.5`` rule.
    """

    labels: list[int] = []
    logits: list[float] = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        contract = _classification_contract(row, index=index)
        kind = str(row["kind"])
        logit = float(contract["raw_logit"])
        labels.append(int(kind == "forged"))
        logits.append(logit)
        by_kind[kind].append(logit)

    labels_array = np.asarray(labels, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    result: dict[str, Any] = {
        "status": "secondary_numerical_saturation_diagnostic_only",
        "eligible_for_primary_t1_metrics": False,
        "official_released_t1_output": "sigmoid_probability",
        "raw_logit_replaces_probability_for_primary_metrics": False,
        "by_kind": {
            kind: {
                **descriptive(values),
                "standard_deviation": float(
                    np.std(np.asarray(values, dtype=np.float64))
                ),
                "unique_float64_values": len(set(values)),
            }
            for kind, values in sorted(by_kind.items())
        },
    }
    if np.any(labels_array == 0) and np.any(labels_array == 1):
        tpr, threshold = _tpr_at_fpr(labels_array, logits_array)
        result["ranking_metrics"] = {
            "auroc": float(roc_auc_score(labels_array, logits_array)),
            "average_precision": float(
                average_precision_score(labels_array, logits_array)
            ),
            "tpr_at_fpr_5_percent": tpr,
            "tpr_at_fpr_5_percent_logit_threshold": threshold,
        }
    else:
        result["ranking_metrics"] = None

    paired_deltas = np.asarray(
        [
            float(
                _classification_contract(
                    pair["forged"],
                    index=2 * index + 1,
                )["raw_logit"]
            )
            - float(
                _classification_contract(
                    pair["real"],
                    index=2 * index,
                )["raw_logit"]
            )
            for index, pair in enumerate(pairs)
        ],
        dtype=np.float64,
    )
    result["paired_logit_delta"] = descriptive(paired_deltas)
    result["paired_logit_ranking_accuracy"] = (
        float(np.mean(paired_deltas > 0.0)) if paired_deltas.size else None
    )
    result["paired_sign_test"] = _sign_test(paired_deltas)
    return result


def summarize_nfa_vit_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None = None,
    *,
    classification_threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
    bootstrap_samples: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Build the paired NFA-ViT T1+T2 summary and joint diagnostics."""

    classification_threshold = _require_fixed_threshold(
        classification_threshold,
        label="T1 detection",
    )
    mask_threshold = _require_fixed_threshold(
        mask_threshold,
        label="T2 localization",
    )
    try:
        summary = _summarize_localization_results(
            rows,
            expected_rows,
            mask_threshold=mask_threshold,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except ValueError as exc:
        raise _rebrand_error(exc) from exc

    selected, expected_count = _select_rows(rows, expected_rows)
    valid = [row for row in selected if row.get("status") == "ok"]
    for index, row in enumerate(valid):
        _classification_contract(row, index=index)
    if summary["coverage"]["expected_images"] != expected_count:
        raise RuntimeError("shared localization summary coverage drifted")

    pairs = _complete_pairs(valid)
    paired_ids = {
        id(row)
        for pair in pairs
        for row in (pair["real"], pair["forged"])
    }
    pair_bootstrap = {
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        **{
            space: (
                summarize_nfa_vit_pair_slice(
                    pairs,
                    iterations=int(bootstrap_samples),
                    seed=int(seed),
                    localization_space=space,
                )
                if pairs
                else None
            )
            for space in LOCALIZATION_SPACES
        },
    }
    deltas = [
        float(pair["forged"]["score"]) - float(pair["real"]["score"])
        for pair in pairs
    ]

    summary["task_scope"] = {
        "primary_task": "T1_detection_and_T2_localization",
        "valid_for_t1": True,
        "valid_for_t2": True,
        "primary_detection_score": "score",
        "primary_detection_semantics": (
            "native_sigmoid_cls_decoder_manipulation_probability"
        ),
        "classification_threshold": classification_threshold,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "primary_localization_space": "native",
        "auxiliary_localization_space": "model_512",
        "localization_semantics": (
            "nfa_vit_sigmoid_segmentation_probability_float32"
        ),
        "probability_dtype": "float32",
        "mask_threshold": mask_threshold,
        "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
        "model_space_probability_source": (
            "sigmoid_bilinear_align_corners_false_decoder_logits_128_to_512"
        ),
        "native_probability_source": (
            "bilinear_align_corners_false_resize_of_model_512_probability"
        ),
        "image_score_independent_of_dense_map": True,
    }
    summary["paired_coverage"] = {
        "complete_pairs": len(pairs),
        "paired_images": len(paired_ids),
        "unpaired_valid_images": len(valid) - len(paired_ids),
    }
    summary["score_by_kind"] = _score_summary(valid)
    summary["paired_score_delta"] = descriptive(deltas)
    summary["paired_ranking_accuracy"] = (
        sum(delta > 0.0 for delta in deltas) / len(deltas)
        if deltas
        else None
    )
    summary["detection"] = image_detection_metrics_strict(
        valid,
        classification_threshold,
    )
    summary["native_t1_saturation_diagnostic"] = (
        _native_t1_saturation_diagnostic(valid, pairs)
    )
    summary["raw_logit_secondary_diagnostic"] = (
        _raw_logit_secondary_diagnostic(valid, pairs)
    )
    summary["pair_bootstrap"] = pair_bootstrap
    summary["joint_diagnostics"] = {
        "status": "claimforge_diagnostic_not_official_nfa_vit_metric",
        "definition": (
            "CLAIMFORGE S_joint = image_F1_at_0.5 * pixel_F1_at_0.5"
        ),
        "classification_source": "native_cls_decoder_only",
        "dense_map_used_for_t1": False,
        "spaces": {
            space: (
                None
                if pair_bootstrap[space] is None
                else {
                    "macro": pair_bootstrap[space][
                        "s_joint_macro_at_0_5"
                    ],
                    "micro": pair_bootstrap[space][
                        "s_joint_micro_at_0_5"
                    ],
                }
            )
            for space in LOCALIZATION_SPACES
        },
    }
    return summary


__all__ = [
    "BOOTSTRAP_METRICS",
    "CLASSIFICATION_THRESHOLD_OPERATOR",
    "FIXED_CLASSIFICATION_THRESHOLD",
    "FIXED_MASK_THRESHOLD",
    "LOCALIZATION_SPACES",
    "MASK_THRESHOLD_OPERATOR",
    "OPTIONAL_BOOTSTRAP_METRICS",
    "binary_pixel_metrics",
    "binary_pixel_metrics_strict",
    "image_detection_metrics",
    "image_detection_metrics_strict",
    "summarize_nfa_vit_pair_slice",
    "summarize_nfa_vit_results",
]
