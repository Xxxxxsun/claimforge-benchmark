"""Shared strict metrics for paired whole-image AIGC detectors.

The primary use is the released FSD score, but the implementation deliberately
does not assume that a detector score is a probability.  ``ai_score`` is a
finite binary64 value and larger values must mean "more likely fake".  Paired
summaries additionally require every forged-minus-real difference to remain
finite, so no overflowed statistic is silently omitted.

The released FSD operating point is the strict rule ``ai_score > 2``.  The
additional 5% FPR operating point is selected without looking at forged
scores: it is the 95th percentile of the real scores using NumPy's ``higher``
quantile rule, followed by the same strict comparison.  Pair bootstrap
replicates resample ``task_id`` pairs and re-select that real-only threshold in
every replicate.
"""

from __future__ import annotations

import math
import numbers
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.opensource.maskclip_metrics import descriptive


FIXED_THRESHOLD = 2.0
THRESHOLD_OPERATOR = ">"
TARGET_FPR = 0.05
FPR_QUANTILE_METHOD = "higher"
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260724
EDIT_VISIBILITIES = frozenset({"none", "partial", "full"})

BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "tpr_at_fpr_5_percent_threshold",
    "tpr_at_fpr_5_percent_actual_fpr",
    "accuracy_at_2",
    "balanced_accuracy_at_2",
    "precision_at_2",
    "recall_at_2",
    "f1_at_2",
    "specificity_at_2",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
)


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.integer, np.floating)
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _binary_label(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Integral, np.integer)
    ):
        raise ValueError(f"{label} is not an integer 0/1 label")
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{label} is not a 0/1 label")
    return result


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Integral, np.integer)
    ):
        raise ValueError(f"{label} is not a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return result


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _validated_edit_visibility(
    value: Any,
    fraction: Any,
    *,
    label: str,
) -> tuple[str, float]:
    visibility = _nonempty_string(value, label=f"{label} edit_visibility")
    if visibility not in EDIT_VISIBILITIES:
        raise ValueError(
            f"{label} edit_visibility has invalid category {visibility!r}"
        )
    visible_fraction = _finite_real(
        fraction,
        label=f"{label} edit_visible_gt_fraction",
    )
    if not 0.0 <= visible_fraction <= 1.0:
        raise ValueError(
            f"{label} edit_visible_gt_fraction falls outside [0, 1]"
        )
    expected = (
        "none"
        if visible_fraction == 0.0
        else "full"
        if visible_fraction == 1.0
        else "partial"
    )
    if visibility != expected:
        raise ValueError(
            f"{label} edit_visibility/category fraction mismatch: "
            f"{visibility!r} requires {expected!r} for fraction "
            f"{visible_fraction}"
        )
    return visibility, visible_fraction


def _finite_pair_deltas(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
) -> np.ndarray:
    real = np.asarray(real_scores, dtype=np.float64)
    forged = np.asarray(forged_scores, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = forged - real
    if not np.isfinite(delta).all():
        raise ValueError(
            "paired score delta is not finite; finite binary64 scores must "
            "also have finite forged-minus-real differences"
        )
    return delta


def _require_fixed_threshold(value: Any) -> float:
    threshold = _finite_real(value, label="classification threshold")
    if not math.isclose(
        threshold,
        FIXED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"whole-image detection uses fixed threshold {FIXED_THRESHOLD}, "
            f"not {threshold}"
        )
    return FIXED_THRESHOLD


def _safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _confusion_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=bool)
    if labels_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("classification arrays must be one-dimensional")
    if labels_array.shape != predictions_array.shape:
        raise ValueError("classification arrays have different lengths")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("classification labels must be binary")

    positive = labels_array == 1
    negative = ~positive
    tp = int(np.count_nonzero(predictions_array & positive))
    fp = int(np.count_nonzero(predictions_array & negative))
    fn = int(np.count_nonzero(~predictions_array & positive))
    tn = int(np.count_nonzero(~predictions_array & negative))
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
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
        "precision": _safe_div(tp, tp + fp) or 0.0,
        "precision_zero_division": 0,
        "recall": recall,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn) or 0.0,
        "specificity": specificity,
    }


def _real_only_fpr_operating_point(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
) -> dict[str, float]:
    real = np.asarray(real_scores, dtype=np.float64)
    forged = np.asarray(forged_scores, dtype=np.float64)
    if real.ndim != 1 or forged.ndim != 1:
        raise ValueError("real and forged score arrays must be one-dimensional")
    if not real.size or not forged.size:
        raise ValueError("5% FPR operating point needs real and forged scores")
    if not np.isfinite(real).all() or not np.isfinite(forged).all():
        raise ValueError("5% FPR operating point received non-finite scores")

    threshold = float(
        np.quantile(
            real,
            1.0 - TARGET_FPR,
            method=FPR_QUANTILE_METHOD,
        )
    )
    return {
        "tpr": float(np.mean(forged > threshold)),
        "threshold": threshold,
        "actual_fpr": float(np.mean(real > threshold)),
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
        raise ValueError("detection labels and scores have different lengths")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("detection labels must be binary")
    if not np.isfinite(scores_array).all():
        raise ValueError("detection scores contain non-finite values")

    result: dict[str, Any] = {
        **_confusion_metrics(labels_array, scores_array > threshold),
        "threshold": float(threshold),
        "threshold_operator": THRESHOLD_OPERATOR,
        "score_direction": "higher_means_fake",
        "auroc": None,
        "average_precision": None,
        "tpr_at_fpr_5_percent": None,
        "tpr_at_fpr_5_percent_threshold": None,
        "tpr_at_fpr_5_percent_actual_fpr": None,
        "tpr_at_fpr_5_percent_threshold_source": (
            "real_scores_quantile_0_95_method_higher"
        ),
        "tpr_at_fpr_5_percent_threshold_operator": THRESHOLD_OPERATOR,
    }
    positive = labels_array == 1
    negative = ~positive
    if positive.any() and negative.any():
        operating_point = _real_only_fpr_operating_point(
            scores_array[negative],
            scores_array[positive],
        )
        result.update(
            {
                "auroc": float(roc_auc_score(labels_array, scores_array)),
                "average_precision": float(
                    average_precision_score(labels_array, scores_array)
                ),
                "tpr_at_fpr_5_percent": operating_point["tpr"],
                "tpr_at_fpr_5_percent_threshold": operating_point["threshold"],
                "tpr_at_fpr_5_percent_actual_fpr": operating_point[
                    "actual_fpr"
                ],
            }
        )
    return result


def image_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    *,
    score_key: str = "ai_score",
) -> dict[str, Any]:
    """Calculate strict whole-image metrics from successful rows.

    A successful row is never silently dropped: its label and selected score
    must satisfy the frozen schema.
    """

    threshold_value = _require_fixed_threshold(threshold)
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"result row {index} is not an object")
        status = row.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"result row {index} has invalid status")
        if status == "error":
            continue
        labels.append(
            _binary_label(
                row.get("label"),
                label=f"successful row {index} label",
            )
        )
        scores.append(
            _finite_real(
                row.get(score_key),
                label=f"successful row {index} {score_key}",
            )
        )

    result = _detection_from_arrays(
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        threshold=threshold_value,
    )
    result["score_key"] = score_key
    return result


image_detection_metrics = image_detection_metrics_strict


def _pair_row(pair: Any, kind: str) -> Mapping[str, Any]:
    row = (
        pair.get(kind)
        if isinstance(pair, Mapping)
        else getattr(pair, kind, None)
    )
    if not isinstance(row, Mapping):
        raise ValueError(f"pair has no {kind!r} result row")
    return row


def _validate_pair_rows(
    pairs: list[Any],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
    list[str],
    np.ndarray,
]:
    real_rows: list[Mapping[str, Any]] = []
    forged_rows: list[Mapping[str, Any]] = []
    domains: list[str] = []
    visibilities: list[str] = []
    visible_fractions: list[float] = []
    seen_task_ids: set[str] = set()

    for index, pair in enumerate(pairs):
        real = _pair_row(pair, "real")
        forged = _pair_row(pair, "forged")
        for kind, expected_label, row in (
            ("real", 0, real),
            ("forged", 1, forged),
        ):
            if row.get("status") != "ok":
                raise ValueError(f"{kind} row in pair {index} is not status ok")
            if row.get("kind") != kind:
                raise ValueError(f"{kind} row in pair {index} has wrong kind")
            if _binary_label(
                row.get("label"),
                label=f"{kind} row in pair {index} label",
            ) != expected_label:
                raise ValueError(f"{kind} row in pair {index} has wrong label")
            _finite_real(
                row.get("ai_score"),
                label=f"{kind} row in pair {index} ai_score",
            )

        real_task = _nonempty_string(
            real.get("task_id"),
            label=f"real row in pair {index} task_id",
        )
        forged_task = _nonempty_string(
            forged.get("task_id"),
            label=f"forged row in pair {index} task_id",
        )
        if real_task != forged_task:
            raise ValueError(f"pair {index} has mismatched task IDs")
        if real_task in seen_task_ids:
            raise ValueError(f"duplicate task pair {real_task}")
        seen_task_ids.add(real_task)

        real_domain = _nonempty_string(
            real.get("domain"),
            label=f"real row in pair {index} domain",
        )
        forged_domain = _nonempty_string(
            forged.get("domain"),
            label=f"forged row in pair {index} domain",
        )
        if real_domain != forged_domain:
            raise ValueError(f"pair {real_task} has mismatched domains")

        real_visibility, real_fraction = _validated_edit_visibility(
            real.get("edit_visibility"),
            real.get("edit_visible_gt_fraction"),
            label=f"real row in pair {index}",
        )
        forged_visibility, forged_fraction = _validated_edit_visibility(
            forged.get("edit_visibility"),
            forged.get("edit_visible_gt_fraction"),
            label=f"forged row in pair {index}",
        )
        if real_visibility != forged_visibility:
            raise ValueError(
                f"pair {real_task} has mismatched edit_visibility"
            )
        if not math.isclose(
            real_fraction,
            forged_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"pair {real_task} has mismatched edit_visible_gt_fraction"
            )

        real_rows.append(real)
        forged_rows.append(forged)
        domains.append(real_domain)
        visibilities.append(real_visibility)
        visible_fractions.append(forged_fraction)

    return (
        real_rows,
        forged_rows,
        domains,
        visibilities,
        np.asarray(visible_fractions, dtype=np.float64),
    )


def _paired_point_metrics(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
) -> dict[str, float]:
    real = np.asarray(real_scores, dtype=np.float64)
    forged = np.asarray(forged_scores, dtype=np.float64)
    if real.ndim != 1 or forged.ndim != 1 or real.shape != forged.shape:
        raise ValueError("paired score arrays have incompatible shapes")
    if not real.size:
        raise ValueError("paired score arrays are empty")
    if not np.isfinite(real).all() or not np.isfinite(forged).all():
        raise ValueError("paired score arrays contain non-finite values")
    delta = _finite_pair_deltas(real, forged)

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
    delta_mean = statistics.fmean(float(value) for value in delta)
    if not math.isfinite(delta_mean):
        raise ValueError("paired score delta mean is not finite")
    return {
        "auroc": float(detection["auroc"]),
        "average_precision": float(detection["average_precision"]),
        "tpr_at_fpr_5_percent": float(
            detection["tpr_at_fpr_5_percent"]
        ),
        "tpr_at_fpr_5_percent_threshold": float(
            detection["tpr_at_fpr_5_percent_threshold"]
        ),
        "tpr_at_fpr_5_percent_actual_fpr": float(
            detection["tpr_at_fpr_5_percent_actual_fpr"]
        ),
        "accuracy_at_2": float(detection["accuracy"]),
        "balanced_accuracy_at_2": float(
            detection["balanced_accuracy"]
        ),
        "precision_at_2": float(detection["precision"]),
        "recall_at_2": float(detection["recall"]),
        "f1_at_2": float(detection["f1"]),
        "specificity_at_2": float(detection["specificity"]),
        "paired_ranking_accuracy": float(np.mean(delta > 0.0)),
        "paired_score_delta_mean": float(delta_mean),
    }


def _percentile_ci(values: Iterable[float]) -> list[float]:
    vector = np.asarray(list(values), dtype=np.float64)
    if not vector.size:
        raise ValueError("cannot calculate a confidence interval from no values")
    if not np.isfinite(vector).all():
        raise ValueError("confidence interval values contain non-finite values")
    return [
        float(np.percentile(vector, 2.5)),
        float(np.percentile(vector, 97.5)),
    ]


def _sign_test(delta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    wins = int(np.count_nonzero(values > 0.0))
    losses = int(np.count_nonzero(values < 0.0))
    ties = int(np.count_nonzero(values == 0.0))
    non_ties = wins + losses
    if non_ties:
        lower_tail = min(wins, losses)
        p_value = min(
            1.0,
            2.0
            * sum(
                math.comb(non_ties, count)
                for count in range(lower_tail + 1)
            )
            / (2**non_ties),
        )
    else:
        p_value = 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_ties": non_ties,
        "alternative": "two-sided",
        "null_win_probability": 0.5,
        "two_sided_exact_p": float(p_value),
    }


def summarize_whole_image_pair_slice(
    pairs: list[Any],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize and pair-bootstrap a non-empty complete-pair slice."""

    if not pairs:
        raise ValueError("pair slice is empty")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (numbers.Integral, np.integer))
        or int(iterations) <= 0
    ):
        raise ValueError("bootstrap iterations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (numbers.Integral, np.integer)
    ):
        raise ValueError("bootstrap seed must be an integer")

    (
        real_rows,
        forged_rows,
        domains,
        visibilities,
        visible_fractions,
    ) = _validate_pair_rows(pairs)
    real_scores = np.asarray(
        [float(row["ai_score"]) for row in real_rows],
        dtype=np.float64,
    )
    forged_scores = np.asarray(
        [float(row["ai_score"]) for row in forged_rows],
        dtype=np.float64,
    )
    point = _paired_point_metrics(real_scores, forged_scores)

    rng = np.random.default_rng(int(seed))
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    pair_count = len(pairs)
    for _ in range(int(iterations)):
        indices = rng.integers(0, pair_count, size=pair_count)
        # _paired_point_metrics recomputes the real-only higher quantile.
        replicate = _paired_point_metrics(
            real_scores[indices],
            forged_scores[indices],
        )
        for name in BOOTSTRAP_METRICS:
            replicates[name].append(float(replicate[name]))

    labels = np.concatenate(
        [
            np.zeros(pair_count, dtype=np.int64),
            np.ones(pair_count, dtype=np.int64),
        ]
    )
    scores = np.concatenate([real_scores, forged_scores])
    confusion = _confusion_metrics(labels, scores > FIXED_THRESHOLD)
    delta = _finite_pair_deltas(real_scores, forged_scores)
    wins = int(np.count_nonzero(delta > 0.0))
    losses = int(np.count_nonzero(delta < 0.0))
    ties = int(np.count_nonzero(delta == 0.0))

    return {
        "pairs": pair_count,
        "images": pair_count * 2,
        "bootstrap_unit": "task_id_pair",
        "bootstrap_samples": int(iterations),
        "seed": int(seed),
        "score_key": "ai_score",
        "score_direction": "higher_means_fake",
        "fixed_threshold": FIXED_THRESHOLD,
        "fixed_threshold_operator": THRESHOLD_OPERATOR,
        "fpr_target": TARGET_FPR,
        "fpr_threshold_source": (
            "real_scores_quantile_0_95_method_higher"
        ),
        "fpr_threshold_operator": THRESHOLD_OPERATOR,
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_2": {
            key: confusion[key]
            for key in ("tp", "fp", "fn", "tn")
        },
        "paired_score_delta": descriptive(delta.tolist()),
        "paired_ranking": {
            "comparison": "forged_ai_score_strictly_greater_than_real",
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "strict_accuracy": wins / pair_count,
        },
        "paired_sign_test": _sign_test(delta),
        "domains": sorted(set(domains)),
        "edit_visibilities": sorted(set(visibilities)),
        "edit_visible_gt_fraction": descriptive(
            visible_fractions.tolist()
        ),
    }


def _expected_identity(
    row: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    sample_id = _nonempty_string(
        row.get("sample_id"),
        label=f"expected row {index} sample_id",
    )
    task_id = _nonempty_string(
        row.get("task_id"),
        label=f"expected row {index} task_id",
    )
    kind = row.get("kind")
    if kind not in ("real", "forged"):
        raise ValueError(f"expected row {index} has invalid kind")
    label = _binary_label(
        row.get("label"),
        label=f"expected row {index} label",
    )
    if label != int(kind == "forged"):
        raise ValueError(f"expected row {index} has kind/label mismatch")
    domain = _nonempty_string(
        row.get("domain"),
        label=f"expected row {index} domain",
    )
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "kind": kind,
        "label": label,
        "domain": domain,
    }


def _validate_expected_rows(
    expected_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    expected_by_id: dict[str, dict[str, Any]] = {}
    kinds_by_task: dict[str, set[str]] = defaultdict(set)
    domains_by_task: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(expected_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"expected row {index} is not an object")
        identity = _expected_identity(row, index=index)
        sample_id = identity["sample_id"]
        task_id = identity["task_id"]
        kind = identity["kind"]
        if sample_id in expected_by_id:
            raise ValueError(f"duplicate expected sample_id {sample_id}")
        if kind in kinds_by_task[task_id]:
            raise ValueError(f"duplicate expected {kind} row for task {task_id}")
        expected_by_id[sample_id] = identity
        kinds_by_task[task_id].add(kind)
        domains_by_task[task_id].add(str(identity["domain"]))
    mismatched_domains = sorted(
        task_id
        for task_id, domains in domains_by_task.items()
        if len(domains) != 1
    )
    if mismatched_domains:
        raise ValueError(
            "expected rows have mismatched domains within task "
            f"{mismatched_domains[0]}"
        )
    return expected_by_id, kinds_by_task


def _validate_result_identity(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    index: int,
) -> None:
    row_id = _nonempty_string(row.get("id"), label=f"result row {index} id")
    if row_id != expected["sample_id"]:
        raise ValueError(f"result row {index} id does not match expected row")
    for key in ("task_id", "kind", "domain"):
        if row.get(key) != expected[key]:
            raise ValueError(
                f"result row {index} {key} does not match expected identity"
            )
    label = _binary_label(
        row.get("label"),
        label=f"result row {index} label",
    )
    if label != expected["label"]:
        raise ValueError(
            f"result row {index} label does not match expected identity"
        )

    _validated_edit_visibility(
        row.get("edit_visibility"),
        row.get("edit_visible_gt_fraction"),
        label=f"result row {index}",
    )

    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError(f"result row {index} has invalid status")
    if status == "ok":
        _finite_real(
            row.get("ai_score"),
            label=f"result row {index} ai_score",
        )
        latency_ms = _finite_real(
            row.get("latency_ms"),
            label=f"result row {index} latency_ms",
        )
        if latency_ms < 0.0:
            raise ValueError(
                f"result row {index} latency_ms is negative"
            )
        if "peak_cuda_memory_bytes" not in row:
            raise ValueError(
                f"result row {index} peak_cuda_memory_bytes is missing"
            )
        if row["peak_cuda_memory_bytes"] is not None:
            _nonnegative_integer(
                row["peak_cuda_memory_bytes"],
                label=f"result row {index} peak_cuda_memory_bytes",
            )
    elif "ai_score" in row and row.get("ai_score") is not None:
        _finite_real(
            row.get("ai_score"),
            label=f"error result row {index} ai_score",
        )


def _complete_pairs(
    valid_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, dict[str, Any]]], dict[str, set[str]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid_rows:
        task_id = str(row["task_id"])
        kind = str(row["kind"])
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate valid {kind} row for task {task_id}")
        by_task[task_id][kind] = row

    pairs = [
        {"real": values["real"], "forged": values["forged"]}
        for task_id, values in sorted(by_task.items())
        if set(values) == {"real", "forged"}
    ]
    _validate_pair_rows(pairs)
    return pairs, {task: set(values) for task, values in by_task.items()}


def _score_by_kind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["kind"])].append(float(row["ai_score"]))
    return {
        kind: descriptive(values)
        for kind, values in sorted(by_kind.items())
    }


def summarize_whole_image_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate coverage and summarize a whole-image detector run.

    ``rows`` may contain retry history; every physical row is validated and
    the last row for each expected ID is authoritative.  ``expected_rows`` may
    intentionally contain incomplete pairs for preflight runs.  The resulting
    coverage fields make that limitation explicit and bootstrap statistics use
    complete successful pairs only.
    """

    threshold_value = _require_fixed_threshold(threshold)
    if (
        isinstance(bootstrap_samples, (bool, np.bool_))
        or not isinstance(bootstrap_samples, (numbers.Integral, np.integer))
        or int(bootstrap_samples) <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (numbers.Integral, np.integer)
    ):
        raise ValueError("bootstrap seed must be an integer")

    expected_by_id, expected_kinds = _validate_expected_rows(expected_rows)
    latest: dict[str, dict[str, Any]] = {}
    result_visibilities: dict[str, set[str]] = defaultdict(set)
    result_visible_fractions: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"result row {index} is not an object")
        row_id = _nonempty_string(row.get("id"), label=f"result row {index} id")
        expected = expected_by_id.get(row_id)
        if expected is None:
            raise ValueError(f"unexpected result id {row_id}")
        _validate_result_identity(row, expected, index=index)
        task_id = str(expected["task_id"])
        result_visibilities[task_id].add(str(row["edit_visibility"]))
        result_visible_fractions[task_id].append(
            float(row["edit_visible_gt_fraction"])
        )
        latest[row_id] = dict(row)

    for task_id, values in result_visibilities.items():
        if len(values) != 1:
            raise ValueError(
                f"physical result rows for task {task_id} have mismatched "
                "edit_visibility"
            )
    for task_id, values in result_visible_fractions.items():
        reference = values[0]
        if any(
            not math.isclose(
                value,
                reference,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for value in values[1:]
        ):
            raise ValueError(
                f"physical result rows for task {task_id} have mismatched "
                "edit_visible_gt_fraction"
            )

    selected = [
        latest[sample_id]
        for sample_id in expected_by_id
        if sample_id in latest
    ]
    valid = [row for row in selected if row["status"] == "ok"]
    pairs, valid_kinds = _complete_pairs(valid)
    paired_ids = {
        str(pair[kind]["id"])
        for pair in pairs
        for kind in ("real", "forged")
    }

    expected_complete_tasks = {
        task_id
        for task_id, kinds in expected_kinds.items()
        if kinds == {"real", "forged"}
    }
    expected_incomplete_tasks = set(expected_kinds) - expected_complete_tasks
    valid_complete_tasks = {
        task_id
        for task_id, kinds in valid_kinds.items()
        if kinds == {"real", "forged"}
    }

    pair_bootstrap = (
        summarize_whole_image_pair_slice(
            pairs,
            iterations=int(bootstrap_samples),
            seed=int(seed),
        )
        if pairs
        else None
    )

    domains = sorted(
        {
            str(pair["real"]["domain"])
            for pair in pairs
        }
    )
    by_domain = {
        domain: summarize_whole_image_pair_slice(
            [
                pair
                for pair in pairs
                if pair["real"]["domain"] == domain
            ],
            iterations=int(bootstrap_samples),
            seed=int(seed) + 1000 + index,
        )
        for index, domain in enumerate(domains)
    }
    visibilities = sorted(
        {
            str(pair["real"]["edit_visibility"])
            for pair in pairs
        }
    )
    by_edit_visibility = {
        visibility: summarize_whole_image_pair_slice(
            [
                pair
                for pair in pairs
                if pair["real"]["edit_visibility"] == visibility
            ],
            iterations=int(bootstrap_samples),
            seed=int(seed) + 2000 + index,
        )
        for index, visibility in enumerate(visibilities)
    }

    real_pair_scores = np.asarray(
        [float(pair["real"]["ai_score"]) for pair in pairs],
        dtype=np.float64,
    )
    forged_pair_scores = np.asarray(
        [float(pair["forged"]["ai_score"]) for pair in pairs],
        dtype=np.float64,
    )
    delta_array = _finite_pair_deltas(
        real_pair_scores,
        forged_pair_scores,
    )
    deltas = delta_array.tolist()
    wins = int(np.count_nonzero(delta_array > 0.0))
    losses = int(np.count_nonzero(delta_array < 0.0))
    ties = int(np.count_nonzero(delta_array == 0.0))
    expected_images = len(expected_by_id)
    result_images = len(selected)
    missing_images = expected_images - result_images
    error_images = result_images - len(valid)

    return {
        "schema_version": "whole_image_detection_summary_v1",
        "task_scope": {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "primary_score": "ai_score",
            "score_direction": "higher_means_fake",
            "score_range": (
                "finite_binary64_with_finite_paired_differences"
            ),
            "released_threshold": threshold_value,
            "released_threshold_operator": THRESHOLD_OPERATOR,
            "fpr_target": TARGET_FPR,
            "fpr_threshold_source": (
                "real_scores_quantile_0_95_method_higher"
            ),
            "fpr_threshold_operator": THRESHOLD_OPERATOR,
            "bootstrap_unit": "task_id_pair",
            "precision_zero_division": 0,
        },
        "coverage": {
            "expected_images": expected_images,
            "physical_result_rows": len(rows),
            "result_images": result_images,
            "valid_images": len(valid),
            "error_images": error_images,
            "missing_images": missing_images,
            "coverage_fraction": (
                result_images / expected_images if expected_images else 1.0
            ),
            "valid_fraction": (
                len(valid) / expected_images if expected_images else 1.0
            ),
            "is_complete": missing_images == 0 and error_images == 0,
        },
        "paired_coverage": {
            "expected_tasks": len(expected_kinds),
            "expected_complete_pairs": len(expected_complete_tasks),
            "expected_incomplete_tasks": len(expected_incomplete_tasks),
            "preflight_expected_incomplete_pairs": bool(
                expected_incomplete_tasks
            ),
            "complete_valid_pairs": len(pairs),
            "paired_valid_images": len(paired_ids),
            "unpaired_valid_images": len(valid) - len(paired_ids),
            "valid_complete_task_fraction": (
                len(valid_complete_tasks) / len(expected_complete_tasks)
                if expected_complete_tasks
                else None
            ),
        },
        "score_by_kind": _score_by_kind(valid),
        "detection": image_detection_metrics_strict(
            valid,
            threshold_value,
        ),
        "paired_score_delta": descriptive(deltas),
        "paired_ranking_accuracy": (
            wins / len(pairs) if pairs else None
        ),
        "paired_ranking": {
            "comparison": "forged_ai_score_strictly_greater_than_real",
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "strict_accuracy": wins / len(pairs) if pairs else None,
        },
        "paired_sign_test": (
            _sign_test(delta_array) if pairs else None
        ),
        "pair_bootstrap": pair_bootstrap,
        "by_domain": by_domain,
        "by_edit_visibility": by_edit_visibility,
        "edit_visible_gt_fraction": descriptive(
            float(pair["forged"]["edit_visible_gt_fraction"])
            for pair in pairs
        ),
        "latency_ms": descriptive(
            float(row["latency_ms"])
            for row in valid
        ),
        "peak_cuda_memory_bytes": descriptive(
            float(row["peak_cuda_memory_bytes"])
            for row in valid
            if row["peak_cuda_memory_bytes"] is not None
        ),
    }


__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EDIT_VISIBILITIES",
    "FIXED_THRESHOLD",
    "FPR_QUANTILE_METHOD",
    "TARGET_FPR",
    "THRESHOLD_OPERATOR",
    "image_detection_metrics",
    "image_detection_metrics_strict",
    "summarize_whole_image_pair_slice",
    "summarize_whole_image_results",
]
