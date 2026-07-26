"""Strict paired metrics for the official B-Free whole-image detector.

B-Free releases one unbounded raw logit per image.  Larger values mean fake
and the official decision is the strict comparison ``raw_logit > 0``.  This
module intentionally retains that raw score rather than converting it to a
probability.
"""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from eval.opensource.maskclip_metrics import descriptive
from eval.opensource.whole_image_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FPR_QUANTILE_METHOD,
    TARGET_FPR,
    THRESHOLD_OPERATOR,
    _confusion_metrics,
    _detection_from_arrays,
    _finite_pair_deltas,
    _finite_real,
    _nonempty_string,
    _nonnegative_integer,
    _percentile_ci,
    _sign_test,
    _validated_edit_visibility,
)


FIXED_THRESHOLD = 0.0
BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "tpr_at_fpr_5_percent_threshold",
    "tpr_at_fpr_5_percent_actual_fpr",
    "accuracy_at_0",
    "balanced_accuracy_at_0",
    "precision_at_0",
    "recall_at_0",
    "f1_at_0",
    "specificity_at_0",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
)


def _require_threshold(value: Any) -> float:
    threshold = _finite_real(value, label="classification threshold")
    if not math.isclose(
        threshold,
        FIXED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"B-Free uses fixed threshold {FIXED_THRESHOLD}, not {threshold}"
        )
    return FIXED_THRESHOLD


def _binary_label(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Integral, np.integer),
    ):
        raise ValueError(f"{label} is not an integer 0/1 label")
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{label} is not a 0/1 label")
    return result


def bfree_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Calculate strict image metrics without dropping malformed successes."""

    threshold_value = _require_threshold(threshold)
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
                row.get("ai_score"),
                label=f"successful row {index} ai_score",
            )
        )
    result = _detection_from_arrays(
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        threshold=threshold_value,
    )
    result["score_key"] = "ai_score"
    result["score_semantics"] = (
        "official_float32_mean_of_five_crop_raw_logits"
    )
    result["score_range"] = "unbounded_finite_raw_logit"
    return result


def bfree_fake_probability_float32(raw_logit: Any) -> float:
    """Return the diagnostic float32 sigmoid without changing the primary."""

    score = _finite_real(raw_logit, label="B-Free raw logit")
    value = np.asarray(score, dtype=np.float32)
    # This branch form avoids overflow warnings while matching float32
    # sigmoid limits exactly.
    if value >= 0:
        probability = np.float32(1.0) / (
            np.float32(1.0) + np.exp(-value, dtype=np.float32)
        )
    else:
        exponential = np.exp(value, dtype=np.float32)
        probability = exponential / (np.float32(1.0) + exponential)
    return float(np.float32(probability))


def official_balanced_calibration(
    rows: list[dict[str, Any]],
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Reproduce the released balanced NLL and balanced ECE definitions."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("calibration bins must be a positive integer")
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if row.get("status") == "error":
            continue
        if row.get("status") != "ok":
            raise ValueError(f"result row {index} has invalid status")
        labels.append(
            _binary_label(
                row.get("label"),
                label=f"calibration row {index} label",
            )
        )
        scores.append(
            _finite_real(
                row.get("ai_score"),
                label=f"calibration row {index} ai_score",
            )
        )
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    classes, inverse, counts = np.unique(
        label_array,
        return_inverse=True,
        return_counts=True,
    )
    if classes.tolist() != [0, 1]:
        return {
            "balanced_nll": None,
            "balanced_ece_15_bins": None,
            "bins": bins,
            "status": "requires_both_classes",
            "definition": "official_B-Free_code_utils.dmetrics",
        }
    signed = 2 * inverse - 1
    losses = np.logaddexp(-signed * score_array, 0.0)
    balanced_nll = 0.5 * (
        float(np.mean(losses[signed < 0]))
        + float(np.mean(losses[signed > 0]))
    )
    probabilities = np.empty_like(score_array)
    nonnegative = score_array >= 0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-score_array[nonnegative])
    )
    exponential = np.exp(score_array[~nonnegative])
    probabilities[~nonnegative] = exponential / (1.0 + exponential)
    sample_weight = (1.0 / counts)[inverse]
    interval = np.floor(bins * probabilities)
    total_weight = float(sample_weight.sum())
    ece = 0.0
    correctness = inverse == 1
    # Intentionally mirrors the official range(bins): probability == 1 maps
    # to interval ``bins`` and is not assigned to a bin.
    for bin_index in range(bins):
        in_bin = interval == bin_index
        if not np.any(in_bin):
            continue
        weight = sample_weight[in_bin]
        weight_mean = float(np.mean(weight))
        accuracy = float(np.mean(weight * correctness[in_bin])) / weight_mean
        confidence = (
            float(np.mean(weight * probabilities[in_bin])) / weight_mean
        )
        proportion = float(weight.sum()) / total_weight
        ece += abs(confidence - accuracy) * proportion
    return {
        "balanced_nll": float(balanced_nll),
        "balanced_ece_15_bins": float(ece),
        "bins": bins,
        "status": "ok",
        "definition": "official_B-Free_code_utils.dmetrics",
        "class_counts": {
            "real": int(counts[0]),
            "fake": int(counts[1]),
        },
    }


def _pair_rows(
    pairs: list[Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    real_rows: list[Mapping[str, Any]] = []
    forged_rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"pair {index} is not an object")
        real, forged = pair.get("real"), pair.get("forged")
        if not isinstance(real, Mapping) or not isinstance(forged, Mapping):
            raise ValueError(f"pair {index} lacks real/forged rows")
        for kind, expected_label, row in (
            ("real", 0, real),
            ("forged", 1, forged),
        ):
            if row.get("status") != "ok":
                raise ValueError(f"{kind} row in pair {index} is not ok")
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
        task_id = _nonempty_string(
            real.get("task_id"),
            label=f"real row in pair {index} task_id",
        )
        if forged.get("task_id") != task_id:
            raise ValueError(f"pair {index} has mismatched task IDs")
        if task_id in seen:
            raise ValueError(f"duplicate task pair {task_id}")
        seen.add(task_id)
        if real.get("domain") != forged.get("domain"):
            raise ValueError(f"pair {task_id} has mismatched domains")
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
        if real_visibility != forged_visibility or not math.isclose(
            real_fraction,
            forged_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"pair {task_id} has mismatched edit visibility")
        real_rows.append(real)
        forged_rows.append(forged)
    return real_rows, forged_rows


def _pair_point(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
) -> dict[str, float]:
    delta = _finite_pair_deltas(real_scores, forged_scores)
    labels = np.concatenate(
        [
            np.zeros(real_scores.size, dtype=np.int64),
            np.ones(forged_scores.size, dtype=np.int64),
        ]
    )
    scores = np.concatenate([real_scores, forged_scores])
    detection = _detection_from_arrays(
        labels,
        scores,
        threshold=FIXED_THRESHOLD,
    )
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
        "accuracy_at_0": float(detection["accuracy"]),
        "balanced_accuracy_at_0": float(detection["balanced_accuracy"]),
        "precision_at_0": float(detection["precision"]),
        "recall_at_0": float(detection["recall"]),
        "f1_at_0": float(detection["f1"]),
        "specificity_at_0": float(detection["specificity"]),
        "paired_ranking_accuracy": float(np.mean(delta > 0.0)),
        "paired_score_delta_mean": float(np.mean(delta)),
    }


def summarize_bfree_pair_slice(
    pairs: list[Any],
    threshold: float = FIXED_THRESHOLD,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize and pair-bootstrap a non-empty complete-pair slice."""

    _require_threshold(threshold)
    if not pairs:
        raise ValueError("pair slice is empty")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (numbers.Integral, np.integer))
        or int(iterations) <= 0
    ):
        raise ValueError("bootstrap iterations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (numbers.Integral, np.integer),
    ):
        raise ValueError("bootstrap seed must be an integer")
    real_rows, forged_rows = _pair_rows(pairs)
    real = np.asarray([row["ai_score"] for row in real_rows], dtype=np.float64)
    forged = np.asarray(
        [row["ai_score"] for row in forged_rows],
        dtype=np.float64,
    )
    point = _pair_point(real, forged)
    rng = np.random.default_rng(int(seed))
    boot = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(int(iterations)):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        replicate = _pair_point(real[indices], forged[indices])
        for name in BOOTSTRAP_METRICS:
            boot[name].append(replicate[name])
    labels = np.concatenate(
        [
            np.zeros(len(pairs), dtype=np.int64),
            np.ones(len(pairs), dtype=np.int64),
        ]
    )
    scores = np.concatenate([real, forged])
    confusion = _confusion_metrics(labels, scores > FIXED_THRESHOLD)
    delta = _finite_pair_deltas(real, forged)
    return {
        "pairs": len(pairs),
        "images": 2 * len(pairs),
        "bootstrap_unit": "task_id_pair",
        "bootstrap_samples": int(iterations),
        "seed": int(seed),
        "score_key": "ai_score",
        "score_semantics": "official_float32_mean_of_five_crop_raw_logits",
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
                "estimate": point[name],
                "ci95_percentile": _percentile_ci(boot[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_0": {
            key: confusion[key] for key in ("tp", "fp", "fn", "tn")
        },
        "paired_score_delta": descriptive(delta.tolist()),
        "paired_ranking": {
            "comparison": "forged_ai_score_strictly_greater_than_real",
            "wins": int(np.count_nonzero(delta > 0.0)),
            "losses": int(np.count_nonzero(delta < 0.0)),
            "ties": int(np.count_nonzero(delta == 0.0)),
            "strict_accuracy": float(np.mean(delta > 0.0)),
        },
        "paired_sign_test": _sign_test(delta),
        "domains": sorted({str(row["domain"]) for row in real_rows}),
        "edit_visibilities": sorted(
            {str(row["edit_visibility"]) for row in real_rows}
        ),
        "edit_visible_gt_fraction": descriptive(
            float(row["edit_visible_gt_fraction"]) for row in real_rows
        ),
    }


def _expected_identity(row: Mapping[str, Any], index: int) -> dict[str, Any]:
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
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "kind": kind,
        "label": label,
        "domain": _nonempty_string(
            row.get("domain"),
            label=f"expected row {index} domain",
        ),
    }


def _validate_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    index: int,
) -> None:
    if row.get("id") != expected["sample_id"]:
        raise ValueError(f"result row {index} id does not match expected")
    for key in ("task_id", "kind", "label", "domain"):
        if row.get(key) != expected[key]:
            raise ValueError(
                f"result row {index} {key} does not match expected identity"
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
        score = _finite_real(
            row.get("ai_score"),
            label=f"result row {index} ai_score",
        )
        raw = _finite_real(
            row.get("raw_logit"),
            label=f"result row {index} raw_logit",
        )
        if score != raw:
            raise ValueError(f"result row {index} raw_logit differs from score")
        latency = _finite_real(
            row.get("latency_ms"),
            label=f"result row {index} latency_ms",
        )
        if latency < 0:
            raise ValueError(f"result row {index} latency is negative")
        if "peak_cuda_memory_bytes" not in row:
            raise ValueError(
                f"result row {index} peak_cuda_memory_bytes is missing"
            )
        if row["peak_cuda_memory_bytes"] is not None:
            _nonnegative_integer(
                row["peak_cuda_memory_bytes"],
                label=f"result row {index} peak_cuda_memory_bytes",
            )


def summarize_bfree_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate retry history and summarize B-Free raw-logit results."""

    _require_threshold(threshold)
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    expected_by_id: dict[str, dict[str, Any]] = {}
    expected_kinds: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(expected_rows):
        identity = _expected_identity(row, index)
        if identity["sample_id"] in expected_by_id:
            raise ValueError("duplicate expected sample_id")
        if identity["kind"] in expected_kinds[identity["task_id"]]:
            raise ValueError("duplicate expected task kind")
        expected_by_id[identity["sample_id"]] = identity
        expected_kinds[identity["task_id"]].add(identity["kind"])

    latest: dict[str, dict[str, Any]] = {}
    physical_visibility: dict[str, set[tuple[str, float]]] = defaultdict(set)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"result row {index} is not an object")
        row_id = _nonempty_string(
            row.get("id"),
            label=f"result row {index} id",
        )
        if row_id not in expected_by_id:
            raise ValueError(f"unexpected result id {row_id}")
        expected = expected_by_id[row_id]
        _validate_result(row, expected, index)
        physical_visibility[str(expected["task_id"])].add(
            (
                str(row["edit_visibility"]),
                float(row["edit_visible_gt_fraction"]),
            )
        )
        latest[row_id] = dict(row)
    if any(len(values) != 1 for values in physical_visibility.values()):
        raise ValueError("physical retry rows have mismatched edit visibility")

    selected = [
        latest[sample_id]
        for sample_id in expected_by_id
        if sample_id in latest
    ]
    valid = [row for row in selected if row["status"] == "ok"]
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid:
        by_task[str(row["task_id"])][str(row["kind"])] = row
    pairs = [
        {"real": values["real"], "forged": values["forged"]}
        for _, values in sorted(by_task.items())
        if set(values) == {"real", "forged"}
    ]
    if pairs:
        _pair_rows(pairs)

    def slices(key: str, offset: int) -> dict[str, Any]:
        values = sorted({str(pair["real"][key]) for pair in pairs})
        return {
            value: summarize_bfree_pair_slice(
                [pair for pair in pairs if str(pair["real"][key]) == value],
                iterations=int(bootstrap_samples),
                seed=int(seed) + offset + index,
            )
            for index, value in enumerate(values)
        }

    delta = _finite_pair_deltas(
        np.asarray(
            [float(pair["real"]["ai_score"]) for pair in pairs],
            dtype=np.float64,
        ),
        np.asarray(
            [float(pair["forged"]["ai_score"]) for pair in pairs],
            dtype=np.float64,
        ),
    )
    expected_complete = sum(
        kinds == {"real", "forged"} for kinds in expected_kinds.values()
    )
    paired_ids = {
        str(pair[kind]["id"])
        for pair in pairs
        for kind in ("real", "forged")
    }
    missing = len(expected_by_id) - len(selected)
    errors = len(selected) - len(valid)
    return {
        "schema_version": "bfree_detection_summary_v1",
        "task_scope": {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "primary_score": "ai_score",
            "score_semantics": (
                "official_float32_mean_of_five_crop_raw_logits"
            ),
            "score_direction": "higher_means_fake",
            "score_range": "unbounded_finite_raw_logit",
            "released_threshold": FIXED_THRESHOLD,
            "released_threshold_operator": THRESHOLD_OPERATOR,
            "bootstrap_unit": "task_id_pair",
        },
        "coverage": {
            "expected_images": len(expected_by_id),
            "physical_result_rows": len(rows),
            "result_images": len(selected),
            "valid_images": len(valid),
            "error_images": errors,
            "missing_images": missing,
            "coverage_fraction": (
                len(selected) / len(expected_by_id) if expected_by_id else 1.0
            ),
            "valid_fraction": (
                len(valid) / len(expected_by_id) if expected_by_id else 1.0
            ),
            "is_complete": missing == 0 and errors == 0,
        },
        "paired_coverage": {
            "expected_tasks": len(expected_kinds),
            "expected_complete_pairs": expected_complete,
            "expected_incomplete_tasks": (
                len(expected_kinds) - expected_complete
            ),
            "preflight_expected_incomplete_pairs": (
                expected_complete != len(expected_kinds)
            ),
            "complete_valid_pairs": len(pairs),
            "paired_valid_images": len(paired_ids),
            "unpaired_valid_images": len(valid) - len(paired_ids),
            "valid_complete_task_fraction": (
                len(pairs) / expected_complete if expected_complete else None
            ),
        },
        "score_by_kind": {
            kind: descriptive(
                float(row["ai_score"])
                for row in valid
                if row["kind"] == kind
            )
            for kind in sorted({str(row["kind"]) for row in valid})
        },
        "detection": bfree_detection_metrics_strict(valid),
        "official_calibration": official_balanced_calibration(valid),
        "paired_score_delta": descriptive(delta.tolist()),
        "paired_ranking_accuracy": (
            float(np.mean(delta > 0.0)) if pairs else None
        ),
        "paired_ranking": {
            "comparison": "forged_ai_score_strictly_greater_than_real",
            "wins": int(np.count_nonzero(delta > 0.0)),
            "losses": int(np.count_nonzero(delta < 0.0)),
            "ties": int(np.count_nonzero(delta == 0.0)),
            "strict_accuracy": (
                float(np.mean(delta > 0.0)) if pairs else None
            ),
        },
        "paired_sign_test": _sign_test(delta) if pairs else None,
        "pair_bootstrap": (
            summarize_bfree_pair_slice(
                pairs,
                iterations=int(bootstrap_samples),
                seed=int(seed),
            )
            if pairs
            else None
        ),
        "by_domain": slices("domain", 1000),
        "by_edit_visibility": slices("edit_visibility", 2000),
        "edit_visible_gt_fraction": descriptive(
            float(pair["forged"]["edit_visible_gt_fraction"])
            for pair in pairs
        ),
        "latency_ms": descriptive(float(row["latency_ms"]) for row in valid),
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
    "FIXED_THRESHOLD",
    "FPR_QUANTILE_METHOD",
    "TARGET_FPR",
    "THRESHOLD_OPERATOR",
    "bfree_detection_metrics_strict",
    "bfree_fake_probability_float32",
    "official_balanced_calibration",
    "summarize_bfree_pair_slice",
    "summarize_bfree_results",
]
