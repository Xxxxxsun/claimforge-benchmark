"""Strict paired metrics for the official CNNDetection classifier.

CNNDetection exposes one scalar logit.  The released image-level score is the
float32 sigmoid of that logit and the released operating point is the strict
rule ``score > 0.5``.  The score is explicitly *uncalibrated* in the official
README, even though it lies in ``[0, 1]``.

The probability-score metric contract is identical to the already shared
UniversalFakeDetect contract, so this module deliberately delegates the
well-tested validation/bootstrap machinery instead of maintaining a second
copy.  CNNDetection-specific schema names and raw-logit diagnostics remain
local to this module.
"""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.opensource.ufd_metrics import (
    BOOTSTRAP_METRICS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    EDIT_VISIBILITIES,
    FIXED_THRESHOLD,
    FPR_QUANTILE_METHOD,
    TARGET_FPR,
    THRESHOLD_OPERATOR,
    summarize_ufd_pair_slice,
    summarize_ufd_results,
    ufd_detection_metrics_strict,
)


def cnndetection_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Validate and summarize released sigmoid scores."""

    result = dict(ufd_detection_metrics_strict(rows, threshold))
    result["score_semantics"] = (
        "official_float32_sigmoid_uncalibrated_fake_score"
    )
    return result


def summarize_cnndetection_pair_slice(
    pairs: list[Any],
    threshold: float = FIXED_THRESHOLD,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pair-bootstrap a non-empty CNNDetection slice."""

    result = dict(
        summarize_ufd_pair_slice(
            pairs,
            threshold=threshold,
            iterations=iterations,
            seed=seed,
        )
    )
    result["score_semantics"] = (
        "official_float32_sigmoid_uncalibrated_fake_score"
    )
    return result


def summarize_cnndetection_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate physical retry history and summarize its latest rows."""

    result = dict(
        summarize_ufd_results(
            rows,
            expected_rows,
            threshold=threshold,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )
    result["schema_version"] = "cnndetection_detection_summary_v1"
    result["task_scope"] = {
        **dict(result["task_scope"]),
        "score_semantics": (
            "official_float32_sigmoid_uncalibrated_fake_score"
        ),
        "calibrated_probability": False,
    }
    result["detection"] = cnndetection_detection_metrics_strict(
        [
            row
            for row in _latest_expected_rows(rows, expected_rows)
            if row.get("status") == "ok"
        ],
        threshold,
    )
    return result


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _latest_expected_rows(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_ids: list[str] = []
    for index, row in enumerate(expected_rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"expected row {index} has invalid sample_id")
        if sample_id in expected_ids:
            raise ValueError(f"duplicate expected sample_id {sample_id}")
        expected_ids.append(sample_id)
    expected_set = set(expected_ids)
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"result row {index} is not an object")
        row_id = row.get("id")
        if row_id not in expected_set:
            raise ValueError(f"unexpected result id {row_id}")
        latest[str(row_id)] = dict(row)
    return [latest[item] for item in expected_ids if item in latest]


def summarize_cnndetection_raw_logits(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Always-report numerical diagnostic using the monotone raw logits.

    This diagnostic never replaces the released sigmoid score or threshold.
    It exists so sigmoid underflow/overflow cannot silently create ranking
    ties in AP/AUROC or paired-delta summaries.
    """

    latest = _latest_expected_rows(rows, expected_rows)
    valid = [row for row in latest if row.get("status") == "ok"]
    labels: list[int] = []
    logits: list[float] = []
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, row in enumerate(valid):
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(
            label,
            (numbers.Integral, np.integer),
        ):
            raise ValueError(f"successful row {index} has invalid label")
        label_value = int(label)
        if label_value not in (0, 1):
            raise ValueError(f"successful row {index} has invalid label")
        logit = _finite_real(
            row.get("raw_logit"),
            f"successful row {index} raw_logit",
        )
        labels.append(label_value)
        logits.append(logit)
        task_id = row.get("task_id")
        kind = row.get("kind")
        if not isinstance(task_id, str) or kind not in ("real", "forged"):
            raise ValueError(f"successful row {index} has invalid pair identity")
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate valid {kind} row for task {task_id}")
        by_task[task_id][str(kind)] = row

    labels_array = np.asarray(labels, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    predictions = logits_array > 0.0
    positive = labels_array == 1
    negative = labels_array == 0
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & negative))
    fn = int(np.count_nonzero(~predictions & positive))
    tn = int(np.count_nonzero(~predictions & negative))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None

    auroc = None
    average_precision = None
    fpr_threshold = None
    actual_fpr = None
    tpr = None
    if positive.any() and negative.any():
        auroc = float(roc_auc_score(labels_array, logits_array))
        average_precision = float(
            average_precision_score(labels_array, logits_array)
        )
        fpr_threshold = float(
            np.quantile(
                logits_array[negative],
                1.0 - TARGET_FPR,
                method=FPR_QUANTILE_METHOD,
            )
        )
        actual_fpr = float(np.mean(logits_array[negative] > fpr_threshold))
        tpr = float(np.mean(logits_array[positive] > fpr_threshold))

    pairs = [
        values
        for _, values in sorted(by_task.items())
        if set(values) == {"real", "forged"}
    ]
    delta = np.asarray(
        [
            _finite_real(pair["forged"].get("raw_logit"), "forged raw_logit")
            - _finite_real(pair["real"].get("raw_logit"), "real raw_logit")
            for pair in pairs
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": "cnndetection_raw_logit_diagnostic_v1",
        "policy": (
            "always reported beside released sigmoid metrics; never used to "
            "replace the official strict sigmoid>0.5 decision"
        ),
        "score_key": "raw_logit",
        "score_direction": "higher_means_fake",
        "threshold": 0.0,
        "threshold_operator": THRESHOLD_OPERATOR,
        "valid_images": int(labels_array.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy_at_logit_0": (
            float(np.mean(predictions == positive))
            if labels_array.size
            else None
        ),
        "balanced_accuracy_at_logit_0": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
        "auroc": auroc,
        "average_precision": average_precision,
        "tpr_at_fpr_5_percent": tpr,
        "tpr_at_fpr_5_percent_threshold": fpr_threshold,
        "tpr_at_fpr_5_percent_actual_fpr": actual_fpr,
        "tpr_at_fpr_5_percent_threshold_source": (
            "real_raw_logits_quantile_0_95_method_higher"
        ),
        "complete_pairs": len(pairs),
        "paired_ranking_accuracy": (
            float(np.mean(delta > 0.0)) if delta.size else None
        ),
        "paired_logit_delta": {
            "count": int(delta.size),
            "mean": float(np.mean(delta)) if delta.size else None,
            "minimum": float(np.min(delta)) if delta.size else None,
            "maximum": float(np.max(delta)) if delta.size else None,
        },
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
    "cnndetection_detection_metrics_strict",
    "summarize_cnndetection_pair_slice",
    "summarize_cnndetection_raw_logits",
    "summarize_cnndetection_results",
]
