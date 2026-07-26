"""Strict paired metrics for the official Effort whole-image detector.

Effort's released DeepfakeBench head returns the class-1 fake probability.
The primary ``ai_score`` and its ``fake_probability`` alias must therefore be
finite probabilities in the closed interval ``[0, 1]``.  Larger values mean
"more likely fake", and the released decision rule is the strict comparison
``ai_score > 0.5``; a score exactly equal to 0.5 is classified as real.

The mature UniversalFakeDetect probability-metric implementation supplies the
shared coverage, complete-pair bootstrap, retry-history, domain, visibility,
and real-only 5% FPR machinery.  This module is a deliberately narrow Effort
adapter around that implementation: it validates Effort score aliases and
rebrands every public result contract rather than exposing a UFD schema.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from eval.opensource import ufd_metrics as _shared


FIXED_THRESHOLD = 0.5
THRESHOLD_OPERATOR = ">"
TARGET_FPR = 0.05
FPR_QUANTILE_METHOD = "higher"
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260724
EDIT_VISIBILITIES = frozenset({"none", "partial", "full"})
BOOTSTRAP_METRICS = _shared.BOOTSTRAP_METRICS

SCHEMA_VERSION = "effort_detection_summary_v1"
SCORE_KEY = "ai_score"
SCORE_ALIASES = ("ai_score", "fake_probability")
SCORE_SEMANTICS = "official_float32_softmax_class1_fake_probability"
SCORE_RANGE = "finite_probability_closed_interval_[0,1]"


_SHARED_CONSTANTS = {
    "FIXED_THRESHOLD": _shared.FIXED_THRESHOLD,
    "THRESHOLD_OPERATOR": _shared.THRESHOLD_OPERATOR,
    "TARGET_FPR": _shared.TARGET_FPR,
    "FPR_QUANTILE_METHOD": _shared.FPR_QUANTILE_METHOD,
    "DEFAULT_BOOTSTRAP_SAMPLES": _shared.DEFAULT_BOOTSTRAP_SAMPLES,
    "DEFAULT_BOOTSTRAP_SEED": _shared.DEFAULT_BOOTSTRAP_SEED,
    "EDIT_VISIBILITIES": _shared.EDIT_VISIBILITIES,
}
_EXPECTED_SHARED_CONSTANTS = {
    "FIXED_THRESHOLD": FIXED_THRESHOLD,
    "THRESHOLD_OPERATOR": THRESHOLD_OPERATOR,
    "TARGET_FPR": TARGET_FPR,
    "FPR_QUANTILE_METHOD": FPR_QUANTILE_METHOD,
    "DEFAULT_BOOTSTRAP_SAMPLES": DEFAULT_BOOTSTRAP_SAMPLES,
    "DEFAULT_BOOTSTRAP_SEED": DEFAULT_BOOTSTRAP_SEED,
    "EDIT_VISIBILITIES": EDIT_VISIBILITIES,
}
if _SHARED_CONSTANTS != _EXPECTED_SHARED_CONSTANTS:
    raise RuntimeError(
        "shared probability-metric constants drifted from the Effort contract"
    )


def _require_fixed_threshold(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise ValueError("Effort classification threshold is not a real number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("Effort classification threshold is not finite")
    if threshold != FIXED_THRESHOLD:
        raise ValueError(
            f"Effort uses fixed threshold {FIXED_THRESHOLD}, not {threshold}"
        )
    return FIXED_THRESHOLD


def _probability(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{label} is not finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label} falls outside [0, 1]")
    return probability


def _normalize_score_row(
    row: Any,
    *,
    label: str,
) -> Any:
    """Return a copy with canonical ``ai_score`` and verified aliases.

    ``ai_score`` is the canonical metrics key.  A row carrying only the native
    ``fake_probability`` name is accepted and normalized.  When two or more
    public aliases are present, their values must agree exactly; the runner
    writes the same Python float to every alias, so tolerance would only hide
    persisted-row drift.
    """

    if not isinstance(row, Mapping):
        return row
    copied = dict(row)
    status = copied.get("status")
    present = [
        key
        for key in ("ai_score", "fake_probability", "score")
        if key in copied
    ]
    non_null = [(key, copied[key]) for key in present if copied[key] is not None]

    if status == "ok":
        if "ai_score" not in copied and "fake_probability" in copied:
            copied["ai_score"] = copied["fake_probability"]
            present.append("ai_score")
        if "ai_score" not in copied:
            # Leave the value missing so the shared strict validator produces
            # the canonical successful-row score error.
            return copied
        canonical = _probability(
            copied["ai_score"],
            label=f"{label} ai_score",
        )
        for key in present:
            alias = _probability(
                copied[key],
                label=f"{label} {key}",
            )
            if alias != canonical:
                raise ValueError(
                    f"{label} {key} differs from Effort ai_score"
                )
    else:
        for key, value in non_null:
            _probability(value, label=f"{label} {key}")
        if "ai_score" not in copied and "fake_probability" in copied:
            copied["ai_score"] = copied["fake_probability"]
        non_null_values = [
            _probability(value, label=f"{label} {key}")
            for key, value in non_null
        ]
        if non_null_values and any(
            value != non_null_values[0] for value in non_null_values[1:]
        ):
            raise ValueError(f"{label} Effort probability aliases disagree")
    return copied


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _normalize_score_row(row, label=f"result row {index}")
        for index, row in enumerate(rows)
    ]


def _normalize_pairs(pairs: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, Mapping):
            real = pair.get("real")
            forged = pair.get("forged")
        else:
            real = getattr(pair, "real", None)
            forged = getattr(pair, "forged", None)
        normalized.append(
            {
                "real": _normalize_score_row(
                    real,
                    label=f"real row in pair {index}",
                ),
                "forged": _normalize_score_row(
                    forged,
                    label=f"forged row in pair {index}",
                ),
            }
        )
    return normalized


def _decorate_detection(result: dict[str, Any]) -> dict[str, Any]:
    decorated = dict(result)
    decorated.update(
        {
            "score_key": SCORE_KEY,
            "score_aliases": list(SCORE_ALIASES),
            "score_semantics": SCORE_SEMANTICS,
            "score_range": SCORE_RANGE,
        }
    )
    return decorated


def _decorate_pair_slice(result: dict[str, Any]) -> dict[str, Any]:
    decorated = dict(result)
    decorated.update(
        {
            "score_key": SCORE_KEY,
            "score_aliases": list(SCORE_ALIASES),
            "score_semantics": SCORE_SEMANTICS,
            "score_range": SCORE_RANGE,
        }
    )
    return decorated


def effort_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Calculate strict image-level Effort probability metrics."""

    threshold_value = _require_fixed_threshold(threshold)
    result = _shared.ufd_detection_metrics_strict(
        _normalize_rows(rows),
        threshold=threshold_value,
    )
    return _decorate_detection(result)


def summarize_effort_pair_slice(
    pairs: list[Any],
    threshold: float = FIXED_THRESHOLD,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize and pair-bootstrap a non-empty complete Effort slice."""

    threshold_value = _require_fixed_threshold(threshold)
    result = _shared.summarize_ufd_pair_slice(
        _normalize_pairs(pairs),
        threshold=threshold_value,
        iterations=iterations,
        seed=seed,
    )
    return _decorate_pair_slice(result)


def summarize_effort_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate retry history and summarize the latest Effort result rows."""

    threshold_value = _require_fixed_threshold(threshold)
    result = _shared.summarize_ufd_results(
        _normalize_rows(rows),
        expected_rows,
        threshold=threshold_value,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    if result.get("schema_version") != "ufd_detection_summary_v1":
        raise RuntimeError("shared probability summary schema changed")

    result["schema_version"] = SCHEMA_VERSION
    scope = dict(result["task_scope"])
    scope.update(
        {
            "primary_score": SCORE_KEY,
            "score_aliases": list(SCORE_ALIASES),
            "score_semantics": SCORE_SEMANTICS,
            "score_range": SCORE_RANGE,
        }
    )
    result["task_scope"] = scope
    result["detection"] = _decorate_detection(result["detection"])
    if result["pair_bootstrap"] is not None:
        result["pair_bootstrap"] = _decorate_pair_slice(
            result["pair_bootstrap"]
        )
    result["by_domain"] = {
        key: _decorate_pair_slice(value)
        for key, value in result["by_domain"].items()
    }
    result["by_edit_visibility"] = {
        key: _decorate_pair_slice(value)
        for key, value in result["by_edit_visibility"].items()
    }
    return result


__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EDIT_VISIBILITIES",
    "FIXED_THRESHOLD",
    "FPR_QUANTILE_METHOD",
    "SCHEMA_VERSION",
    "SCORE_ALIASES",
    "SCORE_KEY",
    "SCORE_RANGE",
    "SCORE_SEMANTICS",
    "TARGET_FPR",
    "THRESHOLD_OPERATOR",
    "effort_detection_metrics_strict",
    "summarize_effort_pair_slice",
    "summarize_effort_results",
]
