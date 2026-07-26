"""Strict paired metrics for the Community Forensics image classifier.

Community Forensics and UniversalFakeDetect expose the same benchmark-facing
contract: one finite fake probability per image, a released strict ``> 0.5``
decision, no native localization output, and pair-level CLAIMFORGE
stratification.  The mature UFD implementation is deliberately reused here
instead of maintaining two subtly different bootstrap implementations.
"""

from __future__ import annotations

from typing import Any

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


def community_forensics_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Calculate image-level metrics without silently dropping invalid rows."""

    return ufd_detection_metrics_strict(rows, threshold)


def summarize_community_forensics_pair_slice(
    pairs: list[Any],
    threshold: float = FIXED_THRESHOLD,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize and bootstrap a non-empty set of complete Mouse pairs."""

    return summarize_ufd_pair_slice(pairs, threshold, iterations, seed)


def summarize_community_forensics_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate retry history and summarize Community Forensics results."""

    result = summarize_ufd_results(
        rows,
        expected_rows,
        threshold,
        bootstrap_samples,
        seed,
    )
    result["schema_version"] = "community_forensics_detection_summary_v1"
    return result


# Short aliases are useful in analysis notebooks while the long names keep the
# method unambiguous in adapter and audit code.
commfor_detection_metrics_strict = (
    community_forensics_detection_metrics_strict
)
summarize_commfor_pair_slice = summarize_community_forensics_pair_slice
summarize_commfor_results = summarize_community_forensics_results


__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EDIT_VISIBILITIES",
    "FIXED_THRESHOLD",
    "FPR_QUANTILE_METHOD",
    "TARGET_FPR",
    "THRESHOLD_OPERATOR",
    "community_forensics_detection_metrics_strict",
    "commfor_detection_metrics_strict",
    "summarize_community_forensics_pair_slice",
    "summarize_community_forensics_results",
    "summarize_commfor_pair_slice",
    "summarize_commfor_results",
]
