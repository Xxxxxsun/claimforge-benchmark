"""Strict paired metrics for the official SPAI whole-image detector.

SPAI emits one finite fake probability per image.  The frozen released
operating point is ``probability > 0.5``; a value equal to 0.5 is real.
SPAI has no native manipulation-localization output, so the mature
UniversalFakeDetect paired/bootstrap implementation supplies the shared T1
benchmark contract.
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


def spai_detection_metrics_strict(
    rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Calculate image-level SPAI metrics without dropping invalid rows."""

    return ufd_detection_metrics_strict(rows, threshold)


def summarize_spai_pair_slice(
    pairs: list[Any],
    threshold: float = FIXED_THRESHOLD,
    iterations: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize and pair-bootstrap a non-empty complete-pair slice."""

    return summarize_ufd_pair_slice(pairs, threshold, iterations, seed)


def summarize_spai_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    threshold: float = FIXED_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate retry history and summarize an SPAI run."""

    result = summarize_ufd_results(
        rows,
        expected_rows,
        threshold,
        bootstrap_samples,
        seed,
    )
    result["schema_version"] = "spai_detection_summary_v1"
    return result


__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EDIT_VISIBILITIES",
    "FIXED_THRESHOLD",
    "FPR_QUANTILE_METHOD",
    "TARGET_FPR",
    "THRESHOLD_OPERATOR",
    "spai_detection_metrics_strict",
    "summarize_spai_pair_slice",
    "summarize_spai_results",
]
