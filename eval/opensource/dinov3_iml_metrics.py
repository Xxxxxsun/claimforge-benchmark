"""Strict T2-only metrics for the official DINOv3-IML CAT checkpoint.

DINOv3-IML exposes a dense manipulation-probability map but no image-level
classification head.  The numerical aggregation contract is deliberately
shared with the already audited 512/native Mesorch localization protocol:
float32 probabilities, a strict ``p > 0.5`` mask, forged-only pixel AP,
authentic false-positive area, macro/micro summaries, and pair bootstrap.

Keeping the common numerical implementation in one place prevents the two
512/native adapters from silently drifting.  The DINOv3-IML runner records
both this file and ``mesorch_metrics.py`` in its immutable adapter contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.opensource.mesorch_metrics import (
    BOOTSTRAP_METRICS as _BOOTSTRAP_METRICS,
    FIXED_MASK_THRESHOLD as _FIXED_MASK_THRESHOLD,
    LOCALIZATION_SPACES as _LOCALIZATION_SPACES,
    THRESHOLD_OPERATOR as _THRESHOLD_OPERATOR,
    binary_pixel_metrics_strict as _binary_pixel_metrics_strict,
    summarize_mesorch_pair_slice as _summarize_pair_slice,
    summarize_mesorch_results as _summarize_results,
)


FIXED_MASK_THRESHOLD = _FIXED_MASK_THRESHOLD
THRESHOLD_OPERATOR = _THRESHOLD_OPERATOR
LOCALIZATION_SPACES = _LOCALIZATION_SPACES
BOOTSTRAP_METRICS = _BOOTSTRAP_METRICS

if LOCALIZATION_SPACES != ("model_512", "native"):
    raise RuntimeError("shared localization metric spaces changed unexpectedly")
if FIXED_MASK_THRESHOLD != 0.5 or THRESHOLD_OPERATOR != ">":
    raise RuntimeError("shared strict localization threshold contract changed")


def _rebrand_error(error: ValueError) -> ValueError:
    return ValueError(str(error).replace("Mesorch", "DINOv3-IML"))


def binary_pixel_metrics_strict(
    probability_map: np.ndarray,
    target: np.ndarray,
    threshold: float = FIXED_MASK_THRESHOLD,
    *,
    include_ap: bool = True,
) -> dict[str, Any]:
    """Evaluate persisted float32 probability with strict ``p > 0.5``.

    Pixel AP is meaningful only for a forged target containing positive
    pixels.  Callers pass ``include_ap=False`` for authentic images; the
    shared implementation also returns ``None`` for an all-negative target.
    """

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


def summarize_dinov3_iml_pair_slice(
    pairs: list[Any],
    *,
    iterations: int,
    seed: int,
    localization_space: str = "native",
) -> dict[str, Any]:
    """Pair-bootstrap DINOv3-IML localization without deriving a T1 score."""

    try:
        return _summarize_pair_slice(
            pairs,
            iterations=iterations,
            seed=seed,
            localization_space=localization_space,
        )
    except ValueError as exc:
        raise _rebrand_error(exc) from exc


def summarize_dinov3_iml_results(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None = None,
    *,
    mask_threshold: float = FIXED_MASK_THRESHOLD,
    bootstrap_samples: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Build the strict model-512/native DINOv3-IML T2 summary."""

    try:
        summary = _summarize_results(
            rows,
            expected_rows,
            mask_threshold=mask_threshold,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except ValueError as exc:
        raise _rebrand_error(exc) from exc
    scope = summary.get("task_scope")
    if not isinstance(scope, dict):
        raise RuntimeError("shared localization summary lost task_scope")
    scope["localization_semantics"] = (
        "dinov3_iml_sigmoid_manipulation_probability_float32"
    )
    scope["model_space_probability_source"] = (
        "sigmoid_bilinear_align_corners_false_seg_head_logits_32_to_512"
    )
    scope["native_probability_source"] = (
        "bilinear_align_corners_false_resize_of_model_512_probability"
    )
    return summary


__all__ = [
    "BOOTSTRAP_METRICS",
    "FIXED_MASK_THRESHOLD",
    "LOCALIZATION_SPACES",
    "THRESHOLD_OPERATOR",
    "binary_pixel_metrics",
    "binary_pixel_metrics_strict",
    "summarize_dinov3_iml_pair_slice",
    "summarize_dinov3_iml_results",
]
