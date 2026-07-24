#!/usr/bin/env python3
"""Materialize Hysteresis-Sam3 splice masks from saved SAM3 results.

Hysteresis-Sam3 treats the selected SAM3 semantic mask as the trusted subject
prior, then grows source/generated residual support outward from that mask.
The support threshold rises with distance from the SAM3 boundary, limiting
background reconstruction leakage while recovering nearby fur and shadows.

This module is offline-only: it reads saved SAM3 masks and never calls an API.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from eval.segmentation.run_fal_sam3 import (
    REPO,
    binary_iou,
    load_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)


METHOD_NAME = "Hysteresis-Sam3"


@dataclass(frozen=True)
class HysteresisSam3Config:
    low_threshold: float = 12.0
    high_threshold: float = 40.0
    far_threshold: float = 28.0
    distance_power: float = 1.5
    reach_scale: float = 0.20
    min_reach_pixels: int = 12
    max_reach_pixels: int = 64
    auto_expand_scale: float = 0.35
    auto_expand_max_reach_pixels: int = 64
    auto_expand_max_growth_over_semantic: float = 0.70
    far_direction_start_ratio: float = 0.35
    far_shadow_channel_min: float = 5.0
    close_iterations: int = 2
    grow_iterations: int = 0
    min_component_pixels: int = 3
    semantic_feather: float = 0.5
    residual_feather: float = 1.0
    core_erosion: int = 2
    unbounded_growth: bool = False
    component_size_cap: bool = True

    def validate(self) -> None:
        if self.low_threshold < 0:
            raise ValueError("low threshold cannot be negative")
        if self.high_threshold <= self.low_threshold:
            raise ValueError("high threshold must exceed low threshold")
        if self.far_threshold < self.low_threshold:
            raise ValueError("far threshold must be at least low threshold")
        if self.distance_power <= 0 or self.reach_scale <= 0:
            raise ValueError("distance power and reach scale must be positive")
        if (
            self.min_reach_pixels < 1
            or self.max_reach_pixels < self.min_reach_pixels
            or self.auto_expand_max_reach_pixels < self.min_reach_pixels
        ):
            raise ValueError("invalid reach bounds")
        if self.auto_expand_scale < 0:
            raise ValueError("auto expand scale cannot be negative")
        if self.auto_expand_max_growth_over_semantic < 0:
            raise ValueError("auto expand growth limit cannot be negative")
        if not 0 <= self.far_direction_start_ratio <= 1:
            raise ValueError("far direction start ratio must be in [0, 1]")
        if self.far_shadow_channel_min < 0:
            raise ValueError("far shadow channel minimum cannot be negative")
        if (
            self.close_iterations < 0
            or self.grow_iterations < 0
            or self.min_component_pixels < 1
            or self.core_erosion < 0
        ):
            raise ValueError("invalid morphology setting")
        if self.semantic_feather < 0 or self.residual_feather < 0:
            raise ValueError("feather radius cannot be negative")


def mask_alpha(mask: np.ndarray, feather: float) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if feather > 0:
        image = image.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(image, dtype=np.uint8)


def _grow_residual_support(
    semantic: np.ndarray,
    difference: np.ndarray,
    shadow_channel_darkening: np.ndarray | None,
    config: HysteresisSam3Config,
    reach_pixels: int,
    *,
    semantic_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow one reach-bounded residual candidate and report its diagnostics."""
    structure = np.ones((3, 3), dtype=bool)
    attach_pixels = int(np.clip(round(reach_pixels * 0.20), 3, 8))

    # For pixels outside the semantic mask this is Euclidean distance to SAM3.
    distance = ndimage.distance_transform_edt(~semantic)
    distance_fraction = np.clip(distance / max(1, reach_pixels), 0.0, 1.0)
    support_threshold = config.low_threshold + (
        config.far_threshold - config.low_threshold
    ) * (distance_fraction**config.distance_power)
    reach = (
        ~semantic
        if config.unbounded_growth
        else (distance <= reach_pixels) & ~semantic
    )
    raw_support = (difference > support_threshold) & reach

    direction_start_distance = max(
        float(attach_pixels),
        reach_pixels * config.far_direction_start_ratio,
    )
    direction_filter_applied = (
        shadow_channel_darkening is not None
        and config.far_shadow_channel_min > 0
    )
    if direction_filter_applied:
        far_zone = distance > direction_start_distance
        far_shadow = (
            shadow_channel_darkening > config.far_shadow_channel_min
        )
        raw_support &= ~far_zone | far_shadow

    support = raw_support
    if config.close_iterations:
        support = ndimage.binary_closing(
            support,
            structure=structure,
            iterations=config.close_iterations,
            border_value=0,
        )
        support &= reach

    # SAM3 itself is the high-confidence anchor. Weak residual may seed only
    # in a narrow attachment band; farther seeds must pass the high threshold.
    attachment_seeds = support & (distance <= attach_pixels)
    strong_seeds = (
        support
        & (difference >= config.high_threshold)
        & (distance <= max(attach_pixels, reach_pixels / 2.0))
    )
    seeds = attachment_seeds | strong_seeds
    propagated = (
        ndimage.binary_propagation(seeds, structure=structure, mask=support)
        if seeds.any()
        else np.zeros_like(semantic)
    )

    labels, count = ndimage.label(propagated, structure=structure)
    kept = np.zeros_like(semantic)
    retained_components = 0
    removed_components = 0
    max_component_pixels = (
        max(
            semantic_pixels * 2,
            int(round(semantic.size * 0.15)),
        )
        if config.component_size_cap
        else None
    )
    for label_id in range(1, count + 1):
        component = labels == label_id
        pixels = int(component.sum())
        if (
            pixels >= config.min_component_pixels
            and (
                max_component_pixels is None
                or pixels <= max_component_pixels
            )
            and np.logical_and(component, seeds).any()
        ):
            kept |= component
            retained_components += 1
        else:
            removed_components += 1

    if kept.any():
        kept = ndimage.binary_fill_holes(kept)
    if config.grow_iterations:
        kept = ndimage.binary_dilation(
            kept,
            structure=structure,
            iterations=config.grow_iterations,
        )
    kept &= ~semantic
    postconnect_nonbright_filter_applied = (
        shadow_channel_darkening is not None
    )
    postconnect_bright_pixels_removed = 0
    if postconnect_nonbright_filter_applied:
        # Connectivity and morphology may bridge through pixels that were not
        # part of the direction-filtered raw support.  A residual is intended
        # to recover shadows, so never paste a generated residual pixel when
        # any RGB channel is brighter than the corresponding source pixel.
        brightening = shadow_channel_darkening < 0
        postconnect_bright_pixels_removed = int(
            np.logical_and(kept, brightening).sum()
        )
        kept &= ~brightening

    context_edge = np.zeros_like(semantic)
    context_edge[[0, -1], :] = True
    context_edge[:, [0, -1]] = True
    reach_boundary = (
        np.zeros_like(semantic)
        if config.unbounded_growth
        else reach & (distance >= max(0.0, reach_pixels - 1.0))
    )
    return kept, {
        "raw_support_pixels": int(raw_support.sum()),
        "closed_support_pixels": int(support.sum()),
        "seed_pixels": int(seeds.sum()),
        "attachment_seed_pixels": int(attachment_seeds.sum()),
        "strong_seed_pixels": int(strong_seeds.sum()),
        "residual_support_pixels": int(kept.sum()),
        "reach_pixels": reach_pixels,
        "attach_pixels": attach_pixels,
        "direction_start_distance": direction_start_distance,
        "shadow_direction_filter_applied": direction_filter_applied,
        "postconnect_nonbright_filter_applied": (
            postconnect_nonbright_filter_applied
        ),
        "postconnect_bright_pixels_removed": (
            postconnect_bright_pixels_removed
        ),
        "retained_components": retained_components,
        "removed_components": removed_components,
        "component_size_cap_pixels": max_component_pixels,
        "max_added_distance": (
            float(distance[kept].max()) if kept.any() else 0.0
        ),
        "touches_reach_boundary": bool(
            np.logical_and(kept, reach_boundary).any()
        ),
        "touches_context_edge": bool(
            np.logical_and(semantic | kept, context_edge).any()
        ),
    }


def hysteresis_sam3_mask(
    semantic: np.ndarray,
    difference: np.ndarray,
    config: HysteresisSam3Config,
    shadow_channel_darkening: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Grow an adaptive residual mask from a SAM3 semantic subject mask.

    Returns ``(combined_mask, residual_support, alpha, stats)``. The full
    semantic mask is always retained. Added support must be reachable from
    either the SAM3 attachment band or a high-threshold seed in the inner half
    of the adaptive reach.
    """
    config.validate()
    semantic = np.asarray(semantic, dtype=bool)
    difference = np.asarray(difference, dtype=np.float32)
    if semantic.ndim != 2 or difference.shape != semantic.shape:
        raise ValueError(
            f"semantic shape {semantic.shape} and difference shape "
            f"{difference.shape} must match in 2D"
        )
    if shadow_channel_darkening is not None:
        shadow_channel_darkening = np.asarray(
            shadow_channel_darkening,
            dtype=np.float32,
        )
        if shadow_channel_darkening.shape != semantic.shape:
            raise ValueError(
                "shadow channel darkening shape "
                f"{shadow_channel_darkening.shape} and semantic shape "
                f"{semantic.shape} must match"
            )
    semantic_pixels = int(semantic.sum())
    if semantic_pixels == 0:
        raise ValueError("SAM3 semantic mask is empty")

    core = semantic.copy()
    if config.core_erosion:
        eroded = ndimage.binary_erosion(
            semantic,
            structure=np.ones((3, 3), dtype=bool),
            iterations=config.core_erosion,
            border_value=0,
        )
        if eroded.any():
            core = eroded

    object_scale = math.sqrt(semantic_pixels)
    initial_reach_pixels = int(
        np.clip(
            round(object_scale * config.reach_scale),
            config.min_reach_pixels,
            config.max_reach_pixels,
        )
    )
    kept, growth_stats = _grow_residual_support(
        semantic,
        difference,
        shadow_channel_darkening,
        config,
        initial_reach_pixels,
        semantic_pixels=semantic_pixels,
    )
    auto_expand_attempted = False
    auto_expand_applied = False
    auto_expand_rejected_reason: str | None = None
    expanded_reach_pixels = initial_reach_pixels
    expanded_growth_over_semantic: float | None = None
    if (
        growth_stats["touches_reach_boundary"]
        and not config.unbounded_growth
        and config.auto_expand_scale > config.reach_scale
    ):
        expanded_reach_pixels = int(
            np.clip(
                round(object_scale * config.auto_expand_scale),
                config.min_reach_pixels,
                config.auto_expand_max_reach_pixels,
            )
        )
        if expanded_reach_pixels > initial_reach_pixels:
            auto_expand_attempted = True
            expanded, expanded_stats = _grow_residual_support(
                semantic,
                difference,
                shadow_channel_darkening,
                config,
                expanded_reach_pixels,
                semantic_pixels=semantic_pixels,
            )
            expanded_growth_over_semantic = float(
                expanded.sum() / max(1, semantic_pixels)
            )
            if (
                expanded_growth_over_semantic
                <= config.auto_expand_max_growth_over_semantic
            ):
                kept = expanded
                growth_stats = expanded_stats
                auto_expand_applied = True
            else:
                auto_expand_rejected_reason = (
                    "expanded_residual_growth_exceeds_limit"
                )

    combined = semantic | kept
    semantic_alpha = mask_alpha(semantic, config.semantic_feather)
    residual_alpha = mask_alpha(kept, config.residual_feather)
    residual_alpha_bright_pixels_zeroed = 0
    if shadow_channel_darkening is not None:
        brightening = shadow_channel_darkening < 0
        residual_alpha_bright_pixels_zeroed = int(
            np.logical_and(residual_alpha > 0, brightening).sum()
        )
        residual_alpha = residual_alpha.copy()
        residual_alpha[brightening] = 0
    alpha = np.maximum(semantic_alpha, residual_alpha)
    final_alpha_bright_pixels_zeroed = 0
    if shadow_channel_darkening is not None:
        # Preserve the trusted SAM3 subject itself, but do not let semantic
        # feathering reintroduce a brighter generated pixel just outside it.
        outside_semantic_brightening = (
            (shadow_channel_darkening < 0) & ~semantic
        )
        final_alpha_bright_pixels_zeroed = int(
            np.logical_and(alpha > 0, outside_semantic_brightening).sum()
        )
        alpha = alpha.copy()
        alpha[outside_semantic_brightening] = 0

    stats = {
        "method": METHOD_NAME,
        "semantic_pixels": semantic_pixels,
        "semantic_core_pixels": int(core.sum()),
        **growth_stats,
        "residual_support_pixels": int(kept.sum()),
        "combined_pixels": int(combined.sum()),
        "growth_over_semantic": float(
            combined.sum() / max(1, semantic_pixels) - 1.0
        ),
        "initial_reach_pixels": initial_reach_pixels,
        "expanded_reach_pixels": expanded_reach_pixels,
        "auto_expand_attempted": auto_expand_attempted,
        "auto_expand_applied": auto_expand_applied,
        "auto_expand_rejected_reason": auto_expand_rejected_reason,
        "expanded_growth_over_semantic": expanded_growth_over_semantic,
        "residual_alpha_bright_pixels_zeroed": (
            residual_alpha_bright_pixels_zeroed
        ),
        "final_alpha_bright_pixels_zeroed": (
            final_alpha_bright_pixels_zeroed
        ),
        "unbounded_growth": config.unbounded_growth,
        "component_size_cap": config.component_size_cap,
    }
    return combined, kept, alpha, stats


def keyed(
    rows: Sequence[dict[str, Any]],
    label: str,
    *,
    endpoint: str | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if endpoint is not None and row.get("endpoint_tag") != endpoint:
            continue
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label} row is missing task_id")
        if task_id in result:
            raise ValueError(f"{label} contains duplicate task_id {task_id!r}")
        result[task_id] = row
    return result


def relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def materialize(
    repo_root: Path,
    sam3_results_dir: Path,
    output_dir: Path,
    config: HysteresisSam3Config,
) -> list[dict[str, Any]]:
    config.validate()
    source_reviews = load_jsonl(sam3_results_dir / "review_manifest.jsonl")
    source_splices = keyed(
        load_jsonl(sam3_results_dir / "splice_results.jsonl"),
        "SAM3 splice results",
        endpoint="sam3",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for review in source_reviews:
        task_id = str(review["task_id"])
        source_splice = source_splices.get(task_id)
        if source_splice is None:
            raise ValueError(f"{task_id}: no saved SAM3 splice row")

        source_path = repo_root / str(review["source_image"])
        generated_path = repo_root / str(review["generated_crop"])
        semantic_path = repo_root / str(source_splice["semantic"]["path"])
        context_box = tuple(int(value) for value in review["context_region_xyxy"])
        with (
            Image.open(source_path) as source_image,
            Image.open(generated_path) as generated_image,
            Image.open(semantic_path) as semantic_image,
        ):
            source = source_image.convert("RGB")
            generated = generated_image.convert("RGB")
            original = source.crop(context_box)
            semantic = np.asarray(semantic_image.convert("L")) >= 128
        if original.size != generated.size:
            raise ValueError(
                f"{task_id}: source context {original.size} != generated "
                f"{generated.size}"
            )
        if semantic.shape != (generated.height, generated.width):
            raise ValueError(
                f"{task_id}: semantic {semantic.shape} != generated "
                f"{(generated.height, generated.width)}"
            )

        original_array = np.asarray(original, dtype=np.int16)
        generated_array = np.asarray(generated, dtype=np.int16)
        difference = np.abs(original_array - generated_array).max(axis=2)
        shadow_channel_darkening = (
            original_array - generated_array
        ).min(axis=2).astype(np.float32)
        combined, support, alpha, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            config,
            shadow_channel_darkening=shadow_channel_darkening,
        )
        alpha_image = Image.fromarray(alpha, mode="L")
        composite = Image.composite(generated, original, alpha_image)
        full = source.copy()
        full.paste(composite, (context_box[0], context_box[1]))

        mask_dir = output_dir / "masks"
        full_dir = output_dir / "spliced_full"
        mask_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)
        combined_path = mask_dir / f"{task_id}_hysteresis_sam3.png"
        support_path = mask_dir / f"{task_id}_residual_support.png"
        alpha_path = mask_dir / f"{task_id}_hysteresis_sam3_alpha.png"
        full_path = full_dir / f"{task_id}.png"
        Image.fromarray(combined.astype(np.uint8) * 255, mode="L").save(
            combined_path
        )
        Image.fromarray(support.astype(np.uint8) * 255, mode="L").save(
            support_path
        )
        alpha_image.save(alpha_path)
        full.save(full_path)

        outside = np.ones((source.height, source.width), dtype=bool)
        x1, y1, x2, y2 = context_box
        outside[y1:y2, x1:x2] = False
        outside_identical = bool(
            np.array_equal(np.asarray(full)[outside], np.asarray(source)[outside])
        )
        row = {
            "schema_version": "hysteresis_sam3_splice_result_v2",
            "task_id": task_id,
            "method": METHOD_NAME,
            "status": "ok",
            "source_image": review["source_image"],
            "generated_crop": review["generated_crop"],
            "sam3_semantic_mask": source_splice["semantic"]["path"],
            "sam3_request_id": source_splice.get("request_id"),
            "context_region_xyxy": list(context_box),
            "edit_region_xyxy": review["edit_region_xyxy"],
            "mask": relative(combined_path, repo_root),
            "residual_support_mask": relative(support_path, repo_root),
            "alpha_mask": relative(alpha_path, repo_root),
            "spliced_full": relative(full_path, repo_root),
            "outside_context_identical_to_source": outside_identical,
            "sam3_iou": binary_iou(combined, semantic),
            "stats": stats,
            "config": asdict(config),
        }
        results.append(row)
        review_rows.append(
            {
                **review,
                "spliced_full": row["spliced_full"],
                "paste_mode": "hysteresis_sam3_v2",
                "hysteresis_sam3": {
                    "method": METHOD_NAME,
                    "mask": row["mask"],
                    "alpha_mask": row["alpha_mask"],
                    "stats": stats,
                },
                "status": "ok",
            }
        )

    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "review_manifest.jsonl", review_rows)
    growth = [float(row["stats"]["growth_over_semantic"]) for row in results]
    summary = {
        "schema_version": "hysteresis_sam3_summary_v2",
        "method": METHOD_NAME,
        "created_at": utc_now(),
        "tasks": len(results),
        "config": asdict(config),
        "source_sam3_results_dir": relative(sam3_results_dir, repo_root),
        "outside_context_failures": [
            row["task_id"]
            for row in results
            if not row["outside_context_identical_to_source"]
        ],
        "context_edge_touches": [
            row["task_id"]
            for row in results
            if row["stats"]["touches_context_edge"]
        ],
        "growth_over_semantic": {
            "mean": float(np.mean(growth)) if growth else None,
            "median": float(np.median(growth)) if growth else None,
            "max": max(growth) if growth else None,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return review_rows


def load_review_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records must be an array")
    labels: dict[str, str] = {}
    for record in records:
        task_id = str(record.get("task_id", ""))
        status = str(record.get("status", ""))
        if not task_id or not status:
            raise ValueError(f"{path}: invalid review record")
        if task_id in labels:
            raise ValueError(f"{path}: duplicate task_id {task_id!r}")
        labels[task_id] = status
    return labels


def export_both_bad_review(
    review_rows: Sequence[dict[str, Any]],
    hysteresis_labels_path: Path,
    sam3_labels_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    hysteresis_labels = load_review_labels(hysteresis_labels_path)
    sam3_labels = load_review_labels(sam3_labels_path)
    both_bad_ids = {
        task_id
        for task_id, status in hysteresis_labels.items()
        if status == "bad" and sam3_labels.get(task_id) == "bad"
    }
    selected = [
        row for row in review_rows if str(row["task_id"]) in both_bad_ids
    ]
    missing = both_bad_ids - {str(row["task_id"]) for row in selected}
    if missing:
        raise ValueError(
            "both-bad labels missing from generated review manifest: "
            + ", ".join(sorted(missing))
        )
    write_jsonl(output_path, selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--sam3-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hysteresis-labels", type=Path)
    parser.add_argument("--sam3-labels", type=Path)
    parser.add_argument(
        "--both-bad-review-output",
        type=Path,
        help="optional review manifest restricted to bad/bad label intersection",
    )
    parser.add_argument("--low-threshold", type=float, default=12.0)
    parser.add_argument("--high-threshold", type=float, default=40.0)
    parser.add_argument("--far-threshold", type=float, default=28.0)
    parser.add_argument("--distance-power", type=float, default=1.5)
    parser.add_argument("--reach-scale", type=float, default=0.20)
    parser.add_argument("--min-reach-pixels", type=int, default=12)
    parser.add_argument("--max-reach-pixels", type=int, default=64)
    parser.add_argument("--auto-expand-scale", type=float, default=0.35)
    parser.add_argument("--auto-expand-max-reach-pixels", type=int, default=64)
    parser.add_argument(
        "--auto-expand-max-growth-over-semantic",
        type=float,
        default=0.70,
    )
    parser.add_argument("--far-direction-start-ratio", type=float, default=0.35)
    parser.add_argument("--far-shadow-channel-min", type=float, default=5.0)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--grow-iterations", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument("--semantic-feather", type=float, default=0.5)
    parser.add_argument("--residual-feather", type=float, default=1.0)
    parser.add_argument("--core-erosion", type=int, default=2)
    parser.add_argument(
        "--unbounded-growth",
        action="store_true",
        help=(
            "remove the hard SAM-distance reach boundary; connected residual "
            "may grow until it naturally terminates"
        ),
    )
    parser.add_argument(
        "--no-component-size-cap",
        action="store_true",
        help="do not reject a seeded residual component because of its area",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    def resolve(path: Path | None) -> Path | None:
        if path is None:
            return None
        return (path if path.is_absolute() else repo_root / path).resolve()

    sam3_results_dir = resolve(args.sam3_results_dir)
    output_dir = resolve(args.output_dir)
    if sam3_results_dir is None or output_dir is None:
        raise AssertionError("required paths were not resolved")
    config = HysteresisSam3Config(
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        far_threshold=args.far_threshold,
        distance_power=args.distance_power,
        reach_scale=args.reach_scale,
        min_reach_pixels=args.min_reach_pixels,
        max_reach_pixels=args.max_reach_pixels,
        auto_expand_scale=args.auto_expand_scale,
        auto_expand_max_reach_pixels=args.auto_expand_max_reach_pixels,
        auto_expand_max_growth_over_semantic=(
            args.auto_expand_max_growth_over_semantic
        ),
        far_direction_start_ratio=args.far_direction_start_ratio,
        far_shadow_channel_min=args.far_shadow_channel_min,
        close_iterations=args.close_iterations,
        grow_iterations=args.grow_iterations,
        min_component_pixels=args.min_component_pixels,
        semantic_feather=args.semantic_feather,
        residual_feather=args.residual_feather,
        core_erosion=args.core_erosion,
        unbounded_growth=args.unbounded_growth,
        component_size_cap=not args.no_component_size_cap,
    )
    review_rows = materialize(
        repo_root,
        sam3_results_dir,
        output_dir,
        config,
    )

    both_bad_rows: list[dict[str, Any]] = []
    if args.both_bad_review_output is not None:
        hysteresis_labels = resolve(args.hysteresis_labels)
        sam3_labels = resolve(args.sam3_labels)
        review_output = resolve(args.both_bad_review_output)
        if hysteresis_labels is None or sam3_labels is None or review_output is None:
            parser.error(
                "--both-bad-review-output requires --hysteresis-labels and "
                "--sam3-labels"
            )
        both_bad_rows = export_both_bad_review(
            review_rows,
            hysteresis_labels,
            sam3_labels,
            review_output,
        )
    print(
        json.dumps(
            {
                "method": METHOD_NAME,
                "output_dir": relative(output_dir, repo_root),
                "generated": len(review_rows),
                "both_bad_review": len(both_bad_rows),
                "status_counts": dict(
                    Counter(row.get("status") for row in review_rows)
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
