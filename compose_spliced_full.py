#!/usr/bin/env python3
"""Compose generated context crops back into full source images.

Input:
  annotations/generation_tasks.jsonl
  generated_crops/<model>/manifest.jsonl

Output:
  spliced_full/<model>/<task_id>.png
  spliced_full/<model>/manifest.jsonl

The output PNG keeps pixels outside context_region_xyxy identical to the source
image. If generated_crops/<model>/manifest.jsonl marks `paste_back: false`, the
script first composites only the insert region from the generated crop into the
original context crop.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

REPO = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def feathered_mask(size, box, feather):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle([int(v) for v in box], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def padded_box(box, pad, size):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = size
    pad = int(max(0, pad))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ]


def box_mask(shape, box):
    h, w = shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mask = np.zeros(shape, bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def object_mask(
    original_crop,
    generated_crop,
    box,
    thr,
    feather,
    object_pad=20,
    min_px=6,
    search_mode="padded",
):
    """Paste only the pixels the model actually *added* (the object), not the
    whole orange rectangle.

    The model re-decodes the whole crop, so the crop can pick up a faint
    low-frequency brightness/color shift on flat background (walls, table). We
    threshold the difference, retain the largest component anchored in the
    orange box, then dilate for contact shadow and feather the boundary. The
    legacy mode searches an orange-box neighborhood; context mode searches the
    complete blue crop so the orange box cannot clip the generated object.
    """
    o = np.asarray(original_crop, np.int16)
    g = np.asarray(generated_crop, np.int16)
    d = np.abs(o - g).max(2)

    if search_mode == "context":
        # Search the complete blue context crop. The orange edit box remains an
        # anchor for selecting the intended connected component, not a hard
        # clipping boundary for the generated object.
        inbox = np.ones_like(d, dtype=bool)
    else:
        search_box = padded_box(box, object_pad, original_crop.size)
        inbox = box_mask(d.shape, search_box)
    original_box = box_mask(d.shape, box)

    m = (d > thr) & inbox
    m = ndimage.binary_opening(m, iterations=1)
    lbl, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        overlaps = ndimage.sum(original_box, lbl, range(1, n + 1))
        candidates = [i for i, overlap in enumerate(overlaps) if overlap > 0]
        if candidates:
            best = max(candidates, key=lambda i: sizes[i])
            m = lbl == (best + 1)
            if search_mode == "context":
                # Restore low-difference holes inside fur, limbs, and other
                # thin object structure before feathering the boundary.
                m = ndimage.binary_closing(m, iterations=2)
                m = ndimage.binary_fill_holes(m)
            m = ndimage.binary_dilation(m, iterations=2) & inbox
        else:
            m = np.zeros_like(m)

    if m.sum() < min_px:
        # subtle edit: fall back to the feathered orange box
        return feathered_mask(original_crop.size, box, feather)

    mask = Image.fromarray((m * 255).astype(np.uint8))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def hysteresis_object_mask(
    reference_crop,
    generated_crop,
    box,
    low_thr,
    high_thr,
    feather,
    close_iterations=3,
    grow_iterations=2,
    min_component_px=6,
    reach_ratio=0.5,
    far_thr=None,
    distance_power=1.0,
    auto_expand_ratio=None,
    auto_expand_max_growth=0.05,
):
    """Grow a complete object mask from high-confidence edit-box seeds.

    A single hard threshold tends to split low-contrast fur, limbs, and shadows
    into separate components. This mode uses high-threshold pixels inside the
    orange box as seeds, closes the lower-threshold support mask *before*
    connected-component selection, and propagates all seeded components with
    8-neighbour connectivity. The reference is the exact context crop supplied
    to generation, rather than a fresh crop decoded from the source image.
    """
    if high_thr <= low_thr:
        raise ValueError(
            f"hysteresis high threshold must exceed low threshold: "
            f"{high_thr} <= {low_thr}"
        )
    if far_thr is not None and far_thr < low_thr:
        raise ValueError(
            f"hysteresis far threshold must be at least the low threshold: "
            f"{far_thr} < {low_thr}"
        )
    if distance_power <= 0:
        raise ValueError(
            f"hysteresis distance power must be positive: {distance_power}"
        )
    if auto_expand_ratio is not None and auto_expand_ratio < reach_ratio:
        raise ValueError(
            f"hysteresis auto-expand ratio must be at least the initial ratio: "
            f"{auto_expand_ratio} < {reach_ratio}"
        )
    if auto_expand_max_growth < 0:
        raise ValueError(
            "hysteresis auto-expand maximum growth cannot be negative: "
            f"{auto_expand_max_growth}"
        )
    if reference_crop.size != generated_crop.size:
        raise ValueError(
            f"hysteresis reference size {reference_crop.size} != "
            f"generated size {generated_crop.size}"
        )

    ref = np.asarray(reference_crop, np.int16)
    generated = np.asarray(generated_crop, np.int16)
    diff = np.abs(ref - generated).max(2)
    anchor = box_mask(diff.shape, box)
    structure = np.ones((3, 3), dtype=bool)

    reach_pad = 0
    reach = np.ones_like(diff, dtype=bool)
    if reach_ratio > 0:
        x1, y1, x2, y2 = [int(value) for value in box]
        reach_pad = max(1, int(round(max(x2 - x1, y2 - y1) * reach_ratio)))
        reach = box_mask(
            diff.shape,
            padded_box(box, reach_pad, reference_crop.size),
        )

    support_threshold = np.full(diff.shape, low_thr, dtype=np.float32)
    if far_thr is not None and far_thr > low_thr:
        # Keep the permissive threshold on and immediately around the orange
        # box, where limbs may extend beyond the annotation. Increase it
        # smoothly with distance so global model reconstruction noise cannot
        # form a low-threshold bridge across most of the context crop.
        distance = ndimage.distance_transform_edt(~anchor)
        distance_scale = reach_pad if reach_pad > 0 else max(diff.shape)
        distance_fraction = np.clip(distance / distance_scale, 0.0, 1.0)
        support_threshold += (far_thr - low_thr) * (
            distance_fraction ** distance_power
        )

    raw_support = (diff > support_threshold) & reach
    support = raw_support
    if close_iterations > 0:
        support = ndimage.binary_closing(
            support,
            structure=structure,
            iterations=close_iterations,
        ) & reach

    seeds = (diff > high_thr) & anchor
    used_low_seed_fallback = False
    if not seeds.any():
        # A subtle edit can have no high-confidence pixel. Fall back to low
        # threshold seeds while retaining the same region-growing constraints.
        seeds = support & anchor
        used_low_seed_fallback = True

    if not seeds.any():
        mask = feathered_mask(reference_crop.size, box, feather)
        return mask, {
            "seed_pixels": 0,
            "raw_support_pixels": int(raw_support.sum()),
            "support_pixels": int(support.sum()),
            "mask_pixels": 0,
            "used_low_seed_fallback": True,
            "used_box_fallback": True,
            "reach_pad": reach_pad,
            "reach_pixels": int(reach.sum()),
            "requested_reach_ratio": reach_ratio,
            "effective_reach_ratio": reach_ratio,
            "touches_reach_boundary": False,
            "touches_context_edge": False,
            "needs_regeneration": False,
            "auto_expand_attempted": False,
            "auto_expand_applied": False,
            "auto_expand_added_pixels": 0,
            "auto_expand_added_fraction": 0.0,
            "auto_expand_rejected_reason": None,
        }

    propagated = ndimage.binary_propagation(
        seeds,
        structure=structure,
        mask=support,
    )

    # Keep every sufficiently large propagated component. Unlike the legacy
    # mode, this does not throw away a disconnected limb or tail merely because
    # another part of the object is larger.
    labels, count = ndimage.label(propagated, structure=structure)
    kept = np.zeros_like(propagated)
    if count > 0:
        component_ids = np.arange(1, count + 1)
        sizes = ndimage.sum(np.ones_like(labels), labels, component_ids)
        seed_overlaps = ndimage.sum(seeds, labels, component_ids)
        for component_id, size, seed_overlap in zip(
            component_ids, sizes, seed_overlaps
        ):
            if seed_overlap > 0 and size >= min_component_px:
                kept |= labels == component_id

    if kept.sum() < min_component_px:
        mask = feathered_mask(reference_crop.size, box, feather)
        return mask, {
            "seed_pixels": int(seeds.sum()),
            "raw_support_pixels": int(raw_support.sum()),
            "support_pixels": int(support.sum()),
            "mask_pixels": int(kept.sum()),
            "used_low_seed_fallback": used_low_seed_fallback,
            "used_box_fallback": True,
            "reach_pad": reach_pad,
            "reach_pixels": int(reach.sum()),
            "requested_reach_ratio": reach_ratio,
            "effective_reach_ratio": reach_ratio,
            "touches_reach_boundary": False,
            "touches_context_edge": False,
            "needs_regeneration": False,
            "auto_expand_attempted": False,
            "auto_expand_applied": False,
            "auto_expand_added_pixels": 0,
            "auto_expand_added_fraction": 0.0,
            "auto_expand_rejected_reason": None,
        }

    kept = ndimage.binary_fill_holes(kept)
    if grow_iterations > 0:
        kept = ndimage.binary_dilation(
            kept,
            structure=structure,
            iterations=grow_iterations,
        )

    mask = Image.fromarray((kept * 255).astype(np.uint8))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    # Distinguish an artificial search boundary from the real context edge.
    # Touching the former means the object may continue beyond the initial
    # reach window; touching the latter with high-confidence changed pixels
    # means the generated subject itself may have been clipped by the crop.
    context_edge = np.zeros_like(kept)
    context_edge[[0, -1], :] = True
    context_edge[:, [0, -1]] = True
    context_edge_band = ndimage.binary_dilation(
        context_edge,
        structure=structure,
        iterations=1,
    )
    reach_boundary = reach & ~ndimage.binary_erosion(
        reach,
        structure=structure,
        border_value=0,
    )
    artificial_reach_boundary = reach_boundary & ~context_edge
    touches_reach_boundary = bool((kept & artificial_reach_boundary).any())
    touches_context_edge = bool(
        (kept & context_edge_band & (diff > high_thr)).any()
    )

    stats = {
        "seed_pixels": int(seeds.sum()),
        "raw_support_pixels": int(raw_support.sum()),
        "support_pixels": int(support.sum()),
        "mask_pixels": int(kept.sum()),
        "used_low_seed_fallback": used_low_seed_fallback,
        "used_box_fallback": False,
        "reach_pad": reach_pad,
        "reach_pixels": int(reach.sum()),
        "requested_reach_ratio": reach_ratio,
        "effective_reach_ratio": reach_ratio,
        "touches_reach_boundary": touches_reach_boundary,
        "touches_context_edge": touches_context_edge,
        "needs_regeneration": touches_context_edge,
        "auto_expand_attempted": False,
        "auto_expand_applied": False,
        "auto_expand_added_pixels": 0,
        "auto_expand_added_fraction": 0.0,
        "auto_expand_rejected_reason": None,
    }

    should_try_expand = (
        auto_expand_ratio is not None
        and auto_expand_ratio > reach_ratio
        and touches_reach_boundary
        and not touches_context_edge
    )
    if should_try_expand:
        expanded_mask, expanded_stats = hysteresis_object_mask(
            reference_crop,
            generated_crop,
            box,
            low_thr,
            high_thr,
            feather,
            close_iterations=close_iterations,
            grow_iterations=grow_iterations,
            min_component_px=min_component_px,
            reach_ratio=auto_expand_ratio,
            far_thr=far_thr,
            distance_power=distance_power,
            auto_expand_ratio=None,
            auto_expand_max_growth=auto_expand_max_growth,
        )
        added_pixels = max(
            0,
            expanded_stats["mask_pixels"] - stats["mask_pixels"],
        )
        added_fraction = added_pixels / diff.size
        expanded_touches_edge = expanded_stats.get("touches_context_edge", False)
        accept_expansion = (
            not expanded_stats.get("used_box_fallback", False)
            and not expanded_touches_edge
            and added_fraction <= auto_expand_max_growth
        )
        if accept_expansion:
            expanded_stats.update({
                "requested_reach_ratio": reach_ratio,
                "effective_reach_ratio": auto_expand_ratio,
                "base_mask_pixels": stats["mask_pixels"],
                "auto_expand_attempted": True,
                "auto_expand_applied": True,
                "auto_expand_added_pixels": added_pixels,
                "auto_expand_added_fraction": added_fraction,
                "auto_expand_rejected_reason": None,
            })
            return expanded_mask, expanded_stats

        if expanded_stats.get("used_box_fallback", False):
            reason = "expanded_mask_fell_back_to_box"
        elif expanded_touches_edge:
            reason = "expanded_mask_touches_context_edge"
            stats["needs_regeneration"] = True
        else:
            reason = "expanded_mask_growth_exceeds_limit"
        stats.update({
            "auto_expand_attempted": True,
            "auto_expand_added_pixels": added_pixels,
            "auto_expand_added_fraction": added_fraction,
            "auto_expand_rejected_reason": reason,
        })

    return mask, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="hunyuan_image3")
    ap.add_argument("--tasks", default="annotations/generation_tasks.jsonl")
    ap.add_argument("--generated-manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--blend",
                    choices=["object", "hysteresis", "hysteresis-distance", "box"],
                    default="object",
                    help="for paste_back:false crops -- 'object' uses the legacy "
                         "single-threshold component mask; 'hysteresis' grows "
                         "low-threshold support from high-confidence edit-box "
                         "seeds; 'hysteresis-distance' also raises the support "
                         "threshold away from the edit box; 'box' pastes the "
                         "feathered orange rectangle")
    ap.add_argument("--object-thr", type=float, default=30.0,
                    help="per-pixel max-channel diff threshold for object detection")
    ap.add_argument("--object-pad", type=int, default=20,
                    help="pixels to expand the edit box when detecting the object mask")
    ap.add_argument("--object-search", choices=["padded", "context"], default="padded",
                    help="search for changed pixels inside the padded orange box "
                         "or across the complete blue context crop")
    ap.add_argument("--hysteresis-low", type=float, default=20.0,
                    help="low difference threshold used as region-growing support")
    ap.add_argument("--hysteresis-high", type=float, default=40.0,
                    help="high difference threshold used for edit-box seeds")
    ap.add_argument("--hysteresis-close", type=int, default=3,
                    help="8-neighbour closing iterations before region growing")
    ap.add_argument("--hysteresis-grow", type=int, default=2,
                    help="8-neighbour dilation iterations after region growing")
    ap.add_argument("--hysteresis-min-component", type=int, default=6,
                    help="minimum propagated component size in pixels")
    ap.add_argument("--hysteresis-reach-ratio", type=float, default=0.5,
                    help="maximum propagation padding as a fraction of the "
                         "larger edit-box side; 0 searches the full context")
    ap.add_argument("--hysteresis-far-thr", type=float, default=40.0,
                    help="support threshold at the outer reach boundary in "
                         "hysteresis-distance mode")
    ap.add_argument("--hysteresis-distance-power", type=float, default=1.0,
                    help="shape of the low-to-far threshold ramp; 1 is linear")
    ap.add_argument("--hysteresis-auto-expand-ratio", type=float, default=0.75,
                    help="retry a boundary-touching distance mask with this "
                         "reach ratio; 0 disables automatic expansion")
    ap.add_argument("--hysteresis-auto-expand-max-growth", type=float, default=0.05,
                    help="maximum fraction of context pixels that automatic "
                         "reach expansion may add")
    ap.add_argument("--only", default=None,
                    help="comma-separated task_ids or 0-based generated-row indices")
    ap.add_argument("--update-existing-manifest", action="store_true",
                    help="with --only, replace matching rows in an existing "
                         "output manifest instead of truncating it")
    args = ap.parse_args()

    if args.update_existing_manifest and not args.only:
        ap.error("--update-existing-manifest requires --only")

    model = args.model_name
    gen_manifest = REPO / Path(args.generated_manifest or f"generated_crops/{model}/manifest.jsonl")
    out_dir = REPO / Path(args.out_dir or f"spliced_full/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.jsonl"
    existing_manifest_rows = []
    if args.update_existing_manifest:
        if not man_path.is_file():
            ap.error(
                f"--update-existing-manifest requires an existing manifest: "
                f"{man_path}"
            )
        existing_manifest_rows = load_jsonl(man_path)

    tasks = {row["task_id"]: row for row in load_jsonl(REPO / args.tasks)}
    generated = [row for row in load_jsonl(REPO / gen_manifest) if row.get("status") == "ok"]
    if args.only:
        selected = set(args.only.split(","))
        generated = [
            row for index, row in enumerate(generated)
            if row["task_id"] in selected or str(index) in selected
        ]
        if not generated:
            ap.error("--only did not match any successful generated rows")

    manifest_rows = []
    for row in generated:
        task_id = row["task_id"]
        task = tasks[task_id]
        source_path = REPO / task["source_image"]
        crop_path = REPO / row["output_crop"]

        source = Image.open(source_path).convert("RGB")
        generated_crop = Image.open(crop_path).convert("RGB")
        x1, y1, x2, y2 = [int(v) for v in task["context_region_xyxy"]]
        expected = (x2 - x1, y2 - y1)
        if generated_crop.size != expected:
            raise ValueError(f"{task_id}: crop size {generated_crop.size} != context size {expected}")

        if row.get("paste_back") is False:
            original_crop = source.crop((x1, y1, x2, y2))
            box = task["edit_region_in_context_xyxy"]
            mask_reference = None
            hysteresis_stats = None
            if args.blend == "object":
                mask = object_mask(original_crop, generated_crop, box,
                                   args.object_thr, args.feather,
                                   object_pad=args.object_pad,
                                   search_mode=args.object_search)
                paste_mode = ("object_only_context_diff"
                              if args.object_search == "context"
                              else "object_only")
            elif args.blend in {"hysteresis", "hysteresis-distance"}:
                mask_reference = (row.get("input_context_crop")
                                  or task.get("context_crop"))
                if not mask_reference:
                    raise ValueError(
                        f"{task_id}: no input context crop for hysteresis mask"
                    )
                reference_crop = Image.open(REPO / mask_reference).convert("RGB")
                mask, hysteresis_stats = hysteresis_object_mask(
                    reference_crop,
                    generated_crop,
                    box,
                    args.hysteresis_low,
                    args.hysteresis_high,
                    args.feather,
                    close_iterations=args.hysteresis_close,
                    grow_iterations=args.hysteresis_grow,
                    min_component_px=args.hysteresis_min_component,
                    reach_ratio=args.hysteresis_reach_ratio,
                    far_thr=(args.hysteresis_far_thr
                             if args.blend == "hysteresis-distance" else None),
                    distance_power=args.hysteresis_distance_power,
                    auto_expand_ratio=(
                        args.hysteresis_auto_expand_ratio
                        if args.blend == "hysteresis-distance"
                        and args.hysteresis_auto_expand_ratio > 0
                        else None
                    ),
                    auto_expand_max_growth=(
                        args.hysteresis_auto_expand_max_growth
                    ),
                )
                paste_mode = ("object_hysteresis_distance"
                              if args.blend == "hysteresis-distance"
                              else "object_hysteresis")
            else:
                mask = feathered_mask(expected, box, args.feather)
                paste_mode = "masked_insert_region"
            crop_to_paste = Image.composite(generated_crop, original_crop, mask)
        else:
            crop_to_paste = generated_crop
            paste_mode = "full_context_crop"
            mask_reference = None
            hysteresis_stats = None

        spliced = source.copy()
        spliced.paste(crop_to_paste, (x1, y1))
        out_path = out_dir / f"{task_id}.png"
        spliced.save(out_path)

        manifest_rows.append({
            "task_id": task_id,
            "source_image": task["source_image"],
            "generated_crop": row["output_crop"],
            "spliced_full": str(out_path.relative_to(REPO)),
            "model": model,
            "image_size": list(source.size),
            "context_region_xyxy": task["context_region_xyxy"],
            "edit_region_xyxy": task["edit_region_xyxy"],
            "edit_region_in_context_xyxy": task["edit_region_in_context_xyxy"],
            "candidates": task["candidates"],
            "paste_mode": paste_mode,
            "object_pad": (args.object_pad
                           if args.blend == "object" and args.object_search == "padded"
                           else 0),
            "object_search": args.object_search if args.blend == "object" else None,
            "object_threshold": args.object_thr if args.blend == "object" else None,
            "mask_reference": mask_reference,
            "hysteresis_low_threshold": (args.hysteresis_low
                                         if args.blend.startswith("hysteresis")
                                         else None),
            "hysteresis_high_threshold": (args.hysteresis_high
                                          if args.blend.startswith("hysteresis")
                                          else None),
            "hysteresis_close_iterations": (args.hysteresis_close
                                             if args.blend.startswith("hysteresis")
                                             else None),
            "hysteresis_grow_iterations": (args.hysteresis_grow
                                            if args.blend.startswith("hysteresis")
                                            else None),
            "hysteresis_reach_ratio": (args.hysteresis_reach_ratio
                                       if args.blend.startswith("hysteresis")
                                       else None),
            "hysteresis_far_threshold": (args.hysteresis_far_thr
                                         if args.blend == "hysteresis-distance"
                                         else None),
            "hysteresis_distance_power": (args.hysteresis_distance_power
                                           if args.blend == "hysteresis-distance"
                                           else None),
            "hysteresis_auto_expand_ratio": (
                args.hysteresis_auto_expand_ratio
                if args.blend == "hysteresis-distance" else None
            ),
            "hysteresis_auto_expand_max_growth": (
                args.hysteresis_auto_expand_max_growth
                if args.blend == "hysteresis-distance" else None
            ),
            "hysteresis_stats": hysteresis_stats,
            "source_generated_paste_back": row.get("paste_back"),
            "status": "ok",
        })
        print(f"{task_id}: {out_path}")

    processed_count = len(manifest_rows)
    if args.update_existing_manifest:
        updates = {row["task_id"]: row for row in manifest_rows}
        merged_rows = []
        for old_row in existing_manifest_rows:
            merged_rows.append(updates.pop(old_row["task_id"], old_row))
        merged_rows.extend(updates.values())
        manifest_rows = merged_rows
    man_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_rows), encoding="utf-8")
    print(
        f"done: processed {processed_count}; manifest has {len(manifest_rows)} "
        f"full spliced images -> {out_dir}"
    )


if __name__ == "__main__":
    main()
