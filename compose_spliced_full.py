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


def object_mask(original_crop, generated_crop, box, thr, feather, min_px=6):
    """Paste only the pixels the model actually *added* (the object), not the
    whole orange rectangle.

    The model re-decodes the whole crop, so the orange box also picks up a faint
    low-frequency brightness/color shift on flat background (walls, table) which
    shows up as a visible bright rectangle when pasted. We isolate the object by:
      diff > thr  ->  morphological opening (drop thin/area background drift)
      ->  keep the largest connected blob (the object)  ->  dilate (contact
      shadow)  ->  feather. Falls back to the box mask if nothing strong is found
      (very subtle edits).
    """
    o = np.asarray(original_crop, np.int16)
    g = np.asarray(generated_crop, np.int16)
    d = np.abs(o - g).max(2)

    x1, y1, x2, y2 = [int(v) for v in box]
    inbox = np.zeros(d.shape, bool)
    inbox[y1:y2, x1:x2] = True

    m = (d > thr) & inbox
    m = ndimage.binary_opening(m, iterations=1)
    lbl, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
        m = ndimage.binary_dilation(m, iterations=2)

    if m.sum() < min_px:
        # subtle edit: fall back to the feathered orange box
        return feathered_mask(original_crop.size, box, feather)

    mask = Image.fromarray((m * 255).astype(np.uint8))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="hunyuan_image3")
    ap.add_argument("--tasks", default="annotations/generation_tasks.jsonl")
    ap.add_argument("--generated-manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--blend", choices=["object", "box"], default="object",
                    help="for paste_back:false crops -- 'object' pastes only the "
                         "added object (no bright-rectangle seam); 'box' is the "
                         "legacy feathered orange-box paste.")
    ap.add_argument("--object-thr", type=float, default=30.0,
                    help="per-pixel max-channel diff threshold for object detection")
    args = ap.parse_args()

    model = args.model_name
    gen_manifest = REPO / Path(args.generated_manifest or f"generated_crops/{model}/manifest.jsonl")
    out_dir = REPO / Path(args.out_dir or f"spliced_full/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = {row["task_id"]: row for row in load_jsonl(REPO / args.tasks)}
    generated = [row for row in load_jsonl(REPO / gen_manifest) if row.get("status") == "ok"]

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
            if args.blend == "object":
                mask = object_mask(original_crop, generated_crop, box,
                                   args.object_thr, args.feather)
                paste_mode = "object_only"
            else:
                mask = feathered_mask(expected, box, args.feather)
                paste_mode = "masked_insert_region"
            crop_to_paste = Image.composite(generated_crop, original_crop, mask)
        else:
            crop_to_paste = generated_crop
            paste_mode = "full_context_crop"

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
            "source_generated_paste_back": row.get("paste_back"),
            "status": "ok",
        })
        print(f"{task_id}: {out_path}")

    man_path = out_dir / "manifest.jsonl"
    man_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_rows), encoding="utf-8")
    print(f"done: {len(manifest_rows)} full spliced images -> {out_dir}")


if __name__ == "__main__":
    main()
