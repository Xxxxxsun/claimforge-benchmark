#!/usr/bin/env python3
"""Export browser-labeler slots into CLAIMFORGE generation handoff files."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "source_pool" / "openimages_v7_600" / "manifest.json"


def rect_xyxy(rect: dict) -> list[int]:
    x1 = int(rect["x"])
    y1 = int(rect["y"])
    return [x1, y1, x1 + int(rect["width"]), y1 + int(rect["height"])]


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normal_slot(slot: dict, index: int) -> dict:
    return {
        "id": slot.get("id") or f"slot_{index + 1:03d}",
        "label": slot.get("label") or f"slot {index + 1}",
        "candidates": (slot.get("candidates") or "mouse").strip() or "mouse",
        "insert_box": slot.get("insert_box"),
        "crop_box": slot.get("crop_box"),
    }


def complete(slot: dict) -> bool:
    return bool(slot.get("insert_box") and slot.get("crop_box"))


def clamp_xyxy(xyxy: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def save_overlay(image: Image.Image, insert_xyxy: list[int], crop_xyxy: list[int], out_path: Path):
    overlay = image.copy()
    max_side = max(overlay.size)
    scale = 1.0
    if max_side > 1600:
        scale = 1600 / max_side
        overlay.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    def scaled(box: list[int]) -> list[int]:
        return [round(v * scale) for v in box]

    draw = ImageDraw.Draw(overlay)
    crop = scaled(crop_xyxy)
    insert = scaled(insert_xyxy)
    line = max(3, round(4 * scale))
    draw.rectangle(crop, outline=(37, 99, 235), width=line)
    draw.rectangle(insert, outline=(234, 122, 24), width=line)
    overlay.save(out_path, "JPEG", quality=90, optimize=True)


def build(args: argparse.Namespace):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items_by_id = {item["id"]: item for item in manifest}
    slots_raw = json.loads(Path(args.slots_json).read_text(encoding="utf-8"))

    slots_by_image = {}
    for item in manifest:
        raw_slots = slots_raw.get(item["id"], [])
        slots_by_image[item["id"]] = [
            normal_slot(slot, index)
            for index, slot in enumerate(raw_slots)
            if isinstance(slot, dict)
        ]

    task_prefix = args.task_prefix
    task_items = [
        item for item in manifest
        if not task_prefix or item["id"].startswith(task_prefix)
    ]

    annotations_dir = REPO / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    crop_context_dir = REPO / "crops" / "context"
    crop_insert_dir = REPO / "crops" / "insert"
    masks_dir = REPO / "masks"
    overlays_dir = REPO / "overlays"
    if args.clean_generated:
        clean_dir(crop_context_dir)
        clean_dir(crop_insert_dir)
        clean_dir(masks_dir)
        clean_dir(overlays_dir)
    else:
        for path in (crop_context_dir, crop_insert_dir, masks_dir, overlays_dir):
            path.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 2,
        "task": "rekey_method_slots",
        "coordinate_space": "original_image_pixels",
        "active_image": args.active_image,
        "images": [
            {
                "id": item["id"],
                "title": f"{item.get('category', 'source')} · {item.get('title') or item.get('source_image_id') or item['id']}",
                "image": item["path"],
                "image_size": item.get("size") or {"width": 0, "height": 0},
                "slots": slots_by_image[item["id"]],
            }
            for item in manifest
        ],
    }

    tasks = []
    rows = []
    source_manifest = []

    for item in task_items:
        source_rel = Path("source_pool") / "openimages_v7_600" / item["path"]
        source_abs = REPO / source_rel
        if not source_abs.exists():
            raise FileNotFoundError(source_abs)

        item_slots = slots_by_image.get(item["id"], [])
        complete_slots = [slot for slot in item_slots if complete(slot)]
        if not complete_slots:
            continue

        with Image.open(source_abs) as src:
            image = src.convert("RGB")
        width, height = image.size
        source_manifest.append({**item, "path": str(source_rel)})

        for slot in complete_slots:
            insert_xyxy = clamp_xyxy(rect_xyxy(slot["insert_box"]), width, height)
            crop_xyxy = clamp_xyxy(rect_xyxy(slot["crop_box"]), width, height)
            if insert_xyxy[2] <= insert_xyxy[0] or insert_xyxy[3] <= insert_xyxy[1]:
                raise ValueError(f"empty insert box for {item['id']} {slot['id']}")
            if crop_xyxy[2] <= crop_xyxy[0] or crop_xyxy[3] <= crop_xyxy[1]:
                raise ValueError(f"empty crop box for {item['id']} {slot['id']}")

            task_id = f"{item['id']}_{slot['id']}"
            context_rel = Path("crops") / "context" / f"{task_id}_context.jpg"
            insert_rel = Path("crops") / "insert" / f"{task_id}_insert.jpg"
            mask_rel = Path("masks") / f"{task_id}_insert_mask.png"
            overlay_rel = Path("overlays") / f"{task_id}_overlay.jpg"

            image.crop(tuple(crop_xyxy)).save(REPO / context_rel, "JPEG", quality=95, optimize=True)
            image.crop(tuple(insert_xyxy)).save(REPO / insert_rel, "JPEG", quality=95, optimize=True)

            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).rectangle(insert_xyxy, fill=255)
            mask.save(REPO / mask_rel)
            save_overlay(image, insert_xyxy, crop_xyxy, REPO / overlay_rel)

            edit_in_context = [
                insert_xyxy[0] - crop_xyxy[0],
                insert_xyxy[1] - crop_xyxy[1],
                insert_xyxy[2] - crop_xyxy[0],
                insert_xyxy[3] - crop_xyxy[1],
            ]
            task = {
                "task_id": task_id,
                "image_id": item["id"],
                "slot_id": slot["id"],
                "source_image": str(source_rel),
                "context_crop": str(context_rel),
                "insert_crop": str(insert_rel),
                "insert_mask": str(mask_rel),
                "overlay": str(overlay_rel),
                "image_size": {"width": width, "height": height},
                "candidates": slot["candidates"],
                "insert_box": slot["insert_box"],
                "crop_box": slot["crop_box"],
                "edit_region_xyxy": insert_xyxy,
                "context_region_xyxy": crop_xyxy,
                "edit_region_in_context_xyxy": edit_in_context,
                "prompt_hint": (
                    f"Add a small realistic {slot['candidates']} inside the insert box "
                    "while preserving the rest of the crop."
                ),
            }
            tasks.append(task)
            rows.append({
                "task_id": task_id,
                "source_image": str(source_rel),
                "edit_region": {"xyxy": insert_xyxy},
                "context_region": {"xyxy": crop_xyxy},
                "add": {"objects": [slot["candidates"]]},
                "metadata": {
                    "image_id": item["id"],
                    "slot_id": slot["id"],
                    "context_crop": str(context_rel),
                    "insert_mask": str(mask_rel),
                    "overlay": str(overlay_rel),
                },
            })

    (annotations_dir / "rekey_method_slots_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (annotations_dir / "slots_by_image.json").write_text(
        json.dumps(slots_by_image, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, data in (
        ("generation_tasks.jsonl", tasks),
        ("slots_flat.jsonl", tasks),
        ("annotation_rows.jsonl", rows),
    ):
        with (annotations_dir / name).open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (REPO / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "manifest_images": len(manifest),
        "task_prefix": task_prefix,
        "task_images_with_complete_slots": len(source_manifest),
        "generation_tasks": len(tasks),
        "payload_images": len(payload["images"]),
    }
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots-json", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--task-prefix", default="")
    parser.add_argument("--active-image", default=None)
    parser.add_argument("--clean-generated", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
