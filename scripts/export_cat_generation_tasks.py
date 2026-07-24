#!/usr/bin/env python3
"""Export a completed slot-labeler payload into generation tasks and crops.

The command-line defaults preserve the original cat workflow.  Other object
families can supply their own paths, task prefix, and fallback candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SLOTS = REPO / "annotations" / "claimforge-good-mouse-source-cat-275-slots.json"
DEFAULT_TASKS = REPO / "annotations" / "cat_generation_tasks.jsonl"
DEFAULT_CROPS = REPO / "crops" / "context_cat"


def rect_xyxy(rect: dict) -> list[int]:
    x1 = int(rect["x"])
    y1 = int(rect["y"])
    return [x1, y1, x1 + int(rect["width"]), y1 + int(rect["height"])]


def source_relative_path(raw_path: str) -> Path:
    """Turn the labeler's ../../source_pool/... URL into a repo-relative path."""
    parts = list(PurePosixPath(raw_path).parts)
    while parts and parts[0] in {".", "..", "/"}:
        parts.pop(0)
    relative = Path(*parts)
    absolute = (REPO / relative).resolve()
    if not absolute.is_relative_to(REPO.resolve()):
        raise ValueError(f"source path escapes repository: {raw_path}")
    return relative


def task_id_for(prefix: str, image_id: str, slot: dict, slot_count: int) -> str:
    if slot_count == 1:
        return f"{prefix}_{image_id}"
    slot_id = str(slot.get("id") or "slot_unknown")
    return f"{prefix}_{image_id}_{slot_id}"


def export(args: argparse.Namespace) -> dict:
    task_prefix = str(getattr(args, "task_prefix", "cat")).strip().strip("_")
    default_candidate = str(getattr(args, "default_candidate", "cat")).strip()
    if not task_prefix or not all(ch.isalnum() or ch == "_" for ch in task_prefix):
        raise ValueError(f"invalid task prefix: {task_prefix!r}")
    if not default_candidate:
        raise ValueError("default candidate must not be empty")

    slots_path = args.slots_json if args.slots_json.is_absolute() else REPO / args.slots_json
    tasks_path = args.tasks if args.tasks.is_absolute() else REPO / args.tasks
    crop_dir = args.crop_dir if args.crop_dir.is_absolute() else REPO / args.crop_dir
    slots_path = slots_path.resolve()
    tasks_path = tasks_path.resolve()
    crop_dir = crop_dir.resolve()
    if not tasks_path.is_relative_to(REPO) or not crop_dir.is_relative_to(REPO):
        raise ValueError("task and crop outputs must stay inside the repository")

    payload = json.loads(slots_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("annotation payload must contain an images list")

    tasks: list[dict] = []
    seen_task_ids: set[str] = set()
    crops_to_write: list[tuple[Path, Image.Image]] = []
    empty_images: list[str] = []
    incomplete_slots: list[str] = []
    skip_incomplete = bool(getattr(args, "skip_incomplete", False))

    for item in images:
        slots = item.get("slots") or []
        if not slots:
            empty_images.append(str(item["id"]))
            continue

        source_rel = source_relative_path(str(item["image"]))
        source_abs = REPO / source_rel
        if not source_abs.is_file():
            raise FileNotFoundError(source_abs)

        with Image.open(source_abs) as source_file:
            source = source_file.convert("RGB")
        width, height = source.size
        declared = item.get("image_size") or {}
        declared_size = (int(declared.get("width", 0)), int(declared.get("height", 0)))
        if declared_size != source.size:
            raise ValueError(
                f"{item['id']}: declared size {declared_size} != source size {source.size}"
            )

        for slot in slots:
            insert_box = slot.get("insert_box")
            crop_box = slot.get("crop_box")
            if not insert_box or not crop_box:
                incomplete_id = f"{item['id']}:{slot.get('id') or 'slot_unknown'}"
                if skip_incomplete:
                    incomplete_slots.append(incomplete_id)
                    continue
                raise ValueError(f"{item['id']} {slot.get('id')}: incomplete slot")

            insert_xyxy = rect_xyxy(insert_box)
            crop_xyxy = rect_xyxy(crop_box)
            ix1, iy1, ix2, iy2 = insert_xyxy
            cx1, cy1, cx2, cy2 = crop_xyxy
            if not (0 <= cx1 < cx2 <= width and 0 <= cy1 < cy2 <= height):
                raise ValueError(f"{item['id']}: crop box outside source: {crop_xyxy}")
            if not (cx1 <= ix1 < ix2 <= cx2 and cy1 <= iy1 < iy2 <= cy2):
                raise ValueError(
                    f"{item['id']}: insert {insert_xyxy} outside crop {crop_xyxy}"
                )

            task_id = task_id_for(task_prefix, str(item["id"]), slot, len(slots))
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task id: {task_id}")
            seen_task_ids.add(task_id)

            context_rel = crop_dir.relative_to(REPO) / f"{task_id}_context.jpg"
            edit_in_context = [
                ix1 - cx1,
                iy1 - cy1,
                ix2 - cx1,
                iy2 - cy1,
            ]
            candidate = (
                str(slot.get("candidates") or default_candidate).strip()
                or default_candidate
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "image_id": item["id"],
                    "slot_id": slot.get("id"),
                    "source_image": source_rel.as_posix(),
                    "context_crop": context_rel.as_posix(),
                    "image_size": {"width": width, "height": height},
                    "context_region_xyxy": crop_xyxy,
                    "edit_region_xyxy": insert_xyxy,
                    "edit_region_in_context_xyxy": edit_in_context,
                    "insert_box": insert_box,
                    "crop_box": crop_box,
                    "candidates": candidate,
                }
            )
            crops_to_write.append((REPO / context_rel, source.crop(tuple(crop_xyxy))))

    if not args.dry_run:
        crop_dir.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        for crop_path, crop in crops_to_write:
            crop.save(crop_path, "JPEG", quality=95, optimize=True)

        temporary = tasks_path.with_suffix(tasks_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        temporary.replace(tasks_path)

    return {
        "annotation_images": len(images),
        "empty_images": len(empty_images),
        "empty_image_ids": empty_images,
        "incomplete_slots": len(incomplete_slots),
        "incomplete_slot_ids": incomplete_slots,
        "generation_tasks": len(tasks),
        "unique_task_ids": len(seen_task_ids),
        "task_prefix": task_prefix,
        "default_candidate": default_candidate,
        "tasks_path": str(tasks_path),
        "crop_dir": str(crop_dir),
        "dry_run": args.dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots-json", type=Path, default=DEFAULT_SLOTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--task-prefix", default="cat")
    parser.add_argument("--default-candidate", default="cat")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="skip partially drawn slots instead of rejecting the whole export",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
