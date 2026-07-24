#!/usr/bin/env python3
"""Build generation tasks with a wider source-image context crop.

This keeps each task ID and absolute edit box unchanged while expanding the
context region around the original crop. It is useful when a tight crop does
not show enough floor or other support geometry for a complete inserted object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]


def repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"path must stay inside repository: {path}")
    return resolved


def expanded_box(
    box: list[int],
    image_size: tuple[int, int],
    scale: float,
) -> list[int]:
    image_width, image_height = image_size
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    target_width = min(image_width, max(width, round(width * scale)))
    target_height = min(image_height, max(height, round(height * scale)))
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    new_left = round(center_x - target_width / 2)
    new_top = round(center_y - target_height / 2)
    new_left = max(0, min(image_width - target_width, new_left))
    new_top = max(0, min(image_height - target_height, new_top))
    return [
        new_left,
        new_top,
        new_left + target_width,
        new_top + target_height,
    ]


def select_tasks(rows: list[dict[str, Any]], only: str) -> list[dict[str, Any]]:
    selected = {item.strip() for item in only.split(",") if item.strip()}
    if not selected:
        raise ValueError("--only must select at least one task")
    output = [
        row
        for index, row in enumerate(rows)
        if row["task_id"] in selected or str(index) in selected
    ]
    matched = {row["task_id"] for row in output}
    unresolved = {
        item
        for item in selected
        if not item.isdigit() and item not in matched
    }
    if unresolved:
        raise ValueError(f"unknown task IDs: {sorted(unresolved)}")
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = repo_path(args.tasks)
    output_tasks = repo_path(args.output_tasks)
    crop_dir = repo_path(args.crop_dir)
    if output_tasks.exists():
        raise FileExistsError(f"refusing to overwrite {output_tasks}")
    if args.scale <= 1:
        raise ValueError("--scale must be greater than 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_tasks(rows, args.only)
    crop_dir.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, Any]] = []
    suffix = ".jpg" if args.image_format == "jpeg" else ".png"

    for task in selected:
        source_path = repo_path(Path(task["source_image"]))
        original_context = [int(value) for value in task["context_region_xyxy"]]
        edit_box = [int(value) for value in task["edit_region_xyxy"]]
        crop_path = crop_dir / f"{task['task_id']}_context{suffix}"
        if crop_path.exists():
            raise FileExistsError(f"refusing to overwrite {crop_path}")

        with Image.open(source_path) as source_image:
            source = source_image.convert("RGB")
            expected_size = (
                int(task["image_size"]["width"]),
                int(task["image_size"]["height"]),
            )
            if source.size != expected_size:
                raise ValueError(
                    f"{task['task_id']}: source size {source.size} "
                    f"!= metadata {expected_size}"
                )
            context_box = expanded_box(original_context, source.size, args.scale)
            left, top, right, bottom = context_box
            if not (
                left <= edit_box[0] < edit_box[2] <= right
                and top <= edit_box[1] < edit_box[3] <= bottom
            ):
                raise ValueError(
                    f"{task['task_id']}: expanded context misses edit box"
                )
            crop = source.crop(context_box)
            if args.image_format == "jpeg":
                crop.save(
                    crop_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                )
            else:
                crop.save(crop_path, format="PNG")

        relative_crop = crop_path.relative_to(REPO)
        new_task = {
            **task,
            "context_crop": str(relative_crop),
            "context_region_xyxy": context_box,
            "edit_region_in_context_xyxy": [
                edit_box[0] - left,
                edit_box[1] - top,
                edit_box[2] - left,
                edit_box[3] - top,
            ],
            "crop_box": {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            },
            "original_context_crop": task["context_crop"],
            "original_context_region_xyxy": original_context,
            "context_revision": f"expanded_{args.scale:g}x",
        }
        built.append(new_task)

    temporary = output_tasks.with_suffix(output_tasks.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in built:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    temporary.replace(output_tasks)

    return {
        "tasks": len(built),
        "output_tasks": str(output_tasks.relative_to(REPO)),
        "crop_dir": str(crop_dir.relative_to(REPO)),
        "scale": args.scale,
        "image_format": args.image_format,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("annotations/trash_can_generation_tasks_remaining_148.jsonl"),
    )
    parser.add_argument("--only", required=True)
    parser.add_argument("--output-tasks", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
