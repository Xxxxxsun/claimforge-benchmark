#!/usr/bin/env python3
"""Validate and build visual-QC sheets for a full-image orange-box run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps


REPO = Path(__file__).resolve().parents[1]
ORANGE = (234, 122, 24)
OBJECT_KIND_BY_CANDIDATE = {
    "mouse": "mouse",
    "cat": "cat",
    "trash can": "trash-can",
}


def repo_path(path: Path) -> Path:
    resolved = (path if path.is_absolute() else REPO / path).resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {path}")
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_tasks(
    rows: list[dict[str, Any]],
    only: str | None,
) -> list[dict[str, Any]]:
    if not only:
        return rows
    tokens = [item.strip() for item in only.split(",") if item.strip()]
    if not tokens:
        raise ValueError("--only did not contain a task ID or index")
    id_to_index = {str(row["task_id"]): index for index, row in enumerate(rows)}
    if len(id_to_index) != len(rows):
        raise ValueError("duplicate task IDs before --only selection")
    selected_indices: set[int] = set()
    for token in tokens:
        matches: set[int] = set()
        if token in id_to_index:
            matches.add(id_to_index[token])
        try:
            numeric_index = int(token)
        except ValueError:
            pass
        else:
            if 0 <= numeric_index < len(rows):
                matches.add(numeric_index)
        if not matches:
            raise ValueError(f"--only token did not resolve: {token!r}")
        if len(matches) > 1:
            raise ValueError(f"--only token is ambiguous: {token!r}")
        index = next(iter(matches))
        if index in selected_indices:
            raise ValueError(f"--only selects the same task twice: {token!r}")
        selected_indices.add(index)
    return [row for index, row in enumerate(rows) if index in selected_indices]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_size(task: dict[str, Any]) -> tuple[int, int]:
    value = task["image_size"]
    if isinstance(value, dict):
        return int(value["width"]), int(value["height"])
    return int(value[0]), int(value[1])


def ring_pixels(
    image: Image.Image,
    box: list[int],
    width: int,
) -> Iterable[tuple[int, int, int]]:
    x1, y1, x2, y2 = [int(value) for value in box]
    width = max(1, int(width))
    pixels = image.load()
    max_x, max_y = image.width - 1, image.height - 1
    seen: set[tuple[int, int]] = set()
    for offset in range(width):
        for x in range(max(0, x1), min(max_x, x2) + 1):
            seen.add((x, min(max_y, max(0, y1 + offset))))
            seen.add((x, min(max_y, max(0, y2 - offset))))
        for y in range(max(0, y1), min(max_y, y2) + 1):
            seen.add((min(max_x, max(0, x1 + offset)), y))
            seen.add((min(max_x, max(0, x2 - offset)), y))
    return (pixels[x, y] for x, y in seen)


def orange_fraction(
    image: Image.Image,
    box: list[int],
    width: int,
    tolerance: int = 40,
) -> float:
    values = list(ring_pixels(image, box, width))
    if not values:
        return 0.0
    matches = sum(
        all(abs(int(pixel[index]) - ORANGE[index]) <= tolerance for index in range(3))
        for pixel in values
    )
    return matches / len(values)


def qc_crop(
    image: Image.Image,
    box: list[int],
    *,
    tile_image_size: tuple[int, int],
) -> Image.Image:
    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    radius = max(60, round(max(box_width, box_height) * 2.75))
    left = max(0, round(center_x - radius))
    top = max(0, round(center_y - radius))
    right = min(image.width, round(center_x + radius))
    bottom = min(image.height, round(center_y + radius))
    crop = image.crop((left, top, right, bottom))
    return ImageOps.contain(crop, tile_image_size, Image.Resampling.LANCZOS)


def build_contact_sheets(
    tasks: list[dict[str, Any]],
    outputs: dict[str, Path],
    out_dir: Path,
    *,
    columns: int = 5,
    rows_per_page: int = 5,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 300, 240
    image_area = (284, 204)
    per_page = columns * rows_per_page
    pages: list[Path] = []
    for page_index in range(math.ceil(len(tasks) / per_page)):
        canvas = Image.new(
            "RGB",
            (columns * tile_width, rows_per_page * tile_height),
            (225, 225, 225),
        )
        page_tasks = tasks[
            page_index * per_page : (page_index + 1) * per_page
        ]
        for index, task in enumerate(page_tasks):
            task_id = str(task["task_id"])
            with Image.open(outputs[task_id]) as image_file:
                image = image_file.convert("RGB")
            target_box = [int(value) for value in task["edit_region_xyxy"]]
            # Draw the annotation only on the disposable QC copy.  Cyan is
            # deliberately distinct from the orange model-input guide, so the
            # reviewer can see whether the generated object's body actually
            # covers the requested location without altering the saved output.
            qc_image = image.copy()
            qc_line_width = max(2, round(min(image.size) * 2 / 512))
            ImageDraw.Draw(qc_image).rectangle(
                target_box,
                outline=(0, 255, 255),
                width=qc_line_width,
            )
            crop = qc_crop(
                qc_image,
                target_box,
                tile_image_size=image_area,
            )
            tile = Image.new("RGB", (tile_width, tile_height), "white")
            tile.paste(
                crop,
                (
                    (tile_width - crop.width) // 2,
                    4 + (image_area[1] - crop.height) // 2,
                ),
            )
            ImageDraw.Draw(tile).text((6, 216), task_id, fill="black")
            canvas.paste(
                tile,
                ((index % columns) * tile_width, (index // columns) * tile_height),
            )
        page_path = out_dir / f"page_{page_index + 1:02d}.jpg"
        canvas.save(page_path, "JPEG", quality=92, optimize=True)
        pages.append(page_path)
    return pages


def validate(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = repo_path(args.tasks)
    output_dir = repo_path(args.output_dir)
    manifest_path = output_dir / "manifest.jsonl"
    tasks = load_jsonl(tasks_path)
    tasks = select_tasks(tasks, args.only)
    manifest_rows = load_jsonl(manifest_path)
    tasks_by_id = {str(row["task_id"]): row for row in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError(f"duplicate task IDs in {tasks_path}")

    latest: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        latest[str(row["task_id"])] = row
    if set(latest) != set(tasks_by_id):
        missing = sorted(set(tasks_by_id) - set(latest))
        extra = sorted(set(latest) - set(tasks_by_id))
        raise ValueError(f"manifest/task mismatch: missing={missing[:5]} extra={extra[:5]}")
    failed = sorted(
        task_id
        for task_id, row in latest.items()
        if row.get("status") != "ok"
    )
    if failed:
        raise ValueError(f"latest manifest rows failed: {failed[:10]}")

    outputs: dict[str, Path] = {}
    orange_flags: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    model_size_pairs: dict[str, int] = {}
    response_size_pairs: dict[str, int] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        row = latest[task_id]
        task_candidate = str(task.get("candidates"))
        expected_object_kind = OBJECT_KIND_BY_CANDIDATE.get(task_candidate)
        if expected_object_kind is None:
            raise ValueError(
                f"{task_id}: unsupported candidate {task_candidate!r}"
            )
        if row.get("candidate") != task.get("candidates"):
            raise ValueError(f"{task_id}: manifest candidate/task mismatch")
        if row.get("object_kind") != expected_object_kind:
            raise ValueError(
                f"{task_id}: object kind {row.get('object_kind')!r} "
                f"!= {expected_object_kind!r}"
            )
        if row.get("input_mode") != "full-image-orange-box":
            raise ValueError(f"{task_id}: unexpected input mode")
        if row.get("input_source_image") != str(task["source_image"]):
            raise ValueError(f"{task_id}: input source path mismatch")
        expected_box = [int(value) for value in task["edit_region_xyxy"]]
        if row.get("orange_box_xyxy") != expected_box:
            raise ValueError(f"{task_id}: orange box/task mismatch")
        if row.get("orange_guide_rgb") != list(ORANGE):
            raise ValueError(f"{task_id}: unexpected orange guide color")
        output = repo_path(Path(str(row["output_image"])))
        if output.parent != output_dir:
            raise ValueError(f"{task_id}: output outside run directory: {output}")
        if not output.is_file():
            raise FileNotFoundError(output)
        source = repo_path(Path(str(task["source_image"])))
        if row.get("input_source_sha256") != sha256_file(source):
            raise ValueError(f"{task_id}: source SHA-256 mismatch")

        expected_size = declared_size(task)
        if row.get("original_size") != list(expected_size):
            raise ValueError(f"{task_id}: manifest original size mismatch")
        if row.get("service_output_size") != row.get("model_input_size"):
            raise ValueError(
                f"{task_id}: service/model size mismatch: "
                f"{row.get('service_output_size')} != "
                f"{row.get('model_input_size')}"
            )
        with Image.open(output) as output_file:
            if output_file.mode != "RGB":
                raise ValueError(f"{task_id}: output mode {output_file.mode}")
            if output_file.size != expected_size:
                raise ValueError(
                    f"{task_id}: output size {output_file.size} != {expected_size}"
                )
            output_image = output_file.copy()
        with Image.open(source) as source_file:
            source_image = source_file.convert("RGB")
        if source_image.size != expected_size:
            raise ValueError(
                f"{task_id}: source size {source_image.size} != {expected_size}"
            )

        ring_width = int(row.get("guide_ring_width_original_pixels") or 3)
        box = [int(value) for value in task["edit_region_xyxy"]]
        output_orange = orange_fraction(output_image, box, ring_width)
        source_orange = orange_fraction(source_image, box, ring_width)
        if output_orange >= 0.30 and output_orange - source_orange >= 0.15:
            orange_flags.append(
                {
                    "task_id": task_id,
                    "output_orange_fraction": round(output_orange, 6),
                    "source_orange_fraction": round(source_orange, 6),
                }
            )

        digest = sha256_file(output)
        if digest in output_hashes:
            raise ValueError(
                f"duplicate output bytes: {task_id} and {output_hashes[digest]}"
            )
        output_hashes[digest] = task_id
        outputs[task_id] = output
        model_key = "x".join(str(value) for value in row["model_input_size"])
        response_key = "x".join(str(value) for value in row["service_output_size"])
        model_size_pairs[model_key] = model_size_pairs.get(model_key, 0) + 1
        response_size_pairs[response_key] = response_size_pairs.get(response_key, 0) + 1

    png_paths = sorted(output_dir.glob("*.png"))
    if len(png_paths) != len(tasks):
        raise ValueError(f"PNG count {len(png_paths)} != task count {len(tasks)}")

    contact_dir = repo_path(args.contact_sheet_dir)
    pages = build_contact_sheets(tasks, outputs, contact_dir)
    summary = {
        "tasks": len(tasks),
        "manifest_rows": len(manifest_rows),
        "latest_successes": len(latest),
        "png_files": len(png_paths),
        "unique_output_sha256": len(output_hashes),
        "orange_residual_flags": orange_flags,
        "model_input_sizes": dict(sorted(model_size_pairs.items())),
        "service_output_sizes": dict(sorted(response_size_pairs.items())),
        "contact_sheet_pages": [
            path.relative_to(REPO).as_posix() for path in pages
        ],
        "status": "structurally_valid",
    }
    summary_path = output_dir / "validation.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--only")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(validate(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
