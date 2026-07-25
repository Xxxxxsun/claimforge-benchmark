#!/usr/bin/env python3
"""Run the Hunyuan image editor on a complete source image with one orange box.

This is the full-context control for the normal CLAIMFORGE crop-and-splice
workflow.  The complete source image is resized as a whole to the model input
resolution, the annotated insert box is drawn in orange, and no coordinate or
crop-position description is added to the prompt.  The model output is resized
back to the exact source dimensions and saved without pasting source pixels
back over it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_hunyuan_generation import call_edit_omni  # noqa: E402


ALIGN = 16
DEFAULT_ORANGE = (234, 122, 24)
HUNYUAN_BASE_SIZE = 1024
EXPECTED_CANDIDATE = {
    "mouse": "mouse",
    "cat": "cat",
    "trash-can": "trash can",
}

MOUSE_PROMPT = (
    "The orange rectangle is a temporary placement guide and is not part of "
    "the scene. The required edit must be visibly completed: add exactly one "
    "small, clearly recognizable gray-brown mouse naturally inside that orange "
    "rectangle. Keep the whole mouse, including its distinct head, ears, body, "
    "feet, and thin tail, visible and inside the marked area. It must read "
    "unambiguously as a mouse rather than a gray blob, dish, tray, napkin, toy, "
    "or patch. Match the source image's "
    "visual style, resolution, level of detail, sharpness or blur, noise, "
    "compression, color, lighting, perspective, and depth of field. Remove the "
    "entire orange guide in the result. Preserve the complete original framing "
    "and everything outside the marked area. Do not crop, zoom, reframe, add "
    "text, add another animal, or make unrelated changes."
)

CAT_POSES = (
    "walking naturally through the scene",
    "standing casually while observing something nearby",
    "sitting in a relaxed, unposed way",
    "crouching as if inspecting or sniffing a nearby surface",
    "resting comfortably",
    "stretching or turning naturally",
)
CAT_ORIENTATIONS = (
    "seen in side profile and facing toward the left side of the scene",
    "seen in side profile and facing toward the right side of the scene",
    "seen from a three-quarter angle and facing away from the viewer",
    "with its back mostly toward the viewer and attention on the surrounding scene",
)
CAT_PROMPT = (
    "The orange rectangle is a temporary placement guide and is not part of "
    "the scene. Add exactly one complete cat naturally at that orange "
    "rectangle, {pose}, {orientation}. The rectangle is the exact target, not a "
    "loose hint: the center of the cat's torso must coincide with the center of "
    "the rectangle and cover its interior, even if a nearby surface looks more "
    "convenient. Keep the whole cat visible; it may extend naturally beyond the "
    "rectangle while remaining centered on it. Its head and eyes must be "
    "directed into the scene rather than toward the viewer or camera, so the "
    "result feels candid and unposed. "
    "Match the source image's visual style, resolution, level of detail, "
    "sharpness or blur, noise, compression, color, lighting, perspective, and "
    "depth of field; do not make the cat cleaner, sharper, or more realistic "
    "than the source. Remove the entire orange guide in the result. Preserve "
    "the complete original framing and everything outside the marked area. Do "
    "not crop, zoom, reframe, add text, add another animal, or make unrelated "
    "changes."
)

TRASH_CAN_PROMPT = (
    "The orange rectangle is a temporary placement guide and is not part of "
    "the scene. Add exactly one complete, ordinary trash can naturally at that "
    "marked location. The rectangle is the exact target, not a loose hint: the "
    "center of the bin's body must coincide with the center of the rectangle "
    "and cover its interior. The surface directly beneath the orange rectangle "
    "is authoritative: keep the bin there, including when that surface is a "
    "bed, table, desk, counter, shelf, stool, or floor; never relocate it to a "
    "different or more convenient surface. Keep the full rim or lid, both side "
    "contours, complete body, and complete base visible; the bin may extend "
    "naturally beyond the rectangle while remaining centered on it. Make the "
    "base rest naturally on the marked support with appropriate contact shadow "
    "or subtle surface indentation. Match the source image's visual style, "
    "resolution, detail, sharpness or blur, noise, compression, color, "
    "lighting, perspective, and depth of field; do not make the bin cleaner, "
    "sharper, or more realistic than the source. Remove the entire orange guide "
    "in the result. Preserve the complete original framing and everything "
    "outside the marked area. Do not crop, zoom, reframe, add text, logos, "
    "loose trash, another bin, or unrelated changes."
)


def hunyuan_buckets(base_size: int = HUNYUAN_BASE_SIZE) -> tuple[tuple[int, int], ...]:
    """Return the model's trained 1024-base buckets as PIL ``(W, H)`` sizes."""
    if base_size != HUNYUAN_BASE_SIZE:
        raise ValueError(
            f"this control runner is pinned to base size {HUNYUAN_BASE_SIZE}"
        )
    height_width: set[tuple[int, int]] = {(base_size, base_size)}

    height, width = base_size, base_size
    while not (height >= base_size * 2 and width <= base_size // 2):
        height = min(height + base_size // 16, base_size * 2)
        width = max(width - base_size // 16, base_size // 2)
        height_width.add((height, width))

    height, width = base_size, base_size
    while not (height <= base_size // 2 and width >= base_size * 2):
        height = max(height - base_size // 16, base_size // 2)
        width = min(width + base_size // 16, base_size * 2)
        height_width.add((height, width))

    # Official extra aspect buckets.  Lower-area square extras are intentionally
    # omitted so every control input keeps the standard 1024-base pixel budget.
    height_width.update(
        {
            (1024, 768),
            (1280, 720),
            (768, 1024),
            (720, 1280),
        }
    )
    return tuple(sorted((width, height) for height, width in height_width))


def model_size(width: int, height: int) -> tuple[int, int]:
    """Choose the nearest exact learned bucket, then resize the whole frame."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {(width, height)}")
    source_ratio = width / height
    return min(
        hunyuan_buckets(),
        key=lambda bucket: (
            abs(math.log((bucket[0] / bucket[1]) / source_ratio)),
            abs(bucket[0] * bucket[1] - HUNYUAN_BASE_SIZE**2),
            bucket,
        ),
    )


def validate_box(box: list[int], size: tuple[int, int], task_id: str) -> list[int]:
    width, height = size
    if len(box) != 4:
        raise ValueError(f"{task_id}: expected xyxy box, got {box!r}")
    x1, y1, x2, y2 = [int(value) for value in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{task_id}: box {box!r} outside {width}x{height}")
    return [x1, y1, x2, y2]


def scale_box(
    box: list[int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x1, y1, x2, y2 = box
    scaled = [
        round(x1 * target_width / source_width),
        round(y1 * target_height / source_height),
        round(x2 * target_width / source_width),
        round(y2 * target_height / source_height),
    ]
    scaled[0] = min(target_width - 2, max(0, scaled[0]))
    scaled[1] = min(target_height - 2, max(0, scaled[1]))
    scaled[2] = min(target_width - 1, max(scaled[0] + 1, scaled[2]))
    scaled[3] = min(target_height - 1, max(scaled[1] + 1, scaled[3]))
    return scaled


def guide_width(target_size: tuple[int, int]) -> int:
    """Use the same orange-box visual weight as the existing label export."""
    return max(3, round(min(target_size) * 4 / 512))


def draw_orange_guide(
    image: Image.Image,
    box: list[int],
    color: tuple[int, int, int] = DEFAULT_ORANGE,
    width: int | None = None,
) -> tuple[Image.Image, int]:
    guided = image.copy()
    line_width = guide_width(guided.size) if width is None else int(width)
    if line_width <= 0:
        raise ValueError(f"invalid guide width: {line_width}")
    ImageDraw.Draw(guided).rectangle(box, outline=color, width=line_width)
    return guided, line_width


def restore_guide_ring(
    output: Image.Image,
    source: Image.Image,
    box: list[int],
    width: int,
) -> Image.Image:
    """Restore only the temporary guide stroke from pristine source pixels."""
    if output.size != source.size:
        raise ValueError(f"ring restore size mismatch: {output.size} != {source.size}")
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle(box, outline=255, width=max(1, int(width)))
    return Image.composite(source, output, mask)


def make_prompt(object_kind: str, task_id: str, variation_key: str = "") -> str:
    if object_kind == "mouse":
        return MOUSE_PROMPT
    if object_kind == "trash-can":
        return TRASH_CAN_PROMPT
    if object_kind != "cat":
        raise ValueError(f"unsupported object kind: {object_kind}")
    digest = hashlib.sha256(
        f"{task_id}\0{variation_key}".encode("utf-8")
    ).digest()
    return CAT_PROMPT.format(
        pose=CAT_POSES[digest[8] % len(CAT_POSES)],
        orientation=CAT_ORIENTATIONS[digest[9] % len(CAT_ORIENTATIONS)],
    )


def stable_seed(task_id: str, seed_salt: str) -> int:
    digest = hashlib.sha256(
        f"{task_id}\0{seed_salt}".encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:8], "big") % 9_000_000) + 1


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path_for(output_dir: Path, task_id: str) -> Path:
    return output_dir / f"{task_id}.png"


def process_task(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    task_id = str(task["task_id"])
    source_rel = Path(str(task["source_image"]))
    source_path = REPO / source_rel
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")
    original_size = source.size

    declared_size = task.get("image_size") or {}
    if declared_size:
        if isinstance(declared_size, dict):
            declared = (
                int(declared_size.get("width", 0)),
                int(declared_size.get("height", 0)),
            )
        else:
            declared = tuple(int(value) for value in declared_size)
        if declared != original_size:
            raise ValueError(
                f"{task_id}: declared size {declared} != source {original_size}"
            )

    box = validate_box(
        [int(value) for value in task["edit_region_xyxy"]],
        original_size,
        task_id,
    )
    target_size = model_size(*original_size)
    resized = source.resize(target_size, Image.Resampling.LANCZOS)
    model_box = scale_box(box, original_size, target_size)
    guided, line_width = draw_orange_guide(
        resized,
        model_box,
        width=args.guide_width,
    )
    prompt = make_prompt(args.object_kind, task_id, args.seed_salt)
    seed = stable_seed(task_id, args.seed_salt)

    if args.save_guides:
        guide_dir = REPO / args.save_guides
        guide_dir.mkdir(parents=True, exist_ok=True)
        guided.save(guide_dir / f"{task_id}.png", "PNG")

    edited_model = call_edit_omni(
        args.url,
        args.model,
        guided,
        prompt,
        target_size[0],
        target_size[1],
        args.steps,
        seed,
        bot_task=args.bot_task,
        sys_type=args.sys_type,
        guidance_scale=args.guidance_scale,
        timeout=args.timeout,
    )
    service_output_size = edited_model.size
    if service_output_size != target_size:
        raise ValueError(
            f"{task_id}: service output size {service_output_size} "
            f"!= requested model input size {target_size}"
        )
    output = edited_model.resize(original_size, Image.Resampling.LANCZOS)
    ring_width_original = max(
        2,
        round(
            line_width
            * max(
                original_size[0] / target_size[0],
                original_size[1] / target_size[1],
            )
        )
        + 2,
    )
    if args.restore_guide_ring:
        output = restore_guide_ring(
            output,
            source,
            box,
            ring_width_original,
        )
    output_path = output_path_for(args.output_dir_abs, task_id)
    temporary = output_path.with_suffix(".png.part")
    output.save(temporary, "PNG")
    temporary.replace(output_path)

    return {
        "task_id": task_id,
        "image_id": task.get("image_id"),
        "slot_id": task.get("slot_id"),
        "candidate": task.get("candidates"),
        "input_mode": "full-image-orange-box",
        "input_source_image": source_rel.as_posix(),
        "input_source_sha256": source_sha256(source_path),
        "output_image": output_path.relative_to(REPO).as_posix(),
        "object_kind": args.object_kind,
        "orange_box_xyxy": box,
        "orange_box_model_xyxy": model_box,
        "orange_guide_rgb": list(DEFAULT_ORANGE),
        "orange_guide_width_model_pixels": line_width,
        "original_size": list(original_size),
        "model_input_size": list(target_size),
        "service_output_size": list(service_output_size),
        "resize_policy": "whole-frame-no-crop-to-nearest-hunyuan-1024-bucket",
        "guide_ring_restored_from_source": args.restore_guide_ring,
        "guide_ring_width_original_pixels": ring_width_original,
        "model": args.model_name,
        "service_model": args.model,
        "api_style": "omni",
        "bot_task": args.bot_task,
        "sys_type": args.sys_type,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "prompt": prompt,
        "seed": seed,
        "seed_salt": args.seed_salt,
        "status": "ok",
        "elapsed_seconds": round(time.time() - started, 3),
    }


def expected_resume_fields(
    task: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    source_rel = Path(str(task["source_image"]))
    source_path = REPO / source_rel
    with Image.open(source_path) as source_file:
        original_size = source_file.size
    box = validate_box(
        [int(value) for value in task["edit_region_xyxy"]],
        original_size,
        task_id,
    )
    target_size = model_size(*original_size)
    line_width = (
        guide_width(target_size)
        if args.guide_width is None
        else int(args.guide_width)
    )
    return {
        "candidate": task.get("candidates"),
        "input_mode": "full-image-orange-box",
        "input_source_image": source_rel.as_posix(),
        "input_source_sha256": source_sha256(source_path),
        "output_image": output_path_for(
            args.output_dir_abs,
            task_id,
        ).relative_to(REPO).as_posix(),
        "object_kind": args.object_kind,
        "orange_box_xyxy": box,
        "orange_box_model_xyxy": scale_box(
            box,
            original_size,
            target_size,
        ),
        "orange_guide_rgb": list(DEFAULT_ORANGE),
        "orange_guide_width_model_pixels": line_width,
        "original_size": list(original_size),
        "model_input_size": list(target_size),
        "guide_ring_restored_from_source": args.restore_guide_ring,
        "model": args.model_name,
        "service_model": args.model,
        "api_style": "omni",
        "bot_task": args.bot_task,
        "sys_type": args.sys_type,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "prompt": make_prompt(args.object_kind, task_id, args.seed_salt),
        "seed": stable_seed(task_id, args.seed_salt),
        "seed_salt": args.seed_salt,
    }


def load_completed(
    manifest_path: Path,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> set[str]:
    completed: set[str] = set()
    if not manifest_path.is_file():
        return completed
    latest: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row.get("task_id") or "")] = row
    for task in tasks:
        task_id = str(task["task_id"])
        row = latest.get(task_id)
        output = output_path_for(output_dir, task_id)
        if not row or row.get("status") != "ok" or not output.is_file():
            continue
        expected = expected_resume_fields(task, args)
        mismatches = [
            key
            for key, value in expected.items()
            if row.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"{task_id}: refusing incompatible --resume; manifest differs "
                f"for {mismatches}. Use a new output directory, or move the "
                f"stale PNG aside before intentionally regenerating it."
            )
        completed.add(task_id)
    return completed


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--object-kind",
        choices=["mouse", "cat", "trash-can"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8001/v1/images/edits",
    )
    parser.add_argument("--model", default="vllm_hunyuan_image3")
    parser.add_argument(
        "--bot-task",
        choices=["think", "recaption", "think_recaption", "vanilla"],
        default="think_recaption",
    )
    parser.add_argument("--sys-type", default="en_unified")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed-salt", default="full-image-orange-box-v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--guide-width",
        type=int,
        help=(
            "orange rectangle stroke width in model-input pixels; defaults to "
            "the annotation-export-equivalent width"
        ),
    )
    parser.add_argument("--save-guides", type=Path)
    parser.add_argument(
        "--restore-guide-ring",
        action="store_true",
        help=(
            "restore the temporary orange stroke from pristine source pixels; "
            "disabled by default because this is a raw full-image control"
        ),
    )
    parser.add_argument("--only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.tasks_abs = (
        args.tasks if args.tasks.is_absolute() else REPO / args.tasks
    ).resolve()
    args.output_dir_abs = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO / args.output_dir
    ).resolve()
    if not args.tasks_abs.is_relative_to(REPO):
        raise ValueError("tasks path must stay inside the repository")
    if not args.output_dir_abs.is_relative_to(REPO):
        raise ValueError("output directory must stay inside the repository")
    if args.save_guides:
        args.save_guides = (
            args.save_guides
            if args.save_guides.is_absolute()
            else REPO / args.save_guides
        ).resolve()
        if not args.save_guides.is_relative_to(REPO):
            raise ValueError("guide directory must stay inside the repository")
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if args.guide_width is not None and args.guide_width < 1:
        raise ValueError("guide width must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.tasks_abs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = select_tasks(rows, args.only)
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task IDs in {args.tasks_abs}")
    expected_candidate = EXPECTED_CANDIDATE[args.object_kind]
    candidate_mismatches = [
        str(row["task_id"])
        for row in rows
        if str(row.get("candidates")) != expected_candidate
    ]
    if candidate_mismatches:
        raise ValueError(
            f"--object-kind {args.object_kind!r} expects candidate "
            f"{expected_candidate!r}; mismatched tasks: "
            f"{candidate_mismatches[:10]}"
        )

    args.output_dir_abs.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir_abs / "manifest.jsonl"
    if args.resume:
        completed = load_completed(
            manifest_path,
            args.output_dir_abs,
            rows,
            args,
        )
        before = len(rows)
        rows = [row for row in rows if str(row["task_id"]) not in completed]
        print(
            f"resume: skipping {before - len(rows)} completed task(s); "
            f"{len(rows)} remaining",
            flush=True,
        )
    elif manifest_path.exists() or any(args.output_dir_abs.glob("*.png")):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: "
            f"{args.output_dir_abs}"
        )

    successes = 0
    failures = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        futures = [pool.submit(process_task, row, args) for row in rows]
        try:
            for index, (task, future) in enumerate(zip(rows, futures), start=1):
                try:
                    result = future.result()
                except Exception as error:
                    failures += 1
                    result = {
                        "task_id": str(task["task_id"]),
                        "input_source_image": str(task.get("source_image") or ""),
                        "input_mode": "full-image-orange-box",
                        "object_kind": args.object_kind,
                        "model": args.model_name,
                        "service_model": args.model,
                        "steps": args.steps,
                        "seed_salt": args.seed_salt,
                        "status": "failed",
                        "error": repr(error),
                    }
                    message = f"FAILED {error!r}"
                else:
                    successes += 1
                    message = (
                        f"{result['original_size'][0]}x"
                        f"{result['original_size'][1]} "
                        f"{result['elapsed_seconds']:.1f}s"
                    )
                manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                manifest.flush()
                print(
                    f"[{index}/{len(rows)}] {task['task_id']} {message}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown()

    print(
        json.dumps(
            {
                "object_kind": args.object_kind,
                "tasks_requested": len(rows),
                "successes": successes,
                "failures": failures,
                "output_dir": args.output_dir_abs.relative_to(REPO).as_posix(),
                "manifest": manifest_path.relative_to(REPO).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
