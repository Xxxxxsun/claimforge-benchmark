#!/usr/bin/env python3
"""Build paired oracle-edit and unchanged crop probes for commercial APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw


SCHEMA_VERSION = "claimforge_mouse_crop_scale_probe_v1"
DEFAULT_MANIFEST = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path("results/analysis/mouse_crop_scale_probe_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
    )


def resolve_repo_path(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def exact_diff_mask(source: Image.Image, forged: Image.Image) -> Image.Image:
    if source.size != forged.size:
        raise ValueError(f"pair size mismatch: {source.size} != {forged.size}")
    red, green, blue = ImageChops.difference(source, forged).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    return maximum.point(lambda value: 255 if value else 0, mode="L")


def square_box(
    bbox: tuple[int, int, int, int],
    side: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if side < 1 or side > min(width, height):
        raise ValueError(f"invalid square side {side} for image {image_size}")
    x1, y1, x2, y2 = bbox
    center_x_twice = x1 + x2
    center_y_twice = y1 + y2
    left = (center_x_twice - side) // 2
    top = (center_y_twice - side) // 2
    left = max(0, min(width - side, left))
    top = max(0, min(height - side, top))
    return left, top, left + side, top + side


def boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def axis_candidates(limit: int) -> list[int]:
    return sorted({round(limit * fraction / 8) for fraction in range(9)})


def real_control_box(
    diff_bbox: tuple[int, int, int, int],
    side: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    max_left = width - side
    max_top = height - side
    if max_left < 0 or max_top < 0:
        return None
    diff_center_x_twice = diff_bbox[0] + diff_bbox[2]
    diff_center_y_twice = diff_bbox[1] + diff_bbox[3]
    candidates: list[tuple[int, int, int, int, int]] = []
    for left in axis_candidates(max_left):
        for top in axis_candidates(max_top):
            box = (left, top, left + side, top + side)
            if boxes_overlap(box, diff_bbox):
                continue
            center_x_twice = 2 * left + side
            center_y_twice = 2 * top + side
            distance = (
                (center_x_twice - diff_center_x_twice) ** 2
                + (center_y_twice - diff_center_y_twice) ** 2
            )
            candidates.append((distance, -top, -left, left, top))
    if not candidates:
        return None
    _, _, _, left, top = max(candidates)
    return left, top, left + side, top + side


def save_canonical_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.convert("RGB").save(
        temporary,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )
    temporary.replace(path)
    with Image.open(path) as reopened:
        if reopened.mode != "RGB" or reopened.format != "JPEG":
            raise ValueError(f"invalid canonical JPEG: {path}")
        if reopened.getexif():
            raise ValueError(f"JPEG contains metadata: {path}")


def load_pair(
    repo_root: Path,
    row: dict[str, Any],
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int, int, int]]:
    source_path = resolve_repo_path(repo_root, str(row["source_image"]))
    forged_path = resolve_repo_path(repo_root, str(row["image"]))
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    with Image.open(forged_path) as opened:
        forged = opened.convert("RGB")
    mask = exact_diff_mask(source, forged)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"{row['task_id']}: exact-difference mask is empty")
    return source, forged, mask, bbox


def choose_tasks(
    candidates: list[dict[str, Any]],
    per_domain: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for domain in sorted({str(row["domain"]) for row in candidates}):
        domain_rows = sorted(
            (row for row in candidates if row["domain"] == domain),
            key=lambda row: (row["tight_side"], row["task_id"]),
        )
        if len(domain_rows) < per_domain:
            raise ValueError(
                f"domain {domain} has only {len(domain_rows)} eligible tasks"
            )
        used: set[int] = set()
        for quantile_index in range(per_domain):
            quantile = (quantile_index + 0.5) / per_domain
            index = round(quantile * (len(domain_rows) - 1))
            if index in used:
                index = next(
                    candidate
                    for candidate in range(len(domain_rows))
                    if candidate not in used
                )
            used.add(index)
            row = dict(domain_rows[index])
            row["selection_quantile"] = quantile
            selected.append(row)
    return sorted(selected, key=lambda row: (row["domain"], row["selection_quantile"]))


def contact_sheet(
    output_path: Path,
    selected_ids: list[str],
    rows: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    tile = 128
    label_height = 34
    columns = sorted(
        {(row["crop_factor"], row["region_kind"]) for row in rows},
        key=lambda item: (item[0], item[1]),
    )
    width = max(640, 220 + len(columns) * tile)
    height = 44 + len(selected_ids) * (tile + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Oracle edit crops vs unchanged same-image controls", fill="black")
    for column, (factor, kind) in enumerate(columns):
        draw.text(
            (220 + column * tile + 4, 25),
            f"{factor}x {kind[:4]}",
            fill="black",
        )
    by_key = {
        (row["task_id"], row["crop_factor"], row["region_kind"]): row for row in rows
    }
    for row_index, task_id in enumerate(selected_ids):
        top = 44 + row_index * (tile + label_height)
        draw.text((8, top + 4), task_id[:32], fill="black")
        task_row = next(row for row in rows if row["task_id"] == task_id)
        draw.text(
            (8, top + 20),
            f"tight={task_row['tight_side']} px",
            fill="black",
        )
        for column, (factor, kind) in enumerate(columns):
            row = by_key[(task_id, factor, kind)]
            with Image.open(repo_root / row["image"]) as opened:
                preview = opened.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(preview, (220 + column * tile, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--factors", default="1,2,4,8")
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    factors = sorted({int(value) for value in args.factors.split(",")})
    if not factors or factors[0] < 1:
        parser.error("--factors must contain positive integers")
    if args.per_domain < 1 or args.output_size < 32:
        parser.error("--per-domain must be positive and --output-size >= 32")

    repo_root = args.repo_root.resolve()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    ).resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    ).resolve()
    output_dir.relative_to(repo_root)

    source_rows = read_jsonl(manifest_path)
    if len(source_rows) != 250:
        raise ValueError(f"expected 250 benchmark rows, found {len(source_rows)}")

    candidates: list[dict[str, Any]] = []
    for source_row in source_rows:
        task_id = str(source_row["task_id"])
        source, forged, mask, bbox = load_pair(repo_root, source_row)
        tight_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        largest_side = tight_side * max(factors)
        if largest_side > min(source.size):
            continue
        if real_control_box(bbox, largest_side, source.size) is None:
            continue
        candidates.append(
            {
                **source_row,
                "diff_bbox_xyxy": list(bbox),
                "tight_side": tight_side,
            }
        )

    selected = choose_tasks(candidates, args.per_domain)
    crop_dir = output_dir / "images"
    result_rows: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    ordered_inputs: list[dict[str, Any]] = []

    for task_rank, selected_row in enumerate(selected):
        task_id = str(selected_row["task_id"])
        source, forged, mask, bbox = load_pair(repo_root, selected_row)
        tight_side = int(selected_row["tight_side"])
        for factor in factors:
            side = tight_side * factor
            suspect_box = square_box(bbox, side, source.size)
            control_box = real_control_box(bbox, side, source.size)
            if control_box is None:
                raise ValueError(f"{task_id}: no real control at factor {factor}")
            suspect_modified = sum(mask.crop(suspect_box).histogram()[1:])
            control_modified = sum(mask.crop(control_box).histogram()[1:])
            if suspect_modified == 0 or control_modified != 0:
                raise ValueError(
                    f"{task_id}: invalid modified pixels at factor {factor}"
                )
            if ImageChops.difference(
                source.crop(control_box), forged.crop(control_box)
            ).getbbox() is not None:
                raise ValueError(f"{task_id}: control crop is not pixel-identical")

            for region_kind, input_image, crop_box, modified_pixels in (
                ("suspicious", forged, suspect_box, suspect_modified),
                ("real_control", source, control_box, 0),
            ):
                crop = input_image.crop(crop_box).resize(
                    (args.output_size, args.output_size),
                    Image.Resampling.BICUBIC,
                )
                factor_name = f"{factor:02d}"
                compat_task_id = (
                    f"{task_id}__oracle_{region_kind}__field_{factor_name}x"
                )
                relative_image = (
                    output_dir.relative_to(repo_root)
                    / "images"
                    / f"{compat_task_id}.jpg"
                )
                absolute_image = repo_root / relative_image
                save_canonical_jpeg(crop, absolute_image, args.jpeg_quality)
                image_sha256 = sha256_file(absolute_image)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "rank": len(result_rows),
                    "task_rank": task_rank,
                    "task_id": task_id,
                    "compat_task_id": compat_task_id,
                    "domain": selected_row["domain"],
                    "region_kind": region_kind,
                    "crop_factor": factor,
                    "selection_quantile": selected_row["selection_quantile"],
                    "source_image": selected_row["source_image"],
                    "forged_image": selected_row["image"],
                    "diff_bbox_xyxy": list(bbox),
                    "tight_side": tight_side,
                    "crop_box_xyxy": list(crop_box),
                    "native_crop_size": [side, side],
                    "output_size": [args.output_size, args.output_size],
                    "resize_scale": args.output_size / side,
                    "interpolation": "Pillow bicubic",
                    "encoding": {
                        "format": "JPEG",
                        "quality": args.jpeg_quality,
                        "subsampling": 0,
                        "metadata": "none",
                    },
                    "modified_pixels": modified_pixels,
                    "modified_fraction_native": modified_pixels / (side * side),
                    "image": relative_image.as_posix(),
                    "image_sha256": image_sha256,
                    "bytes": absolute_image.stat().st_size,
                }
                result_rows.append(row)
                review_records.append(
                    {
                        "task_id": compat_task_id,
                        "status": "good",
                        "candidates": "mouse",
                        "source_image": relative_image.as_posix(),
                        "spliced_image": relative_image.as_posix(),
                        "image_size": [args.output_size, args.output_size],
                        "edit_region_xyxy": [0, 0, args.output_size, args.output_size],
                        "context_region_xyxy": [
                            0,
                            0,
                            args.output_size,
                            args.output_size,
                        ],
                        "crop_probe": {
                            "base_task_id": task_id,
                            "region_kind": region_kind,
                            "crop_factor": factor,
                        },
                    }
                )
                ordered_inputs.append(
                    {
                        "rank": len(ordered_inputs),
                        "task_id": compat_task_id,
                    }
                )

    manifest_relative = manifest_path.relative_to(repo_root)
    selected_ids = [str(row["task_id"]) for row in selected]
    write_jsonl(output_dir / "manifest.jsonl", result_rows)
    write_json(
        output_dir / "compat_review.json",
        {
            "schema_version": SCHEMA_VERSION,
            "records": review_records,
        },
    )
    write_json(
        output_dir / "compat_order.json",
        {
            "schema_version": SCHEMA_VERSION,
            "ordered_inputs": ordered_inputs,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": manifest_relative.as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "selection": {
            "policy": "per-domain exact-diff tight-side quantiles",
            "eligible_tasks": len(candidates),
            "per_domain": args.per_domain,
            "selected_tasks": selected_ids,
            "selected_domain_counts": dict(
                Counter(str(row["domain"]) for row in selected)
            ),
        },
        "protocol": {
            "oracle_region": "minimum square enclosing exact RGB-difference bbox",
            "field_of_view_factors": factors,
            "control": "farthest same-size region with zero exact-diff pixels",
            "output_size": [args.output_size, args.output_size],
            "interpolation": "Pillow bicubic",
            "encoding": {
                "format": "JPEG",
                "quality": args.jpeg_quality,
                "subsampling": 0,
                "metadata": "none",
            },
        },
        "images": len(result_rows),
        "arms": dict(Counter(row["region_kind"] for row in result_rows)),
        "crop_factors": dict(Counter(str(row["crop_factor"]) for row in result_rows)),
    }
    write_json(output_dir / "summary.json", summary)
    contact_sheet(
        output_dir / "contact_sheet.jpg",
        selected_ids,
        result_rows,
        repo_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
