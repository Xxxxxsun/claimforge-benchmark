#!/usr/bin/env python3
"""Build the formal paired crop-scale experiment for commercial detectors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from eval.commercial.build_mouse_crop_scale_probe import (
    choose_tasks,
    load_pair,
    read_jsonl,
    real_control_box,
    save_canonical_jpeg,
    sha256_file,
    square_box,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "claimforge_mouse_crop_scale_formal_v1"
DEFAULT_MANIFEST = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path("results/analysis/mouse_crop_scale_formal_v1")
DEFAULT_FACTORS = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
ARMS = ("suspicious", "real_control")


def factor_slug(factor: float) -> str:
    value = f"{factor:g}".replace(".", "p")
    return value.zfill(2) if "p" not in value else value


def centered_square(
    outer_box: tuple[int, int, int, int],
    side: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    center_x_twice = outer_box[0] + outer_box[2]
    center_y_twice = outer_box[1] + outer_box[3]
    return square_box(
        (
            center_x_twice // 2,
            center_y_twice // 2,
            (center_x_twice + 1) // 2,
            (center_y_twice + 1) // 2,
        ),
        side,
        image_size,
    )


def interleave_domains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    domains = sorted({str(row["domain"]) for row in rows})
    by_domain = {
        domain: sorted(
            (row for row in rows if row["domain"] == domain),
            key=lambda row: (row["selection_quantile"], row["task_id"]),
        )
        for domain in domains
    }
    output: list[dict[str, Any]] = []
    for rank in range(max(len(values) for values in by_domain.values())):
        for domain in domains:
            if rank < len(by_domain[domain]):
                output.append(by_domain[domain][rank])
    return output


def build_contact_sheet(
    output_path: Path,
    task_ids: list[str],
    rows: list[dict[str, Any]],
    repo_root: Path,
    conditions: list[tuple[str, float, str]],
    title: str,
    tile: int = 104,
) -> None:
    label_width = 210
    header_height = 48
    row_height = tile + 25
    width = label_width + len(conditions) * tile
    height = header_height + len(task_ids) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), title, fill="black")
    for column, (render_mode, factor, arm) in enumerate(conditions):
        label = f"{factor:g}x {arm[:4]}"
        if render_mode == "native":
            label = f"native {arm[:4]}"
        draw.text((label_width + column * tile + 3, 27), label, fill="black")

    by_key = {
        (
            str(row["task_id"]),
            str(row["render_mode"]),
            float(row["crop_factor"]),
            str(row["region_kind"]),
        ): row
        for row in rows
    }
    for row_index, task_id in enumerate(task_ids):
        top = header_height + row_index * row_height
        draw.text((8, top + 5), task_id[:31], fill="black")
        sample = next(row for row in rows if row["task_id"] == task_id)
        draw.text((8, top + 21), f"tight={sample['tight_side']} px", fill="black")
        for column, condition in enumerate(conditions):
            row = by_key[(task_id, *condition)]
            with Image.open(repo_root / row["image"]) as opened:
                preview = opened.convert("RGB")
                preview.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            left = label_width + column * tile + (tile - preview.width) // 2
            image_top = top + (tile - preview.height) // 2
            sheet.paste(preview, (left, image_top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, subsampling=0)


def parse_factors(value: str) -> list[float]:
    factors = sorted({float(part) for part in value.split(",") if part.strip()})
    if not factors or factors[0] < 1:
        raise ValueError("factors must be unique numbers >= 1")
    return factors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--factors",
        default=",".join(f"{factor:g}" for factor in DEFAULT_FACTORS),
    )
    parser.add_argument("--per-domain", type=int, default=25)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--contact-tasks", type=int, default=10)
    args = parser.parse_args()

    try:
        factors = parse_factors(args.factors)
    except ValueError as exc:
        parser.error(str(exc))
    if 1.0 not in factors:
        parser.error("--factors must include 1 for the native-size ablation")
    if args.per_domain < 1 or args.output_size < 32:
        parser.error("--per-domain must be positive and --output-size >= 32")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

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

    max_factor = max(factors)
    candidates: list[dict[str, Any]] = []
    for source_row in source_rows:
        source, _, _, bbox = load_pair(repo_root, source_row)
        tight_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        largest_side = int(math.ceil(tight_side * max_factor))
        if largest_side > min(source.size):
            continue
        control_outer = real_control_box(bbox, largest_side, source.size)
        if control_outer is None:
            continue
        candidates.append(
            {
                **source_row,
                "diff_bbox_xyxy": list(bbox),
                "tight_side": tight_side,
                "control_outer_box_xyxy": list(control_outer),
            }
        )

    selected = interleave_domains(choose_tasks(candidates, args.per_domain))
    crop_dir = output_dir / "images"
    generated: dict[tuple[int, str, float, str], dict[str, Any]] = {}
    condition_groups = [("resize512", factor) for factor in factors]
    condition_groups.append(("native", 1.0))

    for task_rank, selected_row in enumerate(selected):
        task_id = str(selected_row["task_id"])
        source, forged, mask, bbox = load_pair(repo_root, selected_row)
        tight_side = int(selected_row["tight_side"])
        control_outer = tuple(int(value) for value in selected_row["control_outer_box_xyxy"])

        for render_mode, factor in condition_groups:
            side = int(math.ceil(tight_side * factor))
            suspect_box = square_box(bbox, side, source.size)
            control_box = centered_square(control_outer, side, source.size)
            suspect_modified = sum(mask.crop(suspect_box).histogram()[1:])
            control_modified = sum(mask.crop(control_box).histogram()[1:])
            if suspect_modified == 0 or control_modified != 0:
                raise ValueError(
                    f"{task_id}: invalid modified pixels at {render_mode}/{factor:g}x"
                )
            if ImageChops.difference(
                source.crop(control_box), forged.crop(control_box)
            ).getbbox() is not None:
                raise ValueError(f"{task_id}: real control is not pixel-identical")

            for arm, input_image, crop_box, modified_pixels in (
                ("suspicious", forged, suspect_box, suspect_modified),
                ("real_control", source, control_box, 0),
            ):
                crop = input_image.crop(crop_box)
                if render_mode == "resize512":
                    rendered = crop.resize(
                        (args.output_size, args.output_size),
                        Image.Resampling.BICUBIC,
                    )
                    output_size = [args.output_size, args.output_size]
                    interpolation = "Pillow bicubic"
                else:
                    rendered = crop
                    output_size = [side, side]
                    interpolation = "none"

                compat_task_id = (
                    f"{task_id}__oracle_{arm}__{render_mode}"
                    f"__field_{factor_slug(factor)}x"
                )
                relative_image = (
                    output_dir.relative_to(repo_root)
                    / "images"
                    / f"{compat_task_id}.jpg"
                )
                absolute_image = repo_root / relative_image
                save_canonical_jpeg(rendered, absolute_image, args.jpeg_quality)
                generated[(task_rank, render_mode, factor, arm)] = {
                    "schema_version": SCHEMA_VERSION,
                    "task_rank": task_rank,
                    "task_id": task_id,
                    "compat_task_id": compat_task_id,
                    "domain": selected_row["domain"],
                    "region_kind": arm,
                    "render_mode": render_mode,
                    "crop_factor": factor,
                    "selection_quantile": selected_row["selection_quantile"],
                    "source_image": selected_row["source_image"],
                    "forged_image": selected_row["image"],
                    "diff_bbox_xyxy": list(bbox),
                    "tight_side": tight_side,
                    "control_outer_box_xyxy": list(control_outer),
                    "crop_box_xyxy": list(crop_box),
                    "native_crop_size": [side, side],
                    "output_size": output_size,
                    "resize_scale": args.output_size / side
                    if render_mode == "resize512"
                    else 1.0,
                    "interpolation": interpolation,
                    "encoding": {
                        "format": "JPEG",
                        "quality": args.jpeg_quality,
                        "subsampling": 0,
                        "metadata": "none",
                    },
                    "modified_pixels": modified_pixels,
                    "modified_fraction_native": modified_pixels / (side * side),
                    "image": relative_image.as_posix(),
                    "image_sha256": sha256_file(absolute_image),
                    "bytes": absolute_image.stat().st_size,
                }

    result_rows: list[dict[str, Any]] = []
    task_count = len(selected)
    for round_index in range(task_count):
        for condition_index, (render_mode, factor) in enumerate(condition_groups):
            task_rank = (round_index + condition_index) % task_count
            arm_order = ARMS if (round_index + condition_index) % 2 == 0 else ARMS[::-1]
            for arm in arm_order:
                row = dict(generated[(task_rank, render_mode, factor, arm)])
                row["rank"] = len(result_rows)
                result_rows.append(row)

    if len(result_rows) != task_count * len(condition_groups) * len(ARMS):
        raise AssertionError("formal manifest cardinality mismatch")
    if len({row["compat_task_id"] for row in result_rows}) != len(result_rows):
        raise AssertionError("duplicate compatibility task IDs")

    review_records: list[dict[str, Any]] = []
    ordered_inputs: list[dict[str, Any]] = []
    for row in result_rows:
        width, height = row["output_size"]
        review_records.append(
            {
                "task_id": row["compat_task_id"],
                "status": "good",
                "candidates": "mouse",
                "source_image": row["image"],
                "spliced_image": row["image"],
                "image_size": [width, height],
                "edit_region_xyxy": [0, 0, width, height],
                "context_region_xyxy": [0, 0, width, height],
                "crop_probe": {
                    "base_task_id": row["task_id"],
                    "region_kind": row["region_kind"],
                    "render_mode": row["render_mode"],
                    "crop_factor": row["crop_factor"],
                },
            }
        )
        ordered_inputs.append(
            {"rank": len(ordered_inputs), "task_id": row["compat_task_id"]}
        )

    write_jsonl(output_dir / "manifest.jsonl", result_rows)
    write_json(
        output_dir / "compat_review.json",
        {"schema_version": SCHEMA_VERSION, "records": review_records},
    )
    write_json(
        output_dir / "compat_order.json",
        {"schema_version": SCHEMA_VERSION, "ordered_inputs": ordered_inputs},
    )
    resized_records = [
        record
        for record in review_records
        if record["crop_probe"]["render_mode"] == "resize512"
    ]
    resized_order = [
        {"rank": rank, "task_id": record["task_id"]}
        for rank, record in enumerate(resized_records)
    ]
    write_json(
        output_dir / "compat_review_resize512.json",
        {"schema_version": SCHEMA_VERSION, "records": resized_records},
    )
    write_json(
        output_dir / "compat_order_resize512.json",
        {"schema_version": SCHEMA_VERSION, "ordered_inputs": resized_order},
    )

    selected_ids = [str(row["task_id"]) for row in selected]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": manifest_path.relative_to(repo_root).as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "selection": {
            "policy": "per-domain exact-diff tight-side quantiles",
            "eligibility": (
                f"{max_factor:g}x crop fits and has a same-image unchanged control"
            ),
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
            "real_control": (
                "concentric crops inside the farthest unchanged max-factor square"
            ),
            "main_render": {
                "mode": "resize512",
                "output_size": [args.output_size, args.output_size],
                "interpolation": "Pillow bicubic",
            },
            "native_ablation": {
                "mode": "native",
                "field_of_view_factor": 1.0,
                "interpolation": "none",
            },
            "encoding": {
                "format": "JPEG",
                "quality": args.jpeg_quality,
                "subsampling": 0,
                "metadata": "none",
            },
            "request_order": (
                "cyclic interleave across domains, conditions, and paired arms"
            ),
            "provider_compatibility": {
                "all_conditions": {
                    "review": "compat_review.json",
                    "order": "compat_order.json",
                    "images": len(result_rows),
                },
                "resize512_only": {
                    "review": "compat_review_resize512.json",
                    "order": "compat_order_resize512.json",
                    "images": len(resized_records),
                },
            },
        },
        "tasks": task_count,
        "condition_groups": len(condition_groups),
        "images": len(result_rows),
        "arms": dict(Counter(row["region_kind"] for row in result_rows)),
        "render_modes": dict(Counter(row["render_mode"] for row in result_rows)),
        "crop_factors": dict(Counter(f"{row['crop_factor']:g}" for row in result_rows)),
    }
    write_json(output_dir / "summary.json", summary)

    preview_ids = selected_ids[: max(0, min(args.contact_tasks, task_count))]
    resized_conditions = [
        ("resize512", factor, arm) for factor in factors for arm in ARMS
    ]
    build_contact_sheet(
        output_dir / "contact_sheet_resized.jpg",
        preview_ids,
        result_rows,
        repo_root,
        resized_conditions,
        "Formal crop-scale probe: suspicious vs same-image real controls",
    )
    ablation_conditions = [
        ("resize512", 1.0, arm) for arm in ARMS
    ] + [("native", 1.0, arm) for arm in ARMS]
    build_contact_sheet(
        output_dir / "contact_sheet_native_ablation.jpg",
        preview_ids,
        result_rows,
        repo_root,
        ablation_conditions,
        "One-times crop: resized to 512 vs native-size input",
        tile=160,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
