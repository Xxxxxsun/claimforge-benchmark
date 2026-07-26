#!/usr/bin/env python3
"""Build paired canonical inputs for the final cat/trash-can benchmark sets.

The final sets combine several splice pipelines, so their localization masks
must follow the provenance of the selected image:

* SAM3 and Hysteresis-SAM3 selections reuse the binary mask emitted by the
  selected splice run.
* trash-can Hysteresis-Distance masks are deterministically reconstructed from
  the recorded generation inputs and parameters.
* relabel variants, whose intermediate segmentation folders are not shipped,
  use the exact decoded source/final difference clipped to the reviewed final
  context box.

Both real and forged images are decoded and encoded with identical JPEG-Q95
settings before inference. This prevents PNG/JPEG container format from being
a class cue while preserving the final benchmark image content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageChops, features

from compose_spliced_full import hysteresis_object_mask
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


SCHEMA_VERSION = "claimforge_final_canonical_v1"
DEFAULT_OUTPUT_ROOT = Path("outputs/our_defense/canonical")

CATEGORY_PATHS = {
    "cat": {
        "selection": Path("annotations/claimforge_cat_final_251_selections.json"),
        "materialized": Path(
            "spliced_final/claimforge_cat_selected_251_20260725/manifest.jsonl"
        ),
        "tasks": Path("annotations/cat_generation_tasks.jsonl"),
        "sam3": Path(
            "results/segmentation/fal_sam3_cat_native_style_v2_full272_20260723/"
            "splice_results.jsonl"
        ),
        "hysteresis": Path(
            "results/segmentation/"
            "hysteresis_sam3_v2_cat_native_style_v2_full272_20260724/"
            "results.jsonl"
        ),
        "expected": 251,
    },
    "trash_can": {
        "selection": Path(
            "annotations/claimforge_trash_can_final_250_selections.json"
        ),
        "materialized": Path(
            "spliced_final/claimforge_trash_can_selected_250_20260725/"
            "manifest.jsonl"
        ),
        "sam3": Path(
            "results/segmentation/mlx_sam3_trash_can_full260_20260724/"
            "splice_results.jsonl"
        ),
        "hysteresis": Path(
            "spliced_full/"
            "hunyuan_image3_distil_trash_can_260_hysteresis_distance_20260724/"
            "manifest.jsonl"
        ),
        "expected": 250,
    },
}


@dataclass(frozen=True)
class Provenance:
    source_relative: str
    forged_relative: str
    context_box: list[int]
    edit_box: list[int]
    context_mask: Image.Image
    mask_source: str
    selection_method: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_task(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label}: row without task_id")
        if task_id in result:
            raise ValueError(f"{label}: duplicate task_id {task_id}")
        result[task_id] = row
    return result


def _repo_file(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing input: {relative}")
    return path


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _load_binary(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        gray = opened.convert("L")
        return gray.point(lambda value: 255 if value > 0 else 0, mode="L")


def _atomic_save_image(image: Image.Image, path: Path, **save_args: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, **save_args)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonicalize(image: Image.Image, destination: Path) -> None:
    _atomic_save_image(
        image,
        destination,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    with Image.open(destination) as reopened:
        if reopened.format != "JPEG" or reopened.mode != "RGB":
            raise ValueError(f"invalid canonical JPEG: {destination}")
        if reopened.size != image.size:
            raise ValueError(f"canonical geometry changed: {destination}")
        if reopened.getexif():
            raise ValueError(f"canonical JPEG contains EXIF: {destination}")


def _sample_id(dataset_id: str, task_id: str, kind: str) -> str:
    payload = f"{dataset_id}\0{task_id}\0{kind}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _validate_box(
    task_id: str,
    box: list[int],
    image_size: tuple[int, int],
    label: str,
) -> list[int]:
    if len(box) != 4:
        raise ValueError(f"{task_id}: invalid {label} box")
    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{task_id}: {label} box outside image: {box}")
    return [x1, y1, x2, y2]


def _mask_pixels(mask: Image.Image) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=np.uint8)))


def _context_to_full(
    task_id: str,
    context_mask: Image.Image,
    context_box: list[int],
    image_size: tuple[int, int],
) -> Image.Image:
    x1, y1, x2, y2 = _validate_box(
        task_id, context_box, image_size, "context"
    )
    expected = (x2 - x1, y2 - y1)
    if context_mask.size != expected:
        raise ValueError(
            f"{task_id}: context mask {context_mask.size} != {expected}"
        )
    full = Image.new("L", image_size, 0)
    full.paste(context_mask, (x1, y1))
    return full


def _observed_context_mask(
    source: Image.Image,
    forged: Image.Image,
    context_box: list[int],
) -> Image.Image:
    x1, y1, x2, y2 = context_box
    source_crop = source.crop((x1, y1, x2, y2))
    forged_crop = forged.crop((x1, y1, x2, y2))
    red, green, blue = ImageChops.difference(source_crop, forged_crop).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    return maximum.point(lambda value: 255 if value > 0 else 0, mode="L")


def _trash_task_context_index(
    repo_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted((repo_root / "annotations").glob(
        "trash_can_generation_tasks*.jsonl"
    )):
        for row in _read_jsonl(path):
            reference = row.get("context_crop") or row.get("input_context_crop")
            if reference:
                grouped.setdefault(
                    (str(row["task_id"]), str(reference)), []
                ).append(row)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        signatures = {
            (
                str(row["source_image"]),
                tuple(int(value) for value in row["context_region_xyxy"]),
                tuple(int(value) for value in row["edit_region_xyxy"]),
            )
            for row in rows
        }
        if len(signatures) != 1:
            raise ValueError(f"conflicting task geometry for {key}")
        index[key] = rows[0]
    return index


def _cat_provenance(
    *,
    repo_root: Path,
    selection: dict[str, Any],
    materialized: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    sam3: dict[str, dict[str, Any]],
    hysteresis: dict[str, dict[str, Any]],
) -> Provenance:
    task_id = str(selection["task_id"])
    method = str(selection["selection"])
    forged = str(materialized["image"])
    if method == "sam3_local_diff":
        task = tasks[task_id]
        result = sam3[task_id]
        mask_path = str(result["hybrid"]["path"])
        return Provenance(
            source_relative=str(task["source_image"]),
            forged_relative=forged,
            context_box=[int(value) for value in task["context_region_xyxy"]],
            edit_box=[int(value) for value in task["edit_region_xyxy"]],
            context_mask=_load_binary(_repo_file(repo_root, mask_path)),
            mask_source=mask_path,
            selection_method=method,
        )
    if method == "hysteresis_sam3_v2":
        result = hysteresis[task_id]
        mask_path = str(result["mask"])
        return Provenance(
            source_relative=str(result["source_image"]),
            forged_relative=forged,
            context_box=[int(value) for value in result["context_region_xyxy"]],
            edit_box=[int(value) for value in result["edit_region_xyxy"]],
            context_mask=_load_binary(_repo_file(repo_root, mask_path)),
            mask_source=mask_path,
            selection_method=method,
        )
    if method.startswith("variant_"):
        required = ("source_image", "context_region_xyxy", "edit_region_xyxy")
        if any(key not in selection for key in required):
            raise ValueError(f"{task_id}: relabel selection lacks provenance")
        source_path = _repo_file(repo_root, str(selection["source_image"]))
        forged_path = _repo_file(repo_root, forged)
        source = _load_rgb(source_path)
        forged_image = _load_rgb(forged_path)
        context = [int(value) for value in selection["context_region_xyxy"]]
        return Provenance(
            source_relative=str(selection["source_image"]),
            forged_relative=forged,
            context_box=context,
            edit_box=[int(value) for value in selection["edit_region_xyxy"]],
            context_mask=_observed_context_mask(source, forged_image, context),
            mask_source="decoded_exact_diff_inside_final_reviewed_context",
            selection_method=method,
        )
    raise ValueError(f"{task_id}: unsupported cat selection {method}")


def _trash_hysteresis_mask(
    repo_root: Path,
    task_id: str,
    row: dict[str, Any],
) -> tuple[Image.Image, str]:
    reference = _load_rgb(_repo_file(repo_root, str(row["mask_reference"])))
    generated = _load_rgb(_repo_file(repo_root, str(row["generated_crop"])))
    context_box = [int(value) for value in row["context_region_xyxy"]]
    edit_box = [int(value) for value in row["edit_region_xyxy"]]
    x1, y1, _, _ = context_box
    edit_in_context = [
        edit_box[0] - x1,
        edit_box[1] - y1,
        edit_box[2] - x1,
        edit_box[3] - y1,
    ]
    mask, stats = hysteresis_object_mask(
        reference,
        generated,
        edit_in_context,
        float(row["hysteresis_low_threshold"]),
        float(row["hysteresis_high_threshold"]),
        0.0,
        close_iterations=int(row["hysteresis_close_iterations"]),
        grow_iterations=int(row["hysteresis_grow_iterations"]),
        reach_ratio=float(row["hysteresis_reach_ratio"]),
        far_thr=float(row["hysteresis_far_threshold"]),
        distance_power=float(row["hysteresis_distance_power"]),
        auto_expand_ratio=float(row["hysteresis_auto_expand_ratio"]),
        auto_expand_max_growth=float(
            row["hysteresis_auto_expand_max_growth"]
        ),
    )
    binary = mask.point(lambda value: 255 if value > 0 else 0, mode="L")
    expected = int(row["hysteresis_stats"]["mask_pixels"])
    actual = _mask_pixels(binary)
    relative_delta = abs(actual - expected) / max(expected, 1)
    if relative_delta > 0.20:
        raise ValueError(
            f"{task_id}: reconstructed mask pixels {actual} != {expected}; "
            f"relative delta={relative_delta:.3f}; "
            f"recomputed stats={stats.get('mask_pixels')}"
        )
    validation = (
        "reconstructed_hysteresis_distance_from_recorded_parameters"
        f":recorded_pixels={expected}:recomputed_pixels={actual}"
    )
    return binary, validation


def _trash_provenance(
    *,
    repo_root: Path,
    selection: dict[str, Any],
    materialized: dict[str, Any],
    sam3: dict[str, dict[str, Any]],
    hysteresis: dict[str, dict[str, Any]],
    context_index: dict[tuple[str, str], dict[str, Any]],
) -> Provenance:
    task_id = str(selection["task_id"])
    method = str(selection["selection"])
    forged = str(materialized["image"])
    if method == "hysteresis_distance":
        row = hysteresis[task_id]
        mask, mask_source = _trash_hysteresis_mask(repo_root, task_id, row)
        return Provenance(
            source_relative=str(row["source_image"]),
            forged_relative=forged,
            context_box=[int(value) for value in row["context_region_xyxy"]],
            edit_box=[int(value) for value in row["edit_region_xyxy"]],
            context_mask=mask,
            mask_source=mask_source,
            selection_method=method,
        )
    if method == "sam3_semantic_hysteresis":
        result = sam3[task_id]
        reference = str(result["difference_reference"])
        task = context_index.get((task_id, reference))
        if task is None:
            raise ValueError(f"{task_id}: no task geometry for {reference}")
        mask_path = str(result["hybrid"]["path"])
        return Provenance(
            source_relative=str(task["source_image"]),
            forged_relative=forged,
            context_box=[int(value) for value in task["context_region_xyxy"]],
            edit_box=[int(value) for value in task["edit_region_xyxy"]],
            context_mask=_load_binary(_repo_file(repo_root, mask_path)),
            mask_source=mask_path,
            selection_method=method,
        )
    if method.startswith("variant_"):
        required = ("source_image", "context_region_xyxy", "edit_region_xyxy")
        if any(key not in selection for key in required):
            raise ValueError(f"{task_id}: relabel selection lacks provenance")
        source_path = _repo_file(repo_root, str(selection["source_image"]))
        forged_path = _repo_file(repo_root, forged)
        source = _load_rgb(source_path)
        forged_image = _load_rgb(forged_path)
        context = [int(value) for value in selection["context_region_xyxy"]]
        return Provenance(
            source_relative=str(selection["source_image"]),
            forged_relative=forged,
            context_box=context,
            edit_box=[int(value) for value in selection["edit_region_xyxy"]],
            context_mask=_observed_context_mask(source, forged_image, context),
            mask_source="decoded_exact_diff_inside_final_reviewed_context",
            selection_method=method,
        )
    raise ValueError(f"{task_id}: unsupported trash-can selection {method}")


def build_dataset(
    *,
    repo_root: Path,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    config = CATEGORY_PATHS[category]
    selection_path = repo_root / config["selection"]
    materialized_path = repo_root / config["materialized"]
    selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
    selections = list(selection_doc["selections"])
    materialized = _by_task(_read_jsonl(materialized_path), "materialized")
    if len(selections) != int(config["expected"]):
        raise ValueError(
            f"{category}: expected {config['expected']} selections, "
            f"found {len(selections)}"
        )
    if {str(row["task_id"]) for row in selections} != set(materialized):
        raise ValueError(f"{category}: selection/materialized task mismatch")

    sam3 = _by_task(_read_jsonl(repo_root / config["sam3"]), "sam3")
    hysteresis = _by_task(
        _read_jsonl(repo_root / config["hysteresis"]), "hysteresis"
    )
    tasks = (
        _by_task(_read_jsonl(repo_root / config["tasks"]), "tasks")
        if category == "cat"
        else {}
    )
    context_index = (
        _trash_task_context_index(repo_root) if category == "trash_can" else {}
    )

    dataset_id = (
        f"claimforge-{category.replace('_', '-')}-final-"
        f"{config['expected']}-canonical-jpeg-q95-v1"
    )
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    inputs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    mask_source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    for pair_rank, selection in enumerate(selections):
        task_id = str(selection["task_id"])
        if category == "cat":
            provenance = _cat_provenance(
                repo_root=repo_root,
                selection=selection,
                materialized=materialized[task_id],
                tasks=tasks,
                sam3=sam3,
                hysteresis=hysteresis,
            )
        else:
            provenance = _trash_provenance(
                repo_root=repo_root,
                selection=selection,
                materialized=materialized[task_id],
                sam3=sam3,
                hysteresis=hysteresis,
                context_index=context_index,
            )
        source_path = _repo_file(repo_root, provenance.source_relative)
        forged_path = _repo_file(repo_root, provenance.forged_relative)
        source = _load_rgb(source_path)
        forged = _load_rgb(forged_path)
        if source.size != forged.size:
            raise ValueError(
                f"{task_id}: source/forged size {source.size} != {forged.size}"
            )
        context = _validate_box(
            task_id, provenance.context_box, source.size, "context"
        )
        edit = _validate_box(task_id, provenance.edit_box, source.size, "edit")
        if not (
            context[0] <= edit[0] < edit[2] <= context[2]
            and context[1] <= edit[1] < edit[3] <= context[3]
        ):
            raise ValueError(f"{task_id}: edit box is outside context")
        full_mask = _context_to_full(
            task_id, provenance.context_mask, context, source.size
        )
        gt_pixels = _mask_pixels(full_mask)
        if gt_pixels == 0:
            raise ValueError(f"{task_id}: empty GT mask")
        changed = np.any(
            np.asarray(source, dtype=np.uint8)
            != np.asarray(forged, dtype=np.uint8),
            axis=2,
        )
        overlap = int(
            np.count_nonzero(
                changed & (np.asarray(full_mask, dtype=np.uint8) > 0)
            )
        )
        if overlap == 0:
            raise ValueError(f"{task_id}: GT has no overlap with changed pixels")

        mask_path = mask_dir / f"{_sample_id(dataset_id, task_id, 'mask')}.png"
        _atomic_save_image(full_mask, mask_path, format="PNG", optimize=False)
        variants: dict[str, dict[str, Any]] = {}
        for kind, label, image, raw_path, raw_relative in (
            ("real", 0, source, source_path, provenance.source_relative),
            ("forged", 1, forged, forged_path, provenance.forged_relative),
        ):
            sample_id = _sample_id(dataset_id, task_id, kind)
            canonical_path = image_dir / f"{sample_id}.jpg"
            _canonicalize(image, canonical_path)
            variant = {
                "sample_id": sample_id,
                "kind": kind,
                "label": label,
                "raw_path": raw_relative,
                "raw_sha256": sha256_file(raw_path),
                "canonical_path": repo_relative(canonical_path, repo_root),
                "canonical_sha256": sha256_file(canonical_path),
                "canonical_bytes": canonical_path.stat().st_size,
            }
            variants[kind] = variant
            inputs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "rank": len(inputs),
                    "pair_rank": pair_rank,
                    "task_id": task_id,
                    "domain": str(selection["domain"]),
                    "candidate": category,
                    "selection_method": provenance.selection_method,
                    "width": source.width,
                    "height": source.height,
                    "edit_region_xyxy": edit,
                    "context_region_xyxy": context,
                    "gt_mask_kind": "all_zero" if kind == "real" else "provenance",
                    "gt_mask_path": (
                        None
                        if kind == "real"
                        else repo_relative(mask_path, repo_root)
                    ),
                    "gt_positive_pixels": 0 if kind == "real" else gt_pixels,
                    **variant,
                }
            )

        bbox = full_mask.getbbox()
        pair = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "pair_rank": pair_rank,
            "task_id": task_id,
            "domain": str(selection["domain"]),
            "candidate": category,
            "selection_method": provenance.selection_method,
            "width": source.width,
            "height": source.height,
            "edit_region_xyxy": edit,
            "context_region_xyxy": context,
            "gt_mask_source": provenance.mask_source,
            "gt_mask_path": repo_relative(mask_path, repo_root),
            "gt_mask_sha256": sha256_file(mask_path),
            "gt_positive_pixels": gt_pixels,
            "gt_fraction": gt_pixels / (source.width * source.height),
            "gt_bbox_xyxy": list(bbox) if bbox else None,
            "gt_pixels_outside_context": 0,
            "observed_changed_pixels_in_gt": overlap,
            "real": variants["real"],
            "forged": variants["forged"],
        }
        pairs.append(pair)
        method_counts[provenance.selection_method] += 1
        mask_source_counts[provenance.mask_source] += 1
        domain_counts[str(selection["domain"])] += 1

    inputs_path = output_dir / "inputs.jsonl"
    pairs_path = output_dir / "pairs.jsonl"
    atomic_write_jsonl(inputs_path, inputs)
    atomic_write_jsonl(pairs_path, pairs)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "category": category,
        "source_selection_sha256": sha256_file(selection_path),
        "source_materialized_manifest_sha256": sha256_file(materialized_path),
        "jpeg": {
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "encoder": {
                "pillow": PIL.__version__,
                "libjpeg": features.version_codec("jpg"),
            },
        },
        "inputs_sha256": sha256_file(inputs_path),
        "pairs_sha256": sha256_file(pairs_path),
    }
    manifest = {
        **contract,
        "contract_sha256": hashlib.sha256(
            stable_json(contract).encode()
        ).hexdigest(),
        "created_at": utc_now(),
        "source_selection": repo_relative(selection_path, repo_root),
        "source_materialized_manifest": repo_relative(
            materialized_path, repo_root
        ),
        "inputs_path": repo_relative(inputs_path, repo_root),
        "pairs_path": repo_relative(pairs_path, repo_root),
        "pairs": len(pairs),
        "images": len(inputs),
        "domains": dict(sorted(domain_counts.items())),
        "selection_methods": dict(sorted(method_counts.items())),
        "mask_sources": dict(sorted(mask_source_counts.items())),
        "gt": {
            "positive_pixels": sum(
                int(pair["gt_positive_pixels"]) for pair in pairs
            ),
            "mean_fraction": float(
                np.mean([float(pair["gt_fraction"]) for pair in pairs])
            ),
            "median_fraction": float(
                np.median([float(pair["gt_fraction"]) for pair in pairs])
            ),
            "empty_masks": 0,
            "pixels_outside_context": 0,
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--category", choices=tuple(CATEGORY_PATHS), required=True
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{args.category}_final_v1"
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    manifest = build_dataset(
        repo_root=repo_root,
        category=args.category,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
