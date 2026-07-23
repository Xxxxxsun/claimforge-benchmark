#!/usr/bin/env python3
"""Build the frozen paired mouse input set used by open-source baselines.

Both variants of every pair are decoded and re-encoded with identical JPEG
settings. The localization target is computed before that re-encoding, from
the decoded source image and the lossless spliced PNG, so container format and
JPEG block propagation do not become part of the ground-truth edit region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageChops, ImageOps, features

from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


SCHEMA_VERSION = "claimforge_mouse_canonical_v1"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/opensource/mouse_canonical_v1")


def _resolved_repo_path(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing input: {relative_path}")
    return path


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


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


def _canonicalize(image: Image.Image, destination: Path, quality: int) -> None:
    _atomic_save_image(
        image,
        destination,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )
    with Image.open(destination) as reopened:
        if reopened.format != "JPEG" or reopened.mode != "RGB":
            raise ValueError(f"invalid canonical JPEG: {destination}")
        if reopened.size != image.size:
            raise ValueError(
                f"canonical size changed for {destination}: "
                f"{reopened.size} != {image.size}"
            )
        if reopened.getexif():
            raise ValueError(f"canonical JPEG still contains EXIF: {destination}")


def _exact_diff_mask(
    source: Image.Image,
    forged: Image.Image,
    threshold: int,
) -> Image.Image:
    if source.size != forged.size:
        raise ValueError(f"pair size mismatch: {source.size} != {forged.size}")
    red, green, blue = ImageChops.difference(source, forged).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    return maximum.point(
        lambda value: 255 if value > threshold else 0,
        mode="L",
    )


def _mask_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return int(sum(histogram[1:]))


def _box_pixels(mask: Image.Image, box: list[int]) -> int:
    if len(box) != 4:
        return 0
    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = mask.size
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0
    return _mask_pixels(mask.crop((x1, y1, x2, y2)))


def _validated_boxes(
    task_id: str,
    edit_box: list[int],
    context_box: list[int],
    image_size: tuple[int, int],
) -> None:
    if len(edit_box) != 4 or len(context_box) != 4:
        raise ValueError(f"{task_id}: expected four-coordinate edit/context boxes")
    ex1, ey1, ex2, ey2 = edit_box
    cx1, cy1, cx2, cy2 = context_box
    width, height = image_size
    if not (0 <= cx1 < cx2 <= width and 0 <= cy1 < cy2 <= height):
        raise ValueError(f"{task_id}: context box is outside image bounds")
    if not (cx1 <= ex1 < ex2 <= cx2 and cy1 <= ey1 < ey2 <= cy2):
        raise ValueError(f"{task_id}: edit box is not contained by context box")


def _sample_id(dataset_id: str, task_id: str, kind: str) -> str:
    payload = f"{dataset_id}\0{task_id}\0{kind}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _load_records(review_path: Path) -> dict[str, dict[str, Any]]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in review.get("records", [])
        if record.get("status") == "good" and record.get("candidates") == "mouse"
    ]
    by_id = {str(record["task_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate task_id among reviewed good mouse records")
    return by_id


def _ordered_task_ids(
    order_manifest_path: Path,
    records: dict[str, dict[str, Any]],
) -> list[str]:
    ordering = json.loads(order_manifest_path.read_text(encoding="utf-8"))
    ordered: list[str] = []
    for row in ordering.get("ordered_inputs", []):
        task_id = str(row.get("task_id", ""))
        if task_id in records and task_id not in ordered:
            ordered.append(task_id)
    missing = sorted(set(records) - set(ordered))
    if missing:
        raise ValueError(
            f"ordering manifest omits {len(missing)} reviewed tasks; first={missing[0]}"
        )
    return ordered


def build_dataset(
    *,
    repo_root: Path,
    review_path: Path,
    order_manifest_path: Path,
    output_dir: Path,
    quality: int = 95,
    diff_threshold: int = 0,
    expected_pairs: int = 275,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    review_path = review_path.resolve()
    order_manifest_path = order_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if quality != 95:
        raise ValueError("canonical v1 requires JPEG quality=95")
    if diff_threshold != 0:
        raise ValueError("canonical v1 requires an exact diff threshold of 0")

    records = _load_records(review_path)
    ordered_ids = _ordered_task_ids(order_manifest_path, records)
    if expected_pairs >= 0 and len(ordered_ids) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} good mouse pairs, found {len(ordered_ids)}"
        )

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "claimforge-mouse-good275-canonical-jpeg-q95-v1"
    input_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    total_gt_pixels = 0
    total_image_pixels = 0
    outside_context_pixels = 0

    for pair_rank, task_id in enumerate(ordered_ids):
        record = records[task_id]
        domain = task_id.split("_", 1)[0]
        domain_counts[domain] += 1

        source_relative = str(record["source_image"])
        forged_relative = str(record["spliced_image"])
        source_path = _resolved_repo_path(repo_root, source_relative)
        forged_path = _resolved_repo_path(repo_root, forged_relative)
        source = _load_rgb(source_path)
        forged = _load_rgb(forged_path)
        if source.size != forged.size:
            raise ValueError(
                f"{task_id}: decoded pair size mismatch "
                f"{source.size} != {forged.size}"
            )
        declared_size = record.get("image_size")
        if (
            isinstance(declared_size, list)
            and len(declared_size) == 2
            and tuple(int(value) for value in declared_size) != source.size
        ):
            raise ValueError(
                f"{task_id}: declared size {declared_size} != decoded {source.size}"
            )

        diff = _exact_diff_mask(source, forged, diff_threshold)
        gt_pixels = _mask_pixels(diff)
        if gt_pixels == 0:
            raise ValueError(f"{task_id}: exact-difference mask is empty")
        edit_box = [int(value) for value in record["edit_region_xyxy"]]
        context_box = [int(value) for value in record["context_region_xyxy"]]
        _validated_boxes(task_id, edit_box, context_box, source.size)
        inside_context = _box_pixels(diff, context_box)
        outside_context = gt_pixels - inside_context
        if outside_context:
            raise ValueError(
                f"{task_id}: {outside_context} exact-diff pixels fall outside "
                "the reviewed context box"
            )
        total_gt_pixels += gt_pixels
        total_image_pixels += source.width * source.height
        outside_context_pixels += outside_context

        mask_name = f"{_sample_id(dataset_id, task_id, 'mask')}.png"
        mask_path = mask_dir / mask_name
        _atomic_save_image(diff, mask_path, format="PNG", optimize=False)

        variants: dict[str, dict[str, Any]] = {}
        for kind, label, image, raw_path, raw_relative in (
            ("real", 0, source, source_path, source_relative),
            ("forged", 1, forged, forged_path, forged_relative),
        ):
            sample_id = _sample_id(dataset_id, task_id, kind)
            canonical_path = image_dir / f"{sample_id}.jpg"
            _canonicalize(image, canonical_path, quality)
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
            input_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "rank": len(input_rows),
                    "pair_rank": pair_rank,
                    "task_id": task_id,
                    "domain": domain,
                    "candidate": "mouse",
                    "width": source.width,
                    "height": source.height,
                    "edit_region_xyxy": edit_box,
                    "context_region_xyxy": context_box,
                    "gt_mask_kind": "all_zero" if kind == "real" else "exact_diff",
                    "gt_mask_path": (
                        None if kind == "real" else repo_relative(mask_path, repo_root)
                    ),
                    "gt_mask_sha256": (
                        None if kind == "real" else sha256_file(mask_path)
                    ),
                    "gt_positive_pixels": 0 if kind == "real" else gt_pixels,
                    **variant,
                }
            )

        pair_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "pair_rank": pair_rank,
                "task_id": task_id,
                "domain": domain,
                "candidate": "mouse",
                "width": source.width,
                "height": source.height,
                "edit_region_xyxy": edit_box,
                "context_region_xyxy": context_box,
                "gt_mask_path": repo_relative(mask_path, repo_root),
                "gt_mask_sha256": sha256_file(mask_path),
                "gt_positive_pixels": gt_pixels,
                "gt_fraction": gt_pixels / (source.width * source.height),
                "gt_bbox_xyxy": list(diff.getbbox()) if diff.getbbox() else None,
                "gt_pixels_outside_context": outside_context,
                "real": variants["real"],
                "forged": variants["forged"],
            }
        )

    inputs_path = output_dir / "inputs.jsonl"
    pairs_path = output_dir / "pairs.jsonl"
    atomic_write_jsonl(inputs_path, input_rows)
    atomic_write_jsonl(pairs_path, pair_rows)

    deterministic_contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_review_sha256": sha256_file(review_path),
        "source_order_manifest_sha256": sha256_file(order_manifest_path),
        "jpeg": {
            "quality": quality,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "encoder": {
                "pillow": PIL.__version__,
                "libjpeg": features.version_codec("jpg"),
            },
        },
        "gt_mask": {
            "space": "decoded_pre_canonicalization_rgb",
            "rule": "max_abs_rgb_difference_gt_threshold",
            "threshold": diff_threshold,
        },
        "inputs_sha256": sha256_file(inputs_path),
        "pairs_sha256": sha256_file(pairs_path),
    }
    manifest = {
        **deterministic_contract,
        "contract_sha256": hashlib.sha256(
            stable_json(deterministic_contract).encode("utf-8")
        ).hexdigest(),
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "source_review": repo_relative(review_path, repo_root),
        "source_order_manifest": repo_relative(order_manifest_path, repo_root),
        "inputs_path": repo_relative(inputs_path, repo_root),
        "pairs_path": repo_relative(pairs_path, repo_root),
        "pairs": len(pair_rows),
        "images": len(input_rows),
        "domains": dict(sorted(domain_counts.items())),
        "gt": {
            "positive_pixels": total_gt_pixels,
            "image_pixels": total_image_pixels,
            "mean_fraction": (
                sum(float(row["gt_fraction"]) for row in pair_rows) / len(pair_rows)
                if pair_rows
                else None
            ),
            "pixels_outside_context": outside_context_pixels,
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--order-manifest", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--diff-threshold", type=int, default=0)
    parser.add_argument("--expected-pairs", type=int, default=275)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    def anchored(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    manifest = build_dataset(
        repo_root=repo_root,
        review_path=anchored(args.review),
        order_manifest_path=anchored(args.order_manifest),
        output_dir=anchored(args.output_dir),
        quality=args.quality,
        diff_threshold=args.diff_threshold,
        expected_pairs=args.expected_pairs,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
