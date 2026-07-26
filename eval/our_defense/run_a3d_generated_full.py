#!/usr/bin/env python3
"""Run frozen A3D detection on every image listed by generated-full manifests.

Unlike the paired benchmark runner, this entry point does not require a real
counterpart or a localization mask.  It emits only blind image-level detection
scores and proposal diagnostics.  The default preprocessing matches the
cat/trash canonical protocol: JPEG quality 95 with 4:4:4 subsampling.
"""

from __future__ import annotations

import argparse
import gc
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    read_latest_by_id,
    repo_relative,
    utc_now,
)
from eval.opensource.maskclip_metrics import descriptive
from eval.opensource.run_trufor import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    DEFAULT_TRUFOR_ROOT,
    infer_one,
    load_model,
)
from eval.our_defense.run_trufor_a3d import (
    _logit_mean_score,
    _rank_proposals,
)
from eval.our_defense.run_trufor_adaptive_scan import _grid_boxes
from eval.our_defense.run_trufor_adaptive_zoom import (
    _anchored,
    _load_rgb,
    _tensor_from_image,
)


SCHEMA_VERSION = "claimforge_a3d_generated_full_v1"
DEFAULT_THRESHOLD = 0.6353510120379108
DEFAULT_OUTPUT_DIR = Path(
    "results/our_defense/generated_full_images_a3d_20260726"
)
DEFAULT_MANIFESTS = (
    Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_mouse_good275_g5_v1_"
        "20260724/manifest.jsonl"
    ),
    Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_cat_latest272_g5_v1_"
        "20260724/manifest.jsonl"
    ),
    Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_trash_can_latest260_"
        "g5_v1_20260724/manifest.jsonl"
    ),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _canonicalize(
    image: Image.Image,
    quality: int | None,
    subsampling: int,
) -> Image.Image:
    if quality is None:
        return image
    if not 1 <= quality <= 100:
        raise ValueError("jpeg_quality must fall in [1, 100]")
    if subsampling not in {0, 1, 2}:
        raise ValueError("jpeg_subsampling must be one of 0, 1, or 2")
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as opened:
        result = opened.convert("RGB")
    if result.size != image.size:
        raise ValueError("JPEG canonicalization changed image geometry")
    return result


def _category(row: dict[str, Any]) -> str:
    value = str(row.get("object_kind") or row.get("candidate") or "").lower()
    normalized = value.replace("-", "_").replace(" ", "_")
    if normalized in {"mouse", "cat", "trash_can"}:
        return normalized
    raise ValueError(f"unsupported generated-full category: {value!r}")


def _load_entries(
    manifest_paths: list[Path],
    repo_root: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_images: set[Path] = set()
    for manifest_path in manifest_paths:
        rows = read_jsonl(manifest_path)
        latest_by_image: dict[Path, dict[str, Any]] = {}
        for row in rows:
            output_value = row.get("output_image")
            if row.get("status") != "ok" or not isinstance(output_value, str):
                continue
            image_path = _anchored(Path(output_value), repo_root)
            latest_by_image[image_path] = row

        disk_images = {
            path.resolve()
            for path in manifest_path.parent.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        manifest_images = set(latest_by_image)
        if disk_images != manifest_images:
            raise ValueError(
                f"{manifest_path}: manifest/disk image mismatch "
                f"(missing={len(disk_images - manifest_images)}, "
                f"stale={len(manifest_images - disk_images)})"
            )

        for image_path, row in latest_by_image.items():
            if image_path in seen_images:
                raise ValueError(f"duplicate generated image: {image_path}")
            seen_images.add(image_path)
            entries.append(
                {
                    "id": repo_relative(image_path, repo_root),
                    "task_id": str(row["task_id"]),
                    "category": _category(row),
                    "image_path": image_path,
                    "input_source_image": row.get("input_source_image"),
                    "manifest_path": manifest_path,
                    "generation_model": row.get("model"),
                    "generation_seed": row.get("seed"),
                }
            )
    return sorted(
        entries,
        key=lambda item: (
            str(item["category"]),
            str(item["task_id"]),
            str(item["id"]),
        ),
    )


def _score_summary(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"images": len(rows)}
    for label, score_key in (
        ("full", "full_score"),
        ("local", "a3d_score"),
        ("fused", "a3d_fused_score"),
    ):
        values = [float(row[score_key]) for row in rows]
        positives = sum(value >= threshold for value in values)
        result[label] = {
            "score_key": score_key,
            "scores": descriptive(values),
            "threshold": threshold,
            "positive_images": positives,
            "positive_rate": positives / len(values) if values else None,
        }
    result["latency_ms"] = descriptive(
        float(row["total_latency_ms"]) for row in rows
    )
    return result


def _summarize(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    categories = sorted({str(row["category"]) for row in rows})
    return {
        "all": _score_summary(rows, threshold),
        "by_category": {
            category: _score_summary(
                [row for row in rows if row["category"] == category],
                threshold,
            )
            for category in categories
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    manifest_paths = [
        _anchored(path, repo_root)
        for path in (args.manifest or list(DEFAULT_MANIFESTS))
    ]
    output_dir = _anchored(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_id}.jsonl"
    entries = _load_entries(manifest_paths, repo_root)
    if not entries:
        raise ValueError("generated-full manifests contain no images")

    latest = read_latest_by_id(result_path)
    expected_ids = [str(entry["id"]) for entry in entries]
    pending = [row_id for row_id in expected_ids if row_id not in latest]
    print(
        f"A3D generated-full {args.run_id}: {len(entries)} images, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    try:
        if pending:
            model, device = load_model(
                trufor_root=args.trufor_root.resolve(),
                checkpoint_path=args.checkpoint.resolve(),
                device_name=args.device,
            )
            print(f"loaded TruFor on {device}", flush=True)
            completed = len(entries) - len(pending)
            for entry in entries:
                row_id = str(entry["id"])
                if row_id in latest:
                    continue
                image = _canonicalize(
                    _load_rgb(entry["image_path"]),
                    args.jpeg_quality,
                    args.jpeg_subsampling,
                )
                width, height = image.size
                boxes = _grid_boxes(
                    width,
                    height,
                    args.crop_side,
                    args.crop_stride,
                )
                (
                    full_score,
                    full_logit,
                    full_map,
                    full_reliability,
                    full_peak_bytes,
                    full_latency_ms,
                ) = infer_one(model, device, _tensor_from_image(image))
                proposals, selected = _rank_proposals(
                    score_map=full_map,
                    reliability=full_reliability,
                    boxes=boxes,
                    budget=args.proposal_budget,
                )

                crops: list[dict[str, Any]] = []
                crop_latency_ms = 0.0
                peak_bytes = full_peak_bytes
                for rank, proposal in enumerate(selected, start=1):
                    box = tuple(int(value) for value in proposal["box_xyxy"])
                    (
                        crop_score,
                        crop_logit,
                        _crop_map,
                        crop_reliability,
                        crop_peak_bytes,
                        latency_ms,
                    ) = infer_one(
                        model,
                        device,
                        _tensor_from_image(image.crop(box)),
                    )
                    crop_latency_ms += latency_ms
                    peak_bytes = max(peak_bytes, crop_peak_bytes)
                    crops.append(
                        {
                            **proposal,
                            "proposal_rank": rank,
                            "score": crop_score,
                            "score_margin": crop_logit,
                            "reliability_mean": float(
                                np.mean(crop_reliability)
                            ),
                            "latency_ms": latency_ms,
                        }
                    )
                if not crops:
                    raise RuntimeError("A3D selected no crops")

                local_score = max(float(crop["score"]) for crop in crops)
                fused_score = _logit_mean_score(full_score, local_score)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "id": row_id,
                    "run_id": args.run_id,
                    "status": "ok",
                    "task_id": entry["task_id"],
                    "category": entry["category"],
                    "image_path": row_id,
                    "input_source_image": entry["input_source_image"],
                    "manifest_path": repo_relative(
                        entry["manifest_path"],
                        repo_root,
                    ),
                    "generation_model": entry["generation_model"],
                    "generation_seed": entry["generation_seed"],
                    "input_transform": (
                        {
                            "name": "jpeg",
                            "quality": args.jpeg_quality,
                            "subsampling": args.jpeg_subsampling,
                        }
                        if args.jpeg_quality is not None
                        else {"name": "identity"}
                    ),
                    "image_width": width,
                    "image_height": height,
                    "crop_side": args.crop_side,
                    "crop_stride": args.crop_stride,
                    "grid_crops": len(proposals),
                    "proposal_budget": args.proposal_budget,
                    "full_score": full_score,
                    "full_score_margin": full_logit,
                    "a3d_score": local_score,
                    "a3d_fused_score": fused_score,
                    "fixed_threshold": args.classification_threshold,
                    "decision": fused_score >= args.classification_threshold,
                    "proposals": proposals,
                    "selected_crops": crops,
                    "full_latency_ms": full_latency_ms,
                    "crop_latency_ms": crop_latency_ms,
                    "total_latency_ms": full_latency_ms + crop_latency_ms,
                    "peak_cuda_memory_bytes": peak_bytes,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "completed_at": utc_now(),
                }
                append_jsonl(result_path, row)
                latest[row_id] = row
                completed += 1
                print(
                    f"[{completed}/{len(entries)}] {entry['category']} "
                    f"{entry['task_id']}: full={full_score:.4f} "
                    f"local={local_score:.4f} fused={fused_score:.4f} "
                    f"decision={row['decision']} "
                    f"latency={row['total_latency_ms']:.1f}ms",
                    flush=True,
                )
    finally:
        if model is not None:
            del model
            gc.collect()

    physical_rows = read_jsonl(result_path)
    latest_rows = {
        str(row["id"]): row
        for row in physical_rows
        if isinstance(row.get("id"), str)
    }
    missing = [row_id for row_id in expected_ids if row_id not in latest_rows]
    if missing:
        raise RuntimeError(
            f"incomplete generated-full A3D run: {len(missing)} rows missing"
        )
    selected_rows = [latest_rows[row_id] for row_id in expected_ids]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "images": len(selected_rows),
        "manifests": [
            repo_relative(path, repo_root) for path in manifest_paths
        ],
        "crop_side": args.crop_side,
        "crop_stride": args.crop_stride,
        "proposal_budget": args.proposal_budget,
        "input_transform": (
            {
                "name": "jpeg",
                "quality": args.jpeg_quality,
                "subsampling": args.jpeg_subsampling,
            }
            if args.jpeg_quality is not None
            else {"name": "identity"}
        ),
        "classification_threshold": args.classification_threshold,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        **_summarize(selected_rows, args.classification_threshold),
        "completed_at": utc_now(),
    }
    summary_path = output_dir / f"{args.run_id}.summary.json"
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": args.run_id,
                "images": len(selected_rows),
                "summary_path": repo_relative(summary_path, repo_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--trufor-root", type=Path, default=DEFAULT_TRUFOR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-side", type=int, default=512)
    parser.add_argument("--crop-stride", type=int, default=384)
    parser.add_argument("--proposal-budget", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--jpeg-subsampling", type=int, default=0)
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
