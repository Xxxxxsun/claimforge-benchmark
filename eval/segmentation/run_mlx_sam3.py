#!/usr/bin/env python3
"""Run local MLX SAM3 segmentation and reuse the existing hybrid splice logic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from eval.segmentation.run_fal_sam3 import (
    REPO,
    HybridConfig,
    PilotItem,
    append_jsonl,
    load_jsonl,
    load_pilot_candidates,
    materialize_all,
    read_latest,
    relative_to_repo,
    select_pilot_items,
    sha256_json,
    utc_now,
    write_json,
    write_jsonl,
)


def start_length_rle(mask: np.ndarray) -> str:
    """Encode a boolean mask using fal's one-based, row-major start/length form."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.pad(flat.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    starts = changes[0::2]
    lengths = changes[1::2] - starts
    tokens: list[str] = []
    for start, length in zip(starts, lengths):
        tokens.extend((str(int(start) + 1), str(int(length))))
    return " ".join(tokens)


def normalize_masks(raw_masks: Any, shape: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(raw_masks)
    if masks.size == 0:
        return np.zeros((0, *shape), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"unexpected SAM3 mask shape: {masks.shape}")
    normalized: list[np.ndarray] = []
    for raw_mask in masks:
        binary = np.asarray(raw_mask) > 0
        if binary.shape != shape:
            resized = Image.fromarray(binary.astype(np.uint8) * 255).resize(
                (shape[1], shape[0]),
                Image.Resampling.NEAREST,
            )
            binary = np.asarray(resized, dtype=np.uint8) >= 128
        normalized.append(binary)
    return np.stack(normalized)


def local_api_row(
    item: PilotItem,
    result: Any,
    prompt: str,
    model_name: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    shape = (item.crop_size[1], item.crop_size[0])
    masks = normalize_masks(result.masks, shape)
    scores = np.asarray(result.scores, dtype=np.float32).reshape(-1)
    boxes = np.asarray(result.boxes, dtype=np.float32).reshape(-1, 4)
    count = min(len(masks), len(scores), len(boxes))
    masks = masks[:count]
    scores = scores[:count]
    boxes = boxes[:count]
    if not count or not any(mask.any() for mask in masks):
        raise ValueError("SAM3 returned no non-empty mask")
    row_id = f"{item.task_id}__sam3"
    return {
        "schema_version": "mlx_sam3_api_result_v1",
        "id": row_id,
        "task_id": item.task_id,
        "domain": item.domain,
        "endpoint_tag": "sam3",
        "endpoint": model_name,
        "request_id": f"local-{sha256_json([item.input_sha256, prompt, model_name])[:24]}",
        "input_image": item.generated_relative,
        "input_sha256": item.input_sha256,
        "request": {
            "prompt": prompt,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
        },
        "provider_output": {
            "rle": [start_length_rle(mask) for mask in masks],
            "scores": [float(score) for score in scores],
            "boxes": [[float(value) for value in box] for box in boxes],
        },
        "local_inference_seconds": elapsed_seconds,
        "completed_at": utc_now(),
        "status": "ok",
    }


def error_api_row(
    item: PilotItem,
    prompt: str,
    model_name: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "mlx_sam3_api_result_v1",
        "id": f"{item.task_id}__sam3",
        "task_id": item.task_id,
        "domain": item.domain,
        "endpoint_tag": "sam3",
        "endpoint": model_name,
        "input_image": item.generated_relative,
        "input_sha256": item.input_sha256,
        "request": {"prompt": prompt},
        "error_type": type(error).__name__,
        "error_message": str(error),
        "completed_at": utc_now(),
        "status": "error",
    }


def selected_items(
    candidates: Sequence[PilotItem],
    count: int,
    requested_ids: Sequence[str],
) -> list[PilotItem]:
    if requested_ids:
        lookup = {item.task_id: item for item in candidates}
        missing = [task_id for task_id in requested_ids if task_id not in lookup]
        if missing:
            raise ValueError(f"unknown task IDs: {', '.join(missing)}")
        return [lookup[task_id] for task_id in requested_ids]
    if count == len(candidates):
        return list(candidates)
    return select_pilot_items(candidates, count)


def build_review_manifest(
    items: Sequence[PilotItem],
    output_dir: Path,
    api_results_path: Path,
) -> dict[str, int]:
    splice_path = output_dir / "splice_results.jsonl"
    splice_rows = load_jsonl(splice_path) if splice_path.is_file() else []
    splice_by_id = {str(row["task_id"]): row for row in splice_rows}
    api_by_id = read_latest(api_results_path)
    review_rows: list[dict[str, Any]] = []
    ok = 0
    fallback = 0
    for item in items:
        row = splice_by_id.get(item.task_id)
        if row is not None:
            ok += 1
            review_rows.append(
                {
                    **row,
                    "source_image": item.source_relative,
                    "generated_crop": item.generated_relative,
                    "image_size": list(Image.open(item.source_path).size),
                    "context_region_xyxy": list(item.context_box),
                    "edit_region_in_context_xyxy": list(item.edit_box),
                    "segmentation_status": "ok",
                    "status": "ok",
                }
            )
            continue

        fallback += 1
        fallback_path = output_dir / "spliced_hybrid" / "sam3" / f"{item.task_id}.png"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(item.source_path) as source_image:
            source = source_image.convert("RGB")
            source.save(fallback_path)
            image_size = list(source.size)
        api_row = api_by_id.get(f"{item.task_id}__sam3", {})
        review_rows.append(
            {
                "schema_version": "mlx_sam3_review_result_v1",
                "task_id": item.task_id,
                "domain": item.domain,
                "source_image": item.source_relative,
                "generated_crop": item.generated_relative,
                "hybrid_spliced_full": relative_to_repo(fallback_path),
                "image_size": image_size,
                "context_region_xyxy": list(item.context_box),
                "edit_region_in_context_xyxy": list(item.edit_box),
                "outside_context_identical_to_source": True,
                "segmentation_status": "no_mask",
                "segmentation_error": api_row.get("error_message", "materialization failed"),
                "status": "ok",
            }
        )
    write_jsonl(output_dir / "review_manifest.jsonl", review_rows)
    return {"total": len(review_rows), "sam3_ok": ok, "no_mask_fallback": fallback}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="mlx-community/sam3-4bit")
    parser.add_argument("--model-name", default="mlx-community/sam3-4bit")
    parser.add_argument("--candidate", default="trash can")
    parser.add_argument("--prompt", default="trash can")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument(
        "--hybrid-mode",
        choices=("local", "semantic_hysteresis", "semantic_shadow"),
        default="semantic_hysteresis",
    )
    parser.add_argument("--diff-threshold", type=float, default=20.0)
    parser.add_argument("--support-radius", type=int, default=6)
    parser.add_argument("--alpha-feather", type=float, default=1.0)
    parser.add_argument("--edge-diff-threshold", type=float, default=8.0)
    parser.add_argument("--edge-radius", type=int, default=6)
    parser.add_argument("--shadow-feather", type=float, default=3.0)
    parser.add_argument("--far-diff-threshold", type=float, default=40.0)
    parser.add_argument("--max-added-fraction", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (REPO / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_pilot_candidates(
        REPO,
        (REPO / args.base_manifest).resolve(),
        candidate=args.candidate,
        task_manifest_path=(
            (REPO / args.task_manifest).resolve() if args.task_manifest else None
        ),
    )
    items = selected_items(candidates, args.tasks, args.task_id)
    write_jsonl(output_dir / "selection.jsonl", [
        {
            "task_id": item.task_id,
            "domain": item.domain,
            "source_image": item.source_relative,
            "generated_crop": item.generated_relative,
            "context_region_xyxy": list(item.context_box),
            "edit_region_in_context_xyxy": list(item.edit_box),
        }
        for item in items
    ])

    try:
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
        from mlx_vlm.utils import get_model_path, load_model
    except ImportError as exc:
        raise SystemExit(
            "mlx-vlm is required; run this script from the dedicated MLX environment"
        ) from exc

    raw_model_path = Path(args.model_path).expanduser()
    model_path = raw_model_path if raw_model_path.exists() else get_model_path(args.model_path)
    print(f"loading {args.model_name} from {model_path}", flush=True)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(
        model,
        processor,
        score_threshold=args.score_threshold,
    )

    api_results_path = output_dir / "api_results.jsonl"
    latest = read_latest(api_results_path)
    for index, item in enumerate(items, 1):
        row_id = f"{item.task_id}__sam3"
        saved = latest.get(row_id)
        if saved and saved.get("status") == "ok" and saved.get("input_sha256") == item.input_sha256:
            print(f"[{index}/{len(items)}] reuse {item.task_id}", flush=True)
            continue
        started = time.perf_counter()
        try:
            with Image.open(item.generated_path) as image:
                result = predictor.predict(image.convert("RGB"), text_prompt=args.prompt)
            row = local_api_row(
                item,
                result,
                args.prompt,
                args.model_name,
                time.perf_counter() - started,
            )
            count = len(row["provider_output"]["rle"])
            print(
                f"[{index}/{len(items)}] {item.task_id}: {count} mask(s), "
                f"{row['local_inference_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            row = error_api_row(item, args.prompt, args.model_name, exc)
            print(f"[{index}/{len(items)}] {item.task_id}: {exc}", flush=True)
        append_jsonl(api_results_path, row)
        latest[row_id] = row

    config = HybridConfig(
        mode=args.hybrid_mode,
        diff_threshold=args.diff_threshold,
        support_radius=args.support_radius,
        alpha_feather=args.alpha_feather,
        edge_diff_threshold=args.edge_diff_threshold,
        edge_radius=args.edge_radius,
        shadow_feather=args.shadow_feather,
        far_diff_threshold=args.far_diff_threshold,
        max_added_fraction=args.max_added_fraction,
    )
    materialize_summary = materialize_all(
        items,
        output_dir,
        ["sam3"],
        config,
        api_results_path=api_results_path,
    )
    review_counts = build_review_manifest(items, output_dir, api_results_path)
    run_summary = {
        "schema_version": "mlx_sam3_splice_run_v1",
        "model": args.model_name,
        "model_path": str(model_path),
        "prompt": args.prompt,
        "score_threshold": args.score_threshold,
        "selection_count": len(items),
        "review_counts": review_counts,
        "materialization": materialize_summary,
        "completed_at": utc_now(),
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(review_counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
