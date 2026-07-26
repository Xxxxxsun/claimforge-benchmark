#!/usr/bin/env python3
"""Evaluate TruFor on paired oracle-centered crops at several native scales.

This is an upper-bound diagnostic for CLAIMFORGE's small-edit failure mode.
Every real/forged pair uses exactly the same crop coordinates, derived from the
pair's forged ground-truth bounding box.  The crop location is therefore not a
deployable proposal, but it cleanly tests whether concentrating a tiny edit
before forensic inference improves detection and localization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
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
    sha256_file,
    utc_now,
)
from eval.opensource.maskclip_metrics import (
    binary_pixel_metrics,
    descriptive,
    finite_float,
    image_detection_metrics,
    safe_div,
)
from eval.opensource.run_trufor import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    DEFAULT_TRUFOR_ROOT,
    infer_one,
    load_model,
)


SCHEMA_VERSION = "claimforge_trufor_adaptive_zoom_v1"
DEFAULT_PAIRS = Path("outputs/opensource/mouse_canonical_v1/pairs.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/our_defense/adaptive_zoom")
DEFAULT_SCALES = ("full", "context", "square128", "square256", "square512")


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8) > 0


def _tensor_from_image(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    tensor /= 256.0
    return tensor


def _validate_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop box: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def _square_box(
    target: tuple[int, int, int, int],
    side: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = target
    target_width = x2 - x1
    target_height = y2 - y1
    requested = max(int(side), target_width, target_height)
    crop_width = min(requested, width)
    crop_height = min(requested, height)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = int(round(center_x - crop_width / 2.0))
    top = int(round(center_y - crop_height / 2.0))
    left = max(0, min(width - crop_width, left))
    top = max(0, min(height - crop_height, top))
    return left, top, left + crop_width, top + crop_height


def _crop_specs(
    pair: dict[str, Any],
    scales: tuple[str, ...],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    width = int(pair["width"])
    height = int(pair["height"])
    bbox_value = pair.get("gt_bbox_xyxy")
    if not isinstance(bbox_value, list) or len(bbox_value) != 4:
        raise ValueError(f"{pair['task_id']}: invalid gt_bbox_xyxy")
    bbox = _validate_box(tuple(int(value) for value in bbox_value), width, height)
    context_value = pair.get("context_region_xyxy")
    if not isinstance(context_value, list) or len(context_value) != 4:
        raise ValueError(f"{pair['task_id']}: invalid context_region_xyxy")
    context = _validate_box(
        tuple(int(value) for value in context_value),
        width,
        height,
    )

    specs: list[tuple[str, tuple[int, int, int, int]]] = []
    for scale in scales:
        if scale == "full":
            box = (0, 0, width, height)
        elif scale == "context":
            box = context
        elif scale.startswith("square"):
            try:
                side = int(scale.removeprefix("square"))
            except ValueError as exc:
                raise ValueError(f"invalid square scale: {scale}") from exc
            if side <= 0:
                raise ValueError(f"invalid square side: {side}")
            box = _square_box(bbox, side, width, height)
        else:
            raise ValueError(f"unsupported scale: {scale}")
        if not (
            box[0] <= bbox[0]
            and box[1] <= bbox[1]
            and box[2] >= bbox[2]
            and box[3] >= bbox[3]
        ):
            raise ValueError(f"{pair['task_id']}: {scale} omits target bbox")
        specs.append((scale, box))
    return specs


def _select_pairs(
    rows: list[dict[str, Any]],
    scope: str,
    pair_limit: int | None,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["pair_rank"]))
    if scope in {"q1", "q5"}:
        by_size = sorted(
            ordered,
            key=lambda row: (float(row["gt_fraction"]), int(row["pair_rank"])),
        )
        quintile = math.ceil(len(by_size) / 5)
        selected_ids = {
            str(row["task_id"])
            for row in (
                by_size[:quintile] if scope == "q1" else by_size[-quintile:]
            )
        }
        ordered = [
            row for row in ordered if str(row["task_id"]) in selected_ids
        ]
    elif scope != "all":
        raise ValueError(f"unsupported scope: {scope}")
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        ordered = ordered[:pair_limit]
    return ordered


def _variant_path(pair: dict[str, Any], kind: str, repo_root: Path) -> Path:
    variant = pair.get(kind)
    if not isinstance(variant, dict):
        raise ValueError(f"{pair['task_id']}: missing {kind} variant")
    value = variant.get("canonical_path")
    expected_sha = variant.get("canonical_sha256")
    if not isinstance(value, str) or not isinstance(expected_sha, str):
        raise ValueError(f"{pair['task_id']}: invalid {kind} metadata")
    path = _anchored(Path(value), repo_root)
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{pair['task_id']}: {kind} hash mismatch")
    return path


def _target_for_pair(pair: dict[str, Any], repo_root: Path) -> np.ndarray:
    value = pair.get("gt_mask_path")
    expected_sha = pair.get("gt_mask_sha256")
    if not isinstance(value, str) or not isinstance(expected_sha, str):
        raise ValueError(f"{pair['task_id']}: invalid GT metadata")
    path = _anchored(Path(value), repo_root)
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{pair['task_id']}: GT hash mismatch")
    target = _load_mask(path)
    expected_shape = (int(pair["height"]), int(pair["width"]))
    if target.shape != expected_shape:
        raise ValueError(
            f"{pair['task_id']}: GT shape {target.shape} != {expected_shape}"
        )
    return target


def _full_space_metrics(
    *,
    score_crop: np.ndarray,
    target_full: np.ndarray,
    box: tuple[int, int, int, int],
    kind: str,
    threshold: float,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    if score_crop.shape != (y2 - y1, x2 - x1):
        raise ValueError(
            f"score crop shape {score_crop.shape} != {(y2 - y1, x2 - x1)}"
        )
    score_full = np.zeros(target_full.shape, dtype=np.float32)
    score_full[y1:y2, x1:x2] = score_crop
    target = target_full if kind == "forged" else np.zeros_like(target_full)
    return binary_pixel_metrics(
        score_full,
        target,
        threshold,
        include_ap=kind == "forged",
    )


def _paired_deltas(rows: list[dict[str, Any]]) -> list[float]:
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        score = finite_float(row.get("score"))
        if score is not None:
            paired[str(row["task_id"])][str(row["kind"])] = score
    return [
        values["forged"] - values["real"]
        for values in paired.values()
        if "real" in values and "forged" in values
    ]


def _summarize_scale(
    rows: list[dict[str, Any]],
    classification_threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    forged_metrics = [
        row["localization"]
        for row in valid
        if row.get("kind") == "forged"
        and isinstance(row.get("localization"), dict)
    ]
    real_metrics = [
        row["localization"]
        for row in valid
        if row.get("kind") == "real"
        and isinstance(row.get("localization"), dict)
    ]
    counts = {
        key: sum(int(metric.get(key, 0)) for metric in forged_metrics)
        for key in ("tp", "fp", "fn", "tn")
    }
    paired_deltas = _paired_deltas(valid)
    return {
        "images": len(valid),
        "pairs": len(paired_deltas),
        "detection": image_detection_metrics(valid, classification_threshold),
        "paired_score_delta": descriptive(paired_deltas),
        "paired_ranking_accuracy": (
            sum(delta > 0 for delta in paired_deltas) / len(paired_deltas)
            if paired_deltas
            else None
        ),
        "forged_localization": {
            metric: descriptive(
                float(row[metric])
                for row in forged_metrics
                if finite_float(row.get(metric)) is not None
            )
            for metric in (
                "pixel_ap",
                "precision",
                "recall",
                "f1",
                "iou",
                "mcc",
                "predicted_positive_fraction",
            )
        },
        "forged_localization_micro": {
            "threshold": mask_threshold,
            **counts,
            "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
            "f1": safe_div(
                2 * counts["tp"],
                2 * counts["tp"] + counts["fp"] + counts["fn"],
            ),
            "iou": safe_div(
                counts["tp"],
                counts["tp"] + counts["fp"] + counts["fn"],
            ),
        },
        "real_localization": {
            "predicted_positive_fraction": descriptive(
                float(row["predicted_positive_fraction"])
                for row in real_metrics
            ),
            "score_max": descriptive(float(row["score_max"]) for row in real_metrics),
        },
        "latency_ms": descriptive(float(row["latency_ms"]) for row in valid),
        "crop_area_fraction": descriptive(
            float(row["crop_area_fraction"]) for row in valid
        ),
    }


def _write_summary(
    *,
    output_dir: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    scales: tuple[str, ...],
    scope: str,
    expected_pairs: int,
    classification_threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    by_scale: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scale[str(row["crop_kind"])].append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scope": scope,
        "expected_pairs": expected_pairs,
        "expected_rows": expected_pairs * 2 * len(scales),
        "physical_rows": len(rows),
        "scales": list(scales),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "classification_threshold": classification_threshold,
        "mask_threshold": mask_threshold,
        "by_scale": {
            scale: _summarize_scale(
                by_scale.get(scale, []),
                classification_threshold,
                mask_threshold,
            )
            for scale in scales
        },
        "completed_at": utc_now(),
    }
    atomic_write_json(output_dir / f"{run_id}.summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    pairs_path = _anchored(args.pairs, repo_root)
    output_dir = _anchored(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_id}.jsonl"
    scales = tuple(args.scales)

    pair_rows = _select_pairs(
        read_jsonl(pairs_path),
        args.scope,
        args.pair_limit,
    )
    if not pair_rows:
        raise ValueError("selection contains no pairs")

    latest = read_latest_by_id(result_path)
    expected_ids = [
        f"{pair['task_id']}|{kind}|{scale}"
        for pair in pair_rows
        for kind in ("real", "forged")
        for scale in scales
    ]
    pending = [row_id for row_id in expected_ids if row_id not in latest]
    print(
        f"adaptive zoom {args.run_id}: {len(pair_rows)} pairs, "
        f"{len(expected_ids)} expected rows, {len(pending)} pending",
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
            completed = len(expected_ids) - len(pending)
            for pair in pair_rows:
                target_full = _target_for_pair(pair, repo_root)
                specs = _crop_specs(pair, scales)
                width = int(pair["width"])
                height = int(pair["height"])
                for kind, label in (("real", 0), ("forged", 1)):
                    image_path = _variant_path(pair, kind, repo_root)
                    image = _load_rgb(image_path)
                    if image.size != (width, height):
                        raise ValueError(
                            f"{pair['task_id']}: image size {image.size} "
                            f"!= {(width, height)}"
                        )
                    for crop_kind, box in specs:
                        row_id = f"{pair['task_id']}|{kind}|{crop_kind}"
                        if row_id in latest:
                            continue
                        x1, y1, x2, y2 = box
                        cropped = image.crop(box)
                        tensor = _tensor_from_image(cropped)
                        (
                            score,
                            logit,
                            score_map,
                            reliability,
                            peak_bytes,
                            latency_ms,
                        ) = infer_one(model, device, tensor)
                        localization = _full_space_metrics(
                            score_crop=score_map,
                            target_full=target_full,
                            box=box,
                            kind=kind,
                            threshold=args.mask_threshold,
                        )
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "id": row_id,
                            "run_id": args.run_id,
                            "status": "ok",
                            "task_id": str(pair["task_id"]),
                            "pair_rank": int(pair["pair_rank"]),
                            "domain": str(pair["domain"]),
                            "kind": kind,
                            "label": label,
                            "gt_fraction": float(pair["gt_fraction"]),
                            "crop_kind": crop_kind,
                            "crop_xyxy": list(box),
                            "crop_width": x2 - x1,
                            "crop_height": y2 - y1,
                            "crop_area_fraction": (
                                (x2 - x1) * (y2 - y1) / (width * height)
                            ),
                            "image_width": width,
                            "image_height": height,
                            "image_path": repo_relative(image_path, repo_root),
                            "score": score,
                            "score_margin": logit,
                            "decision": score >= args.classification_threshold,
                            "localization": localization,
                            "reliability": {
                                "min": float(np.min(reliability)),
                                "mean": float(np.mean(reliability)),
                                "median": float(np.median(reliability)),
                                "max": float(np.max(reliability)),
                            },
                            "latency_ms": latency_ms,
                            "peak_cuda_memory_bytes": peak_bytes,
                            "checkpoint_sha256": CHECKPOINT_SHA256,
                            "completed_at": utc_now(),
                        }
                        append_jsonl(result_path, row)
                        latest[row_id] = row
                        completed += 1
                        print(
                            f"[{completed}/{len(expected_ids)}] "
                            f"{pair['task_id']} {kind} {crop_kind}: "
                            f"score={score:.6f} f1={localization.get('f1')} "
                            f"latency={latency_ms:.1f}ms",
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
            f"incomplete adaptive zoom run: {len(missing)} rows missing"
        )
    selected_rows = [latest_rows[row_id] for row_id in expected_ids]
    summary = _write_summary(
        output_dir=output_dir,
        run_id=args.run_id,
        rows=selected_rows,
        scales=scales,
        scope=args.scope,
        expected_pairs=len(pair_rows),
        classification_threshold=args.classification_threshold,
        mask_threshold=args.mask_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--trufor-root", type=Path, default=DEFAULT_TRUFOR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=list(DEFAULT_SCALES),
    )
    parser.add_argument("--scope", choices=("all", "q1", "q5"), default="all")
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
