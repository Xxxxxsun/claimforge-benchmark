#!/usr/bin/env python3
"""Run the deployable A3D coarse-to-fine defense with a frozen TruFor model.

A3D performs one full-image pass, ranks a deterministic grid using only the
full-pass forensic evidence, and spends a fixed budget on the four strongest
native-resolution crops.  The image score is the maximum crop score.  The
highest-scoring crop supplies the primary localization map; top-two and all-four
fusion are retained as ablations.

No weights or thresholds are updated per test image.  Ground truth is used
only for evaluation after all proposals and predictions have been produced.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from eval.opensource.maskclip_metrics import (
    binary_pixel_metrics,
    descriptive,
    image_detection_metrics,
)
from eval.opensource.run_trufor import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    DEFAULT_TRUFOR_ROOT,
    infer_one,
    load_model,
)
from eval.our_defense.run_trufor_adaptive_scan import (
    _box_intersection_fraction,
    _grid_boxes,
    _proposal_score,
    _split,
)
from eval.our_defense.run_trufor_adaptive_zoom import (
    DEFAULT_PAIRS,
    _anchored,
    _load_rgb,
    _select_pairs,
    _target_for_pair,
    _tensor_from_image,
    _variant_path,
)


SCHEMA_VERSION = "claimforge_trufor_a3d_v2"
DEFAULT_OUTPUT_DIR = Path("results/our_defense/a3d")
LOCALIZATION_STRATEGIES = ("full", "a3d_top1", "a3d_top2", "a3d_all4")


def _jpeg_recompress(image: Image.Image, quality: int | None) -> Image.Image:
    if quality is None:
        return image
    if not 1 <= quality <= 100:
        raise ValueError("jpeg_quality must fall in [1, 100]")
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as opened:
        return opened.convert("RGB")


def _logit_mean_score(
    full_score: float,
    local_score: float,
    epsilon: float = 1e-6,
) -> float:
    """Fuse global and local evidence without letting either branch dominate."""

    def logit(value: float) -> float:
        clipped = min(max(float(value), epsilon), 1.0 - epsilon)
        return math.log(clipped / (1.0 - clipped))

    mean_logit = 0.5 * (logit(full_score) + logit(local_score))
    return 1.0 / (1.0 + math.exp(-mean_logit))


def _score_value(row: dict[str, Any], score_key: str) -> float:
    value = row.get(score_key)
    if value is not None:
        return float(value)
    if score_key == "a3d_fused_score":
        return _logit_mean_score(
            float(row["full_score"]),
            float(row["a3d_score"]),
        )
    raise KeyError(score_key)


def _rank_proposals(
    *,
    score_map: np.ndarray,
    reliability: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals = []
    for grid_index, box in enumerate(boxes):
        weighted, raw = _proposal_score(score_map, reliability, box)
        proposals.append(
            {
                "grid_index": grid_index,
                "box_xyxy": list(box),
                "proposal_score": weighted,
                "proposal_score_raw": raw,
            }
        )
    ranked = sorted(
        proposals,
        key=lambda item: (
            -float(item["proposal_score"]),
            int(item["grid_index"]),
        ),
    )
    selected = ranked[: min(budget, len(ranked))]
    return proposals, selected


def _fused_map(
    shape: tuple[int, int],
    crops: list[dict[str, Any]],
    crop_maps: list[np.ndarray],
    count: int,
) -> tuple[np.ndarray, list[int]]:
    ranked = sorted(
        range(len(crops)),
        key=lambda index: (
            -float(crops[index]["score"]),
            int(crops[index]["grid_index"]),
        ),
    )
    selected = ranked[: min(count, len(ranked))]
    combined = np.zeros(shape, dtype=np.float32)
    for index in selected:
        x1, y1, x2, y2 = crops[index]["box_xyxy"]
        combined[y1:y2, x1:x2] = np.maximum(
            combined[y1:y2, x1:x2],
            crop_maps[index],
        )
    return combined, selected


def _localization_metrics(
    *,
    score_map: np.ndarray,
    target: np.ndarray,
    kind: str,
    threshold: float,
    include_ap: bool,
) -> dict[str, Any]:
    truth = target if kind == "forged" else np.zeros_like(target)
    return binary_pixel_metrics(
        score_map,
        truth,
        threshold,
        include_ap=include_ap and kind == "forged",
    )


def _calibration_threshold(
    rows: list[dict[str, Any]],
    score_key: str,
    alpha: float,
) -> float | None:
    real_dev = np.asarray(
        [
            _score_value(row, score_key)
            for row in rows
            if row["kind"] == "real" and row["split"] == "dev"
        ],
        dtype=np.float64,
    )
    if real_dev.size == 0:
        return None
    quantile = float(np.quantile(real_dev, 1.0 - alpha, method="higher"))
    return float(np.nextafter(quantile, np.inf))


def _operating_point(
    rows: list[dict[str, Any]],
    score_key: str,
    threshold: float | None,
    split: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if threshold is None or not selected:
        return {"images": len(selected), "threshold": threshold}
    real = [row for row in selected if row["kind"] == "real"]
    forged = [row for row in selected if row["kind"] == "forged"]
    return {
        "images": len(selected),
        "real_images": len(real),
        "forged_images": len(forged),
        "threshold": threshold,
        "fpr": (
            sum(_score_value(row, score_key) >= threshold for row in real)
            / len(real)
            if real
            else None
        ),
        "tpr": (
            sum(_score_value(row, score_key) >= threshold for row in forged)
            / len(forged)
            if forged
            else None
        ),
    }


def _detection_rows(
    rows: list[dict[str, Any]],
    score_key: str,
    split: str | None,
) -> list[dict[str, Any]]:
    selected = rows if split is None else [
        row for row in rows if row["split"] == split
    ]
    return [
        {
            "status": row["status"],
            "label": row["label"],
            "score": _score_value(row, score_key),
        }
        for row in selected
    ]


def _summarize(
    rows: list[dict[str, Any]],
    classification_threshold: float,
    calibration_alpha: float,
) -> dict[str, Any]:
    detection: dict[str, Any] = {}
    for method, score_key in (
        ("full", "full_score"),
        ("a3d_local", "a3d_score"),
        ("a3d_fused", "a3d_fused_score"),
    ):
        calibrated = _calibration_threshold(rows, score_key, calibration_alpha)
        detection[method] = {
            "score_key": score_key,
            "by_split": {
                split_name: image_detection_metrics(
                    _detection_rows(rows, score_key, split),
                    classification_threshold,
                )
                for split_name, split in (
                    ("all", None),
                    ("dev", "dev"),
                    ("test", "test"),
                )
            },
            "calibration": {
                "alpha": calibration_alpha,
                "threshold_from_dev_real": calibrated,
                "dev": _operating_point(rows, score_key, calibrated, "dev"),
                "test": _operating_point(rows, score_key, calibrated, "test"),
            },
        }

    localization: dict[str, Any] = {}
    for strategy in LOCALIZATION_STRATEGIES:
        localization[strategy] = {}
        for split_name, split in (("all", None), ("test", "test")):
            selected = [
                row["localization"][strategy]
                for row in rows
                if row["kind"] == "forged"
                and (split is None or row["split"] == split)
            ]
            localization[strategy][split_name] = {
                metric: descriptive(
                    float(item[metric])
                    for item in selected
                    if item.get(metric) is not None
                )
                for metric in ("pixel_ap", "f1", "iou", "mcc")
            }
    return {"detection": detection, "localization": localization}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    pairs_path = _anchored(args.pairs, repo_root)
    output_dir = _anchored(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_id}.jsonl"
    pair_rows = _select_pairs(
        read_jsonl(pairs_path),
        args.scope,
        args.pair_limit,
    )
    if not pair_rows:
        raise ValueError("selection contains no pairs")

    expected_ids = [
        f"{pair['task_id']}|{kind}"
        for pair in pair_rows
        for kind in ("real", "forged")
    ]
    latest = read_latest_by_id(result_path)
    pending = [row_id for row_id in expected_ids if row_id not in latest]
    print(
        f"A3D {args.run_id}: {len(pair_rows)} pairs, "
        f"{len(expected_ids)} images, {len(pending)} pending",
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
                width = int(pair["width"])
                height = int(pair["height"])
                target = _target_for_pair(pair, repo_root)
                boxes = _grid_boxes(
                    width,
                    height,
                    args.crop_side,
                    args.crop_stride,
                )
                for kind, label in (("real", 0), ("forged", 1)):
                    row_id = f"{pair['task_id']}|{kind}"
                    if row_id in latest:
                        continue
                    image_path = _variant_path(pair, kind, repo_root)
                    image = _jpeg_recompress(
                        _load_rgb(image_path),
                        args.jpeg_quality,
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
                    crop_maps: list[np.ndarray] = []
                    crop_latency_ms = 0.0
                    peak_bytes = full_peak_bytes
                    for rank, proposal in enumerate(selected, start=1):
                        box = tuple(int(value) for value in proposal["box_xyxy"])
                        (
                            crop_score,
                            crop_logit,
                            crop_map,
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
                        crop_maps.append(crop_map)
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
                                "target_intersection_fraction": (
                                    _box_intersection_fraction(box, target)
                                    if kind == "forged"
                                    else None
                                ),
                            }
                        )
                    if not crops:
                        raise RuntimeError("A3D selected no crops")
                    a3d_score = max(float(crop["score"]) for crop in crops)
                    a3d_fused_score = _logit_mean_score(
                        full_score,
                        a3d_score,
                    )
                    maps_and_indices: dict[str, tuple[np.ndarray, list[int]]] = {
                        "a3d_top1": _fused_map(
                            target.shape, crops, crop_maps, 1
                        ),
                        "a3d_top2": _fused_map(
                            target.shape, crops, crop_maps, 2
                        ),
                        "a3d_all4": _fused_map(
                            target.shape, crops, crop_maps, len(crops)
                        ),
                    }
                    localization = {
                        "full": {
                            **_localization_metrics(
                                score_map=full_map,
                                target=target,
                                kind=kind,
                                threshold=args.mask_threshold,
                                include_ap=True,
                            ),
                            "selected_crop_indices": [],
                        }
                    }
                    for strategy, (score_map, selected_indices) in (
                        maps_and_indices.items()
                    ):
                        localization[strategy] = {
                            **_localization_metrics(
                                score_map=score_map,
                                target=target,
                                kind=kind,
                                threshold=args.mask_threshold,
                                include_ap=strategy
                                in {"a3d_top1", "a3d_top2"},
                            ),
                            "selected_crop_indices": selected_indices,
                            "proposal_target_recall": (
                                max(
                                    (
                                        float(
                                            crops[index][
                                                "target_intersection_fraction"
                                            ]
                                        )
                                        for index in selected_indices
                                    ),
                                    default=0.0,
                                )
                                if kind == "forged"
                                else None
                            ),
                        }
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
                        "split": _split(str(pair["task_id"])),
                        "gt_fraction": float(pair["gt_fraction"]),
                        "image_width": width,
                        "image_height": height,
                        "image_path": repo_relative(image_path, repo_root),
                        "input_transform": (
                            {"name": "jpeg", "quality": args.jpeg_quality}
                            if args.jpeg_quality is not None
                            else {"name": "identity"}
                        ),
                        "crop_side": args.crop_side,
                        "crop_stride": args.crop_stride,
                        "grid_crops": len(proposals),
                        "proposal_budget": args.proposal_budget,
                        "full_score": full_score,
                        "full_score_margin": full_logit,
                        "a3d_score": a3d_score,
                        "a3d_fused_score": a3d_fused_score,
                        "proposals": proposals,
                        "selected_crops": crops,
                        "localization": localization,
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
                    recall = localization["a3d_top2"].get(
                        "proposal_target_recall"
                    )
                    print(
                        f"[{completed}/{len(expected_ids)}] "
                        f"{pair['task_id']} {kind}: grid={len(proposals)} "
                        f"full={full_score:.4f} a3d={a3d_score:.4f} "
                        f"top2_recall={recall} "
                        f"latency={full_latency_ms + crop_latency_ms:.1f}ms",
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
        raise RuntimeError(f"incomplete A3D run: {len(missing)} rows missing")
    selected_rows = [latest_rows[row_id] for row_id in expected_ids]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "scope": args.scope,
        "pairs": len(pair_rows),
        "images": len(selected_rows),
        "splits": {
            split: sum(
                row["kind"] == "forged" and row["split"] == split
                for row in selected_rows
            )
            for split in ("dev", "test")
        },
        "crop_side": args.crop_side,
        "crop_stride": args.crop_stride,
        "proposal_budget": args.proposal_budget,
        "input_transform": (
            {"name": "jpeg", "quality": args.jpeg_quality}
            if args.jpeg_quality is not None
            else {"name": "identity"}
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "classification_threshold": args.classification_threshold,
        "mask_threshold": args.mask_threshold,
        "calibration_alpha": args.calibration_alpha,
        **_summarize(
            selected_rows,
            args.classification_threshold,
            args.calibration_alpha,
        ),
        "latency_ms": descriptive(
            float(row["total_latency_ms"]) for row in selected_rows
        ),
        "completed_at": utc_now(),
    }
    summary_path = output_dir / f"{args.run_id}.summary.json"
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": args.run_id,
                "pairs": len(pair_rows),
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
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--trufor-root", type=Path, default=DEFAULT_TRUFOR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scope", choices=("all", "q1", "q5"), default="all")
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-side", type=int, default=512)
    parser.add_argument("--crop-stride", type=int, default=384)
    parser.add_argument("--proposal-budget", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
