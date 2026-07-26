#!/usr/bin/env python3
"""Evaluate a deployable coarse-to-fine TruFor scan on CLAIMFORGE.

The oracle zoom experiment shows whether native-resolution crops can recover
tiny edits.  This runner removes the oracle: a full-image TruFor pass produces
an evidence map, fixed 512-pixel crops cover the image, and two budgeted
strategies retain only the four or eight crops with the strongest full-pass
evidence.  The exhaustive scan is evaluated as a non-oracle reference.

Ground truth is used only after inference for metrics and proposal-recall
diagnostics.  It never affects a crop location, score, or strategy selection.
Results are append-only and resumable at image granularity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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
from eval.our_defense.run_trufor_adaptive_zoom import (
    DEFAULT_PAIRS,
    _anchored,
    _load_rgb,
    _select_pairs,
    _target_for_pair,
    _tensor_from_image,
    _variant_path,
)


SCHEMA_VERSION = "claimforge_trufor_adaptive_scan_v1"
DEFAULT_OUTPUT_DIR = Path("results/our_defense/adaptive_scan")
STRATEGIES = ("full", "scan_all", "adaptive_map4", "adaptive_map8")
DETECTION_FEATURES = (
    "full_score",
    "max_score",
    "top2_mean",
    "top3_mean",
    "q90_score",
    "median_score",
    "max_minus_median",
    "robust_z",
    "positive_fraction",
    "full_max_mean",
    "full_top2_mean",
)


def _axis_starts(length: int, side: int, stride: int) -> list[int]:
    if length <= side:
        return [0]
    starts = list(range(0, length - side + 1, stride))
    last = length - side
    if starts[-1] != last:
        starts.append(last)
    return starts


def _grid_boxes(
    width: int,
    height: int,
    side: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    if side <= 0 or stride <= 0:
        raise ValueError("side and stride must be positive")
    crop_width = min(side, width)
    crop_height = min(side, height)
    return [
        (x, y, x + crop_width, y + crop_height)
        for y in _axis_starts(height, crop_height, stride)
        for x in _axis_starts(width, crop_width, stride)
    ]


def _top_mean(values: np.ndarray, fraction: float = 0.001) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    count = min(flat.size, max(16, int(math.ceil(flat.size * fraction))))
    start = flat.size - count
    return float(np.mean(np.partition(flat, start)[start:]))


def _proposal_score(
    score_map: np.ndarray,
    reliability: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    evidence = score_map[y1:y2, x1:x2]
    weighted = evidence * (0.25 + 0.75 * reliability[y1:y2, x1:x2])
    return _top_mean(weighted), _top_mean(evidence)


def _box_intersection_fraction(
    box: tuple[int, int, int, int],
    target: np.ndarray,
) -> float:
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return 0.0
    x1, y1, x2, y2 = box
    return float(np.count_nonzero(target[y1:y2, x1:x2]) / positives)


def _selected_indices(
    crop_records: list[dict[str, Any]],
    budget: int | None,
) -> list[int]:
    order = sorted(
        range(len(crop_records)),
        key=lambda index: (
            -float(crop_records[index]["proposal_score"]),
            int(crop_records[index]["index"]),
        ),
    )
    return order if budget is None else order[: min(budget, len(order))]


def _score_features(
    *,
    full_score: float,
    crop_records: list[dict[str, Any]],
    selected: list[int],
) -> dict[str, float]:
    scores = np.asarray(
        [float(crop_records[index]["score"]) for index in selected],
        dtype=np.float64,
    )
    ordered = np.sort(scores)[::-1]
    maximum = float(ordered[0])
    top2 = float(np.mean(ordered[: min(2, len(ordered))]))
    top3 = float(np.mean(ordered[: min(3, len(ordered))]))
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_z = (maximum - median) / (0.05 + 1.4826 * mad)
    return {
        "full_score": full_score,
        "max_score": maximum,
        "top2_mean": top2,
        "top3_mean": top3,
        "q90_score": float(np.quantile(scores, 0.9)),
        "median_score": median,
        "max_minus_median": maximum - median,
        "robust_z": robust_z,
        "positive_fraction": float(np.mean(scores >= 0.5)),
        "full_max_mean": 0.5 * (full_score + maximum),
        "full_top2_mean": 0.5 * (full_score + top2),
    }


def _strategy_result(
    *,
    name: str,
    full_score: float,
    full_map: np.ndarray,
    crop_records: list[dict[str, Any]],
    crop_maps: list[np.ndarray],
    selected: list[int],
    target: np.ndarray,
    kind: str,
    mask_threshold: float,
) -> dict[str, Any]:
    if name == "full":
        combined = full_map
        features = {feature: full_score for feature in DETECTION_FEATURES}
        selected = []
    else:
        combined = np.zeros_like(full_map, dtype=np.float32)
        for index in selected:
            x1, y1, x2, y2 = crop_records[index]["box_xyxy"]
            combined[y1:y2, x1:x2] = np.maximum(
                combined[y1:y2, x1:x2],
                crop_maps[index],
            )
        features = _score_features(
            full_score=full_score,
            crop_records=crop_records,
            selected=selected,
        )
    localization_target = target if kind == "forged" else np.zeros_like(target)
    return {
        "selected_crop_indices": selected,
        "selected_crops": len(selected),
        "detection_features": features,
        "localization": binary_pixel_metrics(
            combined,
            localization_target,
            mask_threshold,
            include_ap=kind == "forged",
        ),
        "proposal_target_recall": (
            max(
                (
                    float(crop_records[index]["target_intersection_fraction"])
                    for index in selected
                ),
                default=0.0,
            )
            if kind == "forged"
            else None
        ),
    }


def _split(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return "dev" if digest[0] < 85 else "test"


def _feature_rows(
    rows: list[dict[str, Any]],
    strategy: str,
    feature: str,
    split: str | None,
) -> list[dict[str, Any]]:
    selected = rows
    if split is not None:
        selected = [row for row in selected if row["split"] == split]
    result = []
    for row in selected:
        result.append(
            {
                "status": row["status"],
                "label": row["label"],
                "score": row["strategies"][strategy]["detection_features"][feature],
            }
        )
    return result


def _summarize(
    rows: list[dict[str, Any]],
    classification_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in STRATEGIES:
        strategy_result: dict[str, Any] = {"detection": {}}
        for split in (None, "dev", "test"):
            split_name = "all" if split is None else split
            strategy_result["detection"][split_name] = {
                feature: image_detection_metrics(
                    _feature_rows(rows, strategy, feature, split),
                    classification_threshold,
                )
                for feature in DETECTION_FEATURES
            }
        forged = [
            row["strategies"][strategy]
            for row in rows
            if row["kind"] == "forged"
        ]
        test_forged = [
            row["strategies"][strategy]
            for row in rows
            if row["kind"] == "forged" and row["split"] == "test"
        ]
        strategy_result["forged_localization"] = {
            split_name: {
                metric: descriptive(
                    float(item["localization"][metric])
                    for item in items
                    if item["localization"].get(metric) is not None
                )
                for metric in ("pixel_ap", "f1", "iou", "mcc")
            }
            for split_name, items in (("all", forged), ("test", test_forged))
        }
        strategy_result["proposal_target_recall"] = {
            split_name: descriptive(
                float(item["proposal_target_recall"])
                for item in items
                if item["proposal_target_recall"] is not None
            )
            for split_name, items in (("all", forged), ("test", test_forged))
        }
        result[strategy] = strategy_result
    return result


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
        f"adaptive scan {args.run_id}: {len(pair_rows)} pairs, "
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
                    image = _load_rgb(image_path)
                    if image.size != (width, height):
                        raise ValueError(
                            f"{pair['task_id']}: image size mismatch"
                        )
                    (
                        full_score,
                        full_logit,
                        full_map,
                        full_reliability,
                        full_peak_bytes,
                        full_latency_ms,
                    ) = infer_one(model, device, _tensor_from_image(image))

                    crop_records: list[dict[str, Any]] = []
                    crop_maps: list[np.ndarray] = []
                    crop_latency_ms = 0.0
                    peak_bytes = full_peak_bytes
                    for index, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        proposal_score, proposal_score_raw = _proposal_score(
                            full_map,
                            full_reliability,
                            box,
                        )
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
                        crop_records.append(
                            {
                                "index": index,
                                "box_xyxy": list(box),
                                "score": crop_score,
                                "score_margin": crop_logit,
                                "proposal_score": proposal_score,
                                "proposal_score_raw": proposal_score_raw,
                                "reliability_mean": float(
                                    np.mean(crop_reliability)
                                ),
                                "latency_ms": latency_ms,
                                "target_intersection_fraction": (
                                    _box_intersection_fraction(box, target)
                                    if kind == "forged"
                                    else 0.0
                                ),
                            }
                        )

                    all_indices = _selected_indices(crop_records, None)
                    map4 = _selected_indices(crop_records, 4)
                    map8 = _selected_indices(crop_records, 8)
                    strategies = {
                        "full": _strategy_result(
                            name="full",
                            full_score=full_score,
                            full_map=full_map,
                            crop_records=crop_records,
                            crop_maps=crop_maps,
                            selected=[],
                            target=target,
                            kind=kind,
                            mask_threshold=args.mask_threshold,
                        ),
                        "scan_all": _strategy_result(
                            name="scan_all",
                            full_score=full_score,
                            full_map=full_map,
                            crop_records=crop_records,
                            crop_maps=crop_maps,
                            selected=all_indices,
                            target=target,
                            kind=kind,
                            mask_threshold=args.mask_threshold,
                        ),
                        "adaptive_map4": _strategy_result(
                            name="adaptive_map4",
                            full_score=full_score,
                            full_map=full_map,
                            crop_records=crop_records,
                            crop_maps=crop_maps,
                            selected=map4,
                            target=target,
                            kind=kind,
                            mask_threshold=args.mask_threshold,
                        ),
                        "adaptive_map8": _strategy_result(
                            name="adaptive_map8",
                            full_score=full_score,
                            full_map=full_map,
                            crop_records=crop_records,
                            crop_maps=crop_maps,
                            selected=map8,
                            target=target,
                            kind=kind,
                            mask_threshold=args.mask_threshold,
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
                        "crop_side": args.crop_side,
                        "crop_stride": args.crop_stride,
                        "crop_count": len(crop_records),
                        "full_score": full_score,
                        "full_score_margin": full_logit,
                        "full_latency_ms": full_latency_ms,
                        "crop_latency_ms": crop_latency_ms,
                        "total_latency_ms": full_latency_ms + crop_latency_ms,
                        "peak_cuda_memory_bytes": peak_bytes,
                        "crops": crop_records,
                        "strategies": strategies,
                        "checkpoint_sha256": CHECKPOINT_SHA256,
                        "completed_at": utc_now(),
                    }
                    append_jsonl(result_path, row)
                    latest[row_id] = row
                    completed += 1
                    best = max(
                        float(crop["score"]) for crop in crop_records
                    )
                    recall4 = strategies["adaptive_map4"][
                        "proposal_target_recall"
                    ]
                    print(
                        f"[{completed}/{len(expected_ids)}] "
                        f"{pair['task_id']} {kind}: crops={len(crop_records)} "
                        f"full={full_score:.4f} scan_max={best:.4f} "
                        f"map4_recall={recall4} "
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
        raise RuntimeError(f"incomplete adaptive scan: {len(missing)} rows missing")
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
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "classification_threshold": args.classification_threshold,
        "mask_threshold": args.mask_threshold,
        "by_strategy": _summarize(
            selected_rows,
            args.classification_threshold,
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
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
