#!/usr/bin/env python3
"""Audit and statistically analyze a completed paired MaskCLIP run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


DEFAULT_RUN_ID = "maskclip_mouse_canonical_v1_full275_20260723"
DEFAULT_RESULTS_DIR = Path("results/opensource/maskclip")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
HISTOGRAM_BINS = 65_536


@dataclass(frozen=True)
class Pair:
    task_id: str
    domain: str
    real: dict[str, Any]
    forged: dict[str, Any]
    input_row: dict[str, Any]

    @property
    def edit_fraction(self) -> float:
        metrics = self.forged["localization"]["native"]
        return float(metrics["target_positive_pixels"]) / float(metrics["pixels"])


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot calculate a confidence interval from no values")
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _estimate_ci(estimate: float, replicates: list[float]) -> dict[str, Any]:
    return {
        "estimate": float(estimate),
        "ci95_percentile": _percentile_ci(replicates),
    }


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    eligible = np.where(false_positive_rate <= target_fpr)[0]
    return float(np.max(true_positive_rate[eligible]))


def _slice_arrays(pairs: list[Pair]) -> dict[str, np.ndarray]:
    return {
        "real_score": np.asarray(
            [float(pair.real["score"]) for pair in pairs],
            dtype=np.float64,
        ),
        "forged_score": np.asarray(
            [float(pair.forged["score"]) for pair in pairs],
            dtype=np.float64,
        ),
        "pixel_ap": np.asarray(
            [
                float(pair.forged["localization"]["native"]["pixel_ap"])
                for pair in pairs
            ],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [float(pair.forged["localization"]["native"]["f1"]) for pair in pairs],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [float(pair.forged["localization"]["native"]["iou"]) for pair in pairs],
            dtype=np.float64,
        ),
        "tp": np.asarray(
            [int(pair.forged["localization"]["native"]["tp"]) for pair in pairs],
            dtype=np.int64,
        ),
        "fp": np.asarray(
            [int(pair.forged["localization"]["native"]["fp"]) for pair in pairs],
            dtype=np.int64,
        ),
        "fn": np.asarray(
            [int(pair.forged["localization"]["native"]["fn"]) for pair in pairs],
            dtype=np.int64,
        ),
    }


def _point_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    real = arrays["real_score"]
    forged = arrays["forged_score"]
    labels = np.concatenate(
        [np.zeros(real.size, dtype=np.int64), np.ones(forged.size, dtype=np.int64)]
    )
    scores = np.concatenate([real, forged])
    predictions = scores >= 0.5
    tp = int(np.count_nonzero(predictions & (labels == 1)))
    fp = int(np.count_nonzero(predictions & (labels == 0)))
    fn = int(np.count_nonzero(~predictions & (labels == 1)))
    tn = int(np.count_nonzero(~predictions & (labels == 0)))
    pixel_tp = int(np.sum(arrays["tp"]))
    pixel_fp = int(np.sum(arrays["fp"]))
    pixel_fn = int(np.sum(arrays["fn"]))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tpr_at_fpr_5_percent": _tpr_at_fpr(labels, scores, 0.05),
        "accuracy_at_0_5": float(np.mean(predictions == labels)),
        "image_f1_at_0_5": float(f1_score(labels, predictions, zero_division=0)),
        "paired_ranking_accuracy": float(np.mean(forged > real)),
        "paired_score_delta_mean": float(np.mean(forged - real)),
        "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
        "pixel_f1_macro_at_0_5": float(np.mean(arrays["pixel_f1"])),
        "pixel_iou_macro_at_0_5": float(np.mean(arrays["pixel_iou"])),
        "pixel_f1_micro_at_0_5": (
            2.0 * pixel_tp / (2 * pixel_tp + pixel_fp + pixel_fn)
            if 2 * pixel_tp + pixel_fp + pixel_fn
            else 0.0
        ),
        "pixel_iou_micro_at_0_5": (
            pixel_tp / (pixel_tp + pixel_fp + pixel_fn)
            if pixel_tp + pixel_fp + pixel_fn
            else 0.0
        ),
        "image_tp_at_0_5": float(tp),
        "image_fp_at_0_5": float(fp),
        "image_fn_at_0_5": float(fn),
        "image_tn_at_0_5": float(tn),
    }


BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "accuracy_at_0_5",
    "image_f1_at_0_5",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
)


def summarize_pair_slice(
    pairs: list[Pair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pair slice is empty")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    arrays = _slice_arrays(pairs)
    point = _point_metrics(arrays)
    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(iterations):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sampled = {name: values[indices] for name, values in arrays.items()}
        values = _point_metrics(sampled)
        for name in BOOTSTRAP_METRICS:
            replicates[name].append(float(values[name]))

    delta = arrays["forged_score"] - arrays["real_score"]
    wins = int(np.count_nonzero(delta > 0))
    losses = int(np.count_nonzero(delta < 0))
    ties = int(np.count_nonzero(delta == 0))
    non_ties = wins + losses
    lower_tail = min(wins, losses)
    sign_test_p = min(
        1.0,
        2.0
        * sum(math.comb(non_ties, k) for k in range(lower_tail + 1))
        / (2**non_ties),
    )
    return {
        "pairs": len(pairs),
        "images": len(pairs) * 2,
        **{
            name: _estimate_ci(point[name], replicates[name])
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_0_5": {
            "tp": int(point["image_tp_at_0_5"]),
            "fp": int(point["image_fp_at_0_5"]),
            "fn": int(point["image_fn_at_0_5"]),
            "tn": int(point["image_tn_at_0_5"]),
        },
        "paired_sign_test": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "two_sided_exact_p": sign_test_p,
        },
        "edit_fraction": {
            "min": float(min(pair.edit_fraction for pair in pairs)),
            "median": float(np.median([pair.edit_fraction for pair in pairs])),
            "mean": float(np.mean([pair.edit_fraction for pair in pairs])),
            "max": float(max(pair.edit_fraction for pair in pairs)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }


def histogram_best_metrics(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = HISTOGRAM_BINS,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(f"score/target mismatch: {scores.shape} != {truth.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("score map falls outside [0, 1]")
    if not truth.any():
        raise ValueError("best-threshold metrics require a non-empty target")

    indices = np.minimum(
        (scores * (bins - 1)).astype(np.int32),
        bins - 1,
    )
    all_hist = np.bincount(indices.reshape(-1), minlength=bins).astype(np.int64)
    positive_hist = np.bincount(indices[truth], minlength=bins).astype(np.int64)
    tp = np.cumsum(positive_hist[::-1], dtype=np.int64)[::-1]
    predicted = np.cumsum(all_hist[::-1], dtype=np.int64)[::-1]
    fp = predicted - tp
    total_positive = int(np.count_nonzero(truth))
    fn = total_positive - tp
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=f1_denominator > 0,
    )
    iou = np.divide(
        tp,
        iou_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=iou_denominator > 0,
    )
    best_index = int(np.argmax(f1))
    return (
        {
            "histogram_bins": bins,
            "threshold": best_index / (bins - 1),
            "f1": float(f1[best_index]),
            "iou": float(iou[best_index]),
            "tp": int(tp[best_index]),
            "fp": int(fp[best_index]),
            "fn": int(fn[best_index]),
        },
        all_hist,
        positive_hist,
    )


def _load_pairs(
    result_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> list[Pair]:
    latest = {
        str(row["id"]): row
        for row in result_rows
        if isinstance(row.get("id"), str)
    }
    expected_ids = {str(row["sample_id"]) for row in input_rows}
    if set(latest) != expected_ids:
        missing = sorted(expected_ids - set(latest))
        unexpected = sorted(set(latest) - expected_ids)
        raise ValueError(
            f"result/input ID mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    if any(row.get("status") != "ok" for row in latest.values()):
        raise ValueError("analysis requires every latest result row to be successful")

    inputs_by_id = {str(row["sample_id"]): row for row in input_rows}
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in latest.values():
        by_task.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    pairs: list[Pair] = []
    for task_id, values in by_task.items():
        if set(values) != {"real", "forged"}:
            raise ValueError(f"incomplete pair for {task_id}: {sorted(values)}")
        real = values["real"]
        forged = values["forged"]
        input_row = inputs_by_id[str(forged["id"])]
        if real.get("domain") != forged.get("domain"):
            raise ValueError(f"domain mismatch within {task_id}")
        pairs.append(
            Pair(
                task_id=task_id,
                domain=str(forged["domain"]),
                real=real,
                forged=forged,
                input_row=input_row,
            )
        )
    return sorted(pairs, key=lambda pair: int(pair.forged["pair_rank"]))


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"{label} has no expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def audit_and_best_threshold(
    pairs: list[Pair],
    *,
    repo_root: Path,
    bins: int,
) -> dict[str, Any]:
    per_image_best: list[dict[str, Any]] = []
    global_all = np.zeros(bins, dtype=np.int64)
    global_positive = np.zeros(bins, dtype=np.int64)
    box_ious: list[float] = []
    box_hits = 0
    checked_files = 0

    for pair in pairs:
        for result in (pair.real, pair.forged):
            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            model_map_path = _anchored(
                Path(str(result["score_map_model_path"])),
                repo_root,
            )
            native_map_path = _anchored(
                Path(str(result["score_map_native_path"])),
                repo_root,
            )
            mask_path = _anchored(Path(str(result["mask_path"])), repo_root)
            for path, expected, label in (
                (image_path, result["image_sha256"], "canonical image"),
                (
                    model_map_path,
                    result["score_map_model_sha256"],
                    "model-space score map",
                ),
                (
                    native_map_path,
                    result["score_map_native_sha256"],
                    "native score map",
                ),
                (mask_path, result["mask_sha256"], "threshold mask"),
            ):
                _verify_hash(path, expected, f"{label} {result['id']}")
                checked_files += 1

            width, height = (int(value) for value in result["image_size"])
            model_map = np.load(model_map_path, mmap_mode="r", allow_pickle=False)
            native_map = np.load(native_map_path, mmap_mode="r", allow_pickle=False)
            if model_map.shape != (512, 512):
                raise ValueError(f"invalid model map shape for {result['id']}")
            if native_map.shape != (height, width):
                raise ValueError(f"invalid native map shape for {result['id']}")
            if model_map.dtype != np.float32 or native_map.dtype != np.float32:
                raise ValueError(f"invalid score map dtype for {result['id']}")
            for name, score_map in (
                ("model", model_map),
                ("native", native_map),
            ):
                if not np.isfinite(score_map).all():
                    raise ValueError(
                        f"non-finite {name} score map for {result['id']}"
                    )
                if float(score_map.min()) < 0.0 or float(score_map.max()) > 1.0:
                    raise ValueError(
                        f"out-of-range {name} score map for {result['id']}"
                    )
            with Image.open(mask_path) as opened:
                binary_mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
            expected_mask = np.asarray(native_map) >= float(result["mask_threshold"])
            if not np.array_equal(binary_mask, expected_mask):
                raise ValueError(f"threshold mask mismatch for {result['id']}")

        mask_value = pair.input_row.get("gt_mask_path")
        mask_sha = pair.input_row.get("gt_mask_sha256")
        if not isinstance(mask_value, str):
            raise ValueError(f"forged sample has no GT mask: {pair.task_id}")
        target_path = _anchored(Path(mask_value), repo_root)
        _verify_hash(target_path, mask_sha, f"ground-truth mask {pair.task_id}")
        checked_files += 1
        with Image.open(target_path) as opened:
            target = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        native_map_path = _anchored(
            Path(str(pair.forged["score_map_native_path"])),
            repo_root,
        )
        native_map = np.load(native_map_path, mmap_mode="r", allow_pickle=False)
        best, all_hist, positive_hist = histogram_best_metrics(
            native_map,
            target,
            bins=bins,
        )
        per_image_best.append({"task_id": pair.task_id, **best})
        global_all += all_hist
        global_positive += positive_hist

        with Image.open(
            _anchored(Path(str(pair.forged["mask_path"])), repo_root)
        ) as opened:
            prediction = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        x1, y1, x2, y2 = (int(value) for value in pair.input_row["edit_region_xyxy"])
        box_area = (x2 - x1) * (y2 - y1)
        intersection = int(np.count_nonzero(prediction[y1:y2, x1:x2]))
        predicted_area = int(np.count_nonzero(prediction))
        union = predicted_area + box_area - intersection
        box_iou = intersection / union if union else 0.0
        box_ious.append(box_iou)
        box_hits += int(box_iou > 0.3)

    global_tp = np.cumsum(global_positive[::-1], dtype=np.int64)[::-1]
    global_predicted = np.cumsum(global_all[::-1], dtype=np.int64)[::-1]
    global_fp = global_predicted - global_tp
    global_fn = int(np.sum(global_positive)) - global_tp
    global_denominator = 2 * global_tp + global_fp + global_fn
    global_f1 = np.divide(
        2.0 * global_tp,
        global_denominator,
        out=np.zeros_like(global_tp, dtype=np.float64),
        where=global_denominator > 0,
    )
    best_index = int(np.argmax(global_f1))
    best_f1_values = [float(row["f1"]) for row in per_image_best]
    best_iou_values = [float(row["iou"]) for row in per_image_best]
    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": checked_files,
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "checks": [
                "all expected IDs are present exactly once in the latest rows",
                "every latest result has status=ok",
                "canonical image, both score maps, threshold mask, and GT hashes",
                "score-map dtype, dimensions, finiteness, and [0,1] range",
                "saved threshold mask equals native score map >= 0.5",
            ],
        },
        "localization_best_threshold": {
            "approximation": (
                f"native score maps quantized into {bins} uniform bins over [0,1]"
            ),
            "per_image_oracle": {
                "images": len(per_image_best),
                "f1_mean": float(np.mean(best_f1_values)),
                "f1_median": float(np.median(best_f1_values)),
                "iou_mean": float(np.mean(best_iou_values)),
                "iou_median": float(np.median(best_iou_values)),
            },
            "single_global_oracle": {
                "threshold": best_index / (bins - 1),
                "micro_f1": float(global_f1[best_index]),
                "micro_iou": (
                    float(global_tp[best_index])
                    / float(
                        global_tp[best_index]
                        + global_fp[best_index]
                        + global_fn[best_index]
                    )
                ),
                "tp": int(global_tp[best_index]),
                "fp": int(global_fp[best_index]),
                "fn": int(global_fn[best_index]),
            },
        },
        "box_hit_at_mask_threshold_0_5": {
            "definition": "IoU(predicted native binary mask, edit_region_xyxy) > 0.3",
            "hits": box_hits,
            "images": len(pairs),
            "rate": box_hits / len(pairs),
            "iou_mean": float(np.mean(box_ious)),
            "iou_median": float(np.median(box_ious)),
            "iou_max": float(np.max(box_ious)),
        },
    }


def _quintiles(pairs: list[Pair]) -> list[tuple[str, list[Pair]]]:
    ordered = sorted(pairs, key=lambda pair: (pair.edit_fraction, pair.task_id))
    chunks = np.array_split(np.asarray(ordered, dtype=object), 5)
    return [
        (
            (
                f"q{index}_"
                f"{'smallest' if index == 1 else 'largest' if index == 5 else ''}"
            ).rstrip("_"),
            list(chunk),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    results_dir = _anchored(args.results_dir, repo_root)
    result_path = results_dir / f"{args.run_id}.jsonl"
    run_manifest_path = results_dir / f"{args.run_id}.run_manifest.json"
    summary_path = results_dir / f"{args.run_id}.summary.json"
    output_path = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else results_dir / f"{args.run_id}.analysis.json"
    )
    input_path = _anchored(args.inputs, repo_root)
    for path in (result_path, run_manifest_path, summary_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result_rows = read_jsonl(result_path)
    input_rows = read_jsonl(input_path)
    pairs = _load_pairs(result_rows, input_rows)
    if len(result_rows) != len(input_rows):
        raise ValueError(
            "post-hoc analysis requires one physical result row per input; "
            f"got {len(result_rows)} and {len(input_rows)}"
        )
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != args.run_id:
        raise ValueError("run manifest ID does not match requested run")

    overall = summarize_pair_slice(
        pairs,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    by_domain = {
        domain: summarize_pair_slice(
            [pair for pair in pairs if pair.domain == domain],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + index,
        )
        for index, domain in enumerate(
            sorted({pair.domain for pair in pairs}),
            start=1,
        )
    }
    by_edit_quintile = {
        name: summarize_pair_slice(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }
    audit = audit_and_best_threshold(
        pairs,
        repo_root=repo_root,
        bins=args.histogram_bins,
    )

    value = {
        "schema_version": "maskclip_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "sources": {
            "results_path": str(result_path.relative_to(repo_root)),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": str(run_manifest_path.relative_to(repo_root)),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "summary_path": str(summary_path.relative_to(repo_root)),
            "summary_sha256": sha256_file(summary_path),
            "inputs_path": str(input_path.relative_to(repo_root)),
            "inputs_sha256": sha256_file(input_path),
        },
        "bootstrap": {
            "unit": "paired task (real and forged resampled together)",
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "interval": "2.5th and 97.5th percentile",
        },
        "overall": overall,
        "by_domain": by_domain,
        "by_edit_fraction_quintile": by_edit_quintile,
        **audit,
    }
    atomic_write_json(output_path, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--histogram-bins", type=int, default=HISTOGRAM_BINS)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
