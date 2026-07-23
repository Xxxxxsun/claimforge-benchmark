#!/usr/bin/env python3
"""Audit and statistically analyze a completed paired TruFor run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.analyze_maskclip_run import (
    HISTOGRAM_BINS,
    Pair,
    _load_pairs,
    _quintiles,
    histogram_best_metrics,
    summarize_pair_slice,
)
from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.run_trufor import (
    CHECKPOINT_EPOCH,
    CHECKPOINT_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE_SHA256,
    MODEL_SOURCE_COMMIT,
)


DEFAULT_RUN_ID = "trufor_mouse_canonical_v1_full275_20260723"
DEFAULT_RESULTS_DIR = Path("results/opensource/trufor")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"{label} has no expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"result row {line_number} has no non-empty string id"
            )
        histories[row_id].append((line_number, row))
        status_counts[str(row.get("status"))] += 1

    duplicate_histories: list[dict[str, Any]] = []
    recovered_ids: list[str] = []
    latest_status_counts: Counter[str] = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_status_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicate_histories.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line_number for line_number, _ in entries],
                    "statuses": statuses,
                }
            )
        if statuses[-1] == "ok" and "error" in statuses[:-1]:
            recovered_ids.append(row_id)

    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_rows": len(result_rows) - len(histories),
        "ids_with_multiple_rows": len(duplicate_histories),
        "recovered_error_to_ok": len(recovered_ids),
        "recovered_ids": recovered_ids,
        "historical_status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _selection_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def validate_provenance(
    *,
    repo_root: Path,
    run_id: str,
    input_path: Path,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    _require_equal(
        manifest.get("schema_version"),
        "opensource_run_manifest_v1",
        "run manifest schema",
    )
    _require_equal(manifest.get("run_id"), run_id, "run manifest ID")

    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("run manifest has no fingerprint")
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint", "created_at", "adapter", "environment"}
    }
    computed_fingerprint = hashlib.sha256(
        stable_json(immutable).encode("utf-8")
    ).hexdigest()
    _require_equal(
        fingerprint,
        computed_fingerprint,
        "run manifest fingerprint",
    )

    manifest_input = _require_mapping(manifest.get("input"), "manifest input")
    actual_inputs_sha256 = sha256_file(input_path)
    _require_equal(
        manifest_input.get("inputs_sha256"),
        actual_inputs_sha256,
        "manifest/input JSONL SHA-256",
    )
    manifest_inputs_path_value = manifest_input.get("inputs_manifest")
    if not isinstance(manifest_inputs_path_value, str):
        raise ValueError("manifest has no inputs_manifest path")
    _require_equal(
        _anchored(Path(manifest_inputs_path_value), repo_root),
        input_path.resolve(),
        "manifest/input JSONL path",
    )

    dataset_manifest_value = manifest_input.get("dataset_manifest")
    if not isinstance(dataset_manifest_value, str):
        raise ValueError("manifest has no dataset_manifest path")
    dataset_manifest_path = _anchored(Path(dataset_manifest_value), repo_root)
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    release = _require_mapping(
        json.loads(dataset_manifest_path.read_text(encoding="utf-8")),
        "canonical dataset manifest",
    )
    _require_equal(
        release.get("schema_version"),
        "claimforge_mouse_canonical_v1",
        "canonical dataset schema",
    )
    for key in ("dataset_id", "contract_sha256", "inputs_sha256"):
        _require_equal(
            release.get(key),
            manifest_input.get(
                "dataset_contract_sha256" if key == "contract_sha256" else key
            ),
            f"canonical dataset {key}",
        )
    release_inputs_value = release.get("inputs_path")
    if not isinstance(release_inputs_value, str):
        raise ValueError("canonical dataset manifest has no inputs_path")
    _require_equal(
        _anchored(Path(release_inputs_value), repo_root),
        input_path.resolve(),
        "canonical dataset inputs path",
    )

    expected_ids = [str(row["sample_id"]) for row in input_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("canonical inputs contain duplicate sample IDs")
    expected_selection = _selection_contract(input_rows)
    _require_equal(
        manifest.get("ordered_inputs"),
        expected_selection,
        "manifest ordered input selection",
    )
    selection_sha256 = hashlib.sha256(
        stable_json(expected_selection).encode("utf-8")
    ).hexdigest()
    _require_equal(
        manifest_input.get("selection_sha256"),
        selection_sha256,
        "manifest input selection SHA-256",
    )
    _require_equal(
        manifest.get("expected_images"),
        len(input_rows),
        "manifest expected image count",
    )
    expected_pairs = len({int(row["pair_rank"]) for row in input_rows})
    _require_equal(
        manifest.get("expected_pairs"),
        expected_pairs,
        "manifest expected pair count",
    )

    model = _require_mapping(manifest.get("model"), "manifest model")
    checkpoint = _require_mapping(
        model.get("checkpoint"),
        "manifest checkpoint",
    )
    configuration = _require_mapping(
        model.get("configuration"),
        "manifest model configuration",
    )
    license_value = _require_mapping(model.get("license"), "manifest license")
    for actual, expected, label in (
        (model.get("name"), "TruFor", "manifest model name"),
        (model.get("model_slug"), "trufor_cvpr2023", "manifest model slug"),
        (
            model.get("source_commit"),
            MODEL_SOURCE_COMMIT,
            "manifest model source commit",
        ),
        (
            model.get("source_tracked_clean"),
            True,
            "manifest source clean flag",
        ),
        (
            configuration.get("sha256"),
            MODEL_CONFIG_SHA256,
            "manifest model configuration SHA-256",
        ),
        (
            license_value.get("sha256"),
            MODEL_LICENSE_SHA256,
            "manifest model license SHA-256",
        ),
        (
            checkpoint.get("sha256"),
            CHECKPOINT_SHA256,
            "manifest checkpoint SHA-256",
        ),
        (
            checkpoint.get("epoch"),
            CHECKPOINT_EPOCH,
            "manifest checkpoint epoch",
        ),
        (
            checkpoint.get("strict_load"),
            True,
            "manifest strict-load flag",
        ),
        (
            checkpoint.get("safe_weights_only_load"),
            True,
            "manifest safe-load flag",
        ),
    ):
        _require_equal(actual, expected, label)

    expected_by_id = {
        str(row["sample_id"]): row
        for row in input_rows
    }
    seen_ids: set[str] = set()
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"result row {line_number} has no non-empty string id"
            )
        if row_id not in expected_by_id:
            raise ValueError(f"unexpected result ID at row {line_number}: {row_id}")
        seen_ids.add(row_id)
        latest[row_id] = row
        input_row = expected_by_id[row_id]
        expected_values = {
            "schema_version": "opensource_result_v1",
            "run_id": run_id,
            "input_manifest_sha256": actual_inputs_sha256,
            "id": row_id,
            "task_id": str(input_row["task_id"]),
            "pair_rank": int(input_row["pair_rank"]),
            "domain": str(input_row["domain"]),
            "kind": str(input_row["kind"]),
            "label": int(input_row["label"]),
            "image_path": str(input_row["canonical_path"]),
            "image_sha256": str(input_row["canonical_sha256"]),
            "image_size": [
                int(input_row["width"]),
                int(input_row["height"]),
            ],
            "model": "TruFor",
            "model_slug": "trufor_cvpr2023",
            "checkpoint_sha256": CHECKPOINT_SHA256,
        }
        for key, expected in expected_values.items():
            _require_equal(
                row.get(key),
                expected,
                f"result row {line_number} field {key}",
            )
        if row.get("status") not in {"ok", "error"}:
            raise ValueError(
                f"result row {line_number} has invalid status: {row.get('status')!r}"
            )
    if seen_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - seen_ids)
        raise ValueError(f"result history is missing expected IDs: {missing[:5]}")

    _require_equal(
        summary.get("schema_version"),
        "opensource_summary_v1",
        "summary schema",
    )
    for actual, expected, label in (
        (summary.get("run_id"), run_id, "summary run ID"),
        (summary.get("condition"), manifest.get("condition"), "summary condition"),
        (summary.get("model"), "TruFor", "summary model name"),
        (
            summary.get("model_slug"),
            "trufor_cvpr2023",
            "summary model slug",
        ),
        (
            summary.get("checkpoint_sha256"),
            CHECKPOINT_SHA256,
            "summary checkpoint SHA-256",
        ),
        (
            summary.get("input_manifest_sha256"),
            actual_inputs_sha256,
            "summary input manifest SHA-256",
        ),
        (
            summary.get("run_manifest_fingerprint"),
            fingerprint,
            "summary run manifest fingerprint",
        ),
    ):
        _require_equal(actual, expected, label)

    coverage = _require_mapping(summary.get("coverage"), "summary coverage")
    valid_latest = sum(row.get("status") == "ok" for row in latest.values())
    expected_coverage = {
        "expected_images": len(input_rows),
        "result_images": len(latest),
        "valid_images": valid_latest,
        "error_images": len(latest) - valid_latest,
        "missing_images": len(input_rows) - len(latest),
    }
    for key, expected in expected_coverage.items():
        _require_equal(
            coverage.get(key),
            expected,
            f"summary coverage {key}",
        )

    inference = _require_mapping(
        manifest.get("inference"),
        "manifest inference",
    )
    detection_summary = _require_mapping(
        summary.get("detection"),
        "summary detection",
    )
    localization_summary = _require_mapping(
        summary.get("localization_forged"),
        "summary forged localization",
    )
    native_summary = _require_mapping(
        localization_summary.get("native"),
        "summary native localization",
    )
    micro_summary = _require_mapping(
        native_summary.get("micro_at_threshold"),
        "summary native localization threshold metrics",
    )
    _require_equal(
        detection_summary.get("threshold"),
        inference.get("classification_threshold"),
        "summary classification threshold",
    )
    _require_equal(
        micro_summary.get("threshold"),
        inference.get("mask_threshold"),
        "summary localization threshold",
    )

    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical dataset manifest, input path, input hash, and ordered selection",
            "pinned source, configuration, license, checkpoint, strict-load, and safe-load metadata",
            "every physical result row against its canonical input and run/model/checkpoint provenance",
            "summary identity, fingerprint, thresholds, and latest-row coverage",
        ],
    }


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
            score_map_path = _anchored(
                Path(str(result["score_map_native_path"])),
                repo_root,
            )
            reliability_path = _anchored(
                Path(str(result["reliability_map_native_path"])),
                repo_root,
            )
            mask_path = _anchored(Path(str(result["mask_path"])), repo_root)
            for path, expected, label in (
                (image_path, result["image_sha256"], "canonical image"),
                (
                    score_map_path,
                    result["score_map_native_sha256"],
                    "native score map",
                ),
                (
                    reliability_path,
                    result["reliability_map_native_sha256"],
                    "native reliability map",
                ),
                (mask_path, result["mask_sha256"], "threshold mask"),
            ):
                _verify_hash(path, expected, f"{label} {result['id']}")
                checked_files += 1

            width, height = (int(value) for value in result["image_size"])
            score_map = np.load(score_map_path, mmap_mode="r", allow_pickle=False)
            reliability = np.load(
                reliability_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            for name, value in (
                ("score", score_map),
                ("reliability", reliability),
            ):
                if value.shape != (height, width):
                    raise ValueError(
                        f"invalid {name} map shape for {result['id']}"
                    )
                if value.dtype != np.float32:
                    raise ValueError(
                        f"invalid {name} map dtype for {result['id']}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"non-finite {name} map for {result['id']}"
                    )
                if float(value.min()) < 0.0 or float(value.max()) > 1.0:
                    raise ValueError(
                        f"out-of-range {name} map for {result['id']}"
                    )
            with Image.open(mask_path) as opened:
                binary_mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
            expected_mask = np.asarray(score_map) >= float(result["mask_threshold"])
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
        score_map_path = _anchored(
            Path(str(pair.forged["score_map_native_path"])),
            repo_root,
        )
        score_map = np.load(score_map_path, mmap_mode="r", allow_pickle=False)
        best, all_hist, positive_hist = histogram_best_metrics(
            score_map,
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
                "the latest-row ID set exactly matches the canonical inputs",
                "every latest result has status=ok",
                "canonical image, score map, reliability map, mask, and GT hashes",
                "map dtype, native dimensions, finiteness, and [0,1] range",
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
    manifest = _require_mapping(
        json.loads(run_manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    result_history = summarize_result_history(result_rows)
    provenance = validate_provenance(
        repo_root=repo_root,
        run_id=args.run_id,
        input_path=input_path,
        input_rows=input_rows,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
    )
    pairs = _load_pairs(result_rows, input_rows)

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
        "schema_version": "trufor_posthoc_analysis_v1",
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
        "provenance_integrity": provenance,
        "result_history": result_history,
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
