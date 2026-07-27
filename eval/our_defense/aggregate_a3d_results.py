#!/usr/bin/env python3
"""Build one auditable A3D metrics index from the durable result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from eval.opensource.common import (
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    utc_now,
)
from eval.our_defense.analyze_trufor_a3d import (
    METHODS,
    _pair_rows,
    _point_summary,
)
from eval.our_defense.run_trufor_a3d import (
    _calibration_threshold,
    _score_value,
)


SCHEMA_VERSION = "claimforge_a3d_aggregate_v1"
DEFAULT_OUTPUT = Path(
    "results/our_defense/a3d_aggregate_20260727/aggregate_metrics.json"
)
DEFAULT_THRESHOLD = 0.6353510120379108
DEFAULT_INPUTS = {
    "mouse": Path(
        "results/our_defense/mouse_a3d_20260725/clean/"
        "claim_a3d_deployable_all_20260725.jsonl"
    ),
    "mouse_summary": Path(
        "results/our_defense/mouse_a3d_20260725/clean/"
        "claim_a3d_deployable_all_20260725.summary.json"
    ),
    "mouse_analysis": Path(
        "results/our_defense/mouse_a3d_20260725/clean/"
        "claim_a3d_deployable_all_20260725.analysis.json"
    ),
    "mouse_jpeg90": Path(
        "results/our_defense/mouse_a3d_20260725/jpeg90/"
        "claim_a3d_deployable_all_jpeg90_20260725.jsonl"
    ),
    "mouse_jpeg90_summary": Path(
        "results/our_defense/mouse_a3d_20260725/jpeg90/"
        "claim_a3d_deployable_all_jpeg90_20260725.summary.json"
    ),
    "mouse_jpeg90_analysis": Path(
        "results/our_defense/mouse_a3d_20260725/jpeg90/"
        "claim_a3d_deployable_all_jpeg90_20260725.analysis.json"
    ),
    "cat": Path(
        "results/our_defense/cat_trash_a3d_20260725/"
        "cat_final_251_clean_a3d_v1.jsonl"
    ),
    "cat_analysis": Path(
        "results/our_defense/cat_trash_a3d_20260725/"
        "cat_final_251_clean_a3d_v1.analysis.json"
    ),
    "trash_can": Path(
        "results/our_defense/cat_trash_a3d_20260725/"
        "trash_can_final_250_clean_a3d_v1.jsonl"
    ),
    "trash_can_analysis": Path(
        "results/our_defense/cat_trash_a3d_20260725/"
        "trash_can_final_250_clean_a3d_v1.analysis.json"
    ),
    "cat_trash_combined_analysis": Path(
        "results/our_defense/cat_trash_a3d_20260725/"
        "cat_trash_combined_501_clean_a3d_v1.analysis.json"
    ),
    "generated_full": Path(
        "results/our_defense/generated_full_images_a3d_20260726/"
        "generated_full_all_807_a3d_q95_v1.jsonl"
    ),
    "generated_full_summary": Path(
        "results/our_defense/generated_full_images_a3d_20260726/"
        "generated_full_all_807_a3d_q95_v1.summary.json"
    ),
    "adaptive_scan_q1_summary": Path(
        "results/our_defense/mouse_a3d_20260725/diagnostics/"
        "adaptive_scan_q1/claim_a3d_adaptive_scan_q1_20260725.summary.json"
    ),
    "oracle_zoom_q1_summary": Path(
        "results/our_defense/mouse_a3d_20260725/diagnostics/"
        "oracle_zoom_q1/claim_a3d_oracle_zoom_q1_20260725.summary.json"
    ),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_latest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    physical = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    for row in physical:
        row_id = row.get("id")
        if not isinstance(row_id, str):
            raise ValueError(f"{path}: result row has no string id")
        latest[row_id] = row
    rows = list(latest.values())
    ok = [row for row in rows if row.get("status") == "ok"]
    if len(ok) != len(rows):
        raise ValueError(f"{path}: latest results contain failed rows")
    return rows, {
        "physical_rows": len(physical),
        "latest_rows": len(rows),
        "superseded_rows": len(physical) - len(rows),
        "ok_rows": len(ok),
        "unique_ids": len(latest),
        "sha256": _sha256(path),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _roc_operating_point(
    pairs: list[dict[str, Any]],
    score_key: str,
    max_fpr: float,
) -> dict[str, float | None]:
    labels: list[int] = []
    scores: list[float] = []
    for pair in pairs:
        for label, kind in ((0, "real"), (1, "forged")):
            labels.append(label)
            scores.append(_score_value(pair[kind], score_key))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )
    eligible = np.where(false_positive_rate <= max_fpr)[0]
    best_index = int(eligible[np.argmax(true_positive_rate[eligible])])
    threshold = float(thresholds[best_index])
    return {
        "target_fpr": max_fpr,
        "actual_fpr": float(false_positive_rate[best_index]),
        "tpr": float(true_positive_rate[best_index]),
        "threshold": threshold if np.isfinite(threshold) else None,
    }


def _fixed_operating_point(
    pairs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    real = [pair["real"] for pair in pairs]
    forged = [pair["forged"] for pair in pairs]
    return {
        "score": "a3d_fused_score",
        "threshold": threshold,
        "real_images": len(real),
        "forged_images": len(forged),
        "fpr": sum(
            _score_value(row, "a3d_fused_score") >= threshold for row in real
        )
        / len(real),
        "tpr": sum(
            _score_value(row, "a3d_fused_score") >= threshold
            for row in forged
        )
        / len(forged),
    }


def _scope_metrics(
    pairs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    result = _point_summary(pairs)
    for method, score_key in METHODS.items():
        result["detection"][method]["tpr_at_fpr_1_percent"] = (
            _roc_operating_point(pairs, score_key, 0.01)
        )
        result["detection"][method]["tpr_at_fpr_5_percent_detail"] = (
            _roc_operating_point(pairs, score_key, 0.05)
        )
    result["fixed_fused_operating_point"] = _fixed_operating_point(
        pairs,
        threshold,
    )
    return result


def _paired_dataset(
    rows: list[dict[str, Any]],
    threshold: float,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    pairs = _pair_rows(rows)
    if len(rows) != 2 * len(pairs):
        raise ValueError("paired result contains duplicate or incomplete rows")
    test_pairs = [pair for pair in pairs if pair["split"] == "test"]
    return {
        "coverage": {
            **coverage,
            "pairs": len(pairs),
            "images": len(rows),
            "dev_pairs": sum(pair["split"] == "dev" for pair in pairs),
            "test_pairs": len(test_pairs),
        },
        "all": _scope_metrics(pairs, threshold),
        "hash_test": _scope_metrics(test_pairs, threshold),
    }


def _scan_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    strategies: dict[str, Any] = {}
    for name, value in summary["by_strategy"].items():
        detection = value["detection"]["all"]
        strategies[name] = {
            "detection": {
                score_name: {
                    key: score_metrics.get(key)
                    for key in (
                        "auroc",
                        "average_precision",
                        "accuracy",
                        "balanced_accuracy",
                        "f1",
                        "tpr_at_fpr_5_percent",
                    )
                }
                for score_name, score_metrics in detection.items()
            },
            "forged_localization": value["forged_localization"]["all"],
            "proposal_target_recall": value["proposal_target_recall"]["all"],
        }
    return {
        "scope": summary["scope"],
        "pairs": summary["pairs"],
        "images": summary["images"],
        "strategies": strategies,
        "total_latency_ms": summary["latency_ms"],
    }


def _zoom_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": summary["scope"],
        "pairs": summary["expected_pairs"],
        "images_per_scale": summary["expected_rows"],
        "scales": {
            name: {
                "detection": value["detection"],
                "forged_localization": value["forged_localization"],
                "latency_ms": value["latency_ms"],
                "crop_area_fraction": value["crop_area_fraction"],
            }
            for name, value in summary["by_scale"].items()
        },
    }


def _generated_full_inventory(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "generated_full_images"
    by_directory: dict[str, int] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        count = sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in directory.iterdir()
        )
        if count:
            by_directory[directory.name] = count
    return {
        "recursive_image_files": sum(by_directory.values()),
        "directories_with_images": len(by_directory),
        "by_directory": by_directory,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    paths = {
        name: (repo_root / path).resolve()
        for name, path in DEFAULT_INPUTS.items()
    }
    mouse_rows, mouse_coverage = _load_latest(paths["mouse"])
    jpeg_rows, jpeg_coverage = _load_latest(paths["mouse_jpeg90"])
    cat_rows, cat_coverage = _load_latest(paths["cat"])
    trash_rows, trash_coverage = _load_latest(paths["trash_can"])
    generated_rows, generated_coverage = _load_latest(
        paths["generated_full"]
    )

    threshold = _calibration_threshold(
        mouse_rows,
        "a3d_fused_score",
        alpha=args.calibration_alpha,
    )
    if threshold is None:
        raise ValueError("mouse reference set produced no fused threshold")
    if not np.isclose(threshold, DEFAULT_THRESHOLD, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"unexpected fixed threshold: {threshold} != {DEFAULT_THRESHOLD}"
        )

    generated_summary = _json(paths["generated_full_summary"])
    generated_inventory = _generated_full_inventory(repo_root)
    if generated_summary["images"] != len(generated_rows):
        raise ValueError("generated-full JSONL/summary image mismatch")

    paired_inputs = {
        "mouse": (mouse_rows, mouse_coverage),
        "cat": (cat_rows, cat_coverage),
        "trash_can": (trash_rows, trash_coverage),
        "mouse_jpeg90": (jpeg_rows, jpeg_coverage),
        "cat_trash_combined": (
            cat_rows + trash_rows,
            {
                "component_datasets": ["cat", "trash_can"],
                "latest_rows": len(cat_rows) + len(trash_rows),
                "ok_rows": len(cat_rows) + len(trash_rows),
            },
        ),
        "all_spliced_mixed_preprocessing": (
            mouse_rows + cat_rows + trash_rows,
            {
                "component_datasets": ["mouse", "cat", "trash_can"],
                "latest_rows": len(mouse_rows) + len(cat_rows) + len(trash_rows),
                "ok_rows": len(mouse_rows) + len(cat_rows) + len(trash_rows),
            },
        ),
    }
    paired = {
        name: _paired_dataset(rows, threshold, coverage)
        for name, (rows, coverage) in paired_inputs.items()
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "name": "A3D adaptive anomaly-aware defense",
            "base_model": "TruFor",
            "checkpoint_sha256": (
                "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
            ),
            "crop_side": 512,
            "crop_stride": 384,
            "proposal_budget": 4,
            "primary_detection_score": (
                "sigmoid((logit(full_score) + logit(local_score)) / 2)"
            ),
            "primary_localization": "full_image_trufor_map",
            "calibration_alpha": args.calibration_alpha,
            "fixed_threshold": threshold,
            "calibration_real_images": sum(
                row["kind"] == "real" and row["split"] == "dev"
                for row in mouse_rows
            ),
        },
        "paired_spliced": paired,
        "paired_runner_summaries": {
            "mouse": _json(paths["mouse_summary"]),
            "mouse_jpeg90": _json(paths["mouse_jpeg90_summary"]),
        },
        "paired_bootstrap_analyses": {
            "mouse": _json(paths["mouse_analysis"]),
            "mouse_jpeg90": _json(paths["mouse_jpeg90_analysis"]),
            "cat": _json(paths["cat_analysis"]),
            "trash_can": _json(paths["trash_can_analysis"]),
            "cat_trash_combined": _json(
                paths["cat_trash_combined_analysis"]
            ),
        },
        "generated_full": {
            "coverage": generated_coverage,
            "evaluated_final_manifest_images": len(generated_rows),
            "current_recursive_inventory": generated_inventory,
            "not_in_final_manifests": (
                generated_inventory["recursive_image_files"]
                - len(generated_rows)
            ),
            "summary": generated_summary,
            "metric_scope_note": (
                "Generated-only set: report score distributions and fixed-"
                "threshold detected fractions, not AUROC/AP/localization."
            ),
        },
        "exploratory_q1_diagnostics": {
            "adaptive_scan": _scan_diagnostic(
                _json(paths["adaptive_scan_q1_summary"])
            ),
            "oracle_zoom": _zoom_diagnostic(
                _json(paths["oracle_zoom_q1_summary"])
            ),
        },
        "source_artifacts": {
            name: {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "completed_at": utc_now(),
    }
    output = _json_safe(output)
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, output)
    checksum_path = output_path.parent / "artifact_checksums.sha256"
    atomic_write_text(
        checksum_path,
        "".join(
            f"{value['sha256']}  {value['path']}\n"
            for value in output["source_artifacts"].values()
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path.relative_to(repo_root)),
                "checksums": str(checksum_path.relative_to(repo_root)),
                "fixed_threshold": threshold,
                "paired_datasets": sorted(paired),
                "generated_full_images": len(generated_rows),
            },
            indent=2,
        )
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
