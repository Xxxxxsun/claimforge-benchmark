#!/usr/bin/env python3
"""Build the formal Copyleaks Balanced250 table with reused mouse runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from eval.commercial.analyze_alibaba_balanced250 import (
    CONDITIONS,
    bootstrap_bacc,
    fmt_rate,
    latest_by_id,
    read_json,
    read_jsonl,
    relative,
    select_reused_mouse,
    sha256_file,
    utc_now,
    write_json,
)


DEFAULT_LEDGER = Path(
    "results/opensource/cnndetection/"
    "cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/"
    "expected_inputs.jsonl"
)
DEFAULT_MISSING_RESULTS = Path(
    "results/commercial/copyleaks/"
    "claimforge_balanced250_missing1250_canonical_png_20260727.jsonl"
)
DEFAULT_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/copyleaks/"
    "good275_mouse_forged_canonical_png_20260720.jsonl"
)
DEFAULT_LOCAL_MOUSE_MANIFEST = Path(
    "results/commercial/copyleaks/"
    "good275_mouse_forged_canonical_png_20260720.run_manifest.json"
)
DEFAULT_FULL_MOUSE_RESULTS = Path(
    "results/commercial/copyleaks/"
    "claimforge_v1_full_image_mouse250_canonical_png_20260725.jsonl"
)
DEFAULT_FULL_MOUSE_MANIFEST = Path(
    "results/commercial/copyleaks/"
    "claimforge_v1_full_image_mouse250_canonical_png_20260725.run_manifest.json"
)
DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_FULL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/full_image/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/copyleaks/"
    "claimforge_balanced250_main_table_20260727.summary.json"
)
DEFAULT_REPORT = Path("docs/COPYLEAKS_BALANCED250_RESULTS_2026-07-27.md")
DEFERRED_CONDITIONS = ("fullframe_cat", "fullframe_trash_can")


def positive(row: dict[str, Any]) -> bool:
    return bool(row.get("is_ai_detected"))


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return numeric_summary(
        [
            float(row["ai_score"])
            for row in rows
            if isinstance(row.get("ai_score"), (int, float))
        ]
    )


def localization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (row, row.get("localization", {}).get("pixel_diff_gt"))
        for row in rows
    ]
    pairs = [
        (row, target)
        for row, target in pairs
        if isinstance(target, dict)
    ]
    detected = [(row, target) for row, target in pairs if positive(row)]

    def metric(
        selected: list[tuple[dict[str, Any], dict[str, Any]]],
        name: str,
    ) -> dict[str, Any]:
        return numeric_summary(
            [
                float(target[name])
                for _, target in selected
                if isinstance(target.get(name), (int, float))
            ]
        )

    return {
        "evaluated": len(pairs),
        "positive_masks": len(detected),
        "empty_or_negative_masks": len(pairs) - len(detected),
        "any_overlap": sum(bool(target.get("any_overlap")) for _, target in pairs),
        "unconditional": {
            name: metric(pairs, name) for name in ("precision", "recall", "iou")
        },
        "detected_only": {
            name: metric(detected, name) for name in ("precision", "recall", "iou")
        },
    }


def format_optional_rate(value: Any) -> str:
    return fmt_rate(float(value)) if isinstance(value, (int, float)) else "n/a"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    real = summary["real_panel"]
    lines = [
        "# Copyleaks Ultra: CLAIMFORGE Balanced250",
        "",
        f"- Run completed: `{summary['generated_at']}`",
        "- Model: `ai-image-1-ultra`, `sandbox=false`.",
        "- Decision rule: vendor `isAiDetected=true`.",
        "- Upload policy: metadata-free RGB PNG, resized only when required by "
        "the vendor's image-size contract.",
        "- Primary comparison: the same independent real250 panel versus each "
        "250-image forged condition.",
        "",
        "## Coverage",
        "",
        f"- New calls: {summary['coverage']['new_valid']}/"
        f"{summary['coverage']['new_expected']} valid, "
        f"{summary['coverage']['new_errors']} final errors.",
        f"- Reused mouse calls: {summary['coverage']['reused_valid']}/"
        f"{summary['coverage']['reused_expected']} valid.",
        f"- Evaluated primary cells: {summary['coverage']['total_valid']}/"
        f"{summary['coverage']['total_planned']} images.",
        f"- Deferred before submission: "
        f"{summary['coverage']['deferred_images']} full-frame cat/trash-can images.",
        f"- New-call credits reported by Copyleaks: "
        f"{summary['coverage']['new_actual_credits']:.0f}.",
        "",
        "## Image-Level Results",
        "",
        f"Real false positives: {real['false_positives']}/250 "
        f"({fmt_rate(real['false_positive_rate'])}); specificity "
        f"{fmt_rate(real['specificity'])}.",
        "",
        "| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        if row.get("status") == "deferred":
            lines.append(
                f"| `{condition}` | deferred | deferred | "
                "deferred | deferred | deferred |"
            )
            continue
        lines.append(
            f"| `{condition}` | {row['true_positives']} / 250 | "
            f"{fmt_rate(row['sensitivity'])} | "
            f"{fmt_rate(row['specificity'])} | "
            f"{fmt_rate(row['balanced_accuracy'])} | "
            f"{fmt_rate(row['balanced_accuracy_ci95'][0])}-"
            f"{fmt_rate(row['balanced_accuracy_ci95'][1])} |"
        )

    lines.extend(
        [
            "",
            "## Native Localization",
            "",
            "Copyleaks returns a native RLE mask. Local-splice rows are compared "
            "against the frozen exact-difference mask; full-frame rows have no "
            "localization target.",
            "",
            "| Condition | Evaluated | Positive masks | Any GT overlap | Mean IoU, all | Mean IoU, detected |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in ("local_mouse", "local_cat", "local_trash_can"):
        row = summary["conditions"][condition]["localization"]
        lines.append(
            f"| `{condition}` | {row['evaluated']} | "
            f"{row['positive_masks']} | {row['any_overlap']} | "
            f"{format_optional_rate(row['unconditional']['iou']['mean'])} | "
            f"{format_optional_rate(row['detected_only']['iou']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Reuse Audit",
            "",
            f"- Local mouse: "
            f"{summary['reuse_audit']['local_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            f"- Full-frame mouse: "
            f"{summary['reuse_audit']['fullframe_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            "- Full-frame cat and trash-can were deferred at the user's request "
            "before any Copyleaks request was submitted.",
            "- Copyleaks uses its own canonical PNG upload policy, so frozen "
            "Q95 JPEG hashes are not treated as expected upload hashes.",
            "",
            "## Artifacts",
            "",
            f"- Missing-cell raw results: `{summary['artifacts']['new_results']['path']}`",
            f"- Missing-cell run manifest: "
            f"`{summary['artifacts']['new_run_manifest']['path']}`",
            f"- Formal summary: `{summary['artifacts']['summary']['path']}`",
            f"- Reused local mouse results: "
            f"`{summary['artifacts']['local_mouse_results']['path']}`",
            f"- Reused full-frame mouse results: "
            f"`{summary['artifacts']['fullframe_mouse_results']['path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--missing-results", type=Path, default=DEFAULT_MISSING_RESULTS)
    parser.add_argument(
        "--local-mouse-results", type=Path, default=DEFAULT_LOCAL_MOUSE_RESULTS
    )
    parser.add_argument(
        "--local-mouse-manifest", type=Path, default=DEFAULT_LOCAL_MOUSE_MANIFEST
    )
    parser.add_argument(
        "--full-mouse-results", type=Path, default=DEFAULT_FULL_MOUSE_RESULTS
    )
    parser.add_argument(
        "--full-mouse-manifest", type=Path, default=DEFAULT_FULL_MOUSE_MANIFEST
    )
    parser.add_argument(
        "--local-mouse-benchmark", type=Path, default=DEFAULT_LOCAL_MOUSE_BENCHMARK
    )
    parser.add_argument(
        "--full-mouse-benchmark", type=Path, default=DEFAULT_FULL_MOUSE_BENCHMARK
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    ledger_path = resolve(args.ledger)
    missing_path = resolve(args.missing_results)
    local_results_path = resolve(args.local_mouse_results)
    local_manifest_path = resolve(args.local_mouse_manifest)
    full_results_path = resolve(args.full_mouse_results)
    full_manifest_path = resolve(args.full_mouse_manifest)
    local_benchmark_path = resolve(args.local_mouse_benchmark)
    full_benchmark_path = resolve(args.full_mouse_benchmark)
    output_path = resolve(args.output)
    report_path = resolve(args.report)

    panel_rows = [
        row for row in read_jsonl(ledger_path) if row.get("panel") is True
    ]
    if len(panel_rows) != 1750:
        raise ValueError(f"expected 1750 primary panel rows, got {len(panel_rows)}")

    missing_attempts = read_jsonl(missing_path)
    missing_latest = latest_by_id(missing_attempts)
    missing_valid = [
        row for row in missing_latest.values() if row.get("status") == "ok"
    ]
    if len(missing_valid) != 750:
        raise ValueError(f"expected 750 valid new rows, got {len(missing_valid)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in missing_valid:
        grouped.setdefault(str(row["condition"]), []).append(row)
    for condition in (
        "real",
        "local_cat",
        "local_trash_can",
    ):
        if len(grouped.get(condition, [])) != 250:
            raise ValueError(f"{condition}: expected 250 new rows")
    for condition in DEFERRED_CONDITIONS:
        if grouped.get(condition):
            raise ValueError(f"{condition}: expected no submitted rows")

    local_mouse, local_audit = select_reused_mouse(
        condition="local_mouse",
        benchmark_rows=read_jsonl(local_benchmark_path),
        prior_manifest=read_json(local_manifest_path),
        prior_rows=read_jsonl(local_results_path),
        canonical_by_task={},
    )
    full_mouse, full_audit = select_reused_mouse(
        condition="fullframe_mouse",
        benchmark_rows=read_jsonl(full_benchmark_path),
        prior_manifest=read_json(full_manifest_path),
        prior_rows=read_jsonl(full_results_path),
        canonical_by_task={},
    )
    grouped["local_mouse"] = local_mouse
    grouped["fullframe_mouse"] = full_mouse

    real_positive = [positive(row) for row in grouped["real"]]
    false_positives = sum(real_positive)
    specificity = 1 - false_positives / 250
    condition_metrics: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
        if condition in DEFERRED_CONDITIONS:
            condition_metrics[condition] = {
                "status": "deferred",
                "expected": 250,
                "valid": 0,
                "reason": "deferred by user before API submission",
            }
            continue
        rows = grouped[condition]
        forged_positive = [positive(row) for row in rows]
        true_positives = sum(forged_positive)
        sensitivity = true_positives / 250
        bacc = (specificity + sensitivity) / 2
        ci = bootstrap_bacc(
            real_positive,
            forged_positive,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        condition_metrics[condition] = {
            "expected": 250,
            "valid": len(rows),
            "true_positives": true_positives,
            "false_negatives": 250 - true_positives,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": bacc,
            "balanced_accuracy_ci95": list(ci),
            "ai_score": score_summary(rows),
        }
        if condition.startswith("local_"):
            condition_metrics[condition]["localization"] = localization_summary(rows)

    new_actual_credits = sum(
        float(row["actual_credits"])
        for row in missing_valid
        if isinstance(row.get("actual_credits"), (int, float))
    )
    artifacts = {
        "new_results": {
            "path": relative(repo_root, missing_path),
            "sha256": sha256_file(missing_path),
        },
        "new_run_manifest": {
            "path": relative(repo_root, missing_path.with_suffix(".run_manifest.json")),
            "sha256": sha256_file(missing_path.with_suffix(".run_manifest.json")),
        },
        "local_mouse_results": {
            "path": relative(repo_root, local_results_path),
            "sha256": sha256_file(local_results_path),
        },
        "fullframe_mouse_results": {
            "path": relative(repo_root, full_results_path),
            "sha256": sha256_file(full_results_path),
        },
        "input_ledger": {
            "path": relative(repo_root, ledger_path),
            "sha256": sha256_file(ledger_path),
        },
    }
    summary: dict[str, Any] = {
        "schema_version": "copyleaks_ultra_balanced250_main_table_v1",
        "generated_at": utc_now(),
        "dataset_id": "claimforge-commercial-balanced250-mouse-reuse-v1",
        "input_selection": {
            "real_cat_trash": (
                "panel=true rows from "
                "claimforge-balanced250-independent-panel-jpeg-q95-v1"
            ),
            "mouse": "frozen benchmark panels, authorized reuse",
        },
        "provider": {
            "name": "Copyleaks AI Image Detector",
            "model": "ai-image-1-ultra",
            "decision_rule": "vendor isAiDetected",
            "upload_policy": "metadata-free RGB PNG; resize only for contract",
        },
        "coverage": {
            "new_expected": 750,
            "new_valid": len(missing_valid),
            "new_errors": 750 - len(missing_valid),
            "new_raw_attempt_rows": len(missing_attempts),
            "new_actual_credits": new_actual_credits,
            "reused_expected": 500,
            "reused_valid": len(local_mouse) + len(full_mouse),
            "total_planned": 1750,
            "deferred_images": 500,
            "total_evaluated_expected": 1250,
            "total_valid": len(missing_valid) + len(local_mouse) + len(full_mouse),
        },
        "real_panel": {
            "expected": 250,
            "valid": len(real_positive),
            "false_positives": false_positives,
            "true_negatives": 250 - false_positives,
            "false_positive_rate": false_positives / 250,
            "specificity": specificity,
            "ai_score": score_summary(grouped["real"]),
        },
        "conditions": condition_metrics,
        "reuse_audit": {
            "local_mouse": local_audit,
            "fullframe_mouse": full_audit,
        },
        "upload_audit": {
            "new_resized": sum(
                bool(row.get("upload_resized")) for row in missing_valid
            ),
            "new_unresized": sum(
                not bool(row.get("upload_resized")) for row in missing_valid
            ),
            "canonical_jpeg_hash_comparison": "not_applicable",
        },
        "bootstrap": {
            "method": "independent stratified image bootstrap of real and forged panels",
            "samples": args.bootstrap_samples,
            "seed_base": args.bootstrap_seed,
        },
        "artifacts": artifacts,
    }
    summary["artifacts"]["summary"] = {
        "path": relative(repo_root, output_path),
    }
    write_json(output_path, summary)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
