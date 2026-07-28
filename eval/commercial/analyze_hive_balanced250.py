#!/usr/bin/env python3
"""Build the Hive Balanced250 table with reused mouse runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
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
DEFAULT_NEW_RESULTS = Path(
    "results/commercial/hive/"
    "claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl"
)
DEFAULT_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/hive/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl"
)
DEFAULT_LOCAL_MOUSE_MANIFEST = Path(
    "results/commercial/hive/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.run_manifest.json"
)
DEFAULT_FULL_MOUSE_RESULTS = Path(
    "results/commercial/hive/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl"
)
DEFAULT_FULL_MOUSE_MANIFEST = Path(
    "results/commercial/hive/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.run_manifest.json"
)
DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_FULL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/full_image/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/hive/"
    "claimforge_balanced250_main_table_20260727.summary.json"
)
DEFAULT_REPORT = Path("docs/HIVE_BALANCED250_RESULTS_2026-07-27.md")
DEFERRED_CONDITIONS = ("fullframe_cat", "fullframe_trash_can")
THRESHOLD = 0.9


def positive(row: dict[str, Any]) -> bool:
    value = row.get("ai_probability")
    return isinstance(value, (int, float)) and float(value) >= THRESHOLD


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["ai_probability"])
        for row in rows
        if isinstance(row.get("ai_probability"), (int, float))
    ]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    real = summary["real_panel"]
    lines = [
        "# Hive V3: CLAIMFORGE Balanced250",
        "",
        f"- Run completed: `{summary['generated_at']}`",
        f"- Decision rule: `ai_generated >= {THRESHOLD}`.",
        "- Upload policy: metadata-free RGB JPEG Q95 with 4:4:4 subsampling.",
        "",
        "## Coverage",
        "",
        f"- New panel: {summary['coverage']['new_valid']}/"
        f"{summary['coverage']['new_expected']} valid, "
        f"{summary['coverage']['new_errors']} final errors.",
        f"- Raw API attempts: {summary['coverage']['new_raw_attempt_rows']}; "
        f"{summary['coverage']['superseded_attempt_rows']} superseded attempt rows.",
        f"- Canonical upload hashes: "
        f"{summary['canonicalization_audit']['new_matches']}/"
        f"{summary['canonicalization_audit']['new_reference_available']} match.",
        f"- Reused mouse calls: {summary['coverage']['reused_valid']}/"
        f"{summary['coverage']['reused_expected']} valid.",
        f"- Evaluated primary cells: {summary['coverage']['total_valid']}/"
        f"{summary['coverage']['total_planned']} images.",
        f"- Deferred before submission: "
        f"{summary['coverage']['deferred_images']} full-frame cat/trash-can images.",
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
            "## Reuse Audit",
            "",
            f"- Local mouse: "
            f"{summary['reuse_audit']['local_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            f"- Full-frame mouse: "
            f"{summary['reuse_audit']['fullframe_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            "- Full-frame cat and trash-can were deferred before submission.",
            "",
            "## Artifacts",
            "",
            f"- New raw results: `{summary['artifacts']['new_results']['path']}`",
            f"- New run manifest: "
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
    parser.add_argument("--new-results", type=Path, default=DEFAULT_NEW_RESULTS)
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
    new_results_path = resolve(args.new_results)
    new_manifest_path = new_results_path.with_suffix(".run_manifest.json")
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
    canonical = {
        (str(row["condition"]), str(row["task_id"])): str(row["canonical_sha256"])
        for row in panel_rows
    }

    new_attempts = read_jsonl(new_results_path)
    new_latest = latest_by_id(new_attempts)
    new_manifest = read_json(new_manifest_path)
    expected_new_ids = {
        str(row["id"]) for row in new_manifest["ordered_inputs"]
    }
    if set(new_latest) != expected_new_ids:
        raise ValueError("new result IDs do not match the run manifest")
    new_valid = [row for row in new_latest.values() if row.get("status") == "ok"]
    if len(new_valid) != 750:
        raise ValueError(f"expected 750 valid new rows, got {len(new_valid)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in new_valid:
        grouped.setdefault(str(row["condition"]), []).append(row)
    for condition in ("real", "local_cat", "local_trash_can"):
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
        canonical_by_task={
            task_id: digest
            for (condition, task_id), digest in canonical.items()
            if condition == "local_mouse"
        },
    )
    full_mouse, full_audit = select_reused_mouse(
        condition="fullframe_mouse",
        benchmark_rows=read_jsonl(full_benchmark_path),
        prior_manifest=read_json(full_manifest_path),
        prior_rows=read_jsonl(full_results_path),
        canonical_by_task={
            task_id: digest
            for (condition, task_id), digest in canonical.items()
            if condition == "fullframe_mouse"
        },
    )
    grouped["local_mouse"] = local_mouse
    grouped["fullframe_mouse"] = full_mouse

    evaluated = [
        *new_valid,
        *local_mouse,
        *full_mouse,
    ]
    invalid_thresholds = [
        row["id"] for row in evaluated if float(row.get("threshold", -1)) != THRESHOLD
    ]
    if invalid_thresholds:
        raise ValueError(f"unexpected Hive thresholds: {invalid_thresholds[:5]}")
    missing_scores = [
        row["id"]
        for row in evaluated
        if not isinstance(row.get("ai_probability"), (int, float))
    ]
    if missing_scores:
        raise ValueError(f"missing Hive scores: {missing_scores[:5]}")

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
                "reason": "deferred before API submission",
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
            "ai_probability": score_summary(rows),
            "provider_models": dict(
                Counter(str(row.get("provider_model")) for row in rows)
            ),
            "provider_versions": dict(
                Counter(str(row.get("provider_version")) for row in rows)
            ),
        }

    new_canonical_matches = sum(
        row.get("upload_sha256")
        == canonical[(str(row["condition"]), str(row["task_id"]))]
        for row in new_valid
    )
    artifacts = {
        "new_results": {
            "path": relative(repo_root, new_results_path),
            "sha256": sha256_file(new_results_path),
        },
        "new_run_manifest": {
            "path": relative(repo_root, new_manifest_path),
            "sha256": sha256_file(new_manifest_path),
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
        "summary": {"path": relative(repo_root, output_path)},
    }
    summary: dict[str, Any] = {
        "schema_version": "hive_v3_balanced250_main_table_v1",
        "generated_at": utc_now(),
        "dataset_id": "claimforge-commercial-balanced250-mouse-reuse-v1",
        "provider": {
            "name": "Hive V3",
            "decision_rule": f"ai_generated >= {THRESHOLD}",
            "upload_policy": "metadata-free RGB JPEG Q95 4:4:4",
        },
        "coverage": {
            "new_expected": 750,
            "new_valid": len(new_valid),
            "new_errors": 750 - len(new_valid),
            "new_raw_attempt_rows": len(new_attempts),
            "superseded_attempt_rows": len(new_attempts) - len(new_latest),
            "reused_expected": 500,
            "reused_valid": len(local_mouse) + len(full_mouse),
            "total_planned": 1750,
            "deferred_images": 500,
            "total_evaluated_expected": 1250,
            "total_valid": len(new_valid) + len(local_mouse) + len(full_mouse),
        },
        "real_panel": {
            "expected": 250,
            "valid": len(real_positive),
            "false_positives": false_positives,
            "true_negatives": 250 - false_positives,
            "false_positive_rate": false_positives / 250,
            "specificity": specificity,
            "ai_probability": score_summary(grouped["real"]),
        },
        "conditions": condition_metrics,
        "reuse_audit": {
            "local_mouse": local_audit,
            "fullframe_mouse": full_audit,
        },
        "canonicalization_audit": {
            "new_reference_available": 750,
            "new_matches": new_canonical_matches,
            "new_mismatches": 750 - new_canonical_matches,
        },
        "bootstrap": {
            "method": "independent stratified image bootstrap of real and forged panels",
            "samples": args.bootstrap_samples,
            "seed_base": args.bootstrap_seed,
        },
        "artifacts": artifacts,
    }
    write_json(output_path, summary)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
