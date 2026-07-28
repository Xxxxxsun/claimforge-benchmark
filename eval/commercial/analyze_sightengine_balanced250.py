#!/usr/bin/env python3
"""Build the Sightengine Balanced250 table with partial mouse reuse."""

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
    "results/commercial/sightengine/"
    "claimforge_balanced250_core822_original_files_20260727.jsonl"
)
DEFAULT_PRIOR_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.jsonl"
)
DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_FULL_MOUSE_RESULTS = Path(
    "results/commercial/sightengine/"
    "claimforge_v1_full_image_mouse250_original_png_20260725.jsonl"
)
DEFAULT_FULL_MOUSE_MANIFEST = Path(
    "results/commercial/sightengine/"
    "claimforge_v1_full_image_mouse250_original_png_20260725.run_manifest.json"
)
DEFAULT_FULL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/full_image/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/sightengine/"
    "claimforge_balanced250_main_table_20260727.summary.json"
)
DEFAULT_REPORT = Path("docs/SIGHTENGINE_BALANCED250_RESULTS_2026-07-27.md")
DEFERRED_CONDITIONS = ("fullframe_cat", "fullframe_trash_can")
THRESHOLD = 0.5


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


def select_local_mouse(
    benchmark_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(benchmark_rows) != 250:
        raise ValueError("expected 250 local-mouse benchmark rows")
    benchmark_by_task = {str(row["task_id"]): row for row in benchmark_rows}
    prior_by_task = {
        str(row["task_id"]): row
        for row in latest_by_id(prior_rows).values()
        if row.get("status") == "ok"
    }
    new_by_task = {
        str(row["task_id"]): row
        for row in latest_by_id(new_rows).values()
        if row.get("status") == "ok" and row.get("condition") == "local_mouse"
    }
    selected: list[dict[str, Any]] = []
    reused = newly_submitted = 0
    for task_id, benchmark in benchmark_by_task.items():
        row = prior_by_task.get(task_id)
        source = "prior"
        if row is None:
            row = new_by_task.get(task_id)
            source = "new"
        if row is None:
            raise ValueError(f"missing local-mouse result: {task_id}")
        if row.get("image_sha256") != benchmark.get("sha256"):
            raise ValueError(f"local-mouse image SHA-256 mismatch: {task_id}")
        selected.append(row)
        if source == "prior":
            reused += 1
        else:
            newly_submitted += 1
    if reused != 178 or newly_submitted != 72:
        raise ValueError(
            f"unexpected local-mouse split: prior={reused}, new={newly_submitted}"
        )
    return selected, {
        "selected": 250,
        "prior_reused": reused,
        "newly_submitted": newly_submitted,
        "raw_image_sha256_matches": 250,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    real = summary["real_panel"]
    lines = [
        "# Sightengine genai: CLAIMFORGE Balanced250",
        "",
        f"- Run completed: `{summary['generated_at']}`",
        f"- Decision rule: `type.ai_generated >= {THRESHOLD}`.",
        "- Upload policy: exact benchmark raw JPEG/PNG files without resizing "
        "or re-encoding.",
        "",
        "## Coverage",
        "",
        f"- New submissions: {summary['coverage']['new_valid']}/"
        f"{summary['coverage']['new_expected']} valid, "
        f"{summary['coverage']['new_errors']} final errors.",
        f"- Raw API attempts: {summary['coverage']['new_raw_attempt_rows']}; "
        f"{summary['coverage']['new_superseded_attempt_rows']} failed attempt "
        "was superseded by a successful retry.",
        f"- New operations consumed: "
        f"{summary['coverage']['new_operations_consumed']}.",
        f"- Reused results: {summary['coverage']['reused_valid']}/"
        f"{summary['coverage']['reused_expected']} valid "
        "(178 local mouse plus 250 full-frame mouse).",
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
            f"{summary['reuse_audit']['local_mouse']['prior_reused']} prior + "
            f"{summary['reuse_audit']['local_mouse']['newly_submitted']} new; "
            "250/250 raw hashes match.",
            f"- Full-frame mouse: "
            f"{summary['reuse_audit']['fullframe_mouse']['raw_image_sha256_matches']}"
            "/250 raw hashes match.",
            "- Full-frame cat and trash-can were deferred before submission.",
            "",
            "## Artifacts",
            "",
            f"- New raw results: `{summary['artifacts']['new_results']['path']}`",
            f"- New run manifest: "
            f"`{summary['artifacts']['new_run_manifest']['path']}`",
            f"- Formal summary: `{summary['artifacts']['summary']['path']}`",
            f"- Prior local-mouse results: "
            f"`{summary['artifacts']['prior_local_mouse_results']['path']}`",
            f"- Full-frame mouse results: "
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
        "--prior-local-mouse-results",
        type=Path,
        default=DEFAULT_PRIOR_LOCAL_MOUSE_RESULTS,
    )
    parser.add_argument(
        "--local-mouse-benchmark",
        type=Path,
        default=DEFAULT_LOCAL_MOUSE_BENCHMARK,
    )
    parser.add_argument(
        "--full-mouse-results", type=Path, default=DEFAULT_FULL_MOUSE_RESULTS
    )
    parser.add_argument(
        "--full-mouse-manifest", type=Path, default=DEFAULT_FULL_MOUSE_MANIFEST
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
    prior_local_path = resolve(args.prior_local_mouse_results)
    local_benchmark_path = resolve(args.local_mouse_benchmark)
    full_results_path = resolve(args.full_mouse_results)
    full_manifest_path = resolve(args.full_mouse_manifest)
    full_benchmark_path = resolve(args.full_mouse_benchmark)
    output_path = resolve(args.output)
    report_path = resolve(args.report)

    panel_rows = [
        row for row in read_jsonl(ledger_path) if row.get("panel") is True
    ]
    if len(panel_rows) != 1750:
        raise ValueError(f"expected 1750 primary panel rows, got {len(panel_rows)}")

    new_attempts = read_jsonl(new_results_path)
    new_latest = latest_by_id(new_attempts)
    new_valid = [row for row in new_latest.values() if row.get("status") == "ok"]
    if len(new_valid) != 822:
        raise ValueError(f"expected 822 valid new rows, got {len(new_valid)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in new_valid:
        grouped.setdefault(str(row["condition"]), []).append(row)
    for condition, expected in (
        ("real", 250),
        ("local_mouse", 72),
        ("local_cat", 250),
        ("local_trash_can", 250),
    ):
        if len(grouped.get(condition, [])) != expected:
            raise ValueError(f"{condition}: expected {expected} new rows")
    for condition in DEFERRED_CONDITIONS:
        if grouped.get(condition):
            raise ValueError(f"{condition}: expected no submitted rows")

    local_mouse, local_audit = select_local_mouse(
        read_jsonl(local_benchmark_path),
        read_jsonl(prior_local_path),
        new_attempts,
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
            "operations": sum(int(row.get("operations") or 0) for row in rows),
        }

    artifacts = {
        "new_results": {
            "path": relative(repo_root, new_results_path),
            "sha256": sha256_file(new_results_path),
        },
        "new_run_manifest": {
            "path": relative(
                repo_root, new_results_path.with_suffix(".run_manifest.json")
            ),
            "sha256": sha256_file(
                new_results_path.with_suffix(".run_manifest.json")
            ),
        },
        "prior_local_mouse_results": {
            "path": relative(repo_root, prior_local_path),
            "sha256": sha256_file(prior_local_path),
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
        "schema_version": "sightengine_genai_balanced250_main_table_v1",
        "generated_at": utc_now(),
        "dataset_id": "claimforge-commercial-balanced250-mouse-reuse-v1",
        "provider": {
            "name": "Sightengine genai",
            "decision_rule": f"type.ai_generated >= {THRESHOLD}",
            "upload_policy": "exact benchmark raw JPEG/PNG; no resize/re-encode",
        },
        "coverage": {
            "new_expected": 822,
            "new_valid": len(new_valid),
            "new_errors": 822 - len(new_valid),
            "new_raw_attempt_rows": len(new_attempts),
            "new_superseded_attempt_rows": len(new_attempts) - len(new_latest),
            "new_operations_consumed": sum(
                int(row.get("operations") or 0) for row in new_valid
            ),
            "reused_expected": 428,
            "reused_valid": 178 + len(full_mouse),
            "total_planned": 1750,
            "deferred_images": 500,
            "total_evaluated_expected": 1250,
            "total_valid": len(new_valid) + 178 + len(full_mouse),
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
