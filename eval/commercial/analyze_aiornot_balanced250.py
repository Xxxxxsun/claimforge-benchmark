#!/usr/bin/env python3
"""Build the formal AI or Not Balanced250 table with reused mouse runs."""

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
DEFAULT_MISSING_RESULTS = Path(
    "results/commercial/aiornot/"
    "claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl"
)
DEFAULT_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/aiornot/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl"
)
DEFAULT_LOCAL_MOUSE_MANIFEST = Path(
    "results/commercial/aiornot/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.run_manifest.json"
)
DEFAULT_FULL_MOUSE_RESULTS = Path(
    "results/commercial/aiornot/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl"
)
DEFAULT_FULL_MOUSE_MANIFEST = Path(
    "results/commercial/aiornot/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.run_manifest.json"
)
DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_FULL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/full_image/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/aiornot/"
    "claimforge_balanced250_main_table_20260727.summary.json"
)
DEFAULT_REPORT = Path("docs/AIORNOT_BALANCED250_RESULTS_2026-07-27.md")


def positive(row: dict[str, Any]) -> bool:
    return bool(row.get("ai_detected"))


def confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["ai_confidence"])
        for row in rows
        if isinstance(row.get("ai_confidence"), (int, float))
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
        "# AI or Not: CLAIMFORGE Balanced250",
        "",
        f"- Run completed: `{summary['generated_at']}`",
        "- Endpoint/report: `/v2/image/sync`, `only=ai_generated`",
        "- Decision rule: vendor `ai_detected=true`.",
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
        f"- Complete evaluation: {summary['coverage']['total_valid']}/"
        f"{summary['coverage']['total_expected']} valid.",
        f"- Estimated new-call cost: USD "
        f"{summary['coverage']['estimated_new_call_cost_usd']:.2f}.",
        "",
        "## Raw Results",
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
            "## Main Finding",
            "",
            "AI or Not is almost perfectly specific on the shared authentic "
            "panel and detects the full-frame controls strongly, but its three "
            "local-splice sensitivities are near zero. Consequently, local "
            "balanced accuracy is approximately chance despite excellent "
            "full-frame balanced accuracy.",
            "",
            "## Reuse Audit",
            "",
            f"- Local mouse: "
            f"{summary['reuse_audit']['local_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            f"- Full-frame mouse: "
            f"{summary['reuse_audit']['fullframe_mouse']['raw_image_sha256_matches']}"
            "/250 frozen PNGs are byte-identical.",
            f"- Of "
            f"{summary['canonicalization_audit']['reference_available']} uploads "
            "with a frozen canonical JPEG reference, "
            f"{summary['canonicalization_audit']['canonical_matches']} are "
            "byte-identical and "
            f"{summary['canonicalization_audit']['canonical_mismatches']} were "
            "re-encoded by the current Pillow/libjpeg under the same Q95 policy.",
            f"- The raw new-results ledger has "
            f"{summary['coverage']['raw_attempt_rows']} rows because one rejected "
            "old-key preflight was retained; the later success for that ID is "
            "authoritative.",
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
    canonical = {
        (str(row["condition"]), str(row["task_id"])): str(row["canonical_sha256"])
        for row in panel_rows
    }

    missing_attempts = read_jsonl(missing_path)
    missing_latest = latest_by_id(missing_attempts)
    missing_valid = [
        row for row in missing_latest.values() if row.get("status") == "ok"
    ]
    if len(missing_valid) != 1250:
        raise ValueError(f"expected 1250 valid new rows, got {len(missing_valid)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in missing_valid:
        grouped.setdefault(str(row["condition"]), []).append(row)
    for condition in (
        "real",
        "local_cat",
        "local_trash_can",
        "fullframe_cat",
        "fullframe_trash_can",
    ):
        if len(grouped.get(condition, [])) != 250:
            raise ValueError(f"{condition}: expected 250 new rows")

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

    real_positive = [positive(row) for row in grouped["real"]]
    false_positives = sum(real_positive)
    specificity = 1 - false_positives / 250
    condition_metrics: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
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
            "ai_confidence": confidence_summary(rows),
            "generator_attribution_nonempty": sum(
                bool(row.get("generator_scores")) for row in rows
            ),
        }

    new_canonical_matches = sum(
        row.get("upload_sha256")
        == canonical[(str(row["condition"]), str(row["task_id"]))]
        for row in missing_valid
    )
    canonical_reference_available = (
        1250
        + local_audit["canonical_reference_available"]
        + full_audit["canonical_reference_available"]
    )
    total_canonical_matches = (
        new_canonical_matches
        + local_audit["canonical_upload_sha256_matches"]
        + full_audit["canonical_upload_sha256_matches"]
    )
    run_manifest_path = missing_path.with_suffix(".run_manifest.json")
    artifacts = {
        "new_results": {
            "path": relative(repo_root, missing_path),
            "sha256": sha256_file(missing_path),
        },
        "new_run_manifest": {
            "path": relative(repo_root, run_manifest_path),
            "sha256": sha256_file(run_manifest_path),
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
        "schema_version": "aiornot_balanced250_main_table_v1",
        "generated_at": utc_now(),
        "dataset_id": "claimforge-commercial-balanced250-mouse-reuse-v1",
        "input_selection": {
            "real_cat_trash": (
                "panel=true rows from "
                "claimforge-balanced250-independent-panel-jpeg-q95-v1"
            ),
            "mouse": (
                "frozen benchmark/claimforge_v1_250x3x2 panels, authorized reuse"
            ),
        },
        "provider": {
            "name": "AI or Not",
            "endpoint": "https://api.aiornot.com/v2/image/sync",
            "report_selection": ["ai_generated"],
            "decision_rule": "vendor ai_detected boolean",
        },
        "coverage": {
            "raw_attempt_rows": len(missing_attempts),
            "superseded_error_rows": len(missing_attempts) - len(missing_latest),
            "new_expected": 1250,
            "new_valid": len(missing_valid),
            "new_errors": 1250 - len(missing_valid),
            "reused_expected": 500,
            "reused_valid": len(local_mouse) + len(full_mouse),
            "total_expected": 1750,
            "total_valid": len(missing_valid) + len(local_mouse) + len(full_mouse),
            "estimated_new_call_cost_usd": len(missing_valid) * 0.02,
        },
        "real_panel": {
            "expected": 250,
            "valid": 250,
            "false_positives": false_positives,
            "true_negatives": 250 - false_positives,
            "false_positive_rate": false_positives / 250,
            "specificity": specificity,
            "ai_confidence": confidence_summary(grouped["real"]),
        },
        "conditions": condition_metrics,
        "reuse_audit": {
            "local_mouse": local_audit,
            "fullframe_mouse": full_audit,
        },
        "canonicalization_audit": {
            "total_uploads": 1750,
            "reference_available": canonical_reference_available,
            "reference_unavailable": 1750 - canonical_reference_available,
            "canonical_matches": total_canonical_matches,
            "canonical_mismatches": (
                canonical_reference_available - total_canonical_matches
            ),
            "new_matches": new_canonical_matches,
            "new_mismatches": 1250 - new_canonical_matches,
        },
        "bootstrap": {
            "method": "independent stratified image bootstrap of real and forged panels",
            "samples": args.bootstrap_samples,
            "seed_base": args.bootstrap_seed,
        },
        "artifacts": artifacts,
    }
    artifacts["summary"] = {"path": relative(repo_root, output_path)}
    write_json(output_path, summary)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
