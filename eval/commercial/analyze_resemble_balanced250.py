#!/usr/bin/env python3
"""Build the Resemble Balanced250 table with reused mouse runs."""

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
    "results/commercial/resemble/"
    "claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl"
)
DEFAULT_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/resemble/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl"
)
DEFAULT_LOCAL_MOUSE_MANIFEST = Path(
    "results/commercial/resemble/"
    "good275_mouse_forged_canonical_jpeg_q95_20260720.run_manifest.json"
)
DEFAULT_FULL_MOUSE_RESULTS = Path(
    "results/commercial/resemble/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl"
)
DEFAULT_FULL_MOUSE_MANIFEST = Path(
    "results/commercial/resemble/"
    "claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.run_manifest.json"
)
DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_FULL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/full_image/mouse/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/resemble/"
    "claimforge_balanced250_main_table_20260727.summary.json"
)
DEFAULT_REPORT = Path("docs/RESEMBLE_BALANCED250_RESULTS_2026-07-27.md")
DEFERRED_CONDITIONS = ("fullframe_cat", "fullframe_trash_can")


def positive(row: dict[str, Any]) -> bool:
    label = str(row.get("provider_label") or "").strip().lower()
    return label in {"fake", "likely fake"}


def numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
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
        "# Resemble Detect: CLAIMFORGE Balanced250",
        "",
        f"- Run completed: `{summary['generated_at']}`",
        "- Decision rule: provider label is `Fake` or `Likely fake`.",
        "- Upload policy: metadata-free RGB JPEG Q95 with 4:4:4 subsampling.",
        "- Resemble heatmaps are retained as provider-rendered RGB/JPEG "
        "artifacts; they are not interpreted as calibrated pixel scores.",
        "",
        "## Coverage",
        "",
        f"- New panel: {summary['coverage']['new_valid']}/"
        f"{summary['coverage']['new_expected']} valid, "
        f"{summary['coverage']['new_errors']} final errors.",
        f"- Raw API attempts: {summary['coverage']['new_raw_attempt_rows']}; "
        f"{summary['coverage']['superseded_error_rows']} provider-processing "
        "failure was superseded by a successful retry.",
        f"- Reused mouse calls: {summary['coverage']['reused_valid']}/"
        f"{summary['coverage']['reused_expected']} valid.",
        f"- Evaluated primary cells: {summary['coverage']['total_valid']}/"
        f"{summary['coverage']['total_planned']} images.",
        f"- Deferred before submission: "
        f"{summary['coverage']['deferred_images']} full-frame cat/trash-can images.",
        f"- Saved provider artifacts in the new run: "
        f"{summary['coverage']['new_heatmaps']} heatmaps and "
        f"{summary['coverage']['new_visualizations']} visualizations.",
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
            "## Labels",
            "",
            "| Condition | Label counts |",
            "|---|---|",
        ]
    )
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        if row.get("status") == "deferred":
            lines.append(f"| `{condition}` | deferred |")
            continue
        labels = ", ".join(
            f"{label}: {count}"
            for label, count in sorted(row["provider_labels"].items())
        )
        lines.append(f"| `{condition}` | {labels} |")
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
            "provider_labels": dict(
                Counter(str(row.get("provider_label")) for row in rows)
            ),
            "provider_score": numeric_summary(rows, "provider_score"),
            "ifl_score": numeric_summary(rows, "ifl_score"),
            "heatmaps_available": sum(bool(row.get("heatmap")) for row in rows),
            "visualizations_available": sum(
                bool(row.get("visualization")) for row in rows
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
            "path": relative(
                repo_root, new_results_path.with_suffix(".run_manifest.json")
            ),
            "sha256": sha256_file(
                new_results_path.with_suffix(".run_manifest.json")
            ),
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
        "schema_version": "resemble_detect_balanced250_main_table_v1",
        "generated_at": utc_now(),
        "dataset_id": "claimforge-commercial-balanced250-mouse-reuse-v1",
        "provider": {
            "name": "Resemble Detect",
            "decision_rule": "provider label is Fake or Likely fake",
            "upload_policy": "metadata-free RGB JPEG Q95 4:4:4",
            "heatmap_interpretation": (
                "provider-rendered RGB/JPEG artifact; not a calibrated score map"
            ),
        },
        "coverage": {
            "new_expected": 750,
            "new_valid": len(new_valid),
            "new_errors": 750 - len(new_valid),
            "new_raw_attempt_rows": len(new_attempts),
            "superseded_error_rows": len(new_attempts) - len(new_latest),
            "new_heatmaps": sum(bool(row.get("heatmap")) for row in new_valid),
            "new_visualizations": sum(
                bool(row.get("visualization")) for row in new_valid
            ),
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
            "provider_labels": dict(
                Counter(str(row.get("provider_label")) for row in grouped["real"])
            ),
            "provider_score": numeric_summary(grouped["real"], "provider_score"),
            "ifl_score": numeric_summary(grouped["real"], "ifl_score"),
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
