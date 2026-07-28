#!/usr/bin/env python3
"""Run AI or Not on the missing CLAIMFORGE Balanced250 commercial cells."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from eval.commercial.run_aiornot import DEFAULT_ENDPOINT, classify
from eval.commercial.run_alibaba_balanced250 import (
    DEFAULT_CONDITIONS,
    DEFAULT_INPUTS,
    SelectedInput,
    load_inputs,
    parse_conditions,
    selection_digest,
)
from eval.commercial.run_hive import canonicalize
from eval.commercial.run_illuminarty import (
    append_jsonl,
    read_latest,
    sha256_file,
    utc_now,
)


DEFAULT_OUTPUT = Path(
    "results/commercial/aiornot/"
    "claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl"
)


def ensure_run_manifest(
    output_path: Path,
    selected: list[SelectedInput],
    ledger_path: Path,
    repo_root: Path,
    endpoint: str,
    quality: int,
    run_id: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    conditions = list(dict.fromkeys(entry.condition for entry in selected))
    expected = {
        "schema_version": "aiornot_balanced250_run_manifest_v1",
        "run_id": run_id,
        "dataset_id": "claimforge-balanced250-independent-panel-jpeg-q95-v1",
        "selection": "panel=true",
        "conditions": conditions,
        "per_condition": len(selected) // len(conditions),
        "expected_images": len(selected),
        "input_digest": selection_digest(selected),
        "input_ledger": ledger_path.relative_to(repo_root).as_posix(),
        "input_ledger_sha256": sha256_file(ledger_path),
        "endpoint": endpoint,
        "reports": ["ai_generated"],
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "resize": False,
            "filename": "image.jpg",
        },
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"run manifest mismatch: {mismatches}")
        return

    payload = {
        **expected,
        "created_at": utc_now(),
        "adapter": {
            "path": Path(__file__).relative_to(repo_root).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "dependencies": {
            "requests": requests.__version__,
            "pillow": Image.__version__,
        },
        "ordered_inputs": [
            {
                "rank": rank,
                "id": entry.item.id,
                "sample_id": entry.sample_id,
                "task_id": entry.item.task_id,
                "domain": entry.item.domain,
                "kind": entry.item.kind,
                "condition": entry.condition,
                "condition_family": entry.condition_family,
                "selection_rank": entry.selection_rank,
                "source_content_cluster": entry.source_content_cluster,
                "image_path": entry.item.relative_path,
                "image_sha256": entry.item.sha256,
                "canonical_sha256": entry.canonical_sha256,
                "file_bytes": entry.item.file_bytes,
            }
            for rank, entry in enumerate(selected)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def score_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["ai_confidence"])
        for row in rows
        if isinstance(row.get("ai_confidence"), (int, float))
    ]
    detected = sum(bool(row.get("ai_detected")) for row in rows)
    return {
        "count": len(values),
        "detected": detected,
        "detection_rate": detected / len(rows) if rows else None,
        "confidence": {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        },
    }


def write_summary(
    output_path: Path,
    selected: list[SelectedInput],
    run_id: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    expected_by_id = {entry.item.id: entry for entry in selected}
    rows = [latest[identifier] for identifier in expected_by_id if identifier in latest]
    valid = [row for row in rows if row.get("status") == "ok"]
    conditions: dict[str, Any] = {}
    for condition in dict.fromkeys(entry.condition for entry in selected):
        identifiers = {
            entry.item.id for entry in selected if entry.condition == condition
        }
        condition_rows = [row for row in rows if row["id"] in identifiers]
        condition_valid = [
            row for row in condition_rows if row.get("status") == "ok"
        ]
        conditions[condition] = {
            "expected": len(identifiers),
            "completed": len(condition_rows),
            "valid": len(condition_valid),
            "errors": len(condition_rows) - len(condition_valid),
            **score_stats(condition_valid),
        }

    summary = {
        "schema_version": "aiornot_balanced250_summary_v1",
        "run_id": run_id,
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "expected_images": len(selected),
        "completed_images": len(rows),
        "valid_images": len(valid),
        "error_images": len(rows) - len(valid),
        "remaining_images": len(selected) - len(valid),
        "canonical_hash_mismatches": sum(
            row.get("upload_matches_expected_canonical") is False for row in rows
        ),
        "estimated_successful_call_cost_usd": round(len(valid) * 0.02, 2),
        "conditions": conditions,
        "http_error_counts": dict(
            Counter(
                str((row.get("attempts") or [{}])[-1].get("http_status"))
                for row in rows
                if row.get("status") == "error"
            )
        ),
    }
    path = output_path.with_suffix(".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=DEFAULT_CONDITIONS,
        help="comma-separated canonical condition names",
    )
    parser.add_argument("--per-condition", type=int, default=250)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=1.1)
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--allow-canonical-mismatch",
        action="store_true",
        help=(
            "continue when the current Pillow/libjpeg Q95 encoding is not "
            "byte-identical to the frozen canonical JPEG"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="aiornot_balanced250_missing1250_20260727",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.per_condition < 1:
        parser.error("--per-condition must be positive")
    if args.max_attempts < 1 or args.minimum_interval < 0:
        parser.error("max attempts must be positive and interval non-negative")
    if args.max_pending is not None and args.max_pending < 1:
        parser.error("--max-pending must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.jpeg_quality != 95:
        parser.error("Balanced250 requires --jpeg-quality 95")

    repo_root = args.repo_root.resolve()
    ledger_path = (
        args.inputs if args.inputs.is_absolute() else repo_root / args.inputs
    ).resolve()
    output_path = (
        args.output if args.output.is_absolute() else repo_root / args.output
    ).resolve()
    selected = load_inputs(
        repo_root,
        ledger_path,
        args.conditions,
        args.per_condition,
    )
    latest = read_latest(output_path)
    pending = [
        entry
        for entry in selected
        if latest.get(entry.item.id, {}).get("status") != "ok"
    ]
    pending_total = len(pending)
    if args.max_pending is not None:
        pending = pending[: args.max_pending]
    print(
        json.dumps(
            {
                "conditions": args.conditions,
                "expected_images": len(selected),
                "already_valid": len(selected) - pending_total,
                "pending": pending_total,
                "scheduled": len(pending),
                "output": output_path.relative_to(repo_root).as_posix(),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    api_key = os.environ.get("AIORNOT_API_KEY", "")
    if not api_key:
        raise SystemExit("AIORNOT_API_KEY must be set")
    ensure_run_manifest(
        output_path,
        selected,
        ledger_path,
        repo_root,
        args.endpoint,
        args.jpeg_quality,
        args.run_id,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "claimforge-benchmark/aiornot-v1"})

    with tempfile.TemporaryDirectory(prefix="claimforge-aiornot-balanced250-") as temp:
        temporary_dir = Path(temp)
        for index, entry in enumerate(pending, start=1):
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload = canonicalize(entry.item.path, upload_path, args.jpeg_quality)
            canonical_match = upload["upload_sha256"] == entry.canonical_sha256
            if not canonical_match and not args.allow_canonical_mismatch:
                raise ValueError(
                    f"{entry.item.id}: canonical SHA-256 mismatch "
                    f"({upload['upload_sha256']} != {entry.canonical_sha256})"
                )
            row = classify(
                session,
                entry.item,
                upload_path,
                {
                    **upload,
                    "expected_canonical_sha256": entry.canonical_sha256,
                    "upload_matches_expected_canonical": canonical_match,
                },
                args.endpoint,
                api_key,
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                args.run_id,
                selection_digest(selected),
            )
            row.update(
                {
                    "schema_version": "aiornot_balanced250_result_v1",
                    "sample_id": entry.sample_id,
                    "condition": entry.condition,
                    "condition_family": entry.condition_family,
                    "selection_rank": entry.selection_rank,
                    "source_content_cluster": entry.source_content_cluster,
                }
            )
            append_jsonl(output_path, row)

            if (
                index == 1
                or index % args.progress_every == 0
                or index == len(pending)
                or row["status"] != "ok"
            ):
                print(
                    json.dumps(
                        {
                            "scheduled_progress": f"{index}/{len(pending)}",
                            "condition": entry.condition,
                            "id": entry.item.id,
                            "status": row["status"],
                            "ai_detected": row.get("ai_detected"),
                            "ai_confidence": row.get("ai_confidence"),
                            "error": (
                                (row.get("attempts") or [{}])[-1].get(
                                    "error_message"
                                )
                                if row["status"] == "error"
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if row["status"] == "error":
                final = (row.get("attempts") or [{}])[-1]
                if final.get("http_status") in {401, 402, 403, 429}:
                    break
            if index < len(pending):
                time.sleep(args.minimum_interval)

    latest = read_latest(output_path)
    unexpected = sorted(
        set(latest) - {entry.item.id for entry in selected}
    )
    if unexpected:
        raise ValueError(f"result file contains unexpected IDs: {unexpected[:5]}")
    summary = write_summary(output_path, selected, args.run_id)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
