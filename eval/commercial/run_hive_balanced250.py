#!/usr/bin/env python3
"""Run Hive on the real and local-edit CLAIMFORGE Balanced250 cells."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

from eval.commercial.run_alibaba_balanced250 import (
    DEFAULT_INPUTS,
    SelectedInput,
    load_inputs,
    parse_conditions,
    selection_digest,
)
from eval.commercial.run_hive import (
    DEFAULT_ENDPOINT,
    DEFAULT_THRESHOLD,
    classify,
)
from eval.commercial.run_illuminarty import (
    append_jsonl,
    read_latest,
    sha256_file,
    utc_now,
)


DEFAULT_CONDITIONS = ("real", "local_cat", "local_trash_can")
DEFAULT_OUTPUT = Path(
    "results/commercial/hive/"
    "claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl"
)
DEFERRED_CONDITIONS = ("fullframe_cat", "fullframe_trash_can")


def canonicalize_strict(
    source: Path, destination: Path, quality: int
) -> dict[str, Any]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        image.info.clear()
        image.save(
            destination,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
        )
    return {
        "upload_sha256": sha256_file(destination),
        "upload_bytes": destination.stat().st_size,
        "upload_size": [width, height],
    }


def ensure_run_manifest(
    output_path: Path,
    selected: list[SelectedInput],
    ledger_path: Path,
    repo_root: Path,
    endpoint: str,
    quality: int,
    threshold: float,
    run_id: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    conditions = list(dict.fromkeys(entry.condition for entry in selected))
    expected = {
        "schema_version": "hive_balanced250_run_manifest_v1",
        "run_id": run_id,
        "dataset_id": "claimforge-balanced250-independent-panel-jpeg-q95-v1",
        "selection": "panel=true",
        "conditions": conditions,
        "per_condition": len(selected) // len(conditions),
        "expected_images": len(selected),
        "deferred_conditions": list(DEFERRED_CONDITIONS),
        "input_digest": selection_digest(selected),
        "input_ledger": ledger_path.relative_to(repo_root).as_posix(),
        "input_ledger_sha256": sha256_file(ledger_path),
        "endpoint": endpoint,
        "model": "ai-generated-and-deepfake-content-detection",
        "threshold": threshold,
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "resize": False,
            "filename": "image.jpg",
        },
        "decision": f"ai_generated >= {threshold}",
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


def write_summary(
    output_path: Path,
    selected: list[SelectedInput],
    run_id: str,
    threshold: float,
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
            "positive": sum(
                float(row["ai_probability"]) >= threshold
                for row in condition_valid
            ),
            "ai_probability": numeric_summary(
                condition_valid, "ai_probability"
            ),
        }

    summary = {
        "schema_version": "hive_balanced250_summary_v1",
        "run_id": run_id,
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "threshold": threshold,
        "expected_images": len(selected),
        "completed_images": len(rows),
        "valid_images": len(valid),
        "error_images": len(rows) - len(valid),
        "remaining_images": len(selected) - len(valid),
        "canonical_hash_matches": sum(
            row.get("upload_matches_expected_canonical") is True for row in rows
        ),
        "canonical_hash_mismatches": sum(
            row.get("upload_matches_expected_canonical") is False for row in rows
        ),
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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--retry-canonical-mismatches",
        action="store_true",
        help="resubmit successful rows whose upload hash is not canonical",
    )
    parser.add_argument(
        "--run-id",
        default="hive_balanced250_real_local750_20260727",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.per_condition < 1:
        parser.error("--per-condition must be positive")
    if args.jpeg_quality != 95:
        parser.error("Balanced250 requires --jpeg-quality 95")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.max_attempts < 1 or args.workers < 1:
        parser.error("--max-attempts and --workers must be positive")
    if args.minimum_interval < 0:
        parser.error("--minimum-interval must be non-negative")
    if args.max_pending is not None and args.max_pending < 1:
        parser.error("--max-pending must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")

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
        or (
            args.retry_canonical_mismatches
            and latest[entry.item.id].get("upload_sha256")
            != entry.canonical_sha256
        )
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
                "workers": args.workers,
                "retry_canonical_mismatches": args.retry_canonical_mismatches,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    api_key = os.environ.get("HIVE_API_KEY", "")
    if not api_key:
        raise SystemExit("HIVE_API_KEY must be set")
    ensure_run_manifest(
        output_path,
        selected,
        ledger_path,
        repo_root,
        args.endpoint,
        args.jpeg_quality,
        args.threshold,
        args.run_id,
    )
    digest = selection_digest(selected)

    with tempfile.TemporaryDirectory(
        prefix="claimforge-hive-balanced250-"
    ) as temporary:
        temporary_dir = Path(temporary)

        def run_one(index: int, entry: SelectedInput) -> dict[str, Any]:
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload = canonicalize_strict(
                entry.item.path, upload_path, args.jpeg_quality
            )
            canonical_match = upload["upload_sha256"] == entry.canonical_sha256
            with requests.Session() as session:
                session.headers.update(
                    {"User-Agent": "claimforge-benchmark/hive-balanced250-v1"}
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
                    args.threshold,
                    args.run_id,
                    digest,
                )
            row.update(
                {
                    "schema_version": "hive_balanced250_result_v1",
                    "sample_id": entry.sample_id,
                    "condition": entry.condition,
                    "condition_family": entry.condition_family,
                    "selection_rank": entry.selection_rank,
                    "source_content_cluster": entry.source_content_cluster,
                }
            )
            return row

        indexed_pending = iter(enumerate(pending))
        futures: dict[Future[dict[str, Any]], tuple[int, SelectedInput]] = {}
        stop_submitting = False
        completed_this_run = 0

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            try:
                index, entry = next(indexed_pending)
            except StopIteration:
                return False
            futures[executor.submit(run_one, index, entry)] = (index, entry)
            if args.minimum_interval > 0:
                time.sleep(args.minimum_interval)
            return True

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(min(args.workers, len(pending))):
                submit_next(executor)
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    _, entry = futures.pop(future)
                    row = future.result()
                    append_jsonl(output_path, row)
                    completed_this_run += 1
                    if (
                        completed_this_run == 1
                        or completed_this_run % args.progress_every == 0
                        or completed_this_run == len(pending)
                        or row["status"] != "ok"
                    ):
                        print(
                            json.dumps(
                                {
                                    "scheduled_progress": (
                                        f"{completed_this_run}/{len(pending)}"
                                    ),
                                    "condition": entry.condition,
                                    "id": row["id"],
                                    "status": row["status"],
                                    "ai_probability": row.get("ai_probability"),
                                    "detected": row.get("detected"),
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
                        if final.get("http_status") in {401, 402, 403, 405, 429}:
                            stop_submitting = True
                            print(
                                "authentication, credit, or quota failure; "
                                "stopping new submissions",
                                flush=True,
                            )
                    if not stop_submitting:
                        submit_next(executor)

    latest = read_latest(output_path)
    unexpected = sorted(set(latest) - {entry.item.id for entry in selected})
    if unexpected:
        raise ValueError(f"result file contains unexpected IDs: {unexpected[:5]}")
    summary = write_summary(
        output_path, selected, args.run_id, args.threshold
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
