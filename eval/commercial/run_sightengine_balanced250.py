#!/usr/bin/env python3
"""Run Sightengine on missing core CLAIMFORGE Balanced250 images."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from eval.commercial.run_alibaba_balanced250 import (
    DEFAULT_INPUTS,
    SelectedInput,
    load_inputs,
    read_jsonl,
)
from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    input_digest,
    read_latest,
    sha256_file,
    utc_now,
)
from eval.commercial.run_sightengine import (
    AUTH_OR_QUOTA_HTTP,
    DEFAULT_ENDPOINT,
    classify,
)


DEFAULT_LOCAL_MOUSE_BENCHMARK = Path(
    "benchmark/claimforge_v1_250x3x2/local_splice/mouse/manifest.jsonl"
)
DEFAULT_PRIOR_LOCAL_MOUSE_RESULTS = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/sightengine/"
    "claimforge_balanced250_core822_original_files_20260727.jsonl"
)
CORE_CONDITIONS = ("real", "local_cat", "local_trash_can")
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Submission:
    item: ImageItem
    condition: str
    sample_id: str
    selection_rank: int
    source_content_cluster: str
    reuse_reason: str | None = None


def resolve_repo_image(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    path.relative_to(repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"missing input image: {relative_path}")
    return path


def load_missing_local_mouse(
    repo_root: Path,
    benchmark_path: Path,
    prior_results_path: Path,
) -> tuple[list[Submission], dict[str, Any]]:
    benchmark_rows = read_jsonl(benchmark_path)
    if len(benchmark_rows) != 250:
        raise ValueError(f"expected 250 local-mouse rows, got {len(benchmark_rows)}")
    prior_latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(prior_results_path):
        identifier = row.get("id")
        if isinstance(identifier, str):
            prior_latest[identifier] = row
    prior_by_task = {
        str(row["task_id"]): row
        for row in prior_latest.values()
        if row.get("status") == "ok"
    }

    missing: list[Submission] = []
    reusable = 0
    for rank, row in enumerate(benchmark_rows):
        task_id = str(row["task_id"])
        expected_sha256 = str(row["sha256"])
        prior = prior_by_task.get(task_id)
        if prior is not None:
            if prior.get("image_sha256") != expected_sha256:
                raise ValueError(f"{task_id}: prior raw image SHA-256 mismatch")
            reusable += 1
            continue

        relative_path = str(row["image"])
        path = resolve_repo_image(repo_root, relative_path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{task_id}: benchmark image SHA-256 mismatch")
        raw_size = row.get("image_size")
        if not isinstance(raw_size, dict):
            raise ValueError(f"{task_id}: missing image_size")
        missing.append(
            Submission(
                item=ImageItem(
                    id=f"{task_id}__forged",
                    task_id=task_id,
                    domain=str(row["domain"]),
                    kind="forged",
                    label="edited",
                    path=path,
                    relative_path=relative_path,
                    image_size=(
                        int(raw_size["width"]),
                        int(raw_size["height"]),
                    ),
                    sha256=actual_sha256,
                    file_bytes=path.stat().st_size,
                ),
                condition="local_mouse",
                sample_id=str(row.get("sample_id") or task_id),
                selection_rank=rank,
                source_content_cluster=str(row.get("source_content_cluster") or ""),
                reuse_reason=None,
            )
        )
    if reusable + len(missing) != 250:
        raise ValueError("local-mouse reuse accounting mismatch")
    return missing, {
        "benchmark_images": 250,
        "reused_prior_valid": reusable,
        "newly_required": len(missing),
        "prior_results": prior_results_path.relative_to(repo_root).as_posix(),
        "prior_results_sha256": sha256_file(prior_results_path),
    }


def convert_selected(entry: SelectedInput) -> Submission:
    return Submission(
        item=entry.item,
        condition=entry.condition,
        sample_id=entry.sample_id,
        selection_rank=entry.selection_rank,
        source_content_cluster=entry.source_content_cluster,
    )


def ensure_run_manifest(
    output_path: Path,
    submissions: list[Submission],
    ledger_path: Path,
    local_mouse_benchmark_path: Path,
    local_mouse_reuse: dict[str, Any],
    repo_root: Path,
    endpoint: str,
    run_id: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    digest = input_digest([entry.item for entry in submissions])
    expected = {
        "schema_version": "sightengine_balanced250_run_manifest_v1",
        "run_id": run_id,
        "dataset_id": "claimforge-balanced250-independent-panel-original-files-v1",
        "conditions": ["real", "local_mouse", "local_cat", "local_trash_can"],
        "submitted_by_condition": dict(
            Counter(entry.condition for entry in submissions)
        ),
        "expected_images": len(submissions),
        "deferred_conditions": ["fullframe_cat", "fullframe_trash_can"],
        "input_digest": digest,
        "input_ledger": ledger_path.relative_to(repo_root).as_posix(),
        "input_ledger_sha256": sha256_file(ledger_path),
        "local_mouse_benchmark": (
            local_mouse_benchmark_path.relative_to(repo_root).as_posix()
        ),
        "local_mouse_benchmark_sha256": sha256_file(
            local_mouse_benchmark_path
        ),
        "local_mouse_reuse": local_mouse_reuse,
        "endpoint": endpoint,
        "model": "genai",
        "decision": f"ai_generated >= {DEFAULT_THRESHOLD}",
        "expected_operations_per_image": 5,
        "upload": {
            "policy": "exact benchmark raw file",
            "formats": ["JPEG", "PNG"],
            "metadata": "unchanged; provider genai model documented as pixel-based",
            "resize": False,
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
        "requests_version": requests.__version__,
        "ordered_inputs": [
            {
                "rank": rank,
                "id": entry.item.id,
                "sample_id": entry.sample_id,
                "task_id": entry.item.task_id,
                "domain": entry.item.domain,
                "kind": entry.item.kind,
                "condition": entry.condition,
                "selection_rank": entry.selection_rank,
                "source_content_cluster": entry.source_content_cluster,
                "image_path": entry.item.relative_path,
                "image_sha256": entry.item.sha256,
                "file_bytes": entry.item.file_bytes,
            }
            for rank, entry in enumerate(submissions)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "positive": sum(value >= DEFAULT_THRESHOLD for value in values),
    }


def write_summary(
    output_path: Path,
    submissions: list[Submission],
    run_id: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    expected_by_id = {entry.item.id: entry for entry in submissions}
    rows = [latest[identifier] for identifier in expected_by_id if identifier in latest]
    valid = [row for row in rows if row.get("status") == "ok"]
    conditions: dict[str, Any] = {}
    for condition in dict.fromkeys(entry.condition for entry in submissions):
        identifiers = {
            entry.item.id for entry in submissions if entry.condition == condition
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
            "score": numeric_summary(condition_valid),
            "operations": sum(
                int(row.get("operations") or 0) for row in condition_valid
            ),
        }
    summary = {
        "schema_version": "sightengine_balanced250_summary_v1",
        "run_id": run_id,
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "expected_images": len(submissions),
        "completed_images": len(rows),
        "valid_images": len(valid),
        "error_images": len(rows) - len(valid),
        "remaining_images": len(submissions) - len(valid),
        "operations_consumed": sum(
            int(row.get("operations") or 0) for row in valid
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
    parser.add_argument(
        "--local-mouse-benchmark",
        type=Path,
        default=DEFAULT_LOCAL_MOUSE_BENCHMARK,
    )
    parser.add_argument(
        "--prior-local-mouse-results",
        type=Path,
        default=DEFAULT_PRIOR_LOCAL_MOUSE_RESULTS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-successes", type=int, default=822)
    parser.add_argument("--operation-budget", type=int, default=4_110)
    parser.add_argument("--expected-operations-per-image", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--run-id",
        default="sightengine_balanced250_core822_20260727",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (
        args.max_successes < 0
        or args.operation_budget < 0
        or args.max_attempts < 1
        or args.minimum_interval < 0
    ):
        parser.error("limits/interval must be non-negative; attempts must be positive")
    if args.expected_operations_per_image < 1:
        parser.error("--expected-operations-per-image must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")

    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    ledger_path = resolve(args.inputs)
    local_mouse_benchmark_path = resolve(args.local_mouse_benchmark)
    prior_local_mouse_results_path = resolve(args.prior_local_mouse_results)
    output_path = resolve(args.output)

    core = [
        convert_selected(entry)
        for entry in load_inputs(
            repo_root,
            ledger_path,
            CORE_CONDITIONS,
            250,
        )
    ]
    missing_mouse, local_mouse_reuse = load_missing_local_mouse(
        repo_root,
        local_mouse_benchmark_path,
        prior_local_mouse_results_path,
    )
    submissions = core[:250] + missing_mouse + core[250:]
    if len(submissions) != 822:
        raise ValueError(f"expected 822 submissions, got {len(submissions)}")
    ids = [entry.item.id for entry in submissions]
    if len(ids) != len(set(ids)):
        raise ValueError("submission IDs are not unique")

    latest = read_latest(output_path)
    pending = [
        entry
        for entry in submissions
        if latest.get(entry.item.id, {}).get("status") != "ok"
    ]
    print(
        json.dumps(
            {
                "expected_images": len(submissions),
                "submitted_by_condition": dict(
                    Counter(entry.condition for entry in submissions)
                ),
                "local_mouse_reused": local_mouse_reuse["reused_prior_valid"],
                "already_valid": len(submissions) - len(pending),
                "pending": len(pending),
                "max_successes": args.max_successes,
                "operation_budget": args.operation_budget,
                "output": output_path.relative_to(repo_root).as_posix(),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    api_user = os.environ.get("SIGHTENGINE_API_USER", "")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET", "")
    if not api_user or not api_secret:
        raise SystemExit("SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET must be set")
    ensure_run_manifest(
        output_path,
        submissions,
        ledger_path,
        local_mouse_benchmark_path,
        local_mouse_reuse,
        repo_root,
        args.endpoint,
        args.run_id,
    )

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "claimforge-benchmark/sightengine-balanced250-v1"}
    )
    digest = input_digest([entry.item for entry in submissions])
    successes = failures = operations = 0
    last_started = 0.0
    for entry in pending:
        if successes >= args.max_successes:
            break
        if operations + args.expected_operations_per_image > args.operation_budget:
            break
        wait_for = args.minimum_interval - (time.monotonic() - last_started)
        if wait_for > 0:
            time.sleep(wait_for)
        last_started = time.monotonic()
        row = classify(
            session,
            entry.item,
            api_user,
            api_secret,
            args.endpoint,
            (args.connect_timeout, args.read_timeout),
            args.max_attempts,
            args.run_id,
            entry.condition,
            digest,
        )
        row.update(
            {
                "schema_version": "sightengine_balanced250_result_v1",
                "sample_id": entry.sample_id,
                "selection_rank": entry.selection_rank,
                "source_content_cluster": entry.source_content_cluster,
            }
        )
        append_jsonl(output_path, row)
        if row["status"] == "ok":
            successes += 1
            operations += int(row.get("operations") or 0)
        else:
            failures += 1
        if (
            successes + failures == 1
            or (successes + failures) % args.progress_every == 0
            or row["status"] != "ok"
        ):
            final = (row.get("attempts") or [{}])[-1]
            print(
                json.dumps(
                    {
                        "attempted_this_run": successes + failures,
                        "successes_this_run": successes,
                        "failures_this_run": failures,
                        "operations_this_run": operations,
                        "condition": entry.condition,
                        "id": row["id"],
                        "status": row["status"],
                        "ai_probability": row.get("ai_probability"),
                        "http_status": row.get("http_status")
                        or final.get("http_status"),
                        "error": (
                            final.get("error_message")
                            if row["status"] == "error"
                            else None
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        final = (row.get("attempts") or [{}])[-1]
        final_http = final.get("http_status")
        final_error = (
            str(final.get("error_type") or "").lower()
            + " "
            + str(final.get("error_message") or "").lower()
        )
        if final_http in AUTH_OR_QUOTA_HTTP or any(
            marker in final_error
            for marker in (
                "auth",
                "credential",
                "quota",
                "operation limit",
                "usage_limit",
                "usage limit",
            )
        ):
            print("authentication/quota failure detected; stopping batch", flush=True)
            break
        if (
            row["status"] == "ok"
            and row.get("operations") != args.expected_operations_per_image
        ):
            print(
                f"operation cost changed to {row.get('operations')}; stopping",
                flush=True,
            )
            break

    summary = write_summary(output_path, submissions, args.run_id)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
