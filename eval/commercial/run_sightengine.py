#!/usr/bin/env python3
"""Run Sightengine ``genai`` on reviewed CLAIMFORGE images.

Credentials are read only from ``SIGHTENGINE_API_USER`` and
``SIGHTENGINE_API_SECRET``. They are never accepted on the command line or
written to result artifacts. The default run covers the complete good-mouse
forged pool, while ``--max-successes`` can spend a bounded daily quota and
resume later from the same JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    as_probability,
    input_digest,
    load_items,
    quantile,
    read_latest,
    redact_payload,
    redact_text,
    sha256_file,
    utc_now,
)


DEFAULT_ENDPOINT = "https://api.sightengine.com/1.0/check.json"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_OUTPUT = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 500, 502, 503, 504}
AUTH_OR_QUOTA_HTTP = {401, 403, 429}


def stable_domain_order(items: list[ImageItem]) -> list[ImageItem]:
    """Interleave hash-sorted domain/orientation strata for balanced prefixes."""
    grouped: dict[tuple[str, str], list[ImageItem]] = defaultdict(list)
    for item in items:
        if item.image_size is None:
            orientation = "unknown"
        else:
            width, height = item.image_size
            orientation = "landscape" if width > height else "portrait" if height > width else "square"
        grouped[(item.domain, orientation)].append(item)
    for stratum in grouped:
        grouped[stratum].sort(
            key=lambda item: hashlib.sha256(
                f"claimforge-sightengine-v1\0{item.id}".encode()
            ).hexdigest()
        )

    strata = sorted(grouped)
    total = len(items)
    targets = {stratum: len(grouped[stratum]) for stratum in strata}
    used = Counter()
    offsets = Counter()
    ordered: list[ImageItem] = []
    for position in range(1, total + 1):
        available = [stratum for stratum in strata if offsets[stratum] < targets[stratum]]
        stratum = max(
            available,
            key=lambda name: (
                position * targets[name] / total - used[name],
                -strata.index(name),
            ),
        )
        ordered.append(grouped[stratum][offsets[stratum]])
        offsets[stratum] += 1
        used[stratum] += 1
    return ordered


def parse_success(body: dict[str, Any]) -> dict[str, Any] | None:
    if body.get("status") != "success":
        return None
    result_type = body.get("type")
    if not isinstance(result_type, dict):
        return None
    score = as_probability(result_type.get("ai_generated"))
    if score is None:
        return None
    request = body.get("request") if isinstance(body.get("request"), dict) else {}
    operations = request.get("operations")
    if isinstance(operations, bool) or not isinstance(operations, int) or operations < 0:
        operations = None
    generators = result_type.get("ai_generators")
    return {
        "api_status": "success",
        "ai_probability": score,
        "operations": operations,
        "generator_scores": generators if isinstance(generators, dict) else None,
    }


def error_details(body: Any, response_text: str, secret: str) -> dict[str, Any]:
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
    return {
        "error_type": error.get("type") or error.get("code"),
        "error_message": redact_text(str(error.get("message") or response_text[:1000]), secret),
    }


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, min(120.0, float(raw)))
            except ValueError:
                pass
    return min(60.0, 2 ** (attempt - 1) + random.random())


def classify(
    session: requests.Session,
    item: ImageItem,
    api_user: str,
    api_secret: str,
    endpoint: str,
    timeout: tuple[float, float],
    max_attempts: int,
    run_id: str,
    condition: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        response: requests.Response | None = None
        started = time.monotonic()
        try:
            with item.path.open("rb") as image_handle:
                response = session.post(
                    endpoint,
                    data={
                        "models": "genai",
                        "api_user": api_user,
                        "api_secret": api_secret,
                    },
                    files={"media": ("image.png", image_handle, "image/png")},
                    timeout=timeout,
                    allow_redirects=False,
                )
            latency_ms = round((time.monotonic() - started) * 1000)
            try:
                body: Any = response.json()
            except ValueError:
                body = None
            parsed = parse_success(body) if isinstance(body, dict) else None
            if response.status_code == 200 and parsed is not None:
                return {
                    "schema_version": "sightengine_result_v1",
                    "run_id": run_id,
                    "condition": condition,
                    "input_manifest_sha256": manifest_sha256,
                    "id": item.id,
                    "task_id": item.task_id,
                    "domain": item.domain,
                    "kind": item.kind,
                    "label": item.label,
                    "image_path": item.relative_path,
                    "image_sha256": item.sha256,
                    "image_size": list(item.image_size) if item.image_size else None,
                    "file_bytes": item.file_bytes,
                    "status": "ok",
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "attempt_count": attempt,
                    "attempts": attempts,
                    **parsed,
                    "raw_response": redact_payload(body),
                    "completed_at": utc_now(),
                }

            details = error_details(body, response.text, api_secret)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    **details,
                }
            )
            if response.status_code not in RETRYABLE_HTTP:
                break
        except requests.RequestException as exc:
            latency_ms = round((time.monotonic() - started) * 1000)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code if response is not None else None,
                    "latency_ms": latency_ms,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), api_secret),
                }
            )
        if attempt < max_attempts:
            time.sleep(retry_delay(attempt, response))

    return {
        "schema_version": "sightengine_result_v1",
        "run_id": run_id,
        "condition": condition,
        "input_manifest_sha256": manifest_sha256,
        "id": item.id,
        "task_id": item.task_id,
        "domain": item.domain,
        "kind": item.kind,
        "label": item.label,
        "image_path": item.relative_path,
        "image_sha256": item.sha256,
        "image_size": list(item.image_size) if item.image_size else None,
        "file_bytes": item.file_bytes,
        "status": "error",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "completed_at": utc_now(),
    }


def ensure_run_manifest(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    endpoint: str,
    run_id: str,
    condition: str,
) -> Path:
    manifest_path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "sightengine_run_manifest_v1",
        "run_id": run_id,
        "condition": condition,
        "endpoint": endpoint,
        "model": "genai",
        "candidate": "mouse",
        "include": "forged",
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
        "upload_filename": "image.png",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"run manifest mismatch: {mismatches}")
        return manifest_path

    payload = {
        **expected,
        "created_at": utc_now(),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "requests_version": requests.__version__,
        "ordered_inputs": [
            {
                "rank": rank,
                "id": item.id,
                "task_id": item.task_id,
                "domain": item.domain,
                "image_path": item.relative_path,
                "image_sha256": item.sha256,
                "file_bytes": item.file_bytes,
            }
            for rank, item in enumerate(items)
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(
        float(row["ai_probability"])
        for row in rows
        if row.get("status") == "ok" and as_probability(row.get("ai_probability")) is not None
    )
    detected = sum(score >= 0.5 for score in scores)
    return {
        "count": len(scores),
        "min": min(scores) if scores else None,
        "p05": quantile(scores, 0.05),
        "p25": quantile(scores, 0.25),
        "median": statistics.median(scores) if scores else None,
        "mean": statistics.fmean(scores) if scores else None,
        "p75": quantile(scores, 0.75),
        "p95": quantile(scores, 0.95),
        "max": max(scores) if scores else None,
        "detected_at_0_5": detected,
        "detection_rate_at_0_5": detected / len(scores) if scores else None,
        "escaped_at_0_5": len(scores) - detected,
        "escape_rate_at_0_5": (len(scores) - detected) / len(scores) if scores else None,
        "below_0_01": sum(score < 0.01 for score in scores),
        "below_0_05": sum(score < 0.05 for score in scores),
        "below_0_10": sum(score < 0.10 for score in scores),
    }


def write_summary(output_path: Path, items: list[ImageItem], manifest_sha256: str) -> dict[str, Any]:
    latest = read_latest(output_path)
    rows = [latest[item.id] for item in items if item.id in latest]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") == "error"]
    summary = {
        "schema_version": "sightengine_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "expected_images": len(items),
        "completed_images": len(rows),
        "valid_images": len(ok_rows),
        "coverage": len(ok_rows) / len(items) if items else None,
        "error_images": len(error_rows),
        "operations_consumed": sum(
            int(row.get("operations") or 0) for row in ok_rows
        ),
        "score": score_summary(ok_rows),
        "by_domain": {
            domain: score_summary([row for row in ok_rows if row.get("domain") == domain])
            for domain in sorted({item.domain for item in items})
        },
        "http_error_counts": dict(
            Counter(
                str((row.get("attempts") or [{}])[-1].get("http_status"))
                for row in error_rows
            )
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--expected", type=int, default=275)
    parser.add_argument("--max-successes", type=int, default=99)
    parser.add_argument("--operation-budget", type=int, default=495)
    parser.add_argument("--expected-operations-per-image", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=1.05)
    parser.add_argument("--run-id", default="sightengine_pilot_good275_mouse_forged_20260720")
    parser.add_argument("--condition", default="pilot_good275_mouse_forged_original_png")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_successes < 0 or args.operation_budget < 0 or args.max_attempts < 1:
        parser.error("limits must be non-negative and --max-attempts must be positive")
    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    items = stable_domain_order(load_items(repo_root, review_path, "forged", "mouse"))
    if len(items) != args.expected:
        raise SystemExit(f"expected {args.expected} images, found {len(items)}")
    manifest_sha256 = input_digest(items)
    latest = read_latest(output_path)
    pending = [item for item in items if latest.get(item.id, {}).get("status") != "ok"]
    preview = pending[: args.max_successes]
    startup = {
        "selected": len(items),
        "already_valid": len(items) - len(pending),
        "max_successes_this_invocation": args.max_successes,
        "operation_budget_this_invocation": args.operation_budget,
        "preview_domain_counts": dict(Counter(item.domain for item in preview)),
        "input_manifest_sha256": manifest_sha256,
        "output": output_path.as_posix(),
        "dry_run": args.dry_run,
    }
    print(json.dumps(startup, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    api_user = os.environ.get("SIGHTENGINE_API_USER", "")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET", "")
    if not api_user or not api_secret:
        raise SystemExit("SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET must be set")
    ensure_run_manifest(
        output_path,
        items,
        manifest_sha256,
        args.endpoint,
        args.run_id,
        args.condition,
    )

    client = requests.Session()
    client.headers.update({"User-Agent": "claimforge-benchmark/sightengine-pilot-v1"})
    successes = failures = operations = 0
    last_started = 0.0
    for item in pending:
        if successes >= args.max_successes:
            break
        if operations + args.expected_operations_per_image > args.operation_budget:
            break
        wait_for = args.minimum_interval - (time.monotonic() - last_started)
        if wait_for > 0:
            time.sleep(wait_for)
        last_started = time.monotonic()
        row = classify(
            client,
            item,
            api_user,
            api_secret,
            args.endpoint,
            (args.connect_timeout, args.read_timeout),
            args.max_attempts,
            args.run_id,
            args.condition,
            manifest_sha256,
        )
        append_jsonl(output_path, row)
        if row["status"] == "ok":
            successes += 1
            operations += int(row.get("operations") or 0)
        else:
            failures += 1
        final_attempt = (row.get("attempts") or [{}])[-1]
        print(
            json.dumps(
                {
                    "successes_this_invocation": successes,
                    "failures_this_invocation": failures,
                    "operations_this_invocation": operations,
                    "last_id": row["id"],
                    "last_status": row["status"],
                    "last_http_status": row.get("http_status") or final_attempt.get("http_status"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        final_http = final_attempt.get("http_status")
        final_error = str(final_attempt.get("error_type") or "").lower() + " " + str(
            final_attempt.get("error_message") or ""
        ).lower()
        if final_http in AUTH_OR_QUOTA_HTTP or any(
            marker in final_error for marker in ("auth", "credential", "quota", "operation limit")
        ):
            print("authentication/quota failure detected; stopping batch", flush=True)
            break
        if row["status"] == "ok" and row.get("operations") != args.expected_operations_per_image:
            print(
                f"operation cost changed to {row.get('operations')}; stopping before exceeding budget assumptions",
                flush=True,
            )
            break

    summary = write_summary(output_path, items, manifest_sha256)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
