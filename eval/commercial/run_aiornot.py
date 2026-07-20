#!/usr/bin/env python3
"""Run AI or Not on reviewed CLAIMFORGE mouse images.

The API key is read only from ``AIORNOT_API_KEY``. Both real and forged
images are converted to metadata-free JPEGs before upload, and the request
uses ``only=ai_generated`` so unrelated paid reports are not run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from eval.commercial.run_hive import canonicalize, load_selected_items
from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    as_probability,
    input_digest,
    quantile,
    read_latest,
    redact_payload,
    redact_text,
    sha256_file,
    utc_now,
)


DEFAULT_ENDPOINT = "https://api.aiornot.com/v2/image/sync"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/aiornot/"
    "pilot_good_mouse_pairs5_canonical_jpeg_q95_20260720.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


def parse_success(body: dict[str, Any]) -> dict[str, Any] | None:
    report = body.get("report")
    if not isinstance(report, dict):
        return None
    ai_generated = report.get("ai_generated")
    if not isinstance(ai_generated, dict):
        return None
    ai = ai_generated.get("ai")
    human = ai_generated.get("human")
    if not isinstance(ai, dict) or not isinstance(human, dict):
        return None
    ai_detected = ai.get("is_detected")
    human_detected = human.get("is_detected")
    ai_confidence = as_probability(ai.get("confidence"))
    human_confidence = as_probability(human.get("confidence"))
    if not isinstance(ai_detected, bool) or ai_confidence is None:
        return None

    raw_generators = ai_generated.get("generator")
    generator_scores: dict[str, float] = {}
    if isinstance(raw_generators, dict):
        for name, value in raw_generators.items():
            score = as_probability(value)
            if score is not None:
                generator_scores[str(name)] = score
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    return {
        "provider_id": body.get("id"),
        "provider_created_at": body.get("created_at"),
        "provider_external_id": body.get("external_id"),
        "ai_detected": ai_detected,
        "ai_confidence": ai_confidence,
        "human_detected": human_detected if isinstance(human_detected, bool) else None,
        "human_confidence": human_confidence,
        "generator_scores": generator_scores,
        "provider_meta": redact_payload(meta),
    }


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(120.0, float(retry_after)))
            except ValueError:
                pass
    return min(30.0, 2 ** (attempt - 1) + random.random())


def classify(
    session: requests.Session,
    item: ImageItem,
    upload_path: Path,
    upload: dict[str, Any],
    endpoint: str,
    api_key: str,
    timeout: tuple[float, float],
    max_attempts: int,
    run_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        response: requests.Response | None = None
        started = time.monotonic()
        try:
            with upload_path.open("rb") as handle:
                response = session.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"only": "ai_generated", "external_id": item.id},
                    files={"image": ("image.jpg", handle, "image/jpeg")},
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
                    "schema_version": "aiornot_result_v1",
                    "run_id": run_id,
                    "input_manifest_sha256": manifest_sha256,
                    "id": item.id,
                    "task_id": item.task_id,
                    "domain": item.domain,
                    "kind": item.kind,
                    "label": item.label,
                    "image_path": item.relative_path,
                    "image_sha256": item.sha256,
                    "file_bytes": item.file_bytes,
                    **upload,
                    "status": "ok",
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "attempt_count": attempt,
                    **parsed,
                    "completed_at": utc_now(),
                }
            message = response.text[:1500]
            if isinstance(body, (dict, list)):
                message = json.dumps(redact_payload(body), ensure_ascii=False)[:1500]
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "rate_limit": {
                        key: response.headers[key]
                        for key in (
                            "Retry-After",
                            "X-RateLimit-Limit",
                            "X-RateLimit-Remaining",
                            "X-RateLimit-Reset",
                        )
                        if key in response.headers
                    },
                    "error_message": redact_text(message, api_key),
                }
            )
            if response.status_code not in RETRYABLE_HTTP:
                break
        except (requests.RequestException, OSError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code if response is not None else None,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), api_key),
                }
            )
        if attempt < max_attempts:
            time.sleep(retry_delay(attempt, response))

    return {
        "schema_version": "aiornot_result_v1",
        "run_id": run_id,
        "input_manifest_sha256": manifest_sha256,
        "id": item.id,
        "task_id": item.task_id,
        "domain": item.domain,
        "kind": item.kind,
        "label": item.label,
        "image_path": item.relative_path,
        "image_sha256": item.sha256,
        "file_bytes": item.file_bytes,
        **upload,
        "status": "error",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "completed_at": utc_now(),
    }


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(
        float(row["ai_confidence"])
        for row in rows
        if row.get("status") == "ok"
        and as_probability(row.get("ai_confidence")) is not None
    )
    detected = sum(
        bool(row.get("ai_detected")) for row in rows if row.get("status") == "ok"
    )
    return {
        "count": len(scores),
        "min": min(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "mean": statistics.fmean(scores) if scores else None,
        "p95": quantile(scores, 0.95),
        "max": max(scores) if scores else None,
        "detected": detected,
        "detection_rate": detected / len(scores) if scores else None,
    }


def write_summary(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    include: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    rows = [latest[item.id] for item in items if item.id in latest]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in ok_rows:
        by_task.setdefault(row["task_id"], {})[row["kind"]] = row
    deltas = [
        pair["forged"]["ai_confidence"] - pair["real"]["ai_confidence"]
        for pair in by_task.values()
        if "real" in pair and "forged" in pair
    ]
    summary = {
        "schema_version": "aiornot_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "include": include,
        "expected_tasks": len({item.task_id for item in items}),
        "expected_images": len(items),
        "completed_images": len(rows),
        "valid_images": len(ok_rows),
        "error_images": len(rows) - len(ok_rows),
        "score_by_kind": {
            kind: score_summary([row for row in ok_rows if row.get("kind") == kind])
            for kind in ("real", "forged")
        },
        "paired_delta": {
            "count": len(deltas),
            "mean": statistics.fmean(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
        },
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


def ensure_run_manifest(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    endpoint: str,
    quality: int,
    run_id: str,
    include: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "aiornot_run_manifest_v1",
        "run_id": run_id,
        "endpoint": endpoint,
        "reports": ["ai_generated"],
        "candidate": "mouse",
        "include": "paired_real_and_forged" if include == "both" else "forged",
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "metadata": "stripped",
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
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "requests_version": requests.__version__,
        "ordered_inputs": [
            {
                "rank": rank,
                "id": item.id,
                "task_id": item.task_id,
                "domain": item.domain,
                "kind": item.kind,
                "image_path": item.relative_path,
                "image_sha256": item.sha256,
                "file_bytes": item.file_bytes,
            }
            for rank, item in enumerate(items)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--order-manifest", type=Path, default=DEFAULT_ORDER_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--include", choices=("forged", "both"), default="both")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=1.1)
    parser.add_argument("--run-id", default="aiornot_mouse_pairs5_20260720")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tasks < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--tasks must be positive and JPEG quality must be in [1, 100]")
    if args.max_attempts < 1 or args.minimum_interval < 0:
        parser.error("max attempts must be positive and minimum interval non-negative")

    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    order_path = (
        args.order_manifest
        if args.order_manifest.is_absolute()
        else repo_root / args.order_manifest
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    items = load_selected_items(
        repo_root, review_path, order_path, args.tasks, args.include
    )
    manifest_sha256 = input_digest(items)
    latest = read_latest(output_path)
    pending = [item for item in items if latest.get(item.id, {}).get("status") != "ok"]
    print(
        json.dumps(
            {
                "selected_tasks": args.tasks,
                "include": args.include,
                "selected_images": len(items),
                "already_valid": len(items) - len(pending),
                "pending": len(pending),
                "endpoint": args.endpoint,
                "reports": ["ai_generated"],
                "output": output_path.as_posix(),
                "dry_run": args.dry_run,
            }
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
        items,
        manifest_sha256,
        args.endpoint,
        args.jpeg_quality,
        args.run_id,
        args.include,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "claimforge-benchmark/aiornot-v1"})
    with tempfile.TemporaryDirectory(prefix="claimforge-aiornot-") as temporary:
        temporary_dir = Path(temporary)
        for index, item in enumerate(pending):
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload = canonicalize(item.path, upload_path, args.jpeg_quality)
            row = classify(
                session,
                item,
                upload_path,
                upload,
                args.endpoint,
                api_key,
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                args.run_id,
                manifest_sha256,
            )
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "ai_detected": row.get("ai_detected"),
                        "ai_confidence": row.get("ai_confidence"),
                        "error": (row.get("attempts") or [{}])[-1].get("error_message")
                        if row["status"] == "error"
                        else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["status"] == "error":
                final = (row.get("attempts") or [{}])[-1]
                if final.get("http_status") in {401, 402, 403, 429}:
                    print("authentication, billing, or quota failure; stopping batch", flush=True)
                    break
            if index + 1 < len(pending):
                time.sleep(args.minimum_interval)

    summary = write_summary(output_path, items, manifest_sha256, args.include)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
