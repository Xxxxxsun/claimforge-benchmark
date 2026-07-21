#!/usr/bin/env python3
"""Run Hive AI-generated-content detection on reviewed CLAIMFORGE images.

The API secret is read only from ``HIVE_API_KEY``. Source and forged images
are both decoded and re-encoded as metadata-free JPEGs before upload so the
comparison is based on pixels rather than container or metadata differences.
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
from PIL import Image, ImageOps

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


DEFAULT_ENDPOINT = (
    "https://api.thehive.ai/api/v3/hive/"
    "ai-generated-and-deepfake-content-detection"
)
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/hive/"
    "pilot_good_mouse_pairs5_canonical_jpeg_q95_20260720.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 500, 502, 503, 504}
DEFAULT_THRESHOLD = 0.9


def load_selected_items(
    repo_root: Path,
    review_path: Path,
    order_manifest_path: Path,
    task_count: int,
    include: str,
) -> list[ImageItem]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    records = {
        str(record["task_id"]): record
        for record in review.get("records", [])
        if record.get("status") == "good" and record.get("candidates") == "mouse"
    }
    ordering = json.loads(order_manifest_path.read_text(encoding="utf-8"))
    task_ids: list[str] = []
    for entry in ordering.get("ordered_inputs", []):
        task_id = str(entry.get("task_id", ""))
        if task_id in records and task_id not in task_ids:
            task_ids.append(task_id)
        if len(task_ids) == task_count:
            break
    if len(task_ids) != task_count:
        raise ValueError(f"requested {task_count} tasks, found {len(task_ids)}")

    items: list[ImageItem] = []
    for task_id in task_ids:
        record = records[task_id]
        raw_size = record.get("image_size")
        image_size = (
            (int(raw_size[0]), int(raw_size[1]))
            if isinstance(raw_size, list) and len(raw_size) == 2
            else None
        )
        variants = [("forged", "edited", "spliced_image")]
        if include == "both":
            variants.insert(0, ("real", "not_edited", "source_image"))
        for kind, label, field in variants:
            relative_path = str(record[field])
            path = (repo_root / relative_path).resolve()
            try:
                path.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"image path escapes repo: {relative_path}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"missing {kind} image: {relative_path}")
            items.append(
                ImageItem(
                    id=f"{task_id}__{kind}",
                    task_id=task_id,
                    domain=task_id.split("_", 1)[0],
                    kind=kind,
                    label=label,
                    path=path,
                    relative_path=relative_path,
                    image_size=image_size,
                    sha256=sha256_file(path),
                    file_bytes=path.stat().st_size,
                )
            )
    return items


def canonicalize(source: Path, destination: Path, quality: int) -> dict[str, Any]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
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


def parse_success(body: dict[str, Any]) -> dict[str, Any] | None:
    outputs = body.get("output")
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        return None
    classes = outputs[0].get("classes")
    if not isinstance(classes, list):
        return None
    scores: dict[str, float] = {}
    for row in classes:
        if not isinstance(row, dict) or not isinstance(row.get("class"), str):
            continue
        value = as_probability(row.get("value"))
        if value is not None:
            scores[row["class"]] = value
    ai_probability = scores.get("ai_generated")
    if ai_probability is None:
        return None
    return {
        "ai_probability": ai_probability,
        "not_ai_probability": scores.get("not_ai_generated"),
        "class_scores": scores,
        "provider_model": body.get("model"),
        "provider_version": body.get("version"),
        "provider_task_id": body.get("task_id"),
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
    threshold: float,
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
                    files={"media": ("image.jpg", handle, "image/jpeg")},
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
                    "schema_version": "hive_result_v1",
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
                    "threshold": threshold,
                    "detected": parsed["ai_probability"] >= threshold,
                    **parsed,
                    "completed_at": utc_now(),
                }
            message = response.text[:1000]
            if isinstance(body, dict):
                message = json.dumps(redact_payload(body), ensure_ascii=False)[:1000]
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
        except requests.RequestException as exc:
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
        "schema_version": "hive_result_v1",
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


def score_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores = sorted(
        float(row["ai_probability"])
        for row in rows
        if row.get("status") == "ok"
        and as_probability(row.get("ai_probability")) is not None
    )
    detected = sum(score >= threshold for score in scores)
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
    threshold: float,
    include: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    rows = [latest[item.id] for item in items if item.id in latest]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in ok_rows:
        by_task.setdefault(row["task_id"], {})[row["kind"]] = row
    deltas = [
        pair["forged"]["ai_probability"] - pair["real"]["ai_probability"]
        for pair in by_task.values()
        if "real" in pair and "forged" in pair
    ]
    summary = {
        "schema_version": "hive_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "threshold": threshold,
        "include": include,
        "expected_tasks": len({item.task_id for item in items}),
        "expected_pairs": (
            len({item.task_id for item in items}) if include == "both" else 0
        ),
        "expected_images": len(items),
        "completed_images": len(rows),
        "valid_images": len(ok_rows),
        "error_images": len(rows) - len(ok_rows),
        "score_by_kind": {
            kind: score_summary(
                [row for row in ok_rows if row.get("kind") == kind], threshold
            )
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
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return summary


def ensure_run_manifest(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    endpoint: str,
    quality: int,
    threshold: float,
    run_id: str,
    include: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "hive_run_manifest_v1",
        "run_id": run_id,
        "endpoint": endpoint,
        "model": "ai-generated-and-deepfake-content-detection",
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
        "threshold": threshold,
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
        "pillow_version": Image.__version__,
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
    parser.add_argument("--tasks", "--pairs", dest="task_count", type=int, default=5)
    parser.add_argument("--include", choices=("forged", "both"), default="both")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument(
        "--max-pending",
        type=int,
        help="process at most this many currently pending images",
    )
    parser.add_argument("--run-id", default="hive_pilot_good_mouse_pairs5_20260720")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--tasks must be positive and JPEG quality must be in [1, 100]")
    if not 0 <= args.threshold <= 1 or args.max_attempts < 1:
        parser.error("threshold must be in [0, 1] and max attempts must be positive")
    if args.max_pending is not None and args.max_pending < 1:
        parser.error("--max-pending must be positive")

    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    order_path = (
        args.order_manifest
        if args.order_manifest.is_absolute()
        else repo_root / args.order_manifest
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    items = load_selected_items(
        repo_root, review_path, order_path, args.task_count, args.include
    )
    manifest_sha256 = input_digest(items)
    latest = read_latest(output_path)
    pending = [item for item in items if latest.get(item.id, {}).get("status") != "ok"]
    pending_total = len(pending)
    if args.max_pending is not None:
        pending = pending[: args.max_pending]
    print(
        json.dumps(
            {
                "selected_tasks": args.task_count,
                "include": args.include,
                "selected_images": len(items),
                "already_valid": len(items) - pending_total,
                "pending": pending_total,
                "scheduled": len(pending),
                "output": output_path.as_posix(),
                "dry_run": args.dry_run,
            }
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
        items,
        manifest_sha256,
        args.endpoint,
        args.jpeg_quality,
        args.threshold,
        args.run_id,
        args.include,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "claimforge-benchmark/hive-pilot-v1"})
    with tempfile.TemporaryDirectory(prefix="claimforge-hive-") as temporary:
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
                args.threshold,
                args.run_id,
                manifest_sha256,
            )
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "ai_probability": row.get("ai_probability"),
                        "detected": row.get("detected"),
                    }
                ),
                flush=True,
            )
            if index + 1 < len(pending):
                time.sleep(args.minimum_interval)
            if row["status"] == "error":
                final = (row.get("attempts") or [{}])[-1]
                if final.get("http_status") in {401, 403, 429}:
                    print("authentication or quota failure; stopping batch", flush=True)
                    break

    summary = write_summary(
        output_path, items, manifest_sha256, args.threshold, args.include
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
