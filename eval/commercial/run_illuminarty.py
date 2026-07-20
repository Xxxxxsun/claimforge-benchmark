#!/usr/bin/env python3
"""Run Illuminarty classification on reviewed CLAIMFORGE image pairs.

The API key is read only from ``ILLUMINARTY_API_KEY``. It is never accepted as
a command-line argument or written to results. Results are appended one image
at a time so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_ENDPOINT = "https://api.illuminarty.ai/v1/image/classify"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_OUTPUT = Path(
    "results/commercial/illuminarty/"
    "pilot_good275_mouse_forged_original_png_20260719.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ImageItem:
    id: str
    task_id: str
    domain: str
    kind: str
    label: str
    path: Path
    relative_path: str
    image_size: tuple[int, int] | None
    sha256: str
    file_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(
    repo_root: Path,
    review_path: Path,
    include: str,
    candidate: str,
) -> list[ImageItem]:
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"review export has no records list: {review_path}")

    items: list[ImageItem] = []
    for record in records:
        if record.get("status") != "good" or record.get("candidates") != candidate:
            continue
        task_id = str(record["task_id"])
        domain = task_id.split("_", 1)[0]
        variants: list[tuple[str, str, str]] = []
        if include in {"forged", "both"}:
            variants.append(("forged", "edited", str(record["spliced_image"])))
        if include in {"real", "both"}:
            variants.append(("real", "not_edited", str(record["source_image"])))
        raw_size = record.get("image_size")
        image_size = (
            (int(raw_size[0]), int(raw_size[1]))
            if isinstance(raw_size, list) and len(raw_size) == 2
            else None
        )
        for kind, label, relative_path in variants:
            path = (repo_root / relative_path).resolve()
            try:
                path.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ValueError(f"image path escapes repo root for {task_id}: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"missing {kind} image for {task_id}: {path}")
            items.append(
                ImageItem(
                    id=f"{task_id}__{kind}",
                    task_id=task_id,
                    domain=domain,
                    kind=kind,
                    label=label,
                    path=path,
                    relative_path=relative_path,
                    image_size=image_size,
                    sha256=sha256_file(path),
                    file_bytes=path.stat().st_size,
                )
            )

    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("input selection contains duplicate image IDs")
    return sorted(items, key=lambda item: item.id)


def input_digest(items: Iterable[ImageItem]) -> str:
    rows = [
        {
            "id": item.id,
            "label": item.label,
            "relative_path": item.relative_path,
            "sha256": item.sha256,
        }
        for item in items
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def ensure_run_manifest(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    endpoint: str,
    run_id: str,
    condition: str,
    candidate: str,
    include: str,
) -> Path:
    manifest_path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "illuminarty_run_manifest_v1",
        "run_id": run_id,
        "condition": condition,
        "endpoint": endpoint,
        "candidate": candidate,
        "include": include,
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"run manifest does not match requested run: {mismatches}")
        return manifest_path

    payload = {
        **expected,
        "created_at": utc_now(),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "requests_version": requests.__version__,
        "inputs": [
            {
                "id": item.id,
                "task_id": item.task_id,
                "domain": item.domain,
                "kind": item.kind,
                "label": item.label,
                "image_path": item.relative_path,
                "image_sha256": item.sha256,
                "file_bytes": item.file_bytes,
            }
            for item in items
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


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            latest[row["id"]] = row
    return latest


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def redact_text(text: str, secret: str) -> str:
    if secret:
        text = text.replace(secret, "<redacted-api-key>")
    return re.sub(
        r'(?i)("?(?:api[_-]?key|authorization|token|secret)"?\s*[:=]\s*)[^,\s}\]]+',
        r"\1<redacted>",
        text,
    )


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(marker in normalized for marker in ("api_key", "apikey", "authorization", "token", "secret")):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = redact_payload(child)
        return redacted
    if isinstance(value, list):
        return [redact_payload(child) for child in value]
    return value


def nested_get(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 0.0 <= number <= 1.0 and math.isfinite(number):
        return number
    return None


def parse_response(body: dict[str, Any]) -> dict[str, Any]:
    # Lock to the schema exposed by Illuminarty's public classify endpoint.
    # Do not reinterpret unrelated numeric fields as the AI score.
    ai_paths = (("data", "probability"),)
    human_paths = (
        ("data", "details", "breakdown", "human"),
        ("details", "breakdown", "human"),
        ("data", "human_probability"),
        ("human_probability",),
    )
    localization_paths = (
        ("data", "details", "localization"),
        ("details", "localization"),
    )
    ai_probability = next(
        (score for path in ai_paths if (score := as_probability(nested_get(body, path))) is not None),
        None,
    )
    human_probability = next(
        (score for path in human_paths if (score := as_probability(nested_get(body, path))) is not None),
        None,
    )
    localization = next(
        (value for path in localization_paths if (value := nested_get(body, path)) is not None),
        None,
    )
    api_status = body.get("status") if isinstance(body.get("status"), str) else None
    return {
        "api_status": api_status,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "localization": localization,
    }


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, min(120.0, float(raw)))
            except ValueError:
                pass
    return min(60.0, (2 ** (attempt - 1)) + random.random())


_thread_state = threading.local()


def session() -> requests.Session:
    current = getattr(_thread_state, "session", None)
    if current is None:
        current = requests.Session()
        current.headers.update({"User-Agent": "claimforge-benchmark/illuminarty-pilot-v1"})
        _thread_state.session = current
    return current


def classify(
    item: ImageItem,
    api_key: str,
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
            mime = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
            with item.path.open("rb") as image_handle:
                response = session().post(
                    endpoint,
                    headers={"X-API-Key": api_key},
                    # Do not disclose task IDs, domains, or labels via the
                    # multipart filename. The detector should see image bytes only.
                    files={"file": (f"image{item.path.suffix.lower()}", image_handle, mime)},
                    timeout=timeout,
                    allow_redirects=False,
                )
            latency_ms = round((time.monotonic() - started) * 1000)
            try:
                body: Any = response.json()
            except ValueError:
                body = None
            if response.status_code == 200 and isinstance(body, dict):
                parsed = parse_response(body)
                if parsed["ai_probability"] is not None and parsed["api_status"] in {None, "success"}:
                    return {
                        "schema_version": "illuminarty_result_v1",
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
            response_text = redact_text(response.text[:1000], api_key)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "error": (
                        "valid JSON response did not contain an AI probability"
                        if response.status_code == 200 and isinstance(body, dict)
                        else response_text
                    ),
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
                    "error": redact_text(f"{type(exc).__name__}: {exc}", api_key),
                }
            )
        if attempt < max_attempts:
            time.sleep(retry_delay(attempt, response))

    return {
        "schema_version": "illuminarty_result_v1",
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


def quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(
        float(row["ai_probability"])
        for row in rows
        if row.get("status") == "ok" and as_probability(row.get("ai_probability")) is not None
    )
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
        "detected_at_0_5": sum(score >= 0.5 for score in scores),
        "detection_rate_at_0_5": (sum(score >= 0.5 for score in scores) / len(scores)) if scores else None,
        "escaped_at_0_5": sum(score < 0.5 for score in scores),
        "escape_rate_at_0_5": (sum(score < 0.5 for score in scores) / len(scores)) if scores else None,
        "below_0_01": sum(score < 0.01 for score in scores),
        "below_0_05": sum(score < 0.05 for score in scores),
        "below_0_10": sum(score < 0.10 for score in scores),
    }


def write_summary(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    selected_rows = [latest[item.id] for item in items if item.id in latest]
    ok_rows = [row for row in selected_rows if row.get("status") == "ok"]
    error_rows = [row for row in selected_rows if row.get("status") == "error"]
    summary = {
        "schema_version": "illuminarty_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "expected_images": len(items),
        "completed_images": len(selected_rows),
        "valid_images": len(ok_rows),
        "coverage": len(ok_rows) / len(items) if items else None,
        "error_images": len(error_rows),
        "score": score_summary(ok_rows),
        "by_domain": {
            domain: score_summary([row for row in ok_rows if row.get("domain") == domain])
            for domain in sorted({item.domain for item in items})
        },
        "localization_present": sum(row.get("localization") is not None for row in ok_rows),
        "http_error_counts": {
            str(status): sum(
                1
                for row in error_rows
                if (row.get("attempts") or [{}])[-1].get("http_status") == status
            )
            for status in sorted(
                {
                    (row.get("attempts") or [{}])[-1].get("http_status")
                    for row in error_rows
                },
                key=lambda value: (value is None, str(value)),
            )
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--candidate", default="mouse")
    parser.add_argument("--include", choices=("forged", "real", "both"), default="forged")
    parser.add_argument("--expected", type=int, default=275)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--run-id", default="illuminarty_pilot_good275_mouse_forged_20260719")
    parser.add_argument("--condition", default="pilot_good275_mouse_forged_original_png")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_attempts < 1:
        parser.error("--concurrency and --max-attempts must be positive")
    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    items = load_items(repo_root, review_path, args.include, args.candidate)
    if args.expected >= 0 and len(items) != args.expected:
        raise SystemExit(f"expected {args.expected} selected images, found {len(items)}")
    manifest_sha256 = input_digest(items)
    latest = read_latest(output_path)
    pending = [
        item
        for item in items
        if item.id not in latest or latest[item.id].get("status") != "ok"
    ]
    already_recorded = len(items) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]

    startup = {
        "selected": len(items),
        "already_recorded": already_recorded,
        "pending_this_invocation": len(pending),
        "files_over_3_mib": sum(item.file_bytes > 3 * 1024 * 1024 for item in items),
        "input_manifest_sha256": manifest_sha256,
        "output": output_path.as_posix(),
        "dry_run": args.dry_run,
    }
    print(json.dumps(startup, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    api_key = os.environ.get("ILLUMINARTY_API_KEY", "")
    if not api_key:
        raise SystemExit("ILLUMINARTY_API_KEY is not set")
    ensure_run_manifest(
        output_path,
        items,
        manifest_sha256,
        args.endpoint,
        args.run_id,
        args.condition,
        args.candidate,
        args.include,
    )

    invocation_total = len(pending)
    completed = succeeded = failed = 0
    if pending:
        # One synchronous preflight prevents a bad/revoked key from fanning
        # out into hundreds of guaranteed 401/403 requests.
        first, *pending = pending
        row = classify(
            first,
            api_key,
            args.endpoint,
            (args.connect_timeout, args.read_timeout),
            args.max_attempts,
            args.run_id,
            args.condition,
            manifest_sha256,
        )
        append_jsonl(output_path, row)
        completed = 1
        succeeded = int(row["status"] == "ok")
        failed = int(row["status"] != "ok")
        final_http_status = (row.get("attempts") or [{}])[-1].get("http_status")
        print(
            json.dumps(
                {
                    "completed_this_invocation": completed,
                    "total_this_invocation": invocation_total,
                    "succeeded": succeeded,
                    "failed": failed,
                    "last_id": row["id"],
                    "last_status": row["status"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if final_http_status in {401, 403}:
            write_summary(output_path, items, manifest_sha256)
            raise SystemExit(f"authentication failed with HTTP {final_http_status}; stopping before batch fan-out")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                classify,
                item,
                api_key,
                args.endpoint,
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                args.run_id,
                args.condition,
                manifest_sha256,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(output_path, row)
            completed += 1
            if row["status"] == "ok":
                succeeded += 1
            else:
                failed += 1
            print(
                json.dumps(
                    {
                        "completed_this_invocation": completed,
                        "total_this_invocation": invocation_total,
                        "succeeded": succeeded,
                        "failed": failed,
                        "last_id": row["id"],
                        "last_status": row["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = write_summary(output_path, items, manifest_sha256)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
