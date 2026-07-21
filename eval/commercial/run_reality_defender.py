#!/usr/bin/env python3
"""Run a Reality Defender coverage pilot on reviewed CLAIMFORGE images.

The API key is read only from ``REALITY_DEFENDER_API_KEY``. Real and forged
inputs are decoded and re-encoded as metadata-free JPEGs before upload. A
successful upload is appended to the JSONL before polling begins, so an
interrupted run resumes the existing ``requestId`` instead of creating and
potentially billing a duplicate scan.
"""

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

from eval.commercial.run_hive import canonicalize, load_selected_items
from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    input_digest,
    quantile,
    read_latest,
    redact_payload,
    redact_text,
    sha256_file,
    utc_now,
)


DEFAULT_API_BASE = "https://api.prd.realitydefender.xyz"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/reality_defender/"
    "pilot_good_mouse_pairs5_canonical_jpeg_q95_20260721.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
READY_STATUSES = {
    "AUTHENTIC",
    "FAKE",
    "SUSPICIOUS",
    "NOT_APPLICABLE",
    "UNABLE_TO_EVALUATE",
}
NON_APPLICABLE_STATUSES = {"NOT_APPLICABLE", "UNABLE_TO_EVALUATE"}


def endpoint(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def safe_json(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def response_error(response: requests.Response, api_key: str) -> str:
    body = safe_json(response)
    if body is not None:
        text = json.dumps(redact_payload(body), ensure_ascii=False, sort_keys=True)
    else:
        text = response.text
    return redact_text(text[:4000], api_key)


def backoff(attempt: int) -> None:
    time.sleep(min(2 ** (attempt - 1), 8))


def request_presigned_url(
    session: requests.Session,
    api_base: str,
    api_key: str,
    filename: str,
    timeout: tuple[float, float],
    max_attempts: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    url = endpoint(api_base, "/api/files/aws-presigned")
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            response = session.post(
                url,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"fileName": filename},
                timeout=timeout,
                allow_redirects=False,
            )
            latency_ms = round((time.monotonic() - started) * 1000)
            body = safe_json(response)
            attempt_row: dict[str, Any] = {
                "stage": "presign",
                "attempt": attempt,
                "http_status": response.status_code,
                "latency_ms": latency_ms,
            }
            if not 200 <= response.status_code < 300:
                attempt_row["error_message"] = response_error(response, api_key)
                attempts.append(attempt_row)
                if response.status_code in RETRYABLE_HTTP and attempt < max_attempts:
                    backoff(attempt)
                    continue
                return None, attempts
            signed_url = (
                body.get("response", {}).get("signedUrl")
                if isinstance(body, dict) and isinstance(body.get("response"), dict)
                else None
            )
            request_id = body.get("requestId") if isinstance(body, dict) else None
            media_id = body.get("mediaId") if isinstance(body, dict) else None
            if not all(isinstance(value, str) and value for value in (signed_url, request_id)):
                attempt_row["error_message"] = (
                    "invalid presign response: missing requestId or response.signedUrl"
                )
                attempts.append(attempt_row)
                return None, attempts
            attempts.append(attempt_row)
            return {
                "signed_url": signed_url,
                "request_id": request_id,
                "media_id": media_id if isinstance(media_id, str) else None,
            }, attempts
        except requests.RequestException as exc:
            attempts.append(
                {
                    "stage": "presign",
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), api_key),
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
            )
            if attempt < max_attempts:
                backoff(attempt)
                continue
            return None, attempts
    return None, attempts


def upload_to_presigned_url(
    session: requests.Session,
    signed_url: str,
    upload_path: Path,
    timeout: tuple[float, float],
    max_attempts: int,
) -> tuple[bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            with upload_path.open("rb") as handle:
                response = session.put(
                    signed_url,
                    headers={"Content-Type": "image/jpeg"},
                    data=handle,
                    timeout=timeout,
                    allow_redirects=False,
                )
            attempt_row: dict[str, Any] = {
                "stage": "upload",
                "attempt": attempt,
                "http_status": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
            if 200 <= response.status_code < 300:
                attempts.append(attempt_row)
                return True, attempts
            # Never persist the signed URL, response headers, or an AWS body that
            # could echo request-specific upload information.
            attempt_row["error_message"] = (
                f"signed upload returned HTTP {response.status_code}"
            )
            attempts.append(attempt_row)
            if response.status_code in RETRYABLE_HTTP and attempt < max_attempts:
                backoff(attempt)
                continue
            return False, attempts
        except requests.RequestException as exc:
            attempts.append(
                {
                    "stage": "upload",
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    # A requests exception can embed its URL, including the
                    # presigned query string. Keep only the exception type.
                    "error_message": "signed upload request failed",
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
            )
            if attempt < max_attempts:
                backoff(attempt)
                continue
            return False, attempts
    return False, attempts


def submission_row(
    item: ImageItem,
    upload: dict[str, Any],
    request_id: str,
    media_id: str | None,
    attempts: list[dict[str, Any]],
    run_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "reality_defender_result_v1",
        "provider": "reality_defender",
        "run_id": run_id,
        "id": item.id,
        "task_id": item.task_id,
        "domain": item.domain,
        "kind": item.kind,
        "label": item.label,
        "image_path": item.relative_path,
        "input_sha256": item.sha256,
        "input_bytes": item.file_bytes,
        "input_manifest_sha256": manifest_sha256,
        "status": "submitted",
        "request_id": request_id,
        "media_id": media_id,
        "upload_completed": True,
        "upload": upload,
        "attempts": attempts,
        "submitted_at": utc_now(),
    }


def parse_result(body: dict[str, Any]) -> dict[str, Any] | None:
    summary = body.get("resultsSummary")
    if not isinstance(summary, dict):
        return None
    raw_status = summary.get("status")
    if not isinstance(raw_status, str):
        return None
    provider_status = raw_status.strip().upper()
    if provider_status not in READY_STATUSES:
        return None
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    raw_score = metadata.get("finalScore")
    provider_score_raw = (
        float(raw_score)
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
        else None
    )
    provider_score = (
        provider_score_raw / 100.0
        if provider_score_raw is not None and 0.0 <= provider_score_raw <= 100.0
        else None
    )
    reasons = metadata.get("reasons") if isinstance(metadata.get("reasons"), list) else []
    models: list[dict[str, Any]] = []
    raw_models = body.get("models") if isinstance(body.get("models"), list) else []
    for model in raw_models:
        if not isinstance(model, dict):
            continue
        models.append(
            {
                "name": model.get("name"),
                "status": model.get("status"),
                "prediction_number": model.get("predictionNumber"),
            }
        )
    return {
        "provider_status": provider_status,
        "provider_score": provider_score,
        "provider_score_raw_0_100": provider_score_raw,
        "provider_fake": provider_status == "FAKE",
        "provider_flagged": provider_status in {"FAKE", "SUSPICIOUS"},
        "applicable": provider_status not in NON_APPLICABLE_STATUSES,
        "not_applicable_reasons": redact_payload(reasons),
        "release_version": body.get("releaseVersion"),
        "media_type": body.get("mediaType"),
        "models": redact_payload(models),
        "provider_response": {
            "requestId": body.get("requestId"),
            "releaseVersion": body.get("releaseVersion"),
            "mediaType": body.get("mediaType"),
            "resultsSummary": redact_payload(summary),
            "models": redact_payload(models),
        },
    }


def poll_result(
    session: requests.Session,
    api_base: str,
    api_key: str,
    request_id: str,
    timeout: tuple[float, float],
    poll_interval: float,
    max_poll_seconds: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    url = endpoint(api_base, f"/api/media/users/{request_id}")
    deadline = time.monotonic() + max_poll_seconds
    poll_number = 0
    while True:
        poll_number += 1
        started = time.monotonic()
        try:
            response = session.get(
                url,
                headers={"X-API-KEY": api_key},
                timeout=timeout,
                allow_redirects=False,
            )
            attempt_row: dict[str, Any] = {
                "stage": "poll",
                "attempt": poll_number,
                "http_status": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
            body = safe_json(response)
            if response.status_code == 200 and body is not None:
                parsed = parse_result(body)
                summary = body.get("resultsSummary")
                attempt_row["provider_status"] = (
                    summary.get("status") if isinstance(summary, dict) else None
                )
                attempts.append(attempt_row)
                if parsed is not None:
                    return parsed, attempts, "ok"
            elif response.status_code == 404:
                # The official SDK treats a short-lived 404 as not ready yet.
                attempts.append(attempt_row)
            else:
                attempt_row["error_message"] = response_error(response, api_key)
                attempts.append(attempt_row)
                if response.status_code in {401, 402, 403}:
                    return None, attempts, "error"
                if response.status_code not in RETRYABLE_HTTP:
                    return None, attempts, "error"
        except requests.RequestException as exc:
            attempts.append(
                {
                    "stage": "poll",
                    "attempt": poll_number,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), api_key),
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
            )
        if time.monotonic() >= deadline:
            return None, attempts, "poll_timeout"
        time.sleep(poll_interval)


def terminal_row(
    item: ImageItem,
    previous: dict[str, Any],
    parsed: dict[str, Any] | None,
    poll_attempts: list[dict[str, Any]],
    status: str,
    run_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema_version": "reality_defender_result_v1",
        "provider": "reality_defender",
        "run_id": run_id,
        "id": item.id,
        "task_id": item.task_id,
        "domain": item.domain,
        "kind": item.kind,
        "label": item.label,
        "image_path": item.relative_path,
        "input_sha256": item.sha256,
        "input_bytes": item.file_bytes,
        "input_manifest_sha256": manifest_sha256,
        "status": status,
        "request_id": previous.get("request_id"),
        "media_id": previous.get("media_id"),
        "upload_completed": True,
        "upload": previous.get("upload"),
        "attempts": poll_attempts,
        "completed_at": utc_now(),
    }
    if parsed is not None:
        row.update(parsed)
    return row


def submit_image(
    session: requests.Session,
    item: ImageItem,
    upload_path: Path,
    upload: dict[str, Any],
    api_base: str,
    api_key: str,
    timeout: tuple[float, float],
    max_attempts: int,
    run_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    presigned, presign_attempts = request_presigned_url(
        session,
        api_base,
        api_key,
        "image.jpg",
        timeout,
        max_attempts,
    )
    if presigned is None:
        return {
            "schema_version": "reality_defender_result_v1",
            "provider": "reality_defender",
            "run_id": run_id,
            "id": item.id,
            "task_id": item.task_id,
            "domain": item.domain,
            "kind": item.kind,
            "label": item.label,
            "image_path": item.relative_path,
            "input_sha256": item.sha256,
            "input_bytes": item.file_bytes,
            "input_manifest_sha256": manifest_sha256,
            "status": "error",
            "upload_completed": False,
            "upload": upload,
            "attempts": presign_attempts,
            "completed_at": utc_now(),
        }
    uploaded, upload_attempts = upload_to_presigned_url(
        session,
        str(presigned["signed_url"]),
        upload_path,
        timeout,
        max_attempts,
    )
    attempts = presign_attempts + upload_attempts
    if not uploaded:
        return {
            "schema_version": "reality_defender_result_v1",
            "provider": "reality_defender",
            "run_id": run_id,
            "id": item.id,
            "task_id": item.task_id,
            "domain": item.domain,
            "kind": item.kind,
            "label": item.label,
            "image_path": item.relative_path,
            "input_sha256": item.sha256,
            "input_bytes": item.file_bytes,
            "input_manifest_sha256": manifest_sha256,
            "status": "error",
            "request_id": presigned["request_id"],
            "media_id": presigned.get("media_id"),
            "upload_completed": False,
            "upload": upload,
            "attempts": attempts,
            "completed_at": utc_now(),
        }
    return submission_row(
        item,
        upload,
        str(presigned["request_id"]),
        presigned.get("media_id"),
        attempts,
        run_id,
        manifest_sha256,
    )


def score_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p95": quantile(values, 0.95) if values else None,
    }


def write_summary(
    output_path: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    include: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    rows = [latest[item.id] for item in items if item.id in latest]
    completed = [row for row in rows if row.get("status") == "ok"]
    applicable = [row for row in completed if row.get("applicable") is True]
    status_by_kind: dict[str, dict[str, int]] = {}
    score_by_kind: dict[str, dict[str, Any]] = {}
    coverage_by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("real", "forged"):
        selected_count = sum(item.kind == kind for item in items)
        kind_rows = [row for row in completed if row.get("kind") == kind]
        kind_applicable = [row for row in kind_rows if row.get("applicable") is True]
        status_by_kind[kind] = dict(
            Counter(str(row.get("provider_status")) for row in kind_rows)
        )
        score_by_kind[kind] = score_stats(
            [
                float(row["provider_score"])
                for row in kind_applicable
                if isinstance(row.get("provider_score"), (int, float))
            ]
        )
        coverage_by_kind[kind] = {
            "expected": selected_count,
            "terminal": len(kind_rows),
            "applicable": len(kind_applicable),
            "applicable_rate": (
                len(kind_applicable) / selected_count if selected_count else None
            ),
        }

    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in applicable:
        by_task.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    paired_deltas: list[float] = []
    for pair in by_task.values():
        real = pair.get("real", {}).get("provider_score")
        forged = pair.get("forged", {}).get("provider_score")
        if isinstance(real, (int, float)) and isinstance(forged, (int, float)):
            paired_deltas.append(float(forged) - float(real))

    reasons = Counter()
    for row in completed:
        for reason in row.get("not_applicable_reasons") or []:
            if isinstance(reason, dict):
                key = f"{reason.get('code')}: {reason.get('message')}"
                reasons[key] += 1
    summary = {
        "schema_version": "reality_defender_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "include": include,
        "expected_tasks": len({item.task_id for item in items}),
        "expected_images": len(items),
        "completed_images": len(rows),
        "valid_terminal_images": len(completed),
        "applicable_images": len(applicable),
        "applicable_coverage": len(applicable) / len(items) if items else None,
        "pending_images": len(items) - len(completed),
        "status_by_kind": status_by_kind,
        "coverage_by_kind": coverage_by_kind,
        "score_by_kind": score_by_kind,
        "paired_score_delta": score_stats(paired_deltas),
        "not_applicable_reasons": dict(reasons),
        "row_status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "http_error_counts": dict(
            Counter(
                str((row.get("attempts") or [{}])[-1].get("http_status"))
                for row in rows
                if row.get("status") in {"error", "poll_timeout"}
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
    api_base: str,
    quality: int,
    run_id: str,
    include: str,
    poll_interval: float,
    max_poll_seconds: float,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "reality_defender_run_manifest_v1",
        "run_id": run_id,
        "api_base": api_base.rstrip("/"),
        "presign_endpoint": endpoint(api_base, "/api/files/aws-presigned"),
        "result_endpoint_template": endpoint(
            api_base, "/api/media/users/{request_id}"
        ),
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
        "poll_interval_seconds": poll_interval,
        "max_poll_seconds": max_poll_seconds,
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


def blocking_http_status(row: dict[str, Any]) -> int | None:
    for attempt in reversed(row.get("attempts") or []):
        value = attempt.get("http_status")
        if isinstance(value, int):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--order-manifest", type=Path, default=DEFAULT_ORDER_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--include", choices=("forged", "both"), default="both")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-poll-seconds", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--run-id", default="reality_defender_mouse_pairs5_20260721")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tasks < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--tasks must be positive and JPEG quality must be in [1, 100]")
    if args.max_attempts < 1 or min(
        args.poll_interval,
        args.max_poll_seconds,
        args.minimum_interval,
    ) < 0:
        parser.error("attempts must be positive and intervals must be non-negative")
    if args.max_poll_seconds == 0:
        parser.error("--max-poll-seconds must be positive")

    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    order_path = (
        args.order_manifest
        if args.order_manifest.is_absolute()
        else repo_root / args.order_manifest
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    items = load_selected_items(
        repo_root,
        review_path,
        order_path,
        args.tasks,
        args.include,
    )
    manifest_sha256 = input_digest(items)
    latest = read_latest(output_path)
    pending = [item for item in items if latest.get(item.id, {}).get("status") != "ok"]
    resumable = [
        item
        for item in pending
        if latest.get(item.id, {}).get("upload_completed") is True
        and isinstance(latest.get(item.id, {}).get("request_id"), str)
    ]
    print(
        json.dumps(
            {
                "selected_tasks": args.tasks,
                "include": args.include,
                "selected_images": len(items),
                "already_terminal": len(items) - len(pending),
                "pending": len(pending),
                "resumable_request_ids": len(resumable),
                "new_uploads": len(pending) - len(resumable),
                "output": output_path.as_posix(),
                "dry_run": args.dry_run,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return

    api_key = os.environ.get("REALITY_DEFENDER_API_KEY", "")
    if not api_key:
        raise SystemExit("REALITY_DEFENDER_API_KEY must be set")
    ensure_run_manifest(
        output_path,
        items,
        manifest_sha256,
        args.api_base,
        args.jpeg_quality,
        args.run_id,
        args.include,
        args.poll_interval,
        args.max_poll_seconds,
    )

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "claimforge-benchmark/reality-defender-pilot-v1"}
    )
    with tempfile.TemporaryDirectory(prefix="claimforge-reality-defender-") as temporary:
        temporary_dir = Path(temporary)
        for index, item in enumerate(pending):
            prior = latest.get(item.id, {})
            if prior.get("upload_completed") is True and isinstance(
                prior.get("request_id"), str
            ):
                submitted = prior
            else:
                upload_path = temporary_dir / f"upload-{index:04d}.jpg"
                upload = canonicalize(item.path, upload_path, args.jpeg_quality)
                submitted = submit_image(
                    session,
                    item,
                    upload_path,
                    upload,
                    args.api_base,
                    api_key,
                    (args.connect_timeout, args.read_timeout),
                    args.max_attempts,
                    args.run_id,
                    manifest_sha256,
                )
                append_jsonl(output_path, submitted)
                if submitted.get("status") != "submitted":
                    print(
                        json.dumps(
                            {
                                "id": item.id,
                                "status": submitted.get("status"),
                                "stage": "upload",
                                "http_status": blocking_http_status(submitted),
                            }
                        ),
                        flush=True,
                    )
                    if blocking_http_status(submitted) in {401, 402, 403, 429}:
                        print(
                            "authentication, credit, or quota failure; stopping batch",
                            flush=True,
                        )
                        break
                    continue

            parsed, poll_attempts, poll_status = poll_result(
                session,
                args.api_base,
                api_key,
                str(submitted["request_id"]),
                (args.connect_timeout, args.read_timeout),
                args.poll_interval,
                args.max_poll_seconds,
            )
            row = terminal_row(
                item,
                submitted,
                parsed,
                poll_attempts,
                poll_status,
                args.run_id,
                manifest_sha256,
            )
            append_jsonl(output_path, row)
            latest[item.id] = row
            print(
                json.dumps(
                    {
                        "id": item.id,
                        "status": row.get("status"),
                        "provider_status": row.get("provider_status"),
                        "provider_score": row.get("provider_score"),
                        "applicable": row.get("applicable"),
                    }
                ),
                flush=True,
            )
            if poll_status == "error" and blocking_http_status(row) in {
                401,
                402,
                403,
                429,
            }:
                print(
                    "authentication, credit, or quota failure; stopping batch",
                    flush=True,
                )
                break
            if index + 1 < len(pending):
                time.sleep(args.minimum_interval)

    summary = write_summary(output_path, items, manifest_sha256, args.include)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
