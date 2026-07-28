#!/usr/bin/env python3
"""Run Resemble Detect on reviewed CLAIMFORGE image tasks.

The API token is read only from ``RESEMBLE_API_TOKEN``. Inputs use the same
metadata-free JPEG protocol as the Hive paired pilot. Returned IFL heatmaps
and visualization images are decoded into local artifacts rather than being
embedded in JSONL results.
"""

from __future__ import annotations

import argparse
import base64
import io
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
from PIL import Image

from eval.commercial.run_hive import (
    canonicalize,
    load_benchmark_items,
    load_selected_items,
)
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


DEFAULT_ENDPOINT = "https://app.resemble.ai/api/v2/detect"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/resemble/"
    "pilot_good_mouse_pairs5_canonical_jpeg_q95_20260720.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


def artifact_metadata(
    value: Any,
    destination_stem: Path,
    repo_root: Path,
    session: requests.Session,
) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    mime_type: str | None = None
    raw: bytes
    source_type: str
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
            mime_type = header[5:].split(";", 1)[0]
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            return None
        source_type = "data_uri"
    elif value.startswith(("https://", "http://")):
        response = session.get(value, timeout=(15, 120), allow_redirects=True)
        response.raise_for_status()
        raw = response.content
        mime_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        source_type = "url"
    else:
        return None

    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        image_format = str(image.format or "PNG").upper()
        width, height = image.size
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        image_format, ".bin"
    )
    destination = destination_stem.with_suffix(extension)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    try:
        relative_path = destination.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        relative_path = destination.resolve().as_posix()
    return {
        "path": relative_path,
        "sha256": sha256_file(destination),
        "bytes": len(raw),
        "size": [width, height],
        "format": image_format,
        "mime_type": mime_type,
        "source_type": source_type,
    }


def parse_success(
    body: dict[str, Any],
    item: ImageItem,
    artifact_root: Path,
    repo_root: Path,
    session: requests.Session,
) -> dict[str, Any] | None:
    provider_item = body.get("item")
    if body.get("success") is not True or not isinstance(provider_item, dict):
        return None
    metrics = provider_item.get("image_metrics")
    if provider_item.get("status") != "completed" or not isinstance(metrics, dict):
        return None
    score = as_probability(metrics.get("score"))
    label = metrics.get("label")
    if score is None or not isinstance(label, str):
        return None
    ifl = metrics.get("ifl") if isinstance(metrics.get("ifl"), dict) else {}
    ifl_score = as_probability(ifl.get("score"))
    heatmap = artifact_metadata(
        ifl.get("heatmap"), artifact_root / "heatmaps" / item.id, repo_root, session
    )
    visualization = artifact_metadata(
        metrics.get("image"),
        artifact_root / "visualizations" / item.id,
        repo_root,
        session,
    )
    return {
        "provider_status": provider_item.get("status"),
        "provider_uuid": provider_item.get("uuid"),
        "provider_label": label,
        "provider_score": score,
        "provider_fake": label.strip().lower() == "fake",
        "ifl_score": ifl_score,
        "heatmap": heatmap,
        "visualization": visualization,
        "image_metrics_type": metrics.get("type"),
        "image_metrics_children": redact_payload(metrics.get("children")),
        "c2pa_manifest": redact_payload(provider_item.get("c2pa_manifest")),
        "visualize": provider_item.get("visualize"),
        "zero_retention_mode": provider_item.get("zero_retention_mode"),
    }


def classify(
    session: requests.Session,
    item: ImageItem,
    upload_path: Path,
    upload: dict[str, Any],
    endpoint: str,
    token: str,
    timeout: tuple[float, float],
    max_attempts: int,
    run_id: str,
    manifest_sha256: str,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        response: requests.Response | None = None
        started = time.monotonic()
        try:
            with upload_path.open("rb") as handle:
                response = session.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Prefer": "wait",
                    },
                    data={"visualize": "true"},
                    files={"file": ("image.jpg", handle, "image/jpeg")},
                    timeout=timeout,
                    allow_redirects=False,
                )
            latency_ms = round((time.monotonic() - started) * 1000)
            try:
                body: Any = response.json()
            except ValueError:
                body = None
            parsed = (
                parse_success(
                    body, item, artifact_root, repo_root, session
                )
                if response.status_code == 200 and isinstance(body, dict)
                else None
            )
            if parsed is not None:
                return {
                    "schema_version": "resemble_result_v1",
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
            message = response.text[:1000]
            if isinstance(body, dict):
                message = json.dumps(redact_payload(body), ensure_ascii=False)[:1000]
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "error_message": redact_text(message, token),
                }
            )
            if response.status_code not in RETRYABLE_HTTP:
                break
        except (requests.RequestException, OSError, ValueError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code if response is not None else None,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), token),
                }
            )
        if attempt < max_attempts:
            time.sleep(min(30.0, 2 ** (attempt - 1)))

    return {
        "schema_version": "resemble_result_v1",
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


def numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(
        float(row[field])
        for row in rows
        if row.get("status") == "ok" and as_probability(row.get(field)) is not None
    )
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
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
    score_deltas = [
        pair["forged"]["provider_score"] - pair["real"]["provider_score"]
        for pair in by_task.values()
        if "real" in pair and "forged" in pair
    ]
    ifl_deltas = [
        pair["forged"]["ifl_score"] - pair["real"]["ifl_score"]
        for pair in by_task.values()
        if "real" in pair
        and "forged" in pair
        and as_probability(pair["real"].get("ifl_score")) is not None
        and as_probability(pair["forged"].get("ifl_score")) is not None
    ]
    summary = {
        "schema_version": "resemble_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "include": include,
        "expected_tasks": len({item.task_id for item in items}),
        "expected_images": len(items),
        "valid_images": len(ok_rows),
        "error_images": len(rows) - len(ok_rows),
        "heatmaps_saved": sum(bool(row.get("heatmap")) for row in ok_rows),
        "visualizations_saved": sum(
            bool(row.get("visualization")) for row in ok_rows
        ),
        "by_kind": {
            kind: {
                "count": len(kind_rows := [
                    row for row in ok_rows if row.get("kind") == kind
                ]),
                "labels": dict(Counter(row.get("provider_label") for row in kind_rows)),
                "provider_fake": sum(row.get("provider_fake") is True for row in kind_rows),
                "provider_score": numeric_summary(kind_rows, "provider_score"),
                "ifl_score": numeric_summary(kind_rows, "ifl_score"),
            }
            for kind in ("real", "forged")
        },
        "paired_provider_score_delta": {
            "count": len(score_deltas),
            "mean": statistics.fmean(score_deltas) if score_deltas else None,
            "median": statistics.median(score_deltas) if score_deltas else None,
            "min": min(score_deltas) if score_deltas else None,
            "max": max(score_deltas) if score_deltas else None,
        },
        "paired_ifl_score_delta": {
            "count": len(ifl_deltas),
            "mean": statistics.fmean(ifl_deltas) if ifl_deltas else None,
            "median": statistics.median(ifl_deltas) if ifl_deltas else None,
            "min": min(ifl_deltas) if ifl_deltas else None,
            "max": max(ifl_deltas) if ifl_deltas else None,
        },
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
    artifact_root: Path,
    items: list[ImageItem],
    manifest_sha256: str,
    endpoint: str,
    quality: int,
    include: str,
    run_id: str,
    candidate: str = "mouse",
    input_selection: dict[str, Any] | None = None,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "resemble_run_manifest_v1",
        "run_id": run_id,
        "endpoint": endpoint,
        "candidate": candidate,
        "include": include,
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
        "visualize": True,
        "zero_retention_mode": False,
        "artifact_root": artifact_root.as_posix(),
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "metadata": "stripped",
            "filename": "image.jpg",
        },
    }
    if input_selection is not None:
        expected["input_selection"] = input_selection
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
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        help="load images from a CLAIMFORGE benchmark image manifest",
    )
    parser.add_argument("--benchmark-category", default="mouse")
    parser.add_argument(
        "--benchmark-method",
        choices=("local_splice", "full_image"),
        default="full_image",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--include", choices=("forged", "both"), default="both")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-pending",
        type=int,
        help="process at most this many currently pending images",
    )
    parser.add_argument("--run-id", default="resemble_pilot_good_mouse_pairs5_20260720")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tasks < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--tasks must be positive and JPEG quality must be in [1, 100]")
    if args.max_attempts < 1 or args.workers < 1:
        parser.error("--max-attempts and --workers must be positive")
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
    artifact_root = args.artifact_root or output_path.with_suffix("")
    artifact_root = (
        artifact_root if artifact_root.is_absolute() else repo_root / artifact_root
    ).resolve()
    input_selection: dict[str, Any] | None = None
    if args.benchmark_manifest is not None:
        benchmark_manifest = (
            args.benchmark_manifest
            if args.benchmark_manifest.is_absolute()
            else repo_root / args.benchmark_manifest
        )
        benchmark_manifest = benchmark_manifest.resolve()
        try:
            benchmark_manifest.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"benchmark manifest escapes repo: {args.benchmark_manifest}"
            ) from exc
        items = load_benchmark_items(
            repo_root,
            benchmark_manifest,
            args.tasks,
            args.include,
            args.benchmark_category,
            args.benchmark_method,
        )
        input_selection = {
            "kind": "benchmark_manifest",
            "manifest": benchmark_manifest.relative_to(repo_root).as_posix(),
            "category": args.benchmark_category,
            "method": args.benchmark_method,
        }
    else:
        items = load_selected_items(
            repo_root, review_path, order_path, args.tasks, args.include
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
                "selected_tasks": args.tasks,
                "include": args.include,
                "selected_images": len(items),
                "already_valid": len(items) - pending_total,
                "pending": pending_total,
                "scheduled": len(pending),
                "output": output_path.as_posix(),
                "artifact_root": artifact_root.as_posix(),
                "workers": args.workers,
                "dry_run": args.dry_run,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return

    token = os.environ.get("RESEMBLE_API_TOKEN", "")
    if not token:
        raise SystemExit("RESEMBLE_API_TOKEN must be set")
    ensure_run_manifest(
        output_path,
        artifact_root,
        items,
        manifest_sha256,
        args.endpoint,
        args.jpeg_quality,
        args.include,
        args.run_id,
        args.benchmark_category if input_selection is not None else "mouse",
        input_selection,
    )
    with tempfile.TemporaryDirectory(prefix="claimforge-resemble-") as temporary:
        temporary_dir = Path(temporary)

        def run_one(index: int, item: ImageItem) -> dict[str, Any]:
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload = canonicalize(item.path, upload_path, args.jpeg_quality)
            with requests.Session() as session:
                session.headers.update(
                    {"User-Agent": "claimforge-benchmark/resemble-pilot-v1"}
                )
                return classify(
                    session,
                    item,
                    upload_path,
                    upload,
                    args.endpoint,
                    token,
                    (args.connect_timeout, args.read_timeout),
                    args.max_attempts,
                    args.run_id,
                    manifest_sha256,
                    artifact_root,
                    repo_root,
                )

        indexed_pending = iter(enumerate(pending))
        futures: dict[Future[dict[str, Any]], tuple[int, ImageItem]] = {}
        stop_submitting = False
        completed_this_run = 0

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            try:
                index, item = next(indexed_pending)
            except StopIteration:
                return False
            futures[executor.submit(run_one, index, item)] = (index, item)
            if args.minimum_interval > 0:
                time.sleep(args.minimum_interval)
            return True

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(min(args.workers, len(pending))):
                submit_next(executor)
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    _, item = futures.pop(future)
                    row = future.result()
                    append_jsonl(output_path, row)
                    completed_this_run += 1
                    print(
                        json.dumps(
                            {
                                "completed_this_run": completed_this_run,
                                "id": row["id"],
                                "status": row["status"],
                                "provider_label": row.get("provider_label"),
                                "provider_score": row.get("provider_score"),
                                "ifl_score": row.get("ifl_score"),
                                "has_heatmap": bool(row.get("heatmap")),
                            }
                        ),
                        flush=True,
                    )
                    if row["status"] == "error":
                        final = (row.get("attempts") or [{}])[-1]
                        if final.get("http_status") in {401, 402, 403, 429}:
                            stop_submitting = True
                            print(
                                "authentication, credit, or quota failure; "
                                "stopping new submissions",
                                flush=True,
                            )
                    if not stop_submitting:
                        submit_next(executor)

    summary = write_summary(output_path, items, manifest_sha256, args.include)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
