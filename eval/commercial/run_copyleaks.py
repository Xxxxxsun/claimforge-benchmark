#!/usr/bin/env python3
"""Run Copyleaks AI Image Detection Ultra on reviewed CLAIMFORGE images.

Credentials are read only from ``COPYLEAKS_EMAIL`` and
``COPYLEAKS_API_KEY``. Inputs are converted to metadata-free RGB PNGs and,
when necessary, resized to satisfy Copyleaks' 512x512 minimum. The returned
RLE mask is preserved and compared with CLAIMFORGE's exact pixel-difference
mask for forged images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageChops, ImageOps

from eval.commercial.run_hive import load_selected_items
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


DEFAULT_LOGIN_ENDPOINT = "https://id.copyleaks.com/v3/account/login/api"
DEFAULT_ENDPOINT_TEMPLATE = (
    "https://api.copyleaks.com/v1/ai-image-detector/{scan_id}/check"
)
DEFAULT_MODEL = "ai-image-1-ultra"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/copyleaks/"
    "pilot_good_mouse_pairs5_canonical_png_20260720.jsonl"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
MIN_SIDE = 512
MAX_WIDTH = 6000
MAX_HEIGHT = 4500
MAX_PIXELS = 27_000_000


def load_review_records(review_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"review export has no records list: {review_path}")
    return {
        str(record["task_id"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("task_id"), str)
    }


def target_size(width: int, height: int) -> tuple[int, int]:
    minimum_scale = max(1.0, MIN_SIDE / width, MIN_SIDE / height)
    maximum_scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        math.sqrt(MAX_PIXELS / (width * height)),
    )
    if minimum_scale > maximum_scale:
        raise ValueError(
            f"image aspect ratio cannot satisfy Copyleaks limits: {width}x{height}"
        )
    scale = minimum_scale if minimum_scale > 1 else min(1.0, maximum_scale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def canonicalize_png(
    source: Path,
    destination: Path,
    forced_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        upload_size = forced_size or target_size(*original_size)
        if upload_size != original_size:
            image = image.resize(upload_size, Image.Resampling.LANCZOS)
        if (
            upload_size[0] < MIN_SIDE
            or upload_size[1] < MIN_SIDE
            or upload_size[0] > MAX_WIDTH
            or upload_size[1] > MAX_HEIGHT
            or upload_size[0] * upload_size[1] > MAX_PIXELS
        ):
            raise ValueError(f"invalid canonical upload size: {upload_size}")
        image.save(destination, format="PNG", optimize=False, compress_level=6)
    return {
        "upload_sha256": sha256_file(destination),
        "upload_bytes": destination.stat().st_size,
        "upload_size": list(upload_size),
        "source_decoded_size": list(original_size),
        "upload_resized": upload_size != original_size,
    }


def login(
    session: requests.Session,
    endpoint: str,
    email: str,
    api_key: str,
    timeout: tuple[float, float],
) -> str:
    response = session.post(
        endpoint,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"email": email, "key": api_key},
        timeout=timeout,
        allow_redirects=False,
    )
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    token = body.get("access_token") if isinstance(body, dict) else None
    if response.status_code != 200 or not isinstance(token, str) or not token:
        message = response.text[:1500]
        if isinstance(body, (dict, list)):
            message = json.dumps(redact_payload(body), ensure_ascii=False)[:1500]
        message = redact_text(redact_text(message, api_key), email)
        raise RuntimeError(f"Copyleaks login failed ({response.status_code}): {message}")
    return token


def make_scan_id(run_id: str, item: ImageItem) -> str:
    payload = f"{run_id}:{item.id}:{item.sha256}".encode()
    return "cfcl" + hashlib.sha256(payload).hexdigest()[:28]


def parse_success(body: dict[str, Any]) -> dict[str, Any] | None:
    summary = body.get("summary")
    result = body.get("result")
    image_info = body.get("imageInfo")
    scanned = body.get("scannedDocument")
    if not all(isinstance(value, dict) for value in (summary, result, image_info)):
        return None
    ai_score = as_probability(summary.get("ai"))
    human_score = as_probability(summary.get("human"))
    detected = body.get("isAiDetected")
    starts = result.get("starts")
    lengths = result.get("lengths")
    shape = image_info.get("shape")
    if (
        ai_score is None
        or human_score is None
        or not isinstance(detected, bool)
        or not isinstance(starts, list)
        or not isinstance(lengths, list)
        or len(starts) != len(lengths)
        or not isinstance(shape, dict)
    ):
        return None
    try:
        starts = [int(value) for value in starts]
        lengths = [int(value) for value in lengths]
        width = int(shape["width"])
        height = int(shape["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        width < 1
        or height < 1
        or any(start < 0 for start in starts)
        or any(length < 0 for length in lengths)
    ):
        return None
    scanned = scanned if isinstance(scanned, dict) else {}
    return {
        "provider_model": body.get("model"),
        "ai_score": ai_score,
        "human_score": human_score,
        "is_ai_detected": detected,
        "image_info": redact_payload(image_info),
        "provider_scan_id": scanned.get("scanId"),
        "actual_credits": scanned.get("actualCredits"),
        "expected_credits": scanned.get("expectedCredits"),
        "provider_creation_time": scanned.get("creationTime"),
        "rle": {"starts": starts, "lengths": lengths},
        "rle_width": width,
        "rle_height": height,
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
    endpoint_template: str,
    token: str,
    model: str,
    sandbox: bool,
    timeout: tuple[float, float],
    max_attempts: int,
    run_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    scan_id = make_scan_id(run_id, item)
    endpoint = endpoint_template.format(scan_id=scan_id)
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
                        "Accept-Encoding": "gzip",
                    },
                    files={"image": ("image.png", handle, "image/png")},
                    data={
                        "filename": "image.png",
                        "sandbox": str(sandbox).lower(),
                        "model": model,
                    },
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
                    "schema_version": "copyleaks_result_v1",
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
                    "sandbox": sandbox,
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
                    "error_message": redact_text(message, token),
                }
            )
            if (
                response.status_code not in RETRYABLE_HTTP
                and not 500 <= response.status_code < 600
            ):
                break
        except (requests.RequestException, OSError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status_code if response is not None else None,
                    "error_type": type(exc).__name__,
                    "error_message": redact_text(str(exc), token),
                }
            )
        if attempt < max_attempts:
            time.sleep(retry_delay(attempt, response))

    return {
        "schema_version": "copyleaks_result_v1",
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


def decode_rle_mask(
    starts: list[int], lengths: list[int], width: int, height: int
) -> Image.Image:
    total = width * height
    mask = bytearray(total)
    for start, length in zip(starts, lengths, strict=True):
        end = start + length
        if start < 0 or length < 0 or end > total:
            raise ValueError(
                f"RLE run is outside image: start={start}, length={length}, total={total}"
            )
        mask[start:end] = b"\xff" * length
    return Image.frombytes("L", (width, height), bytes(mask))


def binary_pixel_count(mask: Image.Image) -> int:
    return int(mask.histogram()[255])


def binary_overlap(prediction: Image.Image, target: Image.Image) -> dict[str, Any]:
    if prediction.size != target.size:
        raise ValueError(f"mask size mismatch: {prediction.size} != {target.size}")
    pred_pixels = binary_pixel_count(prediction)
    target_pixels = binary_pixel_count(target)
    intersection = binary_pixel_count(ImageChops.multiply(prediction, target))
    union = pred_pixels + target_pixels - intersection
    return {
        "predicted_pixels": pred_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "precision": intersection / pred_pixels if pred_pixels else None,
        "recall": intersection / target_pixels if target_pixels else None,
        "iou": intersection / union if union else None,
        "any_overlap": intersection > 0,
    }


def exact_diff_mask(
    source_path: Path, forged_path: Path, threshold: int
) -> Image.Image:
    with Image.open(source_path) as source_opened, Image.open(forged_path) as forged_opened:
        source = source_opened.convert("RGB")
        forged = forged_opened.convert("RGB")
        if source.size != forged.size:
            raise ValueError(f"canonical pair size mismatch: {source.size} != {forged.size}")
        red, green, blue = ImageChops.difference(source, forged).split()
        maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
        return maximum.point(lambda value: 255 if value > threshold else 0, mode="L")


def scaled_box(
    raw_box: Any,
    original_size: tuple[int, int],
    upload_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw_box)
    except (TypeError, ValueError):
        return None
    scale_x = upload_size[0] / original_size[0]
    scale_y = upload_size[1] / original_size[1]
    box = (
        max(0, min(upload_size[0], math.floor(x1 * scale_x))),
        max(0, min(upload_size[1], math.floor(y1 * scale_y))),
        max(0, min(upload_size[0], math.ceil(x2 * scale_x))),
        max(0, min(upload_size[1], math.ceil(y2 * scale_y))),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def box_overlap(mask: Image.Image, box: tuple[int, int, int, int]) -> dict[str, Any]:
    mask_pixels = binary_pixel_count(mask)
    pixels_in_box = binary_pixel_count(mask.crop(box))
    box_pixels = (box[2] - box[0]) * (box[3] - box[1])
    return {
        "box_xyxy": list(box),
        "box_pixels": box_pixels,
        "predicted_pixels_in_box": pixels_in_box,
        "fraction_of_prediction_in_box": pixels_in_box / mask_pixels
        if mask_pixels
        else None,
        "fraction_of_box_predicted": pixels_in_box / box_pixels if box_pixels else None,
        "any_overlap": pixels_in_box > 0,
    }


def add_localization(
    row: dict[str, Any],
    item: ImageItem,
    upload_path: Path,
    source_upload_path: Path | None,
    record: dict[str, Any],
    diff_threshold: int,
) -> None:
    width = int(row["rle_width"])
    height = int(row["rle_height"])
    upload_size = tuple(int(value) for value in row["upload_size"])
    if (width, height) != upload_size:
        raise ValueError(f"provider shape {(width, height)} != upload {upload_size}")
    rle = row["rle"]
    prediction = decode_rle_mask(rle["starts"], rle["lengths"], width, height)
    predicted_pixels = binary_pixel_count(prediction)
    total_pixels = width * height
    localization: dict[str, Any] = {
        "predicted_pixels": predicted_pixels,
        "predicted_fraction": predicted_pixels / total_pixels,
        "predicted_bbox_xyxy": list(prediction.getbbox()) if prediction.getbbox() else None,
        "summary_ai_minus_rle_fraction": row["ai_score"]
        - predicted_pixels / total_pixels,
    }
    raw_original_size = record.get("image_size")
    if isinstance(raw_original_size, list) and len(raw_original_size) == 2:
        original_size = (int(raw_original_size[0]), int(raw_original_size[1]))
        for name, field in (
            ("edit_box", "edit_region_xyxy"),
            ("context_box", "context_region_xyxy"),
        ):
            box = scaled_box(record.get(field), original_size, upload_size)
            if box is not None:
                localization[name] = box_overlap(prediction, box)
    if item.kind == "forged":
        if source_upload_path is None:
            raise ValueError("forged localization requires canonical source image")
        target = exact_diff_mask(source_upload_path, upload_path, diff_threshold)
        localization["pixel_diff_gt"] = {
            "channel_absolute_difference_threshold": diff_threshold,
            "target_bbox_xyxy": list(target.getbbox()) if target.getbbox() else None,
            **binary_overlap(prediction, target),
        }
    row["localization"] = localization


def value_summary(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["ai_score"])
        for row in rows
        if row.get("status") == "ok" and as_probability(row.get("ai_score")) is not None
    ]
    detected = sum(bool(row.get("is_ai_detected")) for row in rows if row.get("status") == "ok")
    return {
        **value_summary(scores),
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
        pair["forged"]["ai_score"] - pair["real"]["ai_score"]
        for pair in by_task.values()
        if "real" in pair and "forged" in pair
    ]
    forged_rows = [row for row in ok_rows if row.get("kind") == "forged"]
    localization_pairs = [
        (row, row["localization"]["pixel_diff_gt"])
        for row in forged_rows
        if isinstance(row.get("localization", {}).get("pixel_diff_gt"), dict)
    ]

    def localization_metric_summary(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            metric: value_summary(
                [
                    float(localization[metric])
                    for _, localization in pairs
                    if localization.get(metric) is not None
                ]
            )
            for metric in ("precision", "recall", "iou")
        }

    detected_localization_pairs = [
        pair for pair in localization_pairs if pair[0].get("is_ai_detected") is True
    ]
    localization_summary = {
        "evaluated": len(localization_pairs),
        "detected_masks": len(detected_localization_pairs),
        "empty_masks": len(localization_pairs) - len(detected_localization_pairs),
        "any_overlap": sum(
            bool(localization.get("any_overlap"))
            for _, localization in localization_pairs
        ),
        "all_forged": localization_metric_summary(localization_pairs),
        "detected_only": localization_metric_summary(detected_localization_pairs),
    }
    summary = {
        "schema_version": "copyleaks_summary_v1",
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
        "paired_ai_score_delta": value_summary([float(value) for value in deltas]),
        "forged_localization": localization_summary,
        "actual_credits": sum(
            float(row["actual_credits"])
            for row in ok_rows
            if isinstance(row.get("actual_credits"), (int, float))
        ),
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
    login_endpoint: str,
    endpoint_template: str,
    model: str,
    run_id: str,
    include: str,
    sandbox: bool,
    diff_threshold: int,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "copyleaks_run_manifest_v1",
        "run_id": run_id,
        "login_endpoint": login_endpoint,
        "endpoint_template": endpoint_template,
        "model": model,
        "candidate": "mouse",
        "include": "paired_real_and_forged" if include == "both" else "forged",
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
        "sandbox": sandbox,
        "upload": {
            "format": "PNG",
            "color_mode": "RGB",
            "metadata": "stripped",
            "minimum_side": MIN_SIDE,
            "resize_filter": "Lanczos",
            "filename": "image.png",
        },
        "localization": {
            "rle_order": "row_major_zero_based_per_vendor_example",
            "ground_truth": "canonical_source_vs_forged_exact_pixel_difference",
            "channel_absolute_difference_threshold": diff_threshold,
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
    parser.add_argument("--login-endpoint", default=DEFAULT_LOGIN_ENDPOINT)
    parser.add_argument("--endpoint-template", default=DEFAULT_ENDPOINT_TEMPLATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--include", choices=("forged", "both"), default="both")
    parser.add_argument("--diff-threshold", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--run-id", default="copyleaks_mouse_pairs5_20260720")
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.tasks < 1
        or args.task_offset < 0
        or args.max_attempts < 1
        or args.minimum_interval < 0
    ):
        parser.error(
            "tasks/max-attempts must be positive; offset/interval must be non-negative"
        )
    if not 0 <= args.diff_threshold <= 255:
        parser.error("--diff-threshold must be in [0, 255]")
    if "{scan_id}" not in args.endpoint_template:
        parser.error("--endpoint-template must contain {scan_id}")

    repo_root = args.repo_root.resolve()
    review_path = args.review if args.review.is_absolute() else repo_root / args.review
    order_path = (
        args.order_manifest
        if args.order_manifest.is_absolute()
        else repo_root / args.order_manifest
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    ordered_items = load_selected_items(
        repo_root,
        review_path,
        order_path,
        args.tasks + args.task_offset,
        args.include,
    )
    task_order = list(dict.fromkeys(item.task_id for item in ordered_items))
    selected_task_ids = set(
        task_order[args.task_offset : args.task_offset + args.tasks]
    )
    items = [item for item in ordered_items if item.task_id in selected_task_ids]
    records = load_review_records(review_path)
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
                "model": args.model,
                "sandbox": args.sandbox,
                "output": output_path.as_posix(),
                "dry_run": args.dry_run,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return

    email = os.environ.get("COPYLEAKS_EMAIL", "")
    api_key = os.environ.get("COPYLEAKS_API_KEY", "")
    if not email or not api_key:
        raise SystemExit("COPYLEAKS_EMAIL and COPYLEAKS_API_KEY must be set")
    ensure_run_manifest(
        output_path,
        items,
        manifest_sha256,
        args.login_endpoint,
        args.endpoint_template,
        args.model,
        args.run_id,
        args.include,
        args.sandbox,
        args.diff_threshold,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "claimforge-benchmark/copyleaks-v1"})
    token = login(
        session,
        args.login_endpoint,
        email,
        api_key,
        (args.connect_timeout, args.read_timeout),
    )
    with tempfile.TemporaryDirectory(prefix="claimforge-copyleaks-") as temporary:
        temporary_dir = Path(temporary)
        for index, item in enumerate(pending):
            upload_path = temporary_dir / f"upload-{index:04d}.png"
            upload = canonicalize_png(item.path, upload_path)
            row = classify(
                session,
                item,
                upload_path,
                upload,
                args.endpoint_template,
                token,
                args.model,
                args.sandbox,
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                args.run_id,
                manifest_sha256,
            )
            if row["status"] == "ok":
                source_upload_path: Path | None = None
                if item.kind == "forged":
                    source_relative = str(records[item.task_id]["source_image"])
                    source_path = (repo_root / source_relative).resolve()
                    source_upload_path = temporary_dir / f"source-{index:04d}.png"
                    canonicalize_png(
                        source_path,
                        source_upload_path,
                        forced_size=tuple(int(value) for value in upload["upload_size"]),
                    )
                add_localization(
                    row,
                    item,
                    upload_path,
                    source_upload_path,
                    records[item.task_id],
                    args.diff_threshold,
                )
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "ai_score": row.get("ai_score"),
                        "is_ai_detected": row.get("is_ai_detected"),
                        "predicted_fraction": row.get("localization", {}).get(
                            "predicted_fraction"
                        ),
                        "pixel_diff_iou": row.get("localization", {})
                        .get("pixel_diff_gt", {})
                        .get("iou"),
                        "actual_credits": row.get("actual_credits"),
                        "error": (row.get("attempts") or [{}])[-1].get(
                            "error_message"
                        )
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
