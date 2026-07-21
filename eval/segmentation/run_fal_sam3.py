#!/usr/bin/env python3
"""Run a resumable fal SAM 3/3.1 pilot and materialize hybrid splice masks.

The credential is read only from ``FAL_KEY``. It is never accepted as a CLI
argument, written to disk, or included in diagnostic output. Queue request IDs
are appended before polling so an interrupted run can resume without issuing a
duplicate billable request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy import ndimage

from compose_spliced_full import box_mask, object_mask


REPO = Path(__file__).resolve().parents[2]
QUEUE_BASE = "https://queue.fal.run"
ENDPOINTS = {
    "sam3": "fal-ai/sam-3/image-rle",
    "sam3_1": "fal-ai/sam-3-1/image-rle",
}
ENDPOINT_COST_USD = {"sam3": 0.005, "sam3_1": 0.01}
DEFAULT_BASE_MANIFEST = Path(
    "spliced_full/hunyuan_image3_distil_cat_272_fullblue_t30/manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/segmentation/fal_sam3_cat_pilot10_20260721"
)
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class PilotItem:
    task_id: str
    domain: str
    source_path: Path
    source_relative: str
    generated_path: Path
    generated_relative: str
    context_box: tuple[int, int, int, int]
    edit_box: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    input_sha256: str
    input_bytes: int
    edit_area_fraction: float
    threshold_disagreement: float
    selection_reason: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in load_jsonl(path):
        row_id = row.get("id")
        if isinstance(row_id, str):
            latest[row_id] = row
    return latest


def safe_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = (repo_root / raw_path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {raw_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def current_threshold_mask(
    original: Image.Image,
    generated: Image.Image,
    edit_box: Sequence[int],
    threshold: float,
) -> np.ndarray:
    mask = object_mask(
        original,
        generated,
        edit_box,
        threshold,
        0,
        object_pad=0,
        search_mode="context",
    )
    return np.asarray(mask, dtype=np.uint8) >= 128


def domain_from_task_id(task_id: str) -> str:
    parts = task_id.split("_")
    if len(parts) >= 2 and parts[0] == "cat":
        return parts[1]
    return parts[0]


def load_pilot_candidates(repo_root: Path, manifest_path: Path) -> list[PilotItem]:
    candidates: list[PilotItem] = []
    for row in load_jsonl(manifest_path):
        if row.get("status") != "ok" or row.get("candidates") != "cat":
            continue
        task_id = str(row["task_id"])
        source_relative = str(row["source_image"])
        generated_relative = str(row["generated_crop"])
        source_path = safe_repo_path(repo_root, source_relative)
        generated_path = safe_repo_path(repo_root, generated_relative)
        context_box = tuple(int(value) for value in row["context_region_xyxy"])
        edit_box = tuple(int(value) for value in row["edit_region_in_context_xyxy"])
        if len(context_box) != 4 or len(edit_box) != 4:
            raise ValueError(f"{task_id}: invalid box")
        cx1, cy1, cx2, cy2 = context_box
        crop_size = (cx2 - cx1, cy2 - cy1)
        with Image.open(source_path) as source_image, Image.open(generated_path) as generated_image:
            original = source_image.convert("RGB").crop(context_box)
            generated = generated_image.convert("RGB")
            if generated.size != crop_size:
                raise ValueError(
                    f"{task_id}: generated size {generated.size} != context {crop_size}"
                )
            mask30 = current_threshold_mask(original, generated, edit_box, 30)
            mask40 = current_threshold_mask(original, generated, edit_box, 40)
        union = np.logical_or(mask30, mask40).sum()
        disagreement = (
            float(np.logical_xor(mask30, mask40).sum() / union) if union else 0.0
        )
        ex1, ey1, ex2, ey2 = edit_box
        edit_fraction = max(0, ex2 - ex1) * max(0, ey2 - ey1) / float(
            crop_size[0] * crop_size[1]
        )
        candidates.append(
            PilotItem(
                task_id=task_id,
                domain=domain_from_task_id(task_id),
                source_path=source_path,
                source_relative=source_relative,
                generated_path=generated_path,
                generated_relative=generated_relative,
                context_box=context_box,
                edit_box=edit_box,
                crop_size=crop_size,
                input_sha256=sha256_file(generated_path),
                input_bytes=generated_path.stat().st_size,
                edit_area_fraction=edit_fraction,
                threshold_disagreement=disagreement,
            )
        )
    if not candidates:
        raise ValueError(f"no cat candidates found in {manifest_path}")
    ids = [item.task_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate manifest contains duplicate task IDs")
    return sorted(candidates, key=lambda item: item.task_id)


def select_pilot_items(candidates: Sequence[PilotItem], count: int) -> list[PilotItem]:
    """Stratify by domain and intended edit area, prioritizing t30/t40 disagreement."""
    if count < 1 or count > len(candidates):
        raise ValueError("pilot count must be positive and no larger than candidates")
    grouped: dict[str, list[PilotItem]] = defaultdict(list)
    for item in candidates:
        grouped[item.domain].append(item)
    domains = sorted(grouped)
    base, remainder = divmod(count, len(domains))
    quotas = {
        domain: min(len(grouped[domain]), base + (index < remainder))
        for index, domain in enumerate(domains)
    }
    shortfall = count - sum(quotas.values())
    while shortfall:
        changed = False
        for domain in domains:
            if quotas[domain] < len(grouped[domain]):
                quotas[domain] += 1
                shortfall -= 1
                changed = True
                if not shortfall:
                    break
        if not changed:
            raise ValueError("could not satisfy pilot quota")

    selected: list[PilotItem] = []
    for domain in domains:
        pool = sorted(
            grouped[domain], key=lambda item: (item.edit_area_fraction, item.task_id)
        )
        quota = quotas[domain]
        bucketed: dict[str, list[PilotItem]] = {"small": [], "medium": [], "large": []}
        labels = ("small", "medium", "large")
        for rank, item in enumerate(pool):
            bucket = min(2, (rank * 3) // max(1, len(pool)))
            bucketed[labels[bucket]].append(item)
        chosen_ids: set[str] = set()
        for label in labels:
            if len(chosen_ids) >= quota or not bucketed[label]:
                continue
            item = sorted(
                bucketed[label],
                key=lambda candidate: (
                    -candidate.threshold_disagreement,
                    candidate.task_id,
                ),
            )[0]
            selected.append(
                replace(
                    item,
                    selection_reason=f"{domain}:{label}:max_t30_t40_disagreement",
                )
            )
            chosen_ids.add(item.task_id)
        extras = sorted(
            (item for item in pool if item.task_id not in chosen_ids),
            key=lambda item: (-item.threshold_disagreement, item.task_id),
        )
        for item in extras[: max(0, quota - len(chosen_ids))]:
            selected.append(
                replace(item, selection_reason=f"{domain}:high_t30_t40_disagreement")
            )
            chosen_ids.add(item.task_id)
    if len(selected) != count:
        raise AssertionError(f"selected {len(selected)} items, expected {count}")
    return sorted(
        selected,
        key=lambda item: (-item.threshold_disagreement, item.domain, item.task_id),
    )


def item_record(item: PilotItem, rank: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": item.task_id,
        "domain": item.domain,
        "source_image": item.source_relative,
        "generated_crop": item.generated_relative,
        "generated_sha256": item.input_sha256,
        "generated_bytes": item.input_bytes,
        "crop_size": list(item.crop_size),
        "context_region_xyxy": list(item.context_box),
        "edit_region_in_context_xyxy": list(item.edit_box),
        "edit_area_fraction": item.edit_area_fraction,
        "t30_t40_mask_disagreement": item.threshold_disagreement,
        "selection_reason": item.selection_reason,
    }
    if rank is not None:
        row["rank"] = rank
    return row


def selection_digest(items: Sequence[PilotItem]) -> str:
    return sha256_json(
        [
            {
                "task_id": item.task_id,
                "generated_sha256": item.input_sha256,
                "context_box": item.context_box,
                "edit_box": item.edit_box,
            }
            for item in items
        ]
    )


def ensure_run_manifest(
    output_dir: Path,
    base_manifest: Path,
    items: Sequence[PilotItem],
    endpoint_tags: Sequence[str],
    prompt: str,
    max_masks: int,
    diff_threshold: float,
    support_radius: int,
    feather: float,
    use_box_prompt: bool,
) -> Path:
    path = output_dir / "run_manifest.json"
    expected = {
        "schema_version": "fal_sam3_splice_run_v1",
        "provider": "fal",
        "endpoints": [ENDPOINTS[tag] for tag in endpoint_tags],
        "prompt": prompt,
        "max_masks": max_masks,
        "expected_tasks": len(items),
        "expected_requests": len(items) * len(endpoint_tags),
        "selection_sha256": selection_digest(items),
        "base_manifest": base_manifest.relative_to(REPO).as_posix(),
        "base_manifest_sha256": sha256_file(base_manifest),
        "request_input": {
            "prompt": prompt,
            "box_prompt": use_box_prompt,
            "apply_mask": False,
            "return_multiple_masks": True,
            "max_masks": max_masks,
            "include_scores": True,
            "include_boxes": True,
        },
        "postprocess": {
            "diff_threshold": diff_threshold,
            "support_radius": support_radius,
            "alpha_feather": feather,
        },
        "fal_platform_headers": {
            "X-Fal-Store-IO": "0",
            "X-Fal-Object-Lifecycle-Preference": {
                "expiration_duration_seconds": 3600
            },
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
        return path
    payload = {
        **expected,
        "created_at": utc_now(),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "requests_version": requests.__version__,
        "pillow_version": Image.__version__,
        "numpy_version": np.__version__,
        "ordered_inputs": [item_record(item, rank) for rank, item in enumerate(items)],
    }
    write_json(path, payload)
    write_json(output_dir / "selection.json", payload["ordered_inputs"])
    return path


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def request_spec(
    item: PilotItem,
    prompt: str,
    max_masks: int,
    use_box_prompt: bool = True,
) -> dict[str, Any]:
    x1, y1, x2, y2 = item.edit_box
    return {
        "prompt": prompt,
        "box_prompts": (
            [
                {
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "object_id": 1,
                }
            ]
            if use_box_prompt
            else []
        ),
        "point_prompts": [],
        "apply_mask": False,
        "sync_mode": True,
        "output_format": "png",
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }


def request_fingerprint(
    item: PilotItem,
    endpoint_tag: str,
    prompt: str,
    max_masks: int,
    use_box_prompt: bool = True,
) -> str:
    return sha256_json(
        {
            "endpoint": ENDPOINTS[endpoint_tag],
            "input_sha256": item.input_sha256,
            "spec": request_spec(item, prompt, max_masks, use_box_prompt),
        }
    )


def auth_headers(api_key: str, content_type: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Key {api_key}",
        "X-Fal-Store-IO": "0",
        "X-Fal-Object-Lifecycle-Preference": json.dumps(
            {"expiration_duration_seconds": 3600}, separators=(",", ":")
        ),
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def queue_app_id(endpoint_id: str) -> str:
    """Return the queue app root; endpoint subpaths are used only on submit.

    For example, ``fal-ai/sam-3/image-rle`` submits to that full path, while
    status/result routes live under ``fal-ai/sam-3/requests/...``.
    """
    parts = endpoint_id.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid fal endpoint ID: {endpoint_id}")
    return "/".join(parts[:2])


def safe_json(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def redact_error_text(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "<redacted-api-key>")
    text = re.sub(
        r"data:image/[^;,\s]+(?:;base64)?,[A-Za-z0-9+/=]+",
        "<redacted-data-uri>",
        text,
    )
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|token|secret)(.{0,8})[^,\s}\]]+",
        r"\1\2<redacted>",
        text,
    )
    return text[:2000]


def response_error(response: requests.Response, api_key: str) -> str:
    body = safe_json(response)
    if body is not None:
        preferred = body.get("detail", body.get("error", body.get("message", body)))
        text = json.dumps(preferred, ensure_ascii=False, sort_keys=True)
    else:
        text = response.text
    return redact_error_text(text, api_key)


def submit_request(
    session: requests.Session,
    item: PilotItem,
    endpoint_tag: str,
    api_key: str,
    prompt: str,
    max_masks: int,
    timeout: tuple[float, float],
    selection_sha256: str,
    use_box_prompt: bool,
) -> dict[str, Any]:
    endpoint_id = ENDPOINTS[endpoint_tag]
    row_id = f"{item.task_id}__{endpoint_tag}"
    public_spec = request_spec(item, prompt, max_masks, use_box_prompt)
    payload = {"image_url": image_data_uri(item.generated_path), **public_spec}
    started = time.monotonic()
    base = {
        "schema_version": "fal_sam3_api_result_v1",
        "id": row_id,
        "task_id": item.task_id,
        "domain": item.domain,
        "provider": "fal",
        "endpoint_tag": endpoint_tag,
        "endpoint": endpoint_id,
        "input_image": item.generated_relative,
        "input_sha256": item.input_sha256,
        "selection_sha256": selection_sha256,
        "request_fingerprint": request_fingerprint(
            item,
            endpoint_tag,
            prompt,
            max_masks,
            use_box_prompt,
        ),
        "request": public_spec,
    }
    try:
        response = session.post(
            f"{QUEUE_BASE}/{endpoint_id}",
            headers=auth_headers(api_key, content_type=True),
            json=payload,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        return {
            **base,
            "status": "submission_error",
            "submission_outcome": "unknown_not_retried",
            "error_type": type(exc).__name__,
            "error_message": "queue submission failed before a request ID was received",
            "submit_latency_ms": round((time.monotonic() - started) * 1000),
            "updated_at": utc_now(),
        }
    latency_ms = round((time.monotonic() - started) * 1000)
    body = safe_json(response)
    request_id = body.get("request_id") if body else None
    if not 200 <= response.status_code < 300 or not isinstance(request_id, str):
        return {
            **base,
            "status": "submission_error",
            "submission_outcome": "rejected" if response.status_code < 500 else "unknown",
            "http_status": response.status_code,
            "error_message": response_error(response, api_key),
            "submit_latency_ms": latency_ms,
            "updated_at": utc_now(),
        }
    return {
        **base,
        "status": "submitted",
        "request_id": request_id,
        "queue_position": body.get("queue_position"),
        "submit_http_status": response.status_code,
        "submit_latency_ms": latency_ms,
        "submitted_at": utc_now(),
    }


def poll_request(
    session: requests.Session,
    endpoint_id: str,
    request_id: str,
    api_key: str,
    timeout: tuple[float, float],
    poll_interval: float,
    max_poll_seconds: float,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    deadline = time.monotonic() + max_poll_seconds
    attempts: list[dict[str, Any]] = []
    app_id = queue_app_id(endpoint_id)
    while time.monotonic() < deadline:
        started = time.monotonic()
        try:
            response = session.get(
                f"{QUEUE_BASE}/{app_id}/requests/{request_id}/status",
                headers=auth_headers(api_key),
                params={"logs": "0"},
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            attempts.append(
                {
                    "error_type": type(exc).__name__,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
            )
            time.sleep(poll_interval)
            continue
        body = safe_json(response)
        attempt: dict[str, Any] = {
            "http_status": response.status_code,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        if isinstance(body, dict):
            attempt["queue_status"] = body.get("status")
            if isinstance(body.get("queue_position"), int):
                attempt["queue_position"] = body["queue_position"]
        attempts.append(attempt)
        if not 200 <= response.status_code < 300:
            if response.status_code in RETRYABLE_HTTP:
                time.sleep(poll_interval)
                continue
            return "error", {
                "http_status": response.status_code,
                "error_message": response_error(response, api_key),
            }, attempts
        status = body.get("status") if body else None
        if status == "COMPLETED":
            safe_status = {
                "status": status,
                "metrics": body.get("metrics") if isinstance(body.get("metrics"), dict) else {},
                "error": body.get("error"),
                "error_type": body.get("error_type"),
            }
            if body.get("error"):
                return "error", safe_status, attempts
            return "completed", safe_status, attempts
        time.sleep(poll_interval)
    return "pending", None, attempts


def provider_output(body: dict[str, Any]) -> dict[str, Any] | None:
    if "rle" in body:
        return body
    for key in ("data", "payload", "result"):
        child = body.get(key)
        if isinstance(child, dict) and "rle" in child:
            return child
    return None


def fetch_result(
    session: requests.Session,
    endpoint_id: str,
    request_id: str,
    api_key: str,
    timeout: tuple[float, float],
    attempts: int = 3,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    app_id = queue_app_id(endpoint_id)
    for attempt_number in range(1, attempts + 1):
        started = time.monotonic()
        try:
            response = session.get(
                f"{QUEUE_BASE}/{app_id}/requests/{request_id}",
                headers=auth_headers(api_key),
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            history.append(
                {
                    "attempt": attempt_number,
                    "error_type": type(exc).__name__,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
            )
            if attempt_number < attempts:
                time.sleep(min(2 ** (attempt_number - 1), 4))
            continue
        body = safe_json(response)
        history.append(
            {
                "attempt": attempt_number,
                "http_status": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        )
        output = provider_output(body) if body else None
        if 200 <= response.status_code < 300 and output is not None:
            safe_output = {
                key: output[key]
                for key in ("rle", "metadata", "scores", "boxes")
                if key in output
            }
            billable = response.headers.get("X-Fal-Billable-Units")
            if billable is not None:
                safe_output["billable_units"] = billable
            return safe_output, history
        if response.status_code not in RETRYABLE_HTTP:
            error = response_error(response, api_key)
            history[-1]["error_message"] = error
            break
        time.sleep(min(2 ** (attempt_number - 1), 4))
    return None, history


def terminal_api_row(
    submitted: dict[str, Any],
    poll_state: str,
    status_body: dict[str, Any] | None,
    poll_history: list[dict[str, Any]],
    output: dict[str, Any] | None,
    fetch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if poll_state == "pending":
        status = "submitted"
    elif poll_state == "completed" and output is not None:
        status = "ok"
    else:
        status = "terminal_error"
    row = {
        **submitted,
        "status": status,
        "queue_state": poll_state,
        "queue_result": status_body,
        "poll_history": poll_history,
        "fetch_history": fetch_history,
        "updated_at": utc_now(),
    }
    if output is not None:
        row["provider_output"] = output
    return row


def _decode_coco_counts(counts: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    area = shape[0] * shape[1]
    flat = np.zeros(area, dtype=bool)
    position = 0
    foreground = False
    for raw_count in counts:
        count = int(raw_count)
        if count < 0 or position + count > area:
            raise ValueError("invalid COCO RLE run length")
        if foreground and count:
            flat[position : position + count] = True
        position += count
        foreground = not foreground
    if position != area:
        raise ValueError(f"COCO RLE covers {position} pixels, expected {area}")
    return flat.reshape(shape, order="F")


def _decompress_coco_counts(encoded: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        while more:
            if position >= len(encoded):
                raise ValueError("truncated compressed COCO RLE")
            code = ord(encoded[position]) - 48
            position += 1
            if code < 0 or code > 63:
                raise ValueError("invalid compressed COCO RLE character")
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            shift += 1
            if not more and code & 0x10:
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("negative compressed COCO RLE count")
        counts.append(value)
    return counts


def _decode_start_length(tokens: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    if len(tokens) % 2:
        raise ValueError("start-length RLE needs an even number of integers")
    area = shape[0] * shape[1]
    flat = np.zeros(area, dtype=bool)
    starts = [int(value) for value in tokens[0::2]]
    lengths = [int(value) for value in tokens[1::2]]
    zero_based = any(start == 0 for start in starts)
    for start, length in zip(starts, lengths):
        offset = start if zero_based else start - 1
        if offset < 0 or length < 0 or offset + length > area:
            raise ValueError("start-length RLE exceeds mask dimensions")
        flat[offset : offset + length] = True
    # fal's SAM image-rle endpoint emits one-based start/length pairs over a
    # NumPy-style row-major flattening. This is distinct from COCO's
    # alternating counts, which are column-major and handled above.
    return flat.reshape(shape, order="C")


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def decode_rle(value: Any, shape: tuple[int, int]) -> tuple[np.ndarray, str, tuple[int, int]]:
    """Decode fal's observed/simple RLE plus standard COCO RLE variants."""
    if isinstance(value, dict):
        raw_size = value.get("size")
        encoded_shape = (
            (int(raw_size[0]), int(raw_size[1]))
            if isinstance(raw_size, list) and len(raw_size) == 2
            else shape
        )
        counts = value.get("counts", value.get("ucounts"))
        if isinstance(counts, list):
            mask = _decode_coco_counts([int(v) for v in counts], encoded_shape)
            return _resize_mask(mask, shape), "coco_uncompressed", encoded_shape
        if isinstance(counts, str):
            integer_sequence = bool(re.search(r"[\s,;]", counts.strip())) and bool(
                re.fullmatch(r"[\s,;+-]*\d+(?:[\s,;+-]+\d+)*[\s,;]*", counts)
            )
            if integer_sequence:
                tokens = [int(v) for v in re.findall(r"-?\d+", counts)]
                if sum(tokens) == encoded_shape[0] * encoded_shape[1]:
                    mask = _decode_coco_counts(tokens, encoded_shape)
                    kind = "coco_uncompressed_text"
                else:
                    mask = _decode_start_length(tokens, encoded_shape)
                    kind = "start_length_text"
            else:
                mask = _decode_coco_counts(
                    _decompress_coco_counts(counts), encoded_shape
                )
                kind = "coco_compressed"
            return _resize_mask(mask, shape), kind, encoded_shape
        raise ValueError("RLE object has no supported counts field")
    if isinstance(value, list) and all(
        isinstance(child, int) and not isinstance(child, bool) for child in value
    ):
        tokens = [int(child) for child in value]
        if sum(tokens) == shape[0] * shape[1]:
            return _decode_coco_counts(tokens, shape), "coco_uncompressed", shape
        return _decode_start_length(tokens, shape), "start_length", shape
    if not isinstance(value, str):
        raise ValueError(f"unsupported RLE type: {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        return np.zeros(shape, dtype=bool), "empty", shape
    if stripped[0] in "[{":
        try:
            decoded_json = json.loads(stripped)
        except json.JSONDecodeError:
            decoded_json = None
        if decoded_json is not None:
            return decode_rle(decoded_json, shape)
    integer_sequence = bool(re.search(r"[\s,;]", stripped)) and bool(
        re.fullmatch(r"[\s,;+-]*\d+(?:[\s,;+-]+\d+)*[\s,;]*", stripped)
    )
    if integer_sequence:
        tokens = [int(value) for value in re.findall(r"-?\d+", stripped)]
        if sum(tokens) == shape[0] * shape[1]:
            return _decode_coco_counts(tokens, shape), "coco_uncompressed_text", shape
        return _decode_start_length(tokens, shape), "start_length_text", shape
    mask = _decode_coco_counts(_decompress_coco_counts(stripped), shape)
    return mask, "coco_compressed", shape


def rle_entries(raw_rle: Any) -> list[Any]:
    if isinstance(raw_rle, list):
        if all(isinstance(value, int) and not isinstance(value, bool) for value in raw_rle):
            return [raw_rle]
        return list(raw_rle)
    return [raw_rle]


def numeric_list(value: Any) -> list[float | None]:
    if not isinstance(value, list):
        return []
    return [
        float(child)
        if isinstance(child, (int, float)) and not isinstance(child, bool)
        else None
        for child in value
    ]


def mask_candidates(
    provider_result: dict[str, Any],
    shape: tuple[int, int],
    edit_box: Sequence[int],
    residual: np.ndarray,
) -> list[dict[str, Any]]:
    scores = numeric_list(provider_result.get("scores"))
    metadata = provider_result.get("metadata")
    metadata = metadata if isinstance(metadata, list) else []
    boxes = provider_result.get("boxes")
    boxes = boxes if isinstance(boxes, list) else []
    edit = box_mask(shape, edit_box)
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(rle_entries(provider_result.get("rle"))):
        mask, encoding, encoded_shape = decode_rle(entry, shape)
        area = int(mask.sum())
        intersection_edit = int(np.logical_and(mask, edit).sum())
        intersection_residual = int(np.logical_and(mask, residual).sum())
        meta = metadata[index] if index < len(metadata) and isinstance(metadata[index], dict) else {}
        provider_score = scores[index] if index < len(scores) else None
        if provider_score is None and isinstance(meta.get("score"), (int, float)):
            provider_score = float(meta["score"])
        confidence = min(1.0, max(0.0, provider_score or 0.0))
        area_ratio = area / float(mask.size)
        edit_coverage = intersection_edit / max(1, int(edit.sum()))
        edit_precision = intersection_edit / max(1, area)
        residual_precision = intersection_residual / max(1, area)
        plausible = 1.0 if 0.001 <= area_ratio <= 0.7 else 0.0
        # A combined text+box call can return one candidate for each prompt.
        # The box candidate may confidently segment pre-existing furniture in
        # the edit region, while the lower-confidence text candidate is the
        # inserted cat. Source/generated residual agreement therefore carries
        # equal weight to provider confidence.
        rank_score = (
            0.30 * confidence
            + 0.45 * residual_precision
            + 0.10 * min(1.0, edit_coverage)
            + 0.05 * min(1.0, edit_precision)
            + 0.10 * plausible
        )
        if intersection_edit == 0:
            rank_score *= 0.25
        results.append(
            {
                "index": index,
                "mask": mask,
                "encoding": encoding,
                "encoded_shape": list(encoded_shape),
                "provider_score": provider_score,
                "provider_box": boxes[index] if index < len(boxes) else meta.get("box"),
                "area_pixels": area,
                "area_fraction": area_ratio,
                "edit_intersection_pixels": intersection_edit,
                "edit_coverage": edit_coverage,
                "edit_precision": edit_precision,
                "residual_precision": residual_precision,
                "rank_score": rank_score,
            }
        )
    return sorted(results, key=lambda row: (-row["rank_score"], row["index"]))


def hybrid_mask(
    semantic: np.ndarray,
    residual: np.ndarray,
    support_radius: int,
    min_component_pixels: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if not semantic.any():
        return semantic.copy(), np.zeros_like(semantic)
    near = ndimage.binary_dilation(semantic, iterations=max(1, support_radius))
    clip = ndimage.binary_dilation(semantic, iterations=max(2, support_radius * 2))
    labels, count = ndimage.label(residual)
    support = np.zeros_like(semantic)
    max_component = max(int(semantic.sum()) * 4, int(semantic.size * 0.25))
    for label in range(1, count + 1):
        component = labels == label
        size = int(component.sum())
        if (
            size >= min_component_pixels
            and size <= max_component
            and np.logical_and(component, near).any()
        ):
            support |= component & clip
    support &= ~semantic
    return semantic | support, support


def alpha_mask(mask: np.ndarray, feather: float) -> Image.Image:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return image.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else image


def mask_metrics(mask: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    return {
        "pixels": int(mask.sum()),
        "fraction": float(mask.mean()),
        "reference_iou": binary_iou(mask, reference),
    }


def semantic_quality(selected: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if selected["area_fraction"] < 0.002:
        reasons.append("semantic_mask_too_small")
    if selected["area_fraction"] > 0.7:
        reasons.append("semantic_mask_too_large")
    if selected["edit_intersection_pixels"] == 0:
        reasons.append("semantic_mask_not_anchored_in_edit_box")
    if selected["residual_precision"] < 0.5:
        reasons.append("less_than_half_of_semantic_mask_supported_by_source_generated_diff")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "residual_precision_threshold": 0.5,
    }


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def materialize_one(
    item: PilotItem,
    api_row: dict[str, Any],
    output_dir: Path,
    diff_threshold: float,
    support_radius: int,
    feather: float,
) -> tuple[dict[str, Any], dict[str, Image.Image]]:
    endpoint_tag = str(api_row["endpoint_tag"])
    with Image.open(item.source_path) as source_image, Image.open(item.generated_path) as generated_image:
        source = source_image.convert("RGB")
        generated = generated_image.convert("RGB")
    original = source.crop(item.context_box)
    difference = np.abs(
        np.asarray(original, dtype=np.int16) - np.asarray(generated, dtype=np.int16)
    ).max(axis=2)
    residual = difference > diff_threshold
    t30 = current_threshold_mask(original, generated, item.edit_box, 30)
    t40 = current_threshold_mask(original, generated, item.edit_box, 40)
    provider_result = api_row.get("provider_output")
    if not isinstance(provider_result, dict):
        raise ValueError("API row has no provider_output")
    candidates = mask_candidates(
        provider_result,
        (item.crop_size[1], item.crop_size[0]),
        item.edit_box,
        residual,
    )
    if not candidates or not candidates[0]["mask"].any():
        raise ValueError("provider returned no non-empty anchored mask")
    selected = candidates[0]
    quality_gate = semantic_quality(selected)
    semantic = selected["mask"]
    hybrid, support = hybrid_mask(semantic, residual, support_radius)
    semantic_alpha = alpha_mask(semantic, feather)
    hybrid_alpha = alpha_mask(hybrid, feather)
    semantic_crop = Image.composite(generated, original, semantic_alpha)
    hybrid_crop = Image.composite(generated, original, hybrid_alpha)
    full_hybrid = source.copy()
    full_hybrid.paste(hybrid_crop, (item.context_box[0], item.context_box[1]))

    stem = item.task_id
    mask_dir = output_dir / "masks" / endpoint_tag
    context_dir = output_dir / "contexts" / endpoint_tag
    full_dir = output_dir / "spliced_hybrid" / endpoint_tag
    semantic_path = mask_dir / f"{stem}_semantic.png"
    support_path = mask_dir / f"{stem}_residual_support.png"
    hybrid_path = mask_dir / f"{stem}_hybrid.png"
    alpha_path = mask_dir / f"{stem}_hybrid_alpha.png"
    semantic_crop_path = context_dir / f"{stem}_semantic_composite.png"
    hybrid_crop_path = context_dir / f"{stem}_hybrid_composite.png"
    full_path = full_dir / f"{stem}.png"
    save_mask(semantic_path, semantic)
    save_mask(support_path, support)
    save_mask(hybrid_path, hybrid)
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_alpha.save(alpha_path)
    semantic_crop_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_crop.save(semantic_crop_path)
    hybrid_crop.save(hybrid_crop_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_hybrid.save(full_path)

    full_array = np.asarray(full_hybrid)
    source_array = np.asarray(source)
    outside = np.ones(source_array.shape[:2], dtype=bool)
    cx1, cy1, cx2, cy2 = item.context_box
    outside[cy1:cy2, cx1:cx2] = False
    outside_equal = bool(np.array_equal(full_array[outside], source_array[outside]))
    clean_candidates = [
        {key: value for key, value in candidate.items() if key != "mask"}
        for candidate in candidates
    ]
    row = {
        "schema_version": "fal_sam3_splice_result_v1",
        "id": api_row["id"],
        "task_id": item.task_id,
        "domain": item.domain,
        "endpoint_tag": endpoint_tag,
        "endpoint": api_row["endpoint"],
        "request_id": api_row["request_id"],
        "input_image": item.generated_relative,
        "input_sha256": item.input_sha256,
        "prompt": api_row.get("request", {}).get("prompt"),
        "prompt_box_xyxy": list(item.edit_box),
        "selected_candidate_index": selected["index"],
        "selected_provider_score": selected["provider_score"],
        "candidate_rank_score": selected["rank_score"],
        "quality_gate": quality_gate,
        "candidates": clean_candidates,
        "current_t30": mask_metrics(t30, hybrid),
        "current_t40": mask_metrics(t40, hybrid),
        "semantic": {
            **mask_metrics(semantic, t30),
            "path": relative_to_repo(semantic_path),
        },
        "residual_support": {
            "pixels": int(support.sum()),
            "fraction": float(support.mean()),
            "path": relative_to_repo(support_path),
        },
        "hybrid": {
            **mask_metrics(hybrid, t30),
            "growth_over_semantic": float(hybrid.sum() / max(1, semantic.sum()) - 1.0),
            "path": relative_to_repo(hybrid_path),
            "alpha_path": relative_to_repo(alpha_path),
        },
        "semantic_context_composite": relative_to_repo(semantic_crop_path),
        "hybrid_context_composite": relative_to_repo(hybrid_crop_path),
        "hybrid_spliced_full": relative_to_repo(full_path),
        "outside_context_identical_to_source": outside_equal,
        "postprocess": {
            "diff_threshold": diff_threshold,
            "support_radius": support_radius,
            "alpha_feather": feather,
        },
        "status": "ok",
    }
    visuals = {
        "original": original,
        "generated": generated,
        "t30": Image.fromarray(t30.astype(np.uint8) * 255).convert("RGB"),
        "semantic": Image.fromarray(semantic.astype(np.uint8) * 255).convert("RGB"),
        "semantic_composite": semantic_crop,
        "hybrid": Image.fromarray(hybrid.astype(np.uint8) * 255).convert("RGB"),
        "hybrid_composite": hybrid_crop,
    }
    return row, visuals


def fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, (235, 235, 235))
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    tile.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return tile


def make_contact_sheet(
    rows: Sequence[dict[str, Any]],
    visuals: dict[str, dict[str, Image.Image]],
    path: Path,
) -> None:
    if not rows:
        return
    columns = [
        ("original", "source crop"),
        ("generated", "generated"),
        ("t30", "current t30"),
        ("semantic", "SAM mask"),
        ("semantic_composite", "SAM only"),
        ("hybrid", "hybrid mask"),
        ("hybrid_composite", "hybrid splice"),
    ]
    tile_size = (180, 125)
    label_height = 42
    left_width = 290
    header_height = 32
    canvas = Image.new(
        "RGB",
        (
            left_width + tile_size[0] * len(columns),
            header_height + (tile_size[1] + label_height) * len(rows),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column_index, (_, title) in enumerate(columns):
        draw.text(
            (left_width + column_index * tile_size[0] + 5, 9),
            title,
            fill="black",
            font=font,
        )
    for row_index, row in enumerate(rows):
        y = header_height + row_index * (tile_size[1] + label_height)
        score = row.get("selected_provider_score")
        score_text = "n/a" if score is None else f"{score:.3f}"
        label = (
            f"{row['task_id']}\n{row['endpoint_tag']}  score={score_text}\n"
            f"gate={'PASS' if row['quality_gate']['pass'] else 'FAIL'}  "
            f"SAM={row['semantic']['fraction']:.3f} hybrid={row['hybrid']['fraction']:.3f}"
        )
        draw.multiline_text((7, y + 8), label, fill="black", font=font, spacing=3)
        row_visuals = visuals[str(row["id"])]
        for column_index, (key, _) in enumerate(columns):
            tile = fit_tile(row_visuals[key], tile_size)
            canvas.paste(tile, (left_width + column_index * tile_size[0], y))
        draw.line(
            [(0, y + tile_size[1] + label_height - 1), (canvas.width, y + tile_size[1] + label_height - 1)],
            fill=(190, 190, 190),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92, optimize=True)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(clean) if clean else None


def summarize_materialized(
    rows: Sequence[dict[str, Any]],
    endpoint_tags: Sequence[str],
    expected_tasks: int,
    api_latest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_endpoint[row["endpoint_tag"]].append(row)
        by_task[row["task_id"]][row["endpoint_tag"]] = row
    endpoint_summary: dict[str, Any] = {}
    for tag in endpoint_tags:
        group = by_endpoint[tag]
        endpoint_summary[tag] = {
            "endpoint": ENDPOINTS[tag],
            "completed": len(group),
            "mean_provider_score": mean_or_none(
                row.get("selected_provider_score") for row in group
            ),
            "mean_semantic_area_fraction": mean_or_none(
                row["semantic"]["fraction"] for row in group
            ),
            "mean_hybrid_area_fraction": mean_or_none(
                row["hybrid"]["fraction"] for row in group
            ),
            "mean_hybrid_growth": mean_or_none(
                row["hybrid"]["growth_over_semantic"] for row in group
            ),
            "mean_hybrid_iou_with_current_t30": mean_or_none(
                row["hybrid"]["reference_iou"] for row in group
            ),
            "outside_context_invariance_failures": sum(
                not row["outside_context_identical_to_source"] for row in group
            ),
            "quality_gate_failures": sum(
                not row.get("quality_gate", {}).get("pass", False) for row in group
            ),
            "quality_gate_failure_ids": [
                row["id"]
                for row in group
                if not row.get("quality_gate", {}).get("pass", False)
            ],
        }
    pair_semantic: list[float] = []
    pair_hybrid: list[float] = []
    if len(endpoint_tags) == 2:
        for task_rows in by_task.values():
            if all(tag in task_rows for tag in endpoint_tags):
                first, second = (task_rows[tag] for tag in endpoint_tags)
                first_sem = np.asarray(Image.open(REPO / first["semantic"]["path"])) >= 128
                second_sem = np.asarray(Image.open(REPO / second["semantic"]["path"])) >= 128
                first_hyb = np.asarray(Image.open(REPO / first["hybrid"]["path"])) >= 128
                second_hyb = np.asarray(Image.open(REPO / second["hybrid"]["path"])) >= 128
                pair_semantic.append(binary_iou(first_sem, second_sem))
                pair_hybrid.append(binary_iou(first_hyb, second_hyb))
    expected_requests = expected_tasks * len(endpoint_tags)
    api_status_counts = Counter(row.get("status") for row in api_latest.values())
    return {
        "schema_version": "fal_sam3_splice_summary_v1",
        "updated_at": utc_now(),
        "expected_tasks": expected_tasks,
        "expected_requests": expected_requests,
        "api_rows_latest": len(api_latest),
        "api_status_counts": dict(sorted(api_status_counts.items())),
        "materialized_results": len(rows),
        "estimated_cost_if_all_requested_usd": sum(
            ENDPOINT_COST_USD[tag] * expected_tasks for tag in endpoint_tags
        ),
        "by_endpoint": endpoint_summary,
        "endpoint_pair_agreement": {
            "paired_tasks": len(pair_semantic),
            "mean_semantic_iou": mean_or_none(pair_semantic),
            "mean_hybrid_iou": mean_or_none(pair_hybrid),
        },
    }


def materialize_all(
    items: Sequence[PilotItem],
    output_dir: Path,
    endpoint_tags: Sequence[str],
    diff_threshold: float,
    support_radius: int,
    feather: float,
) -> dict[str, Any]:
    api_path = output_dir / "api_results.jsonl"
    latest = read_latest(api_path)
    item_lookup = {item.task_id: item for item in items}
    rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Image.Image]] = {}
    errors: list[dict[str, Any]] = []
    ordered_ids = [
        f"{item.task_id}__{tag}" for item in items for tag in endpoint_tags
    ]
    for row_id in ordered_ids:
        api_row = latest.get(row_id)
        if not api_row or api_row.get("status") != "ok":
            continue
        item = item_lookup[str(api_row["task_id"])]
        try:
            row, row_visuals = materialize_one(
                item,
                api_row,
                output_dir,
                diff_threshold,
                support_radius,
                feather,
            )
        except Exception as exc:
            errors.append(
                {
                    "id": row_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        rows.append(row)
        visuals[row_id] = row_visuals
    write_jsonl(output_dir / "splice_results.jsonl", rows)
    make_contact_sheet(rows, visuals, output_dir / "contact_sheet.jpg")
    summary = summarize_materialized(rows, endpoint_tags, len(items), latest)
    summary["materialization_errors"] = errors
    write_json(output_dir / "summary.json", summary)
    return summary


def load_visuals_from_row(
    row: dict[str, Any], item: PilotItem
) -> dict[str, Image.Image]:
    with Image.open(item.source_path) as source_image, Image.open(item.generated_path) as generated_image:
        original = source_image.convert("RGB").crop(item.context_box)
        generated = generated_image.convert("RGB")
    t30 = current_threshold_mask(original, generated, item.edit_box, 30)
    return {
        "original": original,
        "generated": generated,
        "t30": Image.fromarray(t30.astype(np.uint8) * 255).convert("RGB"),
        "semantic": Image.open(REPO / row["semantic"]["path"]).convert("RGB"),
        "semantic_composite": Image.open(
            REPO / row["semantic_context_composite"]
        ).convert("RGB"),
        "hybrid": Image.open(REPO / row["hybrid"]["path"]).convert("RGB"),
        "hybrid_composite": Image.open(
            REPO / row["hybrid_context_composite"]
        ).convert("RGB"),
    }


def select_quality_fallbacks(
    items: Sequence[PilotItem],
    primary_dir: Path,
    fallback_dir: Path,
    endpoint_tags: Sequence[str],
) -> dict[str, Any]:
    primary_rows = {
        row["id"]: row for row in load_jsonl(primary_dir / "splice_results.jsonl")
    }
    fallback_rows = {
        row["id"]: row for row in load_jsonl(fallback_dir / "splice_results.jsonl")
    }
    item_lookup = {item.task_id: item for item in items}
    selected_rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Image.Image]] = {}
    unresolved: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for item in items:
        for endpoint_tag in endpoint_tags:
            row_id = f"{item.task_id}__{endpoint_tag}"
            primary = primary_rows.get(row_id)
            if primary is None:
                unresolved.append({"id": row_id, "reason": "missing_primary_result"})
                continue
            primary_pass = bool(primary.get("quality_gate", {}).get("pass"))
            fallback = fallback_rows.get(row_id)
            fallback_pass = bool(
                fallback and fallback.get("quality_gate", {}).get("pass")
            )
            if primary_pass:
                selected = dict(primary)
                source = "primary_text_box"
            elif fallback_pass:
                selected = dict(fallback)
                source = "text_only_quality_fallback"
            else:
                unresolved.append(
                    {
                        "id": row_id,
                        "reason": "primary_failed_and_no_passing_fallback",
                        "primary_quality_gate": primary.get("quality_gate"),
                        "fallback_quality_gate": (
                            fallback.get("quality_gate") if fallback else None
                        ),
                    }
                )
                continue
            selected["selection_source"] = source
            selected["primary_quality_gate"] = primary.get("quality_gate")
            selected["selected_from_result_dir"] = relative_to_repo(
                primary_dir if source == "primary_text_box" else fallback_dir
            )
            selected_rows.append(selected)
            source_counts[source] += 1
            visuals[row_id] = load_visuals_from_row(selected, item_lookup[item.task_id])

    write_jsonl(primary_dir / "selected_splice_results.jsonl", selected_rows)
    make_contact_sheet(
        selected_rows,
        visuals,
        primary_dir / "selected_contact_sheet.jpg",
    )
    api_latest = read_latest(primary_dir / "api_results.jsonl")
    summary = summarize_materialized(
        selected_rows,
        endpoint_tags,
        len(items),
        api_latest,
    )
    primary_summary = json.loads((primary_dir / "summary.json").read_text(encoding="utf-8"))
    fallback_summary = json.loads((fallback_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": "fal_sam3_selected_splice_summary_v1",
            "selection_sources": dict(sorted(source_counts.items())),
            "primary_result_dir": relative_to_repo(primary_dir),
            "fallback_result_dir": relative_to_repo(fallback_dir),
            "unresolved": unresolved,
            "all_selected_quality_gates_pass": not unresolved
            and len(selected_rows) == len(items) * len(endpoint_tags),
            "total_provider_requests": int(primary_summary["expected_requests"])
            + int(fallback_summary["expected_requests"]),
            "total_estimated_cost_usd": float(
                primary_summary["estimated_cost_if_all_requested_usd"]
            )
            + float(fallback_summary["estimated_cost_if_all_requested_usd"]),
        }
    )
    write_json(primary_dir / "selected_summary.json", summary)
    return summary


def row_mask_iou(
    left: dict[str, Any], right: dict[str, Any], mask_name: str
) -> float:
    left_mask = np.asarray(Image.open(REPO / left[mask_name]["path"])) >= 128
    right_mask = np.asarray(Image.open(REPO / right[mask_name]["path"])) >= 128
    return binary_iou(left_mask, right_mask)


def assemble_text_only_shards(
    items: Sequence[PilotItem],
    primary_dir: Path,
    shard_dirs: Sequence[Path],
    endpoint_tag: str = "sam3",
) -> dict[str, Any]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    api_latest: dict[str, dict[str, Any]] = {}
    row_sources: dict[str, str] = {}
    for shard_dir in shard_dirs:
        for row in load_jsonl(shard_dir / "splice_results.jsonl"):
            if row.get("endpoint_tag") != endpoint_tag:
                continue
            row_id = str(row["id"])
            if row_id in rows_by_id:
                raise ValueError(f"duplicate text-only result across shards: {row_id}")
            rows_by_id[row_id] = row
            row_sources[row_id] = relative_to_repo(shard_dir)
        for row_id, row in read_latest(shard_dir / "api_results.jsonl").items():
            if row.get("endpoint_tag") == endpoint_tag:
                if row_id in api_latest:
                    raise ValueError(f"duplicate text-only API row across shards: {row_id}")
                api_latest[row_id] = row

    rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Image.Image]] = {}
    missing: list[str] = []
    for item in items:
        row_id = f"{item.task_id}__{endpoint_tag}"
        raw_row = rows_by_id.get(row_id)
        if raw_row is None:
            missing.append(row_id)
            continue
        row = dict(raw_row)
        row["selection_source"] = "text_only"
        row["selected_from_result_dir"] = row_sources[row_id]
        rows.append(row)
        visuals[row_id] = load_visuals_from_row(row, item)

    write_jsonl(primary_dir / "text_only_sam3_full10_results.jsonl", rows)
    make_contact_sheet(
        rows,
        visuals,
        primary_dir / "text_only_sam3_full10_contact_sheet.jpg",
    )
    summary = summarize_materialized(rows, [endpoint_tag], len(items), api_latest)
    primary_rows = {
        row["id"]: row for row in load_jsonl(primary_dir / "splice_results.jsonl")
    }
    selected_rows = {
        row["id"]: row
        for row in load_jsonl(primary_dir / "selected_splice_results.jsonl")
    }
    primary_semantic_iou: list[float] = []
    primary_hybrid_iou: list[float] = []
    passing_primary_semantic_iou: list[float] = []
    passing_primary_hybrid_iou: list[float] = []
    selected_semantic_iou: list[float] = []
    selected_hybrid_iou: list[float] = []
    for row in rows:
        row_id = row["id"]
        primary = primary_rows.get(row_id)
        if primary is not None:
            semantic_iou = row_mask_iou(row, primary, "semantic")
            hybrid_iou = row_mask_iou(row, primary, "hybrid")
            primary_semantic_iou.append(semantic_iou)
            primary_hybrid_iou.append(hybrid_iou)
            if primary.get("quality_gate", {}).get("pass"):
                passing_primary_semantic_iou.append(semantic_iou)
                passing_primary_hybrid_iou.append(hybrid_iou)
        selected = selected_rows.get(row_id)
        if selected is not None:
            selected_semantic_iou.append(row_mask_iou(row, selected, "semantic"))
            selected_hybrid_iou.append(row_mask_iou(row, selected, "hybrid"))
    summary.update(
        {
            "schema_version": "fal_sam3_text_only_full10_summary_v1",
            "source_result_dirs": [relative_to_repo(path) for path in shard_dirs],
            "missing": missing,
            "all_quality_gates_pass": len(rows) == len(items)
            and not missing
            and all(row.get("quality_gate", {}).get("pass") for row in rows),
            "estimated_text_only_cost_usd": len(items) * ENDPOINT_COST_USD[endpoint_tag],
            "comparison_to_text_box_primary": {
                "all_tasks_mean_semantic_iou": mean_or_none(primary_semantic_iou),
                "all_tasks_mean_hybrid_iou": mean_or_none(primary_hybrid_iou),
                "passing_primary_tasks": len(passing_primary_semantic_iou),
                "passing_primary_mean_semantic_iou": mean_or_none(
                    passing_primary_semantic_iou
                ),
                "passing_primary_mean_hybrid_iou": mean_or_none(
                    passing_primary_hybrid_iou
                ),
            },
            "comparison_to_quality_selected_output": {
                "paired_tasks": len(selected_semantic_iou),
                "mean_semantic_iou": mean_or_none(selected_semantic_iou),
                "mean_hybrid_iou": mean_or_none(selected_hybrid_iou),
            },
        }
    )
    write_json(primary_dir / "text_only_sam3_full10_summary.json", summary)
    return summary


def parse_endpoint_tags(raw: str) -> list[str]:
    tags = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [tag for tag in tags if tag not in ENDPOINTS]
    if not tags or unknown or len(tags) != len(set(tags)):
        raise ValueError(f"invalid endpoint tags: {raw}; choices: {sorted(ENDPOINTS)}")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--endpoints", default="sam3,sam3_1")
    parser.add_argument("--prompt", default="cat")
    parser.add_argument(
        "--prompt-mode",
        choices=("text_box", "text_only"),
        default="text_box",
        help="text_box is the primary run; text_only is useful as a quality-gated fallback",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="comma-separated explicit task IDs; overrides the stratified --tasks selection",
    )
    parser.add_argument("--max-masks", type=int, default=3)
    parser.add_argument("--diff-threshold", type=float, default=20.0)
    parser.add_argument("--support-radius", type=int, default=6)
    parser.add_argument("--feather", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-poll-seconds", type=float, default=300.0)
    parser.add_argument(
        "--new-submission-limit",
        type=int,
        default=0,
        help="0 means unlimited; resumptions do not count toward the limit",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="do no network work; rebuild masks/composites from saved API rows",
    )
    parser.add_argument(
        "--fallback-dir",
        type=Path,
        default=None,
        help="select passing text-only rows for primary quality-gate failures",
    )
    parser.add_argument(
        "--text-only-shard-dirs",
        default="",
        help="comma-separated result dirs to assemble a full SAM 3 text-only run",
    )
    args = parser.parse_args()
    if (
        args.tasks < 1
        or not 1 <= args.max_masks <= 32
        or args.diff_threshold < 0
        or args.support_radius < 1
        or args.feather < 0
        or args.poll_interval < 0
        or args.max_poll_seconds <= 0
        or args.new_submission_limit < 0
    ):
        parser.error("invalid numeric argument")
    try:
        endpoint_tags = parse_endpoint_tags(args.endpoints)
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = args.repo_root.resolve()
    base_manifest = (
        args.base_manifest
        if args.base_manifest.is_absolute()
        else repo_root / args.base_manifest
    ).resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    ).resolve()
    fallback_dir = (
        None
        if args.fallback_dir is None
        else (
            args.fallback_dir
            if args.fallback_dir.is_absolute()
            else repo_root / args.fallback_dir
        ).resolve()
    )
    text_only_shard_dirs = [
        (
            Path(value.strip())
            if Path(value.strip()).is_absolute()
            else repo_root / value.strip()
        ).resolve()
        for value in args.text_only_shard_dirs.split(",")
        if value.strip()
    ]
    candidates = load_pilot_candidates(repo_root, base_manifest)
    explicit_ids = [value.strip() for value in args.task_ids.split(",") if value.strip()]
    if explicit_ids:
        if len(explicit_ids) != len(set(explicit_ids)):
            parser.error("--task-ids contains duplicates")
        candidate_lookup = {item.task_id: item for item in candidates}
        missing = [task_id for task_id in explicit_ids if task_id not in candidate_lookup]
        if missing:
            parser.error(f"unknown --task-ids: {missing}")
        items = [
            replace(candidate_lookup[task_id], selection_reason="explicit_quality_fallback")
            for task_id in explicit_ids
        ]
    else:
        items = select_pilot_items(candidates, args.tasks)
    use_box_prompt = args.prompt_mode == "text_box"
    selected_summary = {
        "selected_tasks": len(items),
        "domains": dict(sorted(Counter(item.domain for item in items).items())),
        "endpoints": endpoint_tags,
        "prompt_mode": args.prompt_mode,
        "expected_requests": len(items) * len(endpoint_tags),
        "estimated_cost_usd": sum(
            ENDPOINT_COST_USD[tag] * len(items) for tag in endpoint_tags
        ),
        "selection": [
            {
                "task_id": item.task_id,
                "domain": item.domain,
                "edit_area_fraction": round(item.edit_area_fraction, 6),
                "t30_t40_disagreement": round(item.threshold_disagreement, 6),
                "reason": item.selection_reason,
            }
            for item in items
        ],
        "output_dir": output_dir.as_posix(),
        "dry_run": args.dry_run,
    }
    print(json.dumps(selected_summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return

    ensure_run_manifest(
        output_dir,
        base_manifest,
        items,
        endpoint_tags,
        args.prompt,
        args.max_masks,
        args.diff_threshold,
        args.support_radius,
        args.feather,
        use_box_prompt,
    )
    if args.materialize_only:
        summary = materialize_all(
            items,
            output_dir,
            endpoint_tags,
            args.diff_threshold,
            args.support_radius,
            args.feather,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        if fallback_dir is not None:
            selected = select_quality_fallbacks(
                items,
                output_dir,
                fallback_dir,
                endpoint_tags,
            )
            print(json.dumps(selected, ensure_ascii=False, sort_keys=True), flush=True)
        if text_only_shard_dirs:
            assembled = assemble_text_only_shards(
                items,
                output_dir,
                text_only_shard_dirs,
            )
            print(json.dumps(assembled, ensure_ascii=False, sort_keys=True), flush=True)
        return

    api_key = os.environ.get("FAL_KEY", "")
    if not api_key:
        raise SystemExit("FAL_KEY must be set")
    api_path = output_dir / "api_results.jsonl"
    latest = read_latest(api_path)
    selection_sha = selection_digest(items)
    session = requests.Session()
    session.headers.update({"User-Agent": "claimforge-benchmark/fal-sam3-splice-pilot-v1"})
    new_submissions = 0
    stop = False
    for item in items:
        for endpoint_tag in endpoint_tags:
            row_id = f"{item.task_id}__{endpoint_tag}"
            prior = latest.get(row_id)
            recoverable_legacy_route_error = bool(
                prior
                and prior.get("status") == "terminal_error"
                and isinstance(prior.get("request_id"), str)
                and isinstance(prior.get("queue_result"), dict)
                and prior["queue_result"].get("http_status") == 405
            )
            recoverable_result_fetch_error = bool(
                prior
                and prior.get("status") == "terminal_error"
                and prior.get("queue_state") == "completed"
                and isinstance(prior.get("request_id"), str)
                and "provider_output" not in prior
            )
            if prior and prior.get("status") == "ok":
                continue
            if (
                prior
                and prior.get("status") == "terminal_error"
                and not recoverable_legacy_route_error
                and not recoverable_result_fetch_error
            ):
                continue
            if prior and isinstance(prior.get("request_id"), str):
                submitted = prior
                action = "resume"
            else:
                if prior:
                    print(
                        json.dumps(
                            {
                                "id": row_id,
                                "status": "not_retried",
                                "reason": "previous submission has no request_id",
                            }
                        ),
                        flush=True,
                    )
                    stop = True
                    break
                if args.new_submission_limit and new_submissions >= args.new_submission_limit:
                    stop = True
                    break
                submitted = submit_request(
                    session,
                    item,
                    endpoint_tag,
                    api_key,
                    args.prompt,
                    args.max_masks,
                    (args.connect_timeout, args.read_timeout),
                    selection_sha,
                    use_box_prompt,
                )
                append_jsonl(api_path, submitted)
                latest[row_id] = submitted
                new_submissions += 1
                action = "submit"
                if submitted.get("status") != "submitted":
                    print(
                        json.dumps(
                            {
                                "id": row_id,
                                "action": action,
                                "status": submitted.get("status"),
                                "http_status": submitted.get("http_status"),
                            }
                        ),
                        flush=True,
                    )
                    stop = True
                    break
            poll_state, status_body, poll_history = poll_request(
                session,
                str(submitted["endpoint"]),
                str(submitted["request_id"]),
                api_key,
                (args.connect_timeout, args.read_timeout),
                args.poll_interval,
                args.max_poll_seconds,
            )
            output: dict[str, Any] | None = None
            fetch_history: list[dict[str, Any]] = []
            if poll_state == "completed":
                output, fetch_history = fetch_result(
                    session,
                    str(submitted["endpoint"]),
                    str(submitted["request_id"]),
                    api_key,
                    (args.connect_timeout, args.read_timeout),
                )
            terminal = terminal_api_row(
                submitted,
                poll_state,
                status_body,
                poll_history,
                output,
                fetch_history,
            )
            append_jsonl(api_path, terminal)
            latest[row_id] = terminal
            print(
                json.dumps(
                    {
                        "id": row_id,
                        "action": action,
                        "status": terminal["status"],
                        "queue_state": poll_state,
                        "rle_type": (
                            type(output.get("rle")).__name__ if output else None
                        ),
                    }
                ),
                flush=True,
            )
            if terminal["status"] == "terminal_error":
                stop = True
                break
        if stop:
            break

    summary = materialize_all(
        items,
        output_dir,
        endpoint_tags,
        args.diff_threshold,
        args.support_radius,
        args.feather,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if fallback_dir is not None:
        selected = select_quality_fallbacks(
            items,
            output_dir,
            fallback_dir,
            endpoint_tags,
        )
        print(json.dumps(selected, ensure_ascii=False, sort_keys=True), flush=True)
    if text_only_shard_dirs:
        assembled = assemble_text_only_shards(
            items,
            output_dir,
            text_only_shard_dirs,
        )
        print(json.dumps(assembled, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
