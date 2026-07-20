#!/usr/bin/env python3
"""Run Alibaba Cloud Ultra on reviewed CLAIMFORGE mouse images.

Credentials are read only from ``ALIBABA_CLOUD_ACCESS_KEY_ID`` and
``ALIBABA_CLOUD_ACCESS_KEY_SECRET``. Images are converted to metadata-free
JPEGs, uploaded with Alibaba's temporary Content Moderation OSS token, and
submitted to ``aigcDetector_ultra`` in China (Beijing).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import oss2
from alibabacloud_green20220302 import models
from alibabacloud_green20220302.client import Client
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util import models as util_models

from eval.commercial.run_hive import canonicalize, load_selected_items
from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    input_digest,
    read_latest,
    redact_payload,
    redact_text,
    sha256_file,
    utc_now,
)


DEFAULT_ENDPOINT = "green-cip.cn-beijing.aliyuncs.com"
DEFAULT_REGION = "cn-beijing"
DEFAULT_SERVICE = "aigcDetector_ultra"
DEFAULT_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_ORDER_MANIFEST = Path(
    "results/commercial/sightengine/"
    "pilot_good275_mouse_forged_original_png_20260720.run_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/alibaba/"
    "preflight_good_mouse_forged1_canonical_jpeg_q95_20260720.jsonl"
)
RISK_LABELS = ("risk_aigc", "risk_fake", "risk_edit")


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class TemporaryUploader:
    """Upload local files with the short-lived token issued by Green."""

    def __init__(self, client: Client, endpoint: str) -> None:
        self.client = client
        self.endpoint = endpoint
        self.token: Any = None
        self.bucket: oss2.Bucket | None = None

    def _refresh(self) -> None:
        response = self.client.describe_upload_token()
        body = response.body
        if response.status_code != 200 or body is None or body.code != 200:
            code = getattr(body, "code", None)
            message = getattr(body, "msg", None)
            raise RuntimeError(f"upload token failed: http={response.status_code} code={code} msg={message}")
        token = body.data
        if token is None:
            raise RuntimeError("upload token response has no data")
        auth = oss2.StsAuth(
            token.access_key_id,
            token.access_key_secret,
            token.security_token,
        )
        self.token = token
        self.bucket = oss2.Bucket(auth, token.oss_internet_end_point, token.bucket_name)

    def upload(self, path: Path) -> dict[str, str]:
        expiration = int(getattr(self.token, "expiration", 0) or 0)
        if self.token is None or self.bucket is None or expiration <= int(time.time()) + 60:
            self._refresh()
        extension = path.suffix.lower() or ".jpg"
        object_name = f"{self.token.file_name_prefix}{uuid.uuid4()}{extension}"
        self.bucket.put_object_from_file(object_name, path.as_posix())
        return {
            "ossBucketName": self.token.bucket_name,
            "ossObjectName": object_name,
        }


def parse_response(response: Any) -> dict[str, Any]:
    body = response.body
    if response.status_code != 200 or body is None:
        raise RuntimeError(f"unexpected HTTP status: {response.status_code}")
    if body.code != 200:
        raise RuntimeError(f"provider code={body.code} msg={body.msg}")
    data = body.data
    if data is None:
        raise RuntimeError("provider returned no result data")

    labels: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for result in data.result or []:
        label = getattr(result, "label", None)
        confidence = as_number(getattr(result, "confidence", None))
        if not isinstance(label, str):
            continue
        labels.append({"label": label, "confidence": confidence})
        if confidence is not None:
            scores[label] = confidence

    return {
        "request_id": body.request_id,
        "provider_code": body.code,
        "provider_message": body.msg,
        "provider_data_id": data.data_id,
        "provider_risk_level": data.risk_level,
        "provider_labels": labels,
        "label_scores": scores,
        "risk_aigc_confidence": scores.get("risk_aigc"),
        "risk_fake_confidence": scores.get("risk_fake"),
        "risk_edit_confidence": scores.get("risk_edit"),
        "risk_aigc_detected": "risk_aigc" in scores,
        "risk_fake_detected": "risk_fake" in scores,
        "risk_edit_detected": "risk_edit" in scores,
        "non_label": any(row["label"] == "nonLabel" for row in labels),
    }


def classify(
    client: Client,
    uploader: TemporaryUploader,
    item: ImageItem,
    upload_path: Path,
    upload_metadata: dict[str, Any],
    service: str,
    run_id: str,
    manifest_sha256: str,
    max_attempts: int,
    access_key_id: str,
    access_key_secret: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    service_parameters: dict[str, str] | None = None
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            if service_parameters is None:
                service_parameters = uploader.upload(upload_path)
                service_parameters["dataId"] = item.id
            request = models.ImageModerationRequest(
                service=service,
                service_parameters=json.dumps(service_parameters),
            )
            response = client.image_moderation_with_options(
                request,
                util_models.RuntimeOptions(
                    connect_timeout=15_000,
                    read_timeout=180_000,
                    autoretry=False,
                ),
            )
            parsed = parse_response(response)
            return {
                "schema_version": "alibaba_ultra_result_v1",
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
                **upload_metadata,
                "status": "ok",
                "latency_ms": round((time.monotonic() - started) * 1000),
                "attempt_count": attempt,
                **parsed,
                "completed_at": utc_now(),
            }
        except Exception as exc:
            message = redact_text(str(exc), access_key_secret)
            message = redact_text(message, access_key_id)
            data = getattr(exc, "data", None)
            attempts.append(
                {
                    "attempt": attempt,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "error_message": message[:1500],
                    "provider_data": redact_payload(data),
                }
            )
            if attempt < max_attempts:
                time.sleep(min(8.0, 2 ** (attempt - 1)))

    return {
        "schema_version": "alibaba_ultra_result_v1",
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
        **upload_metadata,
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
    region: str,
    service: str,
    quality: int,
    run_id: str,
    include: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    expected = {
        "schema_version": "alibaba_ultra_run_manifest_v1",
        "run_id": run_id,
        "endpoint": endpoint,
        "region": region,
        "service": service,
        "candidate": "mouse",
        "include": "paired_real_and_forged" if include == "both" else "forged",
        "expected_images": len(items),
        "input_manifest_sha256": manifest_sha256,
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "metadata": "stripped",
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
        "sdk_version": importlib.metadata.version("alibabacloud_green20220302"),
        "oss2_version": getattr(oss2, "__version__", None),
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
    summary = {
        "schema_version": "alibaba_ultra_summary_v1",
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "input_manifest_sha256": manifest_sha256,
        "include": include,
        "expected_tasks": len({item.task_id for item in items}),
        "expected_images": len(items),
        "completed_images": len(rows),
        "valid_images": len(ok_rows),
        "error_images": len(rows) - len(ok_rows),
        "detected_by_kind": {
            kind: {
                label: sum(
                    bool(row.get(f"{label}_detected"))
                    for row in ok_rows
                    if row.get("kind") == kind
                )
                for label in RISK_LABELS
            }
            for kind in ("real", "forged")
        },
        "confidence_by_label": {
            label: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for label in RISK_LABELS
            for values in [
                [
                    float(row[f"{label}_confidence"])
                    for row in ok_rows
                    if as_number(row.get(f"{label}_confidence")) is not None
                ]
            ]
        },
        "paired_risk_edit_outcomes": dict(
            Counter(
                f"real={bool(pair['real'].get('risk_edit_detected'))},"
                f"forged={bool(pair['forged'].get('risk_edit_detected'))}"
                for pair in by_task.values()
                if "real" in pair and "forged" in pair
            )
        ),
        "error_types": dict(
            Counter(
                str((row.get("attempts") or [{}])[-1].get("error_type"))
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
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--order-manifest", type=Path, default=DEFAULT_ORDER_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--include", choices=("forged", "both"), default="forged")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--run-id", default="alibaba_ultra_mouse_preflight1_20260720")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tasks < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--tasks must be positive and JPEG quality must be in [1, 100]")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")

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
                "service": args.service,
                "output": output_path.as_posix(),
                "dry_run": args.dry_run,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return

    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
    if not access_key_id or not access_key_secret:
        raise SystemExit(
            "ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET must be set"
        )
    ensure_run_manifest(
        output_path,
        items,
        manifest_sha256,
        args.endpoint,
        args.region,
        args.service,
        args.jpeg_quality,
        args.run_id,
        args.include,
    )
    client = Client(
        Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=args.region,
            endpoint=args.endpoint,
            connect_timeout=15_000,
            read_timeout=180_000,
        )
    )
    uploader = TemporaryUploader(client, args.endpoint)
    with tempfile.TemporaryDirectory(prefix="claimforge-alibaba-") as temporary:
        temporary_dir = Path(temporary)
        for index, item in enumerate(pending):
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload_metadata = canonicalize(item.path, upload_path, args.jpeg_quality)
            row = classify(
                client,
                uploader,
                item,
                upload_path,
                upload_metadata,
                args.service,
                args.run_id,
                manifest_sha256,
                args.max_attempts,
                access_key_id,
                access_key_secret,
            )
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "risk_level": row.get("provider_risk_level"),
                        "labels": row.get("provider_labels"),
                        "error": (row.get("attempts") or [{}])[-1].get("error_message")
                        if row["status"] == "error"
                        else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["status"] == "error":
                break
            if index + 1 < len(pending):
                time.sleep(args.minimum_interval)

    summary = write_summary(output_path, items, manifest_sha256, args.include)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
