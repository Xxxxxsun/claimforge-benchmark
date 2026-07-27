"""Run frozen CLAIMFORGE MLLM protocols without exposing labels to the model."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .client import RetryableError, VisionClient, retry_delay
from .config import load_config
from .inputs import ImageItem, from_jsonl, from_review_export, manifest_hash
from .masks import boxes_to_1000, boxes_to_pixels, write_union_mask
from .prompts import (
    BBOX1000_PROTOCOL_SUITE_VERSION,
    LOCALIZATION_BBOX1000_PROMPT,
    LOCALIZATION_BBOX1000_PROTOCOL_VERSION,
    PROMPTS,
    PROTOCOL_SUITE_VERSION,
    PROTOCOL_VERSIONS,
    REPAIR_SUFFIX,
    SYSTEM_PROMPT,
)
from .results import append_jsonl, completed_aggregate_keys, completed_raw_keys, successful_raw
from .schema import SchemaError, aggregate, parse

PROTOCOL_IDS = {"detection": "mllm_detection_v1", "localization": "mllm_localization_v1"}


def _size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF: i += 1; continue
            marker = data[i+1]; i += 2
            if marker in {0xD8, 0xD9}: continue
            length = int.from_bytes(data[i:i+2], "big")
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                return int.from_bytes(data[i+5:i+7], "big"), int.from_bytes(data[i+3:i+5], "big")
            i += length
    raise ValueError(f"unsupported image dimensions: {path}")


def _raw_id(run_id: str, item: ImageItem, protocol: str, replicate: int) -> str:
    return f"{run_id}:{item.id}:{protocol}:{replicate}"


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    return value


def _protocol_versions_for_model(
    model: dict[str, Any],
    protocols: list[str],
) -> tuple[dict[str, str], str | None, str]:
    coordinate_space = model.get("localizationCoordinateSpace", "bbox_px")
    versions = {key: PROTOCOL_VERSIONS[key] for key in protocols}
    if "localization" in protocols and coordinate_space == "bbox_1000":
        versions["localization"] = LOCALIZATION_BBOX1000_PROTOCOL_VERSION
    suite_version = None
    if len(protocols) > 1:
        suite_version = (
            BBOX1000_PROTOCOL_SUITE_VERSION
            if coordinate_space == "bbox_1000"
            else PROTOCOL_SUITE_VERSION
        )
    return versions, suite_version, coordinate_space


def _protocol_prompt(
    protocol: str,
    image_size: tuple[int, int] | None = None,
    coordinate_space: str = "bbox_px",
) -> str:
    prompt = (
        LOCALIZATION_BBOX1000_PROMPT
        if protocol == "localization" and coordinate_space == "bbox_1000"
        else PROMPTS[protocol]
    )
    if protocol != "localization":
        return prompt
    if image_size is None:
        raise ValueError("localization requires image dimensions")
    width, height = image_size
    if coordinate_space == "bbox_1000":
        return prompt + f"""

Image coordinate metadata (not an annotation): the original image is {width} pixels wide by {height} pixels high.
Return every region as bbox_1000 in the normalized full-image coordinate system. Use top-left (0, 0), bottom-right (1000, 1000), and require 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. Do not return bbox_px. The evaluator will convert bbox_1000 to original-image pixels before validation and aggregation.
"""
    if coordinate_space != "bbox_px":
        raise ValueError(f"unsupported localization coordinate space: {coordinate_space}")
    return prompt + f"""

Image coordinate metadata (not an annotation): this image is exactly {width} pixels wide by {height} pixels high.
Use the top-left corner as (0, 0). Return every bbox_px directly in this original full-image pixel coordinate system as [x1, y1, x2, y2]. Each coordinate must be an integer and must satisfy 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}. Do not normalize coordinates, do not scale them to 0-1000, and do not return bbox_1000. If no valid region can be supported by visible evidence, return no_localized_edit with an empty regions array.
"""


def _schema_repair_prompt(
    base_prompt: str,
    protocol: str,
    error: SchemaError,
    image_size: tuple[int, int] | None,
    coordinate_space: str = "bbox_px",
) -> str:
    details = f"\n\nYour previous response was invalid: {error}."
    if protocol == "localization" and image_size is not None:
        width, height = image_size
        if coordinate_space == "bbox_1000":
            details += f"""
Correct the JSON using bbox_1000 in the normalized full-image coordinate system for the original {width} x {height} image.
Every bbox_1000 must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
Do not return bbox_px. If you cannot support a valid normalized box, return
decision "no_localized_edit" with an empty regions array."""
        else:
            details += f"""
Correct the JSON using the original {width} x {height} pixel image coordinate system.
Every bbox_px must satisfy 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}.
Do not use normalized 0-1000 coordinates. If you cannot support a valid pixel box, return
decision "no_localized_edit" with an empty regions array."""
    return base_prompt + details + REPAIR_SUFFIX


def _load_signed_url_map(path: Path) -> tuple[dict[str, str], str]:
    """Load a local-only relative-path -> signed HTTP(S) URL map without logging it."""
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    mapping: dict[str, str] = {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        relative_path, url = row.get("relative_path"), row.get("url")
        if not isinstance(relative_path, str) or not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError(f"{path}:{number} requires relative_path and an http(s) url")
        if relative_path in mapping:
            raise ValueError(f"duplicate relative_path in {path}: {relative_path}")
        mapping[relative_path] = url
    if not mapping:
        raise ValueError(f"no signed URLs in {path}")
    return mapping, hashlib.sha256(raw).hexdigest()


def _with_signed_urls(items: list[ImageItem], mapping: dict[str, str], root: Path) -> list[ImageItem]:
    mapped: list[ImageItem] = []
    missing: list[str] = []
    for item in items:
        if item.image_path is None:
            raise ValueError("--image-url-map requires local image_path inputs")
        try:
            relative_path = item.image_path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"image path is outside repo root: {item.image_path}") from exc
        url = mapping.get(relative_path)
        if not url:
            missing.append(relative_path)
            continue
        mapped.append(ImageItem(item.id, item.image_path, url, item.task_id, item.label, item.mask_path, item.metadata))
    if missing:
        raise ValueError(f"--image-url-map lacks {len(missing)} input path(s); first: {missing[0]}")
    return mapped


def _recordable_image_url(value: str | None) -> str | None:
    """Keep the object location but never persist query-string signatures."""
    if not value:
        return None
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")) if parsed.scheme in {"http", "https"} else value


def _protocol_manifest(
    protocols: list[str],
    versions: dict[str, str] | None = None,
    suite_version: str | None = None,
    localization_coordinate_space: str = "bbox_px",
) -> dict[str, Any]:
    versions = versions or {key: PROTOCOL_VERSIONS[key] for key in protocols}
    combined = len(protocols) > 1
    effective_suite_version = suite_version or PROTOCOL_SUITE_VERSION
    payload: dict[str, Any] = {
        "version": (
            effective_suite_version
            if combined
            else versions[protocols[0]]
        ),
        "keys": protocols,
        "versions": versions,
        "replicates_required": 3,
    }
    if combined:
        payload["suite_version"] = effective_suite_version
    if "localization" in protocols and localization_coordinate_space != "bbox_px":
        payload["localization_coordinate_space"] = localization_coordinate_space
    return payload


def _manifest_payload(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: dict[str, Any],
    items: list[ImageItem],
    run_id: str,
    manifest_path: Path,
    raw_path: Path,
    output_path: Path,
    protocols: list[str],
    protocol_versions: dict[str, str],
    protocol_suite_version: str | None,
    localization_coordinate_space: str,
) -> dict[str, Any]:
    """Freeze a useful, secret-free description of one model run."""
    provider = model["provider"]
    metadata = {
        "schema_version": "mllm_run_manifest_v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "model": {"id": model["id"], "slug": model["slug"], "max_tokens": model["maxTokens"], "temperature": None if model.get("omitTemperature", False) else model["temperature"], "concurrency": model["concurrency"], "request_format": model.get("requestFormat", "openai_chat_completions"), "localization_coordinate_space": localization_coordinate_space},
        "protocol": _protocol_manifest(
            protocols,
            protocol_versions,
            protocol_suite_version,
            localization_coordinate_space,
        ),
        "input": {"source": args.source, "review_export": str(args.review_export) if args.review_export else None, "review_status": args.review_status if args.source == "review-export" else None, "include_source_pairs": args.include_source_pairs if args.source == "review-export" else None, "list": str(args.list) if args.list else None, "images": len(items), "manifest_sha256": manifest_hash(items)},
        "api": {"timeout_seconds": cfg["api"]["timeout"], "api_base": provider.get("apiBase"), "provider_header_keys": sorted(provider.get("headers", {}).keys()), "provider_extra_body_keys": sorted(provider.get("extraBody", {}).keys()), "model_omitted_extra_body_keys": sorted(model.get("omitExtraBodyKeys", []))},
        "retry": {"max_retries_per_replicate": cfg["retry"]["maxRetriesPerReplicate"], "base_backoff_seconds": cfg["retry"]["baseBackoffSeconds"]},
        "image": dict(cfg["image"]),
        "outputs": {"aggregate_jsonl": str(output_path), "raw_jsonl": str(raw_path), "run_manifest": str(manifest_path), "metrics_dir": str(output_path.parent / "metrics" / run_id)},
    }
    fingerprint_input = json.dumps({key: value for key, value in metadata.items() if key != "created_at_utc"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metadata["config_fingerprint_sha256"] = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    return metadata


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _existing_or_new_manifest(fresh: dict[str, Any], path: Path) -> dict[str, Any]:
    """A resumed run keeps its original identity and frozen input description."""
    if not path.is_file():
        _write_json(path, fresh)
        return fresh
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing_protocol = existing.get("protocol", {})
    fresh_protocol = fresh.get("protocol", {})
    existing_versions = existing_protocol.get("versions")
    if not isinstance(existing_versions, dict):
        existing_versions = {
            key: existing_protocol.get("version")
            for key in existing_protocol.get("keys", [])
            if key in PROTOCOL_VERSIONS
        }
    checks = (
        ("run_id", existing.get("run_id"), fresh.get("run_id")),
        ("condition", existing.get("condition"), fresh.get("condition")),
        ("model.id", existing.get("model", {}).get("id"), fresh.get("model", {}).get("id")),
        ("model.concurrency", existing.get("model", {}).get("concurrency"), fresh.get("model", {}).get("concurrency")),
        ("protocol.version", existing_protocol.get("version"), fresh_protocol.get("version")),
        ("protocol.keys", existing_protocol.get("keys"), fresh_protocol.get("keys")),
        ("protocol.versions", existing_versions, fresh_protocol.get("versions")),
        (
            "protocol.localization_coordinate_space",
            existing_protocol.get("localization_coordinate_space", "bbox_px"),
            fresh_protocol.get("localization_coordinate_space", "bbox_px"),
        ),
        ("input.manifest_sha256", existing.get("input", {}).get("manifest_sha256"), fresh.get("input", {}).get("manifest_sha256")),
    )
    mismatch = [name for name, old, new in checks if old != new]
    if mismatch:
        raise ValueError(f"run manifest does not match this invocation: {', '.join(mismatch)}")
    # A recovery may change only how the same frozen local inputs are delivered
    # to the model (for example, base64 -> approved OSS URLs). Preserve this
    # explicitly without changing the original run identity or its fingerprint.
    if existing.get("image") != fresh.get("image"):
        history = existing.setdefault("resume_image_configurations", [])
        entry = {"resumed_at_utc": datetime.now(timezone.utc).isoformat(), "image": fresh["image"]}
        if not any(previous.get("image") == entry["image"] for previous in history):
            history.append(entry)
            _write_json(path, existing)
    return existing


def _one_replicate(
    client: VisionClient,
    item: ImageItem,
    protocol: str,
    replicate: int,
    retry: dict[str, Any],
    dry_run: bool,
    image_size: tuple[int, int] | None = None,
    coordinate_space: str = "bbox_px",
) -> dict[str, Any]:
    image = client.image_url(item.image_path, item.image_url)
    attempts, base_prompt = [], _protocol_prompt(protocol, image_size, coordinate_space)
    prompt = base_prompt
    if dry_run:
        parsed = {"reasoning": "dry run", "decision": "no_localized_edit", "p_ai_edited": 50, "regions": []} if protocol == "localization" else {"reasoning": "dry run", "decision": "not_edited", "p_ai_edited": 50, "evidence": []}
        return {"status": "ok", "parsed": parsed, "attempts": attempts, "latency_ms": 0}
    for attempt in range(int(retry["maxRetriesPerReplicate"]) + 1):
        retry_after = None
        try:
            raw, latency = client.call(SYSTEM_PROMPT, prompt, image)
            parsed = parse(protocol, raw, image_size, coordinate_space)
            attempts.append({"attempt": attempt + 1, "status": "ok", "latency_ms": latency})
            return {"status": "ok", "parsed": parsed, "raw_response": raw, "attempts": attempts, "latency_ms": latency}
        except SchemaError as exc:
            attempts.append({
                "attempt": attempt + 1,
                "status": "schema_error",
                "error": str(exc),
                "raw_response": raw,
            })
            prompt = _schema_repair_prompt(
                base_prompt,
                protocol,
                exc,
                image_size,
                coordinate_space,
            )
        except RetryableError as exc:
            retry_after = exc.retry_after
            attempts.append({"attempt": attempt + 1, "status": "retryable_error", "error": str(exc)})
        except Exception as exc:
            attempts.append({"attempt": attempt + 1, "status": "error", "error": str(exc)})
            break
        if attempt < int(retry["maxRetriesPerReplicate"]):
            time.sleep(retry_after if retry_after is not None else retry_delay(attempt, retry["baseBackoffSeconds"]))
    return {"status": "error", "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", choices=["review-export", "list"], required=True)
    parser.add_argument("--review-export", type=Path)
    parser.add_argument("--review-status", default="good")
    parser.add_argument("--include-source-pairs", action="store_true")
    parser.add_argument("--list", type=Path)
    parser.add_argument("--protocol", choices=["detection", "localization", "both"], default="both")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True, help="unique run identifier; used in result filenames and every record")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-root", type=Path, default=Path("results/mllm"))
    parser.add_argument("--limit", type=int, help="process only the first N deterministic input rows (smoke testing)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="accepted for compatibility; successful rows are always resumed")
    parser.add_argument("--model-slug", action="append", help="run only matching model slug(s); can be repeated")
    parser.add_argument("--concurrency", type=int, help="override per-model request concurrency for this run")
    parser.add_argument("--max-tokens", type=int, help="override the selected model output-token limit")
    parser.add_argument("--api-timeout", type=float, help="override the configured per-request timeout in seconds")
    parser.add_argument("--max-retries-per-replicate", type=int, help="override retries inside each replicate; failures remain recorded in raw JSONL")
    parser.add_argument("--retry-until-complete", action="store_true", help="requeue failed replicates before aggregation/metrics until every unit has three valid replies")
    parser.add_argument("--max-recovery-waves", type=int, default=0, help="maximum failed-replicate recovery waves per model; 0 keeps retrying until complete")
    parser.add_argument("--recovery-backoff-seconds", type=float, default=10, help="initial delay between failed-replicate recovery waves")
    parser.add_argument("--write-metrics", action="store_true", help="after each model, write separate detection/localization metric tables from the review export")
    parser.add_argument("--image-transport", choices=["base64", "url"], help="override image transport for this invocation")
    parser.add_argument("--image-url-prefix", help="when using url transport, map repo-relative image paths below this oss/http prefix")
    parser.add_argument("--image-url-map", type=Path, help="local JSONL map of repo-relative paths to signed HTTP(S) URLs; never persisted verbatim")
    parser.add_argument("--skip-id", action="append", default=[], help="skip a specific image ID while resuming aggregation; repeat for multiple IDs")
    parser.add_argument("--aggregate-only", action="store_true", help="write aggregates and metrics from existing successful raw rows without sending model requests")
    args = parser.parse_args()
    if args.replicates != 3:
        raise SystemExit("MLLM protocol requires exactly --replicates 3")
    if args.write_metrics and args.source != "review-export":
        raise SystemExit("--write-metrics currently requires --source review-export")
    if args.concurrency is not None and args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")
    if args.api_timeout is not None and args.api_timeout <= 0:
        raise SystemExit("--api-timeout must be positive")
    if args.max_retries_per_replicate is not None and args.max_retries_per_replicate < 0:
        raise SystemExit("--max-retries-per-replicate must be non-negative")
    if args.recovery_backoff_seconds < 0:
        raise SystemExit("--recovery-backoff-seconds must be non-negative")
    if args.max_recovery_waves < 0:
        raise SystemExit("--max-recovery-waves must be non-negative")
    selected = set(args.model_slug or [])
    run_id = _safe_run_id(args.run_id)
    cfg, root = load_config(args.config, selected or None), args.repo_root.resolve()
    if args.max_tokens is not None:
        for model in cfg["models"]:
            model["maxTokens"] = args.max_tokens
    if args.api_timeout is not None:
        cfg["api"]["timeout"] = args.api_timeout
    if args.max_retries_per_replicate is not None:
        cfg["retry"]["maxRetriesPerReplicate"] = args.max_retries_per_replicate
    if args.image_transport:
        cfg["image"]["transport"] = args.image_transport
    if args.image_url_prefix:
        if cfg["image"].get("transport") != "url":
            raise SystemExit("--image-url-prefix requires --image-transport url")
        if args.image_url_map:
            raise SystemExit("--image-url-prefix and --image-url-map are mutually exclusive")
        cfg["image"]["urlPrefix"] = args.image_url_prefix.rstrip("/")
        cfg["image"]["localRoot"] = str(root)
    manifest_items = from_review_export(args.review_export, root, args.review_status, args.include_source_pairs) if args.source == "review-export" else from_jsonl(args.list, root)
    if args.limit is not None:
        manifest_items = manifest_items[:args.limit]
    items = manifest_items
    if args.image_url_map:
        if cfg["image"].get("transport") != "url":
            raise SystemExit("--image-url-map requires --image-transport url")
        signed_url_map, map_sha256 = _load_signed_url_map(args.image_url_map)
        items = _with_signed_urls(manifest_items, signed_url_map, root)
        cfg["image"]["signedUrlMapEntries"] = len(signed_url_map)
        cfg["image"]["signedUrlMapSha256"] = map_sha256
    protocols = ["detection", "localization"] if args.protocol == "both" else [args.protocol]
    print(json.dumps({
        "run_id": run_id,
        "items": len(manifest_items),
        "manifest_sha256": manifest_hash(manifest_items),
        "protocols": protocols,
        "models": [model["slug"] for model in cfg["models"]],
    }, ensure_ascii=False))
    skipped_ids = set(args.skip_id)
    for model in cfg["models"]:
        if args.concurrency is not None:
            model["concurrency"] = args.concurrency
        protocol_versions, protocol_suite_version, coordinate_space = (
            _protocol_versions_for_model(model, protocols)
        )
        folder = args.results_root / model["slug"]
        raw_path, output_path = folder / f"{run_id}.raw.jsonl", folder / f"{run_id}.jsonl"
        run_manifest_path = folder / f"{run_id}.run_manifest.json"
        fresh_manifest = _manifest_payload(
            args,
            cfg,
            model,
            manifest_items,
            run_id,
            run_manifest_path,
            raw_path,
            output_path,
            protocols,
            protocol_versions,
            protocol_suite_version,
            coordinate_space,
        )
        run_manifest = _existing_or_new_manifest(fresh_manifest, run_manifest_path)
        run_fields = {
            "run_id": run_id,
            "run_manifest_path": str(run_manifest_path),
            "input_manifest_sha256": run_manifest["input"]["manifest_sha256"],
            "config_fingerprint_sha256": run_manifest["config_fingerprint_sha256"],
        }
        if run_manifest["protocol"].get("suite_version"):
            run_fields["protocol_suite_version"] = run_manifest["protocol"]["suite_version"]
        done = completed_raw_keys(raw_path, protocol_versions)
        prior = successful_raw(raw_path, protocol_versions)
        aggregated = completed_aggregate_keys(output_path, protocol_versions)
        client = VisionClient(model, float(cfg["api"]["timeout"]), cfg["image"])
        units: dict[tuple[str, str], dict[str, Any]] = {}
        pending: list[tuple[tuple[str, str], int, ImageItem, str, str | None, tuple[int, int] | None, dict[str, Any]]] = []
        for item in items:
            if item.id in skipped_ids:
                continue
            size = _size(item.image_path) if item.image_path else None
            digest = hashlib.sha256(item.image_path.read_bytes()).hexdigest() if item.image_path else None
            for protocol in protocols:
                if (item.id, protocol) in aggregated:
                    continue
                unit_key = (item.id, protocol)
                units[unit_key] = {"item": item, "protocol": protocol, "size": size, "digest": digest, "replies": []}
                request_params = {"max_tokens": model["maxTokens"]}
                if not model.get("omitTemperature", False):
                    request_params["temperature"] = model["temperature"]
                for replicate in range(1, 4):
                    key = (item.id, protocol, replicate)
                    if key in done:
                        units[unit_key]["replies"].append(prior[key])
                        continue
                    pending.append((unit_key, replicate, item, protocol, digest, size, request_params))
        completed_attempts = 0

        def execute_pending(work: list[tuple[tuple[str, str], int, ImageItem, str, str | None, tuple[int, int] | None, dict[str, Any]]], phase: str) -> list[tuple[tuple[str, str], int, ImageItem, str, str | None, tuple[int, int] | None, dict[str, Any]]]:
            nonlocal completed_attempts
            if args.aggregate_only:
                if work:
                    print(json.dumps({"run_id": run_id, "model_slug": model["slug"], "phase": "aggregate_only", "skipped_pending_replicates": len(work)}, ensure_ascii=False), flush=True)
                return work
            failed = []
            with ThreadPoolExecutor(max_workers=int(model["concurrency"])) as executor:
                futures = {
                    executor.submit(
                        _one_replicate,
                        client,
                        item,
                        protocol,
                        replicate,
                        cfg["retry"],
                        args.dry_run,
                        size,
                        coordinate_space,
                    ): (unit_key, replicate, item, protocol, digest, size, request_params)
                    for unit_key, replicate, item, protocol, digest, size, request_params in work
                }
                for future in as_completed(futures):
                    task = futures[future]
                    unit_key, replicate, item, protocol, digest, size, request_params = task
                    result = future.result()
                    effective_image_url = client.image_url(item.image_path, item.image_url) if cfg["image"].get("transport") == "url" else item.image_url
                    row = {"schema_version": "mllm_raw_v1", "raw_id": _raw_id(run_id, item, protocol, replicate), "id": item.id, "task_id": item.task_id, "image_path": str(item.image_path) if item.image_path else None, "image_url": _recordable_image_url(effective_image_url), "image_sha256": digest, "image_size": size, "image_transport": cfg["image"].get("transport", "base64"), "condition": args.condition, "model": model["id"], "model_slug": model["slug"], "protocol_key": protocol, "protocol_id": PROTOCOL_IDS[protocol], "protocol_version": protocol_versions[protocol], "localization_coordinate_space": coordinate_space if protocol == "localization" else None, "replicate_index": replicate, "request_params": request_params, **run_fields, **result}
                    append_jsonl(raw_path, row)
                    if result["status"] == "ok":
                        units[unit_key]["replies"].append(result["parsed"])
                    else:
                        failed.append(task)
                    completed_attempts += 1
                    if completed_attempts % 25 == 0 or completed_attempts == len(pending):
                        print(json.dumps({"run_id": run_id, "model_slug": model["slug"], "phase": phase, "completed_attempts": completed_attempts, "initial_pending_replicates": len(pending), "failed_in_phase": len(failed)}, ensure_ascii=False), flush=True)
            return failed

        failed = execute_pending(pending, "initial")
        recovery_wave = 0
        while failed and args.retry_until_complete:
            if args.max_recovery_waves and recovery_wave >= args.max_recovery_waves:
                break
            recovery_wave += 1
            delay = min(60.0, args.recovery_backoff_seconds * recovery_wave)
            print(json.dumps({"run_id": run_id, "model_slug": model["slug"], "phase": "recovery", "recovery_wave": recovery_wave, "failed_replicates": len(failed), "backoff_seconds": delay}, ensure_ascii=False), flush=True)
            if delay:
                time.sleep(delay)
            failed = execute_pending(failed, f"recovery_{recovery_wave}")
        if failed:
            print(json.dumps({"run_id": run_id, "model_slug": model["slug"], "phase": "recovery_exhausted", "failed_replicates": len(failed)}, ensure_ascii=False), flush=True)
        for unit in units.values():
            item, protocol, size, digest, replies = unit["item"], unit["protocol"], unit["size"], unit["digest"], unit["replies"]
            request_params = {"max_tokens": model["maxTokens"]}
            if not model.get("omitTemperature", False):
                request_params["temperature"] = model["temperature"]
            if len(replies) == 3:
                summary = aggregate(protocol, replies)
                mask_path = None
                if protocol == "localization" and size is not None:
                    boxes = boxes_to_pixels(summary["regions"], size[0], size[1])
                    normalized_boxes = boxes_to_1000(summary["regions"], size[0], size[1])
                    mask_file = folder / "masks" / run_id / f"{item.id}.png"
                    write_union_mask(mask_file, size[0], size[1], boxes)
                    summary["regions_px"] = boxes
                    summary["regions_1000"] = normalized_boxes
                    mask_path = str(mask_file)
                effective_image_url = client.image_url(item.image_path, item.image_url) if cfg["image"].get("transport") == "url" else item.image_url
                append_jsonl(output_path, {"schema_version": "mllm_result_v1", "id": item.id, "task_id": item.task_id, "image_path": str(item.image_path) if item.image_path else None, "image_url": _recordable_image_url(effective_image_url), "image_sha256": digest, "image_size": size, "image_transport": cfg["image"].get("transport", "base64"), "condition": args.condition, "model": model["id"], "model_slug": model["slug"], "protocol_key": protocol, "protocol_id": PROTOCOL_IDS[protocol], "protocol_version": protocol_versions[protocol], "localization_coordinate_space": coordinate_space if protocol == "localization" else None, "request_params": request_params, **run_fields, "status": "ok", "valid_for_metrics": True, "replicate_count": 3, "successful_replicates": 3, "aggregation": {"decision": "majority", "probability": "median", "regions": "two-vote-iou-clusters-pixel" if protocol == "localization" else None}, "decision": summary["decision"], "p_ai_edited": summary["p_ai_edited"], "score": summary["p_ai_edited"] / 100, "regions_1000": summary.get("regions_1000", []), "regions_px": summary.get("regions_px", []), "mask_path": mask_path, "result": summary})
            else:
                append_jsonl(output_path, {"schema_version": "mllm_result_v1", "id": item.id, "task_id": item.task_id, "image_path": str(item.image_path) if item.image_path else None, "image_sha256": digest, "image_size": size, "condition": args.condition, "model": model["id"], "model_slug": model["slug"], "protocol_key": protocol, "protocol_id": PROTOCOL_IDS[protocol], "protocol_version": protocol_versions[protocol], "localization_coordinate_space": coordinate_space if protocol == "localization" else None, **run_fields, "status": "incomplete_replicates", "valid_for_metrics": False, "successful_replicates": len(replies), "replicate_count": 3})
        if args.write_metrics:
            from .metrics import evaluate_review_export
            metrics_dir = folder / "metrics" / run_id
            summary = evaluate_review_export(output_path, args.review_export, metrics_dir, status=args.review_status, include_source_pairs=args.include_source_pairs, protocol_version=protocol_versions, run_manifest_path=run_manifest_path, repo_root=root)
            print(json.dumps({"model_slug": model["slug"], "metrics_dir": str(metrics_dir), "metrics": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
