"""Run the CLAIMFORGE MLLM zoom-agent detection protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import VisionClient
from .config import load_config
from .inputs import (
    DEFAULT_BENCHMARK1000_LEDGER,
    DEFAULT_BENCHMARK1000_MANIFEST,
    ImageItem,
    from_benchmark1000,
    from_jsonl,
    from_review_export,
    manifest_hash,
)
from .masks import boxes_to_1000, boxes_to_pixels, write_union_mask
from .results import (
    append_jsonl,
    completed_aggregate_keys,
    completed_raw_keys,
    successful_raw,
)
from .run_mllm import (
    _load_signed_url_map,
    _recordable_image_url,
    _safe_run_id,
    _with_signed_urls,
    _write_json,
)
from .schema import aggregate
from .zoom_agent import (
    AGENT_PROTOCOL_VERSION,
    PROTOCOL_VERSIONS,
    ZOOM_TOOL_SCHEMA,
    run_agent_episode,
    summarize_agent_run,
)

PROTOCOL_IDS = {
    "detection": "mllm_zoom_agent_detection_v1",
    "localization": "mllm_zoom_agent_localization_v1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: dict[str, Any],
    items: list[ImageItem],
    run_id: str,
    raw_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    provider = model["provider"]
    payload: dict[str, Any] = {
        "schema_version": "mllm_zoom_agent_run_manifest_v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "model": {
            "id": model["id"],
            "slug": model["slug"],
            "max_tokens": model["maxTokens"],
            "temperature": (
                None if model.get("omitTemperature", False) else model["temperature"]
            ),
            "concurrency": model["concurrency"],
            "request_format": model.get(
                "requestFormat",
                "openai_chat_completions",
            ),
        },
        "protocol": {
            "version": AGENT_PROTOCOL_VERSION,
            "keys": ["detection", "localization"],
            "versions": PROTOCOL_VERSIONS,
            "replicates_required": 3,
            "coordinate_space": "bbox_1000",
            "max_zoom_calls_per_episode": args.max_zoom_calls,
            "max_inference_turns_per_episode": args.max_zoom_calls + 1,
            "zoom_long_side": args.zoom_long_side,
            "single_image_turns": args.single_image_turns,
            "tool_transport": "application_json_action",
            "tool_schema": ZOOM_TOOL_SCHEMA,
            "crop_source": "original_full_resolution_image",
        },
        "input": {
            "source": args.source,
            "review_export": str(args.review_export) if args.review_export else None,
            "review_status": (
                args.review_status if args.source == "review-export" else None
            ),
            "include_source_pairs": (
                args.include_source_pairs
                if args.source == "review-export"
                else None
            ),
            "list": (
                str(args.list)
                if args.list
                else (
                    str(DEFAULT_BENCHMARK1000_LEDGER)
                    if args.source == "benchmark1000"
                    else None
                )
            ),
            "benchmark_manifest": (
                str(args.benchmark_manifest)
                if args.source == "benchmark1000"
                else None
            ),
            "images": len(items),
            "manifest_sha256": manifest_hash(items),
        },
        "api": {
            "timeout_seconds": cfg["api"]["timeout"],
            "api_base": provider.get("apiBase"),
            "provider_header_keys": sorted(provider.get("headers", {}).keys()),
            "provider_extra_body_keys": sorted(
                provider.get("extraBody", {}).keys()
            ),
            "model_omitted_extra_body_keys": sorted(
                model.get("omitExtraBodyKeys", [])
            ),
        },
        "retry": {
            "max_retries_per_replicate": cfg["retry"][
                "maxRetriesPerReplicate"
            ],
            "max_retries_per_turn": cfg["retry"]["maxRetriesPerReplicate"],
            "base_backoff_seconds": cfg["retry"]["baseBackoffSeconds"],
        },
        "image": dict(cfg["image"]),
        "outputs": {
            "aggregate_jsonl": str(output_path),
            "raw_jsonl": str(raw_path),
            "run_manifest": str(output_path.with_suffix(".run_manifest.json")),
            "agent_metrics_json": str(
                output_path.with_suffix(".agent_metrics.json")
            ),
            "agent_metrics_csv": str(
                output_path.with_suffix(".agent_metrics.csv")
            ),
            "crops_dir": str(output_path.parent / "crops" / run_id),
            "masks_dir": str(output_path.parent / "masks" / run_id),
            "metrics_dir": str(output_path.parent / "metrics" / run_id),
        },
    }
    fingerprint_value = {
        key: value for key, value in payload.items() if key != "created_at_utc"
    }
    encoded = json.dumps(
        fingerprint_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["config_fingerprint_sha256"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    return payload


def _existing_or_new_manifest(
    fresh: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        _write_json(path, fresh)
        return fresh
    existing = json.loads(path.read_text(encoding="utf-8"))
    checks = (
        ("run_id", existing.get("run_id"), fresh.get("run_id")),
        ("condition", existing.get("condition"), fresh.get("condition")),
        (
            "model.id",
            existing.get("model", {}).get("id"),
            fresh.get("model", {}).get("id"),
        ),
        (
            "protocol.version",
            existing.get("protocol", {}).get("version"),
            fresh.get("protocol", {}).get("version"),
        ),
        (
            "protocol.max_zoom_calls_per_episode",
            existing.get("protocol", {}).get("max_zoom_calls_per_episode"),
            fresh.get("protocol", {}).get("max_zoom_calls_per_episode"),
        ),
        (
            "protocol.zoom_long_side",
            existing.get("protocol", {}).get("zoom_long_side"),
            fresh.get("protocol", {}).get("zoom_long_side"),
        ),
        (
            "protocol.single_image_turns",
            existing.get("protocol", {}).get("single_image_turns"),
            fresh.get("protocol", {}).get("single_image_turns"),
        ),
        (
            "input.manifest_sha256",
            existing.get("input", {}).get("manifest_sha256"),
            fresh.get("input", {}).get("manifest_sha256"),
        ),
    )
    mismatch = [name for name, old, new in checks if old != new]
    if mismatch:
        raise ValueError(
            "run manifest does not match this invocation: "
            + ", ".join(mismatch)
        )
    return existing


def _run_episode_safely(
    client: VisionClient,
    item: ImageItem,
    replicate: int,
    crop_dir: Path,
    retry: dict[str, Any],
    max_zoom_calls: int,
    zoom_long_side: int,
    single_image_turns: bool,
    dry_run: bool,
) -> dict[str, Any]:
    try:
        if item.image_path is None:
            raise ValueError(
                "zoom_in requires a local original image_path; URL-only inputs "
                "cannot be cropped"
            )
        original_image = client.image_url(item.image_path, item.image_url)
        return run_agent_episode(
            client,
            item.image_path,
            original_image,
            item.id,
            replicate,
            crop_dir,
            retry,
            max_zoom_calls=max_zoom_calls,
            zoom_long_side=zoom_long_side,
            single_image_turns=single_image_turns,
            dry_run=dry_run,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "turns": [],
            "tool_calls": [],
            "latency_ms": 0,
        }


def _append_aggregate(
    output_path: Path,
    folder: Path,
    run_id: str,
    condition: str,
    model: dict[str, Any],
    item: ImageItem,
    image_digest: str,
    image_size: tuple[int, int],
    parsed_episodes: list[dict[str, Any]],
    run_fields: dict[str, Any],
    original_image_url: str | None,
    image_transport: str,
    already_aggregated: set[tuple[str, str]],
) -> None:
    request_params: dict[str, Any] = {"max_tokens": model["maxTokens"]}
    if not model.get("omitTemperature", False):
        request_params["temperature"] = model["temperature"]
    for protocol in ("detection", "localization"):
        if (item.id, protocol) in already_aggregated:
            continue
        replies = [episode[protocol] for episode in parsed_episodes]
        summary = aggregate(protocol, replies)
        mask_path = None
        regions_px: list[list[float]] = []
        regions_1000: list[list[float]] = []
        if protocol == "localization":
            regions_px = boxes_to_pixels(
                summary["regions"],
                image_size[0],
                image_size[1],
            )
            regions_1000 = boxes_to_1000(
                summary["regions"],
                image_size[0],
                image_size[1],
            )
            mask_file = folder / "masks" / run_id / f"{item.id}.png"
            write_union_mask(
                mask_file,
                image_size[0],
                image_size[1],
                regions_px,
            )
            mask_path = str(mask_file)
            summary["regions_px"] = regions_px
            summary["regions_1000"] = regions_1000
        append_jsonl(
            output_path,
            {
                "schema_version": "mllm_result_v1",
                "id": item.id,
                "task_id": item.task_id,
                "image_path": str(item.image_path),
                "image_url": _recordable_image_url(original_image_url),
                "image_sha256": image_digest,
                "image_size": list(image_size),
                "image_transport": image_transport,
                "condition": condition,
                "model": model["id"],
                "model_slug": model["slug"],
                "protocol_key": protocol,
                "protocol_id": PROTOCOL_IDS[protocol],
                "protocol_version": PROTOCOL_VERSIONS[protocol],
                "protocol_suite_version": AGENT_PROTOCOL_VERSION,
                "localization_coordinate_space": (
                    "bbox_1000" if protocol == "localization" else None
                ),
                "request_params": request_params,
                **run_fields,
                "status": "ok",
                "valid_for_metrics": True,
                "replicate_count": 3,
                "successful_replicates": 3,
                "agent": {
                    "max_zoom_calls_per_episode": run_fields[
                        "max_zoom_calls_per_episode"
                    ],
                    "zoom_long_side": run_fields["zoom_long_side"],
                    "single_image_turns": run_fields[
                        "single_image_turns"
                    ],
                    "tool_calls_per_episode": [
                        episode.get("_tool_call_count", 0)
                        for episode in parsed_episodes
                    ],
                },
                "aggregation": {
                    "decision": "majority",
                    "probability": "median",
                    "regions": (
                        "two-vote-iou-clusters-pixel"
                        if protocol == "localization"
                        else None
                    ),
                },
                "decision": summary["decision"],
                "p_ai_edited": summary["p_ai_edited"],
                "score": summary["p_ai_edited"] / 100,
                "regions_1000": regions_1000,
                "regions_px": regions_px,
                "mask_path": mask_path,
                "result": summary,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an MLLM image-forensics agent with up to five zoom calls"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=["review-export", "list", "benchmark1000"],
        required=True,
    )
    parser.add_argument("--review-export", type=Path)
    parser.add_argument("--review-status", default="good")
    parser.add_argument("--include-source-pairs", action="store_true")
    parser.add_argument("--list", type=Path)
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK1000_MANIFEST,
        help=(
            "immutable 750-forged + 250-real manifest used by "
            "--source benchmark1000"
        ),
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/mllm"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-slug", action="append")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-timeout", type=float)
    parser.add_argument("--max-retries-per-turn", type=int)
    parser.add_argument("--retry-until-complete", action="store_true")
    parser.add_argument("--max-recovery-waves", type=int, default=0)
    parser.add_argument("--recovery-backoff-seconds", type=float, default=10)
    parser.add_argument("--write-metrics", action="store_true")
    parser.add_argument(
        "--image-transport",
        choices=["base64", "url"],
    )
    parser.add_argument("--image-url-prefix")
    parser.add_argument("--image-url-map", type=Path)
    parser.add_argument("--skip-id", action="append", default=[])
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--max-zoom-calls",
        type=int,
        default=5,
        help="maximum zoom_in executions per independent episode (0-5)",
    )
    parser.add_argument(
        "--zoom-long-side",
        type=int,
        default=1536,
        help="minimum long-side pixels of the PNG crop observation",
    )
    parser.add_argument(
        "--single-image-turns",
        action="store_true",
        help=(
            "after each zoom, rebuild context with the action transcript and "
            "only the newest crop image (provider compatibility mode)"
        ),
    )
    args = parser.parse_args()

    if args.replicates != 3:
        raise SystemExit("zoom-agent protocol requires exactly --replicates 3")
    if not 0 <= args.max_zoom_calls <= 5:
        raise SystemExit("--max-zoom-calls must be between 0 and 5")
    if args.zoom_long_side < 1:
        raise SystemExit("--zoom-long-side must be positive")
    if args.concurrency is not None and args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.api_timeout is not None and args.api_timeout <= 0:
        raise SystemExit("--api-timeout must be positive")
    if args.max_retries_per_turn is not None and args.max_retries_per_turn < 0:
        raise SystemExit("--max-retries-per-turn must be non-negative")
    if args.max_recovery_waves < 0:
        raise SystemExit("--max-recovery-waves must be non-negative")
    if args.recovery_backoff_seconds < 0:
        raise SystemExit("--recovery-backoff-seconds must be non-negative")
    if args.write_metrics and args.source != "review-export":
        raise SystemExit("--write-metrics requires --source review-export")
    if args.source == "review-export" and args.review_export is None:
        raise SystemExit("--source review-export requires --review-export")
    if args.source == "list" and args.list is None:
        raise SystemExit("--source list requires --list")

    selected = set(args.model_slug or [])
    run_id = _safe_run_id(args.run_id)
    root = args.repo_root.resolve()
    cfg = load_config(args.config, selected or None)
    if args.max_tokens is not None:
        for model in cfg["models"]:
            model["maxTokens"] = args.max_tokens
    if args.api_timeout is not None:
        cfg["api"]["timeout"] = args.api_timeout
    if args.max_retries_per_turn is not None:
        cfg["retry"]["maxRetriesPerReplicate"] = args.max_retries_per_turn
    if args.image_transport:
        cfg["image"]["transport"] = args.image_transport
    if args.image_url_prefix:
        if cfg["image"].get("transport") != "url":
            raise SystemExit("--image-url-prefix requires --image-transport url")
        if args.image_url_map:
            raise SystemExit(
                "--image-url-prefix and --image-url-map are mutually exclusive"
            )
        cfg["image"]["urlPrefix"] = args.image_url_prefix.rstrip("/")
        cfg["image"]["localRoot"] = str(root)

    if args.source == "review-export":
        manifest_items = from_review_export(
            args.review_export,
            root,
            args.review_status,
            args.include_source_pairs,
        )
    elif args.source == "list":
        manifest_items = from_jsonl(args.list, root)
    else:
        manifest_items = from_benchmark1000(
            args.benchmark_manifest,
            args.list or DEFAULT_BENCHMARK1000_LEDGER,
            root,
        )
    if args.limit is not None:
        manifest_items = manifest_items[:args.limit]
    if any(item.image_path is None for item in manifest_items):
        raise SystemExit(
            "zoom-agent inputs require local image_path values so zoom_in can "
            "crop the original pixels"
        )
    items = manifest_items
    if args.image_url_map:
        if cfg["image"].get("transport") != "url":
            raise SystemExit("--image-url-map requires --image-transport url")
        signed_url_map, map_sha256 = _load_signed_url_map(args.image_url_map)
        items = _with_signed_urls(manifest_items, signed_url_map, root)
        cfg["image"]["signedUrlMapEntries"] = len(signed_url_map)
        cfg["image"]["signedUrlMapSha256"] = map_sha256

    print(
        json.dumps(
            {
                "run_id": run_id,
                "items": len(items),
                "manifest_sha256": manifest_hash(manifest_items),
                "models": [model["slug"] for model in cfg["models"]],
                "max_zoom_calls": args.max_zoom_calls,
                "zoom_long_side": args.zoom_long_side,
                "single_image_turns": args.single_image_turns,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    skipped_ids = set(args.skip_id)
    for model in cfg["models"]:
        if args.concurrency is not None:
            model["concurrency"] = args.concurrency
        if model.get("requestFormat") == "gemini_httpstream":
            raise SystemExit(
                f"{model['slug']}: zoom-agent requires the "
                "openai_chat_completions request format"
            )
        folder = args.results_root / model["slug"] / "agent_zoom"
        raw_path = folder / f"{run_id}.raw.jsonl"
        output_path = folder / f"{run_id}.jsonl"
        run_manifest_path = folder / f"{run_id}.run_manifest.json"
        crop_dir = folder / "crops" / run_id
        fresh_manifest = _manifest_payload(
            args,
            cfg,
            model,
            manifest_items,
            run_id,
            raw_path,
            output_path,
        )
        run_manifest = _existing_or_new_manifest(
            fresh_manifest,
            run_manifest_path,
        )
        run_fields = {
            "run_id": run_id,
            "run_manifest_path": str(run_manifest_path),
            "input_manifest_sha256": run_manifest["input"][
                "manifest_sha256"
            ],
            "config_fingerprint_sha256": run_manifest[
                "config_fingerprint_sha256"
            ],
            "max_zoom_calls_per_episode": args.max_zoom_calls,
            "zoom_long_side": args.zoom_long_side,
            "single_image_turns": args.single_image_turns,
        }
        done = completed_raw_keys(raw_path, AGENT_PROTOCOL_VERSION)
        prior = successful_raw(raw_path, AGENT_PROTOCOL_VERSION)
        aggregated = completed_aggregate_keys(output_path, PROTOCOL_VERSIONS)
        client = VisionClient(
            model,
            float(cfg["api"]["timeout"]),
            cfg["image"],
        )
        units: dict[str, dict[str, Any]] = {}
        pending: list[tuple[ImageItem, int]] = []
        for item in items:
            if item.id in skipped_ids:
                continue
            if all(
                (item.id, protocol) in aggregated
                for protocol in ("detection", "localization")
            ):
                continue
            assert item.image_path is not None
            from PIL import Image, ImageOps

            with Image.open(item.image_path) as opened:
                image_size = ImageOps.exif_transpose(opened).size
            unit = {
                "item": item,
                "image_size": image_size,
                "digest": _sha256(item.image_path),
                "episodes": [],
            }
            units[item.id] = unit
            for replicate in range(1, 4):
                key = (item.id, "agent_zoom", replicate)
                if key in done:
                    unit["episodes"].append(prior[key])
                else:
                    pending.append((item, replicate))

        initial_pending = len(pending)
        completed_attempts = 0

        def execute_pending(
            work: list[tuple[ImageItem, int]],
            phase: str,
        ) -> list[tuple[ImageItem, int]]:
            nonlocal completed_attempts
            if args.aggregate_only:
                return work
            failed: list[tuple[ImageItem, int]] = []
            with ThreadPoolExecutor(
                max_workers=int(model["concurrency"])
            ) as executor:
                futures = {
                    executor.submit(
                        _run_episode_safely,
                        client,
                        item,
                        replicate,
                        crop_dir,
                        cfg["retry"],
                        args.max_zoom_calls,
                        args.zoom_long_side,
                        args.single_image_turns,
                        args.dry_run,
                    ): (item, replicate)
                    for item, replicate in work
                }
                for future in as_completed(futures):
                    item, replicate = futures[future]
                    result = future.result()
                    if result["status"] == "ok":
                        parsed = dict(result["parsed"])
                        parsed["_tool_call_count"] = len(
                            result.get("tool_calls", [])
                        )
                        result = {**result, "parsed": parsed}
                    assert item.image_path is not None
                    original_url = (
                        client.image_url(item.image_path, item.image_url)
                        if cfg["image"].get("transport") == "url"
                        else item.image_url
                    )
                    request_params: dict[str, Any] = {
                        "max_tokens": model["maxTokens"],
                    }
                    if not model.get("omitTemperature", False):
                        request_params["temperature"] = model["temperature"]
                    row = {
                        "schema_version": "mllm_zoom_agent_raw_v1",
                        "raw_id": (
                            f"{run_id}:{item.id}:agent_zoom:{replicate}"
                        ),
                        "id": item.id,
                        "task_id": item.task_id,
                        "image_path": str(item.image_path),
                        "image_url": _recordable_image_url(original_url),
                        "image_sha256": units[item.id]["digest"],
                        "image_size": list(units[item.id]["image_size"]),
                        "image_transport": cfg["image"].get(
                            "transport",
                            "base64",
                        ),
                        "condition": args.condition,
                        "model": model["id"],
                        "model_slug": model["slug"],
                        "protocol_key": "agent_zoom",
                        "protocol_id": "mllm_zoom_agent_v1",
                        "protocol_version": AGENT_PROTOCOL_VERSION,
                        "replicate_index": replicate,
                        "request_params": request_params,
                        **run_fields,
                        **result,
                    }
                    append_jsonl(raw_path, row)
                    if result["status"] == "ok":
                        units[item.id]["episodes"].append(result["parsed"])
                    else:
                        failed.append((item, replicate))
                    completed_attempts += 1
                    if (
                        completed_attempts % 25 == 0
                        or completed_attempts == initial_pending
                    ):
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "model_slug": model["slug"],
                                    "phase": phase,
                                    "completed_episode_attempts": (
                                        completed_attempts
                                    ),
                                    "initial_pending_episodes": (
                                        initial_pending
                                    ),
                                    "failed_in_phase": len(failed),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
            return failed

        failed = execute_pending(pending, "initial")
        recovery_wave = 0
        while failed and args.retry_until_complete:
            if (
                args.max_recovery_waves
                and recovery_wave >= args.max_recovery_waves
            ):
                break
            recovery_wave += 1
            delay = min(
                60.0,
                args.recovery_backoff_seconds * recovery_wave,
            )
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "model_slug": model["slug"],
                        "phase": "recovery",
                        "recovery_wave": recovery_wave,
                        "failed_episodes": len(failed),
                        "backoff_seconds": delay,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if delay:
                time.sleep(delay)
            failed = execute_pending(failed, f"recovery_{recovery_wave}")

        for unit in units.values():
            item = unit["item"]
            episodes = unit["episodes"]
            if len(episodes) == 3:
                original_url = (
                    client.image_url(item.image_path, item.image_url)
                    if cfg["image"].get("transport") == "url"
                    else item.image_url
                )
                _append_aggregate(
                    output_path,
                    folder,
                    run_id,
                    args.condition,
                    model,
                    item,
                    unit["digest"],
                    unit["image_size"],
                    episodes,
                    run_fields,
                    original_url,
                    cfg["image"].get("transport", "base64"),
                    aggregated,
                )
            else:
                for protocol in ("detection", "localization"):
                    if (item.id, protocol) in aggregated:
                        continue
                    append_jsonl(
                        output_path,
                        {
                            "schema_version": "mllm_result_v1",
                            "id": item.id,
                            "task_id": item.task_id,
                            "image_path": str(item.image_path),
                            "image_sha256": unit["digest"],
                            "image_size": list(unit["image_size"]),
                            "condition": args.condition,
                            "model": model["id"],
                            "model_slug": model["slug"],
                            "protocol_key": protocol,
                            "protocol_id": PROTOCOL_IDS[protocol],
                            "protocol_version": PROTOCOL_VERSIONS[protocol],
                            "protocol_suite_version": (
                                AGENT_PROTOCOL_VERSION
                            ),
                            **run_fields,
                            "status": "incomplete_replicates",
                            "valid_for_metrics": False,
                            "successful_replicates": len(episodes),
                            "replicate_count": 3,
                        },
                    )
        agent_summary = summarize_agent_run(
            raw_path,
            output_path.with_suffix(".agent_metrics.json"),
            len(
                [
                    item
                    for item in items
                    if item.id not in skipped_ids
                ]
            ),
            run_id,
            model["slug"],
            args.max_zoom_calls,
        )
        print(
            json.dumps(
                {
                    "model_slug": model["slug"],
                    "agent_metrics": agent_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.write_metrics:
            from .metrics import evaluate_review_export

            metrics_dir = folder / "metrics" / run_id
            summary = evaluate_review_export(
                output_path,
                args.review_export,
                metrics_dir,
                status=args.review_status,
                include_source_pairs=args.include_source_pairs,
                protocol_version=PROTOCOL_VERSIONS,
                run_manifest_path=run_manifest_path,
                repo_root=root,
            )
            print(
                json.dumps(
                    {
                        "model_slug": model["slug"],
                        "metrics_dir": str(metrics_dir),
                        "metrics": summary,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
