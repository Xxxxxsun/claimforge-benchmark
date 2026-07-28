"""Provider-portable image-forensics agent loop with a local zoom-in tool."""
from __future__ import annotations

import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .client import RetryableError, VisionClient, retry_delay
from .results import iter_jsonl
from .schema import SchemaError, parse

AGENT_PROTOCOL_VERSION = "mllm_zoom_agent_v2_bboxpx_20260728"
DETECTION_PROTOCOL_VERSION = "mllm_zoom_agent_detection_v2_20260728"
LOCALIZATION_PROTOCOL_VERSION = "mllm_zoom_agent_localization_v2_bboxpx_20260728"
PROTOCOL_VERSIONS = {
    "detection": DETECTION_PROTOCOL_VERSION,
    "localization": LOCALIZATION_PROTOCOL_VERSION,
}
ZOOM_TOOL_SCHEMA: dict[str, Any] = {
    "name": "zoom_in",
    "description": (
        "Crop and enlarge a rectangular region from the original "
        "full-resolution image."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "bbox_px": {
                "type": "array",
                "description": (
                    "Original full-image pixel [x1,y1,x2,y2] box. Bounds "
                    "depend on the original image dimensions supplied in "
                    "the prompt."
                ),
                "items": {"type": "integer", "minimum": 0},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["bbox_px"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are an image-forensics agent. Examine the supplied image for localized digital manipulation. You may inspect the original full image and request high-resolution crops with the zoom_in tool. Pay close attention to inconsistencies in lighting, shadows, texture, edges, perspective, scale, and logical composition. Base the final answer only on visible evidence."""

INITIAL_PROMPT = """Analyze the original full image for digital editing.

You have access to one application tool:
zoom_in(bbox_px): crop a rectangular region from the ORIGINAL full-resolution image and return an enlarged observation.

The ORIGINAL full image is {width} x {height} pixels.
Tool coordinates always use its ORIGINAL full-image pixel coordinate system:
- top-left is (0, 0), bottom-right is ({width}, {height})
- bbox_px is [x1, y1, x2, y2] using integer pixel coordinates
- require 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}
- a crop observation does not create a new coordinate system

You may call zoom_in at most {max_zoom_calls} times. Use it only when closer inspection can resolve a specific uncertainty. You may inspect different or overlapping regions. When enough evidence is available, finish without spending unused calls.

For a tool call, return exactly one JSON object:
{{
  "action": "zoom_in",
  "reasoning": "<why this original-image region needs closer inspection>",
  "bbox_px": [<x1>, <y1>, <x2>, <y2>]
}}

For the final answer, return exactly one JSON object:
{{
  "action": "final",
  "reasoning": "<detailed image-forensics reasoning using the full image and any zoom observations>",
  "decision": "edited" | "not_edited",
  "p_ai_edited": <integer 0-100>,
  "evidence": [<at most 3 short visible-evidence statements>],
  "regions": [{{
    "bbox_px": [<x1>, <y1>, <x2>, <y2>],
    "confidence": <integer 0-100>,
    "evidence": "<short visible-evidence statement>"
  }}]
}}

Final-answer rules:
- regions use the ORIGINAL full-image bbox_px coordinates, never crop-relative coordinates
- return at most 3 regions, ordered by confidence
- if decision is not_edited, regions must be []
- if edited evidence is visible but cannot be localized reliably, decision may be edited with regions []
- output JSON only, without Markdown
"""


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not text.startswith("{"):
        start = text.find("{")
        text = text[start:] if start >= 0 else text
    try:
        # Some OpenAI-compatible gateways/models emit an entire imagined
        # action trajectory as adjacent JSON objects in one assistant turn.
        # Execute only the first object.  Later objects have not seen the
        # requested crop and therefore must never be accepted as observations
        # or final answers.
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SchemaError("response must be a JSON object")
    return value


def _bbox_px(
    value: Any,
    image_size: tuple[int, int],
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and int(x) == x
            for x in value
        )
    ):
        raise SchemaError("bbox_px must contain four integer pixel coordinates")
    box = [int(x) for x in value]
    width, height = image_size
    if not (
        0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    ):
        raise SchemaError(
            f"bbox_px is out of range or degenerate for "
            f"{width}x{height} image"
        )
    return box


def parse_agent_action(
    raw: str,
    image_size: tuple[int, int],
    zoom_calls_used: int,
    max_zoom_calls: int,
) -> dict[str, Any]:
    """Parse one agent action in original full-image pixel coordinates."""
    value = _json_object(raw)
    action = value.get("action")
    reasoning = value.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise SchemaError("reasoning must be a non-empty string")
    if action == "zoom_in":
        if zoom_calls_used >= max_zoom_calls:
            raise SchemaError("zoom_in call limit has been reached; return action=final")
        return {
            "action": "zoom_in",
            "reasoning": reasoning.strip(),
            "bbox_px": _bbox_px(value.get("bbox_px"), image_size),
        }
    if action != "final":
        raise SchemaError("action must be zoom_in or final")

    detection_value = {
        "reasoning": reasoning,
        "decision": value.get("decision"),
        "p_ai_edited": value.get("p_ai_edited"),
        "evidence": value.get("evidence", []),
    }
    detection = parse("detection", json.dumps(detection_value))
    raw_regions = value.get("regions", [])
    if not isinstance(raw_regions, list):
        raise SchemaError("regions must be an array")
    if detection["decision"] == "not_edited" and raw_regions:
        raise SchemaError("not_edited requires an empty regions array")
    localization_value = {
        "reasoning": reasoning,
        "decision": "localized_edit" if raw_regions else "no_localized_edit",
        "p_ai_edited": detection["p_ai_edited"],
        "regions": raw_regions,
    }
    localization = parse(
        "localization",
        json.dumps(localization_value),
        image_size=image_size,
        coordinate_space="bbox_px",
    )
    return {
        "action": "final",
        "detection": detection,
        "localization": localization,
    }


def create_zoom_crop(
    original_path: Path,
    bbox_px: list[int],
    output_path: Path,
    long_side: int,
) -> dict[str, Any]:
    """Crop from the original image and enlarge the observation losslessly as PNG."""
    if long_side < 1:
        raise ValueError("zoom long side must be positive")
    from PIL import Image, ImageOps

    with Image.open(original_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
        bbox_px = _bbox_px(bbox_px, original.size)
        crop = original.crop(tuple(bbox_px))
        input_size = crop.size
        scale = max(1.0, long_side / max(crop.size))
        output_size = (
            max(1, round(crop.width * scale)),
            max(1, round(crop.height * scale)),
        )
        if output_size != crop.size:
            crop = crop.resize(output_size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="PNG", optimize=False)
    return {
        "bbox_px": bbox_px,
        "crop_input_size": list(input_size),
        "crop_output_size": list(output_size),
        "crop_path": str(output_path),
    }


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or "image"


def _repair_message(error: Exception, calls_remaining: int) -> dict[str, Any]:
    instruction = (
        f"Your previous response was invalid: {error}. "
        f"Return one valid JSON action only. zoom_in calls remaining: {calls_remaining}."
    )
    if calls_remaining == 0:
        instruction += " You must return action=final; do not call zoom_in."
    return {"role": "user", "content": instruction}


def run_agent_episode(
    client: VisionClient,
    image_path: Path,
    original_image: str,
    item_id: str,
    replicate: int,
    crop_dir: Path,
    retry: dict[str, Any],
    max_zoom_calls: int = 5,
    zoom_long_side: int = 1536,
    single_image_turns: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one independent full-image → zoom observation(s) → final episode."""
    if not 0 <= max_zoom_calls <= 5:
        raise ValueError("max_zoom_calls must be between 0 and 5")
    from PIL import Image, ImageOps

    with Image.open(image_path) as opened:
        image_size = ImageOps.exif_transpose(opened).size
    if dry_run:
        detection = {
            "reasoning": "dry run",
            "decision": "not_edited",
            "p_ai_edited": 50,
            "evidence": [],
        }
        localization = {
            "reasoning": "dry run",
            "decision": "no_localized_edit",
            "p_ai_edited": 50,
            "regions": [],
        }
        return {
            "status": "ok",
            "parsed": {"detection": detection, "localization": localization},
            "turns": [],
            "tool_calls": [],
            "latency_ms": 0,
        }

    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": INITIAL_PROMPT.format(
                    max_zoom_calls=max_zoom_calls,
                    width=image_size[0],
                    height=image_size[1],
                ),
            },
            client.image_part(original_image),
        ],
    }]
    turns: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    prior_action_summaries: list[str] = []
    total_latency = 0

    for turn_index in range(max_zoom_calls + 1):
        parsed_action: dict[str, Any] | None = None
        turn_attempts: list[dict[str, Any]] = []
        raw = ""
        repair_error: SchemaError | None = None
        for attempt in range(int(retry["maxRetriesPerReplicate"]) + 1):
            retry_after = None
            try:
                call_messages = messages
                if repair_error is not None:
                    call_messages = [
                        *messages,
                        _repair_message(
                            repair_error,
                            max_zoom_calls - len(tool_calls),
                        ),
                    ]
                raw, latency = client.call_messages(SYSTEM_PROMPT, call_messages)
                total_latency += latency
                parsed_action = parse_agent_action(
                    raw,
                    image_size,
                    len(tool_calls),
                    max_zoom_calls,
                )
                turn_attempts.append({
                    "attempt": attempt + 1,
                    "status": "ok",
                    "latency_ms": latency,
                })
                break
            except SchemaError as exc:
                repair_error = exc
                turn_attempts.append({
                    "attempt": attempt + 1,
                    "status": "schema_error",
                    "error": str(exc),
                    "raw_response": raw,
                })
            except RetryableError as exc:
                retry_after = exc.retry_after
                turn_attempts.append({
                    "attempt": attempt + 1,
                    "status": "retryable_error",
                    "error": str(exc),
                })
            except Exception as exc:
                turn_attempts.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "error": str(exc),
                })
                break
            if attempt < int(retry["maxRetriesPerReplicate"]):
                delay = (
                    retry_after
                    if retry_after is not None
                    else retry_delay(attempt, retry["baseBackoffSeconds"])
                )
                time.sleep(delay)
        turns.append({
            "turn_index": turn_index + 1,
            "raw_response": raw or None,
            "parsed_action": parsed_action,
            "attempts": turn_attempts,
        })
        if parsed_action is None:
            return {
                "status": "error",
                "turns": turns,
                "tool_calls": tool_calls,
                "latency_ms": total_latency,
            }
        if parsed_action["action"] == "final":
            return {
                "status": "ok",
                "parsed": {
                    "detection": parsed_action["detection"],
                    "localization": parsed_action["localization"],
                },
                "turns": turns,
                "tool_calls": tool_calls,
                "latency_ms": total_latency,
            }

        zoom_index = len(tool_calls) + 1
        crop_path = (
            crop_dir
            / _safe_component(item_id)
            / f"replicate_{replicate}_zoom_{zoom_index}.png"
        )
        tool_result = create_zoom_crop(
            image_path,
            parsed_action["bbox_px"],
            crop_path,
            zoom_long_side,
        )
        tool_result["tool_call_index"] = zoom_index
        tool_calls.append(tool_result)
        calls_remaining = max_zoom_calls - len(tool_calls)
        tool_text = (
            "zoom_in tool result: this observation is the enlarged crop "
            f"from original bbox_px={tool_result['bbox_px']}. "
            f"zoom_in calls remaining: {calls_remaining}. "
            "Retain the original full-image coordinate system for any "
            "next tool call and for final regions."
            + (
                " You must now return action=final."
                if calls_remaining == 0
                else ""
            )
        )
        if single_image_turns:
            prior_action_summaries.append(raw)
            # Provider compatibility mode: some OpenAI-compatible gateways
            # reset connections when a later request contains the original
            # base64 image plus one or more crop images. Rebuild the next turn
            # with the executed action transcript and only the newest crop.
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            INITIAL_PROMPT.format(
                                max_zoom_calls=max_zoom_calls,
                                width=image_size[0],
                                height=image_size[1],
                            )
                            + "\nPreviously executed action JSON objects:\n"
                            + "\n".join(prior_action_summaries)
                            + "\n"
                            + tool_text
                            + "\nThis compatibility turn contains only the "
                            "newest crop image; use the recorded reasoning "
                            "above as the transcript of the original-image "
                            "inspection."
                        ),
                    },
                    client.image_part(client.image_data_url(crop_path)),
                ],
            }]
        else:
            messages.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": tool_text},
                        client.image_part(client.image_data_url(crop_path)),
                    ],
                },
            ])

    return {
        "status": "error",
        "turns": turns,
        "tool_calls": tool_calls,
        "latency_ms": total_latency,
        "error": "agent loop ended without a final action",
    }


def summarize_agent_run(
    raw_path: Path,
    output_json: Path,
    expected_images: int,
    run_id: str,
    model_slug: str,
    max_zoom_calls: int = 5,
) -> dict[str, Any]:
    """Summarize latest per-episode tool use and validity without using GT."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    if raw_path.is_file():
        for row in iter_jsonl(raw_path):
            if (
                row.get("run_id") == run_id
                and row.get("protocol_version") == AGENT_PROTOCOL_VERSION
            ):
                latest[(str(row["id"]), int(row["replicate_index"]))] = row
    successful = [row for row in latest.values() if row.get("status") == "ok"]
    calls = [len(row.get("tool_calls", [])) for row in successful]
    turns = [len(row.get("turns", [])) for row in successful]
    latencies = [
        int(row["latency_ms"])
        for row in successful
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    decision_counts = {"edited": 0, "not_edited": 0}
    for row in successful:
        decision = row.get("parsed", {}).get("detection", {}).get("decision")
        if decision in decision_counts:
            decision_counts[decision] += 1
    attempt_status_counts: dict[str, int] = {}
    for row in latest.values():
        for turn in row.get("turns", []):
            for attempt in turn.get("attempts", []):
                status = str(attempt.get("status", "unknown"))
                attempt_status_counts[status] = (
                    attempt_status_counts.get(status, 0) + 1
                )
    successful_by_image: dict[str, int] = {}
    for row in successful:
        image_id = str(row["id"])
        successful_by_image[image_id] = successful_by_image.get(image_id, 0) + 1
    expected_episodes = expected_images * 3
    zoom_call_histogram = {
        str(value): sum(count == value for count in calls)
        for value in range(max_zoom_calls + 1)
    }
    summary: dict[str, Any] = {
        "schema_version": "mllm_zoom_agent_metrics_v1",
        "run_id": run_id,
        "model_slug": model_slug,
        "protocol_version": AGENT_PROTOCOL_VERSION,
        "expected_images": expected_images,
        "expected_episodes": expected_episodes,
        "recorded_latest_episodes": len(latest),
        "successful_episodes": len(successful),
        "episode_coverage": (
            len(successful) / expected_episodes if expected_episodes else None
        ),
        "images_with_three_successful_episodes": sum(
            count == 3 for count in successful_by_image.values()
        ),
        "max_zoom_calls_per_episode": max_zoom_calls,
        "zoom_calls_total": sum(calls),
        "mean_zoom_calls_per_successful_episode": (
            statistics.fmean(calls) if calls else None
        ),
        "zoom_call_count_histogram": zoom_call_histogram,
        "episodes_with_zero_zoom_calls": sum(value == 0 for value in calls),
        "episodes_with_one_zoom_call": sum(value == 1 for value in calls),
        "episodes_with_two_zoom_calls": sum(value == 2 for value in calls),
        "episodes_with_any_zoom_rate": (
            sum(value > 0 for value in calls) / len(calls) if calls else None
        ),
        "episodes_using_full_zoom_budget_rate": (
            sum(value == max_zoom_calls for value in calls) / len(calls)
            if calls
            else None
        ),
        "mean_inference_turns_per_successful_episode": (
            statistics.fmean(turns) if turns else None
        ),
        "mean_latency_ms_per_successful_episode": (
            statistics.fmean(latencies) if latencies else None
        ),
        "median_latency_ms_per_successful_episode": (
            statistics.median(latencies) if latencies else None
        ),
        "episode_final_decision_counts": decision_counts,
        "attempt_status_counts": attempt_status_counts,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_json.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        flattened = {
            **{
                key: value
                for key, value in summary.items()
                if not isinstance(value, dict)
            },
            **{
                f"episode_final_{key}": value
                for key, value in decision_counts.items()
            },
            **{
                f"attempt_status_{key}": value
                for key, value in attempt_status_counts.items()
            },
        }
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)
    return summary
