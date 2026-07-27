"""Strict response parsing and deterministic three-replicate aggregation."""
from __future__ import annotations

import json
import statistics
from typing import Any


class SchemaError(ValueError):
    pass


def _json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        text = text[start:end + 1] if start >= 0 and end > start else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SchemaError("response must be a JSON object")
    return value


def _prob(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not 0 <= value <= 100:
        raise SchemaError("p_ai_edited must be an integer in [0,100]")
    return int(value)


def parse(
    protocol: str,
    raw: str,
    image_size: tuple[int, int] | None = None,
    coordinate_space: str = "bbox_px",
) -> dict[str, Any]:
    value = _json(raw)
    p = _prob(value.get("p_ai_edited"))
    reasoning = value.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise SchemaError("reasoning must be a non-empty string")
    if protocol == "detection":
        decision = value.get("decision")
        if decision not in {"edited", "not_edited"}:
            raise SchemaError("invalid detection decision")
        evidence = value.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) > 3 or not all(isinstance(x, str) for x in evidence):
            raise SchemaError("evidence must be <=3 strings")
        return {"reasoning": reasoning, "decision": decision, "p_ai_edited": p, "evidence": evidence}
    decision = value.get("decision")
    if decision not in {"localized_edit", "no_localized_edit"}:
        raise SchemaError("invalid localization decision")
    if image_size is None:
        raise SchemaError("localization parsing requires image dimensions")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise SchemaError("image dimensions must be positive")
    if coordinate_space not in {"bbox_px", "bbox_1000"}:
        raise SchemaError(f"unsupported localization coordinate space: {coordinate_space}")
    regions = value.get("regions", [])
    if not isinstance(regions, list) or len(regions) > 3:
        raise SchemaError("regions must be an array of at most 3 items")
    parsed = []
    for region in regions:
        if not isinstance(region, dict):
            raise SchemaError("region must be an object")
        box = region.get(coordinate_space)
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in box):
            raise SchemaError(f"{coordinate_space} must contain four numbers")
        x1, y1, x2, y2 = map(float, box)
        if coordinate_space == "bbox_1000":
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise SchemaError("bbox_1000 is out of range or degenerate")
            source_box = [x1, y1, x2, y2]
            x1, y1, x2, y2 = (
                x1 * width / 1000,
                y1 * height / 1000,
                x2 * width / 1000,
                y2 * height / 1000,
            )
        else:
            source_box = None
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise SchemaError(f"bbox_px is out of range or degenerate for {width}x{height} image")
        parsed_region = {
            "bbox_px": [x1, y1, x2, y2],
            "confidence": _prob(region.get("confidence")),
            "evidence": str(region.get("evidence", "")),
        }
        if source_box is not None:
            parsed_region["bbox_1000"] = source_box
        parsed.append(parsed_region)
    if decision == "localized_edit" and not parsed:
        raise SchemaError("localized_edit requires a region")
    return {"reasoning": reasoning, "decision": decision, "p_ai_edited": p, "regions": parsed}


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def aggregate(protocol: str, replies: list[dict[str, Any]]) -> dict[str, Any]:
    if len(replies) != 3:
        raise ValueError("aggregation requires exactly three successful replicates")
    probability = int(statistics.median(item["p_ai_edited"] for item in replies))
    if protocol == "detection":
        votes = {key: sum(item["decision"] == key for item in replies) for key in ("edited", "not_edited")}
        decision = max(votes, key=votes.get)
        tie = votes[decision] == 1
        return {"decision": decision, "p_ai_edited": probability, "vote_tie": tie, "reasoning": [x["reasoning"] for x in replies], "evidence": [x["evidence"] for x in replies]}
    positive = [index for index, item in enumerate(replies) if item["decision"] == "localized_edit" and item["regions"]]
    regions = []
    for replicate, item in enumerate(replies):
        for region in item["regions"]:
            placed = False
            for cluster in regions:
                if any(_iou(region["bbox_px"], member["bbox_px"]) >= 0.10 for member in cluster):
                    cluster.append({**region, "replicate": replicate}); placed = True; break
            if not placed:
                regions.append([{**region, "replicate": replicate}])
    consensus = []
    for cluster in regions:
        if len({member["replicate"] for member in cluster}) >= 2:
            coords = [statistics.median(member["bbox_px"][i] for member in cluster) for i in range(4)]
            consensus.append({"bbox_px": coords, "confidence": int(statistics.median(member["confidence"] for member in cluster)), "support": len({member["replicate"] for member in cluster})})
    decision = "localized_edit" if len(positive) >= 2 else "no_localized_edit"
    return {"decision": decision, "p_ai_edited": probability, "regions": consensus, "positive_votes": len(positive), "reasoning": [x["reasoning"] for x in replies]}
