#!/usr/bin/env python3
"""Seed a bbox_1000 Doubao run from legacy pixel-protocol raw responses.

Legacy Doubao replies frequently placed normalized 0-1000 values in bbox_px.
This script reinterprets successful replies offline. By default, it omits a
replicate when an earlier out-of-range positive reply was followed by a final
no_localized_edit reply. Running eval.mllm.run_mllm with the same run ID then
requests only those omitted replicates and aggregates all three canonical
pixel-coordinate replies.

With --use-final-negative-fallback, the final no_localized_edit response is
retained instead. This treats coordinate-format understanding, schema
compliance, and retry stability as part of model performance while preserving
explicit provenance on every affected replicate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mllm.config import load_config
from eval.mllm.inputs import from_jsonl
from eval.mllm.prompts import LOCALIZATION_BBOX1000_PROTOCOL_VERSION
from eval.mllm.run_mllm import _manifest_payload, _write_json
from eval.mllm.schema import SchemaError, parse


def _as_bbox1000_response(raw: str) -> str:
    return raw.replace('"bbox_px"', '"bbox_1000"')


def _canonical_response(parsed: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    regions = []
    for region in parsed.get("regions", []):
        box = region.get("bbox_1000")
        if box is None:
            x1, y1, x2, y2 = region["bbox_px"]
            box = [
                x1 * 1000 / width,
                y1 * 1000 / height,
                x2 * 1000 / width,
                y2 * 1000 / height,
            ]
        regions.append({
            "bbox_1000": box,
            "confidence": region["confidence"],
            "evidence": region.get("evidence", ""),
        })
    return {
        "reasoning": parsed["reasoning"],
        "decision": parsed["decision"],
        "p_ai_edited": parsed["p_ai_edited"],
        "regions": regions,
    }


def _parse_legacy_success(row: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    width, height = row["image_size"]
    raw = row["raw_response"]
    try:
        parsed = parse(
            "localization",
            _as_bbox1000_response(raw),
            (width, height),
            "bbox_1000",
        )
        mode = "bbox_px_reinterpreted_as_bbox_1000"
    except SchemaError:
        # A very small number of legacy replies genuinely followed the old
        # pixel protocol (typically full-width seam boxes containing 1280).
        parsed = parse("localization", raw, (width, height), "bbox_px")
        mode = "legacy_bbox_px_preserved"
    canonical = _canonical_response(parsed, width, height)
    canonical_raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    canonical_parsed = parse(
        "localization",
        canonical_raw,
        (width, height),
        "bbox_1000",
    )
    return canonical_parsed, canonical_raw, mode


def _positive_bbox1000_attempt(attempt: dict[str, Any], image_size: list[int]) -> bool:
    if (
        attempt.get("status") != "schema_error"
        or "bbox_px is out of range" not in attempt.get("error", "")
        or not attempt.get("raw_response")
    ):
        return False
    try:
        parsed = parse(
            "localization",
            _as_bbox1000_response(attempt["raw_response"]),
            tuple(image_size),
            "bbox_1000",
        )
    except SchemaError:
        return False
    return parsed["decision"] == "localized_edit" and bool(parsed["regions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-slug", default="doubao_seed_2_1_pro_260628")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-root", type=Path, default=Path("results/mllm"))
    parser.add_argument(
        "--use-final-negative-fallback",
        action="store_true",
        help=(
            "retain the final no_localized_edit response when a prior positive "
            "attempt failed the legacy pixel-coordinate schema"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    cfg = load_config(args.config, {args.model_slug})
    model = cfg["models"][0]
    if model.get("localizationCoordinateSpace") != "bbox_1000":
        raise SystemExit(
            f"{args.model_slug} must configure localizationCoordinateSpace=bbox_1000"
        )
    model["concurrency"] = args.concurrency
    items = from_jsonl(args.list, root)

    folder = args.results_root / model["slug"]
    raw_path = folder / f"{args.run_id}.raw.jsonl"
    output_path = folder / f"{args.run_id}.jsonl"
    manifest_path = folder / f"{args.run_id}.run_manifest.json"
    report_path = folder / f"{args.run_id}.offline_repair_report.json"
    existing_outputs = [path for path in (raw_path, output_path, manifest_path, report_path) if path.exists()]
    if existing_outputs and not args.force:
        raise SystemExit(f"refusing to overwrite existing output: {existing_outputs[0]}")
    for path in existing_outputs:
        path.unlink()

    manifest_args = argparse.Namespace(
        condition=args.condition,
        source="list",
        review_export=None,
        review_status="good",
        include_source_pairs=False,
        list=args.list,
    )
    manifest = _manifest_payload(
        manifest_args,
        cfg,
        model,
        items,
        args.run_id,
        manifest_path,
        raw_path,
        output_path,
        ["localization"],
        {"localization": LOCALIZATION_BBOX1000_PROTOCOL_VERSION},
        None,
        "bbox_1000",
    )
    _write_json(manifest_path, manifest)

    last_success: dict[tuple[str, int], dict[str, Any]] = {}
    positive_out_of_range: set[tuple[str, int]] = set()
    for line in args.source_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("protocol_key") != "localization":
            continue
        key = (row["id"], int(row["replicate_index"]))
        if row.get("status") == "ok" and isinstance(row.get("parsed"), dict):
            last_success[key] = row
        if any(
            _positive_bbox1000_attempt(attempt, row["image_size"])
            for attempt in row.get("attempts", [])
        ):
            positive_out_of_range.add(key)

    expected_keys = {(item.id, replicate) for item in items for replicate in range(1, 4)}
    missing_source = expected_keys - set(last_success)
    if missing_source:
        raise SystemExit(
            f"source raw lacks {len(missing_source)} successful replicate(s); "
            f"first: {sorted(missing_source)[0]}"
        )

    flagged_keys = {
        key
        for key in positive_out_of_range
        if last_success[key]["parsed"]["decision"] == "no_localized_edit"
    }
    rerun_keys = set() if args.use_final_negative_fallback else flagged_keys
    seeded_rows = []
    modes: dict[str, int] = {}
    for key in sorted(expected_keys):
        if key in rerun_keys:
            continue
        source = last_success[key]
        if key in flagged_keys:
            width, height = source["image_size"]
            canonical = _canonical_response(source["parsed"], width, height)
            canonical_raw = json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
            )
            parsed = parse(
                "localization",
                canonical_raw,
                (width, height),
                "bbox_1000",
            )
            mode = "legacy_retry_final_negative_preserved"
        else:
            parsed, canonical_raw, mode = _parse_legacy_success(source)
        modes[mode] = modes.get(mode, 0) + 1
        row = dict(source)
        source_raw_id = source.get("raw_id")
        row.update({
            "run_id": args.run_id,
            "run_manifest_path": str(manifest_path),
            "input_manifest_sha256": manifest["input"]["manifest_sha256"],
            "config_fingerprint_sha256": manifest["config_fingerprint_sha256"],
            "condition": args.condition,
            "protocol_version": LOCALIZATION_BBOX1000_PROTOCOL_VERSION,
            "localization_coordinate_space": "bbox_1000",
            "raw_id": f"{args.run_id}:{source['id']}:localization:{source['replicate_index']}",
            "status": "ok",
            "parsed": parsed,
            "raw_response": canonical_raw,
            "attempts": [{
                "attempt": 0,
                "status": "offline_repaired",
                "source_raw_id": source_raw_id,
                "mode": mode,
            }],
            "latency_ms": 0,
            "offline_repair": {
                "source_run_id": source.get("run_id"),
                "source_raw_id": source_raw_id,
                "source_protocol_version": source.get("protocol_version"),
                "mode": mode,
            },
        })
        row.pop("protocol_suite_version", None)
        seeded_rows.append(row)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in seeded_rows
        ),
        encoding="utf-8",
    )
    report = {
        "run_id": args.run_id,
        "source_raw": str(args.source_raw),
        "source_successful_replicates": len(last_success),
        "seeded_replicates": len(seeded_rows),
        "flagged_replicates": len(flagged_keys),
        "flagged_images": len({key[0] for key in flagged_keys}),
        "fallback_negative_replicates": (
            len(flagged_keys) if args.use_final_negative_fallback else 0
        ),
        "rerun_replicates": len(rerun_keys),
        "rerun_images": len({key[0] for key in rerun_keys}),
        "repair_modes": modes,
        "flagged_keys": [
            {"id": image_id, "replicate_index": replicate}
            for image_id, replicate in sorted(flagged_keys)
        ],
        "rerun_keys": [
            {"id": image_id, "replicate_index": replicate}
            for image_id, replicate in sorted(rerun_keys)
        ],
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
