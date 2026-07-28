#!/usr/bin/env python3
"""Run Copyleaks Ultra on the missing CLAIMFORGE Balanced250 cells."""

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
from PIL import Image, ImageChops

from eval.commercial.run_alibaba_balanced250 import (
    DEFAULT_CONDITIONS,
    DEFAULT_INPUTS,
    SelectedInput,
    load_inputs,
    parse_conditions,
    read_jsonl,
    selection_digest,
)
from eval.commercial.run_copyleaks import (
    DEFAULT_ENDPOINT_TEMPLATE,
    DEFAULT_LOGIN_ENDPOINT,
    DEFAULT_MODEL,
    binary_overlap,
    binary_pixel_count,
    box_overlap,
    canonicalize_png,
    classify,
    decode_rle_mask,
    login,
    scaled_box,
)
from eval.commercial.run_illuminarty import (
    append_jsonl,
    read_latest,
    sha256_file,
    utc_now,
)


DEFAULT_OUTPUT = Path(
    "results/commercial/copyleaks/"
    "claimforge_balanced250_missing1250_canonical_png_20260727.jsonl"
)


def selected_metadata(
    ledger_path: Path,
    selected: list[SelectedInput],
) -> dict[str, dict[str, Any]]:
    wanted = {entry.item.id for entry in selected}
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(ledger_path):
        identifier = f"{row.get('condition')}/{row.get('sample_id')}"
        if identifier in wanted:
            metadata[identifier] = row
    missing = sorted(wanted - set(metadata))
    if missing:
        raise ValueError(f"missing ledger metadata for {missing[:5]}")
    return metadata


def ensure_run_manifest(
    output_path: Path,
    selected: list[SelectedInput],
    metadata: dict[str, dict[str, Any]],
    ledger_path: Path,
    repo_root: Path,
    login_endpoint: str,
    endpoint_template: str,
    model: str,
    run_id: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    conditions = list(dict.fromkeys(entry.condition for entry in selected))
    expected = {
        "schema_version": "copyleaks_balanced250_run_manifest_v1",
        "run_id": run_id,
        "dataset_id": "claimforge-balanced250-independent-panel-jpeg-q95-v1",
        "selection": "panel=true",
        "conditions": conditions,
        "per_condition": len(selected) // len(conditions),
        "expected_images": len(selected),
        "input_digest": selection_digest(selected),
        "input_ledger": ledger_path.relative_to(repo_root).as_posix(),
        "input_ledger_sha256": sha256_file(ledger_path),
        "login_endpoint": login_endpoint,
        "endpoint_template": endpoint_template,
        "model": model,
        "sandbox": False,
        "upload": {
            "format": "PNG",
            "color_mode": "RGB",
            "metadata": "stripped",
            "minimum_side": 512,
            "maximum_width": 6000,
            "maximum_height": 4500,
            "maximum_pixels": 27_000_000,
            "resize_filter": "Lanczos",
            "filename": "image.png",
        },
        "decision": "vendor isAiDetected boolean",
        "localization": {
            "prediction": "vendor native row-major zero-based RLE",
            "ground_truth": (
                "frozen Balanced250 exact-diff mask where available; otherwise "
                "reconstructed from the hash-verified raw source/forged pair and "
                "checked against frozen pixel-count and bbox metadata"
            ),
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
        "adapter": {
            "path": Path(__file__).relative_to(repo_root).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "dependencies": {
            "requests": requests.__version__,
            "pillow": Image.__version__,
        },
        "ordered_inputs": [
            {
                "rank": rank,
                "id": entry.item.id,
                "sample_id": entry.sample_id,
                "task_id": entry.item.task_id,
                "domain": entry.item.domain,
                "kind": entry.item.kind,
                "condition": entry.condition,
                "condition_family": entry.condition_family,
                "selection_rank": entry.selection_rank,
                "source_content_cluster": entry.source_content_cluster,
                "image_path": entry.item.relative_path,
                "image_sha256": entry.item.sha256,
                "file_bytes": entry.item.file_bytes,
                "gt_mask_kind": metadata[entry.item.id].get("gt_mask_kind"),
                "gt_mask_path": metadata[entry.item.id].get("gt_mask_path"),
                "gt_mask_sha256": metadata[entry.item.id].get("gt_mask_sha256"),
            }
            for rank, entry in enumerate(selected)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def add_balanced250_localization(
    row: dict[str, Any],
    metadata: dict[str, Any],
    repo_root: Path,
) -> None:
    width = int(row["rle_width"])
    height = int(row["rle_height"])
    upload_size = tuple(int(value) for value in row["upload_size"])
    if (width, height) != upload_size:
        raise ValueError(f"provider shape {(width, height)} != upload {upload_size}")

    rle = row["rle"]
    prediction = decode_rle_mask(
        rle["starts"],
        rle["lengths"],
        width,
        height,
    )
    predicted_pixels = binary_pixel_count(prediction)
    localization: dict[str, Any] = {
        "predicted_pixels": predicted_pixels,
        "predicted_fraction": predicted_pixels / (width * height),
        "predicted_bbox_xyxy": (
            list(prediction.getbbox()) if prediction.getbbox() else None
        ),
    }

    original_size = (int(metadata["width"]), int(metadata["height"]))
    for name, field in (
        ("edit_box", "edit_region_xyxy"),
        ("context_box", "context_region_xyxy"),
    ):
        box = scaled_box(metadata.get(field), original_size, upload_size)
        if box is not None:
            localization[name] = box_overlap(prediction, box)

    if metadata.get("gt_mask_kind") == "exact_diff":
        relative = str(metadata["gt_mask_path"])
        mask_path = (repo_root / relative).resolve()
        mask_path.relative_to(repo_root)
        if mask_path.is_file():
            if sha256_file(mask_path) != str(metadata["gt_mask_sha256"]):
                raise ValueError(f"GT mask SHA-256 mismatch: {relative}")
            with Image.open(mask_path) as opened:
                target = opened.convert("L").point(
                    lambda value: 255 if value > 0 else 0,
                    mode="L",
                )
            target_origin = "frozen_mask"
        else:
            source_relative = str(metadata["matched_source_raw_path"])
            source_path = (repo_root / source_relative).resolve()
            forged_path = (repo_root / str(metadata["raw_path"])).resolve()
            source_path.relative_to(repo_root)
            forged_path.relative_to(repo_root)
            if sha256_file(source_path) != str(
                metadata["matched_source_raw_sha256"]
            ):
                raise ValueError(f"source SHA-256 mismatch: {source_relative}")
            with (
                Image.open(source_path) as source_opened,
                Image.open(forged_path) as forged_opened,
            ):
                source = source_opened.convert("RGB")
                forged = forged_opened.convert("RGB")
                if source.size != forged.size:
                    raise ValueError(
                        f"source/forged size mismatch: "
                        f"{source.size} != {forged.size}"
                    )
                red, green, blue = ImageChops.difference(source, forged).split()
                maximum = ImageChops.lighter(
                    red,
                    ImageChops.lighter(green, blue),
                )
                target = maximum.point(
                    lambda value: 255 if value > 0 else 0,
                    mode="L",
                )
            expected_pixels = int(metadata["gt_positive_pixels"])
            actual_pixels = binary_pixel_count(target)
            expected_bbox = tuple(int(value) for value in metadata["gt_bbox_xyxy"])
            if actual_pixels != expected_pixels or target.getbbox() != expected_bbox:
                raise ValueError(
                    f"reconstructed GT mismatch for {relative}: "
                    f"pixels {actual_pixels} != {expected_pixels}, "
                    f"bbox {target.getbbox()} != {expected_bbox}"
                )
            target_origin = "reconstructed_exact_diff_from_verified_raw_pair"
        if target.size != original_size:
            raise ValueError(
                f"GT mask size mismatch for {relative}: "
                f"{target.size} != {original_size}"
            )
        if target.size != upload_size:
            target = target.resize(upload_size, Image.Resampling.NEAREST)
        localization["pixel_diff_gt"] = {
            "mask_path": relative,
            "mask_sha256": metadata["gt_mask_sha256"],
            "mask_file_available": mask_path.is_file(),
            "target_origin": target_origin,
            "target_bbox_xyxy": (
                list(target.getbbox()) if target.getbbox() else None
            ),
            **binary_overlap(prediction, target),
        }

    row["localization"] = localization


def score_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["ai_score"])
        for row in rows
        if isinstance(row.get("ai_score"), (int, float))
    ]
    detected = sum(bool(row.get("is_ai_detected")) for row in rows)
    return {
        "count": len(values),
        "detected": detected,
        "detection_rate": detected / len(rows) if rows else None,
        "score": {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        },
    }


def write_summary(
    output_path: Path,
    selected: list[SelectedInput],
    run_id: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    expected_by_id = {entry.item.id: entry for entry in selected}
    rows = [latest[identifier] for identifier in expected_by_id if identifier in latest]
    valid = [row for row in rows if row.get("status") == "ok"]
    conditions: dict[str, Any] = {}
    for condition in dict.fromkeys(entry.condition for entry in selected):
        identifiers = {
            entry.item.id for entry in selected if entry.condition == condition
        }
        condition_rows = [row for row in rows if row["id"] in identifiers]
        condition_valid = [
            row for row in condition_rows if row.get("status") == "ok"
        ]
        conditions[condition] = {
            "expected": len(identifiers),
            "completed": len(condition_rows),
            "valid": len(condition_valid),
            "errors": len(condition_rows) - len(condition_valid),
            **score_stats(condition_valid),
        }

    summary = {
        "schema_version": "copyleaks_balanced250_summary_v1",
        "run_id": run_id,
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "expected_images": len(selected),
        "completed_images": len(rows),
        "valid_images": len(valid),
        "error_images": len(rows) - len(valid),
        "remaining_images": len(selected) - len(valid),
        "actual_credits": sum(
            float(row["actual_credits"])
            for row in valid
            if isinstance(row.get("actual_credits"), (int, float))
        ),
        "conditions": conditions,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=DEFAULT_CONDITIONS,
        help="comma-separated canonical condition names",
    )
    parser.add_argument("--per-condition", type=int, default=250)
    parser.add_argument("--login-endpoint", default=DEFAULT_LOGIN_ENDPOINT)
    parser.add_argument("--endpoint-template", default=DEFAULT_ENDPOINT_TEMPLATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--minimum-interval", type=float, default=0.1)
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--run-id",
        default="copyleaks_balanced250_missing1250_20260727",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.per_condition < 1:
        parser.error("--per-condition must be positive")
    if args.max_attempts < 1 or args.minimum_interval < 0:
        parser.error("max attempts must be positive and interval non-negative")
    if args.max_pending is not None and args.max_pending < 1:
        parser.error("--max-pending must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if "{scan_id}" not in args.endpoint_template:
        parser.error("--endpoint-template must contain {scan_id}")

    repo_root = args.repo_root.resolve()
    ledger_path = (
        args.inputs if args.inputs.is_absolute() else repo_root / args.inputs
    ).resolve()
    output_path = (
        args.output if args.output.is_absolute() else repo_root / args.output
    ).resolve()
    selected = load_inputs(
        repo_root,
        ledger_path,
        args.conditions,
        args.per_condition,
    )
    metadata = selected_metadata(ledger_path, selected)
    latest = read_latest(output_path)
    pending = [
        entry
        for entry in selected
        if latest.get(entry.item.id, {}).get("status") != "ok"
    ]
    pending_total = len(pending)
    if args.max_pending is not None:
        pending = pending[: args.max_pending]
    print(
        json.dumps(
            {
                "conditions": args.conditions,
                "expected_images": len(selected),
                "already_valid": len(selected) - pending_total,
                "pending": pending_total,
                "scheduled": len(pending),
                "model": args.model,
                "sandbox": False,
                "output": output_path.relative_to(repo_root).as_posix(),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
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
        selected,
        metadata,
        ledger_path,
        repo_root,
        args.login_endpoint,
        args.endpoint_template,
        args.model,
        args.run_id,
    )

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "claimforge-benchmark/copyleaks-balanced250-v1"}
    )
    token = login(
        session,
        args.login_endpoint,
        email,
        api_key,
        (args.connect_timeout, args.read_timeout),
    )
    digest = selection_digest(selected)
    with tempfile.TemporaryDirectory(
        prefix="claimforge-copyleaks-balanced250-"
    ) as temporary:
        temporary_dir = Path(temporary)
        for index, entry in enumerate(pending, start=1):
            upload_path = temporary_dir / f"upload-{index:04d}.png"
            upload = canonicalize_png(entry.item.path, upload_path)
            row = classify(
                session,
                entry.item,
                upload_path,
                upload,
                args.endpoint_template,
                token,
                args.model,
                False,
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                args.run_id,
                digest,
            )
            row.update(
                {
                    "schema_version": "copyleaks_balanced250_result_v1",
                    "sample_id": entry.sample_id,
                    "condition": entry.condition,
                    "condition_family": entry.condition_family,
                    "selection_rank": entry.selection_rank,
                    "source_content_cluster": entry.source_content_cluster,
                }
            )
            if row["status"] == "ok":
                add_balanced250_localization(
                    row,
                    metadata[entry.item.id],
                    repo_root,
                )
            append_jsonl(output_path, row)

            if (
                index == 1
                or index % args.progress_every == 0
                or index == len(pending)
                or row["status"] != "ok"
            ):
                print(
                    json.dumps(
                        {
                            "scheduled_progress": f"{index}/{len(pending)}",
                            "condition": entry.condition,
                            "id": entry.item.id,
                            "status": row["status"],
                            "is_ai_detected": row.get("is_ai_detected"),
                            "ai_score": row.get("ai_score"),
                            "actual_credits": row.get("actual_credits"),
                            "error": (
                                (row.get("attempts") or [{}])[-1].get(
                                    "error_message"
                                )
                                if row["status"] == "error"
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if row["status"] == "error":
                final = (row.get("attempts") or [{}])[-1]
                if final.get("http_status") in {401, 402, 403, 429}:
                    break
            if index < len(pending):
                time.sleep(args.minimum_interval)

    latest = read_latest(output_path)
    unexpected = sorted(set(latest) - {entry.item.id for entry in selected})
    if unexpected:
        raise ValueError(f"result file contains unexpected IDs: {unexpected[:5]}")
    summary = write_summary(output_path, selected, args.run_id)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
