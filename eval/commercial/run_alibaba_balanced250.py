#!/usr/bin/env python3
"""Run Alibaba Cloud Ultra on the missing CLAIMFORGE Balanced250 cells.

The default selection contains the independent real250 panel plus the local
and full-frame cat/trash-can cells. Existing mouse runs are intentionally not
submitted again.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oss2
from alibabacloud_green20220302.client import Client
from alibabacloud_tea_openapi.models import Config
from PIL import Image

from eval.commercial.run_alibaba import (
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    DEFAULT_SERVICE,
    TemporaryUploader,
    classify,
)
from eval.commercial.run_hive import canonicalize
from eval.commercial.run_illuminarty import (
    ImageItem,
    append_jsonl,
    input_digest,
    read_latest,
    sha256_file,
    utc_now,
)


DEFAULT_INPUTS = Path(
    "results/opensource/cnndetection/"
    "cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/"
    "expected_inputs.jsonl"
)
DEFAULT_OUTPUT = Path(
    "results/commercial/alibaba/"
    "claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl"
)
DEFAULT_CONDITIONS = (
    "real",
    "local_cat",
    "local_trash_can",
    "fullframe_cat",
    "fullframe_trash_can",
)
RISK_LABELS = ("risk_aigc", "risk_fake", "risk_edit")


@dataclass(frozen=True)
class SelectedInput:
    item: ImageItem
    condition: str
    condition_family: str
    sample_id: str
    canonical_sha256: str
    source_content_cluster: str
    selection_rank: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"input path escapes repository: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing input image: {relative_path}")
    return path


def load_inputs(
    repo_root: Path,
    ledger_path: Path,
    conditions: tuple[str, ...],
    per_condition: int,
) -> list[SelectedInput]:
    selected: list[SelectedInput] = []
    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    for row in read_jsonl(ledger_path):
        condition = str(row.get("condition") or "")
        if condition in rows_by_condition and row.get("panel") is True:
            rows_by_condition[condition].append(row)

    for condition in conditions:
        rows = sorted(
            rows_by_condition[condition],
            key=lambda row: (int(row["rank"]), str(row["sample_id"])),
        )
        if len(rows) < per_condition:
            raise ValueError(
                f"requested {per_condition} {condition} rows, found {len(rows)}"
            )
        for row in rows[:per_condition]:
            sample_id = str(row["sample_id"])
            task_id = str(row["task_id"])
            relative_path = str(row["raw_path"])
            path = resolve_repo_path(repo_root, relative_path)
            actual_sha256 = sha256_file(path)
            expected_raw_sha256 = str(row["raw_sha256"])
            if actual_sha256 != expected_raw_sha256:
                raise ValueError(
                    f"{condition}/{sample_id}: raw SHA-256 mismatch "
                    f"({actual_sha256} != {expected_raw_sha256})"
                )
            kind = "real" if condition == "real" else "forged"
            selected.append(
                SelectedInput(
                    item=ImageItem(
                        id=f"{condition}/{sample_id}",
                        task_id=task_id,
                        domain=str(row["domain"]),
                        kind=kind,
                        label="not_edited" if kind == "real" else "edited",
                        path=path,
                        relative_path=relative_path,
                        image_size=(int(row["width"]), int(row["height"])),
                        sha256=actual_sha256,
                        file_bytes=path.stat().st_size,
                    ),
                    condition=condition,
                    condition_family=str(row["condition_family"]),
                    sample_id=sample_id,
                    canonical_sha256=str(row["canonical_sha256"]),
                    source_content_cluster=str(row["source_content_cluster"]),
                    selection_rank=int(row["selection_rank"]),
                )
            )

    ids = [entry.item.id for entry in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected input IDs are not unique")
    return selected


def selection_digest(selected: list[SelectedInput]) -> str:
    return input_digest([entry.item for entry in selected])


def ensure_run_manifest(
    output_path: Path,
    selected: list[SelectedInput],
    ledger_path: Path,
    repo_root: Path,
    endpoint: str,
    region: str,
    service: str,
    quality: int,
    run_id: str,
) -> None:
    path = output_path.with_suffix(".run_manifest.json")
    conditions = list(dict.fromkeys(entry.condition for entry in selected))
    expected = {
        "schema_version": "alibaba_ultra_balanced250_run_manifest_v1",
        "run_id": run_id,
        "dataset_id": "claimforge-balanced250-independent-panel-jpeg-q95-v1",
        "selection": "panel=true",
        "conditions": conditions,
        "per_condition": len(selected) // len(conditions),
        "expected_images": len(selected),
        "input_digest": selection_digest(selected),
        "input_ledger": ledger_path.relative_to(repo_root).as_posix(),
        "input_ledger_sha256": sha256_file(ledger_path),
        "endpoint": endpoint,
        "region": region,
        "service": service,
        "upload": {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "resize": False,
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
            "alibabacloud_green20220302": importlib.metadata.version(
                "alibabacloud_green20220302"
            ),
            "oss2": getattr(oss2, "__version__", None),
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
                "canonical_sha256": entry.canonical_sha256,
                "file_bytes": entry.item.file_bytes,
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


def any_risk(row: dict[str, Any]) -> bool:
    return any(bool(row.get(f"{label}_detected")) for label in RISK_LABELS)


def write_summary(
    output_path: Path,
    selected: list[SelectedInput],
    run_id: str,
) -> dict[str, Any]:
    latest = read_latest(output_path)
    by_id = {entry.item.id: entry for entry in selected}
    rows = [latest[item_id] for item_id in by_id if item_id in latest]
    valid = [row for row in rows if row.get("status") == "ok"]
    conditions: dict[str, Any] = {}
    for condition in dict.fromkeys(entry.condition for entry in selected):
        expected_ids = {
            entry.item.id for entry in selected if entry.condition == condition
        }
        condition_rows = [row for row in rows if row["id"] in expected_ids]
        condition_valid = [
            row for row in condition_rows if row.get("status") == "ok"
        ]
        positives = sum(any_risk(row) for row in condition_valid)
        conditions[condition] = {
            "expected": len(expected_ids),
            "completed": len(condition_rows),
            "valid": len(condition_valid),
            "errors": len(condition_rows) - len(condition_valid),
            "any_risk_positive": positives,
            "any_risk_rate": positives / len(condition_valid)
            if condition_valid
            else None,
            "label_positive": {
                label: sum(
                    bool(row.get(f"{label}_detected")) for row in condition_valid
                )
                for label in RISK_LABELS
            },
        }

    confidence_values = {
        label: [
            float(row[f"{label}_confidence"])
            for row in valid
            if isinstance(row.get(f"{label}_confidence"), (int, float))
        ]
        for label in RISK_LABELS
    }
    summary = {
        "schema_version": "alibaba_ultra_balanced250_summary_v1",
        "run_id": run_id,
        "generated_at": utc_now(),
        "results_path": output_path.as_posix(),
        "expected_images": len(selected),
        "completed_images": len(rows),
        "valid_images": len(valid),
        "error_images": len(rows) - len(valid),
        "remaining_images": len(selected) - len(valid),
        "estimated_successful_call_cost_cny": round(len(valid) * 0.02, 2),
        "canonical_hash_mismatches": sum(
            row.get("upload_matches_expected_canonical") is False for row in rows
        ),
        "conditions": conditions,
        "confidence": {
            label: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for label, values in confidence_values.items()
        },
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


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(part.strip() for part in value.split(",") if part.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    if len(conditions) != len(set(conditions)):
        raise argparse.ArgumentTypeError("conditions must be unique")
    return conditions


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
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--minimum-interval", type=float, default=0.25)
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--allow-canonical-mismatch",
        action="store_true",
        help=(
            "continue when the current Pillow/libjpeg Q95 encoding is not "
            "byte-identical to the frozen canonical JPEG"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="alibaba_ultra_balanced250_missing1250_20260727",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.per_condition < 1:
        parser.error("--per-condition must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.max_pending is not None and args.max_pending < 1:
        parser.error("--max-pending must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.jpeg_quality != 95:
        parser.error("Balanced250 requires --jpeg-quality 95")

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
                "output": output_path.relative_to(repo_root).as_posix(),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
    if not access_key_id or not access_key_secret:
        raise SystemExit(
            "ALIBABA_CLOUD_ACCESS_KEY_ID and "
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET must be set"
        )

    ensure_run_manifest(
        output_path,
        selected,
        ledger_path,
        repo_root,
        args.endpoint,
        args.region,
        args.service,
        args.jpeg_quality,
        args.run_id,
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
    selected_by_id = {entry.item.id: entry for entry in selected}

    with tempfile.TemporaryDirectory(prefix="claimforge-alibaba-balanced250-") as temp:
        temporary_dir = Path(temp)
        for index, entry in enumerate(pending, start=1):
            upload_path = temporary_dir / f"upload-{index:04d}.jpg"
            upload = canonicalize(entry.item.path, upload_path, args.jpeg_quality)
            canonical_match = upload["upload_sha256"] == entry.canonical_sha256
            if not canonical_match and not args.allow_canonical_mismatch:
                raise ValueError(
                    f"{entry.item.id}: canonical SHA-256 mismatch "
                    f"({upload['upload_sha256']} != {entry.canonical_sha256})"
                )
            row = classify(
                client,
                uploader,
                entry.item,
                upload_path,
                {
                    **upload,
                    "expected_canonical_sha256": entry.canonical_sha256,
                    "upload_matches_expected_canonical": canonical_match,
                },
                args.service,
                args.run_id,
                selection_digest(selected),
                args.max_attempts,
                access_key_id,
                access_key_secret,
            )
            row.update(
                {
                    "schema_version": "alibaba_ultra_balanced250_result_v1",
                    "sample_id": entry.sample_id,
                    "condition": entry.condition,
                    "condition_family": entry.condition_family,
                    "selection_rank": entry.selection_rank,
                    "source_content_cluster": entry.source_content_cluster,
                }
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
                            "risk_level": row.get("provider_risk_level"),
                            "labels": row.get("provider_labels"),
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
                break
            if index < len(pending):
                time.sleep(args.minimum_interval)

    # Catch accidental output IDs outside the immutable run selection.
    latest = read_latest(output_path)
    unexpected = sorted(set(latest) - set(selected_by_id))
    if unexpected:
        raise ValueError(f"result file contains unexpected IDs: {unexpected[:5]}")
    summary = write_summary(output_path, selected, args.run_id)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
