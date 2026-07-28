#!/usr/bin/env python3
"""Build the authoritative MLLM Balanced250 local750 + real250 ledger.

The selection is inherited verbatim from the Commercial API Balanced250
expected-input ledger.  This script materializes no images: it binds the raw
repository paths and SHA-256 values into the MLLM evaluation ledger.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "claimforge_mllm_benchmark1000_v2"
ROW_SCHEMA_VERSION = "claimforge_mllm_benchmark_image_v2"
DATASET_ID = "claimforge-mllm-balanced250-local750-real250-v2"
SOURCE_DATASET_ID = "claimforge-balanced250-independent-panel-jpeg-q95-v1"
DEFAULT_SOURCE = Path(
    "results/opensource/community_forensics/"
    "community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/"
    "expected_inputs.jsonl"
)
DEFAULT_LEDGER = Path(
    "annotations/claimforge_mllm_benchmark1000_v2.jsonl"
)
DEFAULT_MANIFEST = Path(
    "annotations/claimforge_mllm_benchmark1000_v2.manifest.json"
)
CONDITIONS = ("real", "local_mouse", "local_cat", "local_trash_can")
CANDIDATE = {
    "real": "real",
    "local_mouse": "mouse",
    "local_cat": "cat",
    "local_trash_can": "trash_can",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _load_source(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if row.get("panel") is True and row.get("condition") in CONDITIONS
    ]
    counts = collections.Counter(str(row["condition"]) for row in selected)
    expected = {condition: 250 for condition in CONDITIONS}
    if dict(counts) != expected:
        raise ValueError(
            f"Balanced250 selection mismatch: {dict(counts)} != {expected}"
        )
    if len({str(row["raw_sha256"]) for row in selected}) != 1000:
        raise ValueError("Balanced250 MLLM scope must contain 1000 unique images")
    return sorted(
        selected,
        key=lambda row: (
            CONDITIONS.index(str(row["condition"])),
            int(row["selection_rank"]),
            str(row["sample_id"]),
        ),
    )


def _row(source: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    condition = str(source["condition"])
    candidate = CANDIDATE[condition]
    raw_path = Path(str(source["raw_path"]))
    source_path = Path(str(source["matched_source_raw_path"]))
    resolved_raw = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved_source = (
        source_path if source_path.is_absolute() else repo_root / source_path
    )
    if not resolved_raw.is_file():
        raise FileNotFoundError(resolved_raw)
    if not resolved_source.is_file():
        raise FileNotFoundError(resolved_source)
    if _sha256(resolved_raw) != source["raw_sha256"]:
        raise ValueError(f"raw SHA-256 mismatch: {raw_path}")
    if _sha256(resolved_source) != source["matched_source_raw_sha256"]:
        raise ValueError(f"source SHA-256 mismatch: {source_path}")

    sample_id = str(source["sample_id"])
    task_id = str(source["task_id"])
    benchmark_id = f"balanced250__{condition}__{sample_id}"
    return {
        "benchmark_id": benchmark_id,
        "candidate": candidate,
        "condition": condition,
        "context_region_xyxy": source.get("context_region_xyxy"),
        "edit_region_xyxy": source.get("edit_region_xyxy"),
        "id": benchmark_id,
        "image_bytes": int(source["raw_bytes"]),
        "image_path": str(raw_path),
        "image_sha256": str(source["raw_sha256"]),
        "image_size": {
            "height": int(source["height"]),
            "width": int(source["width"]),
        },
        "label": "real" if condition == "real" else "forged",
        "metadata": {
            "balanced250_sample_id": sample_id,
            "candidate": candidate,
            "condition": condition,
            "dataset_id": DATASET_ID,
            "panel": True,
            "selection_rank": int(source["selection_rank"]),
            "source_dataset_id": str(source["dataset_id"]),
            "source_release_sample_id": source.get("source_release_sample_id"),
        },
        "scene_id": str(source["normalized_task_id"]),
        "schema_version": ROW_SCHEMA_VERSION,
        "source_image": str(source_path),
        "source_sha256": str(source["matched_source_raw_sha256"]),
        "task_id": task_id,
    }


def build(
    repo_root: Path,
    source_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    source = source_path if source_path.is_absolute() else repo_root / source_path
    ledger = ledger_path if ledger_path.is_absolute() else repo_root / ledger_path
    manifest = (
        manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    )
    selected = _load_source(source)
    rows = [_row(item, repo_root) for item in selected]
    _dump_jsonl(ledger, rows)

    candidate_counts = collections.Counter(row["candidate"] for row in rows)
    label_counts = collections.Counter(row["label"] for row in rows)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "release_ready": True,
        "counts": {
            "total": len(rows),
            "forged": label_counts["forged"],
            "real": label_counts["real"],
            "mouse": candidate_counts["mouse"],
            "cat": candidate_counts["cat"],
            "trash_can": candidate_counts["trash_can"],
        },
        "policy": {
            "selection_source": SOURCE_DATASET_ID,
            "selection": (
                "Commercial API Balanced250 panel=true rows; raw image paths; "
                "real + local_mouse + local_cat + local_trash_can only"
            ),
            "aggregation_must_filter_to_ledger_sha256": True,
            "image_materialization_required": False,
            "localization_strict_missing_policy": (
                "Every forged sample is in the denominator. Missing, invalid, "
                "empty, or out-of-bounds bbox output is a localization miss."
            ),
        },
        "built_from": {
            "path": str(source_path),
            "rows": sum(
                1
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "sha256": _sha256(source),
        },
        "formal_ledger": {
            "path": str(ledger_path),
            "rows": len(rows),
            "sha256": _sha256(ledger),
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the authoritative Balanced250 MLLM benchmark1000 ledger"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = build(
        args.repo_root.resolve(),
        args.source,
        args.ledger,
        args.manifest,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
