"""Blind input adapters: labels stay local and are never sent to models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BENCHMARK1000_DATASET_ID = "claimforge-mllm-local750-real250-v1"
DEFAULT_BENCHMARK1000_MANIFEST = Path(
    "annotations/claimforge_mllm_benchmark1000_v1.manifest.json"
)
DEFAULT_BENCHMARK1000_LEDGER = Path(
    "annotations/claimforge_mllm_benchmark1000_v1.jsonl"
)


@dataclass(frozen=True)
class ImageItem:
    id: str
    image_path: Path | None
    image_url: str | None
    task_id: str | None = None
    label: str | None = None
    mask_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def from_review_export(export_path: Path, root: Path, status: str, include_source_pairs: bool) -> list[ImageItem]:
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("review export must contain a records array")
    items: list[ImageItem] = []
    seen_real: set[Path] = set()
    for row in records:
        if row.get("status") != status:
            continue
        task_id = str(row["task_id"])
        forged = _resolve(root, str(row["spliced_image"]))
        source = _resolve(root, str(row["source_image"]))
        if not forged.is_file() or not source.is_file() or forged == source:
            raise FileNotFoundError(f"invalid review pair for {task_id}: forged={forged}, source={source}")
        items.append(ImageItem(f"{task_id}__forged", forged, None, task_id, "forged", metadata={"review_status": status}))
        if include_source_pairs and source not in seen_real:
            items.append(ImageItem(f"{task_id}__real", source, None, task_id, "real", metadata={"review_status": status}))
            seen_real.add(source)
    if not items:
        raise ValueError(f"no review records with status={status!r}")
    return items


def from_jsonl(path: Path, root: Path) -> list[ImageItem]:
    items: list[ImageItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("id") or not (row.get("image_path") or row.get("image_url")):
            raise ValueError(f"{path}:{number} requires id and image_path or image_url")
        image_path = _resolve(root, row["image_path"]) if row.get("image_path") else None
        if image_path is not None and not image_path.is_file():
            raise FileNotFoundError(image_path)
        items.append(ImageItem(str(row["id"]), image_path, row.get("image_url"), row.get("task_id"), row.get("label"), row.get("mask_path"), row.get("metadata") or {}))
    return items


def from_benchmark1000(
    manifest_path: Path,
    ledger_path: Path,
    root: Path,
) -> list[ImageItem]:
    """Load the immutable 750-forged + 250-real aggregation ledger."""
    resolved_manifest = _resolve(root, str(manifest_path))
    if not resolved_manifest.is_file():
        raise FileNotFoundError(resolved_manifest)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != BENCHMARK1000_DATASET_ID:
        raise ValueError("unexpected benchmark1000 dataset_id")
    counts = manifest.get("counts") or {}
    if not manifest.get("release_ready"):
        raise ValueError("benchmark1000 is not release-ready")
    expected_counts = {
        "total": 1000,
        "forged": 750,
        "real": 250,
        "trash_can": 250,
        "mouse": 250,
        "cat": 250,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise ValueError(
                f"benchmark1000 count mismatch: {key}="
                f"{counts.get(key)!r}, expected {expected}"
            )

    resolved_ledger = _resolve(root, str(ledger_path))
    if not resolved_ledger.is_file():
        raise FileNotFoundError(resolved_ledger)
    formal_ledger = manifest.get("formal_ledger") or {}
    bound_path = formal_ledger.get("path")
    if not isinstance(bound_path, str):
        raise ValueError("benchmark1000 manifest lacks formal ledger path")
    if _resolve(root, bound_path) != resolved_ledger:
        raise ValueError(
            "benchmark1000 ledger path does not match manifest"
        )
    raw = resolved_ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest() != formal_ledger.get("sha256"):
        raise ValueError(
            "benchmark1000 ledger SHA-256 does not match manifest"
        )
    if formal_ledger.get("rows") != 1000:
        raise ValueError("benchmark1000 ledger row binding is not 1000")
    raw_rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        raw.decode("utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        if metadata.get("dataset_id") != BENCHMARK1000_DATASET_ID:
            raise ValueError(
                f"{resolved_ledger}:{number}: unexpected dataset_id"
            )
        raw_rows.append(row)
    if len(raw_rows) != 1000:
        raise ValueError(
            f"benchmark1000 ledger has {len(raw_rows)} rows, expected 1000"
        )
    ids = [str(row.get("id", "")) for row in raw_rows]
    if len(set(ids)) != 1000 or any(not value for value in ids):
        raise ValueError("benchmark1000 IDs must be 1000 unique values")
    category_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in raw_rows:
        category = str(row.get("candidate", ""))
        category_counts[category] = category_counts.get(category, 0) + 1
        label = str(row.get("label", ""))
        label_counts[label] = label_counts.get(label, 0) + 1
    if category_counts != {
        "mouse": 250,
        "cat": 250,
        "trash_can": 250,
        "real": 250,
    }:
        raise ValueError(
            "benchmark1000 candidate counts mismatch: "
            f"{category_counts}"
        )
    if label_counts != {"forged": 750, "real": 250}:
        raise ValueError(
            f"benchmark1000 label counts mismatch: {label_counts}"
        )
    return from_jsonl(resolved_ledger, root)


def manifest_hash(items: list[ImageItem]) -> str:
    serial = [{"id": x.id, "path": str(x.image_path) if x.image_path else None, "url": x.image_url} for x in items]
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
