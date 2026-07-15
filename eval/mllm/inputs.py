"""Blind input adapters: labels stay local and are never sent to models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


def manifest_hash(items: list[ImageItem]) -> str:
    serial = [{"id": x.id, "path": str(x.image_path) if x.image_path else None, "url": x.image_url} for x in items]
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
