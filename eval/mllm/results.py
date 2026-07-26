from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ProtocolVersionSelector = str | Mapping[str, str] | None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _protocol_key(row: dict[str, Any]) -> str:
    value = row.get("protocol_key") or row.get("protocol_id")
    if not isinstance(value, str):
        raise KeyError("protocol_key")
    return value


def protocol_version_matches(
    row: dict[str, Any],
    protocol_version: ProtocolVersionSelector,
) -> bool:
    if protocol_version is None:
        return True
    if isinstance(protocol_version, str):
        return row.get("protocol_version") == protocol_version
    expected = protocol_version.get(_protocol_key(row))
    return expected is not None and row.get("protocol_version") == expected


def completed_raw_keys(
    path: Path,
    protocol_version: ProtocolVersionSelector = None,
) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok" and protocol_version_matches(row, protocol_version):
                keys.add((row["id"], _protocol_key(row), int(row["replicate_index"])))
    return keys


def successful_raw(
    path: Path,
    protocol_version: ProtocolVersionSelector = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load parsed successful replicates so --resume can still aggregate them."""
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and protocol_version_matches(row, protocol_version) and isinstance(row.get("parsed"), dict):
            rows[(row["id"], _protocol_key(row), int(row["replicate_index"]))] = row["parsed"]
    return rows


def completed_aggregate_keys(
    path: Path,
    protocol_version: ProtocolVersionSelector = None,
) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {
        (row["id"], _protocol_key(row))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("status") == "ok" and protocol_version_matches(row, protocol_version)
    }
