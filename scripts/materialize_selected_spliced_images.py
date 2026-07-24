#!/usr/bin/env python3
"""Materialize manually selected splice results into one final image directory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def repo_path(path: Path) -> Path:
    absolute = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    if not absolute.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {path}")
    return absolute


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def materialize(args: argparse.Namespace) -> dict:
    selection_path = repo_path(args.selection_json)
    output_dir = repo_path(args.output_dir)
    if output_dir == REPO:
        raise ValueError("output directory cannot be the repository root")

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selections = payload.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("selection payload must contain a non-empty selections list")

    accepted: list[dict] = []
    rejected = 0
    selection_counts: Counter[str] = Counter()
    task_ids: set[str] = set()
    output_names: set[str] = set()

    for item in selections:
        task_id = str(item.get("task_id") or "")
        selection = str(item.get("selection") or "")
        if not task_id or not SAFE_TASK_ID.fullmatch(task_id):
            raise ValueError(f"unsafe task id: {task_id!r}")
        if task_id in task_ids:
            raise ValueError(f"duplicate task id: {task_id}")
        task_ids.add(task_id)
        if not selection:
            raise ValueError(f"{task_id}: missing method selection")
        selection_counts[selection] += 1

        if selection == args.reject_selection:
            rejected += 1
            continue

        candidates = item.get("candidates")
        if not isinstance(candidates, dict):
            raise ValueError(f"{task_id}: missing candidates")
        selected_path = item.get("selected_spliced_full")
        if selected_path != candidates.get(selection):
            raise ValueError(
                f"{task_id}: selected path does not match candidate {selection!r}"
            )

        source = repo_path(Path(str(selected_path)))
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stem != task_id:
            raise ValueError(
                f"{task_id}: source filename does not match task id: {source.name}"
            )

        output_name = source.name
        if output_name in output_names:
            raise ValueError(f"duplicate output filename: {output_name}")
        output_names.add(output_name)
        accepted.append(
            {
                "task_id": task_id,
                "domain": item.get("domain"),
                "selection": selection,
                "output_name": output_name,
                "source": source,
            }
        )

    expected_names = output_names | {"manifest.jsonl", "summary.json"}
    if output_dir.exists():
        unexpected = sorted(
            path.name for path in output_dir.iterdir() if path.name not in expected_names
        )
        if unexpected:
            raise ValueError(
                f"{repo_relative(output_dir)} contains unexpected files: {unexpected[:5]}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for item in accepted:
        source = item.pop("source")
        destination = output_dir / item.pop("output_name")
        shutil.copy2(source, destination)
        width, height = png_size(destination)
        manifest_rows.append(
            {
                **item,
                "image": repo_relative(destination),
                "source_image": repo_relative(source),
                "image_size": {"width": width, "height": height},
                "bytes": destination.stat().st_size,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")

    summary = {
        "schema_version": 1,
        "task": payload.get("task"),
        "selection_json": repo_relative(selection_path),
        "selection_updated_at": payload.get("updated_at"),
        "output_dir": repo_relative(output_dir),
        "manifest": repo_relative(manifest_path),
        "reviewed_total": len(selections),
        "accepted": len(accepted),
        "rejected": rejected,
        "reject_selection": args.reject_selection,
        "selection_counts": dict(sorted(selection_counts.items())),
        "accepted_selection_counts": dict(
            sorted(
                (method, count)
                for method, count in selection_counts.items()
                if method != args.reject_selection
            )
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reject-selection", default="reject_both")
    args = parser.parse_args()
    print(json.dumps(materialize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
