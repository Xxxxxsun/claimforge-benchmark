#!/usr/bin/env python3
"""Build original-source relabel manifests from splice-method rejections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = (
    {
        "name": "cat",
        "candidate": "cat",
        "selection": Path(
            "annotations/claimforge_cat_native_style_v2_splice_method_selections.json"
        ),
        "tasks": Path("annotations/cat_generation_tasks.jsonl"),
        "output": Path("source_pool/relabel_reject_both_cat_40/manifest.json"),
    },
    {
        "name": "trash_can",
        "candidate": "trash can",
        "selection": Path(
            "annotations/claimforge_trash_can_splice_method_selections.json"
        ),
        "tasks": Path(
            "annotations/trash_can_generation_tasks_260_reviewed_mixed_context_20260724.jsonl"
        ),
        "output": Path(
            "source_pool/relabel_reject_both_trash_can_61/manifest.json"
        ),
    },
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def repo_path(path: Path) -> Path:
    absolute = path if path.is_absolute() else REPO / path
    absolute = absolute.resolve()
    if not absolute.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {path}")
    return absolute


def manifest_entry(
    task: dict[str, Any],
    selection: dict[str, Any],
    candidate: str,
) -> dict[str, Any]:
    source_image = str(task["source_image"])
    source_path = repo_path(Path(source_image))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    image_id = str(task["image_id"])
    domain = str(selection.get("domain") or task["task_id"].split("_")[1])
    return {
        "id": image_id,
        "category": domain,
        "path": f"../../{source_image}",
        "size": task["image_size"],
        "title": f"{candidate} reject_both relabel · {task['task_id']}",
        "source_image_id": source_image,
        "selection_status": "reject_both",
        "relabel_candidate": candidate,
        "rejected_task_id": task["task_id"],
        "previous_edit_region_xyxy": task["edit_region_xyxy"],
        "previous_context_region_xyxy": task["context_region_xyxy"],
        "previous_selected_candidates": selection.get("candidates"),
    }


def build_spec(spec: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    selection_path = repo_path(spec["selection"])
    tasks_path = repo_path(spec["tasks"])
    output_path = repo_path(spec["output"])
    payload = load_json(selection_path)
    selections = payload.get("selections")
    if not isinstance(selections, list):
        raise ValueError(f"missing selections list: {selection_path}")
    rejects = [
        row
        for row in selections
        if isinstance(row, dict) and row.get("selection") == "reject_both"
    ]
    task_rows = load_jsonl(tasks_path)
    tasks_by_id = {str(row["task_id"]): row for row in task_rows}
    if len(tasks_by_id) != len(task_rows):
        raise ValueError(f"duplicate task IDs in {tasks_path}")

    missing = [
        str(row.get("task_id"))
        for row in rejects
        if str(row.get("task_id")) not in tasks_by_id
    ]
    if missing:
        raise ValueError(f"{spec['name']}: missing task rows: {', '.join(missing)}")

    entries = [
        manifest_entry(
            tasks_by_id[str(selection["task_id"])],
            selection,
            str(spec["candidate"]),
        )
        for selection in rejects
    ]
    ids = [str(entry["id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{spec['name']}: duplicate image IDs in relabel manifest")
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "name": spec["name"],
        "candidate": spec["candidate"],
        "selection_path": spec["selection"].as_posix(),
        "task_path": spec["tasks"].as_posix(),
        "manifest_path": spec["output"].as_posix(),
        "reject_both": len(rejects),
        "manifest_entries": len(entries),
        "restaurant": sum(entry["category"] == "restaurant" for entry in entries),
        "lodging": sum(entry["category"] == "lodging" for entry in entries),
        "image_ids": ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("annotations/reject_both_relabel_batches_20260724.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batches = [build_spec(dict(spec), args.dry_run) for spec in DEFAULT_SPECS]
    cat_ids = set(batches[0]["image_ids"])
    trash_ids = set(batches[1]["image_ids"])
    summary = {
        "schema_version": "claimforge_reject_both_relabel_batches_v1",
        "source_selection": "reject_both",
        "batches": batches,
        "overlap": {
            "image_count": len(cat_ids & trash_ids),
            "image_ids": sorted(cat_ids & trash_ids),
        },
    }
    if not args.dry_run:
        summary_path = repo_path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
