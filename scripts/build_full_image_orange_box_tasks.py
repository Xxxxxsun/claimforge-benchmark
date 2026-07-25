#!/usr/bin/env python3
"""Build the frozen task lists for the full-image orange-box control.

Mouse uses the human-reviewed good subset.  Cat and trash-can use the complete
base task list, with the newest completed reject-both relabel boxes replacing
their older rows by task ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]

DEFAULT_MOUSE_BASE = Path("annotations/generation_tasks.jsonl")
DEFAULT_MOUSE_REVIEW = Path("claimforge_generation_review_labels.json")
DEFAULT_CAT_BASE = Path("annotations/cat_generation_tasks.jsonl")
DEFAULT_CAT_REPLACEMENTS = Path(
    "annotations/cat_generation_tasks_relabel_reject_both_partial28_20260724.jsonl"
)
DEFAULT_TRASH_BASE = Path(
    "annotations/trash_can_generation_tasks_260_reviewed_mixed_context_20260724.jsonl"
)
DEFAULT_TRASH_REPLACEMENTS = Path(
    "annotations/"
    "trash_can_generation_tasks_relabel_reject_both_partial55_positioned_20260724.jsonl"
)

DEFAULT_MOUSE_OUTPUT = Path(
    "annotations/full_image_orange_box_mouse_good275_latest_20260724.jsonl"
)
DEFAULT_CAT_OUTPUT = Path(
    "annotations/full_image_orange_box_cat_latest272_20260724.jsonl"
)
DEFAULT_TRASH_OUTPUT = Path(
    "annotations/full_image_orange_box_trash_can_latest260_20260724.jsonl"
)


def repo_path(path: Path) -> Path:
    resolved = (path if path.is_absolute() else REPO / path).resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {path}")
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def by_task_id(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate task IDs in {source}")
    return indexed


def select_good_mouse(
    base_rows: list[dict[str, Any]],
    review_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    base = by_task_id(base_rows, DEFAULT_MOUSE_BASE)
    records = review_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("mouse review payload has no records list")
    good_records = [
        row
        for row in records
        if row.get("status") == "good" and row.get("candidates") == "mouse"
    ]
    good = by_task_id(good_records, DEFAULT_MOUSE_REVIEW)
    missing = sorted(good.keys() - base.keys())
    if missing:
        raise ValueError(f"reviewed mouse IDs missing from base tasks: {missing[:5]}")

    # The review file is an immutable snapshot of the exact source, size, and
    # box that a human judged.  Task IDs have historically been reused when the
    # lodging source pool was refreshed, so joining only by task_id can silently
    # attach a "good" decision to a different image.  Preserve base ordering and
    # stable image/slot metadata, but take all reviewed visual fields from the
    # review snapshot.
    selected: list[dict[str, Any]] = []
    for base_row in base_rows:
        task_id = str(base_row["task_id"])
        if task_id not in good:
            continue
        reviewed = dict(good[task_id])
        reviewed["image_id"] = base_row.get("image_id")
        reviewed["slot_id"] = base_row.get("slot_id")
        x1, y1, x2, y2 = [
            int(value) for value in reviewed["edit_region_xyxy"]
        ]
        reviewed["insert_box"] = {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        selected.append(reviewed)
    if len(selected) != len(good):
        raise AssertionError((len(selected), len(good)))
    return selected


def merge_replacements(
    base_rows: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
    *,
    base_source: Path,
    replacement_source: Path,
) -> list[dict[str, Any]]:
    base = by_task_id(base_rows, base_source)
    replacements = by_task_id(replacement_rows, replacement_source)
    unexpected = sorted(replacements.keys() - base.keys())
    if unexpected:
        raise ValueError(
            f"replacement IDs absent from base tasks: {unexpected[:5]}"
        )
    return [
        replacements.get(str(row["task_id"]), row)
        for row in base_rows
    ]


def freeze_row(
    row: dict[str, Any],
    *,
    annotation_source: str,
    replacement: bool,
) -> dict[str, Any]:
    frozen = {
        "task_id": str(row["task_id"]),
        "image_id": row.get("image_id"),
        "slot_id": row.get("slot_id"),
        "source_image": str(row["source_image"]),
        "image_size": row["image_size"],
        "edit_region_xyxy": [int(value) for value in row["edit_region_xyxy"]],
        "insert_box": row.get("insert_box"),
        "candidates": str(row["candidates"]),
        "annotation_source": annotation_source,
        "latest_relabel_replacement": replacement,
    }
    if row.get("context_region_xyxy") is not None:
        frozen["context_region_xyxy"] = [
            int(value) for value in row["context_region_xyxy"]
        ]
    return frozen


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_candidate: str,
) -> None:
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs in frozen rows")
    for row in rows:
        if str(row["candidates"]) != expected_candidate:
            raise ValueError(
                f"{row['task_id']}: candidate {row['candidates']!r} "
                f"!= {expected_candidate!r}"
            )
        source = repo_path(Path(str(row["source_image"])))
        with Image.open(source) as image:
            actual_size = image.size
        declared = row["image_size"]
        if isinstance(declared, dict):
            declared_size = (
                int(declared["width"]),
                int(declared["height"]),
            )
        else:
            declared_size = tuple(int(value) for value in declared)
        if declared_size != actual_size:
            raise ValueError(
                f"{row['task_id']}: declared {declared_size} != {actual_size}"
            )
        x1, y1, x2, y2 = row["edit_region_xyxy"]
        if not (
            0 <= x1 < x2 <= actual_size[0]
            and 0 <= y1 < y2 <= actual_size[1]
        ):
            raise ValueError(
                f"{row['task_id']}: box {row['edit_region_xyxy']} "
                f"outside {actual_size}"
            )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    mouse_base_path = repo_path(args.mouse_base)
    mouse_review_path = repo_path(args.mouse_review)
    cat_base_path = repo_path(args.cat_base)
    cat_replacements_path = repo_path(args.cat_replacements)
    trash_base_path = repo_path(args.trash_base)
    trash_replacements_path = repo_path(args.trash_replacements)

    mouse_base = load_jsonl(mouse_base_path)
    mouse_review = json.loads(mouse_review_path.read_text(encoding="utf-8"))
    mouse_selected = select_good_mouse(mouse_base, mouse_review)
    mouse_rows = [
        freeze_row(
            row,
            annotation_source=(
                "claimforge_generation_review_labels.json"
                "(status=good,candidates=mouse;review-snapshot-visual-fields)"
            ),
            replacement=False,
        )
        for row in mouse_selected
    ]

    cat_base = load_jsonl(cat_base_path)
    cat_replacements = load_jsonl(cat_replacements_path)
    cat_replacement_ids = {
        str(row["task_id"]) for row in cat_replacements
    }
    cat_merged = merge_replacements(
        cat_base,
        cat_replacements,
        base_source=cat_base_path,
        replacement_source=cat_replacements_path,
    )
    cat_rows = [
        freeze_row(
            row,
            annotation_source=(
                cat_replacements_path.relative_to(REPO).as_posix()
                if str(row["task_id"]) in cat_replacement_ids
                else cat_base_path.relative_to(REPO).as_posix()
            ),
            replacement=str(row["task_id"]) in cat_replacement_ids,
        )
        for row in cat_merged
    ]

    trash_base = load_jsonl(trash_base_path)
    trash_replacements = load_jsonl(trash_replacements_path)
    trash_replacement_ids = {
        str(row["task_id"]) for row in trash_replacements
    }
    trash_merged = merge_replacements(
        trash_base,
        trash_replacements,
        base_source=trash_base_path,
        replacement_source=trash_replacements_path,
    )
    trash_rows = [
        freeze_row(
            row,
            annotation_source=(
                trash_replacements_path.relative_to(REPO).as_posix()
                if str(row["task_id"]) in trash_replacement_ids
                else trash_base_path.relative_to(REPO).as_posix()
            ),
            replacement=str(row["task_id"]) in trash_replacement_ids,
        )
        for row in trash_merged
    ]

    validate_rows(mouse_rows, expected_candidate="mouse")
    validate_rows(cat_rows, expected_candidate="cat")
    validate_rows(trash_rows, expected_candidate="trash can")

    expected_default_counts = {
        "mouse": (
            args.mouse_output == DEFAULT_MOUSE_OUTPUT,
            len(mouse_rows),
            275,
        ),
        "cat": (
            args.cat_output == DEFAULT_CAT_OUTPUT,
            len(cat_rows),
            272,
        ),
        "trash_can": (
            args.trash_output == DEFAULT_TRASH_OUTPUT,
            len(trash_rows),
            260,
        ),
    }
    for kind, (is_default_output, actual, expected) in (
        expected_default_counts.items()
    ):
        if is_default_output and actual != expected:
            raise ValueError(
                f"refusing to overwrite frozen default {kind} task list: "
                f"count {actual} != {expected}"
            )

    outputs = {
        "mouse": (repo_path(args.mouse_output), mouse_rows),
        "cat": (repo_path(args.cat_output), cat_rows),
        "trash_can": (repo_path(args.trash_output), trash_rows),
    }
    if not args.dry_run:
        for output_path, rows in outputs.values():
            write_jsonl(output_path, rows)

    return {
        "mouse": {
            "tasks": len(mouse_rows),
            "replacements": 0,
            "output": outputs["mouse"][0].relative_to(REPO).as_posix(),
        },
        "cat": {
            "tasks": len(cat_rows),
            "replacements": len(cat_replacement_ids),
            "output": outputs["cat"][0].relative_to(REPO).as_posix(),
        },
        "trash_can": {
            "tasks": len(trash_rows),
            "replacements": len(trash_replacement_ids),
            "output": outputs["trash_can"][0].relative_to(REPO).as_posix(),
        },
        "total_tasks": len(mouse_rows) + len(cat_rows) + len(trash_rows),
        "dry_run": args.dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse-base", type=Path, default=DEFAULT_MOUSE_BASE)
    parser.add_argument("--mouse-review", type=Path, default=DEFAULT_MOUSE_REVIEW)
    parser.add_argument("--cat-base", type=Path, default=DEFAULT_CAT_BASE)
    parser.add_argument(
        "--cat-replacements",
        type=Path,
        default=DEFAULT_CAT_REPLACEMENTS,
    )
    parser.add_argument("--trash-base", type=Path, default=DEFAULT_TRASH_BASE)
    parser.add_argument(
        "--trash-replacements",
        type=Path,
        default=DEFAULT_TRASH_REPLACEMENTS,
    )
    parser.add_argument("--mouse-output", type=Path, default=DEFAULT_MOUSE_OUTPUT)
    parser.add_argument("--cat-output", type=Path, default=DEFAULT_CAT_OUTPUT)
    parser.add_argument("--trash-output", type=Path, default=DEFAULT_TRASH_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
