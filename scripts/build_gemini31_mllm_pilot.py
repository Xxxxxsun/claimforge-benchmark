#!/usr/bin/env python3
"""Build a blind, matched-scene Gemini 3.1 Pro MLLM pilot.

Selection uses only canonical membership and a fixed SHA-256 ordering. Model
outputs, review outcomes beyond canonical inclusion, and existing MLLM results
are never consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CANDIDATES = ("mouse", "cat", "trash_can")
TASK_PREFIXES = {"mouse": "", "cat": "cat_", "trash_can": "trash_can_"}
SELECTION_SALT = "gemini31-pilot-v1:"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _candidate(row: dict[str, Any]) -> str:
    return str(row["metadata"]["candidate"]).replace("-", "_")


def _task_id(row: dict[str, Any]) -> str:
    task_id = row.get("task_id") or str(row["id"]).split("__", 1)[-1]
    return str(task_id)


def _base_task(task_id: str, candidate: str) -> str:
    prefix = TASK_PREFIXES[candidate]
    if not task_id.startswith(prefix):
        raise ValueError(f"{task_id}: expected prefix {prefix!r}")
    return task_id[len(prefix):]


def _selection_hash(base_task: str) -> str:
    return hashlib.sha256((SELECTION_SALT + base_task).encode()).hexdigest()


def _annotation_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in _read_jsonl(path)}


def _existing_path(root: Path, raw: str) -> str:
    path = Path(raw)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--partial-list",
        type=Path,
        default=Path("config/mllm_doubao_main_local776_20260726.local.jsonl"),
    )
    parser.add_argument(
        "--fullai-list",
        type=Path,
        default=Path("config/mllm_full_image_orange_box_all807_20260725.local.jsonl"),
    )
    parser.add_argument(
        "--real-list",
        type=Path,
        default=Path("config/mllm_doubao_real_source_union_20260726.local.jsonl"),
    )
    parser.add_argument(
        "--cat-annotations",
        type=Path,
        default=Path("annotations/full_image_orange_box_cat_latest272_20260724.jsonl"),
    )
    parser.add_argument(
        "--trash-can-annotations",
        type=Path,
        default=Path(
            "annotations/full_image_orange_box_trash_can_latest260_20260724.jsonl"
        ),
    )
    parser.add_argument(
        "--partial-review-output",
        type=Path,
        default=Path("config/gemini31_pilot_partial_matched9_20260727.local.json"),
    )
    parser.add_argument(
        "--fullai-output",
        type=Path,
        default=Path("config/gemini31_pilot_fullai_matched9_20260727.local.jsonl"),
    )
    parser.add_argument(
        "--real-output",
        type=Path,
        default=Path("config/gemini31_pilot_real_matched3_20260727.local.jsonl"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("config/gemini31_pilot_selection_20260727.local.json"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()

    partial_rows = _read_jsonl(args.partial_list)
    fullai_rows = _read_jsonl(args.fullai_list)
    real_rows = _read_jsonl(args.real_list)
    cat_annotations = _annotation_map(args.cat_annotations)
    trash_annotations = _annotation_map(args.trash_can_annotations)

    partial = {
        (_candidate(row), _base_task(_task_id(row), _candidate(row))): row
        for row in partial_rows
    }
    fullai = {
        (_candidate(row), _base_task(_task_id(row), _candidate(row))): row
        for row in fullai_rows
    }
    real_by_task = {
        str(task_id): row
        for row in real_rows
        for task_id in row["metadata"]["task_ids"]
    }

    valid: list[tuple[str, dict[str, Any]]] = []
    base_tasks = sorted({key[1] for key in partial} & {key[1] for key in fullai})
    for base_task in base_tasks:
        keys = [(candidate, base_task) for candidate in CANDIDATES]
        if not all(key in partial and key in fullai for key in keys):
            continue
        matched_real = [
            real_by_task.get(_task_id(partial[key]))
            for key in keys
        ]
        if not all(matched_real):
            continue
        if len({str(row["id"]) for row in matched_real}) != 1:
            continue
        valid.append((base_task, matched_real[0]))

    ranked = sorted(valid, key=lambda item: _selection_hash(item[0]))
    selected: list[tuple[str, dict[str, Any]]] = []
    for domain in ("lodging", "restaurant"):
        selected.append(
            next(item for item in ranked if item[0].startswith(domain + "_"))
        )
    selected.append(next(item for item in ranked if item not in selected))

    review_records: list[dict[str, Any]] = []
    pilot_fullai: list[dict[str, Any]] = []
    pilot_real: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for base_task, real_row in selected:
        source_image = _existing_path(root, str(real_row["image_path"]))
        pilot_real.append({
            **real_row,
            "metadata": {
                **(real_row.get("metadata") or {}),
                "pilot": "gemini31_matched_scene_v1",
                "matched_scene": base_task,
                "selection_hash": _selection_hash(base_task),
            },
        })
        selected_entry = {
            "matched_scene": base_task,
            "selection_hash": _selection_hash(base_task),
            "real_id": real_row["id"],
            "real_image_path": source_image,
            "candidates": {},
        }

        for candidate in CANDIDATES:
            key = (candidate, base_task)
            partial_row = partial[key]
            fullai_row = fullai[key]
            task_id = _task_id(partial_row)
            if task_id not in set(real_row["metadata"]["task_ids"]):
                raise ValueError(f"{task_id}: source row mismatch")

            if candidate == "mouse":
                annotation = partial_row["metadata"]
            elif candidate == "cat":
                annotation = cat_annotations[task_id]
            else:
                annotation = trash_annotations[task_id]

            if str(annotation["source_image"]) != source_image:
                raise ValueError(
                    f"{task_id}: annotation source {annotation['source_image']} "
                    f"!= matched source {source_image}"
                )
            edit_box = annotation.get("edit_region_xyxy")
            if (
                not isinstance(edit_box, list)
                or len(edit_box) != 4
                or not all(isinstance(value, int) for value in edit_box)
            ):
                raise ValueError(f"{task_id}: invalid edit_region_xyxy")

            spliced_image = _existing_path(root, str(partial_row["image_path"]))
            fullai_image = _existing_path(root, str(fullai_row["image_path"]))
            pilot_task_id = f"gemini31pilot_{candidate}_{task_id}"
            review_records.append({
                "task_id": pilot_task_id,
                "status": "good",
                "source_image": source_image,
                "spliced_image": spliced_image,
                "edit_region_xyxy": edit_box,
                "candidate": candidate,
                "matched_scene": base_task,
                "canonical_task_id": task_id,
                "selection_hash": _selection_hash(base_task),
            })
            pilot_fullai.append({
                **fullai_row,
                "metadata": {
                    **(fullai_row.get("metadata") or {}),
                    "pilot": "gemini31_matched_scene_v1",
                    "matched_scene": base_task,
                    "selection_hash": _selection_hash(base_task),
                },
            })
            selected_entry["candidates"][candidate] = {
                "canonical_task_id": task_id,
                "partial_image_path": spliced_image,
                "fullai_image_path": fullai_image,
                "edit_region_xyxy": edit_box,
            }
        selection_rows.append(selected_entry)

    _write_json(
        args.partial_review_output,
        {
            "schema_version": "gemini31_mllm_pilot_review_v1",
            "selection_policy": {
                "salt": SELECTION_SALT,
                "eligible_matched_scenes": len(valid),
                "method": (
                    "lowest SHA-256 scene per lodging and restaurant, then "
                    "lowest remaining SHA-256 scene"
                ),
                "uses_model_outputs": False,
            },
            "records": review_records,
        },
    )
    _write_jsonl(args.fullai_output, pilot_fullai)
    _write_jsonl(args.real_output, pilot_real)
    _write_json(
        args.selection_output,
        {
            "schema_version": "gemini31_mllm_pilot_selection_v1",
            "partial_images": len(review_records),
            "fullai_images": len(pilot_fullai),
            "real_images": len(pilot_real),
            "scenes": selection_rows,
        },
    )
    print(json.dumps({
        "eligible_matched_scenes": len(valid),
        "selected_scenes": [base_task for base_task, _ in selected],
        "partial_images": len(review_records),
        "fullai_images": len(pilot_fullai),
        "real_images": len(pilot_real),
        "partial_review_output": str(args.partial_review_output),
        "fullai_output": str(args.fullai_output),
        "real_output": str(args.real_output),
        "selection_output": str(args.selection_output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
