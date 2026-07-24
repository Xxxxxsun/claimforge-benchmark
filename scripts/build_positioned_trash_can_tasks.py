#!/usr/bin/env python3
"""Attach concise, per-image trash-can placement prompts to generation tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE = (
    "Add exactly one clearly visible small ordinary trash can centered near "
    "{center_x_pct}% from the left and {center_y_pct}% from the top. Place it "
    "{placement}. The named support surface and this exact placement are "
    "authoritative: keep the complete base visibly touching that surface and do "
    "not relocate the bin away from it. Keep the bin modest "
    "in scale and well inside the image. Show its complete rim or lid, both side "
    "contours, entire body, and full base in front of every surrounding object, "
    "with a clear background gap around the whole silhouette and at least 8% "
    "frame margin. Do not omit, crop, hide, or duplicate the bin. Match the "
    "source perspective, lighting, color, depth of field, resolution, detail, "
    "sharpness or blur, noise, and compression. Change nothing else; do not "
    "move, erase, sharpen, or redraw existing people, furniture, objects, or "
    "background."
)


def repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"path must stay inside repository: {path}")
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = repo_path(args.tasks)
    placements_path = repo_path(args.placements)
    output_path = repo_path(args.output_tasks)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    tasks = load_jsonl(tasks_path)
    placements = load_jsonl(placements_path)
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("task IDs must be unique")

    placement_by_id: dict[str, dict[str, Any]] = {}
    for placement in placements:
        task_id = str(placement["task_id"])
        if task_id in placement_by_id:
            raise ValueError(f"duplicate placement: {task_id}")
        if task_id not in task_by_id:
            raise ValueError(f"placement has unknown task: {task_id}")
        center_x = int(placement["center_x_pct"])
        center_y = int(placement["center_y_pct"])
        placement_text = str(
            placement.get(
                "placement",
                placement.get(
                    "support_location",
                    placement.get("support", ""),
                ),
            )
        ).strip()
        if not 8 <= center_x <= 92 or not 8 <= center_y <= 92:
            raise ValueError(f"{task_id}: placement must stay in 8..92%")
        if not placement_text:
            raise ValueError(f"{task_id}: placement must not be empty")
        placement_by_id[task_id] = {
            **placement,
            "center_x_pct": center_x,
            "center_y_pct": center_y,
            "placement": placement_text,
        }

    output: list[dict[str, Any]] = []
    for task in tasks:
        placement = placement_by_id.get(task["task_id"])
        if placement is None:
            continue
        prompt = PROMPT_TEMPLATE.format(**placement)
        output.append(
            {
                **task,
                "prompt_override": prompt,
                "placement_plan": {
                    "center_x_pct": placement["center_x_pct"],
                    "center_y_pct": placement["center_y_pct"],
                    "placement": placement["placement"],
                },
            }
        )

    if len(output) != len(placements):
        raise RuntimeError("not every placement produced one output task")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in output:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return {
        "tasks": len(output),
        "output_tasks": str(output_path.relative_to(REPO)),
        "placements": str(placements_path.relative_to(REPO)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--output-tasks", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
