#!/usr/bin/env python3
"""Export completed trash-can slots into generation tasks and context crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_cat_generation_tasks import export


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SLOTS = (
    REPO / "annotations" / "claimforge-good-mouse-source-trash-can-275-slots.json"
)
DEFAULT_TASKS = REPO / "annotations" / "trash_can_generation_tasks.jsonl"
DEFAULT_CROPS = REPO / "crops" / "context_trash_can"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots-json", type=Path, default=DEFAULT_SLOTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.task_prefix = "trash_can"
    args.default_candidate = "trash can"
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
