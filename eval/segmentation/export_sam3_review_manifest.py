#!/usr/bin/env python3
"""Join SAM3 splice results with task metadata for the generation-review UI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label} row is missing task_id")
        if task_id in result:
            raise ValueError(f"{label} contains duplicate task_id {task_id!r}")
        result[task_id] = row
    return result


def export_review_manifest(
    base_manifest: Path,
    splice_results: Path,
    output: Path,
    candidate: str,
    task_manifest: Path | None = None,
    fallback_splice_results: Path | None = None,
) -> list[dict[str, Any]]:
    tasks = keyed(load_jsonl(task_manifest), "task manifest") if task_manifest else {}
    splices = keyed(
        [
            row
            for row in load_jsonl(splice_results)
            if row.get("endpoint_tag") == "sam3" and row.get("status") == "ok"
        ],
        "SAM3 splice results",
    )
    fallback_splices = (
        keyed(
            [
                row
                for row in load_jsonl(fallback_splice_results)
                if row.get("endpoint_tag") == "sam3" and row.get("status") == "ok"
            ],
            "SAM3 fallback splice results",
        )
        if fallback_splice_results
        else {}
    )
    review_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for base in load_jsonl(base_manifest):
        if base.get("status") != "ok":
            continue
        task_id = str(base["task_id"])
        merged = {**tasks.get(task_id, {}), **base}
        if merged.get("candidates") != candidate:
            continue
        selected_ids.append(task_id)
        primary = splices.get(task_id)
        fallback = fallback_splices.get(task_id)
        primary_pass = bool(primary and primary["quality_gate"]["pass"])
        fallback_pass = bool(fallback and fallback["quality_gate"]["pass"])
        if primary_pass:
            splice = primary
            selection_source = "primary_pass"
        elif fallback_pass:
            splice = fallback
            selection_source = "fallback_pass"
        elif primary is not None:
            splice = primary
            selection_source = "primary_failed_gate_no_passing_fallback"
        elif fallback is not None:
            splice = fallback
            selection_source = "fallback_failed_gate_only_available"
        else:
            splice = None
            selection_source = "missing"
        if splice is None:
            continue
        generated_crop = merged.get("generated_crop") or merged.get("output_crop")
        review_rows.append(
            {
                "task_id": task_id,
                "source_image": merged["source_image"],
                "generated_crop": generated_crop,
                "spliced_full": splice["hybrid_spliced_full"],
                "image_size": merged["image_size"],
                "context_region_xyxy": merged["context_region_xyxy"],
                "edit_region_xyxy": merged["edit_region_xyxy"],
                "edit_region_in_context_xyxy": merged[
                    "edit_region_in_context_xyxy"
                ],
                "candidates": candidate,
                "paste_mode": "fal_sam3_text_only_hybrid",
                "sam3_quality_gate": splice["quality_gate"],
                "sam3_request_id": splice["request_id"],
                "sam3_selection_source": selection_source,
                "sam3_primary_quality_gate": (
                    primary["quality_gate"] if primary else None
                ),
                "sam3_fallback_quality_gate": (
                    fallback["quality_gate"] if fallback else None
                ),
                "status": "ok",
            }
        )
    available = set(splices) | set(fallback_splices)
    missing = sorted(set(selected_ids) - available)
    if missing:
        raise ValueError(
            f"{len(missing)} selected tasks have no successful SAM3 splice: "
            + ", ".join(missing[:10])
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return review_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--splice-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--fallback-splice-results", type=Path)
    args = parser.parse_args()
    rows = export_review_manifest(
        args.base_manifest,
        args.splice_results,
        args.output,
        args.candidate,
        args.task_manifest,
        args.fallback_splice_results,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "rows": len(rows),
                "quality_gate_failures": sum(
                    not row["sam3_quality_gate"]["pass"] for row in rows
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
