#!/usr/bin/env python3
"""Build legacy review/order inputs for a frozen full-image generation prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-order", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    repo_root = args.repo_root.resolve()
    generation_manifest = (repo_root / args.generation_manifest).resolve()
    annotation_path = (repo_root / args.annotations).resolve()
    output_review = (repo_root / args.output_review).resolve()
    output_order = (repo_root / args.output_order).resolve()

    annotations = {
        str(row["task_id"]): row for row in read_jsonl(annotation_path)
    }
    generated = [
        row
        for row in read_jsonl(generation_manifest)
        if row.get("status") == "ok"
    ][: args.limit]
    if len(generated) != args.limit:
        raise ValueError(
            f"requested {args.limit} successful generations, found {len(generated)}"
        )

    records: list[dict[str, Any]] = []
    ordered_inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, generated_row in enumerate(generated, start=1):
        task_id = str(generated_row["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate task_id in frozen prefix: {task_id}")
        seen.add(task_id)

        annotation = annotations.get(task_id)
        if annotation is None:
            raise ValueError(f"missing annotation for {task_id}")
        relative_image = str(generated_row["output_image"])
        image_path = (repo_root / relative_image).resolve()
        try:
            image_path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"image path escapes repository: {relative_image}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"missing generated image: {relative_image}")

        records.append(
            {
                **annotation,
                "status": "good",
                "candidates": "mouse",
                "spliced_image": relative_image,
                "full_image_generation_sha256": sha256_file(image_path),
                "compat_input_rank": rank,
            }
        )
        ordered_inputs.append({"rank": rank, "task_id": task_id})

    provenance = {
        "selection": "first successful rows in generation manifest",
        "limit": args.limit,
        "generation_manifest": args.generation_manifest.as_posix(),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "annotations": args.annotations.as_posix(),
        "annotations_sha256": sha256_file(annotation_path),
    }
    write_json(
        output_review,
        {
            "schema_version": "claimforge_full_image_compat_review_v1",
            **provenance,
            "records": records,
        },
    )
    write_json(
        output_order,
        {
            "schema_version": "claimforge_full_image_compat_order_v1",
            **provenance,
            "ordered_inputs": ordered_inputs,
        },
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "first_task_id": records[0]["task_id"],
                "last_task_id": records[-1]["task_id"],
                "review": output_review.relative_to(repo_root).as_posix(),
                "order": output_order.relative_to(repo_root).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
