#!/usr/bin/env python3
"""Build the paired 3-class, 2-generation-route CLAIMFORGE benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("benchmark/claimforge_v1_250x3x2")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCHEMA_VERSION = "claimforge_benchmark_v1"


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    candidate: str
    tasks: Path
    local_kind: str
    local_source: Path
    full_run: Path


CATEGORIES = (
    CategoryConfig(
        name="mouse",
        candidate="mouse",
        tasks=Path(
            "annotations/full_image_orange_box_mouse_good275_latest_20260724.jsonl"
        ),
        local_kind="review_good",
        local_source=Path("claimforge_generation_review_labels.json"),
        full_run=Path(
            "generated_full_images/"
            "hunyuan_image3_distil_full_input_orange_box_mouse_good275_g5_v1_20260724"
        ),
    ),
    CategoryConfig(
        name="cat",
        candidate="cat",
        tasks=Path("annotations/full_image_orange_box_cat_latest272_20260724.jsonl"),
        local_kind="selected_final",
        local_source=Path(
            "spliced_final/claimforge_cat_selected_251_20260725/manifest.jsonl"
        ),
        full_run=Path(
            "generated_full_images/"
            "hunyuan_image3_distil_full_input_orange_box_cat_latest272_g5_v1_20260724"
        ),
    ),
    CategoryConfig(
        name="trash_can",
        candidate="trash can",
        tasks=Path(
            "annotations/full_image_orange_box_trash_can_latest260_20260724.jsonl"
        ),
        local_kind="selected_final",
        local_source=Path(
            "spliced_final/claimforge_trash_can_selected_250_20260725/manifest.jsonl"
        ),
        full_run=Path(
            "generated_full_images/"
            "hunyuan_image3_distil_full_input_orange_box_trash_can_latest260_g5_v1_20260724"
        ),
    ),
)


def repo_path(path: Path) -> Path:
    absolute = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    if not absolute.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {path}")
    return absolute


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path, cache: dict[Path, str]) -> str:
    cached = cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    cache[path] = value
    return value


def sha256_lines(values: list[str]) -> str:
    text = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def unique_by_task(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"missing task_id in {source}")
        if task_id in output:
            raise ValueError(f"duplicate task_id {task_id!r} in {source}")
        output[task_id] = row
    return output


def load_local_candidates(config: CategoryConfig) -> dict[str, dict[str, Any]]:
    source = repo_path(config.local_source)
    if config.local_kind == "review_good":
        payload = load_json(source)
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"missing records list: {source}")
        rows = [
            {
                "task_id": row["task_id"],
                "image": row["spliced_image"],
                "selection": "human_review_good",
                "selection_source": repo_relative(source),
                "review_manifest": row.get("review_manifest"),
            }
            for row in records
            if row.get("status") == "good" and row.get("candidates") == config.candidate
        ]
    elif config.local_kind == "selected_final":
        rows = [
            {
                "task_id": row["task_id"],
                "image": row["image"],
                "selection": row.get("selection"),
                "selection_source": repo_relative(source),
                "selected_candidate_source": row.get("source_image"),
            }
            for row in load_jsonl(source)
        ]
    else:
        raise ValueError(f"unsupported local source kind: {config.local_kind}")
    return unique_by_task(rows, source)


def load_latest_full_results(config: CategoryConfig) -> dict[str, dict[str, Any]]:
    manifest = repo_path(config.full_run / "manifest.jsonl")
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(manifest):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"missing task_id in {manifest}")
        latest[task_id] = row
    return latest


def load_trash_can_qc() -> tuple[Path, dict[str, dict[str, Any]]]:
    path = repo_path(
        Path(
            "annotations/"
            "full_image_orange_box_trash_can_single_shot_manual_qc_20260724.json"
        )
    )
    payload = load_json(path)
    failures = payload.get("failures")
    if not isinstance(failures, list):
        raise ValueError(f"missing failures list: {path}")
    return path, unique_by_task(failures, path)


def prepare_output(output_dir: Path, force: bool) -> Path:
    if output_dir == REPO:
        raise ValueError("output directory cannot be repository root")
    temporary = output_dir.parent / f".{output_dir.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{repo_relative(output_dir)} already exists; use --force")
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"refusing to replace unrecognized directory: {output_dir}")
        summary = load_json(summary_path)
        if summary.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"refusing to replace unrecognized directory: {output_dir}")
    temporary.mkdir(parents=True)
    return temporary


def copy_benchmark_image(
    *,
    source: Path,
    temporary: Path,
    final_output: Path,
    method: str,
    category: str,
    task_id: str,
    expected_size: tuple[int, int],
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".png":
        raise ValueError(f"expected PNG source: {source}")
    if source.stem != task_id:
        raise ValueError(f"task/source filename mismatch: {task_id} != {source.name}")
    actual_size = png_size(source)
    if actual_size != expected_size:
        raise ValueError(
            f"{task_id}: expected {expected_size}, got {actual_size} from {source}"
        )

    relative = Path(method) / category / source.name
    destination = temporary / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    source_sha256 = sha256_file(source, hash_cache)
    destination_sha256 = sha256_file(destination, hash_cache)
    if source_sha256 != destination_sha256:
        raise ValueError(f"copy hash mismatch: {source} -> {destination}")
    return {
        "image": repo_relative(final_output / relative),
        "source_artifact": repo_relative(source),
        "sha256": destination_sha256,
        "bytes": destination.stat().st_size,
        "image_size": {"width": actual_size[0], "height": actual_size[1]},
    }


def render_readme(summary: dict[str, Any]) -> str:
    trash_qc = summary["categories"]["trash_can"]["full_image_manual_qc"]
    return f"""# CLAIMFORGE v1 250x3x2

This directory is the frozen edited-image benchmark slice with:

- 3 object categories: mouse, cat, and trash can;
- 2 generation routes: local crop generation plus splice-back, and full-image
  orange-box conditional generation;
- 250 matched tasks per category and route.

The result contains {summary["total_images"]} edited PNGs arranged as:

```text
local_splice/<category>/*.png
full_image/<category>/*.png
```

`pairs.jsonl` contains the {summary["total_pairs"]} matched task pairs.
`manifest.jsonl` contains one row per edited image. Each method/category
directory also has its own 250-row `manifest.jsonl`.

## Selection policy

For each category, the eligible local-splice quality-approved task set is
filtered through the frozen full-image task-list order. The first 250 task IDs
are retained, and both generation routes use exactly those same task IDs and
real source images. This is a deterministic prefix, not a random sample.

Mouse eligibility is `status=good` in
`claimforge_generation_review_labels.json`. Cat and trash-can eligibility comes
from the manually selected final splice manifests. The full-image side always
uses the fixed single-shot primary output; retries are not substituted.

## Full-image semantic QC

The selected trash-can full-image slice retains the existing manual QC labels:
{trash_qc["usable"]}/250 usable and {trash_qc["failed"]}/250 failed. These
failures remain in the benchmark because the full-image control is a fixed
single-shot generation condition. Use `full_image_manual_qc` in the manifests
for stratified analysis; do not silently drop failures from headline metrics.

Cat and mouse full-image outputs have structural validation but no equivalent
complete manual semantic-QC file in this repository snapshot.

## Rebuild

From the repository root:

```bash
python3 scripts/build_final_benchmark.py --force
```
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    limit = args.per_category
    if limit <= 0:
        raise ValueError("--per-category must be positive")
    final_output = repo_path(args.output_dir)
    temporary = prepare_output(final_output, args.force)
    hash_cache: dict[Path, str] = {}
    trash_qc_path, trash_failures = load_trash_can_qc()

    global_manifest: list[dict[str, Any]] = []
    pair_manifest: list[dict[str, Any]] = []
    category_summaries: dict[str, Any] = {}
    per_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}

    try:
        for config in CATEGORIES:
            task_path = repo_path(config.tasks)
            task_rows = load_jsonl(task_path)
            task_map = unique_by_task(task_rows, task_path)
            local_map = load_local_candidates(config)
            full_map = load_latest_full_results(config)

            missing_from_tasks = sorted(set(local_map) - set(task_map))
            if missing_from_tasks:
                raise ValueError(
                    f"{config.name}: local candidates missing from frozen tasks: "
                    f"{missing_from_tasks[:5]}"
                )

            eligible_tasks = [
                row for row in task_rows if str(row["task_id"]) in local_map
            ]
            if len(eligible_tasks) < limit:
                raise ValueError(
                    f"{config.name}: only {len(eligible_tasks)} eligible tasks, "
                    f"need {limit}"
                )
            selected_tasks = eligible_tasks[:limit]
            selected_ids = [str(row["task_id"]) for row in selected_tasks]

            qc_counts: Counter[str] = Counter()
            selected_source_images: set[str] = set()
            for rank, task in enumerate(selected_tasks, start=1):
                task_id = str(task["task_id"])
                candidate = str(task.get("candidates") or "")
                if candidate != config.candidate:
                    raise ValueError(
                        f"{task_id}: expected candidate {config.candidate!r}, "
                        f"got {candidate!r}"
                    )
                declared_size = task.get("image_size")
                if isinstance(declared_size, dict):
                    expected_size = (
                        int(declared_size["width"]),
                        int(declared_size["height"]),
                    )
                elif isinstance(declared_size, list) and len(declared_size) == 2:
                    expected_size = (int(declared_size[0]), int(declared_size[1]))
                else:
                    raise ValueError(f"{task_id}: invalid image_size")
                source_image = repo_path(Path(str(task["source_image"])))
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                selected_source_images.add(repo_relative(source_image))
                domain = str(task.get("image_id") or "").split("_", maxsplit=1)[0]
                pair_id = f"{config.name}/{task_id}"

                local = local_map[task_id]
                local_source = repo_path(Path(str(local["image"])))
                local_artifact = copy_benchmark_image(
                    source=local_source,
                    temporary=temporary,
                    final_output=final_output,
                    method="local_splice",
                    category=config.name,
                    task_id=task_id,
                    expected_size=expected_size,
                    hash_cache=hash_cache,
                )

                full = full_map.get(task_id)
                if not full or full.get("status") != "ok":
                    raise ValueError(f"{task_id}: missing latest successful full-image row")
                full_source = repo_path(Path(str(full["output_image"])))
                full_artifact = copy_benchmark_image(
                    source=full_source,
                    temporary=temporary,
                    final_output=final_output,
                    method="full_image",
                    category=config.name,
                    task_id=task_id,
                    expected_size=expected_size,
                    hash_cache=hash_cache,
                )

                full_qc: dict[str, Any] | None = None
                if config.name == "trash_can":
                    failure = trash_failures.get(task_id)
                    full_qc = {
                        "reviewed": True,
                        "usable": failure is None,
                        "source": repo_relative(trash_qc_path),
                    }
                    if failure is not None:
                        full_qc["categories"] = failure.get("categories", [])
                        full_qc["reason"] = failure.get("reason", "")
                        qc_counts["failed"] += 1
                    else:
                        qc_counts["usable"] += 1

                common = {
                    "schema_version": SCHEMA_VERSION,
                    "pair_id": pair_id,
                    "rank": rank,
                    "category": config.name,
                    "task_id": task_id,
                    "domain": domain,
                    "label": "ai_edited",
                    "source_image": repo_relative(source_image),
                    "source_image_sha256": sha256_file(source_image, hash_cache),
                    "edit_region_xyxy": task.get("edit_region_xyxy"),
                    "context_region_xyxy": task.get("context_region_xyxy"),
                }
                local_row = {
                    **common,
                    "method": "local_splice",
                    **local_artifact,
                    "local_selection": {
                        key: value
                        for key, value in local.items()
                        if key not in {"task_id", "image"}
                    },
                }
                full_row = {
                    **common,
                    "method": "full_image",
                    **full_artifact,
                    "full_image_manual_qc": full_qc,
                }
                global_manifest.extend((local_row, full_row))
                per_cell.setdefault(("local_splice", config.name), []).append(local_row)
                per_cell.setdefault(("full_image", config.name), []).append(full_row)
                pair_manifest.append(
                    {
                        **common,
                        "local_splice_image": local_artifact["image"],
                        "local_splice_sha256": local_artifact["sha256"],
                        "full_image": full_artifact["image"],
                        "full_image_sha256": full_artifact["sha256"],
                        "full_image_manual_qc": full_qc,
                    }
                )

            excluded_ids = [
                str(row["task_id"]) for row in eligible_tasks[limit:]
            ]
            category_summary: dict[str, Any] = {
                "frozen_tasks": len(task_rows),
                "eligible_local_splice": len(eligible_tasks),
                "selected_tasks": len(selected_tasks),
                "local_splice_images": len(selected_tasks),
                "full_image_images": len(selected_tasks),
                "unique_source_images": len(selected_source_images),
                "task_order_source": repo_relative(task_path),
                "task_ids_sha256": sha256_lines(selected_ids),
                "excluded_eligible_task_ids": excluded_ids,
                "local_splice_source": repo_relative(repo_path(config.local_source)),
                "full_image_source": repo_relative(repo_path(config.full_run)),
            }
            if config.name == "trash_can":
                category_summary["full_image_manual_qc"] = {
                    "reviewed": len(selected_tasks),
                    "usable": qc_counts["usable"],
                    "failed": qc_counts["failed"],
                    "source": repo_relative(trash_qc_path),
                }
            category_summaries[config.name] = category_summary

        expected_pairs = limit * len(CATEGORIES)
        expected_images = expected_pairs * 2
        if len(pair_manifest) != expected_pairs:
            raise ValueError(
                f"expected {expected_pairs} pairs, got {len(pair_manifest)}"
            )
        if len(global_manifest) != expected_images:
            raise ValueError(
                f"expected {expected_images} images, got {len(global_manifest)}"
            )
        if len({row["pair_id"] for row in pair_manifest}) != expected_pairs:
            raise ValueError("duplicate pair_id in final benchmark")
        if len({row["image"] for row in global_manifest}) != expected_images:
            raise ValueError("duplicate image path in final benchmark")

        for (method, category), rows in per_cell.items():
            if len(rows) != limit:
                raise ValueError(f"{method}/{category}: expected {limit}, got {len(rows)}")
            write_jsonl(temporary / method / category / "manifest.jsonl", rows)

        write_jsonl(temporary / "manifest.jsonl", global_manifest)
        write_jsonl(temporary / "pairs.jsonl", pair_manifest)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "selection_policy": (
                "first eligible local-splice tasks in each frozen full-image task-list "
                "order; identical task IDs used for both generation routes"
            ),
            "per_category_per_method": limit,
            "methods": ["local_splice", "full_image"],
            "category_order": [config.name for config in CATEGORIES],
            "total_pairs": expected_pairs,
            "total_images": expected_images,
            "manifest": repo_relative(final_output / "manifest.jsonl"),
            "pairs_manifest": repo_relative(final_output / "pairs.jsonl"),
            "categories": category_summaries,
        }
        write_json(temporary / "summary.json", summary)
        (temporary / "README.md").write_text(
            render_readme(summary),
            encoding="utf-8",
        )

        if final_output.exists():
            shutil.rmtree(final_output)
        temporary.rename(final_output)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
