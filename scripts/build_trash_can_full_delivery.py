#!/usr/bin/env python3
"""Build lossless reviewed deliveries for the remaining 148 and all 260 tasks.

Every task is retained in the delivery. Strict visual QA is represented by the
``review_status`` field (``usable`` or ``needs_review``); it never removes an
image from either output directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]

REMAINING_TASKS = Path(
    "annotations/trash_can_generation_tasks_remaining_148.jsonl"
)
ALL_TASKS = Path("annotations/trash_can_generation_tasks.jsonl")
PRIOR_TASKS = Path("annotations/trash_can_generation_tasks_natural_112.jsonl")
PRIOR_REVIEW = Path(
    "annotations/trash_can_complete_natural_review_20260723.jsonl"
)
PRIOR_DELIVERY = Path(
    "generated_crops/"
    "hunyuan_image3_distil_trash_can_112_complete_natural_reviewed_20260723"
)

REVIEW_148 = Path(
    "annotations/trash_can_remaining_148_review_20260724.jsonl"
)
TASKS_148 = Path(
    "annotations/"
    "trash_can_generation_tasks_remaining_148_reviewed_mixed_context_20260724.jsonl"
)
DELIVERY_148 = Path(
    "generated_crops/"
    "hunyuan_image3_distil_trash_can_remaining_148_reviewed_20260724"
)

REVIEW_260 = Path("annotations/trash_can_full_260_review_20260724.jsonl")
TASKS_260 = Path(
    "annotations/"
    "trash_can_generation_tasks_260_reviewed_mixed_context_20260724.jsonl"
)
DELIVERY_260 = Path(
    "generated_crops/"
    "hunyuan_image3_distil_trash_can_260_complete_reviewed_20260724"
)


def ids(text: str) -> set[str]:
    return {
        f"trash_can_{short_id}_slot_001"
        for short_id in text.split()
        if short_id
    }


# These sets are the union of the strict, task-by-task visual QA passes. A later
# revision takes precedence when the same task passed in more than one round.
STRICT_PASS: dict[str, set[str]] = {
    "v5": ids(
        """
        restaurant_008 restaurant_042 restaurant_051 restaurant_063
        restaurant_071 restaurant_078 restaurant_089 restaurant_090
        restaurant_101 restaurant_110 restaurant_111 restaurant_122
        restaurant_134 restaurant_137 restaurant_149 restaurant_164
        restaurant_183 restaurant_199 restaurant_205 restaurant_225
        restaurant_233 restaurant_236 restaurant_263 restaurant_264
        lodging_007 lodging_020 lodging_028 lodging_049 lodging_070
        lodging_085 lodging_091 lodging_092 lodging_097 lodging_120
        lodging_137 lodging_148 lodging_169 lodging_172 lodging_179
        lodging_190 lodging_191 lodging_208 lodging_217 lodging_219
        lodging_230 lodging_235 lodging_238 lodging_256 lodging_269
        lodging_270 lodging_273 lodging_280 lodging_287 lodging_295
        """
    ),
    "v1": ids(
        """
        restaurant_049 restaurant_176 lodging_032 lodging_108 lodging_180
        """
    ),
    "v6": ids(
        """
        restaurant_022 restaurant_024 restaurant_029 restaurant_045
        restaurant_072 restaurant_091 restaurant_112 restaurant_115
        restaurant_116 restaurant_151 restaurant_206 restaurant_215
        restaurant_241 restaurant_250 restaurant_262 restaurant_276
        restaurant_298 lodging_061 lodging_075 lodging_087 lodging_197
        lodging_199 lodging_224
        """
    ),
    "v7": ids(
        """
        lodging_118 lodging_122 lodging_205 lodging_249 lodging_261
        lodging_281
        """
    ),
    "v8a": ids(
        """
        restaurant_003 restaurant_037 restaurant_052 lodging_013
        lodging_118 lodging_122 lodging_205 lodging_261 lodging_263
        lodging_272 lodging_281
        """
    ),
    "v8b": set(),
}


@dataclass(frozen=True)
class CandidateSource:
    revision: str
    output_dir: Path
    tasks: Path


SOURCES = {
    source.revision: source
    for source in (
        CandidateSource(
            "v1",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_260_native_style_v1_20260722"
            ),
            REMAINING_TASKS,
        ),
        CandidateSource(
            "v5",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_148_complete_natural_v5_20260724"
            ),
            REMAINING_TASKS,
        ),
        CandidateSource(
            "v6",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_remaining_89_expanded_v6_20260724"
            ),
            Path(
                "annotations/"
                "trash_can_generation_tasks_remaining_89_expanded_v6.jsonl"
            ),
        ),
        CandidateSource(
            "v7",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_lodging_32_positioned_v7_20260724"
            ),
            Path(
                "annotations/"
                "trash_can_generation_tasks_lodging_32_positioned_v7.jsonl"
            ),
        ),
        CandidateSource(
            "v8a",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_positioned_49_"
                "think_recaption_v8a_20260724"
            ),
            Path(
                "annotations/"
                "trash_can_generation_tasks_positioned_49_v8a.jsonl"
            ),
        ),
        CandidateSource(
            "v8b",
            Path(
                "generated_crops/"
                "hunyuan_image3_distil_trash_can_positioned_17_"
                "think_recaption_v8b_20260724"
            ),
            Path(
                "annotations/"
                "trash_can_generation_tasks_positioned_66_v8.jsonl"
            ),
        ),
    )
}

PASS_PRIORITY = ("v8b", "v8a", "v7", "v6", "v1", "v5")
FALLBACK_PRIORITY = ("v8b", "v8a", "v7", "v6", "v5", "v1")


def path(relative: Path) -> Path:
    resolved = (REPO / relative).resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError(f"path escapes repository: {relative}")
    return resolved


def load_jsonl(relative: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path(relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def by_task_id(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{label} contains duplicate task IDs")
    return indexed


def write_jsonl(destination: Path, rows: list[dict[str, Any]]) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_png(file_path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(file_path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"{file_path}: expected PNG, got {image.format}")
        if image.mode != "RGB":
            raise ValueError(f"{file_path}: expected RGB, got {image.mode}")
        if image.size != expected_size:
            raise ValueError(
                f"{file_path}: size {image.size} != expected {expected_size}"
            )


def prepare_outputs() -> tuple[Path, Path, dict[Path, Path]]:
    final_dirs = (path(DELIVERY_148), path(DELIVERY_260))
    annotation_paths = (REVIEW_148, TASKS_148, REVIEW_260, TASKS_260)
    for final_dir in final_dirs:
        temporary = final_dir.with_name(final_dir.name + ".tmp")
        if final_dir.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {final_dir}")
    temporary_annotations: dict[Path, Path] = {}
    for relative in annotation_paths:
        final_file = path(relative)
        temporary = final_file.with_suffix(final_file.suffix + ".tmp")
        if final_file.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {final_file}")
        temporary_annotations[relative] = temporary
    temporary_148 = final_dirs[0].with_name(final_dirs[0].name + ".tmp")
    temporary_260 = final_dirs[1].with_name(final_dirs[1].name + ".tmp")
    temporary_148.mkdir(parents=True)
    temporary_260.mkdir(parents=True)
    return temporary_148, temporary_260, temporary_annotations


def select_remaining() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Path],
]:
    remaining = load_jsonl(REMAINING_TASKS)
    remaining_by_id = by_task_id(remaining, "remaining tasks")
    if len(remaining) != 148:
        raise ValueError(f"expected 148 remaining tasks, got {len(remaining)}")

    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    task_variants: dict[str, dict[str, dict[str, Any]]] = {}
    for revision, source in SOURCES.items():
        manifest_rows = load_jsonl(source.output_dir / "manifest.jsonl")
        manifests[revision] = by_task_id(
            manifest_rows, f"{revision} candidate manifest"
        )
        task_variants[revision] = by_task_id(
            load_jsonl(source.tasks), f"{revision} generation tasks"
        )

    remaining_ids = set(remaining_by_id)
    strict_ids = set().union(*STRICT_PASS.values())
    if not strict_ids <= remaining_ids:
        raise ValueError("strict-pass QA contains IDs outside the remaining 148")
    if len(strict_ids) != 94:
        raise ValueError(f"expected 94 strict passes, got {len(strict_ids)}")

    review_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_files: dict[str, Path] = {}

    for task in remaining:
        task_id = str(task["task_id"])
        strict_usable = task_id in strict_ids
        if strict_usable:
            revision = next(
                candidate
                for candidate in PASS_PRIORITY
                if task_id in STRICT_PASS[candidate]
            )
        else:
            revision = next(
                candidate
                for candidate in FALLBACK_PRIORITY
                if task_id in manifests[candidate]
            )

        source = SOURCES[revision]
        source_manifest = manifests[revision].get(task_id)
        source_task = task_variants[revision].get(task_id)
        if source_manifest is None or source_task is None:
            raise ValueError(f"{task_id}: incomplete {revision} provenance")
        source_file = path(Path(source_manifest["output_crop"]))
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        if source_manifest["input_context_crop"] != source_task["context_crop"]:
            raise ValueError(f"{task_id}: {revision} context provenance mismatch")
        expected_size = (
            int(source_task["crop_box"]["width"]),
            int(source_task["crop_box"]["height"]),
        )
        validate_png(source_file, expected_size)
        digest = sha256(source_file)
        review_status = "usable" if strict_usable else "needs_review"
        reason = (
            "strict visual QA pass: complete, naturally placed trash can with "
            "source-matched visual style"
            if strict_usable
            else "generated and retained; no retry passed the strict "
            "complete-and-natural visual QA threshold"
        )
        output_crop = DELIVERY_148 / f"{task_id}.png"
        source_files[task_id] = source_file
        task_rows.append(source_task)
        review_rows.append(
            {
                "task_id": task_id,
                "usable": strict_usable,
                "status": review_status,
                "selected_revision": revision,
                "selected_output": str(output_crop),
                "selected_from": str(source.output_dir),
                "source_context_crop": source_task["context_crop"],
                "sha256": digest,
                "reason": reason,
            }
        )
        manifest_rows.append(
            {
                **source_manifest,
                "output_crop": str(output_crop),
                "model": DELIVERY_148.name,
                "selected_from": str(source.output_dir),
                "selected_revision": revision,
                "original_output_crop": source_manifest["output_crop"],
                "strict_usable": strict_usable,
                "review_status": review_status,
                "sha256": digest,
            }
        )

    return review_rows, task_rows, manifest_rows, source_files


def build() -> dict[str, Any]:
    temporary_148, temporary_260, temporary_annotations = prepare_outputs()
    temporary_dirs = (temporary_148, temporary_260)
    try:
        (
            review_148,
            tasks_148,
            manifest_148,
            source_files_148,
        ) = select_remaining()
        for row in review_148:
            task_id = row["task_id"]
            shutil.copy2(source_files_148[task_id], temporary_148 / f"{task_id}.png")
        write_jsonl(temporary_148 / "manifest.jsonl", manifest_148)

        all_tasks = load_jsonl(ALL_TASKS)
        prior_tasks = load_jsonl(PRIOR_TASKS)
        prior_task_by_id = by_task_id(prior_tasks, "prior 112 tasks")
        prior_manifest_by_id = by_task_id(
            load_jsonl(PRIOR_DELIVERY / "manifest.jsonl"),
            "prior reviewed manifest",
        )
        prior_review_by_id = by_task_id(
            load_jsonl(PRIOR_REVIEW), "prior reviewed QA"
        )
        remaining_task_by_id = by_task_id(tasks_148, "reviewed 148 tasks")
        remaining_manifest_by_id = by_task_id(
            manifest_148, "reviewed 148 manifest"
        )
        remaining_review_by_id = by_task_id(review_148, "reviewed 148 QA")

        prior_ids = set(prior_task_by_id)
        remaining_ids = set(remaining_task_by_id)
        all_ids = {str(task["task_id"]) for task in all_tasks}
        if len(prior_ids) != 112 or len(remaining_ids) != 148:
            raise ValueError("unexpected 112/148 cohort size")
        if prior_ids & remaining_ids or prior_ids | remaining_ids != all_ids:
            raise ValueError("112 + 148 cohorts do not exactly cover all 260 tasks")

        review_260: list[dict[str, Any]] = []
        tasks_260: list[dict[str, Any]] = []
        manifest_260: list[dict[str, Any]] = []
        for original_task in all_tasks:
            task_id = str(original_task["task_id"])
            output_crop = DELIVERY_260 / f"{task_id}.png"
            if task_id in prior_ids:
                selected_task = prior_task_by_id[task_id]
                old_manifest = prior_manifest_by_id[task_id]
                old_review = prior_review_by_id[task_id]
                source_file = path(Path(old_manifest["output_crop"]))
                strict_usable = bool(old_review["usable"])
                review_status = (
                    "usable" if strict_usable else "needs_review"
                )
                digest = sha256(source_file)
                review_row = {
                    **old_review,
                    "status": review_status,
                    "usable": strict_usable,
                    "selected_output": str(output_crop),
                    "cohort": "prior_112",
                    "original_review_status": old_review["status"],
                    "sha256": digest,
                }
                manifest_row = {
                    **old_manifest,
                    "output_crop": str(output_crop),
                    "model": DELIVERY_260.name,
                    "copied_from_reviewed_delivery": str(PRIOR_DELIVERY),
                    "strict_usable": strict_usable,
                    "review_status": review_status,
                    "sha256": digest,
                }
            else:
                selected_task = remaining_task_by_id[task_id]
                old_manifest = remaining_manifest_by_id[task_id]
                old_review = remaining_review_by_id[task_id]
                source_file = source_files_148[task_id]
                strict_usable = bool(old_review["usable"])
                review_status = old_review["status"]
                digest = old_review["sha256"]
                review_row = {
                    **old_review,
                    "selected_output": str(output_crop),
                    "cohort": "remaining_148",
                }
                manifest_row = {
                    **old_manifest,
                    "output_crop": str(output_crop),
                    "model": DELIVERY_260.name,
                    "copied_from_reviewed_delivery": str(DELIVERY_148),
                }

            expected_size = (
                int(selected_task["crop_box"]["width"]),
                int(selected_task["crop_box"]["height"]),
            )
            validate_png(source_file, expected_size)
            shutil.copy2(source_file, temporary_260 / f"{task_id}.png")
            tasks_260.append(selected_task)
            review_260.append(review_row)
            manifest_260.append(manifest_row)

        write_jsonl(temporary_260 / "manifest.jsonl", manifest_260)
        write_jsonl(temporary_annotations[REVIEW_148], review_148)
        write_jsonl(temporary_annotations[TASKS_148], tasks_148)
        write_jsonl(temporary_annotations[REVIEW_260], review_260)
        write_jsonl(temporary_annotations[TASKS_260], tasks_260)

        temporary_148.replace(path(DELIVERY_148))
        temporary_260.replace(path(DELIVERY_260))
        for relative, temporary in temporary_annotations.items():
            temporary.replace(path(relative))
    except Exception:
        for temporary in temporary_dirs:
            if temporary.exists():
                shutil.rmtree(temporary)
        for temporary in temporary_annotations.values():
            temporary.unlink(missing_ok=True)
        raise

    strict_148 = sum(bool(row["usable"]) for row in review_148)
    strict_260 = sum(bool(row["usable"]) for row in review_260)
    return {
        "remaining_148": {
            "images": len(manifest_148),
            "strict_usable": strict_148,
            "needs_review": len(manifest_148) - strict_148,
            "directory": str(DELIVERY_148),
        },
        "full_260": {
            "images": len(manifest_260),
            "strict_usable": strict_260,
            "needs_review": len(manifest_260) - strict_260,
            "directory": str(DELIVERY_260),
        },
        "policy": "all tasks retained; QA status never removes an image",
    }


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
