#!/usr/bin/env python3
"""Build the frozen seven-condition CLAIMFORGE Balanced250 release.

The main panel contains 250 independently selected examples for each of:
real, three local insertion classes, and three full-frame conditional-edit
classes.  The score cache keeps all 275 curated real images so that every
selected forged image also has a genuine source-matched control.

Selection is deterministic and score-blind.  New model scores must never be
used to choose rows for this release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import PIL
from PIL import Image, ImageChops, ImageOps, features

from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


SCHEMA_VERSION = "claimforge_balanced250_canonical_v1"
DATASET_ID = "claimforge-balanced250-independent-panel-jpeg-q95-v1"
PANEL_SIZE = 250
EXPECTED_NEW_CANONICAL_IMAGES = 1775
EXPECTED_NEW_LOCAL_MASKS = 750

CONDITION_ORDER = (
    "real",
    "local_mouse",
    "local_cat",
    "local_trash_can",
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
FORGED_CONDITIONS = CONDITION_ORDER[1:]
LOCAL_CONDITIONS = (
    "local_mouse",
    "local_cat",
    "local_trash_can",
)
FULLFRAME_CONDITIONS = (
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
CANDIDATE_BY_CONDITION = {
    "local_mouse": "mouse",
    "local_cat": "cat",
    "local_trash_can": "trash_can",
    "fullframe_mouse": "mouse",
    "fullframe_cat": "cat",
    "fullframe_trash_can": "trash_can",
}
EXPECTED_ELIGIBLE_ROWS = {
    "real": 275,
    "local_mouse": 275,
    "local_cat": 251,
    "local_trash_can": 250,
    "fullframe_mouse": 275,
    "fullframe_cat": 272,
    "fullframe_trash_can": 260,
}

DEFAULT_OUTPUT_DIR = Path("outputs/opensource/balanced250_v1")
MOUSE_RELEASE_MANIFEST = Path("outputs/opensource/mouse_canonical_v1/manifest.json")
CAT_SELECTION = Path("annotations/claimforge_cat_final_251_selections.json")
CAT_MATERIALIZED = Path(
    "spliced_final/claimforge_cat_selected_251_20260725/manifest.jsonl"
)
TRASH_SELECTION = Path(
    "annotations/claimforge_trash_can_final_250_selections.json"
)
TRASH_MATERIALIZED = Path(
    "spliced_final/claimforge_trash_can_selected_250_20260725/manifest.jsonl"
)
WHOLE_TASKS = {
    "fullframe_mouse": Path(
        "annotations/full_image_orange_box_mouse_good275_latest_20260724.jsonl"
    ),
    "fullframe_cat": Path(
        "annotations/full_image_orange_box_cat_latest272_20260724.jsonl"
    ),
    "fullframe_trash_can": Path(
        "annotations/full_image_orange_box_trash_can_latest260_20260724.jsonl"
    ),
}
WHOLE_RUNS = {
    "fullframe_mouse": Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_mouse_good275_g5_v1_20260724/"
        "manifest.jsonl"
    ),
    "fullframe_cat": Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_cat_latest272_g5_v1_20260724/"
        "manifest.jsonl"
    ),
    "fullframe_trash_can": Path(
        "generated_full_images/"
        "hunyuan_image3_distil_full_input_orange_box_trash_can_latest260_g5_v1_20260724/"
        "manifest.jsonl"
    ),
}
TRASH_WHOLE_QC = Path(
    "annotations/full_image_orange_box_trash_can_single_shot_manual_qc_20260724.json"
)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _id_list_hash(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_task_id(task_id: str) -> str:
    if task_id.startswith("trash_can_"):
        return task_id[len("trash_can_") :]
    if task_id.startswith("cat_"):
        return task_id[len("cat_") :]
    return task_id


def _domain(normalized_task_id: str) -> str:
    prefix = normalized_task_id.split("_", 1)[0]
    if prefix not in {"lodging", "restaurant"}:
        raise ValueError(f"invalid task domain in {normalized_task_id}")
    return prefix


def _selection_key(condition: str, normalized_task_id: str) -> str:
    payload = f"{DATASET_ID}\0{condition}\0{normalized_task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_id(condition: str, normalized_task_id: str) -> str:
    payload = (
        f"{DATASET_ID}\0{condition}\0{normalized_task_id}\0sample".encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _pair_id(condition: str, normalized_task_id: str) -> str:
    payload = (
        f"{DATASET_ID}\0{condition}\0{normalized_task_id}\0source-pair".encode(
            "utf-8"
        )
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _resolve_repo_file(repo_root: Path, value: str | Path, label: str) -> Path:
    resolved = _resolve_repo_path(repo_root, value, label)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {value}")
    return resolved


def _resolve_repo_path(repo_root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository: {value}") from exc
    return resolved


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _image_file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    image = _load_rgb(path)
    return {
        **_file_record(path, repo_root),
        "decoded_width": image.width,
        "decoded_height": image.height,
    }


def _eligibility_set_hash(
    records: Iterable[dict[str, Any]],
    condition: str,
) -> str:
    materialized = list(records)
    normalized_ids = [str(row.get("normalized_task_id")) for row in materialized]
    if (
        any(not value or value == "None" for value in normalized_ids)
        or len(normalized_ids) != len(set(normalized_ids))
    ):
        raise ValueError(f"{condition} eligibility records have invalid identities")
    ordered = sorted(materialized, key=lambda row: str(row["normalized_task_id"]))
    return _rows_hash(ordered)


def _validate_binary_mask(
    path: Path,
    *,
    expected_size: tuple[int, int],
    expected_sha256: str,
    label: str,
) -> Image.Image:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    with Image.open(path) as opened:
        if opened.mode != "L":
            raise ValueError(f"{label} must use native L mode")
        opened.load()
        mask = opened.copy()
    if mask.size != expected_size:
        raise ValueError(f"{label} size mismatch: {mask.size} != {expected_size}")
    histogram = mask.histogram()
    if any(histogram[index] for index in range(1, 255)):
        raise ValueError(f"{label} is not binary 0/255")
    return mask


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _unique_by(
    rows: Iterable[dict[str, Any]],
    field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row has no valid {field}")
        if value in result:
            raise ValueError(f"duplicate {field}={value} in {label}")
        result[value] = row
    return result


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _atomic_save_image(image: Image.Image, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, **kwargs)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _assert_stage_inventory(
    staging: Path,
    *,
    expected_image_names: set[str],
    expected_mask_names: set[str],
    include_manifest: bool,
) -> None:
    if len(expected_image_names) != EXPECTED_NEW_CANONICAL_IMAGES:
        raise ValueError(
            f"expected {EXPECTED_NEW_CANONICAL_IMAGES} new canonical image names, got "
            f"{len(expected_image_names)}"
        )
    if len(expected_mask_names) != EXPECTED_NEW_LOCAL_MASKS:
        raise ValueError(
            f"expected {EXPECTED_NEW_LOCAL_MASKS} new local mask names, got "
            f"{len(expected_mask_names)}"
        )
    expected = {
        "inputs.jsonl",
        "panel.jsonl",
        "source_pairs.jsonl",
        *(f"images/{name}" for name in expected_image_names),
        *(f"masks/{name}" for name in expected_mask_names),
    }
    if include_manifest:
        expected.add("manifest.json")
    actual: set[str] = set()
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staging inventory contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(staging).as_posix())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "staging inventory mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )


def _canonicalize(image: Image.Image, destination: Path) -> None:
    rgb = image.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    _atomic_save_image(
        clean,
        destination,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    with Image.open(destination) as opened:
        if opened.format != "JPEG" or opened.mode != "RGB":
            raise ValueError(f"invalid canonical JPEG: {destination}")
        if opened.size != image.size:
            raise ValueError(
                f"canonical size changed: {opened.size} != {image.size}"
            )
        if opened.getexif():
            raise ValueError(f"canonical image retained EXIF: {destination}")
        allowed_info = {"jfif", "jfif_version", "jfif_unit", "jfif_density"}
        forbidden_info = sorted(set(opened.info) - allowed_info)
        if forbidden_info:
            raise ValueError(
                f"canonical image retained metadata {forbidden_info}: "
                f"{destination}"
            )


def _exact_diff_mask(source: Image.Image, forged: Image.Image) -> Image.Image:
    if source.size != forged.size:
        raise ValueError(f"local pair size mismatch: {source.size} != {forged.size}")
    red, green, blue = ImageChops.difference(source, forged).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    return maximum.point(lambda value: 255 if value > 0 else 0, mode="L")


def _mask_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return int(sum(histogram[1:]))


def _outside_box_pixels(mask: Image.Image, box: list[int] | None) -> int | None:
    if box is None:
        return None
    if len(box) != 4:
        raise ValueError("context box must have four coordinates")
    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = mask.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"context box is outside image bounds: {box} vs {mask.size}")
    return _mask_pixels(mask) - _mask_pixels(mask.crop((x1, y1, x2, y2)))


def _validated_box(
    box: Any,
    image_size: tuple[int, int],
    label: str,
) -> list[int]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{label} must be a four-coordinate list")
    values = [int(value) for value in box]
    x1, y1, x2, y2 = values
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{label} is outside {image_size}: {values}")
    return values


def _validated_declared_size(
    value: Any,
    actual_size: tuple[int, int],
    label: str,
) -> tuple[int, int]:
    """Validate both frozen size encodings used by the source ledgers."""

    if isinstance(value, dict):
        declared = (value.get("width"), value.get("height"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        declared = (value[0], value[1])
    else:
        raise ValueError(f"{label} is not a width/height object or pair")
    try:
        normalized = (int(declared[0]), int(declared[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has non-integer dimensions: {declared}") from exc
    if normalized != actual_size:
        raise ValueError(f"{label} mismatch: {normalized} != {actual_size}")
    return normalized


def _latest_rows(
    rows: list[dict[str, Any]],
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    latest: dict[str, dict[str, Any]] = {}
    latest_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label} row {index} has no task_id")
        latest[task_id] = row
        latest_index[task_id] = index
    return latest, latest_index


def _ranked_selection(
    rows: list[dict[str, Any]],
    condition: str,
    *,
    count: int,
    deduplicate_raw_sha: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ids = [_normalized_task_id(str(row["task_id"])) for row in rows]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"{condition} has duplicate normalized task IDs")
    selection_keys = [
        _selection_key(condition, normalized_task_id)
        for normalized_task_id in normalized_ids
    ]
    if len(selection_keys) != len(set(selection_keys)):
        raise ValueError(f"{condition} has a selection-key collision")
    ranked = sorted(
        rows,
        key=lambda row: (
            _selection_key(condition, _normalized_task_id(str(row["task_id"]))),
            _normalized_task_id(str(row["task_id"])),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for eligibility_rank, row in enumerate(ranked):
        row["_eligibility_rank"] = eligibility_rank
        row["_selection_key"] = _selection_key(
            condition,
            _normalized_task_id(str(row["task_id"])),
        )
        if deduplicate_raw_sha:
            digest = row.get("raw_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{condition} row has no frozen raw SHA-256")
            if digest in seen_content:
                continue
            seen_content.add(digest)
        if len(selected) < count:
            selected.append(row)
    if len(selected) < count and deduplicate_raw_sha:
        selected_ids = {id(row) for row in selected}
        for row in ranked:
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(
            f"{condition} cannot provide {count} selected rows; got {len(selected)}"
        )
    for selection_rank, row in enumerate(selected):
        row["_selection_rank"] = selection_rank
    return ranked, selected


def _source_contract(
    repo_root: Path,
    relative_path: Path,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    path = _resolve_repo_file(repo_root, relative_path, str(relative_path))
    value: dict[str, Any] = {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _canonical_reference(
    path: Path,
    *,
    repo_root: Path,
    raw_path: Path,
    canonical_origin: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical input: {path}")
    with Image.open(path) as opened:
        if opened.format != "JPEG" or opened.mode != "RGB" or opened.getexif():
            raise ValueError(f"invalid reused canonical JPEG: {path}")
        width, height = opened.size
    return {
        "canonical_path": repo_relative(path, repo_root),
        "canonical_sha256": sha256_file(path),
        "canonical_bytes": path.stat().st_size,
        "canonical_origin": canonical_origin,
        "width": width,
        "height": height,
        "raw_path": repo_relative(raw_path, repo_root),
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_path.stat().st_size,
    }


def _new_canonical_reference(
    *,
    image: Image.Image,
    raw_path: Path,
    stage_dir: Path,
    final_output_dir: Path,
    repo_root: Path,
    sample_id: str,
) -> dict[str, Any]:
    stage_path = stage_dir / "images" / f"{sample_id}.jpg"
    _canonicalize(image, stage_path)
    final_path = final_output_dir / "images" / f"{sample_id}.jpg"
    return {
        "canonical_path": repo_relative(final_path, repo_root),
        "canonical_sha256": sha256_file(stage_path),
        "canonical_bytes": stage_path.stat().st_size,
        "canonical_origin": "balanced250_v1_reencode",
        "width": image.width,
        "height": image.height,
        "raw_path": repo_relative(raw_path, repo_root),
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_path.stat().st_size,
    }


def _base_input_row(
    *,
    condition: str,
    task_id: str,
    normalized_task_id: str,
    selection_key: str,
    eligibility_rank: int,
    selection_rank: int | None,
    panel: bool,
    eligible_set_sha256: str,
    canonical: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "sample_id": _sample_id(condition, normalized_task_id),
        "condition": condition,
        "condition_family": (
            "real"
            if condition == "real"
            else "local_splice"
            if condition in LOCAL_CONDITIONS
            else "full_frame_conditional_edit"
        ),
        "kind": "real" if condition == "real" else "forged",
        "label": 0 if condition == "real" else 1,
        "manipulation_scope": (
            "authentic"
            if condition == "real"
            else "local_insertion"
            if condition in LOCAL_CONDITIONS
            else "conditional_full_frame_edit"
        ),
        "candidate": CANDIDATE_BY_CONDITION.get(condition),
        "task_id": task_id,
        "normalized_task_id": normalized_task_id,
        "domain": _domain(normalized_task_id),
        "selection_key": selection_key,
        "eligibility_rank": eligibility_rank,
        "selection_rank": selection_rank,
        "panel": panel,
        "eligible_set_sha256": eligible_set_sha256,
        **canonical,
    }


def _content_clusters(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["matched_source_raw_sha256"])].append(
            str(row["normalized_task_id"])
        )
    duplicate_clusters = [
        {"source_sha256": digest, "normalized_task_ids": sorted(task_ids)}
        for digest, task_ids in grouped.items()
        if len(task_ids) > 1
    ]
    duplicate_clusters.sort(key=lambda row: str(row["source_sha256"]))
    return {
        "rows": sum(len(task_ids) for task_ids in grouped.values()),
        "unique_source_sha256": len(grouped),
        "duplicate_cluster_count": len(duplicate_clusters),
        "duplicate_row_count": sum(
            len(row["normalized_task_ids"]) - 1 for row in duplicate_clusters
        ),
        "duplicate_clusters": duplicate_clusters,
    }


def _load_mouse_release(
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    manifest_path = _resolve_repo_file(
        repo_root,
        MOUSE_RELEASE_MANIFEST,
        "Mouse release manifest",
    )
    manifest = _load_json(manifest_path, "Mouse release manifest")
    if manifest.get("schema_version") != "claimforge_mouse_canonical_v1":
        raise ValueError("unexpected Mouse release schema")
    inputs_value = manifest.get("inputs_path")
    if not isinstance(inputs_value, str):
        raise ValueError("Mouse release has no inputs_path")
    inputs_path = _resolve_repo_file(repo_root, inputs_value, "Mouse inputs")
    if sha256_file(inputs_path) != manifest.get("inputs_sha256"):
        raise ValueError("Mouse inputs SHA-256 mismatch")
    rows = read_jsonl(inputs_path)
    if len(rows) != 550:
        raise ValueError(f"expected 550 Mouse input rows, got {len(rows)}")
    real = _unique_by(
        [row for row in rows if row.get("kind") == "real"],
        "task_id",
        "Mouse real inputs",
    )
    forged = _unique_by(
        [row for row in rows if row.get("kind") == "forged"],
        "task_id",
        "Mouse forged inputs",
    )
    if len(real) != 275 or set(real) != set(forged):
        raise ValueError("Mouse release is not 275 complete pairs")
    pairs_value = manifest.get("pairs_path")
    if not isinstance(pairs_value, str):
        raise ValueError("Mouse release has no pairs_path")
    pairs_path = _resolve_repo_file(repo_root, pairs_value, "Mouse pairs")
    if sha256_file(pairs_path) != manifest.get("pairs_sha256"):
        raise ValueError("Mouse pairs SHA-256 mismatch")
    pairs = read_jsonl(pairs_path)
    pair_by_task = _unique_by(pairs, "task_id", "Mouse pairs")
    if len(pairs) != 275 or set(pair_by_task) != set(real):
        raise ValueError("Mouse pairs are not 275 complete task links")
    return manifest, rows, real, forged, pairs, pair_by_task


def _materialized_local_rows(
    *,
    repo_root: Path,
    selection_path: Path,
    materialized_path: Path,
    tasks_path: Path,
    label: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    selection = _load_json(
        _resolve_repo_file(repo_root, selection_path, f"{label} selection"),
        f"{label} selection",
    )
    selected_rows = selection.get("selections")
    if not isinstance(selected_rows, list):
        raise ValueError(f"{label} selection has no selections array")
    selected = _unique_by(selected_rows, "task_id", f"{label} selections")
    counts = selection.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"{label} selection has no counts object")
    selection_methods = Counter(
        str(row.get("selection")) for row in selected_rows
    )
    if (
        counts.get("total_accepted") != len(selected_rows)
        or counts.get("selection_methods")
        != dict(sorted(selection_methods.items()))
    ):
        raise ValueError(f"{label} selection counts drifted")
    materialized_rows = read_jsonl(
        _resolve_repo_file(repo_root, materialized_path, f"{label} materialized")
    )
    materialized = _unique_by(
        materialized_rows,
        "task_id",
        f"{label} materialized",
    )
    tasks_rows = read_jsonl(
        _resolve_repo_file(repo_root, tasks_path, f"{label} frozen tasks")
    )
    tasks = _unique_by(tasks_rows, "task_id", f"{label} frozen tasks")
    if set(selected) != set(materialized):
        raise ValueError(f"{label} selection/materialized task sets differ")
    missing = sorted(set(materialized) - set(tasks))
    if missing:
        raise ValueError(f"{label} materialized tasks missing frozen task: {missing[0]}")
    return materialized_rows, tasks, selected


def _trash_qc(
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    qc = _load_json(
        _resolve_repo_file(repo_root, TRASH_WHOLE_QC, "Trash whole-frame QC"),
        "Trash whole-frame QC",
    )
    failures_value = qc.get("failures")
    if not isinstance(failures_value, list):
        raise ValueError("Trash whole-frame QC has no failures array")
    failures = _unique_by(
        failures_value,
        "task_id",
        "Trash whole-frame QC failures",
    )
    summary = qc.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Trash whole-frame QC has no summary")
    if summary.get("total") != 260 or summary.get("failed") != len(failures):
        raise ValueError("Trash whole-frame QC counts do not match failures")
    usable = summary.get("usable")
    if not isinstance(usable, int) or usable + len(failures) != 260:
        raise ValueError("Trash whole-frame QC usable/failed partition is invalid")
    return failures, qc


def _whole_sources(
    repo_root: Path,
    condition: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
]:
    task_rows = read_jsonl(
        _resolve_repo_file(
            repo_root,
            WHOLE_TASKS[condition],
            f"{condition} frozen tasks",
        )
    )
    tasks = _unique_by(task_rows, "task_id", f"{condition} frozen tasks")
    run_rows = read_jsonl(
        _resolve_repo_file(
            repo_root,
            WHOLE_RUNS[condition],
            f"{condition} run manifest",
        )
    )
    latest, indexes = _latest_rows(run_rows, f"{condition} run manifest")
    if set(tasks) != set(latest):
        missing = sorted(set(tasks) - set(latest))
        extra = sorted(set(latest) - set(tasks))
        raise ValueError(
            f"{condition} task/latest sets differ: missing={missing[:1]} extra={extra[:1]}"
        )
    for task_id, row in latest.items():
        if row.get("status") != "ok":
            raise ValueError(f"{condition} latest row is not ok: {task_id}")
    return task_rows, latest, indexes, run_rows


def build_release(
    *,
    repo_root: Path,
    output_dir: Path,
    panel_size: int = PANEL_SIZE,
) -> dict[str, Any]:
    """Build and atomically publish the Balanced250 release."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if panel_size != PANEL_SIZE:
        raise ValueError("balanced250 v1 requires panel_size=250")
    try:
        output_relative = output_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"output directory escapes repository: {output_dir}") from exc
    if not output_relative.parts:
        raise ValueError("output directory cannot be the repository root")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.build-",
            dir=output_dir.parent,
        )
    )
    try:
        manifest = _build_into(
            repo_root=repo_root,
            staging=staging,
            final_output_dir=output_dir,
        )
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_into(
    *,
    repo_root: Path,
    staging: Path,
    final_output_dir: Path,
) -> dict[str, Any]:
    (
        mouse_manifest,
        mouse_inputs,
        real_by_task,
        mouse_forged_by_task,
        mouse_pairs,
        mouse_pair_by_task,
    ) = _load_mouse_release(repo_root)
    real_eligible = [dict(row) for row in real_by_task.values()]
    mouse_eligible = [dict(row) for row in mouse_forged_by_task.values()]

    cat_materialized, cat_tasks, cat_selections = _materialized_local_rows(
        repo_root=repo_root,
        selection_path=CAT_SELECTION,
        materialized_path=CAT_MATERIALIZED,
        tasks_path=WHOLE_TASKS["fullframe_cat"],
        label="Cat",
    )
    trash_materialized, trash_tasks, trash_selections = _materialized_local_rows(
        repo_root=repo_root,
        selection_path=TRASH_SELECTION,
        materialized_path=TRASH_MATERIALIZED,
        tasks_path=WHOLE_TASKS["fullframe_trash_can"],
        label="Trash-can",
    )
    local_eligible = {
        "local_mouse": mouse_eligible,
        "local_cat": [dict(row) for row in cat_materialized],
        "local_trash_can": [dict(row) for row in trash_materialized],
    }

    whole_task_rows: dict[str, list[dict[str, Any]]] = {}
    whole_latest: dict[str, dict[str, dict[str, Any]]] = {}
    whole_indexes: dict[str, dict[str, int]] = {}
    whole_run_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in FULLFRAME_CONDITIONS:
        tasks_rows, latest, indexes, run_rows = _whole_sources(
            repo_root,
            condition,
        )
        whole_task_rows[condition] = tasks_rows
        whole_latest[condition] = latest
        whole_indexes[condition] = indexes
        whole_run_rows[condition] = run_rows

    whole_tasks_by_id = {
        "fullframe_mouse": _unique_by(
            whole_task_rows["fullframe_mouse"],
            "task_id",
            "Mouse whole tasks",
        ),
        "fullframe_cat": cat_tasks,
        "fullframe_trash_can": trash_tasks,
    }
    whole_eligible = {
        condition: [dict(row) for row in whole_task_rows[condition]]
        for condition in FULLFRAME_CONDITIONS
    }

    trash_failures, trash_qc = _trash_qc(repo_root)
    if not set(trash_failures).issubset(
        set(whole_tasks_by_id["fullframe_trash_can"])
    ):
        raise ValueError("Trash whole-frame QC contains unknown task IDs")

    eligible_pools = {
        "real": real_eligible,
        **local_eligible,
        **whole_eligible,
    }
    for condition in CONDITION_ORDER:
        actual = len(eligible_pools[condition])
        expected = EXPECTED_ELIGIBLE_ROWS[condition]
        if actual != expected:
            raise ValueError(
                f"{condition} eligible count drift: expected {expected}, got {actual}"
            )

    image_audit_cache: dict[Path, dict[str, Any]] = {}

    def audit_image(value: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
        path = _resolve_repo_file(repo_root, value, label)
        if path not in image_audit_cache:
            image_audit_cache[path] = _image_file_record(path, repo_root)
        return path, dict(image_audit_cache[path])

    eligibility_records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITION_ORDER
    }

    # Audit every Mouse eligible row, including the 25 local rows not selected
    # into the panel. The reused Mouse pair ledger is an independent frozen
    # source for exact-mask and source-link semantics.
    for task_id, real_row in real_by_task.items():
        normalized = _normalized_task_id(task_id)
        real_raw_path, real_raw = audit_image(
            str(real_row["raw_path"]),
            f"Mouse real raw {task_id}",
        )
        real_canonical_path, real_canonical = audit_image(
            str(real_row["canonical_path"]),
            f"Mouse real canonical {task_id}",
        )
        reused = _canonical_reference(
            real_canonical_path,
            repo_root=repo_root,
            raw_path=real_raw_path,
            canonical_origin="reused_mouse_canonical_v1_exact_bytes",
        )
        if (
            reused["raw_sha256"] != real_row.get("raw_sha256")
            or reused["canonical_sha256"] != real_row.get("canonical_sha256")
            or reused["canonical_bytes"] != real_row.get("canonical_bytes")
            or reused["width"] != real_row.get("width")
            or reused["height"] != real_row.get("height")
        ):
            raise ValueError(f"Mouse real frozen fields drifted: {task_id}")
        real_record = {
            "condition": "real",
            "task_id": task_id,
            "normalized_task_id": normalized,
            "source": real_raw,
            "canonical": real_canonical,
        }
        real_row["_eligibility_record"] = real_record
        eligibility_records["real"].append(real_record)

        forged_row = mouse_forged_by_task[task_id]
        forged_raw_path, forged_raw = audit_image(
            str(forged_row["raw_path"]),
            f"Mouse forged raw {task_id}",
        )
        forged_canonical_path, forged_canonical = audit_image(
            str(forged_row["canonical_path"]),
            f"Mouse forged canonical {task_id}",
        )
        forged_reused = _canonical_reference(
            forged_canonical_path,
            repo_root=repo_root,
            raw_path=forged_raw_path,
            canonical_origin="reused_mouse_canonical_v1_exact_bytes",
        )
        if (
            forged_reused["raw_sha256"] != forged_row.get("raw_sha256")
            or forged_reused["canonical_sha256"]
            != forged_row.get("canonical_sha256")
            or forged_reused["canonical_bytes"] != forged_row.get("canonical_bytes")
            or forged_reused["width"] != forged_row.get("width")
            or forged_reused["height"] != forged_row.get("height")
        ):
            raise ValueError(f"Mouse forged frozen fields drifted: {task_id}")
        if (
            (forged_raw["decoded_width"], forged_raw["decoded_height"])
            != (real_raw["decoded_width"], real_raw["decoded_height"])
        ):
            raise ValueError(f"Mouse raw pair size mismatch: {task_id}")
        pair = mouse_pair_by_task[task_id]
        if (
            pair.get("real") != {
                key: real_row.get(key)
                for key in (
                    "canonical_bytes",
                    "canonical_path",
                    "canonical_sha256",
                    "kind",
                    "label",
                    "raw_path",
                    "raw_sha256",
                    "sample_id",
                )
            }
            or pair.get("forged")
            != {
                key: forged_row.get(key)
                for key in (
                    "canonical_bytes",
                    "canonical_path",
                    "canonical_sha256",
                    "kind",
                    "label",
                    "raw_path",
                    "raw_sha256",
                    "sample_id",
                )
            }
        ):
            raise ValueError(f"Mouse pair nested provenance drifted: {task_id}")
        mask_path = _resolve_repo_file(
            repo_root,
            str(forged_row["gt_mask_path"]),
            f"Mouse GT mask {task_id}",
        )
        expected_mask_sha = str(forged_row.get("gt_mask_sha256"))
        if pair.get("gt_mask_sha256") != expected_mask_sha:
            raise ValueError(f"Mouse pair/input mask SHA differs: {task_id}")
        mask = _validate_binary_mask(
            mask_path,
            expected_size=(
                int(forged_row["width"]),
                int(forged_row["height"]),
            ),
            expected_sha256=expected_mask_sha,
            label=f"Mouse GT mask {task_id}",
        )
        exact = _exact_diff_mask(
            _load_rgb(real_raw_path),
            _load_rgb(forged_raw_path),
        )
        if ImageChops.difference(mask, exact).getbbox() is not None:
            raise ValueError(f"Mouse GT mask is not the exact decoded diff: {task_id}")
        positive = _mask_pixels(mask)
        context_box = _validated_box(
            forged_row.get("context_region_xyxy"),
            mask.size,
            f"Mouse context box {task_id}",
        )
        outside = _outside_box_pixels(mask, context_box)
        if (
            positive != forged_row.get("gt_positive_pixels")
            or positive != pair.get("gt_positive_pixels")
            or list(mask.getbbox() or ()) != pair.get("gt_bbox_xyxy")
            or outside != pair.get("gt_pixels_outside_context")
        ):
            raise ValueError(f"Mouse GT mask ledger fields drifted: {task_id}")
        mouse_record = {
            "condition": "local_mouse",
            "task_id": task_id,
            "normalized_task_id": normalized,
            "source": real_raw,
            "candidate": forged_raw,
            "canonical": forged_canonical,
            "mask": _file_record(mask_path, repo_root),
        }
        forged_row["_eligibility_record"] = mouse_record
        forged_row["_verified_pair"] = pair
        eligibility_records["local_mouse"].append(mouse_record)

    local_audit_sources = {
        "local_cat": (cat_tasks, cat_materialized, cat_selections),
        "local_trash_can": (
            trash_tasks,
            trash_materialized,
            trash_selections,
        ),
    }
    for condition in ("local_cat", "local_trash_can"):
        tasks, materialized_rows, selections = local_audit_sources[condition]
        materialized_by_task = _unique_by(
            materialized_rows,
            "task_id",
            f"{condition} materialized audit",
        )
        for task_id, delivered in materialized_by_task.items():
            normalized = _normalized_task_id(task_id)
            task = tasks[task_id]
            selection = selections[task_id]
            candidates = selection.get("candidates")
            if (
                selection.get("selection") != delivered.get("selection")
                or selection.get("selected_spliced_full")
                != delivered.get("source_image")
                or not isinstance(candidates, dict)
                or candidates.get(str(selection.get("selection")))
                != selection.get("selected_spliced_full")
            ):
                raise ValueError(
                    f"{condition} selection/materialized provenance drift: {task_id}"
                )
            provenance_value = str(delivered.get("source_image"))
            _resolve_repo_path(
                repo_root,
                provenance_value,
                f"{condition} candidate provenance {task_id}",
            )
            source_path, source = audit_image(
                str(task["source_image"]),
                f"{condition} frozen source {task_id}",
            )
            forged_path, candidate = audit_image(
                str(delivered["image"]),
                f"{condition} materialized image {task_id}",
            )
            if (
                delivered.get("domain") != _domain(normalized)
                or selection.get("domain") != _domain(normalized)
                or forged_path.name != f"{task_id}.png"
            ):
                raise ValueError(f"{condition} identity fields drifted: {task_id}")
            source_real = real_by_task.get(normalized)
            if source_real is None:
                raise ValueError(f"{condition} has no Mouse real source: {task_id}")
            if (
                source["path"] != source_real.get("raw_path")
                or source["sha256"] != source_real.get("raw_sha256")
            ):
                raise ValueError(f"{condition} frozen source binding drift: {task_id}")
            actual_size = (
                int(candidate["decoded_width"]),
                int(candidate["decoded_height"]),
            )
            if actual_size != (
                int(source["decoded_width"]),
                int(source["decoded_height"]),
            ):
                raise ValueError(f"{condition} decoded pair size mismatch: {task_id}")
            _validated_declared_size(
                delivered.get("image_size"),
                actual_size,
                f"{condition} materialized size {task_id}",
            )
            _validated_declared_size(
                task.get("image_size"),
                actual_size,
                f"{condition} frozen size {task_id}",
            )
            if delivered.get("bytes") != forged_path.stat().st_size:
                raise ValueError(f"{condition} materialized byte count drift: {task_id}")
            _validated_box(
                task.get("edit_region_xyxy"),
                actual_size,
                f"{condition} edit box {task_id}",
            )
            _validated_box(
                task.get("context_region_xyxy"),
                actual_size,
                f"{condition} context box {task_id}",
            )
            record = {
                "condition": condition,
                "task_id": task_id,
                "normalized_task_id": normalized,
                "selection": str(delivered["selection"]),
                "selected_candidate_path": provenance_value,
                "source": source,
                "candidate": candidate,
            }
            delivered["_eligibility_record"] = record
            eligibility_records[condition].append(record)

    expected_whole_metadata = {
        "fullframe_mouse": ("mouse", "mouse"),
        "fullframe_cat": ("cat", "cat"),
        "fullframe_trash_can": ("trash can", "trash-can"),
    }
    for condition in FULLFRAME_CONDITIONS:
        tasks = whole_tasks_by_id[condition]
        latest = whole_latest[condition]
        output_paths: set[str] = set()
        for task_id, task in tasks.items():
            normalized = _normalized_task_id(task_id)
            run = latest[task_id]
            source_path, source = audit_image(
                str(task["source_image"]),
                f"{condition} frozen source {task_id}",
            )
            output_path, candidate = audit_image(
                str(run["output_image"]),
                f"{condition} latest output {task_id}",
            )
            if output_path.name != f"{task_id}.png":
                raise ValueError(f"{condition} output basename drift: {task_id}")
            if candidate["path"] in output_paths:
                raise ValueError(f"{condition} reuses an output path: {task_id}")
            output_paths.add(str(candidate["path"]))
            source_real = real_by_task.get(normalized)
            if source_real is None:
                raise ValueError(f"{condition} has no Mouse real source: {task_id}")
            if (
                source["path"] != source_real.get("raw_path")
                or source["sha256"] != source_real.get("raw_sha256")
                or run.get("input_source_image") != source["path"]
                or run.get("input_source_sha256") != source["sha256"]
            ):
                raise ValueError(f"{condition} source binding drift: {task_id}")
            actual_size = (
                int(source["decoded_width"]),
                int(source["decoded_height"]),
            )
            if (
                int(candidate["decoded_width"]),
                int(candidate["decoded_height"]),
            ) != actual_size:
                raise ValueError(f"{condition} decoded output size mismatch: {task_id}")
            _validated_declared_size(
                task.get("image_size"),
                actual_size,
                f"{condition} frozen size {task_id}",
            )
            _validated_declared_size(
                run.get("original_size"),
                actual_size,
                f"{condition} run original size {task_id}",
            )
            edit_box = _validated_box(
                task.get("edit_region_xyxy"),
                actual_size,
                f"{condition} edit box {task_id}",
            )
            context_box = _validated_box(
                task.get("context_region_xyxy"),
                actual_size,
                f"{condition} context box {task_id}",
            )
            insert_box = task.get("insert_box")
            if not isinstance(insert_box, dict) or [
                int(insert_box.get("x", -1)),
                int(insert_box.get("y", -1)),
                int(insert_box.get("x", -1))
                + int(insert_box.get("width", -1)),
                int(insert_box.get("y", -1))
                + int(insert_box.get("height", -1)),
            ] != edit_box:
                raise ValueError(f"{condition} insert/edit box drift: {task_id}")
            if not (
                context_box[0] <= edit_box[0]
                and context_box[1] <= edit_box[1]
                and context_box[2] >= edit_box[2]
                and context_box[3] >= edit_box[3]
            ):
                raise ValueError(f"{condition} edit box escapes context: {task_id}")
            expected_candidate, expected_object_kind = expected_whole_metadata[
                condition
            ]
            if (
                run.get("input_mode") != "full-image-orange-box"
                or run.get("orange_box_xyxy") != edit_box
                or task.get("candidates") != expected_candidate
                or run.get("candidate") != expected_candidate
                or run.get("object_kind") != expected_object_kind
            ):
                raise ValueError(f"{condition} generation semantics drift: {task_id}")
            record = {
                "condition": condition,
                "task_id": task_id,
                "normalized_task_id": normalized,
                "source": source,
                "candidate": candidate,
                "latest_run_row_index": whole_indexes[condition][task_id],
                "conditioning_box_xyxy": edit_box,
                "input_mode": str(run["input_mode"]),
                "object_kind": str(run["object_kind"]),
            }
            task["_eligibility_record"] = record
            eligibility_records[condition].append(record)

    eligibility_hashes = {
        condition: _eligibility_set_hash(eligibility_records[condition], condition)
        for condition in CONDITION_ORDER
    }

    ranked: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    ranked["real"], selected["real"] = _ranked_selection(
        real_eligible,
        "real",
        count=PANEL_SIZE,
        deduplicate_raw_sha=True,
    )
    for condition in LOCAL_CONDITIONS:
        ranked[condition], selected[condition] = _ranked_selection(
            local_eligible[condition],
            condition,
            count=PANEL_SIZE,
        )
    for condition in FULLFRAME_CONDITIONS:
        ranked[condition], selected[condition] = _ranked_selection(
            whole_eligible[condition],
            condition,
            count=PANEL_SIZE,
        )

    selected_real_tasks = {
        _normalized_task_id(str(row["task_id"])) for row in selected["real"]
    }
    input_rows: list[dict[str, Any]] = []
    input_by_sample: dict[str, dict[str, Any]] = {}
    input_by_condition_task: dict[tuple[str, str], dict[str, Any]] = {}

    # Retain all 275 curated real inputs in the score cache.
    for row in ranked["real"]:
        task_id = str(row["task_id"])
        normalized = _normalized_task_id(task_id)
        raw_path = _resolve_repo_file(repo_root, str(row["raw_path"]), "Mouse real")
        canonical = _new_canonical_reference(
            image=_load_rgb(raw_path),
            raw_path=raw_path,
            stage_dir=staging,
            final_output_dir=final_output_dir,
            repo_root=repo_root,
            sample_id=_sample_id("real", normalized),
        )
        if canonical["raw_sha256"] != row.get("raw_sha256"):
            raise ValueError(f"Mouse real raw SHA mismatch: {task_id}")
        is_panel = normalized in selected_real_tasks
        selection_rank = (
            next(
                int(item["_selection_rank"])
                for item in selected["real"]
                if _normalized_task_id(str(item["task_id"])) == normalized
            )
            if is_panel
            else None
        )
        built = _base_input_row(
            condition="real",
            task_id=task_id,
            normalized_task_id=normalized,
            selection_key=str(row["_selection_key"]),
            eligibility_rank=int(row["_eligibility_rank"]),
            selection_rank=selection_rank,
            panel=is_panel,
            eligible_set_sha256=eligibility_hashes["real"],
            canonical=canonical,
        )
        built.update(
            {
                "matched_source_task_id": normalized,
                "matched_source_raw_path": canonical["raw_path"],
                "matched_source_raw_sha256": canonical["raw_sha256"],
                "gt_mask_kind": "all_zero",
                "gt_mask_path": None,
                "gt_mask_sha256": None,
                "gt_positive_pixels": 0,
                "support_semantics": "authentic_all_zero",
                "edit_region_xyxy": row.get("edit_region_xyxy"),
                "context_region_xyxy": row.get("context_region_xyxy"),
                "source_release_sample_id": row.get("sample_id"),
            }
        )
        input_rows.append(built)
        input_by_sample[str(built["sample_id"])] = built
        input_by_condition_task[("real", normalized)] = built

    def add_forged_input(built: dict[str, Any]) -> None:
        sample_id = str(built["sample_id"])
        key = (str(built["condition"]), str(built["normalized_task_id"]))
        if sample_id in input_by_sample or key in input_by_condition_task:
            raise ValueError(f"duplicate forged input identity: {key}")
        input_rows.append(built)
        input_by_sample[sample_id] = built
        input_by_condition_task[key] = built

    # Local Mouse reuses the already frozen canonical bytes and exact mask.
    for row in selected["local_mouse"]:
        task_id = str(row["task_id"])
        normalized = _normalized_task_id(task_id)
        source_real = real_by_task[normalized]
        source_pair = mouse_pair_by_task[task_id]
        raw_path = _resolve_repo_file(repo_root, str(row["raw_path"]), "Mouse forged")
        sample_id = _sample_id("local_mouse", normalized)
        canonical = _new_canonical_reference(
            image=_load_rgb(raw_path),
            raw_path=raw_path,
            stage_dir=staging,
            final_output_dir=final_output_dir,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        if canonical["raw_sha256"] != row.get("raw_sha256"):
            raise ValueError(f"Mouse forged raw SHA mismatch: {task_id}")
        mask_path = _resolve_repo_file(
            repo_root,
            str(row["gt_mask_path"]),
            "Mouse GT mask",
        )
        mask = _validate_binary_mask(
            mask_path,
            expected_size=(int(row["width"]), int(row["height"])),
            expected_sha256=str(row["gt_mask_sha256"]),
            label=f"Mouse GT mask {task_id}",
        )
        mask_pixels = _mask_pixels(mask)
        if mask_pixels != row.get("gt_positive_pixels"):
            raise ValueError(f"Mouse mask pixel count mismatch: {task_id}")
        stage_mask = staging / "masks" / f"{sample_id}.png"
        _atomic_save_image(mask, stage_mask, format="PNG", optimize=False)
        final_mask = final_output_dir / "masks" / f"{sample_id}.png"
        built = _base_input_row(
            condition="local_mouse",
            task_id=task_id,
            normalized_task_id=normalized,
            selection_key=str(row["_selection_key"]),
            eligibility_rank=int(row["_eligibility_rank"]),
            selection_rank=int(row["_selection_rank"]),
            panel=True,
            eligible_set_sha256=eligibility_hashes["local_mouse"],
            canonical=canonical,
        )
        built.update(
            {
                "matched_source_task_id": normalized,
                "matched_source_raw_path": str(source_real["raw_path"]),
                "matched_source_raw_sha256": str(source_real["raw_sha256"]),
                "gt_mask_kind": "exact_diff",
                "gt_mask_path": repo_relative(final_mask, repo_root),
                "gt_mask_sha256": sha256_file(stage_mask),
                "gt_positive_pixels": mask_pixels,
                "gt_bbox_xyxy": source_pair["gt_bbox_xyxy"],
                "gt_pixels_outside_context": source_pair[
                    "gt_pixels_outside_context"
                ],
                "gt_fraction": source_pair["gt_fraction"],
                "support_semantics": "decoded_source_vs_local_forged_exact_diff",
                "edit_region_xyxy": row.get("edit_region_xyxy"),
                "context_region_xyxy": row.get("context_region_xyxy"),
                "source_release_sample_id": source_real.get("sample_id"),
                "forged_source_release_sample_id": row.get("sample_id"),
                "local_selection_method": "mouse_human_review_good",
            }
        )
        add_forged_input(built)

    local_sources = {
        "local_cat": (
            cat_tasks,
            _unique_by(cat_materialized, "task_id", "Cat materialized"),
        ),
        "local_trash_can": (
            trash_tasks,
            _unique_by(trash_materialized, "task_id", "Trash materialized"),
        ),
    }
    for condition in ("local_cat", "local_trash_can"):
        tasks, materialized = local_sources[condition]
        for row in selected[condition]:
            task_id = str(row["task_id"])
            normalized = _normalized_task_id(task_id)
            task = tasks[task_id]
            delivered = materialized[task_id]
            source_real = real_by_task.get(normalized)
            if source_real is None:
                raise ValueError(f"{condition} has no matched Mouse real: {task_id}")
            source_path = _resolve_repo_file(
                repo_root,
                str(task["source_image"]),
                f"{condition} real source",
            )
            if repo_relative(source_path, repo_root) != source_real.get("raw_path"):
                raise ValueError(f"{condition} source path mismatch: {task_id}")
            if sha256_file(source_path) != source_real.get("raw_sha256"):
                raise ValueError(f"{condition} source SHA mismatch: {task_id}")
            forged_path = _resolve_repo_file(
                repo_root,
                str(delivered["image"]),
                f"{condition} final image",
            )
            source_image = _load_rgb(source_path)
            forged_image = _load_rgb(forged_path)
            if source_image.size != forged_image.size:
                raise ValueError(f"{condition} pair size mismatch: {task_id}")
            _validated_declared_size(
                task.get("image_size"),
                source_image.size,
                f"{condition} declared size {task_id}",
            )
            edit_box = _validated_box(
                task.get("edit_region_xyxy"),
                source_image.size,
                f"{condition} edit box {task_id}",
            )
            context_box = _validated_box(
                task.get("context_region_xyxy"),
                source_image.size,
                f"{condition} context box {task_id}",
            )
            diff = _exact_diff_mask(source_image, forged_image)
            positive_pixels = _mask_pixels(diff)
            if positive_pixels <= 0:
                raise ValueError(f"{condition} exact diff is empty: {task_id}")
            outside_context = _outside_box_pixels(diff, context_box)
            sample_id = _sample_id(condition, normalized)
            stage_mask = staging / "masks" / f"{sample_id}.png"
            _atomic_save_image(diff, stage_mask, format="PNG", optimize=False)
            final_mask = final_output_dir / "masks" / f"{sample_id}.png"
            canonical = _new_canonical_reference(
                image=forged_image,
                raw_path=forged_path,
                stage_dir=staging,
                final_output_dir=final_output_dir,
                repo_root=repo_root,
                sample_id=sample_id,
            )
            built = _base_input_row(
                condition=condition,
                task_id=task_id,
                normalized_task_id=normalized,
                selection_key=str(row["_selection_key"]),
                eligibility_rank=int(row["_eligibility_rank"]),
                selection_rank=int(row["_selection_rank"]),
                panel=True,
                eligible_set_sha256=eligibility_hashes[condition],
                canonical=canonical,
            )
            built.update(
                {
                    "matched_source_task_id": normalized,
                    "matched_source_raw_path": repo_relative(source_path, repo_root),
                    "matched_source_raw_sha256": sha256_file(source_path),
                    "gt_mask_kind": "exact_diff",
                    "gt_mask_path": repo_relative(final_mask, repo_root),
                    "gt_mask_sha256": sha256_file(stage_mask),
                    "gt_positive_pixels": positive_pixels,
                    "gt_bbox_xyxy": list(diff.getbbox()) if diff.getbbox() else None,
                    "gt_pixels_outside_context": outside_context,
                    "gt_fraction": positive_pixels
                    / (source_image.width * source_image.height),
                    "support_semantics": (
                        "decoded_source_vs_local_forged_exact_diff"
                    ),
                    "edit_region_xyxy": edit_box,
                    "context_region_xyxy": context_box,
                    "local_selection_method": delivered.get("selection"),
                    "local_materialized_manifest_path": (
                        CAT_MATERIALIZED.as_posix()
                        if condition == "local_cat"
                        else TRASH_MATERIALIZED.as_posix()
                    ),
                    "local_materialized_candidate_path": delivered.get(
                        "source_image"
                    ),
                }
            )
            add_forged_input(built)

    for condition in (
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ):
        tasks = whole_tasks_by_id[condition]
        latest = whole_latest[condition]
        indexes = whole_indexes[condition]
        for row in selected[condition]:
            task_id = str(row["task_id"])
            normalized = _normalized_task_id(task_id)
            task = tasks[task_id]
            run = latest[task_id]
            source_real = real_by_task.get(normalized)
            if source_real is None:
                raise ValueError(f"{condition} has no matched Mouse real: {task_id}")
            source_path = _resolve_repo_file(
                repo_root,
                str(task["source_image"]),
                f"{condition} source",
            )
            source_relative = repo_relative(source_path, repo_root)
            source_sha = sha256_file(source_path)
            if source_relative != source_real.get("raw_path"):
                raise ValueError(f"{condition} source path mismatch: {task_id}")
            if source_sha != source_real.get("raw_sha256"):
                raise ValueError(f"{condition} source SHA mismatch: {task_id}")
            if run.get("input_source_image") != source_relative:
                raise ValueError(f"{condition} run source path mismatch: {task_id}")
            if run.get("input_source_sha256") != source_sha:
                raise ValueError(f"{condition} run source SHA mismatch: {task_id}")
            output_path = _resolve_repo_file(
                repo_root,
                str(run["output_image"]),
                f"{condition} output",
            )
            source_image = _load_rgb(source_path)
            output_image = _load_rgb(output_path)
            if source_image.size != output_image.size:
                raise ValueError(f"{condition} output size mismatch: {task_id}")
            _validated_declared_size(
                task.get("image_size"),
                source_image.size,
                f"{condition} declared size {task_id}",
            )
            conditioning_box = _validated_box(
                task.get("edit_region_xyxy"),
                source_image.size,
                f"{condition} conditioning box {task_id}",
            )
            context_box = _validated_box(
                task.get("context_region_xyxy"),
                source_image.size,
                f"{condition} context box {task_id}",
            )
            sample_id = _sample_id(condition, normalized)
            canonical = _new_canonical_reference(
                image=output_image,
                raw_path=output_path,
                stage_dir=staging,
                final_output_dir=final_output_dir,
                repo_root=repo_root,
                sample_id=sample_id,
            )
            qc_status = "not_reviewed"
            qc_categories: list[str] = []
            qc_reason: str | None = None
            if condition == "fullframe_trash_can":
                failure = trash_failures.get(task_id)
                if failure is None:
                    qc_status = "usable"
                else:
                    qc_status = "failed"
                    categories = failure.get("categories")
                    if isinstance(categories, list):
                        qc_categories = [str(value) for value in categories]
                    reason = failure.get("reason")
                    qc_reason = str(reason) if reason is not None else None
            built = _base_input_row(
                condition=condition,
                task_id=task_id,
                normalized_task_id=normalized,
                selection_key=str(row["_selection_key"]),
                eligibility_rank=int(row["_eligibility_rank"]),
                selection_rank=int(row["_selection_rank"]),
                panel=True,
                eligible_set_sha256=eligibility_hashes[condition],
                canonical=canonical,
            )
            built.update(
                {
                    "matched_source_task_id": normalized,
                    "matched_source_raw_path": source_relative,
                    "matched_source_raw_sha256": source_sha,
                    "gt_mask_kind": "not_applicable",
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": None,
                    "support_semantics": (
                        "full_frame_conditional_edit_no_localization_target"
                    ),
                    "conditioning_box_xyxy": conditioning_box,
                    "context_region_xyxy": context_box,
                    "generation_manifest_path": WHOLE_RUNS[condition].as_posix(),
                    "generation_manifest_latest_row_index": indexes[task_id],
                    "generation_model": run.get("model"),
                    "generation_service_model": run.get("service_model"),
                    "generation_seed": run.get("seed"),
                    "generation_steps": run.get("steps"),
                    "generation_guidance_scale": run.get("guidance_scale"),
                    "generation_bot_task": run.get("bot_task"),
                    "fullframe_semantic_qc_status": qc_status,
                    "fullframe_semantic_qc_categories": qc_categories,
                    "fullframe_semantic_qc_reason": qc_reason,
                }
            )
            add_forged_input(built)

    source_cluster_counts = {
        condition: Counter(
            str(row["matched_source_raw_sha256"])
            for row in input_rows
            if row["condition"] == condition
        )
        for condition in CONDITION_ORDER
    }
    for row in input_rows:
        cluster = str(row["matched_source_raw_sha256"])
        cluster_size = source_cluster_counts[str(row["condition"])][cluster]
        row["source_content_cluster"] = cluster
        row["source_content_cluster_size_within_condition"] = cluster_size
        row["source_content_is_duplicated_within_condition"] = cluster_size > 1

    # Freeze a stable input order: all real cache rows, then six 250-row sets.
    condition_index = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    input_rows.sort(
        key=lambda row: (
            condition_index[str(row["condition"])],
            int(row["eligibility_rank"])
            if row["condition"] == "real"
            else int(row["selection_rank"]),
            str(row["normalized_task_id"]),
        )
    )
    for rank_value, row in enumerate(input_rows):
        row["rank"] = rank_value
    if len(input_rows) != 1775:
        raise ValueError(f"expected 1775 score-cache rows, got {len(input_rows)}")

    panel_rows: list[dict[str, Any]] = []
    panel_conditions: Counter[str] = Counter()
    for condition in CONDITION_ORDER:
        rows = [
            row
            for row in input_rows
            if row["condition"] == condition and bool(row["panel"])
        ]
        rows.sort(key=lambda row: int(row["selection_rank"]))
        if len(rows) != PANEL_SIZE:
            raise ValueError(f"{condition} panel has {len(rows)} rows")
        for condition_rank, row in enumerate(rows):
            panel_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "panel_rank": len(panel_rows),
                    "condition": condition,
                    "condition_rank": condition_rank,
                    "sample_id": row["sample_id"],
                    "input_rank": row["rank"],
                    "task_id": row["task_id"],
                    "normalized_task_id": row["normalized_task_id"],
                    "label": row["label"],
                    "domain": row["domain"],
                    "kind": row["kind"],
                    "condition_family": row["condition_family"],
                    "manipulation_scope": row["manipulation_scope"],
                    "selection_key": row["selection_key"],
                    "eligible_set_sha256": row["eligible_set_sha256"],
                    "canonical_path": row["canonical_path"],
                    "canonical_sha256": row["canonical_sha256"],
                    "canonical_bytes": row["canonical_bytes"],
                    "width": row["width"],
                    "height": row["height"],
                    "source_content_cluster": row["source_content_cluster"],
                    "source_content_cluster_size_within_condition": row[
                        "source_content_cluster_size_within_condition"
                    ],
                    "gt_mask_kind": row["gt_mask_kind"],
                    "gt_mask_path": row["gt_mask_path"],
                    "gt_mask_sha256": row["gt_mask_sha256"],
                    "gt_positive_pixels": row["gt_positive_pixels"],
                }
            )
            panel_conditions[condition] += 1
    if len(panel_rows) != 1750 or set(panel_conditions.values()) != {PANEL_SIZE}:
        raise ValueError("panel is not seven complete 250-row conditions")

    source_pair_rows: list[dict[str, Any]] = []
    for condition in FORGED_CONDITIONS:
        rows = [
            row for row in input_rows if row["condition"] == condition
        ]
        rows.sort(key=lambda row: int(row["selection_rank"]))
        if len(rows) != PANEL_SIZE:
            raise ValueError(f"{condition} has incomplete selected forged rows")
        for condition_pair_rank, forged in enumerate(rows):
            normalized = str(forged["normalized_task_id"])
            real = input_by_condition_task.get(("real", normalized))
            if real is None:
                raise ValueError(f"{condition} has no cached matched real: {normalized}")
            if (
                real["raw_sha256"] != forged["matched_source_raw_sha256"]
                or real["raw_path"] != forged["matched_source_raw_path"]
            ):
                raise ValueError(f"{condition} matched real mismatch: {normalized}")
            source_pair_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "rank": len(source_pair_rows),
                    "pair_id": _pair_id(condition, normalized),
                    "condition": condition,
                    "pair_rank": len(source_pair_rows),
                    "condition_pair_rank": condition_pair_rank,
                    "normalized_task_id": normalized,
                    "domain": forged["domain"],
                    "selection_key": forged["selection_key"],
                    "eligible_set_sha256": forged["eligible_set_sha256"],
                    "real_sample_id": real["sample_id"],
                    "forged_sample_id": forged["sample_id"],
                    "real": {
                        "canonical_path": real["canonical_path"],
                        "canonical_sha256": real["canonical_sha256"],
                        "canonical_bytes": real["canonical_bytes"],
                        "width": real["width"],
                        "height": real["height"],
                    },
                    "forged": {
                        "canonical_path": forged["canonical_path"],
                        "canonical_sha256": forged["canonical_sha256"],
                        "canonical_bytes": forged["canonical_bytes"],
                        "width": forged["width"],
                        "height": forged["height"],
                    },
                    "source_raw_path": real["raw_path"],
                    "source_raw_sha256": real["raw_sha256"],
                    "source_content_cluster": real["raw_sha256"],
                    "source_content_cluster_size_within_condition": forged[
                        "source_content_cluster_size_within_condition"
                    ],
                    "comparison_design": "source_matched_secondary",
                }
            )
    if len(source_pair_rows) != 1500:
        raise ValueError(
            f"expected 1500 source-pair rows, got {len(source_pair_rows)}"
        )

    inputs_path = staging / "inputs.jsonl"
    panel_path = staging / "panel.jsonl"
    pairs_path = staging / "source_pairs.jsonl"
    atomic_write_jsonl(inputs_path, input_rows)
    atomic_write_jsonl(panel_path, panel_rows)
    atomic_write_jsonl(pairs_path, source_pair_rows)
    expected_image_names = {
        Path(str(row["canonical_path"])).name
        for row in input_rows
        if row["canonical_origin"] == "balanced250_v1_reencode"
    }
    expected_mask_names = {
        Path(str(row["gt_mask_path"])).name
        for row in input_rows
        if row["condition"] in LOCAL_CONDITIONS
    }
    _assert_stage_inventory(
        staging,
        expected_image_names=expected_image_names,
        expected_mask_names=expected_mask_names,
        include_manifest=False,
    )

    source_contracts = {
        "mouse_release_manifest": _source_contract(
            repo_root,
            MOUSE_RELEASE_MANIFEST,
        ),
        "mouse_inputs": _source_contract(
            repo_root,
            Path(str(mouse_manifest["inputs_path"])),
            rows=len(mouse_inputs),
        ),
        "mouse_pairs": _source_contract(
            repo_root,
            Path(str(mouse_manifest["pairs_path"])),
            rows=len(mouse_pairs),
        ),
        "cat_selection": _source_contract(repo_root, CAT_SELECTION),
        "cat_materialized": _source_contract(
            repo_root,
            CAT_MATERIALIZED,
            rows=len(cat_materialized),
        ),
        "trash_selection": _source_contract(repo_root, TRASH_SELECTION),
        "trash_materialized": _source_contract(
            repo_root,
            TRASH_MATERIALIZED,
            rows=len(trash_materialized),
        ),
        "trash_whole_qc": _source_contract(repo_root, TRASH_WHOLE_QC),
    }
    for condition in FULLFRAME_CONDITIONS:
        source_contracts[f"{condition}_tasks"] = _source_contract(
            repo_root,
            WHOLE_TASKS[condition],
            rows=len(whole_task_rows[condition]),
        )
        source_contracts[f"{condition}_run"] = _source_contract(
            repo_root,
            WHOLE_RUNS[condition],
            rows=len(whole_run_rows[condition]),
        )

    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_ORDER:
        eligible_rows = ranked[condition]
        selected_rows = selected[condition]
        selected_ids = [
            _normalized_task_id(str(row["task_id"])) for row in selected_rows
        ]
        input_condition_rows = [
            row for row in input_rows if row["condition"] == condition
        ]
        domain_counts = Counter(str(row["domain"]) for row in input_condition_rows)
        source_cluster_summary = _content_clusters(input_condition_rows)
        summary: dict[str, Any] = {
            "eligible_rows": len(eligible_rows),
            "expected_eligible_rows": EXPECTED_ELIGIBLE_ROWS[condition],
            "eligible_set_sha256": eligibility_hashes[condition],
            "cache_rows": len(input_condition_rows),
            "panel_rows": sum(bool(row["panel"]) for row in input_condition_rows),
            "eligible_normalized_task_ids_sha256": _id_list_hash(
                sorted(
                    _normalized_task_id(str(row["task_id"]))
                    for row in eligible_rows
                )
            ),
            "selected_normalized_task_ids_sha256": _id_list_hash(selected_ids),
            "selection_key_sha256": _id_list_hash(
                str(row["_selection_key"]) for row in selected_rows
            ),
            "domains": dict(sorted(domain_counts.items())),
            "source_content": source_cluster_summary,
        }
        if condition in LOCAL_CONDITIONS:
            summary["gt_positive_pixels"] = sum(
                int(row["gt_positive_pixels"]) for row in input_condition_rows
            )
            summary["gt_pixels_outside_context"] = sum(
                int(row.get("gt_pixels_outside_context") or 0)
                for row in input_condition_rows
            )
            summary["rows_with_gt_outside_context"] = sum(
                int(row.get("gt_pixels_outside_context") or 0) > 0
                for row in input_condition_rows
            )
        if condition == "fullframe_trash_can":
            summary["semantic_qc"] = dict(
                sorted(
                    Counter(
                        str(row["fullframe_semantic_qc_status"])
                        for row in input_condition_rows
                    ).items()
                )
            )
        condition_summaries[condition] = summary

    deterministic_contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "design": {
            "primary": "independent_seven_condition_panel",
            "secondary": "source_matched_six_condition_pairs",
            "panel_conditions": list(CONDITION_ORDER),
            "panel_rows_per_condition": PANEL_SIZE,
            "real_cache_rows": 275,
            "forged_cache_rows_per_condition": PANEL_SIZE,
            "self_contained_canonical_inputs": True,
            "release_canonical_images": EXPECTED_NEW_CANONICAL_IMAGES,
            "release_local_masks": EXPECTED_NEW_LOCAL_MASKS,
        },
        "selection": {
            "score_blind": True,
            "key": (
                "sha256(dataset_id + NUL + condition + NUL + "
                "normalized_task_id)"
            ),
            "collision_policy": "reject duplicate selection keys",
            "real_policy": (
                "rank all eligible real tasks by key; retain the first task "
                "per raw_sha256 until 250 content-unique panel rows are selected"
            ),
            "forged_policy": "first 250 eligible unique normalized task IDs",
            "semantic_qc_used_for_selection": False,
        },
        "canonicalization": {
            "decode": "Pillow ImageOps.exif_transpose then RGB",
            "format": "JPEG",
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "resize": False,
            "all_inputs_reencoded_from_frozen_raw": True,
            "encoder": {
                "pillow": PIL.__version__,
                "libjpeg": features.version_codec("jpg"),
            },
        },
        "localization": {
            "local_conditions": sorted(LOCAL_CONDITIONS),
            "mask_space": "decoded_pre_canonicalization_rgb",
            "mask_rule": "max_abs_rgb_difference_gt_0",
            "context_box_is_not_ground_truth": True,
            "fullframe_gt_mask_kind": "not_applicable",
        },
        "ledgers": {
            "inputs": {
                "path": repo_relative(
                    final_output_dir / "inputs.jsonl",
                    repo_root,
                ),
                "rows": len(input_rows),
                "sha256": sha256_file(inputs_path),
            },
            "panel": {
                "path": repo_relative(
                    final_output_dir / "panel.jsonl",
                    repo_root,
                ),
                "rows": len(panel_rows),
                "sha256": sha256_file(panel_path),
            },
            "source_pairs": {
                "path": repo_relative(
                    final_output_dir / "source_pairs.jsonl",
                    repo_root,
                ),
                "rows": len(source_pair_rows),
                "sha256": sha256_file(pairs_path),
            },
        },
        "source_contracts": source_contracts,
        "conditions": condition_summaries,
        "fullframe_semantics": {
            "label": "conditional_full_frame_edit",
            "fully_synthetic": False,
            "trash_primary_qc_summary": trash_qc["summary"],
        },
    }
    manifest = {
        **deterministic_contract,
        "contract_sha256": _stable_hash(deterministic_contract),
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "output_dir": repo_relative(final_output_dir, repo_root),
        "inputs_rows": len(input_rows),
        "panel_rows": len(panel_rows),
        "source_pair_rows": len(source_pair_rows),
        "new_canonical_images": len(expected_image_names),
        "new_local_masks": len(expected_mask_names),
        "status": "complete",
    }
    atomic_write_json(staging / "manifest.json", manifest)
    _assert_stage_inventory(
        staging,
        expected_image_names=expected_image_names,
        expected_mask_names=expected_mask_names,
        include_manifest=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (repo_root / args.output_dir).resolve()
    )
    manifest = build_release(
        repo_root=repo_root,
        output_dir=output_dir,
        panel_size=args.panel_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
