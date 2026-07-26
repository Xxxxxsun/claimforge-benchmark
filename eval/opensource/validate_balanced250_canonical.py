#!/usr/bin/env python3
"""Validate the frozen balanced250 canonical release independently.

The validator intentionally does not import the dataset builder.  It
recomputes the release identities, hashes, selections, image properties, and
localization targets from the frozen files and their source ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import PIL
from PIL import Image, ImageChops, ImageOps, JpegImagePlugin, features

from eval.opensource.common import (
    atomic_write_json,
    sha256_file,
    stable_json,
    utc_now,
)


SCHEMA_VERSION = "claimforge_balanced250_canonical_v1"
DATASET_ID = "claimforge-balanced250-independent-panel-jpeg-q95-v1"
DEFAULT_RELEASE_DIR = Path("outputs/opensource/balanced250_v1")

REAL_CONDITION = "real"
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
FORGED_CONDITIONS = LOCAL_CONDITIONS + FULLFRAME_CONDITIONS
CONDITIONS = (REAL_CONDITION,) + FORGED_CONDITIONS
CANDIDATE_BY_CONDITION = {
    "local_mouse": "mouse",
    "local_cat": "cat",
    "local_trash_can": "trash_can",
    "fullframe_mouse": "mouse",
    "fullframe_cat": "cat",
    "fullframe_trash_can": "trash_can",
}
SOURCE_CONTRACT_NAMES = {
    "mouse_release_manifest",
    "mouse_inputs",
    "mouse_pairs",
    "cat_selection",
    "cat_materialized",
    "trash_selection",
    "trash_materialized",
    "trash_whole_qc",
    "fullframe_mouse_tasks",
    "fullframe_mouse_run",
    "fullframe_cat_tasks",
    "fullframe_cat_run",
    "fullframe_trash_can_tasks",
    "fullframe_trash_can_run",
}


class ValidationError(ValueError):
    """The release violates its frozen contract."""


@dataclass(frozen=True)
class ValidationSpec:
    """Expected row counts.

    ``FROZEN_SPEC`` is the only specification exposed by the CLI.  Tests use a
    deliberately small specification so fail-closed cases do not need to
    materialize thousands of images.
    """

    real_cache: int
    forged_cache_per_condition: int
    panel_per_condition: int
    local_cat_eligible: int
    local_trash_can_eligible: int
    fullframe_cat_eligible: int
    fullframe_trash_can_eligible: int

    @property
    def inputs(self) -> int:
        return self.real_cache + len(FORGED_CONDITIONS) * (
            self.forged_cache_per_condition
        )

    @property
    def panel(self) -> int:
        return len(CONDITIONS) * self.panel_per_condition

    @property
    def source_pairs(self) -> int:
        return len(FORGED_CONDITIONS) * self.forged_cache_per_condition

    @property
    def local_masks(self) -> int:
        return len(LOCAL_CONDITIONS) * self.forged_cache_per_condition

    @property
    def new_canonical_images(self) -> int:
        return self.inputs

    @property
    def new_local_masks(self) -> int:
        return self.local_masks

    @property
    def cache_counts(self) -> dict[str, int]:
        return {
            REAL_CONDITION: self.real_cache,
            **{
                condition: self.forged_cache_per_condition
                for condition in FORGED_CONDITIONS
            },
        }

    @property
    def panel_counts(self) -> dict[str, int]:
        return {
            condition: self.panel_per_condition for condition in CONDITIONS
        }

    @property
    def source_pair_counts(self) -> dict[str, int]:
        return {
            condition: self.forged_cache_per_condition
            for condition in FORGED_CONDITIONS
        }

    @property
    def eligible_counts(self) -> dict[str, int]:
        return {
            REAL_CONDITION: self.real_cache,
            "local_mouse": self.real_cache,
            "local_cat": self.local_cat_eligible,
            "local_trash_can": self.local_trash_can_eligible,
            "fullframe_mouse": self.real_cache,
            "fullframe_cat": self.fullframe_cat_eligible,
            "fullframe_trash_can": self.fullframe_trash_can_eligible,
        }


FROZEN_SPEC = ValidationSpec(
    real_cache=275,
    forged_cache_per_condition=250,
    panel_per_condition=250,
    local_cat_eligible=251,
    local_trash_can_eligible=250,
    fullframe_cat_eligible=272,
    fullframe_trash_can_eligible=260,
)


def _fail(message: str) -> None:
    raise ValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON number: {value}")


def _loads_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except ValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label}: malformed JSON: {exc}") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = _loads_json(path.read_text(encoding="utf-8"), label)
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"{label}: cannot read UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label}: expected a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"{label}: cannot read UTF-8 JSONL: {exc}") from exc
    if text and not text.endswith("\n"):
        _fail(f"{label}: JSONL must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            _fail(f"{label}:{line_number}: blank JSONL line")
        value = _loads_json(line, f"{label}:{line_number}")
        if not isinstance(value, dict):
            _fail(f"{label}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        _fail(f"{label}: expected lowercase SHA-256")
    return str(value)


def _expect_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}: expected a non-empty string")
    if "\x00" in value:
        _fail(f"{label}: NUL is forbidden")
    return value


def _expect_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label}: expected an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label}: expected value >= {minimum}")
    return value


def _expect_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: expected a number")
    number = float(value)
    if not (float("-inf") < number < float("inf")):
        _fail(f"{label}: expected a finite number")
    return number


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    relative = _expect_str(value, label)
    if "\\" in relative:
        _fail(f"{label}: path must use POSIX separators")
    pure = PurePosixPath(relative)
    if pure.is_absolute():
        _fail(f"{label}: absolute paths are forbidden")
    if (
        pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"{label}: non-canonical or traversing path")
    lexical = repo_root.joinpath(*pure.parts)
    current = repo_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{label}: symlink paths are forbidden")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValidationError(f"{label}: path escapes repository") from exc
    if not resolved.is_file():
        _fail(f"{label}: missing file: {relative}")
    return resolved


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    """Validate a repository-relative provenance path that may be unavailable."""

    relative = _expect_str(value, label)
    if "\\" in relative:
        _fail(f"{label}: path must use POSIX separators")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"{label}: non-canonical, absolute, or traversing path")
    lexical = repo_root.joinpath(*pure.parts)
    current = repo_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{label}: symlink paths are forbidden")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValidationError(f"{label}: path escapes repository") from exc
    return resolved


def _safe_release_file(
    repo_root: Path,
    release_dir: Path,
    value: Any,
    label: str,
    expected_parent: str | None = None,
) -> Path:
    path = _safe_repo_file(repo_root, value, label)
    try:
        relative = path.relative_to(release_dir)
    except ValueError as exc:
        raise ValidationError(f"{label}: path is outside release directory") from exc
    if expected_parent is not None:
        _require(
            relative.parent == Path(expected_parent),
            f"{label}: expected a direct child of {expected_parent}/",
        )
    return path


def _resolved_release_dir(repo_root: Path, release_dir: Path) -> Path:
    resolved = (
        release_dir.resolve()
        if release_dir.is_absolute()
        else (repo_root / release_dir).resolve()
    )
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValidationError("release directory escapes repository") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail("release directory may not contain symlink components")
    if not resolved.is_dir():
        _fail(f"missing release directory: {resolved}")
    return resolved


def _verify_file(path: Path, expected_hash: Any, label: str) -> str:
    expected = _expect_sha256(expected_hash, f"{label} hash")
    actual = sha256_file(path)
    if actual != expected:
        _fail(f"{label}: SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(repo_root).as_posix(),
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


def _validated_declared_size(
    value: Any,
    actual: tuple[int, int],
    label: str,
) -> tuple[int, int]:
    if isinstance(value, dict):
        declared = (value.get("width"), value.get("height"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        declared = (value[0], value[1])
    else:
        _fail(f"{label}: expected width/height object or pair")
    try:
        normalized = (int(declared[0]), int(declared[1]))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}: non-integer dimensions") from exc
    _require(normalized == actual, f"{label}: declared dimensions mismatch")
    return normalized


def _rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    text = "".join(f"{stable_json(row)}\n" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _id_list_hash(values: Iterable[str]) -> str:
    text = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _eligibility_set_hash(
    records: Iterable[Mapping[str, Any]],
    condition: str,
) -> str:
    materialized = list(records)
    identities = [
        _expect_str(
            row.get("normalized_task_id"),
            f"{condition}.eligibility.normalized_task_id",
        )
        for row in materialized
    ]
    _require(
        len(identities) == len(set(identities)),
        f"{condition}: duplicate eligibility identities",
    )
    ordered = sorted(materialized, key=lambda row: str(row["normalized_task_id"]))
    return _rows_hash(ordered)


def _unique_by(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        value = _expect_str(row.get(field), f"{label}[{index}].{field}")
        if value in result:
            _fail(f"{label}: duplicate {field}={value}")
        result[value] = row
    return result


def _latest_by_task(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, int]]:
    latest: dict[str, Mapping[str, Any]] = {}
    indexes: dict[str, int] = {}
    for index, row in enumerate(rows):
        task_id = _expect_str(row.get("task_id"), f"{label}[{index}].task_id")
        latest[task_id] = row
        indexes[task_id] = index
    return latest, indexes


def _rank_eligible(
    rows: Iterable[Mapping[str, Any]],
    condition: str,
    count: int,
    *,
    deduplicate_raw_sha: bool = False,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    materialized = list(rows)
    normalized = [
        _normalized_task_id(
            _expect_str(row.get("task_id"), f"{condition}.task_id")
        )
        for row in materialized
    ]
    _require(
        len(normalized) == len(set(normalized)),
        f"{condition}: duplicate normalized task IDs",
    )
    selection_keys = [
        _selection_key(DATASET_ID, condition, task_id) for task_id in normalized
    ]
    _require(
        len(selection_keys) == len(set(selection_keys)),
        f"{condition}: selection-key collision",
    )
    ranked = sorted(
        materialized,
        key=lambda row: (
            _selection_key(
                DATASET_ID,
                condition,
                _normalized_task_id(str(row["task_id"])),
            ),
            _normalized_task_id(str(row["task_id"])),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    seen_raw: set[str] = set()
    for row in ranked:
        if deduplicate_raw_sha:
            digest = _expect_sha256(
                row.get("raw_sha256"),
                f"{condition}.{row.get('task_id')}.raw_sha256",
            )
            if digest in seen_raw:
                continue
            seen_raw.add(digest)
        if len(selected) < count:
            selected.append(row)
    if len(selected) < count and deduplicate_raw_sha:
        selected_identities = {
            _normalized_task_id(str(row["task_id"])) for row in selected
        }
        for row in ranked:
            normalized_id = _normalized_task_id(str(row["task_id"]))
            if normalized_id not in selected_identities:
                selected.append(row)
                selected_identities.add(normalized_id)
                if len(selected) == count:
                    break
    _require(
        len(selected) == count,
        f"{condition}: cannot select {count} eligible rows",
    )
    return ranked, selected


def _content_clusters(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    row_count = 0
    for row in rows:
        digest = _expect_sha256(
            row.get("matched_source_raw_sha256"),
            "matched_source_raw_sha256",
        )
        task_id = _expect_str(
            row.get("normalized_task_id"),
            "normalized_task_id",
        )
        grouped.setdefault(digest, []).append(task_id)
        row_count += 1
    duplicates = [
        {
            "source_sha256": digest,
            "normalized_task_ids": sorted(task_ids),
        }
        for digest, task_ids in grouped.items()
        if len(task_ids) > 1
    ]
    duplicates.sort(key=lambda row: str(row["source_sha256"]))
    return {
        "rows": row_count,
        "unique_source_sha256": len(grouped),
        "duplicate_cluster_count": len(duplicates),
        "duplicate_row_count": sum(
            len(row["normalized_task_ids"]) - 1 for row in duplicates
        ),
        "duplicate_clusters": duplicates,
    }


def _expect_schema_and_dataset(
    row: Mapping[str, Any],
    dataset_id: str,
    label: str,
) -> None:
    _require(
        row.get("schema_version") == SCHEMA_VERSION,
        f"{label}: wrong schema_version",
    )
    _require(row.get("dataset_id") == dataset_id, f"{label}: wrong dataset_id")


def _expect_contiguous_ranks(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    label: str,
) -> None:
    ranks = [
        _expect_int(row.get(field), f"{label}[{index}].{field}", minimum=0)
        for index, row in enumerate(rows)
    ]
    expected = list(range(len(rows)))
    if ranks != expected:
        _fail(f"{label}: {field} must be contiguous physical order 0..N-1")


def _selection_key(dataset_id: str, condition: str, task_id: str) -> str:
    payload = f"{dataset_id}\0{condition}\0{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_id(dataset_id: str, condition: str, task_id: str) -> str:
    payload = f"{dataset_id}\0{condition}\0{task_id}\0sample".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _pair_id(dataset_id: str, condition: str, task_id: str) -> str:
    payload = (
        f"{dataset_id}\0{condition}\0{task_id}\0source-pair".encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _normalized_task_id(task_id: str) -> str:
    if task_id.startswith("trash_can_"):
        return task_id[len("trash_can_") :]
    if task_id.startswith("cat_"):
        return task_id[len("cat_") :]
    return task_id


def _domain(normalized_task_id: str) -> str:
    value = normalized_task_id.split("_", 1)[0]
    if value not in {"lodging", "restaurant"}:
        _fail(f"invalid normalized task domain: {normalized_task_id}")
    return value


def _load_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.info.clear()
            return image
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot decode image {path}: {exc}") from exc


def _q95_tables() -> dict[int, list[int]]:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as opened:
        return {
            int(key): [int(item) for item in table]
            for key, table in opened.quantization.items()
        }


Q95_TABLES = _q95_tables()


def _reencode_canonical(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    return buffer.getvalue()


def _validate_jpeg(
    path: Path,
    row: Mapping[str, Any],
    label: str,
    *,
    allow_comment: bool = False,
) -> None:
    width = _expect_int(row.get("width"), f"{label}.width", minimum=1)
    height = _expect_int(row.get("height"), f"{label}.height", minimum=1)
    try:
        with Image.open(path) as opened:
            opened.load()
            _require(opened.format == "JPEG", f"{label}: canonical input is not JPEG")
            _require(opened.mode == "RGB", f"{label}: canonical JPEG is not RGB")
            _require(
                opened.size == (width, height),
                f"{label}: canonical dimensions do not match row",
            )
            _require(not opened.getexif(), f"{label}: canonical JPEG contains EXIF")
            _require(
                JpegImagePlugin.get_sampling(opened) == 0,
                f"{label}: canonical JPEG is not 4:4:4",
            )
            tables = {
                int(key): [int(item) for item in table]
                for key, table in opened.quantization.items()
            }
            _require(tables == Q95_TABLES, f"{label}: JPEG is not quality 95")
            allowed_metadata = {
                "jfif",
                "jfif_version",
                "jfif_unit",
                "jfif_density",
            }
            forbidden_metadata = set(opened.info) - allowed_metadata
            if allow_comment:
                forbidden_metadata.discard("comment")
            _require(
                not forbidden_metadata,
                f"{label}: forbidden JPEG metadata: {sorted(forbidden_metadata)}",
            )
    except ValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{label}: invalid canonical JPEG: {exc}") from exc


def _exact_diff_mask(source: Image.Image, forged: Image.Image) -> Image.Image:
    if source.size != forged.size:
        _fail(f"local source/forged size mismatch: {source.size} != {forged.size}")
    red, green, blue = ImageChops.difference(source, forged).split()
    maximum = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    return maximum.point(lambda value: 255 if value > 0 else 0, mode="L")


def _mask_positive_pixels(mask: Image.Image) -> int:
    return int(sum(mask.histogram()[1:]))


def _validate_box(value: Any, size: tuple[int, int], label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"{label}: expected four coordinates")
    box = [_expect_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    x1, y1, x2, y2 = box
    width, height = size
    _require(
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
        f"{label}: box is outside image bounds",
    )
    return box


def _count_outside(mask: Image.Image, box: Sequence[int] | None) -> int | None:
    if box is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in box)
    total = _mask_positive_pixels(mask)
    inside = _mask_positive_pixels(mask.crop((x1, y1, x2, y2)))
    return total - inside


def _validate_inventory(
    release_dir: Path,
    expected_files: Iterable[Path],
) -> tuple[int, int]:
    expected = {path.resolve() for path in expected_files}
    actual: set[Path] = set()
    directories: set[Path] = set()
    for path in release_dir.rglob("*"):
        if path.is_symlink():
            _fail(
                "release inventory contains a symlink: "
                f"{path.relative_to(release_dir).as_posix()}"
            )
        if path.is_dir():
            directories.add(path.resolve())
        elif path.is_file():
            actual.add(path.resolve())
        else:
            _fail(
                "release inventory contains an unsupported entry: "
                f"{path.relative_to(release_dir).as_posix()}"
            )
    missing = sorted(
        path.relative_to(release_dir).as_posix() for path in expected - actual
    )
    extra = sorted(
        path.relative_to(release_dir).as_posix() for path in actual - expected
    )
    if missing or extra:
        _fail(
            "release inventory mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected_directories = {release_dir.resolve()}
    for path in expected:
        parent = path.parent
        while parent != release_dir.parent:
            expected_directories.add(parent)
            if parent == release_dir:
                break
            parent = parent.parent
    extra_directories = sorted(
        path.relative_to(release_dir).as_posix()
        for path in directories - expected_directories
    )
    if extra_directories:
        _fail(f"release inventory has extra directories: {extra_directories[:5]}")
    return len(actual), len(directories)


@dataclass
class SourceData:
    paths: dict[str, Path]
    jsonl: dict[str, list[dict[str, Any]]]
    json: dict[str, dict[str, Any]]
    eligible: dict[str, list[Mapping[str, Any]]]
    selected: dict[str, list[Mapping[str, Any]]]
    eligibility_ranks: dict[str, dict[str, int]]
    selection_ranks: dict[str, dict[str, int]]
    eligibility_hashes: dict[str, str]
    mouse_real: dict[str, Mapping[str, Any]]
    mouse_forged: dict[str, Mapping[str, Any]]
    mouse_pairs: dict[str, Mapping[str, Any]]
    local_tasks: dict[str, dict[str, Mapping[str, Any]]]
    local_materialized: dict[str, dict[str, Mapping[str, Any]]]
    whole_tasks: dict[str, dict[str, Mapping[str, Any]]]
    whole_latest: dict[str, dict[str, Mapping[str, Any]]]
    whole_latest_indexes: dict[str, dict[str, int]]
    trash_failures: dict[str, Mapping[str, Any]]


def _validate_source_contracts(
    manifest: Mapping[str, Any],
    repo_root: Path,
    spec: ValidationSpec,
) -> SourceData:
    contracts_value = manifest.get("source_contracts")
    if not isinstance(contracts_value, dict):
        _fail("manifest.source_contracts must be an object")
    _require(
        set(contracts_value) == SOURCE_CONTRACT_NAMES,
        "manifest.source_contracts has missing or extra ledgers",
    )

    paths: dict[str, Path] = {}
    jsonl_values: dict[str, list[dict[str, Any]]] = {}
    json_values: dict[str, dict[str, Any]] = {}
    jsonl_names = {
        "mouse_inputs",
        "mouse_pairs",
        "cat_materialized",
        "trash_materialized",
        "fullframe_mouse_tasks",
        "fullframe_mouse_run",
        "fullframe_cat_tasks",
        "fullframe_cat_run",
        "fullframe_trash_can_tasks",
        "fullframe_trash_can_run",
    }
    json_names = SOURCE_CONTRACT_NAMES - jsonl_names
    seen_paths: set[Path] = set()
    for name in sorted(SOURCE_CONTRACT_NAMES):
        contract = contracts_value.get(name)
        if not isinstance(contract, dict):
            _fail(f"source_contracts.{name} must be an object")
        required_keys = {"path", "sha256", "bytes"}
        allowed_keys = required_keys | ({"rows"} if name in jsonl_names else set())
        _require(
            set(contract) == allowed_keys,
            f"source_contracts.{name} has missing or extra fields",
        )
        path = _safe_repo_file(
            repo_root,
            contract.get("path"),
            f"source_contracts.{name}.path",
        )
        _require(path not in seen_paths, f"duplicate source ledger path: {name}")
        seen_paths.add(path)
        _verify_file(path, contract.get("sha256"), f"source ledger {name}")
        _require(
            path.stat().st_size
            == _expect_int(
                contract.get("bytes"),
                f"source_contracts.{name}.bytes",
                minimum=0,
            ),
            f"source ledger {name}: byte count mismatch",
        )
        paths[name] = path
        if name in jsonl_names:
            rows = _read_jsonl(path, f"source ledger {name}")
            _require(
                len(rows)
                == _expect_int(
                    contract.get("rows"),
                    f"source_contracts.{name}.rows",
                    minimum=0,
                ),
                f"source ledger {name}: row count mismatch",
            )
            jsonl_values[name] = rows
        elif name in json_names:
            json_values[name] = _read_json(path, f"source ledger {name}")

    mouse_manifest = json_values["mouse_release_manifest"]
    _require(
        mouse_manifest.get("schema_version") == "claimforge_mouse_canonical_v1",
        "source Mouse release has the wrong schema",
    )
    mouse_inputs_path = _safe_repo_file(
        repo_root,
        mouse_manifest.get("inputs_path"),
        "source Mouse manifest inputs_path",
    )
    _require(
        mouse_inputs_path == paths["mouse_inputs"],
        "Mouse source manifest inputs_path does not match source contract",
    )
    _verify_file(
        mouse_inputs_path,
        mouse_manifest.get("inputs_sha256"),
        "source Mouse inputs",
    )
    mouse_inputs = jsonl_values["mouse_inputs"]
    _require(
        len(mouse_inputs) == 2 * spec.real_cache,
        "source Mouse input row count is not the frozen paired count",
    )
    mouse_real_raw = [
        row for row in mouse_inputs if row.get("kind") == "real"
    ]
    mouse_forged_raw = [
        row for row in mouse_inputs if row.get("kind") == "forged"
    ]
    mouse_real_by_task = _unique_by(
        mouse_real_raw,
        "task_id",
        "source Mouse real rows",
    )
    mouse_forged_by_task = _unique_by(
        mouse_forged_raw,
        "task_id",
        "source Mouse forged rows",
    )
    _require(
        len(mouse_real_by_task) == spec.real_cache
        and set(mouse_real_by_task) == set(mouse_forged_by_task),
        "source Mouse ledger is not a complete frozen pair set",
    )
    mouse_pairs_path = _safe_repo_file(
        repo_root,
        mouse_manifest.get("pairs_path"),
        "source Mouse manifest pairs_path",
    )
    _require(
        mouse_pairs_path == paths["mouse_pairs"],
        "Mouse source manifest pairs_path does not match source contract",
    )
    _verify_file(
        mouse_pairs_path,
        mouse_manifest.get("pairs_sha256"),
        "source Mouse pairs",
    )
    mouse_pairs_by_task = _unique_by(
        jsonl_values["mouse_pairs"],
        "task_id",
        "source Mouse pairs",
    )
    _require(
        len(mouse_pairs_by_task) == spec.real_cache
        and set(mouse_pairs_by_task) == set(mouse_real_by_task),
        "source Mouse pair ledger task closure mismatch",
    )

    local_tasks: dict[str, dict[str, Mapping[str, Any]]] = {}
    local_materialized: dict[str, dict[str, Mapping[str, Any]]] = {}
    local_selections: dict[str, dict[str, Mapping[str, Any]]] = {}
    for condition, prefix in (
        ("local_cat", "cat"),
        ("local_trash_can", "trash"),
    ):
        selection = json_values[f"{prefix}_selection"].get("selections")
        if not isinstance(selection, list):
            _fail(f"{prefix}_selection.selections must be an array")
        selection_by_task = _unique_by(
            selection,
            "task_id",
            f"{prefix} source selections",
        )
        materialized_by_task = _unique_by(
            jsonl_values[f"{prefix}_materialized"],
            "task_id",
            f"{prefix} source materialized",
        )
        _require(
            set(selection_by_task) == set(materialized_by_task),
            f"{prefix} selection/materialized task sets differ",
        )
        tasks_name = (
            "fullframe_cat_tasks"
            if condition == "local_cat"
            else "fullframe_trash_can_tasks"
        )
        tasks_by_task = _unique_by(
            jsonl_values[tasks_name],
            "task_id",
            f"{prefix} frozen tasks",
        )
        _require(
            set(materialized_by_task) <= set(tasks_by_task),
            f"{prefix} materialized rows lack frozen source tasks",
        )
        local_tasks[condition] = tasks_by_task
        local_materialized[condition] = materialized_by_task
        local_selections[condition] = selection_by_task

    whole_tasks: dict[str, dict[str, Mapping[str, Any]]] = {}
    whole_latest: dict[str, dict[str, Mapping[str, Any]]] = {}
    whole_indexes: dict[str, dict[str, int]] = {}
    for condition in FULLFRAME_CONDITIONS:
        task_name = f"{condition}_tasks"
        run_name = f"{condition}_run"
        tasks_by_id = _unique_by(
            jsonl_values[task_name],
            "task_id",
            f"{condition} source tasks",
        )
        latest, indexes = _latest_by_task(
            jsonl_values[run_name],
            f"{condition} source run",
        )
        _require(
            set(tasks_by_id) == set(latest),
            f"{condition}: task/latest generation sets differ",
        )
        for task_id, row in latest.items():
            _require(
                row.get("status") == "ok",
                f"{condition}: latest generation is not ok: {task_id}",
            )
        whole_tasks[condition] = tasks_by_id
        whole_latest[condition] = latest
        whole_indexes[condition] = indexes

    qc = json_values["trash_whole_qc"]
    failures_value = qc.get("failures")
    if not isinstance(failures_value, list):
        _fail("trash_whole_qc.failures must be an array")
    trash_failures = _unique_by(
        failures_value,
        "task_id",
        "trash whole-frame QC failures",
    )
    qc_summary = qc.get("summary")
    if not isinstance(qc_summary, dict):
        _fail("trash_whole_qc.summary must be an object")
    _require(
        qc_summary.get("total") == len(whole_tasks["fullframe_trash_can"])
        and qc_summary.get("failed") == len(trash_failures),
        "trash whole-frame QC summary is inconsistent",
    )
    _require(
        isinstance(qc_summary.get("usable"), int)
        and qc_summary.get("usable") + len(trash_failures)
        == len(whole_tasks["fullframe_trash_can"]),
        "trash whole-frame QC usable/failed partition is inconsistent",
    )
    _require(
        set(trash_failures) <= set(whole_tasks["fullframe_trash_can"]),
        "trash whole-frame QC contains unknown tasks",
    )

    eligible: dict[str, list[Mapping[str, Any]]] = {
        "real": list(mouse_real_by_task.values()),
        "local_mouse": list(mouse_forged_by_task.values()),
        "local_cat": list(local_materialized["local_cat"].values()),
        "local_trash_can": list(
            local_materialized["local_trash_can"].values()
        ),
        **{
            condition: list(whole_tasks[condition].values())
            for condition in FULLFRAME_CONDITIONS
        },
    }
    selected: dict[str, list[Mapping[str, Any]]] = {}
    eligibility_ranks: dict[str, dict[str, int]] = {}
    selection_ranks: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        _require(
            len(eligible[condition]) == spec.eligible_counts[condition],
            f"{condition}: source eligible count drift",
        )
        ranked, chosen = _rank_eligible(
            eligible[condition],
            condition,
            spec.panel_per_condition,
            deduplicate_raw_sha=(condition == REAL_CONDITION),
        )
        eligible[condition] = ranked
        selected[condition] = chosen
        eligibility_ranks[condition] = {
            _normalized_task_id(str(row["task_id"])): index
            for index, row in enumerate(ranked)
        }
        selection_ranks[condition] = {
            _normalized_task_id(str(row["task_id"])): index
            for index, row in enumerate(chosen)
        }

    image_records: dict[Path, dict[str, Any]] = {}

    def audited_image(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
        path = _safe_repo_file(repo_root, value, label)
        if path not in image_records:
            image_records[path] = _image_file_record(path, repo_root)
        return path, dict(image_records[path])

    eligibility_records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for task_id, real_row in mouse_real_by_task.items():
        normalized = _normalized_task_id(task_id)
        real_raw_path, real_raw = audited_image(
            real_row.get("raw_path"),
            f"Mouse real raw {task_id}",
        )
        real_canonical_path, real_canonical = audited_image(
            real_row.get("canonical_path"),
            f"Mouse real canonical {task_id}",
        )
        _verify_file(
            real_raw_path,
            real_row.get("raw_sha256"),
            f"Mouse real raw {task_id}",
        )
        _verify_file(
            real_canonical_path,
            real_row.get("canonical_sha256"),
            f"Mouse real canonical {task_id}",
        )
        _require(
            real_canonical_path.stat().st_size == real_row.get("canonical_bytes"),
            f"Mouse real canonical byte drift: {task_id}",
        )
        _validate_jpeg(
            real_canonical_path,
            real_row,
            f"Mouse real canonical {task_id}",
            allow_comment=True,
        )
        eligibility_records["real"].append(
            {
                "condition": "real",
                "task_id": task_id,
                "normalized_task_id": normalized,
                "source": real_raw,
                "canonical": real_canonical,
            }
        )

        forged_row = mouse_forged_by_task[task_id]
        forged_raw_path, forged_raw = audited_image(
            forged_row.get("raw_path"),
            f"Mouse forged raw {task_id}",
        )
        forged_canonical_path, forged_canonical = audited_image(
            forged_row.get("canonical_path"),
            f"Mouse forged canonical {task_id}",
        )
        _verify_file(
            forged_raw_path,
            forged_row.get("raw_sha256"),
            f"Mouse forged raw {task_id}",
        )
        _verify_file(
            forged_canonical_path,
            forged_row.get("canonical_sha256"),
            f"Mouse forged canonical {task_id}",
        )
        _require(
            forged_canonical_path.stat().st_size
            == forged_row.get("canonical_bytes"),
            f"Mouse forged canonical byte drift: {task_id}",
        )
        _validate_jpeg(
            forged_canonical_path,
            forged_row,
            f"Mouse forged canonical {task_id}",
            allow_comment=True,
        )
        _require(
            (real_raw["decoded_width"], real_raw["decoded_height"])
            == (forged_raw["decoded_width"], forged_raw["decoded_height"]),
            f"Mouse raw pair size mismatch: {task_id}",
        )
        pair = mouse_pairs_by_task[task_id]
        expected_real = {
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
        expected_forged = {
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
        _require(
            pair.get("real") == expected_real
            and pair.get("forged") == expected_forged,
            f"Mouse nested pair provenance drift: {task_id}",
        )
        mask_path = _safe_repo_file(
            repo_root,
            forged_row.get("gt_mask_path"),
            f"Mouse GT mask {task_id}",
        )
        mask_sha = _verify_file(
            mask_path,
            forged_row.get("gt_mask_sha256"),
            f"Mouse GT mask {task_id}",
        )
        _require(
            pair.get("gt_mask_sha256") == mask_sha,
            f"Mouse pair/input mask SHA mismatch: {task_id}",
        )
        try:
            with Image.open(mask_path) as opened:
                opened.load()
                _require(opened.format == "PNG", "Mouse GT mask must be PNG")
                _require(opened.mode == "L", "Mouse GT mask must be mode L")
                _require(
                    sum(opened.histogram()[1:255]) == 0,
                    "Mouse GT mask must be binary",
                )
                mask = opened.copy()
        except ValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise ValidationError(f"invalid Mouse mask {task_id}: {exc}") from exc
        exact = _exact_diff_mask(
            _load_rgb(real_raw_path),
            _load_rgb(forged_raw_path),
        )
        _require(
            ImageChops.difference(mask, exact).getbbox() is None,
            f"Mouse GT mask is not exact decoded diff: {task_id}",
        )
        positive = _mask_positive_pixels(mask)
        context = _validate_box(
            forged_row.get("context_region_xyxy"),
            mask.size,
            f"Mouse context {task_id}",
        )
        outside = _count_outside(mask, context)
        _require(
            forged_row.get("gt_positive_pixels") == positive
            and pair.get("gt_positive_pixels") == positive
            and pair.get("gt_bbox_xyxy")
            == (list(mask.getbbox()) if mask.getbbox() else None)
            and pair.get("gt_pixels_outside_context") == outside,
            f"Mouse GT ledger metrics drift: {task_id}",
        )
        eligibility_records["local_mouse"].append(
            {
                "condition": "local_mouse",
                "task_id": task_id,
                "normalized_task_id": normalized,
                "source": real_raw,
                "candidate": forged_raw,
                "canonical": forged_canonical,
                "mask": _file_record(mask_path, repo_root),
            }
        )

    for condition in ("local_cat", "local_trash_can"):
        for task_id, delivered in local_materialized[condition].items():
            normalized = _normalized_task_id(task_id)
            task = local_tasks[condition][task_id]
            selection = local_selections[condition][task_id]
            _require(
                selection.get("selection") == delivered.get("selection")
                and selection.get("selected_spliced_full")
                == delivered.get("source_image"),
                f"{condition}: selection/materialized provenance drift: {task_id}",
            )
            provenance = _expect_str(
                delivered.get("source_image"),
                f"{condition}.{task_id}.source_image",
            )
            _safe_repo_path(
                repo_root,
                provenance,
                f"{condition}.{task_id}.candidate provenance",
            )
            source_path, source = audited_image(
                task.get("source_image"),
                f"{condition} source {task_id}",
            )
            candidate_path, candidate = audited_image(
                delivered.get("image"),
                f"{condition} materialized {task_id}",
            )
            mouse_real = mouse_real_by_task.get(normalized)
            _require(
                mouse_real is not None
                and source["path"] == mouse_real.get("raw_path")
                and source["sha256"] == mouse_real.get("raw_sha256"),
                f"{condition}: Mouse real source binding drift: {task_id}",
            )
            size = (
                int(candidate["decoded_width"]),
                int(candidate["decoded_height"]),
            )
            _require(
                size
                == (
                    int(source["decoded_width"]),
                    int(source["decoded_height"]),
                ),
                f"{condition}: decoded pair size mismatch: {task_id}",
            )
            _validated_declared_size(
                delivered.get("image_size"),
                size,
                f"{condition} materialized size {task_id}",
            )
            _validated_declared_size(
                task.get("image_size"),
                size,
                f"{condition} task size {task_id}",
            )
            _require(
                delivered.get("bytes") == candidate_path.stat().st_size,
                f"{condition}: materialized byte count drift: {task_id}",
            )
            _validate_box(
                task.get("edit_region_xyxy"),
                size,
                f"{condition} edit box {task_id}",
            )
            _validate_box(
                task.get("context_region_xyxy"),
                size,
                f"{condition} context box {task_id}",
            )
            eligibility_records[condition].append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "normalized_task_id": normalized,
                    "selection": str(delivered["selection"]),
                    "selected_candidate_path": provenance,
                    "source": source,
                    "candidate": candidate,
                }
            )

    expected_whole = {
        "fullframe_mouse": ("mouse", "mouse"),
        "fullframe_cat": ("cat", "cat"),
        "fullframe_trash_can": ("trash can", "trash-can"),
    }
    for condition in FULLFRAME_CONDITIONS:
        seen_outputs: set[str] = set()
        for task_id, task in whole_tasks[condition].items():
            normalized = _normalized_task_id(task_id)
            run = whole_latest[condition][task_id]
            source_path, source = audited_image(
                task.get("source_image"),
                f"{condition} source {task_id}",
            )
            output_path, candidate = audited_image(
                run.get("output_image"),
                f"{condition} output {task_id}",
            )
            _require(
                candidate["path"] not in seen_outputs,
                f"{condition}: output path reused: {task_id}",
            )
            seen_outputs.add(str(candidate["path"]))
            mouse_real = mouse_real_by_task.get(normalized)
            _require(
                mouse_real is not None
                and source["path"] == mouse_real.get("raw_path")
                and source["sha256"] == mouse_real.get("raw_sha256")
                and run.get("input_source_image") == source["path"]
                and run.get("input_source_sha256") == source["sha256"],
                f"{condition}: source binding drift: {task_id}",
            )
            size = (
                int(source["decoded_width"]),
                int(source["decoded_height"]),
            )
            _require(
                (
                    int(candidate["decoded_width"]),
                    int(candidate["decoded_height"]),
                )
                == size,
                f"{condition}: decoded output size mismatch: {task_id}",
            )
            _validated_declared_size(
                task.get("image_size"),
                size,
                f"{condition} task size {task_id}",
            )
            _validated_declared_size(
                run.get("original_size"),
                size,
                f"{condition} run original_size {task_id}",
            )
            edit = _validate_box(
                task.get("edit_region_xyxy"),
                size,
                f"{condition} edit box {task_id}",
            )
            _validate_box(
                task.get("context_region_xyxy"),
                size,
                f"{condition} context box {task_id}",
            )
            expected_candidate, expected_object = expected_whole[condition]
            _require(
                run.get("input_mode") == "full-image-orange-box"
                and run.get("orange_box_xyxy") == edit
                and task.get("candidates") == expected_candidate
                and run.get("candidate") == expected_candidate
                and run.get("object_kind") == expected_object,
                f"{condition}: generation semantics drift: {task_id}",
            )
            eligibility_records[condition].append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "normalized_task_id": normalized,
                    "source": source,
                    "candidate": candidate,
                    "latest_run_row_index": whole_indexes[condition][task_id],
                    "conditioning_box_xyxy": edit,
                    "input_mode": str(run["input_mode"]),
                    "object_kind": str(run["object_kind"]),
                }
            )

    eligibility_hashes = {
        condition: _eligibility_set_hash(eligibility_records[condition], condition)
        for condition in CONDITIONS
    }

    return SourceData(
        paths=paths,
        jsonl=jsonl_values,
        json=json_values,
        eligible=eligible,
        selected=selected,
        eligibility_ranks=eligibility_ranks,
        selection_ranks=selection_ranks,
        eligibility_hashes=eligibility_hashes,
        mouse_real=mouse_real_by_task,
        mouse_forged=mouse_forged_by_task,
        mouse_pairs=mouse_pairs_by_task,
        local_tasks=local_tasks,
        local_materialized=local_materialized,
        whole_tasks=whole_tasks,
        whole_latest=whole_latest,
        whole_latest_indexes=whole_indexes,
        trash_failures=trash_failures,
    )


def _load_release_ledgers(
    manifest: Mapping[str, Any],
    repo_root: Path,
    release_dir: Path,
    spec: ValidationSpec,
) -> tuple[
    dict[str, Path],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    ledgers = manifest.get("ledgers")
    if not isinstance(ledgers, dict) or set(ledgers) != {
        "inputs",
        "panel",
        "source_pairs",
    }:
        _fail("manifest.ledgers must contain exactly inputs/panel/source_pairs")
    expected = {
        "inputs": ("inputs.jsonl", spec.inputs),
        "panel": ("panel.jsonl", spec.panel),
        "source_pairs": ("source_pairs.jsonl", spec.source_pairs),
    }
    paths: dict[str, Path] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, (filename, expected_rows) in expected.items():
        contract = ledgers.get(name)
        if not isinstance(contract, dict):
            _fail(f"manifest.ledgers.{name} must be an object")
        _require(
            {"path", "rows", "sha256"} <= set(contract),
            f"manifest.ledgers.{name} lacks required fields",
        )
        path = _safe_release_file(
            repo_root,
            release_dir,
            contract.get("path"),
            f"manifest.ledgers.{name}.path",
        )
        _require(path.name == filename, f"{name} ledger has wrong filename")
        _verify_file(path, contract.get("sha256"), f"{name} ledger")
        if "bytes" in contract:
            _require(
                path.stat().st_size
                == _expect_int(
                    contract.get("bytes"),
                    f"manifest.ledgers.{name}.bytes",
                    minimum=0,
                ),
                f"{name} ledger byte count mismatch",
            )
        rows = _read_jsonl(path, f"{name} ledger")
        declared_rows = _expect_int(
            contract.get("rows"),
            f"manifest.ledgers.{name}.rows",
            minimum=0,
        )
        _require(
            len(rows) == declared_rows == expected_rows,
            f"{name} ledger row count drift",
        )
        _require(
            _rows_hash(rows) == str(contract.get("sha256")),
            f"{name} ledger is not stable canonical JSONL",
        )
        paths[name] = path
        rows_by_name[name] = rows

    _require(
        manifest.get("inputs_rows") == spec.inputs,
        "manifest.inputs_rows count drift",
    )
    _require(
        manifest.get("panel_rows") == spec.panel,
        "manifest.panel_rows count drift",
    )
    _require(
        manifest.get("source_pair_rows") == spec.source_pairs,
        "manifest.source_pair_rows count drift",
    )
    _require(
        manifest.get("new_canonical_images") == spec.new_canonical_images,
        "manifest.new_canonical_images count drift",
    )
    _require(
        manifest.get("new_local_masks") == spec.new_local_masks,
        "manifest.new_local_masks count drift",
    )
    _require(manifest.get("status") == "complete", "manifest status is not complete")
    return (
        paths,
        rows_by_name["inputs"],
        rows_by_name["panel"],
        rows_by_name["source_pairs"],
    )


def _row_path_matches(
    row: Mapping[str, Any],
    field: str,
    expected: Any,
    label: str,
) -> None:
    _require(row.get(field) == expected, f"{label}.{field} source-ledger mismatch")


def _validate_input_rows(
    rows: list[dict[str, Any]],
    sources: SourceData,
    repo_root: Path,
    release_dir: Path,
    spec: ValidationSpec,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    set[Path],
    set[Path],
]:
    _expect_contiguous_ranks(rows, "rank", "inputs")
    condition_counts = Counter(str(row.get("condition")) for row in rows)
    _require(
        dict(condition_counts) == spec.cache_counts,
        f"input condition counts drift: {dict(condition_counts)}",
    )
    cluster_counts = {
        condition: Counter(
            str(row.get("matched_source_raw_sha256"))
            for row in rows
            if row.get("condition") == condition
        )
        for condition in CONDITIONS
    }

    expected_tasks: dict[str, list[str]] = {
        REAL_CONDITION: [
            _normalized_task_id(str(row["task_id"]))
            for row in sources.eligible[REAL_CONDITION]
        ],
        **{
            condition: [
                _normalized_task_id(str(row["task_id"]))
                for row in sources.selected[condition]
            ]
            for condition in FORGED_CONDITIONS
        },
    }
    expected_physical = [
        (condition, normalized)
        for condition in CONDITIONS
        for normalized in expected_tasks[condition]
    ]
    actual_physical: list[tuple[str, str]] = []
    by_sample: dict[str, dict[str, Any]] = {}
    by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    by_condition_task: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_files: set[Path] = set()
    mask_files: set[Path] = set()

    for index, row in enumerate(rows):
        label = f"inputs[{index}]"
        _expect_schema_and_dataset(row, DATASET_ID, label)
        condition = _expect_str(row.get("condition"), f"{label}.condition")
        _require(condition in CONDITIONS, f"{label}: unknown condition")
        task_id = _expect_str(row.get("task_id"), f"{label}.task_id")
        normalized = _expect_str(
            row.get("normalized_task_id"),
            f"{label}.normalized_task_id",
        )
        _require(
            normalized == _normalized_task_id(task_id),
            f"{label}: normalized_task_id mismatch",
        )
        _require(
            row.get("domain") == _domain(normalized),
            f"{label}: domain mismatch",
        )
        actual_physical.append((condition, normalized))

        expected_selection_key = _selection_key(DATASET_ID, condition, normalized)
        _require(
            row.get("selection_key") == expected_selection_key,
            f"{label}: selection_key mismatch",
        )
        expected_sample = _sample_id(DATASET_ID, condition, normalized)
        _require(
            row.get("sample_id") == expected_sample,
            f"{label}: sample_id mismatch",
        )
        sample_id = expected_sample
        _require(sample_id not in by_sample, f"{label}: duplicate sample_id")
        identity = (condition, normalized)
        _require(
            identity not in by_condition_task,
            f"{label}: duplicate condition/task identity",
        )
        by_sample[sample_id] = row
        by_condition[condition].append(row)
        by_condition_task[identity] = row

        eligibility_rank = _expect_int(
            row.get("eligibility_rank"),
            f"{label}.eligibility_rank",
            minimum=0,
        )
        _require(
            eligibility_rank
            == sources.eligibility_ranks[condition].get(normalized),
            f"{label}: eligibility_rank drift",
        )
        selected_rank = sources.selection_ranks[condition].get(normalized)
        if selected_rank is None:
            _require(
                condition == REAL_CONDITION,
                f"{label}: unselected forged input entered cache",
            )
            _require(
                row.get("selection_rank") is None and row.get("panel") is False,
                f"{label}: non-panel real selection semantics mismatch",
            )
        else:
            _require(
                row.get("selection_rank") == selected_rank,
                f"{label}: selection_rank drift",
            )
            _require(row.get("panel") is True, f"{label}: selected row not in panel")

        is_real = condition == REAL_CONDITION
        expected_family = (
            "real"
            if is_real
            else (
                "local_splice"
                if condition in LOCAL_CONDITIONS
                else "full_frame_conditional_edit"
            )
        )
        _require(
            row.get("condition_family") == expected_family,
            f"{label}: condition_family mismatch",
        )
        _require(
            row.get("kind") == ("real" if is_real else "forged"),
            f"{label}: kind mismatch",
        )
        _require(
            row.get("label") == (0 if is_real else 1),
            f"{label}: label mismatch",
        )
        _require(
            row.get("candidate") == CANDIDATE_BY_CONDITION.get(condition),
            f"{label}: candidate mismatch",
        )
        expected_scope = (
            "authentic"
            if is_real
            else (
                "local_insertion"
                if condition in LOCAL_CONDITIONS
                else "conditional_full_frame_edit"
            )
        )
        _require(
            row.get("manipulation_scope") == expected_scope,
            f"{label}: manipulation_scope mismatch",
        )
        _require(
            row.get("eligible_set_sha256")
            == sources.eligibility_hashes[condition],
            f"{label}: eligible_set_sha256 mismatch",
        )

        raw_path = _safe_repo_file(
            repo_root,
            row.get("raw_path"),
            f"{label}.raw_path",
        )
        raw_sha = _verify_file(raw_path, row.get("raw_sha256"), f"{label} raw")
        _require(
            raw_path.stat().st_size
            == _expect_int(
                row.get("raw_bytes"),
                f"{label}.raw_bytes",
                minimum=1,
            ),
            f"{label}: raw byte count mismatch",
        )
        raw_image = _load_rgb(raw_path)

        canonical_path = _safe_repo_file(
            repo_root,
            row.get("canonical_path"),
            f"{label}.canonical_path",
        )
        canonical_sha = _verify_file(
            canonical_path,
            row.get("canonical_sha256"),
            f"{label} canonical",
        )
        _require(
            canonical_path.stat().st_size
            == _expect_int(
                row.get("canonical_bytes"),
                f"{label}.canonical_bytes",
                minimum=1,
            ),
            f"{label}: canonical byte count mismatch",
        )
        _validate_jpeg(canonical_path, row, label)
        _require(
            raw_image.size
            == (
                _expect_int(row.get("width"), f"{label}.width", minimum=1),
                _expect_int(row.get("height"), f"{label}.height", minimum=1),
            ),
            f"{label}: decoded raw dimensions do not match row",
        )
        regenerated_sha = hashlib.sha256(
            _reencode_canonical(raw_image)
        ).hexdigest()
        _require(
            regenerated_sha == canonical_sha,
            f"{label}: canonical bytes do not reproduce q95/4:4:4 contract",
        )
        expected_origin = "balanced250_v1_reencode"
        _require(
            row.get("canonical_origin") == expected_origin,
            f"{label}: canonical_origin mismatch",
        )
        canonical_path = _safe_release_file(
            repo_root,
            release_dir,
            row.get("canonical_path"),
            f"{label}.canonical_path",
            expected_parent="images",
        )
        _require(
            canonical_path.name == f"{sample_id}.jpg",
            f"{label}: canonical filename is not sample-addressed",
        )
        canonical_files.add(canonical_path)

        matched_path = _safe_repo_file(
            repo_root,
            row.get("matched_source_raw_path"),
            f"{label}.matched_source_raw_path",
        )
        matched_sha = _verify_file(
            matched_path,
            row.get("matched_source_raw_sha256"),
            f"{label} matched source",
        )
        _require(
            row.get("matched_source_task_id") == normalized,
            f"{label}: matched_source_task_id mismatch",
        )
        cluster_size = cluster_counts[condition][matched_sha]
        _require(
            row.get("source_content_cluster") == matched_sha
            and row.get("source_content_cluster_size_within_condition")
            == cluster_size
            and row.get("source_content_is_duplicated_within_condition")
            is (cluster_size > 1),
            f"{label}: source-content cluster fields mismatch",
        )

        source_row: Mapping[str, Any]
        if condition == REAL_CONDITION:
            source_row = sources.mouse_real[task_id]
            for field in ("raw_path", "raw_sha256"):
                _row_path_matches(row, field, source_row.get(field), label)
            _require(
                row.get("source_release_sample_id") == source_row.get("sample_id"),
                f"{label}: source_release_sample_id mismatch",
            )
            _require(
                matched_path == raw_path and matched_sha == raw_sha,
                f"{label}: authentic matched-source identity mismatch",
            )
            _require(
                row.get("gt_mask_kind") == "all_zero"
                and row.get("gt_mask_path") is None
                and row.get("gt_mask_sha256") is None
                and row.get("gt_positive_pixels") == 0,
                f"{label}: authentic GT contract mismatch",
            )
            _require(
                row.get("support_semantics") == "authentic_all_zero",
                f"{label}: authentic support_semantics mismatch",
            )
        elif condition == "local_mouse":
            source_row = sources.mouse_forged[task_id]
            for field in ("raw_path", "raw_sha256"):
                _row_path_matches(row, field, source_row.get(field), label)
            _require(
                row.get("source_release_sample_id")
                == sources.mouse_real[normalized].get("sample_id")
                and row.get("forged_source_release_sample_id")
                == source_row.get("sample_id"),
                f"{label}: Mouse source release IDs mismatch",
            )
        elif condition in {"local_cat", "local_trash_can"}:
            delivered = sources.local_materialized[condition][task_id]
            task = sources.local_tasks[condition][task_id]
            _row_path_matches(row, "raw_path", delivered.get("image"), label)
            _require(
                row.get("matched_source_raw_path") == task.get("source_image"),
                f"{label}: local matched source path mismatch",
            )
            if row.get("local_materialized_candidate_path") is not None:
                _safe_repo_path(
                    repo_root,
                    row.get("local_materialized_candidate_path"),
                    f"{label}.local_materialized_candidate_path",
                )
                _require(
                    row.get("local_materialized_candidate_path")
                    == delivered.get("source_image"),
                    f"{label}: local candidate provenance mismatch",
                )
            contract_name = (
                "cat_materialized"
                if condition == "local_cat"
                else "trash_materialized"
            )
            _require(
                row.get("local_materialized_manifest_path")
                == sources.paths[contract_name].relative_to(repo_root).as_posix(),
                f"{label}: local materialized manifest path mismatch",
            )
        else:
            task = sources.whole_tasks[condition][task_id]
            run = sources.whole_latest[condition][task_id]
            _row_path_matches(row, "raw_path", run.get("output_image"), label)
            _require(
                row.get("matched_source_raw_path") == task.get("source_image"),
                f"{label}: full-frame matched source path mismatch",
            )
            _require(
                row.get("generation_manifest_latest_row_index")
                == sources.whole_latest_indexes[condition][task_id],
                f"{label}: latest generation row index mismatch",
            )
            _require(
                run.get("input_source_image")
                == row.get("matched_source_raw_path")
                and run.get("input_source_sha256") == matched_sha,
                f"{label}: generation input provenance mismatch",
            )
            for input_field, run_field in (
                ("generation_model", "model"),
                ("generation_service_model", "service_model"),
                ("generation_seed", "seed"),
                ("generation_steps", "steps"),
                ("generation_guidance_scale", "guidance_scale"),
                ("generation_bot_task", "bot_task"),
            ):
                _require(
                    row.get(input_field) == run.get(run_field),
                    f"{label}: {input_field} provenance mismatch",
                )
            _require(
                row.get("generation_manifest_path")
                == sources.paths[f"{condition}_run"].relative_to(
                    repo_root
                ).as_posix(),
                f"{label}: generation_manifest_path mismatch",
            )

        if condition in LOCAL_CONDITIONS:
            _require(
                row.get("gt_mask_kind") == "exact_diff",
                f"{label}: local row must use exact_diff",
            )
            mask_path = _safe_repo_file(
                repo_root,
                row.get("gt_mask_path"),
                f"{label}.gt_mask_path",
            )
            _verify_file(mask_path, row.get("gt_mask_sha256"), f"{label} GT mask")
            mask_path = _safe_release_file(
                repo_root,
                release_dir,
                row.get("gt_mask_path"),
                f"{label}.gt_mask_path",
                expected_parent="masks",
            )
            _require(
                mask_path.name == f"{sample_id}.png",
                f"{label}: mask filename is not sample-addressed",
            )
            mask_files.add(mask_path)
            try:
                with Image.open(mask_path) as opened:
                    opened.load()
                    _require(opened.format == "PNG", f"{label}: mask is not PNG")
                    _require(opened.mode == "L", f"{label}: mask is not mode L")
                    _require(
                        opened.size == raw_image.size,
                        f"{label}: mask dimensions mismatch",
                    )
                    histogram = opened.histogram()
                    _require(
                        sum(histogram[1:255]) == 0,
                        f"{label}: mask is not binary 0/255",
                    )
                    actual_mask = opened.copy()
            except ValidationError:
                raise
            except (OSError, ValueError) as exc:
                raise ValidationError(f"{label}: invalid mask: {exc}") from exc
            source_image = _load_rgb(matched_path)
            expected_mask = _exact_diff_mask(source_image, raw_image)
            _require(
                ImageChops.difference(actual_mask, expected_mask).getbbox() is None,
                f"{label}: GT mask differs from independently recomputed diff",
            )
            positive = _mask_positive_pixels(expected_mask)
            _require(positive > 0, f"{label}: local exact diff is empty")
            _require(
                row.get("gt_positive_pixels") == positive,
                f"{label}: gt_positive_pixels mismatch",
            )
            bbox = list(expected_mask.getbbox()) if expected_mask.getbbox() else None
            _require(
                row.get("gt_bbox_xyxy") == bbox,
                f"{label}: gt_bbox_xyxy mismatch",
            )
            context_value = row.get("context_region_xyxy")
            context = (
                _validate_box(context_value, raw_image.size, f"{label}.context")
                if context_value is not None
                else None
            )
            outside = _count_outside(expected_mask, context)
            _require(
                row.get("gt_pixels_outside_context") == outside,
                f"{label}: gt_pixels_outside_context mismatch",
            )
            if row.get("gt_fraction") is not None:
                fraction = positive / (raw_image.width * raw_image.height)
                _require(
                    abs(
                        _expect_number(row.get("gt_fraction"), f"{label}.gt_fraction")
                        - fraction
                    )
                    <= 1e-15,
                    f"{label}: gt_fraction mismatch",
                )
            _require(
                row.get("support_semantics")
                == "decoded_source_vs_local_forged_exact_diff",
                f"{label}: local support_semantics mismatch",
            )
            if row.get("edit_region_xyxy") is not None:
                _validate_box(
                    row.get("edit_region_xyxy"),
                    raw_image.size,
                    f"{label}.edit_region_xyxy",
                )
        elif condition in FULLFRAME_CONDITIONS:
            _require(
                row.get("gt_mask_kind") == "not_applicable"
                and row.get("gt_mask_path") is None
                and row.get("gt_mask_sha256") is None
                and row.get("gt_positive_pixels") is None,
                f"{label}: full-frame GT must be not_applicable",
            )
            _require(
                row.get("support_semantics")
                == "full_frame_conditional_edit_no_localization_target",
                f"{label}: full-frame support_semantics mismatch",
            )
            _validate_box(
                row.get("conditioning_box_xyxy"),
                raw_image.size,
                f"{label}.conditioning_box_xyxy",
            )
            _validate_box(
                row.get("context_region_xyxy"),
                raw_image.size,
                f"{label}.context_region_xyxy",
            )
            if condition == "fullframe_trash_can":
                failure = sources.trash_failures.get(task_id)
                expected_status = "failed" if failure is not None else "usable"
                _require(
                    row.get("fullframe_semantic_qc_status") == expected_status,
                    f"{label}: Trash semantic QC status mismatch",
                )
                expected_categories = (
                    [str(value) for value in failure.get("categories", [])]
                    if failure is not None
                    else []
                )
                expected_reason = (
                    str(failure.get("reason"))
                    if failure is not None and failure.get("reason") is not None
                    else None
                )
                _require(
                    row.get("fullframe_semantic_qc_categories")
                    == expected_categories
                    and row.get("fullframe_semantic_qc_reason")
                    == expected_reason,
                    f"{label}: Trash semantic QC detail mismatch",
                )
            else:
                _require(
                    row.get("fullframe_semantic_qc_status") == "not_reviewed"
                    and row.get("fullframe_semantic_qc_categories") == []
                    and row.get("fullframe_semantic_qc_reason") is None,
                    f"{label}: unexpected semantic QC for unreviewed condition",
                )

    _require(
        actual_physical == expected_physical,
        "input physical condition/selection order drift",
    )
    _require(len(by_sample) == spec.inputs, "input sample identity count drift")

    for row in rows:
        condition = str(row["condition"])
        normalized = str(row["normalized_task_id"])
        real = by_condition_task.get((REAL_CONDITION, normalized))
        _require(real is not None, f"{condition}/{normalized}: missing cached real")
        _require(
            row.get("matched_source_raw_path") == real.get("raw_path")
            and row.get("matched_source_raw_sha256") == real.get("raw_sha256"),
            f"{condition}/{normalized}: matched cached real mismatch",
        )

    return by_sample, by_condition, canonical_files, mask_files


def _validate_panel_rows(
    rows: list[dict[str, Any]],
    inputs_by_sample: Mapping[str, Mapping[str, Any]],
    inputs_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: ValidationSpec,
) -> None:
    _expect_contiguous_ranks(rows, "panel_rank", "panel")
    counts = Counter(str(row.get("condition")) for row in rows)
    _require(dict(counts) == spec.panel_counts, "panel condition counts drift")
    seen: set[str] = set()
    expected_physical: list[tuple[str, int]] = []
    actual_physical: list[tuple[str, int]] = []
    for condition in CONDITIONS:
        expected_physical.extend(
            (condition, rank) for rank in range(spec.panel_per_condition)
        )

    for index, row in enumerate(rows):
        label = f"panel[{index}]"
        _expect_schema_and_dataset(row, DATASET_ID, label)
        condition = _expect_str(row.get("condition"), f"{label}.condition")
        _require(condition in CONDITIONS, f"{label}: unknown condition")
        condition_rank = _expect_int(
            row.get("condition_rank"),
            f"{label}.condition_rank",
            minimum=0,
        )
        actual_physical.append((condition, condition_rank))
        sample_id = _expect_str(row.get("sample_id"), f"{label}.sample_id")
        _require(sample_id not in seen, f"{label}: duplicate sample_id")
        seen.add(sample_id)
        source = inputs_by_sample.get(sample_id)
        _require(source is not None, f"{label}: dangling sample_id")
        assert source is not None
        _require(source.get("panel") is True, f"{label}: references non-panel input")
        _require(
            source.get("condition") == condition
            and source.get("selection_rank") == condition_rank,
            f"{label}: input condition/rank reference mismatch",
        )
        for field in (
            "task_id",
            "normalized_task_id",
            "label",
            "domain",
            "kind",
            "condition_family",
            "manipulation_scope",
            "selection_key",
            "eligible_set_sha256",
            "canonical_path",
            "canonical_sha256",
            "canonical_bytes",
            "width",
            "height",
            "source_content_cluster",
            "source_content_cluster_size_within_condition",
            "gt_mask_kind",
            "gt_mask_path",
            "gt_mask_sha256",
            "gt_positive_pixels",
        ):
            _require(
                row.get(field) == source.get(field),
                f"{label}: {field} does not match input",
            )
        _require(
            row.get("input_rank") == source.get("rank"),
            f"{label}: input_rank does not match input",
        )

    _require(
        actual_physical == expected_physical,
        "panel physical condition/rank order drift",
    )
    expected_samples = {
        str(row["sample_id"])
        for condition_rows in inputs_by_condition.values()
        for row in condition_rows
        if row.get("panel") is True
    }
    _require(seen == expected_samples, "panel/input membership closure mismatch")


def _validate_source_pair_rows(
    rows: list[dict[str, Any]],
    inputs_by_sample: Mapping[str, Mapping[str, Any]],
    spec: ValidationSpec,
) -> None:
    _expect_contiguous_ranks(rows, "rank", "source_pairs")
    _expect_contiguous_ranks(rows, "pair_rank", "source_pairs")
    counts = Counter(str(row.get("condition")) for row in rows)
    _require(
        dict(counts) == spec.source_pair_counts,
        "source-pair condition counts drift",
    )
    expected_order = [
        (condition, rank)
        for condition in FORGED_CONDITIONS
        for rank in range(spec.forged_cache_per_condition)
    ]
    actual_order: list[tuple[str, int]] = []
    seen_pair_ids: set[str] = set()
    seen_forged: set[str] = set()

    for index, row in enumerate(rows):
        label = f"source_pairs[{index}]"
        _expect_schema_and_dataset(row, DATASET_ID, label)
        condition = _expect_str(row.get("condition"), f"{label}.condition")
        _require(condition in FORGED_CONDITIONS, f"{label}: invalid condition")
        condition_pair_rank = _expect_int(
            row.get("condition_pair_rank"),
            f"{label}.condition_pair_rank",
            minimum=0,
        )
        actual_order.append((condition, condition_pair_rank))
        normalized = _expect_str(
            row.get("normalized_task_id"),
            f"{label}.normalized_task_id",
        )
        expected_pair_id = _pair_id(DATASET_ID, condition, normalized)
        _require(
            row.get("pair_id") == expected_pair_id,
            f"{label}: pair_id mismatch",
        )
        _require(expected_pair_id not in seen_pair_ids, f"{label}: duplicate pair_id")
        seen_pair_ids.add(expected_pair_id)

        real_id = _expect_str(
            row.get("real_sample_id"),
            f"{label}.real_sample_id",
        )
        forged_id = _expect_str(
            row.get("forged_sample_id"),
            f"{label}.forged_sample_id",
        )
        real = inputs_by_sample.get(real_id)
        forged = inputs_by_sample.get(forged_id)
        _require(real is not None and forged is not None, f"{label}: dangling sample ref")
        assert real is not None and forged is not None
        _require(
            real.get("condition") == REAL_CONDITION
            and forged.get("condition") == condition,
            f"{label}: wrong input condition refs",
        )
        _require(
            real.get("normalized_task_id")
            == forged.get("normalized_task_id")
            == normalized,
            f"{label}: normalized task linkage mismatch",
        )
        _require(
            forged.get("selection_rank") == condition_pair_rank,
            f"{label}: condition_pair_rank does not match selection rank",
        )
        _require(forged_id not in seen_forged, f"{label}: forged input paired twice")
        seen_forged.add(forged_id)
        _require(
            row.get("domain") == real.get("domain") == forged.get("domain"),
            f"{label}: source-pair domain mismatch",
        )
        _require(
            row.get("source_raw_path") == real.get("raw_path")
            and row.get("source_raw_sha256") == real.get("raw_sha256")
            and row.get("source_content_cluster") == real.get("raw_sha256"),
            f"{label}: source-content identity mismatch",
        )
        _require(
            forged.get("matched_source_raw_path") == real.get("raw_path")
            and forged.get("matched_source_raw_sha256") == real.get("raw_sha256"),
            f"{label}: forged matched-source closure mismatch",
        )
        _require(
            row.get("comparison_design") == "source_matched_secondary",
            f"{label}: comparison_design mismatch",
        )
        _require(
            row.get("selection_key") == forged.get("selection_key")
            and row.get("eligible_set_sha256")
            == forged.get("eligible_set_sha256"),
            f"{label}: selection audit fields mismatch",
        )
        _require(
            row.get("source_content_cluster_size_within_condition")
            == forged.get("source_content_cluster_size_within_condition"),
            f"{label}: source cluster size mismatch",
        )
        expected_real_ref = {
            field: real.get(field)
            for field in (
                "canonical_path",
                "canonical_sha256",
                "canonical_bytes",
                "width",
                "height",
            )
        }
        expected_forged_ref = {
            field: forged.get(field)
            for field in (
                "canonical_path",
                "canonical_sha256",
                "canonical_bytes",
                "width",
                "height",
            )
        }
        _require(
            row.get("real") == expected_real_ref
            and row.get("forged") == expected_forged_ref,
            f"{label}: nested canonical refs mismatch",
        )

    _require(
        actual_order == expected_order,
        "source-pair physical condition/pair-rank order drift",
    )
    expected_forged = {
        sample_id
        for sample_id, row in inputs_by_sample.items()
        if row.get("condition") in FORGED_CONDITIONS
    }
    _require(
        seen_forged == expected_forged,
        "source-pair/forged-input membership closure mismatch",
    )


def _require_expected_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            _fail(f"{label}: missing {key}")
        if actual[key] != expected_value:
            _fail(f"{label}.{key}: recomputed value mismatch")


def _validate_condition_summaries(
    manifest: Mapping[str, Any],
    sources: SourceData,
    inputs_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: ValidationSpec,
) -> dict[str, dict[str, Any]]:
    summaries = manifest.get("conditions")
    if not isinstance(summaries, dict) or set(summaries) != set(CONDITIONS):
        _fail("manifest.conditions has missing or extra conditions")
    recomputed: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        eligible = sources.eligible[condition]
        selected = sources.selected[condition]
        input_rows = list(inputs_by_condition[condition])
        selected_ids = [
            _normalized_task_id(str(row["task_id"])) for row in selected
        ]
        expected: dict[str, Any] = {
            "eligible_rows": spec.eligible_counts[condition],
            "expected_eligible_rows": spec.eligible_counts[condition],
            "eligible_set_sha256": sources.eligibility_hashes[condition],
            "cache_rows": spec.cache_counts[condition],
            "panel_rows": spec.panel_counts[condition],
            "eligible_normalized_task_ids_sha256": _id_list_hash(
                sorted(
                    _normalized_task_id(str(row["task_id"]))
                    for row in eligible
                )
            ),
            "selected_normalized_task_ids_sha256": _id_list_hash(selected_ids),
            "selection_key_sha256": _id_list_hash(
                _selection_key(DATASET_ID, condition, task_id)
                for task_id in selected_ids
            ),
            "domains": dict(
                sorted(Counter(str(row["domain"]) for row in input_rows).items())
            ),
            "source_content": _content_clusters(input_rows),
        }
        if condition in LOCAL_CONDITIONS:
            expected.update(
                {
                    "gt_positive_pixels": sum(
                        int(row["gt_positive_pixels"]) for row in input_rows
                    ),
                    "gt_pixels_outside_context": sum(
                        int(row.get("gt_pixels_outside_context") or 0)
                        for row in input_rows
                    ),
                    "rows_with_gt_outside_context": sum(
                        int(row.get("gt_pixels_outside_context") or 0) > 0
                        for row in input_rows
                    ),
                }
            )
        if condition == "fullframe_trash_can":
            expected["semantic_qc"] = dict(
                sorted(
                    Counter(
                        str(row["fullframe_semantic_qc_status"])
                        for row in input_rows
                    ).items()
                )
            )
        actual = summaries.get(condition)
        if not isinstance(actual, dict):
            _fail(f"manifest.conditions.{condition} must be an object")
        _require_expected_fields(
            actual,
            expected,
            f"manifest.conditions.{condition}",
        )
        recomputed[condition] = expected
    return recomputed


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    repo_root: Path,
    release_dir: Path,
    sources: SourceData,
    spec: ValidationSpec,
) -> None:
    _expect_schema_and_dataset(manifest, DATASET_ID, "manifest")
    design = manifest.get("design")
    if not isinstance(design, dict):
        _fail("manifest.design must be an object")
    _require_expected_fields(
        design,
        {
            "primary": "independent_seven_condition_panel",
            "secondary": "source_matched_six_condition_pairs",
            "panel_conditions": list(CONDITIONS),
            "panel_rows_per_condition": spec.panel_per_condition,
            "real_cache_rows": spec.real_cache,
            "forged_cache_rows_per_condition": spec.forged_cache_per_condition,
            "self_contained_canonical_inputs": True,
            "release_canonical_images": spec.inputs,
            "release_local_masks": spec.local_masks,
        },
        "manifest.design",
    )
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        _fail("manifest.selection must be an object")
    _require_expected_fields(
        selection,
        {
            "score_blind": True,
            "key": (
                "sha256(dataset_id + NUL + condition + NUL + "
                "normalized_task_id)"
            ),
            "collision_policy": "reject duplicate selection keys",
            "semantic_qc_used_for_selection": False,
        },
        "manifest.selection",
    )
    _require(
        isinstance(selection.get("real_policy"), str)
        and "raw_sha256" in str(selection.get("real_policy")),
        "manifest.selection.real_policy does not freeze content deduplication",
    )
    _require(
        selection.get("forged_policy")
        == "first 250 eligible unique normalized task IDs",
        "manifest.selection.forged_policy drift",
    )

    canonicalization = manifest.get("canonicalization")
    if not isinstance(canonicalization, dict):
        _fail("manifest.canonicalization must be an object")
    _require_expected_fields(
        canonicalization,
        {
            "decode": "Pillow ImageOps.exif_transpose then RGB",
            "format": "JPEG",
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "resize": False,
            "all_inputs_reencoded_from_frozen_raw": True,
        },
        "manifest.canonicalization",
    )
    encoder = canonicalization.get("encoder")
    if not isinstance(encoder, dict):
        _fail("manifest.canonicalization.encoder must be an object")
    _require_expected_fields(
        encoder,
        {
            "pillow": PIL.__version__,
            "libjpeg": features.version_codec("jpg"),
        },
        "manifest.canonicalization.encoder",
    )
    localization = manifest.get("localization")
    if not isinstance(localization, dict):
        _fail("manifest.localization must be an object")
    _require_expected_fields(
        localization,
        {
            "local_conditions": sorted(LOCAL_CONDITIONS),
            "mask_space": "decoded_pre_canonicalization_rgb",
            "mask_rule": "max_abs_rgb_difference_gt_0",
            "context_box_is_not_ground_truth": True,
            "fullframe_gt_mask_kind": "not_applicable",
        },
        "manifest.localization",
    )
    fullframe = manifest.get("fullframe_semantics")
    if not isinstance(fullframe, dict):
        _fail("manifest.fullframe_semantics must be an object")
    _require_expected_fields(
        fullframe,
        {
            "label": "conditional_full_frame_edit",
            "fully_synthetic": False,
            "trash_primary_qc_summary": sources.json["trash_whole_qc"]["summary"],
        },
        "manifest.fullframe_semantics",
    )

    expected_output = release_dir.relative_to(repo_root).as_posix()
    _require(
        manifest.get("repo_root") == str(repo_root),
        "manifest.repo_root mismatch",
    )
    _require(
        manifest.get("output_dir") == expected_output,
        "manifest.output_dir mismatch",
    )
    created_at = _expect_str(manifest.get("created_at"), "manifest.created_at")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValidationError("manifest.created_at is not ISO-8601") from exc
    _require(timestamp.tzinfo is not None, "manifest.created_at lacks timezone")

    nondeterministic = {
        "contract_sha256",
        "created_at",
        "repo_root",
        "output_dir",
        "inputs_rows",
        "panel_rows",
        "source_pair_rows",
        "new_canonical_images",
        "new_local_masks",
        "status",
    }
    deterministic = {
        key: value for key, value in manifest.items() if key not in nondeterministic
    }
    expected_contract = hashlib.sha256(
        stable_json(deterministic).encode("utf-8")
    ).hexdigest()
    _require(
        manifest.get("contract_sha256") == expected_contract,
        "manifest.contract_sha256 mismatch",
    )


def validate_release(
    release_dir: Path,
    *,
    repo_root: Path,
    spec: ValidationSpec = FROZEN_SPEC,
) -> dict[str, Any]:
    """Validate one release and return a machine-readable summary."""

    repo_root = repo_root.resolve()
    _require(repo_root.is_dir(), f"missing repo root: {repo_root}")
    release_dir = _resolved_release_dir(repo_root, release_dir)
    manifest_path = release_dir / "manifest.json"
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "release has no regular manifest.json",
    )
    manifest = _read_json(manifest_path, "manifest")
    _expect_schema_and_dataset(manifest, DATASET_ID, "manifest")

    sources = _validate_source_contracts(manifest, repo_root, spec)
    ledger_paths, inputs, panel, source_pairs = _load_release_ledgers(
        manifest,
        repo_root,
        release_dir,
        spec,
    )
    (
        inputs_by_sample,
        inputs_by_condition,
        canonical_files,
        mask_files,
    ) = _validate_input_rows(
        inputs,
        sources,
        repo_root,
        release_dir,
        spec,
    )
    _validate_panel_rows(
        panel,
        inputs_by_sample,
        inputs_by_condition,
        spec,
    )
    _validate_source_pair_rows(source_pairs, inputs_by_sample, spec)
    recomputed_conditions = _validate_condition_summaries(
        manifest,
        sources,
        inputs_by_condition,
        spec,
    )
    _validate_manifest_contract(
        manifest,
        repo_root,
        release_dir,
        sources,
        spec,
    )

    _require(
        len(canonical_files) == spec.new_canonical_images,
        "new canonical image identity count drift",
    )
    _require(
        len(mask_files) == spec.new_local_masks,
        "new local mask identity count drift",
    )
    inventory_files, inventory_directories = _validate_inventory(
        release_dir,
        {
            manifest_path,
            *ledger_paths.values(),
            *canonical_files,
            *mask_files,
        },
    )
    ledger_hashes = {
        name: sha256_file(path) for name, path in sorted(ledger_paths.items())
    }
    return {
        "schema_version": "claimforge_balanced250_validation_v1",
        "status": "valid",
        "validated_at": utc_now(),
        "release_dir": release_dir.relative_to(repo_root).as_posix(),
        "release_schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "contract_sha256": str(manifest["contract_sha256"]),
        "manifest_sha256": sha256_file(manifest_path),
        "ledger_sha256": ledger_hashes,
        "counts": {
            "inputs": len(inputs),
            "panel": len(panel),
            "source_pairs": len(source_pairs),
            "new_canonical_images": len(canonical_files),
            "new_local_masks": len(mask_files),
            "inventory_files": inventory_files,
            "inventory_directories": inventory_directories,
        },
        "conditions": recomputed_conditions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional atomic JSON summary path; validation is otherwise read-only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = validate_release(
        args.release_dir,
        repo_root=repo_root,
    )
    if args.output is not None:
        output = (
            args.output.resolve()
            if args.output.is_absolute()
            else (repo_root / args.output).resolve()
        )
        try:
            output.relative_to(repo_root)
        except ValueError as exc:
            raise ValidationError("output path escapes repository") from exc
        release_dir = _resolved_release_dir(repo_root, args.release_dir)
        try:
            output.relative_to(release_dir)
        except ValueError:
            pass
        else:
            _fail("validation summary may not be written inside the release")
        atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
