"""Strict adapters for ClaimForge canonical evaluation releases.

The two supported schemas have intentionally different sampling designs:

* ``claimforge_balanced250_canonical_v1`` is an independent seven-condition
  panel with a larger 1,775-image score cache and separate source-matched
  secondary pairs.
* ``claimforge_mouse_canonical_v1`` is the legacy 275-pair Mouse release.

This module normalizes only the small set of fields runners need.  It does not
silently infer unknown schemas or reinterpret the legacy ``pair_limit`` on the
independent Balanced250 panel.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import sha256_file, stable_json


BALANCED_SCHEMA = "claimforge_balanced250_canonical_v1"
BALANCED_DATASET_ID = "claimforge-balanced250-independent-panel-jpeg-q95-v1"
BALANCED_CONTRACT_SHA256 = (
    "671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c"
)
MOUSE_SCHEMA = "claimforge_mouse_canonical_v1"
MOUSE_DATASET_ID = "claimforge-mouse-good275-canonical-jpeg-q95-v1"

BALANCED_RELEASE_KIND = "balanced250"
LEGACY_MOUSE_RELEASE_KIND = "legacy_mouse"

BALANCED_CONDITIONS = (
    "real",
    "local_mouse",
    "local_cat",
    "local_trash_can",
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
LOCALIZATION_CONDITIONS = (
    "real",
    "local_mouse",
    "local_cat",
    "local_trash_can",
)
LOCAL_FORGED_CONDITIONS = frozenset(LOCALIZATION_CONDITIONS[1:])
FULLFRAME_CONDITIONS = frozenset(BALANCED_CONDITIONS[4:])

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHORT_ID = re.compile(r"[0-9a-f]{24}\Z")


class Capability(str, Enum):
    """The tasks for which a model's native outputs are valid."""

    WHOLE_IMAGE_T1 = "whole_image_t1"
    LOCAL_T1_T2 = "local_t1_t2"
    LOCAL_T2_ONLY = "local_t2_only"

    @property
    def conditions(self) -> tuple[str, ...]:
        """Conditions that must be inferred for a complete model run."""

        if self is Capability.LOCAL_T2_ONLY:
            return LOCALIZATION_CONDITIONS
        return BALANCED_CONDITIONS

    @property
    def valid_for_t1(self) -> bool:
        return self is not Capability.LOCAL_T2_ONLY

    @property
    def valid_for_t2(self) -> bool:
        return self is not Capability.WHOLE_IMAGE_T1


@dataclass(frozen=True)
class LedgerView:
    """A verified immutable JSONL ledger."""

    name: str
    path: Path
    sha256: str
    rows: int


@dataclass(frozen=True)
class CanonicalRelease:
    """A normalized, verified canonical release."""

    repo_root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    schema_version: str
    dataset_id: str
    release_kind: str
    contract_sha256: str
    inputs_ledger: LedgerView
    inputs: tuple[dict[str, Any], ...]
    panel_ledger: LedgerView | None
    panel: tuple[dict[str, Any], ...]
    source_pairs_ledger: LedgerView | None
    source_pairs: tuple[dict[str, Any], ...]
    legacy_pairs_ledger: LedgerView | None
    legacy_pairs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SelectionSpec:
    """An explicit inference selection.

    ``conditions`` and ``per_condition_limit`` are Balanced250-only.
    ``pair_limit`` is legacy-Mouse-only.  ``sample_id`` is supported by both
    schemas and is mutually exclusive with every other limiting selector.
    """

    capability: Capability
    conditions: tuple[str, ...] | None = None
    per_condition_limit: int | None = None
    sample_id: str | None = None
    pair_limit: int | None = None


class CanonicalReleaseError(ValueError):
    """A canonical release or selection violates its frozen contract."""


def _fail(message: str) -> None:
    raise CanonicalReleaseError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _require_short_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHORT_ID.fullmatch(value) is None:
        _fail(f"{label} must be a 24-character lowercase hexadecimal ID")
    return value


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_without_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalReleaseError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    _fail(f"{label}:{line_number} lacks a terminating newline")
                if not line.strip():
                    _fail(f"{label}:{line_number} is blank")
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_without_duplicate_keys,
                    )
                except json.JSONDecodeError as exc:
                    raise CanonicalReleaseError(
                        f"{label}:{line_number} is invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    _fail(f"{label}:{line_number} must be a JSON object")
                if line != f"{stable_json(value)}\n":
                    _fail(f"{label}:{line_number} is not canonical JSONL")
                rows.append(value)
    except (OSError, UnicodeDecodeError) as exc:
        raise CanonicalReleaseError(f"cannot read {label}: {path}") from exc
    return rows


def _safe_repo_relative(
    repo_root: Path,
    value: Any,
    label: str,
    *,
    require_file: bool,
) -> Path:
    relative = _require_str(value, label)
    if "\\" in relative:
        _fail(f"{label} must use POSIX separators")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"{label} is absolute, non-canonical, or traversing")
    current = repo_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{label} contains a symlink component")
    resolved = current.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise CanonicalReleaseError(f"{label} escapes repository") from exc
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _resolve_manifest_path(repo_root: Path, manifest_path: Path) -> Path:
    if manifest_path.is_absolute():
        try:
            relative = manifest_path.relative_to(repo_root)
        except ValueError as exc:
            raise CanonicalReleaseError(
                "canonical manifest path is outside repository"
            ) from exc
        value = relative.as_posix()
    else:
        value = manifest_path.as_posix()
    return _safe_repo_relative(
        repo_root,
        value,
        "canonical manifest path",
        require_file=True,
    )


def _require_release_child(
    path: Path,
    release_dir: Path,
    label: str,
    *,
    parent: str | None = None,
) -> None:
    try:
        relative = path.relative_to(release_dir)
    except ValueError as exc:
        raise CanonicalReleaseError(
            f"{label} is outside the canonical release directory"
        ) from exc
    if parent is not None and relative.parent != Path(parent):
        _fail(f"{label} must be a direct child of {parent}/")


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ledger(
    *,
    repo_root: Path,
    release_dir: Path,
    name: str,
    path_value: Any,
    sha256_value: Any,
    rows_value: Any,
    expected_filename: str,
    expected_rows: int,
) -> tuple[LedgerView, list[dict[str, Any]]]:
    path = _safe_repo_relative(
        repo_root,
        path_value,
        f"{name} ledger path",
        require_file=True,
    )
    _require_release_child(path, release_dir, f"{name} ledger")
    _require(path.name == expected_filename, f"{name} ledger filename changed")
    expected_sha256 = _require_sha256(sha256_value, f"{name} ledger SHA-256")
    actual_sha256 = sha256_file(path)
    _require(
        actual_sha256 == expected_sha256,
        f"{name} ledger SHA-256 mismatch",
    )
    declared_rows = _require_int(rows_value, f"{name} ledger rows", minimum=0)
    _require(
        declared_rows == expected_rows,
        f"{name} ledger declared row count changed",
    )
    rows = _read_jsonl_strict(path, f"{name} ledger")
    _require(
        len(rows) == declared_rows,
        f"{name} ledger physical row count mismatch",
    )
    _require(
        _rows_sha256(rows) == expected_sha256,
        f"{name} ledger canonical row hash mismatch",
    )
    return (
        LedgerView(
            name=name,
            path=path,
            sha256=expected_sha256,
            rows=declared_rows,
        ),
        rows,
    )


def _require_manifest_time(value: Any, label: str) -> None:
    text = _require_str(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CanonicalReleaseError(f"{label} is not ISO-8601") from exc
    _require(parsed.tzinfo is not None, f"{label} has no timezone")


def _require_dimensions(row: Mapping[str, Any], label: str) -> tuple[int, int]:
    width = _require_int(row.get("width"), f"{label}.width", minimum=1)
    height = _require_int(row.get("height"), f"{label}.height", minimum=1)
    return width, height


def _expected_sample_id(
    dataset_id: str,
    condition: str,
    normalized_task_id: str,
) -> str:
    payload = (
        f"{dataset_id}\0{condition}\0{normalized_task_id}\0sample".encode(
            "utf-8"
        )
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _expected_mouse_sample_id(
    dataset_id: str,
    task_id: str,
    kind: str,
) -> str:
    payload = f"{dataset_id}\0{task_id}\0{kind}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _expected_pair_id(
    dataset_id: str,
    condition: str,
    normalized_task_id: str,
) -> str:
    payload = (
        f"{dataset_id}\0{condition}\0{normalized_task_id}\0source-pair".encode(
            "utf-8"
        )
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _expected_selection_key(
    dataset_id: str,
    condition: str,
    normalized_task_id: str,
) -> str:
    payload = f"{dataset_id}\0{condition}\0{normalized_task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _jpeg_q95_quantization() -> dict[int, list[int]]:
    """Return Pillow/libjpeg's deterministic quality-95 quantization tables."""

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as reference:
        return {
            int(table_id): list(values)
            for table_id, values in reference.quantization.items()
        }


def _validate_canonical_path(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    release_dir: Path,
    label: str,
    verify_file: bool,
) -> None:
    sample_id = _require_short_id(row.get("sample_id"), f"{label}.sample_id")
    digest = _require_sha256(
        row.get("canonical_sha256"),
        f"{label}.canonical_sha256",
    )
    declared_bytes = _require_int(
        row.get("canonical_bytes"),
        f"{label}.canonical_bytes",
        minimum=1,
    )
    width, height = _require_dimensions(row, label)
    path = _safe_repo_relative(
        repo_root,
        row.get("canonical_path"),
        f"{label}.canonical_path",
        require_file=verify_file,
    )
    _require_release_child(
        path,
        release_dir,
        f"{label}.canonical_path",
        parent="images",
    )
    _require(
        path.name == f"{sample_id}.jpg",
        f"{label}.canonical_path filename does not match sample_id",
    )
    if not verify_file:
        return
    _require(path.stat().st_size == declared_bytes, f"{label} byte count changed")
    _require(sha256_file(path) == digest, f"{label} image SHA-256 mismatch")
    try:
        with Image.open(path) as opened:
            opened.load()
            _require(opened.format == "JPEG", f"{label} is not JPEG")
            _require(opened.mode == "RGB", f"{label} JPEG is not RGB")
            _require(
                opened.size == (width, height),
                f"{label} image dimensions changed",
            )
            _require(not opened.getexif(), f"{label} JPEG retained EXIF")
            base_info = {
                "jfif",
                "jfif_version",
                "jfif_unit",
                "jfif_density",
            }
            info_keys = set(opened.info)
            if row.get("schema_version") == BALANCED_SCHEMA:
                _require(
                    info_keys == base_info,
                    f"{label} Balanced250 JPEG retained metadata",
                )
            else:
                _require(
                    info_keys in (base_info, base_info | {"comment"}),
                    f"{label} legacy JPEG retained unsupported metadata",
                )
            _require(
                opened.layer
                == [(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
                f"{label} JPEG is not 4:4:4",
            )
            _require(
                opened.quantization == _jpeg_q95_quantization(),
                f"{label} JPEG quantization is not quality 95",
            )
    except OSError as exc:
        raise CanonicalReleaseError(f"cannot decode {label} image") from exc


def _validate_gt_metadata(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    release_dir: Path,
    label: str,
    verify_file: bool,
) -> None:
    kind = row.get("kind")
    image_label = row.get("label")
    condition = row.get("condition")
    scope = row.get("manipulation_scope")
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "all_zero":
        _require(kind == "real" and image_label == 0, f"{label} invalid all-zero GT")
        _require(condition == "real", f"{label} all-zero GT condition changed")
        _require(scope == "authentic", f"{label} authentic scope changed")
        _require(row.get("gt_mask_path") is None, f"{label} all-zero GT has a path")
        _require(
            row.get("gt_mask_sha256") is None,
            f"{label} all-zero GT has a hash",
        )
        _require(
            row.get("gt_positive_pixels") == 0,
            f"{label} all-zero GT pixel count changed",
        )
        return
    if gt_kind == "not_applicable":
        _require(
            kind == "forged" and image_label == 1,
            f"{label} invalid not-applicable GT",
        )
        _require(
            condition in FULLFRAME_CONDITIONS,
            f"{label} not-applicable GT is not full-frame",
        )
        _require(
            scope == "conditional_full_frame_edit",
            f"{label} full-frame scope changed",
        )
        for key in ("gt_mask_path", "gt_mask_sha256", "gt_positive_pixels"):
            _require(row.get(key) is None, f"{label}.{key} must be null")
        return
    if gt_kind != "exact_diff":
        _fail(f"{label}.gt_mask_kind is unsupported")
    _require(
        kind == "forged" and image_label == 1,
        f"{label} invalid exact-difference GT",
    )
    _require(
        condition in LOCAL_FORGED_CONDITIONS,
        f"{label} exact-difference GT condition changed",
    )
    _require(scope == "local_insertion", f"{label} local scope changed")
    _require_int(
        row.get("gt_positive_pixels"),
        f"{label}.gt_positive_pixels",
        minimum=1,
    )
    mask_path = _safe_repo_relative(
        repo_root,
        row.get("gt_mask_path"),
        f"{label}.gt_mask_path",
        require_file=verify_file,
    )
    _require_release_child(
        mask_path,
        release_dir,
        f"{label}.gt_mask_path",
        parent="masks",
    )
    _require_sha256(row.get("gt_mask_sha256"), f"{label}.gt_mask_sha256")
    if verify_file:
        _load_exact_mask(row, repo_root)


def _load_exact_mask(row: Mapping[str, Any], repo_root: Path) -> np.ndarray:
    label = f"ground truth {row.get('sample_id')}"
    path = _safe_repo_relative(
        repo_root,
        row.get("gt_mask_path"),
        f"{label} path",
        require_file=True,
    )
    digest = _require_sha256(row.get("gt_mask_sha256"), f"{label} SHA-256")
    _require(sha256_file(path) == digest, f"{label} SHA-256 mismatch")
    width, height = _require_dimensions(row, label)
    try:
        with Image.open(path) as opened:
            _require(opened.format == "PNG", f"{label} is not PNG")
            _require(opened.mode == "L", f"{label} is not an L-mode mask")
            pixels = np.asarray(opened, dtype=np.uint8)
    except OSError as exc:
        raise CanonicalReleaseError(f"cannot decode {label}") from exc
    _require(pixels.shape == (height, width), f"{label} dimensions changed")
    _require(
        bool(np.isin(pixels, (0, 255)).all()),
        f"{label} is not binary 0/255",
    )
    positive = int(np.count_nonzero(pixels == 255))
    _require(
        positive == row.get("gt_positive_pixels") and positive > 0,
        f"{label} positive-pixel count changed",
    )
    if "gt_fraction" in row:
        fraction = row.get("gt_fraction")
        _require(
            isinstance(fraction, (int, float))
            and not isinstance(fraction, bool)
            and math.isclose(
                float(fraction),
                positive / float(width * height),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{label} fraction changed",
        )
    if "gt_bbox_xyxy" in row:
        ys, xs = np.nonzero(pixels == 255)
        expected_bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
        _require(
            row.get("gt_bbox_xyxy") == expected_bbox,
            f"{label} bounding box changed",
        )
    return pixels == 255


def load_ground_truth(
    row: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    """Load a native-resolution boolean GT mask.

    ``all_zero`` returns an allocated all-false mask, ``exact_diff`` returns
    the verified binary mask, and ``not_applicable`` returns ``None``.
    """

    root = repo_root.resolve()
    _require(root.is_dir(), f"repository root is not a directory: {root}")
    width, height = _require_dimensions(row, "ground-truth row")
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "all_zero":
        _require(
            row.get("kind") == "real"
            and row.get("label") == 0
            and row.get("condition") == "real"
            and row.get("manipulation_scope") == "authentic"
            and row.get("gt_mask_path") is None
            and row.get("gt_mask_sha256") is None
            and row.get("gt_positive_pixels") == 0,
            "invalid all-zero ground-truth contract",
        )
        return np.zeros((height, width), dtype=bool)
    if gt_kind == "not_applicable":
        _require(
            row.get("kind") == "forged"
            and row.get("label") == 1
            and row.get("condition") in FULLFRAME_CONDITIONS
            and row.get("manipulation_scope")
            == "conditional_full_frame_edit"
            and row.get("gt_mask_path") is None
            and row.get("gt_mask_sha256") is None
            and row.get("gt_positive_pixels") is None,
            "invalid not-applicable ground-truth contract",
        )
        return None
    if gt_kind != "exact_diff":
        _fail("unsupported ground-truth kind")
    _require(
        row.get("kind") == "forged"
        and row.get("label") == 1
        and row.get("condition") in LOCAL_FORGED_CONDITIONS
        and row.get("manipulation_scope") == "local_insertion",
        "invalid exact-difference ground-truth contract",
    )
    return _load_exact_mask(row, root)


def _validate_common_input(
    row: Mapping[str, Any],
    *,
    index: int,
    schema_version: str,
    dataset_id: str,
    repo_root: Path,
    release_dir: Path,
    verify_files: bool,
) -> None:
    label = f"inputs[{index}]"
    _require(row.get("schema_version") == schema_version, f"{label} schema changed")
    _require(row.get("dataset_id") == dataset_id, f"{label} dataset changed")
    _require(row.get("rank") == index, f"{label} rank is not contiguous")
    sample_id = _require_short_id(row.get("sample_id"), f"{label}.sample_id")
    task_id = _require_str(row.get("task_id"), f"{label}.task_id")
    domain = _require_str(row.get("domain"), f"{label}.domain")
    kind = row.get("kind")
    _require(kind in {"real", "forged"}, f"{label}.kind is invalid")
    _require(
        row.get("label") == (0 if kind == "real" else 1),
        f"{label} label/kind mismatch",
    )
    _validate_canonical_path(
        row,
        repo_root=repo_root,
        release_dir=release_dir,
        label=label,
        verify_file=verify_files,
    )
    _require_sha256(row.get("raw_sha256"), f"{label}.raw_sha256")
    _safe_repo_relative(
        repo_root,
        row.get("raw_path"),
        f"{label}.raw_path",
        require_file=False,
    )
    _validate_gt_metadata(
        row,
        repo_root=repo_root,
        release_dir=release_dir,
        label=label,
        verify_file=verify_files,
    )
    _require_short_id(sample_id, f"{label}.sample_id")


def _validate_balanced_inputs(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    release_dir: Path,
    manifest: Mapping[str, Any],
    verify_files: bool,
) -> dict[str, dict[str, Any]]:
    _require(len(rows) == 1775, "Balanced250 inputs must contain 1775 rows")
    by_id: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    selection_ranks: dict[str, set[int]] = defaultdict(set)
    panel_selection_ranks: dict[str, set[int]] = defaultdict(set)
    source_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    condition_summaries = manifest.get("conditions")
    _require(
        isinstance(condition_summaries, dict)
        and set(condition_summaries) == set(BALANCED_CONDITIONS),
        "Balanced250 condition summaries changed",
    )
    for index, row in enumerate(rows):
        _validate_common_input(
            row,
            index=index,
            schema_version=BALANCED_SCHEMA,
            dataset_id=BALANCED_DATASET_ID,
            repo_root=repo_root,
            release_dir=release_dir,
            verify_files=verify_files,
        )
        label = f"inputs[{index}]"
        condition = row.get("condition")
        _require(condition in BALANCED_CONDITIONS, f"{label} condition is invalid")
        normalized = _require_str(
            row.get("normalized_task_id"),
            f"{label}.normalized_task_id",
        )
        _require(
            normalized.startswith(f"{row['domain']}_"),
            f"{label} normalized task/domain mismatch",
        )
        _require(
            row.get("sample_id")
            == _expected_sample_id(BALANCED_DATASET_ID, str(condition), normalized),
            f"{label} sample ID derivation changed",
        )
        _require(
            row.get("selection_key")
            == _expected_selection_key(
                BALANCED_DATASET_ID,
                str(condition),
                normalized,
            ),
            f"{label} selection key changed",
        )
        summary = condition_summaries[str(condition)]
        _require(isinstance(summary, dict), f"{condition} summary is invalid")
        eligible_hash = _require_sha256(
            summary.get("eligible_set_sha256"),
            f"conditions.{condition}.eligible_set_sha256",
        )
        _require(
            row.get("eligible_set_sha256") == eligible_hash,
            f"{label} eligible-set hash changed",
        )
        eligibility_rank = _require_int(
            row.get("eligibility_rank"),
            f"{label}.eligibility_rank",
            minimum=0,
        )
        if condition == "real":
            _require(
                eligibility_rank == counts[str(condition)],
                f"{label} real eligibility rank changed",
            )
        else:
            _require(
                eligibility_rank < int(summary.get("eligible_rows", -1)),
                f"{label} eligibility rank is outside eligible set",
            )
        selection_rank = row.get("selection_rank")
        if selection_rank is not None:
            selection_rank = _require_int(
                selection_rank,
                f"{label}.selection_rank",
                minimum=0,
            )
            _require(
                selection_rank not in selection_ranks[str(condition)],
                f"{label} duplicate condition selection rank",
            )
            selection_ranks[str(condition)].add(selection_rank)
        panel = row.get("panel")
        _require(isinstance(panel, bool), f"{label}.panel must be boolean")
        if panel:
            _require(
                selection_rank is not None,
                f"{label} panel row lacks selection rank",
            )
            panel_selection_ranks[str(condition)].add(int(selection_rank))
        else:
            _require(
                condition == "real" and selection_rank is None,
                f"{label} non-panel row is invalid",
            )
        expected_family = (
            "real"
            if condition == "real"
            else "local_splice"
            if condition in LOCAL_FORGED_CONDITIONS
            else "full_frame_conditional_edit"
        )
        _require(
            row.get("condition_family") == expected_family,
            f"{label} condition family changed",
        )
        expected_kind = "real" if condition == "real" else "forged"
        _require(row.get("kind") == expected_kind, f"{label} condition/kind mismatch")
        cluster = _require_sha256(
            row.get("source_content_cluster"),
            f"{label}.source_content_cluster",
        )
        _require(
            cluster == row.get("matched_source_raw_sha256"),
            f"{label} source-content cluster/provenance mismatch",
        )
        _require_int(
            row.get("source_content_cluster_size_within_condition"),
            f"{label}.source_content_cluster_size_within_condition",
            minimum=1,
        )
        _require(
            isinstance(
                row.get("source_content_is_duplicated_within_condition"),
                bool,
            ),
            f"{label}.source_content_is_duplicated_within_condition "
            "must be boolean",
        )
        source_cluster_counts[str(condition)][cluster] += 1
        sample_id = str(row["sample_id"])
        _require(sample_id not in by_id, f"duplicate input sample ID: {sample_id}")
        by_id[sample_id] = row
        counts[str(condition)] += 1

    expected_counts = {"real": 275, **{name: 250 for name in BALANCED_CONDITIONS[1:]}}
    _require(dict(counts) == expected_counts, "Balanced250 condition counts changed")
    expected_order = [
        condition
        for condition in BALANCED_CONDITIONS
        for _ in range(expected_counts[condition])
    ]
    _require(
        [str(row["condition"]) for row in rows] == expected_order,
        "Balanced250 input condition order changed",
    )
    for condition in BALANCED_CONDITIONS:
        _require(
            panel_selection_ranks[condition] == set(range(250)),
            f"{condition} panel selection ranks changed",
        )
        summary = condition_summaries[condition]
        _require(
            summary.get("cache_rows") == expected_counts[condition]
            and summary.get("panel_rows") == 250,
            f"{condition} manifest counts changed",
        )
    for index, row in enumerate(rows):
        label = f"inputs[{index}]"
        condition = str(row["condition"])
        cluster = str(row["source_content_cluster"])
        cluster_size = source_cluster_counts[condition][cluster]
        _require(
            row.get("source_content_cluster_size_within_condition")
            == cluster_size,
            f"{label} source-content cluster size changed",
        )
        _require(
            row.get("source_content_is_duplicated_within_condition")
            is (cluster_size > 1),
            f"{label} source-content duplicate flag changed",
        )
    return by_id


_PANEL_IDENTITY_FIELDS = (
    "schema_version",
    "dataset_id",
    "condition",
    "condition_family",
    "sample_id",
    "task_id",
    "normalized_task_id",
    "domain",
    "kind",
    "label",
    "manipulation_scope",
    "selection_key",
    "eligible_set_sha256",
    "source_content_cluster",
    "source_content_cluster_size_within_condition",
    "canonical_path",
    "canonical_sha256",
    "canonical_bytes",
    "width",
    "height",
    "gt_mask_kind",
    "gt_mask_path",
    "gt_mask_sha256",
    "gt_positive_pixels",
)


def _validate_balanced_panel(
    rows: list[dict[str, Any]],
    inputs_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(len(rows) == 1750, "Balanced250 panel must contain 1750 rows")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        label = f"panel[{index}]"
        _require(
            row.get("schema_version") == BALANCED_SCHEMA,
            f"{label} schema changed",
        )
        _require(
            row.get("dataset_id") == BALANCED_DATASET_ID,
            f"{label} dataset changed",
        )
        _require(row.get("panel_rank") == index, f"{label} rank changed")
        condition = row.get("condition")
        _require(condition in BALANCED_CONDITIONS, f"{label} condition is invalid")
        _require(
            row.get("condition_rank") == counts[str(condition)],
            f"{label} condition rank changed",
        )
        sample_id = _require_short_id(row.get("sample_id"), f"{label}.sample_id")
        _require(sample_id not in seen, f"{label} duplicate sample ID")
        seen.add(sample_id)
        source = inputs_by_id.get(sample_id)
        _require(source is not None, f"{label} has a dangling input reference")
        _require(source.get("panel") is True, f"{label} references a non-panel input")
        _require(
            row.get("input_rank") == source.get("rank"),
            f"{label} input rank changed",
        )
        for key in _PANEL_IDENTITY_FIELDS:
            _require(
                key in row and row.get(key) == source.get(key),
                f"{label}.{key} does not match inputs ledger",
            )
        counts[str(condition)] += 1
    _require(
        dict(counts) == {condition: 250 for condition in BALANCED_CONDITIONS},
        "Balanced250 panel condition counts changed",
    )
    _require(
        seen
        == {
            sample_id
            for sample_id, row in inputs_by_id.items()
            if row.get("panel") is True
        },
        "Balanced250 panel/input membership changed",
    )


_CANONICAL_REFERENCE_FIELDS = (
    "canonical_path",
    "canonical_sha256",
    "canonical_bytes",
    "width",
    "height",
)


def _validate_balanced_source_pairs(
    rows: list[dict[str, Any]],
    inputs_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        len(rows) == 1500,
        "Balanced250 source-pairs ledger must contain 1500 rows",
    )
    counts: Counter[str] = Counter()
    seen_pair_ids: set[str] = set()
    for index, row in enumerate(rows):
        label = f"source_pairs[{index}]"
        _require(
            row.get("schema_version") == BALANCED_SCHEMA,
            f"{label} schema changed",
        )
        _require(
            row.get("dataset_id") == BALANCED_DATASET_ID,
            f"{label} dataset changed",
        )
        _require(
            row.get("rank") == index and row.get("pair_rank") == index,
            f"{label} global rank changed",
        )
        _require(
            row.get("comparison_design") == "source_matched_secondary",
            f"{label} comparison design changed",
        )
        condition = row.get("condition")
        _require(
            condition in BALANCED_CONDITIONS[1:],
            f"{label} condition is invalid",
        )
        _require(
            row.get("condition_pair_rank") == counts[str(condition)],
            f"{label} condition pair rank changed",
        )
        normalized = _require_str(
            row.get("normalized_task_id"),
            f"{label}.normalized_task_id",
        )
        pair_id = _require_short_id(row.get("pair_id"), f"{label}.pair_id")
        _require(
            pair_id
            == _expected_pair_id(BALANCED_DATASET_ID, str(condition), normalized),
            f"{label} pair ID derivation changed",
        )
        _require(pair_id not in seen_pair_ids, f"{label} duplicate pair ID")
        seen_pair_ids.add(pair_id)
        forged_id = _require_short_id(
            row.get("forged_sample_id"),
            f"{label}.forged_sample_id",
        )
        real_id = _require_short_id(
            row.get("real_sample_id"),
            f"{label}.real_sample_id",
        )
        forged = inputs_by_id.get(forged_id)
        real = inputs_by_id.get(real_id)
        _require(forged is not None and real is not None, f"{label} dangling sample")
        _require(
            forged.get("condition") == condition
            and forged.get("kind") == "forged"
            and real.get("condition") == "real"
            and real.get("kind") == "real",
            f"{label} sample roles changed",
        )
        for source in (forged, real):
            _require(
                source.get("normalized_task_id") == normalized,
                f"{label} source task mismatch",
            )
            _require(
                source.get("domain") == row.get("domain"),
                f"{label} source domain mismatch",
            )
        for nested_name, source in (("forged", forged), ("real", real)):
            nested = row.get(nested_name)
            _require(isinstance(nested, dict), f"{label}.{nested_name} is invalid")
            for key in _CANONICAL_REFERENCE_FIELDS:
                _require(
                    nested.get(key) == source.get(key),
                    f"{label}.{nested_name}.{key} changed",
                )
        _require(
            row.get("selection_key") == forged.get("selection_key")
            and row.get("eligible_set_sha256")
            == forged.get("eligible_set_sha256"),
            f"{label} selection provenance changed",
        )
        _require(
            row.get("source_raw_path") == forged.get("matched_source_raw_path")
            and row.get("source_raw_sha256")
            == forged.get("matched_source_raw_sha256"),
            f"{label} source provenance changed",
        )
        cluster = _require_sha256(
            row.get("source_content_cluster"),
            f"{label}.source_content_cluster",
        )
        _require(
            cluster
            == row.get("source_raw_sha256")
            == forged.get("source_content_cluster")
            == real.get("source_content_cluster"),
            f"{label} source-content cluster changed",
        )
        _require(
            row.get("source_content_cluster_size_within_condition")
            == forged.get("source_content_cluster_size_within_condition"),
            f"{label} source-content cluster size changed",
        )
        counts[str(condition)] += 1
    _require(
        dict(counts)
        == {condition: 250 for condition in BALANCED_CONDITIONS[1:]},
        "Balanced250 source-pair condition counts changed",
    )


def _validate_balanced_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    manifest_path: Path,
    verify_files: bool,
) -> CanonicalRelease:
    _require(
        manifest.get("dataset_id") == BALANCED_DATASET_ID,
        "unsupported Balanced250 dataset ID",
    )
    _require(manifest.get("status") == "complete", "Balanced250 release is incomplete")
    _require(manifest.get("repo_root") == str(repo_root), "manifest repo_root changed")
    _require_manifest_time(manifest.get("created_at"), "manifest.created_at")
    release_dir = manifest_path.parent
    expected_output = release_dir.relative_to(repo_root).as_posix()
    _require(
        manifest.get("output_dir") == expected_output,
        "Balanced250 output_dir changed",
    )
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
    contract_sha256 = _require_sha256(
        manifest.get("contract_sha256"),
        "manifest.contract_sha256",
    )
    _require(
        contract_sha256 == expected_contract,
        "Balanced250 manifest contract SHA-256 mismatch",
    )
    _require(
        contract_sha256 == BALANCED_CONTRACT_SHA256,
        "Balanced250 release contract is not the frozen published contract",
    )
    design = manifest.get("design")
    _require(isinstance(design, dict), "Balanced250 manifest.design is invalid")
    expected_design = {
        "primary": "independent_seven_condition_panel",
        "secondary": "source_matched_six_condition_pairs",
        "panel_conditions": list(BALANCED_CONDITIONS),
        "panel_rows_per_condition": 250,
        "real_cache_rows": 275,
        "forged_cache_rows_per_condition": 250,
        "self_contained_canonical_inputs": True,
        "release_canonical_images": 1775,
        "release_local_masks": 750,
    }
    for key, expected in expected_design.items():
        _require(design.get(key) == expected, f"Balanced250 design.{key} changed")
    localization = manifest.get("localization")
    _require(
        isinstance(localization, dict)
        and localization.get("local_conditions") == sorted(LOCAL_FORGED_CONDITIONS)
        and localization.get("mask_space")
        == "decoded_pre_canonicalization_rgb"
        and localization.get("mask_rule") == "max_abs_rgb_difference_gt_0"
        and localization.get("context_box_is_not_ground_truth") is True
        and localization.get("fullframe_gt_mask_kind") == "not_applicable",
        "Balanced250 localization contract changed",
    )
    canonicalization = manifest.get("canonicalization")
    _require(
        isinstance(canonicalization, dict)
        and canonicalization.get("format") == "JPEG"
        and canonicalization.get("quality") == 95
        and canonicalization.get("subsampling") == 0
        and canonicalization.get("optimize") is False
        and canonicalization.get("metadata") == "stripped"
        and canonicalization.get("resize") is False
        and canonicalization.get("all_inputs_reencoded_from_frozen_raw") is True,
        "Balanced250 canonicalization contract changed",
    )
    ledgers = manifest.get("ledgers")
    _require(
        isinstance(ledgers, dict)
        and set(ledgers) == {"inputs", "panel", "source_pairs"},
        "Balanced250 ledgers changed",
    )
    expected_ledgers = {
        "inputs": ("inputs.jsonl", 1775),
        "panel": ("panel.jsonl", 1750),
        "source_pairs": ("source_pairs.jsonl", 1500),
    }
    loaded: dict[str, tuple[LedgerView, list[dict[str, Any]]]] = {}
    for name, (filename, count) in expected_ledgers.items():
        record = ledgers.get(name)
        _require(isinstance(record, dict), f"manifest.ledgers.{name} is invalid")
        loaded[name] = _load_ledger(
            repo_root=repo_root,
            release_dir=release_dir,
            name=name,
            path_value=record.get("path"),
            sha256_value=record.get("sha256"),
            rows_value=record.get("rows"),
            expected_filename=filename,
            expected_rows=count,
        )
        if "bytes" in record:
            _require(
                loaded[name][0].path.stat().st_size
                == _require_int(
                    record.get("bytes"),
                    f"manifest.ledgers.{name}.bytes",
                    minimum=0,
                ),
                f"{name} ledger byte count changed",
            )
    for key, expected in (
        ("inputs_rows", 1775),
        ("panel_rows", 1750),
        ("source_pair_rows", 1500),
        ("new_canonical_images", 1775),
        ("new_local_masks", 750),
    ):
        _require(manifest.get(key) == expected, f"manifest.{key} changed")
    inputs_ledger, inputs = loaded["inputs"]
    panel_ledger, panel = loaded["panel"]
    source_pairs_ledger, source_pairs = loaded["source_pairs"]
    inputs_by_id = _validate_balanced_inputs(
        inputs,
        repo_root=repo_root,
        release_dir=release_dir,
        manifest=manifest,
        verify_files=verify_files,
    )
    _validate_balanced_panel(panel, inputs_by_id)
    _validate_balanced_source_pairs(source_pairs, inputs_by_id)
    return CanonicalRelease(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        schema_version=BALANCED_SCHEMA,
        dataset_id=BALANCED_DATASET_ID,
        release_kind=BALANCED_RELEASE_KIND,
        contract_sha256=contract_sha256,
        inputs_ledger=inputs_ledger,
        inputs=tuple(inputs),
        panel_ledger=panel_ledger,
        panel=tuple(panel),
        source_pairs_ledger=source_pairs_ledger,
        source_pairs=tuple(source_pairs),
        legacy_pairs_ledger=None,
        legacy_pairs=(),
    )


def _mouse_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "dataset_id",
        "source_review_sha256",
        "source_order_manifest_sha256",
        "jpeg",
        "gt_mask",
        "inputs_sha256",
        "pairs_sha256",
    )
    _require(all(key in manifest for key in keys), "Mouse contract fields are missing")
    return {key: manifest[key] for key in keys}


def _validate_mouse_pairs(
    rows: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
) -> None:
    _require(len(rows) == 275, "Mouse pairs ledger must contain 275 rows")
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in inputs:
        by_pair[int(row["pair_rank"])][str(row["kind"])] = row
    _require(set(by_pair) == set(range(275)), "Mouse input pair ranks changed")
    for index, pair in enumerate(rows):
        label = f"pairs[{index}]"
        _require(pair.get("schema_version") == MOUSE_SCHEMA, f"{label} schema changed")
        _require(pair.get("dataset_id") == MOUSE_DATASET_ID, f"{label} dataset changed")
        _require(pair.get("pair_rank") == index, f"{label} rank changed")
        members = by_pair[index]
        _require(set(members) == {"real", "forged"}, f"{label} is incomplete")
        real, forged = members["real"], members["forged"]
        for source in (real, forged):
            _require(
                source.get("task_id") == pair.get("task_id")
                and source.get("domain") == pair.get("domain"),
                f"{label} task/domain changed",
            )
        for nested_name, source in (("real", real), ("forged", forged)):
            nested = pair.get(nested_name)
            _require(isinstance(nested, dict), f"{label}.{nested_name} is invalid")
            for key in (
                "kind",
                "label",
                "sample_id",
                "raw_path",
                "raw_sha256",
                "canonical_path",
                "canonical_sha256",
                "canonical_bytes",
            ):
                _require(
                    nested.get(key) == source.get(key),
                    f"{label}.{nested_name}.{key} changed",
                )
        for key in ("gt_mask_path", "gt_mask_sha256", "gt_positive_pixels"):
            _require(
                pair.get(key) == forged.get(key),
                f"{label}.{key} changed",
            )


def _validate_mouse_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    manifest_path: Path,
    verify_files: bool,
) -> CanonicalRelease:
    _require(
        manifest.get("dataset_id") == MOUSE_DATASET_ID,
        "unsupported Mouse dataset ID",
    )
    _require(manifest.get("repo_root") == str(repo_root), "manifest repo_root changed")
    _require_manifest_time(manifest.get("created_at"), "manifest.created_at")
    for key in (
        "source_review_sha256",
        "source_order_manifest_sha256",
        "inputs_sha256",
        "pairs_sha256",
    ):
        _require_sha256(manifest.get(key), f"manifest.{key}")
    jpeg = manifest.get("jpeg")
    _require(
        isinstance(jpeg, dict)
        and jpeg.get("quality") == 95
        and jpeg.get("subsampling") == 0
        and jpeg.get("optimize") is False
        and jpeg.get("metadata") == "stripped",
        "Mouse JPEG contract changed",
    )
    gt_mask = manifest.get("gt_mask")
    _require(
        isinstance(gt_mask, dict)
        and gt_mask.get("space") == "decoded_pre_canonicalization_rgb"
        and gt_mask.get("rule") == "max_abs_rgb_difference_gt_threshold"
        and gt_mask.get("threshold") == 0,
        "Mouse GT contract changed",
    )
    contract_sha256 = _require_sha256(
        manifest.get("contract_sha256"),
        "manifest.contract_sha256",
    )
    expected_contract = hashlib.sha256(
        stable_json(_mouse_contract(manifest)).encode("utf-8")
    ).hexdigest()
    _require(
        contract_sha256 == expected_contract,
        "Mouse manifest contract SHA-256 mismatch",
    )
    _require(
        manifest.get("pairs") == 275 and manifest.get("images") == 550,
        "Mouse release counts changed",
    )
    release_dir = manifest_path.parent
    inputs_ledger, raw_inputs = _load_ledger(
        repo_root=repo_root,
        release_dir=release_dir,
        name="inputs",
        path_value=manifest.get("inputs_path"),
        sha256_value=manifest.get("inputs_sha256"),
        rows_value=manifest.get("images"),
        expected_filename="inputs.jsonl",
        expected_rows=550,
    )
    legacy_pairs_ledger, legacy_pairs = _load_ledger(
        repo_root=repo_root,
        release_dir=release_dir,
        name="pairs",
        path_value=manifest.get("pairs_path"),
        sha256_value=manifest.get("pairs_sha256"),
        rows_value=manifest.get("pairs"),
        expected_filename="pairs.jsonl",
        expected_rows=275,
    )
    normalized_inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    pair_kinds: dict[int, set[str]] = defaultdict(set)
    for index, source in enumerate(raw_inputs):
        row = dict(source)
        kind = row.get("kind")
        row["condition"] = "real" if kind == "real" else "local_mouse"
        row["condition_family"] = "real" if kind == "real" else "local_splice"
        row["manipulation_scope"] = (
            "authentic" if kind == "real" else "local_insertion"
        )
        _validate_common_input(
            row,
            index=index,
            schema_version=MOUSE_SCHEMA,
            dataset_id=MOUSE_DATASET_ID,
            repo_root=repo_root,
            release_dir=release_dir,
            verify_files=verify_files,
        )
        label = f"inputs[{index}]"
        pair_rank = _require_int(
            row.get("pair_rank"),
            f"{label}.pair_rank",
            minimum=0,
        )
        _require(pair_rank == index // 2, f"{label} pair rank/order changed")
        _require(
            row.get("sample_id")
            == _expected_mouse_sample_id(
                MOUSE_DATASET_ID,
                str(row["task_id"]),
                str(kind),
            ),
            f"{label} sample ID derivation changed",
        )
        _require(
            str(row["task_id"]).startswith(f"{row['domain']}_"),
            f"{label} task/domain identity changed",
        )
        sample_id = str(row["sample_id"])
        _require(sample_id not in seen, f"duplicate Mouse sample ID: {sample_id}")
        seen.add(sample_id)
        _require(kind not in pair_kinds[pair_rank], f"{label} duplicate pair kind")
        pair_kinds[pair_rank].add(str(kind))
        normalized_inputs.append(row)
    _require(
        set(pair_kinds) == set(range(275))
        and all(kinds == {"real", "forged"} for kinds in pair_kinds.values()),
        "Mouse inputs contain incomplete pairs",
    )
    _validate_mouse_pairs(legacy_pairs, raw_inputs)
    return CanonicalRelease(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        schema_version=MOUSE_SCHEMA,
        dataset_id=MOUSE_DATASET_ID,
        release_kind=LEGACY_MOUSE_RELEASE_KIND,
        contract_sha256=contract_sha256,
        inputs_ledger=inputs_ledger,
        inputs=tuple(normalized_inputs),
        panel_ledger=None,
        panel=(),
        source_pairs_ledger=None,
        source_pairs=(),
        legacy_pairs_ledger=legacy_pairs_ledger,
        legacy_pairs=tuple(legacy_pairs),
    )


def load_canonical_release(
    repo_root: Path,
    manifest_path: Path,
    verify_files: bool = True,
) -> CanonicalRelease:
    """Load and strictly verify one supported canonical release.

    Ledger hashes, canonical JSONL, ranks, identities, references, and GT
    metadata are always verified.  ``verify_files=False`` skips only the
    expensive per-JPEG/per-mask byte and decode verification.
    """

    if not isinstance(verify_files, bool):
        _fail("verify_files must be boolean")
    root = repo_root.resolve()
    _require(root.is_dir(), f"repository root is not a directory: {root}")
    resolved_manifest = _resolve_manifest_path(root, manifest_path)
    manifest = _load_json_object(resolved_manifest, "canonical manifest")
    schema = manifest.get("schema_version")
    if schema == BALANCED_SCHEMA:
        return _validate_balanced_manifest(
            manifest,
            repo_root=root,
            manifest_path=resolved_manifest,
            verify_files=verify_files,
        )
    if schema == MOUSE_SCHEMA:
        return _validate_mouse_manifest(
            manifest,
            repo_root=root,
            manifest_path=resolved_manifest,
            verify_files=verify_files,
        )
    _fail(f"unsupported canonical release schema: {schema!r}")


def _validate_selection_spec(spec: SelectionSpec) -> None:
    if not isinstance(spec, SelectionSpec):
        _fail("spec must be a SelectionSpec")
    if not isinstance(spec.capability, Capability):
        _fail("spec.capability must be a Capability")
    if spec.sample_id is not None:
        _require_str(spec.sample_id, "sample_id")
        if (
            spec.conditions is not None
            or spec.per_condition_limit is not None
            or spec.pair_limit is not None
        ):
            _fail("sample_id is mutually exclusive with all other selectors")
    for value, label in (
        (spec.per_condition_limit, "per_condition_limit"),
        (spec.pair_limit, "pair_limit"),
    ):
        if value is not None:
            _require_int(value, label, minimum=1)


def _select_balanced(
    release: CanonicalRelease,
    spec: SelectionSpec,
) -> list[dict[str, Any]]:
    if spec.pair_limit is not None:
        _fail("pair_limit is unsupported for the independent Balanced250 panel")
    if spec.sample_id is not None:
        selected = [
            row for row in release.inputs if row.get("sample_id") == spec.sample_id
        ]
        _require(
            len(selected) == 1,
            f"sample_id must select exactly one row: {spec.sample_id}",
        )
        _require(
            selected[0].get("condition") in spec.capability.conditions,
            "sample_id is outside the requested capability",
        )
        return [selected[0]]
    if spec.conditions is None:
        conditions = spec.capability.conditions
    else:
        if isinstance(spec.conditions, (str, bytes)) or not isinstance(
            spec.conditions,
            Sequence,
        ):
            _fail("conditions must be a sequence of condition names")
        conditions = tuple(spec.conditions)
        _require(bool(conditions), "conditions must not be empty")
        _require(
            all(isinstance(value, str) and value for value in conditions),
            "conditions contains an invalid name",
        )
        _require(
            len(conditions) == len(set(conditions)),
            "conditions contains duplicates",
        )
        unknown = set(conditions) - set(BALANCED_CONDITIONS)
        _require(not unknown, f"unknown Balanced250 conditions: {sorted(unknown)}")
        unsupported = set(conditions) - set(spec.capability.conditions)
        _require(
            not unsupported,
            "conditions are outside the requested capability: "
            f"{sorted(unsupported)}",
        )
        conditions = tuple(
            condition
            for condition in BALANCED_CONDITIONS
            if condition in set(conditions)
        )
    allowed = set(conditions)
    rows = [row for row in release.inputs if row.get("condition") in allowed]
    if spec.per_condition_limit is not None:
        limited: list[dict[str, Any]] = []
        for condition in conditions:
            candidates = [
                row for row in rows if row.get("condition") == condition
            ]
            candidates.sort(
                key=lambda row: (
                    row.get("panel") is not True,
                    row.get("selection_rank")
                    if row.get("selection_rank") is not None
                    else math.inf,
                    int(row["rank"]),
                )
            )
            limited.extend(candidates[: spec.per_condition_limit])
        rows = sorted(limited, key=lambda row: int(row["rank"]))
    _require(bool(rows), "selection is empty")
    return list(rows)


def _select_mouse(
    release: CanonicalRelease,
    spec: SelectionSpec,
) -> list[dict[str, Any]]:
    if spec.conditions is not None or spec.per_condition_limit is not None:
        _fail(
            "conditions/per_condition_limit are unsupported for legacy Mouse; "
            "use pair_limit"
        )
    if spec.sample_id is not None:
        selected = [
            row for row in release.inputs if row.get("sample_id") == spec.sample_id
        ]
        _require(
            len(selected) == 1,
            f"sample_id must select exactly one row: {spec.sample_id}",
        )
        return [selected[0]]
    pair_ranks = list(range(275))
    if spec.pair_limit is not None:
        pair_ranks = pair_ranks[: spec.pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [
        row
        for row in release.inputs
        if int(row["pair_rank"]) in selected_ranks
    ]
    by_pair: dict[int, set[str]] = defaultdict(set)
    for row in selected:
        by_pair[int(row["pair_rank"])].add(str(row["kind"]))
    _require(
        all(kinds == {"real", "forged"} for kinds in by_pair.values()),
        "legacy Mouse selection contains incomplete pairs",
    )
    return list(selected)


def select_inputs(
    release: CanonicalRelease,
    spec: SelectionSpec,
) -> list[dict[str, Any]]:
    """Select inference inputs without conflating the two release designs."""

    if not isinstance(release, CanonicalRelease):
        _fail("release must be a CanonicalRelease")
    _validate_selection_spec(spec)
    if release.release_kind == BALANCED_RELEASE_KIND:
        return _select_balanced(release, spec)
    if release.release_kind == LEGACY_MOUSE_RELEASE_KIND:
        return _select_mouse(release, spec)
    _fail(f"unsupported normalized release kind: {release.release_kind!r}")


__all__ = [
    "BALANCED_CONDITIONS",
    "BALANCED_CONTRACT_SHA256",
    "BALANCED_DATASET_ID",
    "BALANCED_RELEASE_KIND",
    "BALANCED_SCHEMA",
    "CanonicalRelease",
    "CanonicalReleaseError",
    "Capability",
    "LEGACY_MOUSE_RELEASE_KIND",
    "LOCALIZATION_CONDITIONS",
    "LedgerView",
    "MOUSE_DATASET_ID",
    "MOUSE_SCHEMA",
    "SelectionSpec",
    "load_canonical_release",
    "load_ground_truth",
    "select_inputs",
]
