#!/usr/bin/env python3
"""Independently audit an official FSD v1.2 whole-image detection run.

The runner records the expensive, image-derived 960-dimensional FSD vector.
This analyzer treats that vector as an intermediate artifact and independently
replays the complete released detection tail:

    20 learned residual transforms -> released GMM log likelihood
    -> released z score -> CLAIMFORGE ``ai_score = -z``

FSD is an image-level detector.  This module therefore also rejects invented
T2/localization and S_joint outputs.  Crop visibility is reported only as a
diagnostic condition explaining how much of the exact-difference edit mask was
inside FSD's effective center crop; it is not a predicted localization map.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.whole_image_metrics import (
    FIXED_THRESHOLD,
    summarize_whole_image_pair_slice,
    summarize_whole_image_results,
)


DEFAULT_RUN_ID = "fsd_v1_2_0_mouse_canonical_v1_full275_20260724"
DEFAULT_RESULTS_DIR = Path("results/opensource/fsd")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_FSD_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "Forensic-Self-Descriptions-CVPR25-50f2eae"
)

MODEL_NAME = "FSD"
MODEL_SLUG = "fsd_v1_2_0_official"
DESCRIPTOR_DIMENSION = 960
DESCRIPTOR_DTYPE = np.dtype("float64")
EXPECTED_TRANSFORMS = 20
THRESHOLD_OPERATOR = ">"

# The replay is performed in the exact recorded torch/device environment.
# These bounds catch corrupted/tampered values while allowing only the final
# float64 reduction's platform-level last-bit variation.
RAW_SCORE_ABSOLUTE_TOLERANCE = 1e-9
DERIVED_SCORE_ABSOLUTE_TOLERANCE = 1e-12

_T2_KEYS = frozenset(
    {
        "t2",
        "valid_for_t2",
        "localization",
        "localisation",
        "localization_metrics",
        "score_map",
        "score_map_path",
        "mask_path",
        "predicted_mask",
        "pixel_metrics",
        "iou",
        "miou",
        "pixel_f1",
        "dice",
    }
)
_S_JOINT_KEYS = frozenset({"s_joint", "joint_score", "joint_metrics"})
_NA_STRINGS = frozenset(
    {
        "n/a",
        "na",
        "not_applicable",
        "not applicable",
        "unsupported",
        "unsupported_image_level_only",
    }
)
_REPLAY_FIELDS = (
    "score",
    "score_semantics",
    "raw_descriptor_sha256",
    "raw_descriptor_shape",
    "raw_descriptor_dtype",
    "raw_descriptor_semantics",
    "raw_likelihood",
    "released_z_score",
    "ai_score",
    "released_is_fake",
    "released_threshold",
    "released_threshold_operator",
    "classification_decision",
    "classification_threshold",
    "classification_threshold_operator",
    "classification",
    "t1",
    "manual_replay",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
    "preprocess",
)


@dataclass(frozen=True)
class DetectionPair:
    """A complete real/forged task pair."""

    task_id: str
    domain: str
    real: dict[str, Any]
    forged: dict[str, Any]
    forged_input: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """Exact runtime selected for the independent torch replay."""

    torch: ModuleType
    device: Any
    recorded_device: str
    evidence: dict[str, Any]


def _load_runner_pins() -> SimpleNamespace:
    """Load immutable constants without importing the official FSD package."""

    from eval.opensource import run_fsd

    required = (
        "MODEL_NAME",
        "MODEL_SLUG",
        "MODEL_REPO_URL",
        "MODEL_SOURCE_COMMIT",
        "RELEASE_TAG",
        "SOURCE_FILES",
        "WEIGHT_FILES",
        "FSD_DIMENSION",
        "RELEASED_Z_THRESHOLD",
        "AI_SCORE_THRESHOLD",
        "EXPECTED_CONFIG",
        "PAPER_RELEASE_DRIFT",
        "SOURCE_TAG_DRIFT",
        "RELEASE_TAG_COMMIT",
        "PROJECTION_COUNT",
    )
    missing = [name for name in required if not hasattr(run_fsd, name)]
    if missing:
        raise RuntimeError(f"FSD runner does not export audit pins: {missing}")
    return SimpleNamespace(
        **{name: copy.deepcopy(getattr(run_fsd, name)) for name in required}
    )


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON array")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    digest = _require_sha256(expected, f"{label} expected hash")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {digest}")


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint", "created_at", "adapter", "environment"}
    }
    return hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()


def _selection_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


def _select_manifest_inputs(
    input_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(input_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"canonical row {line_number} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"canonical inputs repeat sample_id {sample_id}")
        by_id[sample_id] = row

    selection = manifest.get("selection")
    ordered = (
        selection.get("rows")
        if isinstance(selection, Mapping)
        else manifest.get("ordered_inputs")
    )
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("run manifest ordered_inputs is empty or invalid")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(ordered):
        item = _require_mapping(raw, f"ordered input {index}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ordered input {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"ordered inputs repeat sample_id {sample_id}")
        if sample_id not in by_id:
            raise ValueError(f"ordered inputs select unknown ID {sample_id}")
        seen.add(sample_id)
        selected.append(by_id[sample_id])
    _require_equal(
        ordered,
        _selection_contract(selected),
        "manifest ordered input contract",
    )
    return selected


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {line_number} has no id")
        histories[row_id].append((line_number, row))
        status_counts[str(row.get("status"))] += 1

    duplicate_histories: list[dict[str, Any]] = []
    recovered_ids: list[str] = []
    latest_counts: Counter[str] = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicate_histories.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line for line, _ in entries],
                    "statuses": statuses,
                }
            )
        if statuses[-1] == "ok" and "error" in statuses[:-1]:
            recovered_ids.append(row_id)
    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_rows": len(result_rows) - len(histories),
        "ids_with_multiple_rows": len(duplicate_histories),
        "recovered_error_to_ok": len(recovered_ids),
        "recovered_ids": recovered_ids,
        "historical_status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {line_number} has no valid id")
        latest[row_id] = row
    return latest


def _normalise_source_files(value: Any, label: str) -> dict[str, str]:
    if isinstance(value, Mapping):
        result = {str(path): str(digest) for path, digest in value.items()}
    elif isinstance(value, list):
        result = {}
        for index, raw in enumerate(value):
            item = _require_mapping(raw, f"{label} entry {index}")
            path = item.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{label} entry {index} has no path")
            if path in result:
                raise ValueError(f"{label} repeats path {path}")
            result[path] = _require_sha256(
                item.get("sha256"),
                f"{label} entry {index} SHA-256",
            )
    else:
        raise ValueError(f"{label} is not a mapping/list")
    if not result:
        raise ValueError(f"{label} is empty")
    for path, digest in result.items():
        _require_sha256(digest, f"{label} {path}")
    return result


def _verify_source_tree(
    source_root: Path,
    *,
    expected_commit: str,
    expected_files: Mapping[str, str],
    label: str,
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing {label} source root: {source_root}")
    _require_equal(
        _git_value(source_root, "rev-parse", "HEAD"),
        expected_commit,
        f"{label} checked-out commit",
    )
    tracked_status = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status is None:
        raise ValueError(f"{label} source is not a readable git repository")
    if tracked_status:
        raise ValueError(f"{label} source has tracked modifications")
    for relative, digest in expected_files.items():
        _verify_hash(source_root / relative, digest, f"{label} source {relative}")


def _expected_weight_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or len(value) != 4:
        raise ValueError("runner WEIGHT_FILES must describe exactly four files")
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in value.items():
        name = str(raw_name)
        if isinstance(raw_record, str):
            record = {"sha256": raw_record}
        else:
            record = _require_mapping(
                copy.deepcopy(raw_record),
                f"runner weight {name}",
            )
        _require_sha256(record.get("sha256"), f"runner weight {name}")
        result[name] = record
    required = {"config.json", "fre.pt", "gmm.pt", "fsd_transforms.pt"}
    if set(result) != required:
        raise ValueError(
            f"runner WEIGHT_FILES names mismatch: {sorted(result)}"
        )
    return result


def _normalise_manifest_weights(value: Any) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        iterator = value.items()
    elif isinstance(value, list):
        iterator = []
        for index, raw in enumerate(value):
            item = _require_mapping(raw, f"manifest weight {index}")
            name = item.get("filename", item.get("name"))
            iterator.append((name, item))
    else:
        raise ValueError("manifest weights are not a mapping/list")
    for raw_name, raw_record in iterator:
        name = str(raw_name)
        record = _require_mapping(raw_record, f"manifest weight {name}")
        if name in records:
            raise ValueError(f"manifest weights repeat {name}")
        records[name] = record
    if len(records) != 4:
        raise ValueError("manifest must contain exactly four FSD weight files")
    return records


def _find_weights_mapping(model: Mapping[str, Any]) -> Any:
    weights = model.get("weights")
    if isinstance(weights, Mapping) and "files" in weights:
        return weights["files"]
    for key in ("weight_files", "release_weights"):
        if key in model:
            return model[key]
    raise ValueError("manifest model has no four-file weight contract")


def _weight_bundle_sha256(records: Mapping[str, Mapping[str, Any]]) -> str:
    payload = [
        {
            "filename": name,
            "sha256": str(records[name]["sha256"]),
            "bytes": (
                int(records[name]["bytes"])
                if records[name].get("bytes") is not None
                else None
            ),
        }
        for name in sorted(records)
    ]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _validate_weight_files(
    *,
    model: Mapping[str, Any],
    repo_root: Path,
    pins: SimpleNamespace,
) -> tuple[Path, dict[str, dict[str, Any]], str, dict[str, Any]]:
    expected = _expected_weight_records(pins.WEIGHT_FILES)
    audit = _require_mapping(model.get("weights"), "manifest weights audit")
    recorded = _normalise_manifest_weights(_find_weights_mapping(model))
    if set(recorded) != set(expected):
        raise ValueError("manifest FSD weight filenames mismatch")

    weight_dir_value = audit.get("weights_dir")
    if not isinstance(weight_dir_value, str) or not weight_dir_value:
        raise ValueError("manifest model has no weights_dir")
    weights_dir = _anchored(Path(weight_dir_value), repo_root)

    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        item = recorded[name]
        expected_item = expected[name]
        digest = _require_sha256(item.get("sha256"), f"manifest weight {name}")
        _require_equal(
            digest,
            expected_item["sha256"],
            f"manifest frozen weight {name} SHA-256",
        )
        path_value = item.get("path")
        path = (
            _anchored(Path(path_value), repo_root)
            if isinstance(path_value, str) and path_value
            else (weights_dir / name).resolve()
        )
        _require_equal(path, (weights_dir / name).resolve(), f"weight {name} path")
        _verify_hash(path, digest, f"official FSD weight {name}")
        actual_bytes = path.stat().st_size
        if expected_item.get("bytes") is not None:
            _require_equal(
                actual_bytes,
                int(expected_item["bytes"]),
                f"runner-frozen weight {name} bytes",
            )
        if item.get("bytes") is not None:
            _require_equal(
                actual_bytes,
                int(item["bytes"]),
                f"manifest weight {name} bytes",
            )
        normalized[name] = {
            **item,
            "path": str(path),
            "sha256": digest,
            "bytes": actual_bytes,
        }
        _require_equal(
            item.get("kind"),
            expected_item.get("kind"),
            f"manifest weight {name} kind",
        )
        if name.endswith(".pt"):
            safety = _require_mapping(
                item.get("serialization_safety"),
                f"manifest weight {name} serialization safety",
            )
            _require_equal(
                safety.get("unsafe_globals"),
                [],
                f"manifest weight {name} unsafe globals",
            )
            _require_equal(
                safety.get("required_unsafe_globals"),
                [],
                f"manifest weight {name} required unsafe globals",
            )
            _require_equal(
                safety.get("weights_only"),
                True,
                f"manifest weight {name} weights-only loader",
            )

    bundle = _weight_bundle_sha256(normalized)
    recorded_bundle = audit.get("bundle_sha256")
    _require_equal(
        _require_sha256(recorded_bundle, "manifest weights bundle"),
        bundle,
        "manifest weights bundle SHA-256",
    )

    config_path = weights_dir / "config.json"
    config = _require_mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        "official FSD config",
    )
    expected_config = copy.deepcopy(pins.EXPECTED_CONFIG)
    _require_equal(config, expected_config, "official FSD v1.2 config")
    _require_equal(
        audit.get("config"),
        expected_config,
        "manifest embedded FSD config",
    )
    for actual, expected_value, label in (
        (
            audit.get("provider"),
            "official_github_release",
            "manifest weight provider",
        ),
        (audit.get("release_tag"), pins.RELEASE_TAG, "manifest weight release tag"),
        (
            audit.get("explicit_weights_dir_required"),
            True,
            "manifest explicit weights-dir flag",
        ),
        (
            audit.get("automatic_download_used"),
            False,
            "manifest automatic-download flag",
        ),
    ):
        _require_equal(actual, expected_value, label)
    _require_equal(
        float(config["scoring"]["default_threshold"]),
        float(pins.RELEASED_Z_THRESHOLD),
        "release z threshold",
    )
    _require_equal(
        -float(pins.RELEASED_Z_THRESHOLD),
        float(pins.AI_SCORE_THRESHOLD),
        "AI-score threshold direction",
    )
    return weights_dir, normalized, bundle, config


def _is_explicit_na(value: Any, *, task: str) -> bool:
    if value is None:
        return True
    if value is False and task == "t2":
        return True
    if isinstance(value, str) and value.strip().lower() in _NA_STRINGS:
        return True
    if isinstance(value, Mapping):
        valid = value.get("valid")
        supported = value.get("supported")
        status = value.get("status", value.get("value"))
        if task == "t2" and valid is False:
            return True
        if supported is False:
            return True
        if isinstance(status, str) and status.strip().lower() in _NA_STRINGS:
            return True
    return False


def _reject_localization_contract(value: Any, *, label: str) -> None:
    """Reject fabricated localization/S_joint while permitting explicit N/A."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    for key, child in value.items():
        normalized = str(key).lower()
        if normalized in _T2_KEYS:
            if not _is_explicit_na(child, task="t2"):
                raise ValueError(
                    f"{label}.{key} fabricates localization for image-only FSD"
                )
            continue
        if normalized in _S_JOINT_KEYS:
            if not _is_explicit_na(child, task="s_joint"):
                raise ValueError(f"{label}.{key} fabricates S_joint")
            continue
        # Crop/edit visibility evidence is an input diagnostic, not model
        # localization.  It is intentionally not traversed as a prediction.
        if normalized in {
            "preprocess",
            "edit_visibility_evidence",
            "by_edit_visibility",
            "edit_visible_gt_fraction",
        }:
            continue
        if isinstance(child, Mapping):
            _reject_localization_contract(child, label=f"{label}.{key}")
        elif isinstance(child, list):
            for index, element in enumerate(child):
                if isinstance(element, Mapping):
                    _reject_localization_contract(
                        element,
                        label=f"{label}.{key}[{index}]",
                    )


def _box_area(box: Iterable[float]) -> float:
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError("box must have four coordinates")
    return max(0.0, values[2] - values[0]) * max(
        0.0, values[3] - values[1]
    )


def _intersection_box(
    first: Iterable[float],
    second: Iterable[float],
) -> list[float]:
    a = [float(value) for value in first]
    b = [float(value) for value in second]
    if len(a) != 4 or len(b) != 4:
        raise ValueError("intersection boxes must have four coordinates")
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = max(left, min(a[2], b[2]))
    bottom = max(top, min(a[3], b[3]))
    return [left, top, right, bottom]


def _official_preprocess_geometry(
    *,
    width: int,
    height: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fre = _require_mapping(config.get("fre"), "FSD config fre")
    fsd = _require_mapping(config.get("fsd"), "FSD config fsd")
    kernel = int(fre["kernel_size"])
    border = kernel // 2
    residual_width = int(width) - 2 * border
    residual_height = int(height) - 2 * border
    if residual_width <= 0 or residual_height <= 0:
        raise ValueError("image is smaller than the official FRE trim")
    max_size = int(fsd["max_size"])
    resize_mode = str(fsd["resize_mode"])
    if "resize" not in resize_mode or "crop" not in resize_mode:
        raise ValueError("analyzer supports only released resize_and_crop")
    nominal_scale = max_size / min(residual_height, residual_width)
    new_height = round(residual_height * nominal_scale)
    new_width = round(residual_width * nominal_scale)
    crop_height = min(max_size, new_height)
    crop_width = min(max_size, new_width)
    start_y = (new_height - crop_height) // 2
    start_x = (new_width - crop_width) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    effective = [
        border + start_x * residual_width / new_width,
        border + start_y * residual_height / new_height,
        border + end_x * residual_width / new_width,
        border + end_y * residual_height / new_height,
    ]
    scale_sizes = [
        [
            int(math.floor(crop_width / (2**level))),
            int(math.floor(crop_height / (2**level))),
        ]
        for level in range(int(fsd["num_scales"]))
    ]
    return {
        "decoder": "Pillow.Image.open_then_convert_L",
        "exif_transpose": False,
        "icc_conversion": False,
        "native_size": [int(width), int(height)],
        "grayscale_dtype": "uint8",
        "grayscale_range": [0, 255],
        "fre": {
            "kernel_size": kernel,
            "output_channels": int(fre["out_channels"]),
            "padding": border,
            "border_each_side": border,
            "post_trim_size": [residual_width, residual_height],
        },
        "resize": {
            "enabled": True,
            "mode": "torch.nn.functional.interpolate_bilinear",
            "align_corners": False,
            "antialias": False,
            "scale_factor_nominal": nominal_scale,
            "source_size": [residual_width, residual_height],
            "destination_size": [new_width, new_height],
            "rounding": "python_round",
            "rule": "short_side_to_1024",
        },
        "center_crop": {
            "enabled": True,
            "source_size": [new_width, new_height],
            "start_xy": [start_x, start_y],
            "size": [crop_width, crop_height],
            "end_xy": [end_x, end_y],
        },
        "effective_native_crop_xyxy": effective,
        "pixel_center_mapping": (
            "d=(native_index-border+0.5)*resized_size/"
            "post_trim_size-0.5"
        ),
        "scales": {
            "count": int(fsd["num_scales"]),
            "sizes": scale_sizes,
            "mode": "torch_bilinear",
            "align_corners": False,
            "antialias": False,
        },
        "descriptor": {
            "shape": [DESCRIPTOR_DIMENSION],
            "dtype": "float64",
            "neighborhood": [int(fsd["kernel_size"]), int(fsd["kernel_size"])],
            "solver": "float64_KKT_constrained_least_squares",
            "lambda_regularization": 1e-5,
        },
    }


def _visible_gt_evidence(
    *,
    forged_input: Mapping[str, Any],
    geometry: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    gt_value = forged_input.get("gt_mask_path")
    if not isinstance(gt_value, str) or not gt_value:
        raise ValueError("forged canonical row has no exact-difference GT path")
    gt_path = _anchored(Path(gt_value), repo_root)
    _verify_hash(
        gt_path,
        forged_input.get("gt_mask_sha256"),
        "forged exact-difference GT",
    )
    with Image.open(gt_path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    native = geometry["native_size"]
    if pixels.shape != (int(native[1]), int(native[0])):
        raise ValueError("GT shape does not match canonical image geometry")
    unique = set(int(value) for value in np.unique(pixels))
    if not unique.issubset({0, 255}):
        raise ValueError(f"GT mask is not binary 0/255: {sorted(unique)}")
    positive_y, positive_x = np.nonzero(pixels == 255)
    total = int(positive_x.size)
    if total <= 0:
        raise ValueError("forged exact-difference GT contains no positives")

    fre = geometry["fre"]
    resize = geometry["resize"]
    crop = geometry["center_crop"]
    border = int(fre["border_each_side"])
    residual_width, residual_height = [
        int(value) for value in fre["post_trim_size"]
    ]
    new_width, new_height = [
        int(value) for value in resize["destination_size"]
    ]
    start_x, start_y = [float(value) for value in crop["start_xy"]]
    end_x, end_y = [float(value) for value in crop["end_xy"]]

    # Exact align_corners=False pixel-centre mapping from the native residual
    # support to the resized residual tensor used by the released code.
    destination_x = (
        (positive_x.astype(np.float64) - border + 0.5)
        * new_width
        / residual_width
        - 0.5
    )
    destination_y = (
        (positive_y.astype(np.float64) - border + 0.5)
        * new_height
        / residual_height
        - 0.5
    )
    visible_mask = (
        (destination_x >= start_x)
        & (destination_x < end_x)
        & (destination_y >= start_y)
        & (destination_y < end_y)
    )
    visible = int(np.count_nonzero(visible_mask))
    fraction = visible / total
    if visible == 0:
        category = "none"
    elif visible == total:
        category = "full"
    else:
        category = "partial"
    return {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": str(forged_input["sample_id"]),
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_after_FRE_trim_"
            "with_align_corners_false_formula_into_official_center_crop"
        ),
        "formula": (
            "d=(native_index-7+0.5)*resized_size/(native_size-14)-0.5; "
            "visible iff crop_start <= d < crop_start+1024"
        ),
    }


def _edit_box_visibility(
    edit_region: Any,
    native_crop: Iterable[float],
) -> dict[str, Any]:
    if (
        not isinstance(edit_region, list)
        or len(edit_region) != 4
        or any(not isinstance(value, int) for value in edit_region)
    ):
        raise ValueError("canonical edit_region_xyxy is invalid")
    box = [float(value) for value in edit_region]
    crop = [float(value) for value in native_crop]
    edit_area = _box_area(box)
    if edit_area <= 0:
        raise ValueError("canonical edit region has non-positive area")
    left = max(box[0], crop[0])
    top = max(box[1], crop[1])
    right = min(box[2], crop[2])
    bottom = min(box[3], crop[3])
    intersection = (
        [left, top, right, bottom]
        if right > left and bottom > top
        else None
    )
    visible_area = 0.0 if intersection is None else _box_area(intersection)
    fraction = min(1.0, max(0.0, visible_area / edit_area))
    category = (
        "none"
        if fraction == 0.0
        else "full"
        if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else "partial"
    )
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": crop,
        "intersection_xyxy": intersection,
        "edit_area": edit_area,
        "visible_area": visible_area,
        "visible_fraction": fraction,
        "category": category,
        "basis": (
            "continuous_edit_box_area_intersection_with_effective_native_crop"
        ),
    }


def _compare_nested(
    recorded: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 1e-12,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(recorded, Mapping):
            raise ValueError(f"{label} is not a mapping")
        if set(recorded) != set(expected):
            raise ValueError(
                f"{label} keys mismatch: {sorted(recorded)} != "
                f"{sorted(expected)}"
            )
        for key in expected:
            _compare_nested(
                recorded[key],
                expected[key],
                label=f"{label}.{key}",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(recorded, list) or len(recorded) != len(expected):
            raise ValueError(f"{label} list mismatch")
        for index, (actual_item, expected_item) in enumerate(
            zip(recorded, expected, strict=True)
        ):
            _compare_nested(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(expected, float):
        actual = _require_finite(recorded, label)
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=float_tolerance,
        ):
            raise ValueError(f"{label} mismatch: {actual} != {expected}")
        return
    _require_equal(recorded, expected, label)


def _preprocess_and_visibility_audit(
    *,
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    forged_input: Mapping[str, Any],
    config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    width = int(canonical["width"])
    height = int(canonical["height"])
    geometry = _official_preprocess_geometry(
        width=width,
        height=height,
        config=config,
    )
    recorded = _require_mapping(row.get("preprocess"), "result preprocess")
    _compare_nested(
        recorded,
        geometry,
        label=f"row {row.get('id')} preprocess",
        float_tolerance=1e-12,
    )
    evidence = _visible_gt_evidence(
        forged_input=forged_input,
        geometry=geometry,
        repo_root=repo_root,
    )
    _require_equal(
        row.get("edit_visibility"),
        evidence["category"],
        f"row {row.get('id')} edit visibility",
    )
    recorded_fraction = _require_finite(
        row.get("edit_visible_gt_fraction"),
        f"row {row.get('id')} visible GT fraction",
    )
    if not math.isclose(
        recorded_fraction,
        float(evidence["visible_fraction"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"row {row.get('id')} visible GT fraction mismatch: "
            f"{recorded_fraction} != {evidence['visible_fraction']}"
        )
    detail = row.get(
        "edit_visibility_evidence",
        row.get("edit_visibility_diagnostic"),
    )
    detail_map = _require_mapping(detail, "edit visibility evidence")
    expected_detail = {
        "gt": evidence,
        "edit_box": _edit_box_visibility(
            forged_input.get("edit_region_xyxy"),
            geometry["effective_native_crop_xyxy"],
        ),
    }
    _compare_nested(
        detail_map,
        expected_detail,
        label=f"row {row.get('id')} edit visibility evidence",
        float_tolerance=0.0,
    )
    return evidence


def _visibility_identity_audit(
    *,
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    forged_input: Mapping[str, Any],
    config: Mapping[str, Any],
    repo_root: Path,
    require_preprocess: bool,
) -> dict[str, Any]:
    geometry = _official_preprocess_geometry(
        width=int(canonical["width"]),
        height=int(canonical["height"]),
        config=config,
    )
    evidence = _visible_gt_evidence(
        forged_input=forged_input,
        geometry=geometry,
        repo_root=repo_root,
    )
    _require_equal(
        row.get("edit_visibility"),
        evidence["category"],
        f"row {row.get('id')} edit visibility",
    )
    _compare_float(
        row.get("edit_visible_gt_fraction"),
        float(evidence["visible_fraction"]),
        label=f"row {row.get('id')} edit visible GT fraction",
        absolute_tolerance=0.0,
    )
    expected_detail = {
        "gt": evidence,
        "edit_box": _edit_box_visibility(
            forged_input.get("edit_region_xyxy"),
            geometry["effective_native_crop_xyxy"],
        ),
    }
    detail = row.get(
        "edit_visibility_evidence",
        row.get("edit_visibility_diagnostic"),
    )
    _compare_nested(
        _require_mapping(detail, "edit visibility evidence"),
        expected_detail,
        label=f"row {row.get('id')} edit visibility evidence",
        float_tolerance=0.0,
    )
    if require_preprocess:
        _compare_nested(
            _require_mapping(row.get("preprocess"), "result preprocess"),
            geometry,
            label=f"row {row.get('id')} preprocess",
            float_tolerance=1e-12,
        )
    elif "preprocess" in row:
        raise ValueError(f"error row {row.get('id')} must not claim preprocessing")
    return evidence


def _runtime_package_version(runtime: Mapping[str, Any], name: str) -> str:
    packages = _require_mapping(runtime.get("packages"), "runtime packages")
    value = packages.get(name)
    if isinstance(value, Mapping):
        value = value.get("version")
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime has no exact {name} version")
    return value


def _recorded_device(runtime: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    accelerator = _require_mapping(
        runtime.get("accelerator"),
        "runtime accelerator",
    )
    value = accelerator.get(
        "inference_device",
        accelerator.get(
            "requested_device",
            accelerator.get("device", runtime.get("device")),
        ),
    )
    if not isinstance(value, str) or not value:
        raise ValueError("runtime has no recorded inference device")
    return value, accelerator


def _kernel_replay_runtime(
    manifest: Mapping[str, Any],
    *,
    requested_device: str | None = None,
) -> ReplayRuntime:
    runtime = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime contract",
    )
    import torch

    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    seed = protocol.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("cannot replay FSD without the recorded integer seed")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    recorded_python = _require_mapping(runtime.get("python"), "runtime python")
    for actual, expected, label in (
        (
            platform.python_implementation(),
            recorded_python.get("implementation"),
            "replay Python implementation",
        ),
        (
            platform.python_version(),
            recorded_python.get("version"),
            "replay Python version",
        ),
        (
            str(Path(sys.executable).resolve()),
            recorded_python.get("executable"),
            "replay Python executable",
        ),
    ):
        if expected is None:
            raise RuntimeError(f"cannot replay FSD without recorded {label}")
        _require_equal(actual, expected, label)
    recorded_torch = _runtime_package_version(runtime, "torch")
    if str(torch.__version__) != recorded_torch:
        raise RuntimeError(
            "cannot replay FSD: exact torch version differs "
            f"({torch.__version__!s} != {recorded_torch})"
        )
    torch_package = _require_mapping(
        _require_mapping(runtime.get("packages"), "runtime packages").get(
            "torch"
        ),
        "runtime torch package",
    )
    _require_equal(
        torch_package.get("full_version"),
        str(torch.__version__),
        "replay full torch version",
    )
    _require_equal(
        torch_package.get("distribution_version"),
        importlib.metadata.version("torch"),
        "replay torch distribution version",
    )
    _require_equal(
        _runtime_package_version(runtime, "numpy"),
        str(np.__version__),
        "replay NumPy version",
    )
    _require_equal(
        _runtime_package_version(runtime, "Pillow"),
        importlib.metadata.version("Pillow"),
        "replay Pillow version",
    )
    device_string, accelerator = _recorded_device(runtime)
    _require_equal(
        accelerator.get("torch_version"),
        str(torch.__version__),
        "replay accelerator torch version",
    )
    _require_equal(
        accelerator.get("torch_distribution_version"),
        importlib.metadata.version("torch"),
        "replay accelerator torch distribution version",
    )
    if requested_device is not None and requested_device != device_string:
        raise RuntimeError(
            "cannot replay FSD on a device different from the recorded run"
        )
    device = torch.device(device_string)
    _require_equal(
        device.type,
        accelerator.get("device_type"),
        "replay device type",
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cannot replay CUDA FSD run: CUDA unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        _require_equal(
            index,
            accelerator.get("device_index"),
            "replay CUDA device index",
        )
        actual_name = torch.cuda.get_device_name(index)
        actual_capability = list(torch.cuda.get_device_capability(index))
        recorded_name = accelerator.get(
            "gpu_name",
            accelerator.get("device_name"),
        )
        recorded_capability = accelerator.get(
            "gpu_capability",
            accelerator.get("compute_capability"),
        )
        if recorded_name is None or recorded_capability is None:
            raise RuntimeError(
                "cannot replay CUDA FSD run without GPU name/capability"
            )
        _require_equal(actual_name, recorded_name, "replay GPU name")
        _require_equal(
            actual_capability,
            list(recorded_capability),
            "replay GPU capability",
        )
        recorded_cuda = accelerator.get(
            "torch_cuda_version",
            accelerator.get(
                "torch_cuda",
                accelerator.get("cuda_version"),
            ),
        )
        if recorded_cuda is None:
            raise RuntimeError("cannot replay CUDA FSD run without CUDA version")
        _require_equal(
            str(torch.version.cuda),
            str(recorded_cuda),
            "replay CUDA version",
        )
        _require_equal(
            (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            accelerator.get("cudnn_version"),
            "replay cuDNN version",
        )
    elif device.type == "cpu":
        machine = accelerator.get("machine")
        processor = accelerator.get("processor")
        if machine is not None:
            _require_equal(platform.machine(), machine, "replay CPU machine")
        if processor is not None:
            _require_equal(
                platform.processor(),
                processor,
                "replay CPU processor",
            )
    else:
        raise RuntimeError(f"unsupported recorded replay device: {device}")
    recorded_flags = _require_mapping(
        runtime.get("numerical_flags"),
        "runtime numerical flags",
    )
    actual_flags = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    _require_equal(
        actual_flags,
        recorded_flags,
        "replay numerical flags",
    )
    return ReplayRuntime(
        torch=torch,
        device=device,
        recorded_device=device_string,
        evidence={
            "torch": str(torch.__version__),
            "device": device_string,
            "device_type": device.type,
            "fail_closed": True,
        },
    )


@contextlib.contextmanager
def _pinned_fsd_modules(source_root: Path):
    """Import ``fsd`` exclusively from the verified pinned source tree."""

    source = source_root.resolve()
    old_path = list(sys.path)
    old_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "fsd" or key.startswith("fsd.")
    }
    for key in list(old_modules):
        del sys.modules[key]
    sys.path.insert(0, str(source))
    try:
        projection = importlib.import_module("fsd.projection")
        gmm = importlib.import_module("fsd.gmm")
        for module in (projection, gmm):
            module_file = Path(str(module.__file__)).resolve()
            try:
                module_file.relative_to(source)
            except ValueError as error:
                raise RuntimeError(
                    f"official FSD replay imported outside pinned source: "
                    f"{module_file}"
                ) from error
        yield projection, gmm
    finally:
        for key in [
            name
            for name in list(sys.modules)
            if name == "fsd" or name.startswith("fsd.")
        ]:
            del sys.modules[key]
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def _load_descriptor(row: Mapping[str, Any], repo_root: Path) -> np.ndarray:
    path_value = row.get("raw_descriptor_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"row {row.get('id')} has no raw descriptor path")
    path = _anchored(Path(path_value), repo_root)
    _verify_hash(
        path,
        row.get("raw_descriptor_sha256"),
        f"row {row.get('id')} raw descriptor",
    )
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"cannot load raw descriptor {path}: {error}") from error
    if not isinstance(value, np.ndarray):
        raise ValueError("raw descriptor artifact is not an ndarray")
    if value.shape != (DESCRIPTOR_DIMENSION,):
        raise ValueError(
            f"raw descriptor shape mismatch: {value.shape} != "
            f"{(DESCRIPTOR_DIMENSION,)}"
        )
    if value.dtype != DESCRIPTOR_DTYPE:
        raise ValueError(
            f"raw descriptor dtype mismatch: {value.dtype} != float64"
        )
    if not np.isfinite(value).all():
        raise ValueError("raw descriptor contains non-finite values")
    _require_equal(
        row.get("raw_descriptor_shape"),
        [DESCRIPTOR_DIMENSION],
        f"row {row.get('id')} descriptor shape metadata",
    )
    _require_equal(
        row.get("raw_descriptor_dtype"),
        "float64",
        f"row {row.get('id')} descriptor dtype metadata",
    )
    array_digest = row.get("raw_descriptor_array_sha256")
    if array_digest is not None:
        _require_equal(
            _require_sha256(array_digest, "descriptor array SHA-256"),
            _array_sha256(value),
            f"row {row.get('id')} descriptor array SHA-256",
        )
    return value


def _compare_float(
    actual: Any,
    expected: float,
    *,
    label: str,
    absolute_tolerance: float,
) -> None:
    value = _require_finite(actual, label)
    if not math.isclose(
        value,
        float(expected),
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(f"{label} mismatch: {value} != {expected}")


def _audit_score_fields(
    *,
    row: Mapping[str, Any],
    raw_likelihood: float,
    train_mean: float,
    train_std: float,
    z_threshold: float,
    ai_threshold: float,
) -> dict[str, Any]:
    z_score = (raw_likelihood - train_mean) / train_std
    ai_score = -z_score
    released_is_fake = z_score < z_threshold
    classification_decision = ai_score > ai_threshold
    if released_is_fake != classification_decision:
        raise AssertionError("released/CLAIMFORGE threshold directions diverged")

    _compare_float(
        row.get("raw_likelihood"),
        raw_likelihood,
        label=f"row {row.get('id')} raw likelihood",
        absolute_tolerance=RAW_SCORE_ABSOLUTE_TOLERANCE,
    )
    _compare_float(
        row.get("released_z_score"),
        z_score,
        label=f"row {row.get('id')} released z score",
        absolute_tolerance=DERIVED_SCORE_ABSOLUTE_TOLERANCE,
    )
    _compare_float(
        row.get("ai_score"),
        ai_score,
        label=f"row {row.get('id')} AI score",
        absolute_tolerance=DERIVED_SCORE_ABSOLUTE_TOLERANCE,
    )
    _compare_float(
        row.get("score"),
        ai_score,
        label=f"row {row.get('id')} top-level score",
        absolute_tolerance=DERIVED_SCORE_ABSOLUTE_TOLERANCE,
    )
    _require_equal(
        row.get("score_semantics"),
        "negative_released_FSD_z_score",
        f"row {row.get('id')} score semantics",
    )
    _require_equal(
        row.get("released_is_fake"),
        released_is_fake,
        f"row {row.get('id')} released decision",
    )
    _require_equal(
        row.get("classification_decision"),
        classification_decision,
        f"row {row.get('id')} classification decision",
    )
    _compare_float(
        row.get("classification_threshold"),
        ai_threshold,
        label=f"row {row.get('id')} classification threshold",
        absolute_tolerance=0.0,
    )
    _require_equal(
        row.get("classification_threshold_operator"),
        THRESHOLD_OPERATOR,
        f"row {row.get('id')} threshold operator",
    )
    _compare_float(
        row.get("released_threshold"),
        z_threshold,
        label=f"row {row.get('id')} released threshold",
        absolute_tolerance=0.0,
    )
    _require_equal(
        row.get("released_threshold_operator"),
        "<",
        f"row {row.get('id')} released threshold operator",
    )
    _require_equal(
        row.get("valid_for_t1"),
        True,
        f"row {row.get('id')} valid_for_t1",
    )
    _require_equal(
        row.get("valid_for_t2"),
        False,
        f"row {row.get('id')} valid_for_t2",
    )

    t1 = _require_mapping(row.get("t1"), f"row {row.get('id')} T1 evidence")
    aliases = {
        "score": ai_score,
        "raw_likelihood": raw_likelihood,
        "released_z_score": z_score,
        "threshold": ai_threshold,
        "threshold_operator": THRESHOLD_OPERATOR,
        "decision": classification_decision,
    }
    for container_name, container in (
        ("T1", t1),
        (
            "classification",
            _require_mapping(
                row.get("classification"),
                f"row {row.get('id')} classification evidence",
            ),
        ),
    ):
        for key, expected in aliases.items():
            if key not in container:
                raise ValueError(
                    f"row {row.get('id')} {container_name} lacks {key}"
                )
            if isinstance(expected, float):
                tolerance = (
                    RAW_SCORE_ABSOLUTE_TOLERANCE
                    if key == "raw_likelihood"
                    else DERIVED_SCORE_ABSOLUTE_TOLERANCE
                )
                _compare_float(
                    container[key],
                    expected,
                    label=f"row {row.get('id')} {container_name} {key}",
                    absolute_tolerance=tolerance,
                )
            else:
                _require_equal(
                    container[key],
                    expected,
                    f"row {row.get('id')} {container_name} {key}",
                )
    _require_equal(
        t1.get("policy"),
        "released_FSD_whole_image_score_sign_inverted",
        f"row {row.get('id')} T1 policy",
    )
    classification = _require_mapping(
        row.get("classification"),
        f"row {row.get('id')} classification evidence",
    )
    _require_equal(
        classification.get("semantics"),
        "higher_is_more_AI_negative_released_z",
        f"row {row.get('id')} classification semantics",
    )
    replay = _require_mapping(
        row.get("manual_replay"),
        f"row {row.get('id')} manual replay",
    )
    for key, expected in {
        "raw_likelihood": raw_likelihood,
        "released_z_score": z_score,
        "ai_score": ai_score,
        "released_is_fake": released_is_fake,
        "classification_decision": classification_decision,
    }.items():
        if key not in replay:
            raise ValueError(f"row {row.get('id')} replay lacks {key}")
        if isinstance(expected, float):
            _compare_float(
                replay[key],
                expected,
                label=f"row {row.get('id')} replay {key}",
                absolute_tolerance=(
                    RAW_SCORE_ABSOLUTE_TOLERANCE
                    if key == "raw_likelihood"
                    else DERIVED_SCORE_ABSOLUTE_TOLERANCE
                ),
            )
        else:
            _require_equal(
                replay[key],
                expected,
                f"row {row.get('id')} replay {key}",
            )
    for key, expected in {
        "official_raw_exact_match": True,
        "official_z_exact_match": True,
        "compute_fsd_calls": 1,
    }.items():
        _require_equal(
            replay.get(key),
            expected,
            f"row {row.get('id')} replay {key}",
        )
    return {
        "raw_likelihood": raw_likelihood,
        "released_z_score": z_score,
        "ai_score": ai_score,
        "released_is_fake": released_is_fake,
        "classification_decision": classification_decision,
    }


def _validate_physical_result_payload(
    row: Mapping[str, Any],
    *,
    row_label: str,
    repo_root: Path,
    pins: SimpleNamespace,
) -> None:
    status = row.get("status")
    if status not in {"ok", "error"}:
        raise ValueError(f"{row_label} has invalid status {status!r}")
    visibility = row.get("edit_visibility")
    if visibility not in {"none", "partial", "full"}:
        raise ValueError(f"{row_label} has invalid edit_visibility")
    fraction = _require_finite(
        row.get("edit_visible_gt_fraction"),
        f"{row_label} edit_visible_gt_fraction",
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{row_label} edit_visible_gt_fraction outside [0, 1]")
    _require_mapping(
        row.get("edit_visibility_evidence"),
        f"{row_label} edit visibility evidence",
    )
    _require_equal(row.get("valid_for_t1"), True, f"{row_label} valid_for_t1")
    _require_equal(row.get("valid_for_t2"), False, f"{row_label} valid_for_t2")
    _require_equal(
        row.get("t1_policy"),
        "released_FSD_v1.2.0_whole_image_z_score",
        f"{row_label} T1 policy",
    )
    _require_equal(
        row.get("t2_policy"),
        "unsupported_whole_image_detector",
        f"{row_label} T2 policy",
    )

    if status == "error":
        _require_equal(
            row.get("valid_for_metrics"),
            False,
            f"{row_label} valid_for_metrics",
        )
        for key in (
            "error_type",
            "error_message",
            "traceback",
            "completed_at",
        ):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{row_label} has invalid {key}")
        forbidden = {
            "score",
            "raw_likelihood",
            "released_z_score",
            "ai_score",
            "classification",
            "t1",
            "manual_replay",
            "raw_descriptor_path",
            "preprocess",
            "latency_ms",
            "peak_cuda_memory_bytes",
        }.intersection(row)
        if forbidden:
            raise ValueError(
                f"{row_label} error payload claims success fields: "
                f"{sorted(forbidden)}"
            )
        return

    _require_equal(
        row.get("valid_for_metrics"),
        True,
        f"{row_label} valid_for_metrics",
    )
    raw = _require_finite(row.get("raw_likelihood"), f"{row_label} raw likelihood")
    _audit_score_fields(
        row=row,
        raw_likelihood=raw,
        train_mean=float(pins.EXPECTED_CONFIG["scoring"]["train_mean"]),
        train_std=float(pins.EXPECTED_CONFIG["scoring"]["train_std"]),
        z_threshold=float(pins.RELEASED_Z_THRESHOLD),
        ai_threshold=float(pins.AI_SCORE_THRESHOLD),
    )
    _load_descriptor(row, repo_root)
    latency = _require_finite(row.get("latency_ms"), f"{row_label} latency_ms")
    if latency < 0.0:
        raise ValueError(f"{row_label} latency_ms is negative")
    peak = row.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError(f"{row_label} peak CUDA memory is invalid")
    if not isinstance(row.get("completed_at"), str):
        raise ValueError(f"{row_label} completed_at is invalid")


def _row_provenance_identity(
    row: Mapping[str, Any],
    *,
    row_label: str,
    run_id: str,
    fingerprint: str,
    source_commit: str,
    weights_bundle_sha256: str,
    inputs_sha256: str,
) -> None:
    _require_equal(row.get("run_id"), run_id, f"{row_label} run ID")
    _require_equal(
        row.get("run_manifest_fingerprint"),
        fingerprint,
        f"{row_label} run fingerprint",
    )
    _require_equal(
        row.get("input_manifest_sha256"),
        inputs_sha256,
        f"{row_label} input manifest SHA-256",
    )
    source_value = row.get(
        "source_commit",
        row.get("model_source_commit"),
    )
    _require_equal(
        source_value,
        source_commit,
        f"{row_label} source commit",
    )
    bundle_value = row.get(
        "weights_bundle_sha256",
        row.get("weight_bundle_sha256"),
    )
    _require_equal(
        bundle_value,
        weights_bundle_sha256,
        f"{row_label} weights bundle SHA-256",
    )


def _validate_row_identity(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
    row_label: str,
) -> None:
    expected = {
        "id": canonical["sample_id"],
        "rank": canonical["rank"],
        "task_id": canonical["task_id"],
        "pair_rank": canonical["pair_rank"],
        "domain": canonical["domain"],
        "kind": canonical["kind"],
        "label": canonical["label"],
        "gt_mask_kind": canonical["gt_mask_kind"],
        "gt_mask_sha256": canonical.get("gt_mask_sha256"),
        "edit_region_xyxy": canonical.get("edit_region_xyxy"),
    }
    for key, value in expected.items():
        _require_equal(row.get(key), value, f"{row_label} {key}")
    image_value = row.get("image_path")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{row_label} has no image path")
    image_path = _anchored(Path(image_value), repo_root)
    canonical_path = _anchored(Path(str(canonical["canonical_path"])), repo_root)
    _require_equal(image_path, canonical_path, f"{row_label} canonical path")
    _verify_hash(
        image_path,
        canonical["canonical_sha256"],
        f"{row_label} canonical input",
    )
    _require_equal(
        row.get("image_sha256"),
        canonical["canonical_sha256"],
        f"{row_label} image SHA-256",
    )
    _require_equal(
        row.get("image_size"),
        [int(canonical["width"]), int(canonical["height"])],
        f"{row_label} image size",
    )
    _require_equal(
        row.get("model"),
        MODEL_NAME,
        f"{row_label} model",
    )
    _require_equal(
        row.get("model_slug"),
        MODEL_SLUG,
        f"{row_label} model slug",
    )
    with Image.open(image_path) as image:
        _require_equal(
            [image.width, image.height],
            [int(canonical["width"]), int(canonical["height"])],
            f"{row_label} decoded dimensions",
        )


def _paper_release_drift(
    manifest: Mapping[str, Any],
    pins: SimpleNamespace,
) -> dict:
    value = manifest.get("paper_release_drift")
    drift = _require_mapping(value, "manifest paper/release drift")
    _require_equal(
        drift,
        pins.PAPER_RELEASE_DRIFT,
        "frozen paper/release drift record",
    )
    claim = drift.get("evaluation_claim")
    if (
        not isinstance(claim, str)
        or "inference release" not in claim
        or "paper-protocol parity" not in claim
    ):
        raise ValueError("paper/release drift evaluation claim is incomplete")
    source_drift = _require_mapping(
        manifest.get("source_tag_drift"),
        "manifest source/tag drift",
    )
    _require_equal(
        source_drift,
        pins.SOURCE_TAG_DRIFT,
        "frozen source/tag drift record",
    )
    return dict(drift)


def validate_provenance(
    *,
    repo_root: Path,
    fsd_root: Path,
    run_id: str,
    inputs_path: Path,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate every physical row and every immutable source/input asset."""

    pins = _load_runner_pins()
    _require_equal(
        manifest.get("schema_version"),
        "opensource_run_manifest_v1",
        "run manifest schema",
    )
    _require_equal(manifest.get("run_id"), run_id, "run manifest ID")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "run manifest fingerprint",
    )
    _require_equal(
        fingerprint,
        _manifest_fingerprint(manifest),
        "run manifest fingerprint",
    )

    model = _require_mapping(manifest.get("model"), "manifest model")
    for actual, expected, label in (
        (model.get("name"), pins.MODEL_NAME, "manifest model name"),
        (model.get("slug"), pins.MODEL_SLUG, "manifest model slug"),
        (model.get("repo_url"), pins.MODEL_REPO_URL, "manifest repository URL"),
        (
            model.get("source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "manifest source commit",
        ),
        (
            model.get("release_tag"),
            pins.RELEASE_TAG,
            "manifest release tag",
        ),
    ):
        _require_equal(actual, expected, label)
    source_root_value = model.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ValueError("manifest model has no source_root")
    _require_equal(
        Path(source_root_value).resolve(),
        fsd_root.resolve(),
        "manifest FSD source root",
    )
    _require_equal(
        model.get("task_support"),
        {"t1": True, "t2": False},
        "manifest task support",
    )
    license_record = _require_mapping(
        model.get("license"),
        "manifest FSD license",
    )
    _require_equal(
        license_record,
        {
            "spdx": "CC-BY-NC-SA-4.0",
            "commercial_use": False,
            "share_alike": True,
        },
        "manifest FSD license",
    )
    expected_source = _normalise_source_files(
        pins.SOURCE_FILES,
        "runner source files",
    )
    source_audit = _require_mapping(
        model.get("source_audit"),
        "manifest source audit",
    )
    recorded_source_raw = _require_mapping(
        source_audit.get("source_files"),
        "manifest source files",
    )
    recorded_source = _normalise_source_files(
        {
            path: _require_mapping(record, f"source record {path}").get(
                "sha256"
            )
            for path, record in recorded_source_raw.items()
        },
        "manifest source files",
    )
    _require_equal(
        recorded_source,
        expected_source,
        "manifest frozen source file hashes",
    )
    _verify_source_tree(
        fsd_root,
        expected_commit=pins.MODEL_SOURCE_COMMIT,
        expected_files=expected_source,
        label="official FSD",
    )
    for actual, expected, label in (
        (source_audit.get("repo_url"), pins.MODEL_REPO_URL, "source audit URL"),
        (
            Path(str(source_audit.get("root"))).resolve(),
            fsd_root.resolve(),
            "source audit root",
        ),
        (
            source_audit.get("commit"),
            pins.MODEL_SOURCE_COMMIT,
            "source audit commit",
        ),
        (
            source_audit.get("tracked_dirty"),
            False,
            "source audit dirty flag",
        ),
        (
            source_audit.get("source_tag_drift"),
            pins.SOURCE_TAG_DRIFT,
            "source audit tag drift",
        ),
    ):
        _require_equal(actual, expected, label)
    _require_equal(
        _git_value(fsd_root, "rev-parse", f"{pins.RELEASE_TAG}^{{commit}}"),
        pins.RELEASE_TAG_COMMIT,
        "physical release tag commit",
    )
    _require_equal(
        _git_value(
            fsd_root,
            "rev-list",
            "--count",
            f"{pins.RELEASE_TAG}..HEAD",
        ),
        str(pins.SOURCE_TAG_DRIFT["commits_ahead_of_tag"]),
        "physical source commits ahead of tag",
    )
    changed_text = _git_value(
        fsd_root,
        "diff",
        "--name-only",
        f"{pins.RELEASE_TAG}..HEAD",
    )
    _require_equal(
        sorted(changed_text.splitlines()) if changed_text else [],
        sorted(pins.SOURCE_TAG_DRIFT["changed_files"]),
        "physical source/tag changed files",
    )
    for path, record in recorded_source_raw.items():
        item = _require_mapping(record, f"source record {path}")
        _require_equal(
            Path(str(item.get("path"))).resolve(),
            (fsd_root / path).resolve(),
            f"source record {path} path",
        )
    weights_dir, weights, bundle, config = _validate_weight_files(
        model=model,
        repo_root=repo_root,
        pins=pins,
    )
    drift = _paper_release_drift(manifest, pins)
    runtime_contract = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime contract",
    )
    _require_equal(
        manifest.get("environment"),
        runtime_contract,
        "manifest environment/runtime compatibility copy",
    )
    _runtime_package_version(runtime_contract, "torch")
    _runtime_package_version(runtime_contract, "numpy")
    _runtime_package_version(runtime_contract, "Pillow")
    accelerator = _require_mapping(
        runtime_contract.get("accelerator"),
        "runtime accelerator",
    )
    for key in ("requested_device", "device_type", "machine", "processor"):
        if key not in accelerator:
            raise ValueError(f"runtime accelerator lacks {key}")
    flags = _require_mapping(
        runtime_contract.get("numerical_flags"),
        "runtime numerical flags",
    )
    for key, expected in {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }.items():
        _require_equal(flags.get(key), expected, f"runtime numerical flag {key}")

    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    descriptor_contract = _require_mapping(
        protocol.get("descriptor"),
        "manifest descriptor protocol",
    )
    _require_equal(
        descriptor_contract,
        {
            "shape": [int(pins.FSD_DIMENSION)],
            "dtype": "float64",
            "persisted_for_every_ok_image": True,
        },
        "manifest descriptor protocol",
    )
    classification_contract = _require_mapping(
        protocol.get("classification"),
        "manifest classification protocol",
    )
    _require_equal(
        classification_contract,
        {
            "released_threshold": float(pins.RELEASED_Z_THRESHOLD),
            "released_operator": "<",
            "ai_score_threshold": float(pins.AI_SCORE_THRESHOLD),
            "ai_score_operator": ">",
            "strict": True,
        },
        "manifest classification protocol",
    )
    _require_equal(
        protocol.get("bootstrap_unit"),
        "task_id_pair",
        "manifest bootstrap unit",
    )
    if (
        isinstance(protocol.get("bootstrap_samples"), bool)
        or not isinstance(protocol.get("bootstrap_samples"), int)
        or int(protocol["bootstrap_samples"]) <= 0
    ):
        raise ValueError("manifest bootstrap sample count is invalid")
    if isinstance(protocol.get("seed"), bool) or not isinstance(
        protocol.get("seed"), int
    ):
        raise ValueError("manifest bootstrap seed is invalid")

    manifest_input = _require_mapping(manifest.get("dataset"), "manifest dataset")
    inputs_digest = sha256_file(inputs_path)
    _require_equal(
        manifest_input.get("inputs_sha256"),
        inputs_digest,
        "manifest input SHA-256",
    )
    input_value = manifest_input.get("inputs_path")
    if not isinstance(input_value, str):
        raise ValueError("manifest input has no inputs_manifest path")
    _require_equal(
        _anchored(Path(input_value), repo_root),
        inputs_path.resolve(),
        "manifest canonical input path",
    )
    release_value = manifest_input.get("manifest_path")
    if not isinstance(release_value, str) or not release_value:
        raise ValueError("manifest dataset has no canonical release path")
    release_path = _anchored(Path(release_value), repo_root)
    _verify_hash(
        release_path,
        manifest_input.get("manifest_sha256"),
        "canonical dataset manifest",
    )
    release = _require_mapping(
        json.loads(release_path.read_text(encoding="utf-8")),
        "canonical dataset manifest",
    )
    _require_equal(
        release.get("schema_version"),
        "claimforge_mouse_canonical_v1",
        "canonical dataset schema",
    )
    _require_equal(
        manifest_input.get("dataset_id"),
        release.get("dataset_id"),
        "manifest dataset ID",
    )
    _require_equal(
        release.get("inputs_sha256"),
        inputs_digest,
        "canonical release input SHA-256",
    )
    _require_equal(
        _anchored(Path(str(release.get("inputs_path"))), repo_root),
        inputs_path.resolve(),
        "canonical release input path",
    )

    selection = _selection_contract(input_rows)
    selection_record = _require_mapping(
        manifest.get("selection"),
        "manifest selection",
    )
    _require_equal(
        selection_record.get("rows"),
        selection,
        "manifest selected input contract",
    )
    _require_equal(
        manifest.get("expected_images"),
        len(input_rows),
        "manifest expected images",
    )
    pair_kinds: dict[str, set[str]] = defaultdict(set)
    for row in input_rows:
        pair_kinds[str(row["task_id"])].add(str(row["kind"]))
    expected_pairs = sum(value == {"real", "forged"} for value in pair_kinds.values())
    _require_equal(
        manifest.get("expected_complete_pairs"),
        expected_pairs,
        "manifest expected complete pairs",
    )

    expected_by_id = {str(row["sample_id"]): row for row in input_rows}
    all_canonical_rows = read_jsonl(inputs_path)
    forged_by_task = {
        str(row["task_id"]): row
        for row in all_canonical_rows
        if row.get("kind") == "forged"
    }
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if row_id not in expected_by_id:
            raise ValueError(
                f"physical result row {line_number} has unexpected ID {row_id!r}"
            )
        label = f"physical result row {line_number} ({row_id})"
        _validate_row_identity(
            row,
            expected_by_id[str(row_id)],
            repo_root=repo_root,
            row_label=label,
        )
        _row_provenance_identity(
            row,
            row_label=label,
            run_id=run_id,
            fingerprint=fingerprint,
            source_commit=pins.MODEL_SOURCE_COMMIT,
            weights_bundle_sha256=bundle,
            inputs_sha256=inputs_digest,
        )
        _validate_physical_result_payload(
            row,
            row_label=label,
            repo_root=repo_root,
            pins=pins,
        )
        forged_input = forged_by_task.get(
            str(expected_by_id[str(row_id)]["task_id"])
        )
        if forged_input is None:
            raise ValueError(f"{label} has no canonical forged visibility reference")
        _visibility_identity_audit(
            row=row,
            canonical=expected_by_id[str(row_id)],
            forged_input=forged_input,
            config=config,
            repo_root=repo_root,
            require_preprocess=row.get("status") == "ok",
        )
        _reject_localization_contract(row, label=label)

    latest = _latest_by_id(result_rows)
    if set(latest) != set(expected_by_id):
        raise ValueError("latest physical result IDs do not equal manifest inputs")
    failed = [
        row_id
        for row_id, row in latest.items()
        if row.get("status") != "ok"
    ]
    if failed:
        raise ValueError(f"latest physical FSD rows are not ok: {failed[:5]}")

    _require_equal(summary.get("run_id"), run_id, "summary run ID")
    _require_equal(
        summary.get("run_manifest_fingerprint"),
        fingerprint,
        "summary run fingerprint",
    )
    summary_bundle = summary.get(
        "weights_bundle_sha256",
        summary.get("weight_bundle_sha256"),
    )
    _require_equal(summary_bundle, bundle, "summary weights bundle SHA-256")
    _require_equal(
        summary.get("input_manifest_sha256"),
        inputs_digest,
        "summary input manifest SHA-256",
    )
    for actual, expected, label in (
        (summary.get("model"), pins.MODEL_NAME, "summary model name"),
        (summary.get("model_slug"), pins.MODEL_SLUG, "summary model slug"),
        (
            summary.get("model_source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "summary source commit",
        ),
        (summary.get("release_tag"), pins.RELEASE_TAG, "summary release tag"),
        (summary.get("valid_for_t1"), True, "summary valid_for_t1"),
        (summary.get("valid_for_t2"), False, "summary valid_for_t2"),
    ):
        _require_equal(actual, expected, label)
    adapter_contract = _require_list(
        manifest.get("adapter_contract"),
        "manifest adapter contract",
    )
    if len(adapter_contract) != 3:
        raise ValueError("FSD adapter contract must bind runner/common/metrics")
    adapter_paths: set[Path] = set()
    for index, raw in enumerate(adapter_contract):
        item = _require_mapping(raw, f"adapter contract entry {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"adapter contract entry {index} has no path")
        path = _anchored(Path(path_value), repo_root)
        if path in adapter_paths:
            raise ValueError("adapter contract repeats a file")
        adapter_paths.add(path)
        _verify_hash(
            path,
            item.get("sha256"),
            f"adapter contract entry {index}",
        )
    expected_adapter_names = {"run_fsd.py", "common.py", "whole_image_metrics.py"}
    _require_equal(
        {path.name for path in adapter_paths},
        expected_adapter_names,
        "adapter contract filenames",
    )

    _reject_localization_contract(summary, label="summary")
    _reject_localization_contract(manifest, label="manifest")
    return {
        "run_manifest_fingerprint": fingerprint,
        "source_commit": pins.MODEL_SOURCE_COMMIT,
        "source_files_validated": len(expected_source),
        "weights_dir": _relative_or_absolute(weights_dir, repo_root),
        "weight_files_validated": len(weights),
        "weights_bundle_sha256": bundle,
        "config": config,
        "paper_release_drift": drift,
        "canonical_dataset_manifest_sha256": sha256_file(release_path),
        "adapter_files_validated": len(adapter_contract),
        "physical_result_rows_validated": len(result_rows),
        "latest_rows_validated": len(latest),
        "input_images_validated": len(input_rows),
    }


def _load_pairs(
    latest: Mapping[str, dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> list[DetectionPair]:
    canonical_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    result_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for canonical in input_rows:
        task_id = str(canonical["task_id"])
        kind = str(canonical["kind"])
        canonical_by_task[task_id][kind] = canonical
        result_by_task[task_id][kind] = latest[str(canonical["sample_id"])]
    pairs: list[DetectionPair] = []
    for task_id, canonical in canonical_by_task.items():
        if set(canonical) != {"real", "forged"}:
            continue
        results = result_by_task[task_id]
        if set(results) != {"real", "forged"}:
            raise ValueError(f"pair {task_id} has incomplete result kinds")
        real = results["real"]
        forged = results["forged"]
        _require_equal(
            real.get("edit_visibility"),
            forged.get("edit_visibility"),
            f"pair {task_id} visibility",
        )
        _compare_float(
            real.get("edit_visible_gt_fraction"),
            _require_finite(
                forged.get("edit_visible_gt_fraction"),
                f"pair {task_id} forged visible fraction",
            ),
            label=f"pair {task_id} visible fraction",
            absolute_tolerance=0.0,
        )
        pairs.append(
            DetectionPair(
                task_id=task_id,
                domain=str(forged["domain"]),
                real=real,
                forged=forged,
                forged_input=canonical["forged"],
            )
        )
    pairs.sort(key=lambda pair: int(pair.forged["pair_rank"]))
    return pairs


def audit_artifacts(
    *,
    repo_root: Path,
    fsd_root: Path,
    manifest: Mapping[str, Any],
    input_rows: list[dict[str, Any]],
    latest: Mapping[str, dict[str, Any]],
    runtime: ReplayRuntime | None = None,
    projection_module: ModuleType | None = None,
    gmm_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Replay all descriptors through official transforms and released GMM."""

    pins = _load_runner_pins()
    model = _require_mapping(manifest.get("model"), "manifest model")
    weights_dir, _, _, config = _validate_weight_files(
        model=model,
        repo_root=repo_root,
        pins=pins,
    )
    replay_runtime = runtime or _kernel_replay_runtime(manifest)
    contexts: Any
    if projection_module is None or gmm_module is None:
        contexts = _pinned_fsd_modules(fsd_root)
    else:
        contexts = contextlib.nullcontext((projection_module, gmm_module))

    input_by_id = {str(row["sample_id"]): row for row in input_rows}
    dataset_record = _require_mapping(manifest.get("dataset"), "manifest dataset")
    all_inputs_value = dataset_record.get("inputs_path")
    if not isinstance(all_inputs_value, str) or not all_inputs_value:
        raise ValueError("manifest dataset has no inputs path")
    all_canonical_rows = read_jsonl(
        _anchored(Path(all_inputs_value), repo_root)
    )
    forged_by_task = {
        str(row["task_id"]): row
        for row in all_canonical_rows
        if row.get("kind") == "forged"
    }
    score_differences: list[float] = []
    raw_differences: list[float] = []
    visibility_by_task: dict[str, dict[str, Any]] = {}
    selected_kinds_by_task: dict[str, set[str]] = defaultdict(set)
    with contexts as (projection, gmm_api):
        transforms = projection.load_transforms(
            weights_dir / "fsd_transforms.pt",
            device=replay_runtime.device,
        )
        if len(transforms) != EXPECTED_TRANSFORMS:
            raise ValueError(
                f"released FSD transform count mismatch: {len(transforms)} "
                f"!= {EXPECTED_TRANSFORMS}"
            )
        gmm = gmm_api.load_gmm(
            weights_dir / "gmm.pt",
            device=replay_runtime.device,
        )
        train_mean = float(config["scoring"]["train_mean"])
        train_std = float(config["scoring"]["train_std"])
        if not math.isfinite(train_std) or train_std <= 0:
            raise ValueError("official FSD train_std is invalid")

        for canonical in input_rows:
            sample_id = str(canonical["sample_id"])
            row = latest[sample_id]
            descriptor = _load_descriptor(row, repo_root)
            tensor = replay_runtime.torch.from_numpy(descriptor).to(
                replay_runtime.device,
                dtype=replay_runtime.torch.float64,
            )[None, :]
            with replay_runtime.torch.no_grad():
                transformed = projection.apply_projections(tensor, transforms)
                raw_likelihood = float(gmm.score_samples(transformed).item())
            if not math.isfinite(raw_likelihood):
                raise ValueError(f"row {sample_id} replay produced non-finite score")
            recorded_raw = _require_finite(
                row.get("raw_likelihood"),
                f"row {sample_id} raw likelihood",
            )
            recorded_ai = _require_finite(
                row.get("ai_score"),
                f"row {sample_id} AI score",
            )
            replay = _audit_score_fields(
                row=row,
                raw_likelihood=raw_likelihood,
                train_mean=train_mean,
                train_std=train_std,
                z_threshold=float(pins.RELEASED_Z_THRESHOLD),
                ai_threshold=float(pins.AI_SCORE_THRESHOLD),
            )
            raw_differences.append(abs(recorded_raw - raw_likelihood))
            score_differences.append(abs(recorded_ai - replay["ai_score"]))
            forged_input = forged_by_task.get(str(canonical["task_id"]))
            if forged_input is None:
                raise ValueError(
                    f"row {sample_id} has no paired forged canonical input"
                )
            visibility = _preprocess_and_visibility_audit(
                row=row,
                canonical=canonical,
                forged_input=forged_input,
                config=config,
                repo_root=repo_root,
            )
            selected_kinds_by_task[str(canonical["task_id"])].add(
                str(canonical["kind"])
            )
            prior = visibility_by_task.setdefault(
                str(canonical["task_id"]),
                visibility,
            )
            _require_equal(
                visibility,
                prior,
                f"pair {canonical['task_id']} independent visibility replay",
            )

    visibility_task_counts = Counter(
        str(value["category"]) for value in visibility_by_task.values()
    )
    complete_pair_visibility_counts = Counter(
        str(visibility_by_task[task_id]["category"])
        for task_id, kinds in selected_kinds_by_task.items()
        if kinds == {"real", "forged"}
    )
    incomplete_tasks = sorted(
        task_id
        for task_id, kinds in selected_kinds_by_task.items()
        if kinds != {"real", "forged"}
    )
    return {
        "images_replayed": len(input_rows),
        "descriptors_validated": len(input_rows),
        "descriptor_shape": [DESCRIPTOR_DIMENSION],
        "descriptor_dtype": "float64",
        "transforms_replayed": EXPECTED_TRANSFORMS,
        "gmm_likelihood_replayed": True,
        "normalization_replayed": (
            "released_z=(raw_likelihood-train_mean)/train_std; ai_score=-z"
        ),
        "strict_decision_replayed": "ai_score > 2.0",
        "maximum_raw_likelihood_absolute_difference": max(
            raw_differences,
            default=0.0,
        ),
        "maximum_ai_score_absolute_difference": max(
            score_differences,
            default=0.0,
        ),
        "runtime": replay_runtime.evidence,
        "edit_visibility_tasks": dict(sorted(visibility_task_counts.items())),
        "complete_pair_edit_visibility": dict(
            sorted(complete_pair_visibility_counts.items())
        ),
        "incomplete_selection_tasks": incomplete_tasks,
        "crop_geometry_and_exact_gt_visibility_replayed": True,
    }


def _compare_summary_payload(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    label: str = "summary",
) -> None:
    """Require every recomputed metric key/value in the recorded summary."""

    for key, expected in recomputed.items():
        if key not in recorded:
            raise ValueError(f"{label} lacks recomputed field {key}")
        _compare_nested(
            recorded[key],
            expected,
            label=f"{label}.{key}",
            float_tolerance=1e-12,
        )


def recompute_summary(
    *,
    latest: Mapping[str, dict[str, Any]],
    result_rows: list[dict[str, Any]] | None = None,
    input_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    recorded_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[DetectionPair]]:
    physical_rows = (
        result_rows
        if result_rows is not None
        else [latest[str(row["sample_id"])] for row in input_rows]
    )
    metrics = _require_mapping(manifest.get("protocol"), "manifest protocol")
    classification = _require_mapping(
        metrics.get("classification"),
        "manifest classification protocol",
    )
    threshold = _require_finite(
        classification.get("ai_score_threshold"),
        "manifest classification threshold",
    )
    _require_equal(threshold, FIXED_THRESHOLD, "fixed FSD AI threshold")
    iterations = metrics.get(
        "bootstrap_samples",
    )
    seed = metrics.get("seed")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("manifest bootstrap sample count is not an integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest bootstrap seed is not an integer")
    recomputed = summarize_whole_image_results(
        physical_rows,
        input_rows,
        threshold=threshold,
        bootstrap_samples=iterations,
        seed=seed,
    )
    _compare_summary_payload(recorded_summary, recomputed)
    return recomputed, _load_pairs(latest, input_rows)


def _diagnostic_slices(
    pairs: list[DetectionPair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, visibility in enumerate(("none", "partial", "full")):
        selected = [
            {"real": pair.real, "forged": pair.forged}
            for pair in pairs
            if pair.forged.get("edit_visibility") == visibility
        ]
        result[visibility] = (
            summarize_whole_image_pair_slice(
                selected,
                iterations=iterations,
                seed=seed + offset,
            )
            if selected
            else {
                "pairs": 0,
                "images": 0,
                "status": "empty_slice",
                "bootstrap_samples": iterations,
                "seed": seed + offset,
            }
        )
    return result


def audit_prefix_reproducibility(
    *,
    repo_root: Path,
    full_run_id: str,
    full_manifest: Mapping[str, Any],
    full_rows: list[dict[str, Any]],
    prefix_run_id: str,
    prefix_manifest: Mapping[str, Any],
    prefix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove deterministic equality without accepting copied full-run rows."""

    if prefix_run_id == full_run_id:
        raise ValueError("prefix/full must have independent run IDs")
    _require_equal(
        prefix_manifest.get("run_id"),
        prefix_run_id,
        "prefix manifest run ID",
    )
    full_fingerprint = _require_sha256(
        full_manifest.get("fingerprint"),
        "full manifest fingerprint",
    )
    prefix_fingerprint = _require_sha256(
        prefix_manifest.get("fingerprint"),
        "prefix manifest fingerprint",
    )
    if prefix_fingerprint == full_fingerprint:
        raise ValueError("prefix/full must have independent fingerprints")
    _require_equal(
        prefix_fingerprint,
        _manifest_fingerprint(prefix_manifest),
        "prefix manifest fingerprint",
    )
    full_selection = _require_mapping(
        full_manifest.get("selection"),
        "full selection",
    )
    prefix_selection = _require_mapping(
        prefix_manifest.get("selection"),
        "prefix selection",
    )
    full_ordered = _require_list(full_selection.get("rows"), "full ordered inputs")
    prefix_ordered = _require_list(
        prefix_selection.get("rows"),
        "prefix ordered inputs",
    )
    if not prefix_ordered:
        raise ValueError("prefix selection is empty")
    if full_ordered[: len(prefix_ordered)] != prefix_ordered:
        raise ValueError("prefix selection is not an exact full-run prefix")

    full_model = _require_mapping(full_manifest.get("model"), "full model")
    prefix_model = _require_mapping(prefix_manifest.get("model"), "prefix model")
    for key in (
        "name",
        "slug",
        "repo_url",
        "source_commit",
        "release_tag",
        "source_audit",
        "weights",
    ):
        if key in full_model or key in prefix_model:
            _require_equal(
                prefix_model.get(key),
                full_model.get(key),
                f"prefix/full model {key}",
            )
    _require_equal(
        prefix_manifest.get("runtime_contract"),
        full_manifest.get("runtime_contract"),
        "prefix/full runtime contract",
    )
    _require_equal(
        prefix_manifest.get("protocol"),
        full_manifest.get("protocol"),
        "prefix/full protocol contract",
    )

    full_latest = _latest_by_id(full_rows)
    prefix_latest = _latest_by_id(prefix_rows)
    prefix_ids = [str(item["sample_id"]) for item in prefix_ordered]
    if set(prefix_latest) != set(prefix_ids):
        raise ValueError("prefix latest IDs do not equal prefix manifest")
    if any(sample_id not in full_latest for sample_id in prefix_ids):
        raise ValueError("full run is missing a prefix sample")

    compared = 0
    for sample_id in prefix_ids:
        prefix = prefix_latest[sample_id]
        full = full_latest[sample_id]
        _require_equal(prefix.get("status"), "ok", f"prefix {sample_id} status")
        _require_equal(full.get("status"), "ok", f"full {sample_id} status")
        # These checks specifically reject a byte-for-byte copied full row.
        _require_equal(
            prefix.get("run_id"),
            prefix_run_id,
            f"prefix {sample_id} own run ID",
        )
        _require_equal(
            prefix.get("run_manifest_fingerprint"),
            prefix_fingerprint,
            f"prefix {sample_id} own fingerprint",
        )
        _require_equal(
            full.get("run_id"),
            full_run_id,
            f"full {sample_id} own run ID",
        )
        _require_equal(
            full.get("run_manifest_fingerprint"),
            full_fingerprint,
            f"full {sample_id} own fingerprint",
        )
        prefix_path = prefix.get("raw_descriptor_path")
        full_path = full.get("raw_descriptor_path")
        if not isinstance(prefix_path, str) or not isinstance(full_path, str):
            raise ValueError(f"prefix/full {sample_id} has no descriptor path")
        if _anchored(Path(prefix_path), repo_root) == _anchored(
            Path(full_path), repo_root
        ):
            raise ValueError(
                f"prefix {sample_id} reuses full-run descriptor path"
            )
        prefix_descriptor = _load_descriptor(prefix, repo_root)
        full_descriptor = _load_descriptor(full, repo_root)
        _require_equal(
            prefix.get("raw_descriptor_sha256"),
            full.get("raw_descriptor_sha256"),
            f"prefix/full {sample_id} descriptor file SHA-256",
        )
        if not np.array_equal(prefix_descriptor, full_descriptor):
            raise ValueError(
                f"prefix/full {sample_id} descriptor values differ"
            )
        for field in _REPLAY_FIELDS:
            if field == "raw_descriptor_sha256":
                continue
            _require_equal(
                prefix.get(field),
                full.get(field),
                f"prefix/full {sample_id} {field}",
            )
        compared += 1
    return {
        "policy": (
            "independent run identities and artifact paths; exact ordered "
            "prefix; byte-identical descriptor values/hashes and scores"
        ),
        "prefix_run_id": prefix_run_id,
        "full_run_id": full_run_id,
        "prefix_manifest_fingerprint": prefix_fingerprint,
        "full_manifest_fingerprint": full_fingerprint,
        "prefix_images": len(prefix_ids),
        "prefix_pairs": len(prefix_ids) // 2,
        "samples_compared": compared,
        "copied_full_rows_rejected": True,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    fsd_root = Path(args.fsd_root).resolve()
    result_path = results_dir / f"{args.run_id}.jsonl"
    manifest_path = results_dir / f"{args.run_id}.run_manifest.json"
    summary_path = results_dir / f"{args.run_id}.summary.json"
    for path in (result_path, manifest_path, summary_path, inputs_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    all_inputs = read_jsonl(inputs_path)
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    input_rows = _select_manifest_inputs(all_inputs, manifest)
    result_rows = read_jsonl(result_path)
    history = summarize_result_history(result_rows)
    provenance = validate_provenance(
        repo_root=repo_root,
        fsd_root=fsd_root,
        run_id=args.run_id,
        inputs_path=inputs_path,
        input_rows=input_rows,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
    )
    latest = _latest_by_id(result_rows)
    artifacts = audit_artifacts(
        repo_root=repo_root,
        fsd_root=fsd_root,
        manifest=manifest,
        input_rows=input_rows,
        latest=latest,
    )
    recomputed, pairs = recompute_summary(
        latest=latest,
        result_rows=result_rows,
        input_rows=input_rows,
        manifest=manifest,
        recorded_summary=summary,
    )
    metrics_contract = _require_mapping(
        manifest.get("protocol"),
        "manifest protocol",
    )
    iterations = int(
        metrics_contract.get(
            "bootstrap_samples",
        )
    )
    seed = int(metrics_contract["seed"])
    slices = _diagnostic_slices(
        pairs,
        iterations=iterations,
        seed=seed,
    )

    prefix: dict[str, Any] | None = None
    if args.prefix_run_id is not None:
        prefix_dir = _anchored(Path(args.prefix_results_dir), repo_root)
        prefix_result_path = prefix_dir / f"{args.prefix_run_id}.jsonl"
        prefix_manifest_path = (
            prefix_dir / f"{args.prefix_run_id}.run_manifest.json"
        )
        prefix_summary_path = prefix_dir / f"{args.prefix_run_id}.summary.json"
        prefix_manifest = _require_mapping(
            json.loads(prefix_manifest_path.read_text(encoding="utf-8")),
            "prefix manifest",
        )
        prefix_input_rows = _select_manifest_inputs(all_inputs, prefix_manifest)
        prefix_rows = read_jsonl(prefix_result_path)
        prefix_summary = _require_mapping(
            json.loads(prefix_summary_path.read_text(encoding="utf-8")),
            "prefix summary",
        )
        prefix_provenance = validate_provenance(
            repo_root=repo_root,
            fsd_root=fsd_root,
            run_id=args.prefix_run_id,
            inputs_path=inputs_path,
            input_rows=prefix_input_rows,
            result_rows=prefix_rows,
            manifest=prefix_manifest,
            summary=prefix_summary,
        )
        prefix = {
            **audit_prefix_reproducibility(
                repo_root=repo_root,
                full_run_id=args.run_id,
                full_manifest=manifest,
                full_rows=result_rows,
                prefix_run_id=args.prefix_run_id,
                prefix_manifest=prefix_manifest,
                prefix_rows=prefix_rows,
            ),
            "prefix_provenance": prefix_provenance,
            "prefix_results_path": _relative_or_absolute(
                prefix_result_path,
                repo_root,
            ),
            "prefix_results_sha256": sha256_file(prefix_result_path),
            "prefix_manifest_path": _relative_or_absolute(
                prefix_manifest_path,
                repo_root,
            ),
            "prefix_manifest_sha256": sha256_file(prefix_manifest_path),
        }

    report = {
        "schema_version": "claimforge_fsd_analysis_v1",
        "created_at": utc_now(),
        "method": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "task_scope": {
            "T1": "native_official_whole_image_score",
            "T2": "N/A",
            "S_joint": "N/A",
            "crop_visibility": (
                "input_condition_diagnostic_only_not_model_localization"
            ),
        },
        "run_id": args.run_id,
        "condition": manifest.get("condition"),
        "history": history,
        "provenance": provenance,
        "artifact_and_score_replay": artifacts,
        "recomputed_whole_image_metrics": recomputed,
        "diagnostic_slices_by_edit_visibility": slices,
        "prefix_reproducibility": prefix,
        "evidence": {
            "results_path": _relative_or_absolute(result_path, repo_root),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": _relative_or_absolute(
                manifest_path,
                repo_root,
            ),
            "run_manifest_sha256": sha256_file(manifest_path),
            "runner_summary_path": _relative_or_absolute(
                summary_path,
                repo_root,
            ),
            "runner_summary_sha256": sha256_file(summary_path),
            "inputs_path": _relative_or_absolute(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
        },
    }
    output = (
        _anchored(Path(args.output), repo_root)
        if args.output is not None
        else results_dir / f"{args.run_id}.analysis.json"
    )
    atomic_write_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit an official FSD v1.2 run"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--fsd-root", type=Path, default=DEFAULT_FSD_ROOT)
    parser.add_argument("--prefix-run-id")
    parser.add_argument(
        "--prefix-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    report = analyze(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
