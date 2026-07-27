#!/usr/bin/env python3
"""Run the frozen official MVSS-Net CASIAv2 checkpoint on Balanced250.

The audited Mouse-v1 adapter remains unchanged.  This orchestration layer
binds its exact official source, checkpoint, BGR preprocessing, float32
forward, map-derived image score, legacy PNG postprocessing, and strict
threshold semantics to the independent Balanced250 release.

MVSS-Net has a native image-level score (global maximum pooling over its
segmentation probability map) and a native dense output.  T1 therefore covers
all 1,775 cache images.  T2 covers authentic images and the three local
insertion conditions only; full-frame edits retain diagnostic maps but are
explicitly localization-not-applicable.

Every source, asset, environment, adapter, and strict CPU model-load check
finishes before accelerator configuration.  The CPU gate performs no model
forward and computes no Balanced250 score.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_mvssnet as legacy
from eval.opensource.balanced_run_contract import (
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
    load_ground_truth,
    select_inputs,
)
from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.mvssnet_metrics import binary_pixel_metrics_strict


RUN_MANIFEST_SCHEMA = "mvssnet_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "mvssnet_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "mvssnet_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "mvssnet_balanced_cpu_preflight_v2"

MODEL_NAME = legacy.MODEL_NAME
MODEL_SLUG = legacy.MODEL_SLUG
MODEL_ARCHITECTURE = "original_MVSSNet_ResNet50_RGB_Bayar_Sobel_DAHead_one_class"
CHECKPOINT_ID = "mvssnet_casia"
PREPROCESS_PROFILE = "official_mvssnet_opencv_bgr_stretch512_imagenet_norm_in_bgr_order"

MODEL_SEED = 42
CLASSIFICATION_THRESHOLD = legacy.CLASSIFICATION_THRESHOLD
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD = legacy.MASK_THRESHOLD
MASK_THRESHOLD_OPERATOR = ">"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/mvssnet")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/mvssnet")
DEFAULT_FORMAL_RUN_ID = "mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727"
DEFAULT_SMOKE_RUN_ID_A = "mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_a_20260727"
DEFAULT_SMOKE_RUN_ID_B = "mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_b_20260727"
DEFAULT_SMOKE_LIMIT = 5

EXPECTED_VENV_ROOT = Path("/root/.cache/claimforge/venvs/mvssnet-cc2aed7")
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_BYTES = 206
EXPECTED_PYVENV_SHA256 = (
    "ea545e5a4f8e6a5a8b02e203cf72707d275a908f6dd2fb1d70bb0e8e5ba9849e"
)
EXPECTED_PACKAGES = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "opencv-python-headless": "4.12.0.88",
    "scikit-learn": "1.6.1",
    "scipy": "1.16.0",
}
EXPECTED_CV2_VERSION = "4.10.0"

CHECKPOINT_BYTES = legacy.CHECKPOINT_BYTES
CHECKPOINT_STATE_KEYS = legacy.CHECKPOINT_STATE_KEYS
CHECKPOINT_STATE_ELEMENTS = legacy.CHECKPOINT_STATE_ELEMENTS
CHECKPOINT_FLOAT32_TENSORS = 675
CHECKPOINT_INT64_TENSORS = 125
CHECKPOINT_ORDERED_KEYS_SHA256 = (
    "6f44de38d505a59fb9b0c2e548bd75ef822d00594c5e60d40ff629a371e3dcbf"
)
CHECKPOINT_TENSOR_SCHEMA_SHA256 = (
    "5755c949b4cc66709f8a2e6f6c6d9c19abba01b6e1e483c0b6647e2862560bbf"
)
MODEL_ORDERED_KEYS_SHA256 = (
    "b2605c6da0bd696cd3f41302f42dfdc821b84708d101122cba2dd148a52ff1ed"
)
CHECKPOINT_UNSAFE_GLOBALS: tuple[str, ...] = ()

EXPECTED_MODEL_PARAMETERS = legacy.MODEL_PARAMETER_COUNT
EXPECTED_TRAINABLE_PARAMETERS = 146_811_215
EXPECTED_MODEL_BUFFERS = legacy.MODEL_BUFFER_ELEMENTS
EXPECTED_MODEL_MODULES = 400

MODEL_ARRAY_FILE_BYTES = (
    legacy.MODEL_INPUT_SIZE * legacy.MODEL_INPUT_SIZE * np.dtype(np.float32).itemsize
    + 128
)
MIN_DISK_RESERVE_BYTES = 2_000_000_000
PNG_CONSERVATIVE_OVERHEAD_BYTES = 4_096
STATIC_CPU_SIGMOID_ABS_TOLERANCE = float(2 * np.finfo(np.float32).eps)

MVSSNET_SOURCE_FILES: dict[str, tuple[int, str]] = {
    ".gitignore": (
        2_394,
        "2a42d4a01a1d5b6a996422ea551da6d46b9c182558ef28dcd2ed2c4d1f5486ac",
    ),
    "README.md": (
        5_640,
        "25d78d642a52985dc7aaf666df4754cc8726c63a8de831e8711d66ecf60d07c6",
    ),
    "requirements.txt": (
        193,
        "6384e98cf751596671c4a9cb9c396000a0c4eede9baadc354e4b91d320e3860d",
    ),
    "models/mvssnet.py": (
        18_367,
        legacy.MODEL_NETWORK_SHA256,
    ),
    "inference.py": (
        3_227,
        legacy.MODEL_INFERENCE_SHA256,
    ),
    "evaluate.py": (
        3_078,
        legacy.MODEL_EVALUATE_SHA256,
    ),
    "common/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "common/tools.py": (
        1_163,
        legacy.MODEL_TOOLS_SHA256,
    ),
    "common/transforms.py": (
        324,
        legacy.MODEL_TRANSFORMS_SHA256,
    ),
    "common/utils.py": (
        8_858,
        "621e8fb8abc482c3911afc2f12b13cd16704ee0d0dbe8c96ff34857d451da266",
    ),
}

ADAPTER_SOURCE_PATHS = (
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_mvssnet_balanced.py",
    "eval/opensource/run_mvssnet.py",
    "eval/opensource/mvssnet_metrics.py",
    "eval/opensource/balanced250_localization_metrics.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/common.py",
)

MOUSE_REFERENCE_FILES: dict[str, tuple[int, str]] = {
    (
        "results/opensource/mvssnet/"
        "mvssnet_casia_mouse_canonical_v1_full275_20260723.run_manifest.json"
    ): (
        248_864,
        "5176adfef8b950bd42cc8be89dc89d996dcdd5471ec8c3f734b063fd8515b7cb",
    ),
    (
        "results/opensource/mvssnet/"
        "mvssnet_casia_mouse_canonical_v1_full275_20260723.jsonl"
    ): (
        2_313_263,
        "57872dc4030f5b93fc5bdf6eee6c1d2203ffde846275f95478de4a0f8a07831a",
    ),
    (
        "results/opensource/mvssnet/"
        "mvssnet_casia_mouse_canonical_v1_full275_20260723.summary.json"
    ): (
        15_381,
        "9f813029bcfbf6623d4f0bcbb46f1d8fb9e0e06829eade4db419a83a106d8d95",
    ),
    (
        "results/opensource/mvssnet/"
        "mvssnet_casia_mouse_canonical_v1_full275_20260723.analysis.json"
    ): (
        44_890,
        "c2db65790e895042a713a2c34e4491af05002cd46727a5ed71a39f203117abe1",
    ),
    "docs/MVSSNET_CASIA_MOUSE_FULL_RESULTS_2026-07-23.md": (
        18_205,
        "da5850283372315a7120798b7f224c51dd74a96eef188eb5bb869dad15aad390",
    ),
}

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}
FORMAL_T2_IMAGES = 1_025

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=CLASSIFICATION_THRESHOLD,
    threshold_operator=CLASSIFICATION_THRESHOLD_OPERATOR,
)

T2_SPEC: dict[str, Any] = {
    "valid_conditions": [
        "real",
        "local_mouse",
        "local_cat",
        "local_trash_can",
    ],
    "not_applicable_conditions": [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ],
    "model_probability_map": {
        "source": "sigmoid_one_channel_segmentation_logits",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "saved_for": "all_successful_inputs",
    },
    "native_probability_map": {
        "source": (
            "torchvision_0_6_1_ToPILImage_float_uint8_quantization_then_"
            "opencv_INTER_LINEAR_native_resize"
        ),
        "storage": "PNG_L_uint8",
        "reported_probability": "saved_uint8_divided_by_255",
        "saved_for": "all_successful_inputs",
        "fullframe_role": "diagnostic_and_secondary_T1_only",
    },
    "native_binary_mask": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "encoding": "PNG_L_0_or_255",
        "saved_for": "successful_T2_applicable_inputs_only",
    },
    "ground_truth": {
        "real": "all_zero_false_positive_area",
        "local": "exact_diff_local_insertion",
        "fullframe": "not_applicable",
        "fullframe_conditioning_box_is_not_ground_truth": True,
    },
}

TASK_SCOPE: dict[str, Any] = {
    "primary_task": "T1_image_detection_and_T2_localization",
    "valid_for_t1": True,
    "valid_for_t2": True,
    "fullframe_t2_not_applicable": True,
    "native_dense_output": True,
    "separate_image_classification_head": False,
    "native_image_score": "global_max_pooling_of_segmentation_probability",
}

LICENSE_RECORD: dict[str, Any] = {
    "source_repository_license_file_present": False,
    "source_repository_spdx_identifier": None,
    "checkpoint_separate_terms_present": False,
    "classification": "source_available_research_release_no_grant_found",
    "copyright_default": "all_rights_reserved_absent_an_explicit_grant",
    "commercial_use_permission_established": False,
    "redistribution_permission_established": False,
    "benchmark_use_does_not_establish_downstream_permission": True,
}

ARTIFACT_CONTRACT: dict[str, Any] = {
    "raw_logits_model_512": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "semantics": "official_one_channel_segmentation_logits",
        "inventory": "one_per_successful_input",
    },
    "score_map_model_512": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "official_sigmoid_segmentation_probability",
        "inventory": "one_per_successful_input",
    },
    "score_map_native_official": {
        "format": "PNG",
        "mode": "L",
        "dtype": "uint8",
        "shape": "native_height_by_native_width",
        "semantics": ("official_quantize_probability_to_uint8_before_native_resize"),
        "inventory": "one_per_successful_input",
    },
    "mask_native": {
        "format": "PNG",
        "mode": "L",
        "values": [0, 255],
        "shape": "native_height_by_native_width",
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "inventory": "one_per_successful_T2_applicable_input",
    },
}

ARTIFACT_DIRECTORIES = (
    "raw_logits_model_512",
    "score_maps_model_512",
    "score_maps_native_official",
    "masks_native",
)

_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_manifest_fingerprint",
        "dataset_id",
        "id",
        "sample_id",
        "rank",
        "condition",
        "condition_family",
        "manipulation_scope",
        "normalized_task_id",
        "task_id",
        "kind",
        "label",
        "domain",
        "gt_mask_kind",
        "input_path",
        "input_sha256",
        "input_width",
        "input_height",
        "valid_for_metrics",
        "valid_for_t1",
        "valid_for_t2",
        "model",
        "model_slug",
        "model_architecture",
        "preprocess_profile",
        "checkpoint_id",
        "checkpoint_sha256",
        "config_fingerprint",
        "task_scope",
        "t2_applicable",
        "t2_target_semantics",
    }
)

_OK_RESULT_KEYS = _IDENTITY_KEYS | frozenset(
    {
        "status",
        "completed_at",
        "preprocess",
        "raw_outputs",
        "ai_score",
        "probability",
        "score",
        "score_margin",
        "score_semantics",
        "calibrated_probability",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "official_png_score",
        "official_png_score_semantics",
        "official_png_decision",
        "official_png_threshold",
        "official_png_threshold_operator",
        "raw_logits_model_path",
        "raw_logits_model_sha256",
        "raw_logits_model_bytes",
        "raw_logits_model_array_sha256",
        "raw_logits_model_shape",
        "raw_logits_model_dtype",
        "raw_logits_model_semantics",
        "score_map_model_path",
        "score_map_model_sha256",
        "score_map_model_bytes",
        "score_map_model_array_sha256",
        "score_map_model_shape",
        "score_map_model_dtype",
        "score_map_model_semantics",
        "score_map_native_path",
        "score_map_native_sha256",
        "score_map_native_bytes",
        "score_map_native_array_sha256",
        "score_map_native_shape",
        "score_map_native_dtype",
        "score_map_native_mode",
        "score_map_native_semantics",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_array_sha256",
        "mask_shape",
        "mask_dtype",
        "mask_mode",
        "mask_semantics",
        "artifact_paths",
        "mask_threshold",
        "mask_threshold_operator",
        "localization",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)

_ERROR_RESULT_KEYS = _IDENTITY_KEYS | frozenset(
    {
        "status",
        "completed_at",
        "error_type",
        "error",
        "traceback",
    }
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _unresolved_anchored(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_child(root: Path, name: str, label: str) -> Path:
    candidate = root / name
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its configured root") from error
    if resolved == root.resolve():
        raise ValueError(f"{label} must be below its configured root")
    return resolved


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _valid_run_id(value: Any) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789_.-"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(character not in allowed for character in value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run-id must be one safe ASCII path component (max 160 chars)")
    return value


def _without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_object_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"{path}:{line_number} lacks terminating newline")
                if not line.strip():
                    raise ValueError(f"{path}:{line_number} is blank")
                value = json.loads(
                    line,
                    object_pairs_hook=_without_duplicate_keys,
                    parse_constant=_reject_nonfinite_json,
                )
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                if line != f"{stable_json(value)}\n":
                    raise ValueError(f"{path}:{line_number} is not canonical JSONL")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSONL: {path}") from error
    return rows


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local source that determines inference/result semantics."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        candidate = repo_root / relative
        _reject_symlink_components(
            candidate,
            f"MVSS-Net adapter source {relative}",
        )
        path = candidate.resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe MVSS-Net adapter source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_mouse_reference(repo_root: Path) -> dict[str, Any]:
    """Bind the prior Mouse evidence used to freeze this protocol."""

    files: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in MOUSE_REFERENCE_FILES.items():
        candidate = repo_root / relative
        _reject_symlink_components(
            candidate,
            f"MVSS-Net Mouse reference {relative}",
        )
        path = candidate.resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe MVSS-Net Mouse reference: {path}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"MVSS-Net Mouse reference byte size changed: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"MVSS-Net Mouse reference hash changed: {relative}")
        files[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    value = {
        "run_id": "mvssnet_casia_mouse_canonical_v1_full275_20260723",
        "expected_tasks": 275,
        "expected_images": 550,
        "role": ("protocol_and_regression_anchor_only_not_score_based_selection"),
        "files": files,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_environment() -> dict[str, Any]:
    """Fail unless the audited MVSS-Net virtual environment is active."""

    import cv2

    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    if executable != EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError(
            "MVSS-Net must run with the pinned interpreter "
            f"{EXPECTED_PYTHON_EXECUTABLE}, got {executable}"
        )
    if prefix != EXPECTED_VENV_ROOT:
        raise ValueError(
            f"MVSS-Net venv prefix changed: {prefix} != {EXPECTED_VENV_ROOT}"
        )
    if platform.python_version() != "3.12.3":
        raise ValueError("MVSS-Net Python version changed")
    pyvenv_path = prefix / "pyvenv.cfg"
    if (
        not pyvenv_path.is_file()
        or pyvenv_path.is_symlink()
        or pyvenv_path.stat().st_size != EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv_path) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("MVSS-Net pyvenv.cfg changed")
    versions = {name: _package_version(name) for name in EXPECTED_PACKAGES}
    if versions != EXPECTED_PACKAGES:
        changed = {
            name: {
                "expected": EXPECTED_PACKAGES[name],
                "actual": versions.get(name),
            }
            for name in EXPECTED_PACKAGES
            if versions.get(name) != EXPECTED_PACKAGES[name]
        }
        raise ValueError(f"MVSS-Net package environment changed: {changed}")
    if cv2.__version__ != EXPECTED_CV2_VERSION:
        raise ValueError(
            "MVSS-Net imported cv2 version changed: "
            f"{cv2.__version__} != {EXPECTED_CV2_VERSION}"
        )
    value = {
        "python_executable": str(executable),
        "python_prefix": str(prefix),
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv_path),
            "bytes": EXPECTED_PYVENV_BYTES,
            "sha256": EXPECTED_PYVENV_SHA256,
            "include_system_site_packages": True,
        },
        "packages": versions,
        "cv2_import": {
            "version": cv2.__version__,
            "path": str(Path(cv2.__file__).resolve()),
        },
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_source(mvssnet_root: Path) -> dict[str, Any]:
    """Verify the clean official commit and every source-bound file."""

    _reject_symlink_components(mvssnet_root, "MVSS-Net source root")
    root = mvssnet_root.resolve()
    if root.name != "MVSS-Net" or not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe MVSS-Net source root: {root}")
    commit = legacy._git_value(root, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"MVSS-Net source commit changed: {commit} != "
            f"{legacy.MODEL_SOURCE_COMMIT}"
        )
    status = legacy._git_value(
        root,
        "status",
        "--short",
        "--untracked-files=all",
    )
    if status is None:
        raise ValueError("cannot inspect MVSS-Net source worktree")
    if status:
        raise ValueError("MVSS-Net source worktree is dirty")

    bindings: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in MVSSNET_SOURCE_FILES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing/unsafe MVSS-Net source-bound file: {path}"
            )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"MVSS-Net source file size changed: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"MVSS-Net source file hash changed: {relative}")
        tracked = legacy._git_value(
            root,
            "ls-files",
            "--error-unmatch",
            relative,
        )
        if tracked != relative:
            raise ValueError(f"MVSS-Net source file is not tracked: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "git_tracked": True,
        }
    value = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bindings,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Verify the exact author-hosted CASIAv2 checkpoint bytes."""

    _reject_symlink_components(
        checkpoint_path,
        "MVSS-Net official checkpoint",
    )
    path = checkpoint_path.resolve()
    if path.name != "mvssnet_casia.pt" or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe MVSS-Net checkpoint: {path}")
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("MVSS-Net checkpoint byte size changed")
    if sha256_file(path) != legacy.CHECKPOINT_SHA256:
        raise ValueError("MVSS-Net checkpoint SHA-256 changed")
    value = {
        "path": str(path),
        "filename": path.name,
        "bytes": CHECKPOINT_BYTES,
        "sha256": legacy.CHECKPOINT_SHA256,
        "provider": "official_author_google_drive",
        "drive_file_id": legacy.CHECKPOINT_DRIVE_ID,
        "format": "raw_collections_OrderedDict_state_dict",
        "weights_only": True,
        "strict_model_load": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _checkpoint_tensor_schema(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in state.items()
    ]


def _audit_checkpoint_state(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    import torch

    unsafe = tuple(
        sorted(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path))
    )
    if unsafe != CHECKPOINT_UNSAFE_GLOBALS:
        raise ValueError(f"MVSS-Net checkpoint unsafe globals changed: {unsafe}")
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if type(state).__name__ != "OrderedDict":
        raise ValueError("MVSS-Net checkpoint is not an OrderedDict")
    if len(state) != CHECKPOINT_STATE_KEYS:
        raise ValueError("MVSS-Net checkpoint state-key count changed")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("MVSS-Net checkpoint has a non-string key")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("MVSS-Net checkpoint has a non-tensor value")

    dtype_counts = Counter(str(value.dtype) for value in state.values())
    elements = sum(int(value.numel()) for value in state.values())
    ordered_keys_sha256 = hashlib.sha256("\n".join(state).encode("utf-8")).hexdigest()
    tensor_schema = _checkpoint_tensor_schema(state)
    tensor_schema_sha256 = _fingerprint(tensor_schema)
    if dtype_counts != {
        "torch.float32": CHECKPOINT_FLOAT32_TENSORS,
        "torch.int64": CHECKPOINT_INT64_TENSORS,
    }:
        raise ValueError("MVSS-Net checkpoint dtype inventory changed")
    if elements != CHECKPOINT_STATE_ELEMENTS:
        raise ValueError("MVSS-Net checkpoint element count changed")
    if ordered_keys_sha256 != CHECKPOINT_ORDERED_KEYS_SHA256:
        raise ValueError("MVSS-Net checkpoint ordered-key hash changed")
    if tensor_schema_sha256 != CHECKPOINT_TENSOR_SCHEMA_SHA256:
        raise ValueError("MVSS-Net checkpoint tensor-schema hash changed")
    for name, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(
                "MVSS-Net checkpoint has non-finite tensor values: " f"{name}"
            )
    ordered_keys = tuple(state)
    value = {
        "outer_type": "collections.OrderedDict",
        "state_dict_tensors": CHECKPOINT_STATE_KEYS,
        "state_dict_elements": CHECKPOINT_STATE_ELEMENTS,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": ordered_keys_sha256,
        "tensor_schema_sha256": tensor_schema_sha256,
        "all_floating_tensors_finite": True,
        "static_unsafe_globals": list(unsafe),
        "weights_only": True,
        "map_location": "cpu",
    }
    del state
    gc.collect()
    return {**value, "contract_sha256": _fingerprint(value)}, ordered_keys


def _build_cpu_model_audit(
    *,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly construct/load on CPU without a model forward."""

    import torch

    checkpoint_audit, ordered_checkpoint_keys = _audit_checkpoint_state(checkpoint_path)
    torch.manual_seed(MODEL_SEED)
    model, device = legacy.load_model(
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )
    try:
        if str(device) != "cpu":
            raise ValueError("MVSS-Net CPU preflight loaded on non-CPU")
        model_keys = tuple(model.state_dict())
        model_key_hash = hashlib.sha256(
            "\n".join(model_keys).encode("utf-8")
        ).hexdigest()
        if (
            set(model_keys) != set(ordered_checkpoint_keys)
            or model_key_hash != MODEL_ORDERED_KEYS_SHA256
        ):
            raise ValueError("MVSS-Net model/checkpoint state keys changed")
        parameter_count = sum(int(value.numel()) for value in model.parameters())
        trainable_count = sum(
            int(value.numel()) for value in model.parameters() if value.requires_grad
        )
        buffer_count = sum(int(value.numel()) for value in model.buffers())
        module_count = sum(1 for _ in model.modules())
        if (
            parameter_count != EXPECTED_MODEL_PARAMETERS
            or trainable_count != EXPECTED_TRAINABLE_PARAMETERS
            or buffer_count != EXPECTED_MODEL_BUFFERS
            or module_count != EXPECTED_MODEL_MODULES
            or model.training
        ):
            raise ValueError("MVSS-Net constructed model inventory changed")
        value = {
            "construction_device": "cpu",
            "constructor": {
                "backbone": "resnet50",
                "pretrained_base": True,
                "nclass": 1,
                "sobel": True,
                "constrain": True,
                "n_input": 3,
                "external_ImageNet_downloads_suppressed": True,
                "complete_checkpoint_replaces_constructor_state": True,
            },
            "strict_state_dict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "state_key_set_matches_checkpoint": True,
            "model_ordered_keys_sha256": model_key_hash,
            "checkpoint_and_model_key_order_differ": True,
            "eval_mode": True,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "buffer_elements": buffer_count,
            "module_count": module_count,
            "forward_performed": False,
            "bayar_constraint_mutates_kernel_per_forward": True,
        }
        model_audit = {
            **value,
            "contract_sha256": _fingerprint(value),
        }
    finally:
        del model
        gc.collect()
    return checkpoint_audit, model_audit


def run_cpu_preflight(
    *,
    repo_root: Path,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Perform the complete no-forward, no-CUDA validation gate."""

    import torch

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise RuntimeError("CUDA was initialized before MVSS-Net CPU preflight")
    environment = verify_environment()
    source = verify_source(mvssnet_root)
    checkpoint = verify_checkpoint(checkpoint_path)
    adapter_sources = adapter_source_contract(repo_root)
    mouse_reference = verify_mouse_reference(repo_root)
    checkpoint_audit, model_audit = _build_cpu_model_audit(
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after:
        raise RuntimeError("MVSS-Net CPU preflight initialized CUDA")
    value = {
        "schema_version": CPU_PREFLIGHT_SCHEMA,
        "environment": environment,
        "source": source,
        "checkpoint": checkpoint,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "adapter_sources": adapter_sources,
        "adapter_sources_sha256": _fingerprint(adapter_sources),
        "mouse_reference": mouse_reference,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "balanced_scores_computed": False,
        "model_forward_performed": False,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    """Configure deterministic execution only after the CPU gate."""

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_CONFIG}, got {current!r}"
        )
    import torch

    device = torch.device(device_text)
    if device.type != "cuda":
        raise ValueError(
            "Balanced250 MVSS-Net inference requires an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.index is None:
        raise ValueError("CUDA device must include an explicit logical index")
    torch.cuda.set_device(device)
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(device)
    value = {
        "device": str(device),
        "device_type": "cuda",
        "gpu_name": properties.name,
        "gpu_compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "apex": False,
        "seed": MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    }
    return device, {**value, "contract_sha256": _fingerprint(value)}


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.LOCAL_T1_T2)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or len(selected) != 1_775
        or counts != FORMAL_COUNTS
        or sum(_t2_semantics(row)[0] for row in selected) != FORMAL_T2_IMAGES
    ):
        raise ValueError("MVSS-Net formal selection is not exact Balanced250 full1775")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if per_condition_limit != DEFAULT_SMOKE_LIMIT:
        raise ValueError("MVSS-Net smoke mode requires exactly 5 inputs per condition")
    spec = SelectionSpec(
        capability=Capability.LOCAL_T1_T2,
        conditions=BALANCED_CONDITIONS,
        per_condition_limit=DEFAULT_SMOKE_LIMIT,
    )
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        len(selected) != 35
        or counts
        != {condition: DEFAULT_SMOKE_LIMIT for condition in BALANCED_CONDITIONS}
        or any(row.get("panel") is not True for row in selected)
    ):
        raise ValueError("MVSS-Net smoke selection drifted")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode accepts no selection override")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode accepts no --sample-id")
        return _smoke_selection(release, per_condition_limit)
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode accepts no per-condition limit")
        if sample_id is None:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.LOCAL_T1_T2,
            sample_id=sample_id,
        )
        return spec, select_inputs(release, spec)
    raise ValueError(f"unsupported inference mode {mode!r}")


def _t2_semantics(row: Mapping[str, Any]) -> tuple[bool, str]:
    kind = row.get("gt_mask_kind")
    if kind == "all_zero":
        return True, "all_zero_real_false_positive_area"
    if kind == "exact_diff":
        return True, "exact_diff_local_insertion"
    if kind == "not_applicable":
        return False, "not_applicable_fullframe"
    raise ValueError("unsupported Balanced250 GT kind")


def result_task_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    applicable, _ = _t2_semantics(row)
    return {
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "native_dense_output": True,
        "model_512_output_role": (
            "t2_and_T1" if applicable else "T1_and_diagnostic_only"
        ),
        "native_map_output_role": (
            "t2_and_secondary_T1" if applicable else "secondary_T1_and_diagnostic_only"
        ),
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    applicable, semantics = _t2_semantics(row)
    return {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "model_architecture": MODEL_ARCHITECTURE,
        "preprocess_profile": PREPROCESS_PROFILE,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "config_fingerprint": run_manifest_fingerprint,
        "task_scope": result_task_scope(row),
        "t2_applicable": applicable,
        "t2_target_semantics": semantics,
    }


def _preprocess_with_audit(
    input_path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    tensor, native_size, audit = legacy.preprocess_image(input_path)
    if (
        tensor.shape != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("MVSS-Net preprocessed tensor is invalid")
    return tensor, native_size, audit


def _score_payload(
    raw_logits: np.ndarray,
    model_score_map: np.ndarray,
    native_uint8: np.ndarray,
) -> dict[str, Any]:
    logits = np.ascontiguousarray(raw_logits, dtype=np.float32)
    scores = np.ascontiguousarray(model_score_map, dtype=np.float32)
    native = np.ascontiguousarray(native_uint8, dtype=np.uint8)
    expected_shape = (
        legacy.MODEL_INPUT_SIZE,
        legacy.MODEL_INPUT_SIZE,
    )
    if (
        logits.shape != expected_shape
        or scores.shape != expected_shape
        or not np.isfinite(logits).all()
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
        or native.ndim != 2
        or native.size == 0
    ):
        raise ValueError("MVSS-Net raw score outputs are invalid")
    logits64 = logits.astype(np.float64)
    cpu_sigmoid64 = np.empty_like(logits64)
    nonnegative = logits64 >= 0.0
    cpu_sigmoid64[nonnegative] = 1.0 / (1.0 + np.exp(-logits64[nonnegative]))
    negative_exponential = np.exp(logits64[~nonnegative])
    cpu_sigmoid64[~nonnegative] = negative_exponential / (1.0 + negative_exponential)
    cpu_sigmoid = cpu_sigmoid64.astype(np.float32)
    max_abs = float(np.max(np.abs(cpu_sigmoid - scores)))
    if max_abs > STATIC_CPU_SIGMOID_ABS_TOLERANCE:
        raise ValueError(
            "MVSS-Net logit/probability static sanity check failed: " f"{max_abs}"
        )
    score = float(np.max(scores))
    official_png_score = float(np.max(native)) / 255.0
    return {
        "raw_outputs": {
            "segmentation_logits_shape": list(logits.shape),
            "auxiliary_edge_output": "discarded_as_in_official_inference",
            "static_cpu_sigmoid_max_abs_diff": max_abs,
            "static_cpu_sigmoid_abs_tolerance": (STATIC_CPU_SIGMOID_ABS_TOLERANCE),
        },
        "ai_score": score,
        "probability": score,
        "score": score,
        "score_margin": None,
        "score_semantics": ("continuous_global_max_of_model_512_sigmoid_probability"),
        "calibrated_probability": False,
        "classification_decision": score > CLASSIFICATION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (CLASSIFICATION_THRESHOLD_OPERATOR),
        "official_png_score": official_png_score,
        "official_png_score_semantics": (
            "maximum_of_saved_native_uint8_map_divided_by_255"
        ),
        "official_png_decision": (official_png_score > CLASSIFICATION_THRESHOLD),
        "official_png_threshold": CLASSIFICATION_THRESHOLD,
        "official_png_threshold_operator": (CLASSIFICATION_THRESHOLD_OPERATOR),
    }


def artifact_paths(
    artifact_root: Path,
    sample_id: str,
) -> dict[str, Path]:
    if (
        not isinstance(sample_id, str)
        or len(sample_id) != 24
        or any(character not in "0123456789abcdef" for character in sample_id)
    ):
        raise ValueError("MVSS-Net artifact sample_id is unsafe")
    return {
        "raw_logits": (artifact_root / "raw_logits_model_512" / f"{sample_id}.npy"),
        "model_score": (artifact_root / "score_maps_model_512" / f"{sample_id}.npy"),
        "native_score": (
            artifact_root / "score_maps_native_official" / f"{sample_id}.png"
        ),
        "mask": (artifact_root / "masks_native" / f"{sample_id}.png"),
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    raw_logits: np.ndarray,
    model_score_map: np.ndarray,
    native_uint8: np.ndarray,
    mask_path: Path | None,
) -> dict[str, Any]:
    logits_path = paths["raw_logits"]
    model_path = paths["model_score"]
    native_path = paths["native_score"]
    logits_relative = repo_relative(logits_path, repo_root)
    model_relative = repo_relative(model_path, repo_root)
    native_relative = repo_relative(native_path, repo_root)
    mask_relative = (
        repo_relative(mask_path, repo_root) if mask_path is not None else None
    )
    fields: dict[str, Any] = {
        "raw_logits_model_path": logits_relative,
        "raw_logits_model_sha256": sha256_file(logits_path),
        "raw_logits_model_bytes": logits_path.stat().st_size,
        "raw_logits_model_array_sha256": _array_sha256(raw_logits),
        "raw_logits_model_shape": list(raw_logits.shape),
        "raw_logits_model_dtype": str(raw_logits.dtype),
        "raw_logits_model_semantics": ("official_one_channel_segmentation_logits"),
        "score_map_model_path": model_relative,
        "score_map_model_sha256": sha256_file(model_path),
        "score_map_model_bytes": model_path.stat().st_size,
        "score_map_model_array_sha256": _array_sha256(model_score_map),
        "score_map_model_shape": list(model_score_map.shape),
        "score_map_model_dtype": str(model_score_map.dtype),
        "score_map_model_semantics": ("official_sigmoid_segmentation_probability"),
        "score_map_native_path": native_relative,
        "score_map_native_sha256": sha256_file(native_path),
        "score_map_native_bytes": native_path.stat().st_size,
        "score_map_native_array_sha256": _array_sha256(native_uint8),
        "score_map_native_shape": list(native_uint8.shape),
        "score_map_native_dtype": str(native_uint8.dtype),
        "score_map_native_mode": "L",
        "score_map_native_semantics": (
            "official_probability_times_255_uint8_truncate_then_"
            "opencv_INTER_LINEAR_native_resize"
        ),
        "mask_path": mask_relative,
        "mask_sha256": (sha256_file(mask_path) if mask_path is not None else None),
        "mask_bytes": (mask_path.stat().st_size if mask_path is not None else None),
        "mask_array_sha256": (
            _array_sha256(
                np.where(
                    native_uint8.astype(np.float32) / np.float32(255.0)
                    > MASK_THRESHOLD,
                    255,
                    0,
                ).astype(np.uint8)
            )
            if mask_path is not None
            else None
        ),
        "mask_shape": (list(native_uint8.shape) if mask_path is not None else None),
        "mask_dtype": "uint8" if mask_path is not None else None,
        "mask_mode": "L" if mask_path is not None else None,
        "mask_semantics": (
            "official_native_uint8_divide_255_strict_gt_0_5"
            if mask_path is not None
            else None
        ),
        "artifact_paths": {
            "raw_logits_model_512": logits_relative,
            "score_map_model_512": model_relative,
            "score_map_native_official": native_relative,
            "mask_native": mask_relative,
        },
    }
    return fields


def _localization_payload(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    model_score_map: np.ndarray,
    native_uint8: np.ndarray,
) -> dict[str, Any]:
    target_native = load_ground_truth(row, repo_root)
    if target_native is None:
        raise ValueError("T2-applicable MVSS-Net input has no GT")
    target_model = legacy.resize_target(
        target_native,
        legacy.MODEL_INPUT_SIZE,
        legacy.MODEL_INPUT_SIZE,
    )
    native_score_map = native_uint8.astype(np.float32) / np.float32(255.0)
    include_ap = row.get("gt_mask_kind") == "exact_diff"
    return {
        "model_512": binary_pixel_metrics_strict(
            model_score_map,
            target_model,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
        "native": binary_pixel_metrics_strict(
            native_score_map,
            target_native,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _resolve_exact_artifact(
    *,
    repo_root: Path,
    artifact_root: Path,
    relative: Any,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(relative, str):
        raise ValueError(f"{label} artifact path is invalid")
    path = _anchored(Path(relative), repo_root)
    if path != expected_path.resolve():
        raise ValueError(f"{label} artifact path drifted")
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} artifact escapes root") from error
    _reject_symlink_components(path, f"{label} artifact")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label} artifact: {path}")
    return path


def _load_model_array(
    *,
    path: Path,
    row: Mapping[str, Any],
    prefix: str,
) -> np.ndarray:
    if path.stat().st_size != MODEL_ARRAY_FILE_BYTES:
        raise ValueError(f"{prefix} .npy byte size changed")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {prefix} artifact") from error
    expected_shape = (
        legacy.MODEL_INPUT_SIZE,
        legacy.MODEL_INPUT_SIZE,
    )
    if (
        type(value) is not np.ndarray
        or value.shape != expected_shape
        or value.dtype != np.float32
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{prefix} array contract changed")
    if row.get(f"{prefix}_sha256") != sha256_file(path):
        raise ValueError(f"{prefix} file hash changed")
    if row.get(f"{prefix}_bytes") != path.stat().st_size:
        raise ValueError(f"{prefix} byte metadata changed")
    if row.get(f"{prefix}_array_sha256") != _array_sha256(value):
        raise ValueError(f"{prefix} array hash changed")
    if row.get(f"{prefix}_shape") != list(value.shape) or row.get(
        f"{prefix}_dtype"
    ) != str(value.dtype):
        raise ValueError(f"{prefix} array metadata changed")
    return value


def _load_png(
    *,
    path: Path,
    expected_shape: tuple[int, int],
    row: Mapping[str, Any],
    prefix: str,
    binary: bool,
) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            opened.load()
            if (
                opened.format != "PNG"
                or opened.mode != "L"
                or opened.size != (expected_shape[1], expected_shape[0])
            ):
                raise ValueError(f"{prefix} PNG metadata changed")
            value = np.asarray(opened, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load valid {prefix} PNG") from error
    if (
        value.shape != expected_shape
        or value.dtype != np.uint8
        or (binary and not np.isin(value, (0, 255)).all())
    ):
        raise ValueError(f"{prefix} PNG array contract changed")
    if row.get(f"{prefix}_sha256") != sha256_file(path):
        raise ValueError(f"{prefix} PNG file hash changed")
    if row.get(f"{prefix}_bytes") != path.stat().st_size:
        raise ValueError(f"{prefix} PNG byte metadata changed")
    if row.get(f"{prefix}_array_sha256") != _array_sha256(value):
        raise ValueError(f"{prefix} PNG array hash changed")
    if (
        row.get(f"{prefix}_shape") != list(value.shape)
        or row.get(f"{prefix}_dtype") != str(value.dtype)
        or row.get(f"{prefix}_mode") != "L"
    ):
        raise ValueError(f"{prefix} PNG array metadata changed")
    return value


def _validate_preprocess(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    recompute: bool,
) -> None:
    value = attempt.get("preprocess")
    if not isinstance(value, Mapping):
        raise ValueError("MVSS-Net result preprocess audit is missing")
    required = {
        "native_size",
        "model_size",
        "decoder",
        "channel_order",
        "resize",
        "normalization",
        "decoded_bgr_dtype",
        "decoded_bgr_shape",
        "decoded_bgr_sha256",
        "resized_bgr_dtype",
        "resized_bgr_shape",
        "resized_bgr_sha256",
        "normalized_chw_dtype",
        "normalized_chw_shape",
        "normalized_chw_sha256",
    }
    if set(value) != required:
        raise ValueError("MVSS-Net preprocess audit key set changed")
    if (
        value.get("native_size") != [int(input_row["width"]), int(input_row["height"])]
        or value.get("model_size") != [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE]
        or value.get("decoder") != "opencv_imread_color"
        or value.get("channel_order") != "BGR"
        or value.get("resize") != "opencv_inter_linear_stretch"
    ):
        raise ValueError("MVSS-Net preprocess audit semantics changed")
    if recompute:
        input_path = _anchored(
            Path(str(input_row["canonical_path"])),
            repo_root,
        )
        _, native_size, expected = _preprocess_with_audit(input_path)
        if (
            native_size
            != (
                int(input_row["width"]),
                int(input_row["height"]),
            )
            or dict(value) != expected
        ):
            raise ValueError("MVSS-Net preprocess replay changed")


def _validate_score_payload(
    attempt: Mapping[str, Any],
    sample_id: str,
) -> None:
    score = _finite_number(
        attempt.get("ai_score"),
        f"{sample_id} ai_score",
    )
    official = _finite_number(
        attempt.get("official_png_score"),
        f"{sample_id} official_png_score",
    )
    if not 0.0 <= score <= 1.0 or not 0.0 <= official <= 1.0:
        raise ValueError(f"{sample_id} score falls outside [0, 1]")
    if (
        attempt.get("probability") != score
        or attempt.get("score") != score
        or attempt.get("score_margin") is not None
        or attempt.get("calibrated_probability") is not False
        or attempt.get("classification_decision")
        is not (score > CLASSIFICATION_THRESHOLD)
        or attempt.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or attempt.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
        or attempt.get("official_png_decision")
        is not (official > CLASSIFICATION_THRESHOLD)
        or attempt.get("official_png_threshold") != CLASSIFICATION_THRESHOLD
        or attempt.get("official_png_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} score payload changed")


def _validate_ok_artifacts(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(artifact_root, sample_id)
    logits_path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        relative=attempt.get("raw_logits_model_path"),
        expected_path=paths["raw_logits"],
        label="raw logits",
    )
    model_path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        relative=attempt.get("score_map_model_path"),
        expected_path=paths["model_score"],
        label="model score map",
    )
    native_path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        relative=attempt.get("score_map_native_path"),
        expected_path=paths["native_score"],
        label="native score map",
    )
    logits = _load_model_array(
        path=logits_path,
        row=attempt,
        prefix="raw_logits_model",
    )
    model_score = _load_model_array(
        path=model_path,
        row=attempt,
        prefix="score_map_model",
    )
    if float(model_score.min()) < 0.0 or float(model_score.max()) > 1.0:
        raise ValueError("MVSS-Net model score map range changed")
    expected_shape = (
        int(input_row["height"]),
        int(input_row["width"]),
    )
    native = _load_png(
        path=native_path,
        expected_shape=expected_shape,
        row=attempt,
        prefix="score_map_native",
        binary=False,
    )
    replay_native = legacy.official_postprocess(
        model_score,
        expected_shape[1],
        expected_shape[0],
    )
    if not np.array_equal(native, replay_native):
        raise ValueError("MVSS-Net official native postprocess changed")
    replay_score = _score_payload(logits, model_score, native)
    for key, value in replay_score.items():
        if attempt.get(key) != value:
            raise ValueError(f"MVSS-Net replay score field changed: {key}")

    applicable, _ = _t2_semantics(input_row)
    if applicable:
        mask_path = _resolve_exact_artifact(
            repo_root=repo_root,
            artifact_root=artifact_root,
            relative=attempt.get("mask_path"),
            expected_path=paths["mask"],
            label="native mask",
        )
        mask = _load_png(
            path=mask_path,
            expected_shape=expected_shape,
            row=attempt,
            prefix="mask",
            binary=True,
        )
        expected_mask = np.where(
            native.astype(np.float32) / np.float32(255.0) > MASK_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)
        if not np.array_equal(mask, expected_mask):
            raise ValueError("MVSS-Net native mask thresholding changed")
        expected_localization = _localization_payload(
            row=input_row,
            repo_root=repo_root,
            model_score_map=model_score,
            native_uint8=native,
        )
        if attempt.get("localization") != expected_localization:
            raise ValueError("MVSS-Net localization artifact replay changed")
    else:
        for key in (
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_array_sha256",
            "mask_shape",
            "mask_dtype",
            "mask_mode",
            "mask_semantics",
            "localization",
        ):
            if attempt.get(key) is not None:
                raise ValueError(f"full-frame MVSS-Net result has non-N/A {key}")
        if paths["mask"].exists():
            raise ValueError("full-frame MVSS-Net result has a forbidden mask artifact")


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    verify_artifacts: bool,
    recompute_preprocess: bool = False,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("MVSS-Net result attempt status is invalid")
    expected_identity = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    for key, value in expected_identity.items():
        if attempt.get(key) != value:
            raise ValueError(f"MVSS-Net result identity drifted: {key}")
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("MVSS-Net result completed_at is invalid")
    if status == "error":
        if set(attempt) != _ERROR_RESULT_KEYS:
            raise ValueError("MVSS-Net error result key set changed")
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt.get("error_type")
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt.get("traceback")
        ):
            raise ValueError("MVSS-Net error payload is invalid")
        return
    if set(attempt) != _OK_RESULT_KEYS:
        raise ValueError("MVSS-Net successful result key set changed")
    sample_id = str(input_row["sample_id"])
    _validate_score_payload(attempt, sample_id)
    _validate_preprocess(
        attempt,
        input_row=input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    if (
        attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError("MVSS-Net mask threshold changed")
    latency = _finite_number(
        attempt.get("latency_ms"),
        f"{sample_id} latency_ms",
    )
    if latency < 0.0:
        raise ValueError("MVSS-Net latency is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("MVSS-Net peak memory is invalid")
    if verify_artifacts:
        _validate_ok_artifacts(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )


def _validate_physical_attempt_history(
    selected: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require stateful attempts to follow one exact selected prefix."""

    expected_ids = [str(row["sample_id"]) for row in selected]
    selected_index = 0
    errors = 0
    recovered = 0
    pending_had_error = False
    for line_number, attempt in enumerate(attempts, start=1):
        if selected_index >= len(expected_ids):
            raise ValueError("MVSS-Net history appends after full success")
        sample_id = attempt.get("sample_id")
        expected = expected_ids[selected_index]
        if sample_id != expected:
            raise ValueError(
                "MVSS-Net stateful result history is out of selected order "
                f"at row {line_number}: {sample_id!r} != {expected!r}"
            )
        status = attempt.get("status")
        if status == "error":
            errors += 1
            pending_had_error = True
        elif status == "ok":
            if pending_had_error:
                recovered += 1
            pending_had_error = False
            selected_index += 1
        else:
            raise ValueError(f"MVSS-Net history row {line_number} has invalid status")
    return {
        "policy": ("exact_selected_prefix_with_zero_or_more_errors_before_each_ok"),
        "physical_attempts": len(attempts),
        "successful_prefix": selected_index,
        "errors": errors,
        "recovered_error_to_ok": recovered,
    }


def _prepare_artifact_root(artifact_root: Path) -> None:
    expected = set(ARTIFACT_DIRECTORIES)
    if artifact_root.exists():
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise ValueError(f"MVSS-Net artifact root is unsafe: {artifact_root}")
        for entry in artifact_root.iterdir():
            if entry.name not in expected or not entry.is_dir() or entry.is_symlink():
                raise ValueError(
                    "MVSS-Net artifact root has unexpected/unsafe entry: " f"{entry}"
                )
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_DIRECTORIES:
        (artifact_root / name).mkdir(exist_ok=True)


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    expected_directories = set(ARTIFACT_DIRECTORIES)
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe MVSS-Net artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != expected_directories or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("MVSS-Net artifact root inventory mismatch")
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    successful = {
        sample_id
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    expected: dict[str, set[str]] = {
        "raw_logits_model_512": {f"{sample_id}.npy" for sample_id in successful},
        "score_maps_model_512": {f"{sample_id}.npy" for sample_id in successful},
        "score_maps_native_official": {f"{sample_id}.png" for sample_id in successful},
        "masks_native": {
            f"{sample_id}.png"
            for sample_id in successful
            if _t2_semantics(inputs_by_id[sample_id])[0]
        },
    }
    counts: dict[str, int] = {}
    for directory_name, expected_names in expected.items():
        directory = artifact_root / directory_name
        children = list(directory.iterdir())
        if any(child.is_symlink() or not child.is_file() for child in children):
            raise ValueError(f"MVSS-Net {directory_name} has unsafe/non-file entries")
        actual_names = {child.name for child in children}
        if actual_names != expected_names:
            raise ValueError(
                f"MVSS-Net {directory_name} inventory mismatch: "
                f"missing={sorted(expected_names - actual_names)[:1]}, "
                f"extra={sorted(actual_names - expected_names)[:1]}"
            )
        counts[directory_name] = len(actual_names)
    return counts


def _required_artifact_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Conservative uncompressed upper bound plus a fixed reserve."""

    if not rows:
        return 0
    model_arrays = len(rows) * 2 * MODEL_ARRAY_FILE_BYTES
    native_png_upper_bound = sum(
        int(row["width"]) * int(row["height"]) + PNG_CONSERVATIVE_OVERHEAD_BYTES
        for row in rows
    )
    mask_png_upper_bound = sum(
        int(row["width"]) * int(row["height"]) + PNG_CONSERVATIVE_OVERHEAD_BYTES
        for row in rows
        if _t2_semantics(row)[0]
    )
    return (
        model_arrays
        + native_png_upper_bound
        + mask_png_upper_bound
        + MIN_DISK_RESERVE_BYTES
    )


def _verify_disk_capacity(
    artifact_root: Path,
    pending: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    required = _required_artifact_bytes(pending)
    usage = shutil.disk_usage(artifact_root.parent)
    if usage.free < required:
        raise OSError(
            "insufficient disk for MVSS-Net raw artifacts: "
            f"required={required}, free={usage.free}"
        )
    return {
        "free_bytes_before_inference": int(usage.free),
        "conservative_pending_bytes_plus_reserve": int(required),
        "fixed_reserve_bytes": MIN_DISK_RESERVE_BYTES,
    }


def _validate_run_directory_safety(
    run_dir: Path,
    *,
    resume: bool,
) -> None:
    if not run_dir.exists():
        if resume:
            raise FileNotFoundError(f"resume run directory is missing: {run_dir}")
        return
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"MVSS-Net run directory is unsafe: {run_dir}")
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
    }
    entries = list(run_dir.iterdir())
    unexpected = {entry.name for entry in entries if entry.name not in allowed}
    if unexpected:
        raise ValueError(
            "MVSS-Net run directory has unexpected entries: "
            f"{sorted(unexpected)[:1]}"
        )
    for entry in entries:
        _reject_symlink_components(
            entry,
            f"MVSS-Net run file {entry.name}",
        )
        if not entry.is_file() or entry.is_symlink():
            raise ValueError(f"MVSS-Net run entry is not a regular file: {entry}")
    if resume:
        names = {entry.name for entry in entries}
        if not {
            "manifest.json",
            "expected_inputs.jsonl",
        }.issubset(names):
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    cpu_preflight: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(cpu_preflight["adapter_sources"]),
        "adapter_sources_sha256": cpu_preflight["adapter_sources_sha256"],
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repository": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "training_dataset": "CASIAv2",
            "variant": "original_MVSS-Net_not_MVSS-Net++",
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "decode": "opencv_imread_color",
            "channel_order": "BGR",
            "resize": "opencv_INTER_LINEAR_stretch_512x512",
            "scale": "uint8_divide_255",
            "normalization_mean_in_BGR_order": (legacy.NORMALIZE_MEAN.tolist()),
            "normalization_std_in_BGR_order": (legacy.NORMALIZE_STD.tolist()),
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "batch_size": 1,
            "autocast": False,
            "apex": False,
        },
        "inference": {
            "raw_output": "one_channel_segmentation_logits_512",
            "probability_map": "sigmoid_segmentation_logits",
            "auxiliary_edge_output": "discarded_as_official_inference",
            "primary_T1": ("continuous_global_max_of_model_512_probability"),
            "secondary_T1": ("official_native_saved_uint8_PNG_global_max_divide_255"),
            "native_restore": (
                "probability_times_255_uint8_truncate_before_"
                "opencv_INTER_LINEAR_native_resize"
            ),
            "bayar_state": ("constraint_kernel_normalized_in_place_each_forward"),
            "resume": ("fresh_checkpoint_and_replay_of_every_successful_prefix_input"),
        },
        "score_spec": SCORE_SPEC.as_dict(),
        "t2_spec": T2_SPEC,
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "source": dict(cpu_preflight["source"]),
        "checkpoint": dict(cpu_preflight["checkpoint"]),
        "environment": dict(cpu_preflight["environment"]),
        "checkpoint_audit": dict(cpu_preflight["checkpoint_audit"]),
        "model_audit": dict(cpu_preflight["model_audit"]),
        "mouse_reference": dict(cpu_preflight["mouse_reference"]),
        "license": LICENSE_RECORD,
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_root": repo_relative(artifact_root, repo_root),
            **{
                f"{name}_dir": repo_relative(
                    artifact_root / name,
                    repo_root,
                )
                for name in ARTIFACT_DIRECTORIES
            },
        },
    }


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.mode == "formal":
        run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
        if run_id != DEFAULT_FORMAL_RUN_ID:
            raise ValueError(f"formal run-id must be exactly {DEFAULT_FORMAL_RUN_ID}")
        return run_id
    if args.mode == "smoke":
        if args.run_id is None:
            raise ValueError("smoke mode requires explicit A/B --run-id")
        run_id = _valid_run_id(args.run_id)
        if run_id not in {
            DEFAULT_SMOKE_RUN_ID_A,
            DEFAULT_SMOKE_RUN_ID_B,
        }:
            raise ValueError("smoke run-id must be the frozen A or B ID")
        return run_id
    if args.mode == "single":
        if args.run_id is None:
            raise ValueError("single mode requires explicit --run-id")
        return _valid_run_id(args.run_id)
    raise ValueError(f"mode {args.mode!r} has no run ID")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--mvssnet-root",
        type=Path,
        default=legacy.DEFAULT_MVSSNET_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=legacy.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke", "single", "preflight"),
        default="formal",
    )
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--device",
        help="explicit cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _prior_manifest_started_at(
    *,
    repo_root: Path,
    manifest_path: Path,
    expected_immutable: Mapping[str, Any],
    expected_fingerprint: str,
    expected_inputs_path: Path,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    if not manifest_path.is_file() or not expected_inputs_path.is_file():
        raise FileNotFoundError(
            "resume requires manifest.json and expected_inputs.jsonl"
        )
    prior = _load_json_object_strict(manifest_path)
    if (
        prior.get("schema_version") != RUN_MANIFEST_SCHEMA
        or prior.get("run_id") != expected_immutable["run_id"]
        or prior.get("fingerprint") != expected_fingerprint
        or prior.get("immutable") != dict(expected_immutable)
        or prior.get("status") not in ("running", "incomplete", "complete")
    ):
        raise ValueError("resume MVSS-Net manifest/config drifted")
    status = str(prior["status"])
    base_keys = {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "completed_at",
        "fingerprint",
        "immutable",
        "dataset",
        "outputs",
        "disk_preflight",
    }
    expected_keys = base_keys if status == "running" else base_keys | {"execution"}
    if set(prior) != expected_keys:
        raise ValueError("resume MVSS-Net manifest key set changed")
    if _read_jsonl_strict(expected_inputs_path) != list(selected):
        raise ValueError("resume MVSS-Net expected inputs drifted")
    started_at = prior.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume MVSS-Net started_at is invalid")
    completed_at = prior.get("completed_at")
    if status == "running":
        if completed_at is not None:
            raise ValueError("resume running MVSS-Net manifest has completed_at")
    elif not isinstance(completed_at, str) or not completed_at:
        raise ValueError("resume finalized MVSS-Net completed_at is invalid")
    disk = prior.get("disk_preflight")
    if (
        not isinstance(disk, Mapping)
        or set(disk)
        != {
            "free_bytes_before_inference",
            "conservative_pending_bytes_plus_reserve",
            "fixed_reserve_bytes",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in disk.values()
        )
        or disk.get("fixed_reserve_bytes") != MIN_DISK_RESERVE_BYTES
    ):
        raise ValueError("resume MVSS-Net disk preflight changed")
    outputs = prior.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resume MVSS-Net outputs are missing")
    base_output_keys = set(expected_immutable["outputs"])
    expected_output_keys = (
        base_output_keys
        if status == "running"
        else base_output_keys
        | {
            "results_sha256",
            "summary_sha256",
            "artifact_inventory",
        }
    )
    if set(outputs) != expected_output_keys:
        raise ValueError("resume MVSS-Net output key set changed")
    for key, value in expected_immutable["outputs"].items():
        if outputs.get(key) != value:
            raise ValueError(f"resume MVSS-Net output {key} changed")
    if status in ("incomplete", "complete"):
        for relative_key, hash_key in (
            ("results_path", "results_sha256"),
            ("summary_path", "summary_sha256"),
        ):
            relative = expected_immutable["outputs"][relative_key]
            path = _anchored(Path(relative), repo_root)
            digest = outputs.get(hash_key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != digest
            ):
                raise ValueError(f"resume finalized {relative_key} hash changed")
        inventory = outputs.get("artifact_inventory")
        if (
            not isinstance(inventory, Mapping)
            or set(inventory) != set(ARTIFACT_DIRECTORIES)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in inventory.values()
            )
        ):
            raise ValueError("resume MVSS-Net artifact inventory changed")
        execution = prior.get("execution")
        if (
            not isinstance(execution, Mapping)
            or set(execution)
            != {
                "new_successes",
                "resume_skips",
                "new_errors",
                "physical_result_rows",
                "latest_result_rows",
                "superseded_attempts",
                "stateful_prefix_replayed",
            }
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in execution.values()
            )
        ):
            raise ValueError("resume MVSS-Net execution record changed")
    return prior, started_at


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    mvssnet_root = _unresolved_anchored(
        args.mvssnet_root,
        repo_root,
    )
    checkpoint_path = _unresolved_anchored(
        args.checkpoint,
        repo_root,
    )

    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    if args.mode == "preflight":
        if (
            args.resume
            or args.run_id is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError("preflight accepts no run/selection/resume/CUDA options")
        report = run_cpu_preflight(
            repo_root=repo_root,
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint_path,
        )
        print(
            json.dumps(
                {
                    **report,
                    "dataset": {
                        "dataset_id": release.dataset_id,
                        "manifest_sha256": release.manifest_sha256,
                        "verified_images": len(release.inputs),
                        "verify_files": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    run_id = _resolve_run_id(args)
    selection_spec, selected = select_mode_inputs(
        release,
        mode=args.mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    )

    requested_results = _unresolved_anchored(
        args.results_dir,
        repo_root,
    )
    expected_results = repo_root / DEFAULT_RESULTS_DIR
    _reject_symlink_components(
        requested_results,
        "MVSS-Net results root",
    )
    _reject_symlink_components(
        expected_results,
        "expected MVSS-Net results root",
    )
    results_root = requested_results.resolve()
    if results_root != expected_results.resolve():
        raise ValueError(f"--results-dir must be exactly {DEFAULT_RESULTS_DIR}")
    requested_artifacts = _unresolved_anchored(
        args.artifacts_dir,
        repo_root,
    )
    expected_artifacts = repo_root / DEFAULT_ARTIFACTS_DIR
    _reject_symlink_components(
        requested_artifacts,
        "MVSS-Net artifacts root",
    )
    _reject_symlink_components(
        expected_artifacts,
        "expected MVSS-Net artifacts root",
    )
    artifacts_root = requested_artifacts.resolve()
    if artifacts_root != expected_artifacts.resolve():
        raise ValueError(f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}")
    run_dir = _safe_child(
        results_root,
        run_id,
        "MVSS-Net run directory",
    )
    artifact_root = _safe_child(
        artifacts_root,
        run_id,
        "MVSS-Net artifact directory",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("MVSS-Net result and artifact directories must be disjoint")
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    _validate_run_directory_safety(run_dir, resume=args.resume)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory is non-empty; pass --resume: {run_dir}")
    if artifact_root.exists() and any(artifact_root.iterdir()) and not args.resume:
        raise FileExistsError(
            "artifact directory is non-empty; pass --resume: " f"{artifact_root}"
        )

    # This full strict model-load gate intentionally precedes every CUDA API
    # that configures or initializes an accelerator.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    device, runtime = configure_runtime(args.device or "cuda:0")
    import torch

    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=args.mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        cpu_preflight=cpu_preflight,
        runtime=runtime,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    fingerprint = _fingerprint(immutable)
    if args.resume:
        prior_manifest, started_at = _prior_manifest_started_at(
            repo_root=repo_root,
            manifest_path=manifest_path,
            expected_immutable=immutable,
            expected_fingerprint=fingerprint,
            expected_inputs_path=expected_path,
            selected=selected,
        )
    else:
        prior_manifest = None
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(expected_path, selected)
        started_at = utc_now()

    dataset_record = {
        "contract": dataset_contract.as_dict(),
        "manifest_path": repo_relative(
            dataset_manifest_path,
            repo_root,
        ),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t2_applicable_images": sum(_t2_semantics(row)[0] for row in selected),
    }
    if prior_manifest is not None and prior_manifest.get("dataset") != dataset_record:
        raise ValueError("resume MVSS-Net dataset envelope changed")
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "fingerprint": fingerprint,
        "immutable": immutable,
        "dataset": dataset_record,
        "outputs": dict(immutable["outputs"]),
    }

    physical_before = _read_jsonl_strict(results_path) if results_path.is_file() else []
    history_before = _validate_physical_attempt_history(
        selected,
        physical_before,
    )
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_before:
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[str(attempt["sample_id"])],
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=False,
        )
    completed_prefix = int(history_before["successful_prefix"])
    for row in selected[:completed_prefix]:
        prior = latest_before.latest_by_sample_id[str(row["sample_id"])]
        if prior.get("status") != "ok":
            raise ValueError("MVSS-Net stateful success prefix/latest view disagree")
        _validate_runner_attempt(
            prior,
            input_row=row,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=True,
            recompute_preprocess=True,
        )
    pending = list(selected[completed_prefix:])
    if prior_manifest is None:
        _prepare_artifact_root(artifact_root)
    validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest_before.latest_by_sample_id,
    )
    disk_audit = _verify_disk_capacity(artifact_root, pending)
    manifest["disk_preflight"] = disk_audit
    atomic_write_json(manifest_path, manifest)

    model = None
    new_successes = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            model, loaded_device = legacy.load_model(
                mvssnet_root=mvssnet_root,
                checkpoint_path=checkpoint_path,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("MVSS-Net loaded on an unexpected device")
            if completed_prefix:
                print(
                    "replaying "
                    f"{completed_prefix} successful prefix inputs to restore "
                    "the official stateful Bayar constraint",
                    flush=True,
                )
                for replay_row in selected[:completed_prefix]:
                    replay_path = _anchored(
                        Path(str(replay_row["canonical_path"])),
                        repo_root,
                    )
                    replay_tensor, _, _ = _preprocess_with_audit(replay_path)
                    legacy.infer_one(model, device, replay_tensor)
        for index, input_row in enumerate(pending, start=1):
            sample_id = str(input_row["sample_id"])
            paths = artifact_paths(artifact_root, sample_id)
            expected_ok = result_identity(
                input_row,
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
                valid_for_metrics=True,
            )
            try:
                input_path = _anchored(
                    Path(str(input_row["canonical_path"])),
                    repo_root,
                )
                tensor, (width, height), preprocess = _preprocess_with_audit(input_path)
                if (width, height) != (
                    int(input_row["width"]),
                    int(input_row["height"]),
                ):
                    raise ValueError("canonical MVSS-Net image dimensions changed")
                assert model is not None
                (
                    raw_logits,
                    model_score_map,
                    peak_bytes,
                    latency_ms,
                ) = legacy.infer_one(model, device, tensor)
                raw_logits = np.ascontiguousarray(
                    raw_logits,
                    dtype=np.float32,
                )
                model_score_map = np.ascontiguousarray(
                    model_score_map,
                    dtype=np.float32,
                )
                native_uint8 = legacy.official_postprocess(
                    model_score_map,
                    width,
                    height,
                )
                legacy._atomic_save_npy(
                    paths["raw_logits"],
                    raw_logits,
                )
                legacy._atomic_save_npy(
                    paths["model_score"],
                    model_score_map,
                )
                legacy._atomic_save_gray_png(
                    paths["native_score"],
                    native_uint8,
                )
                applicable, _ = _t2_semantics(input_row)
                mask_path: Path | None = None
                localization: dict[str, Any] | None = None
                if applicable:
                    mask_path = paths["mask"]
                    native_mask = np.where(
                        native_uint8.astype(np.float32) / np.float32(255.0)
                        > MASK_THRESHOLD,
                        255,
                        0,
                    ).astype(np.uint8)
                    legacy._atomic_save_gray_png(mask_path, native_mask)
                    localization = _localization_payload(
                        row=input_row,
                        repo_root=repo_root,
                        model_score_map=model_score_map,
                        native_uint8=native_uint8,
                    )
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **_score_payload(
                        raw_logits,
                        model_score_map,
                        native_uint8,
                    ),
                    **_artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        raw_logits=raw_logits,
                        model_score_map=model_score_map,
                        native_uint8=native_uint8,
                        mask_path=mask_path,
                    ),
                    "mask_threshold": MASK_THRESHOLD,
                    "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
                    "localization": localization,
                    "latency_ms": float(latency_ms),
                    "peak_cuda_memory_bytes": int(peak_bytes),
                }
                _validate_runner_attempt(
                    result,
                    input_row=input_row,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    run_manifest_fingerprint=fingerprint,
                    verify_artifacts=True,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(pending)}] ok {sample_id} "
                    f"score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                for path in paths.values():
                    path.unlink(missing_ok=True)
                new_errors += 1
                error_result = {
                    **result_identity(
                        input_row,
                        run_id=run_id,
                        run_manifest_fingerprint=fingerprint,
                        valid_for_metrics=False,
                    ),
                    "status": "error",
                    "completed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(results_path, error_result)
                print(
                    f"[{index}/{len(pending)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                fatal_error = error
                break
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    physical_results = (
        _read_jsonl_strict(results_path) if results_path.is_file() else []
    )
    history = _validate_physical_attempt_history(
        selected,
        physical_results,
    )
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    for attempt in physical_results:
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[str(attempt["sample_id"])],
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=False,
        )
    coverage = summarize_coverage(latest)
    inventory = validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_mvssnet_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": args.mode,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "score_spec": SCORE_SPEC.as_dict(),
        "t2_spec": T2_SPEC,
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": coverage.as_dict(),
        "stateful_history": history,
        "artifact_inventory": inventory,
        "generated_at": utc_now(),
    }
    atomic_write_json(summary_path, summary)
    manifest["status"] = summary["status"]
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": new_successes,
        "resume_skips": completed_prefix,
        "new_errors": new_errors,
        "physical_result_rows": len(physical_results),
        "latest_result_rows": len(latest.latest_by_sample_id),
        "superseded_attempts": latest.superseded_attempts,
        "stateful_prefix_replayed": (completed_prefix if pending else 0),
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_inventory": inventory,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "mode": args.mode,
                "coverage": coverage.as_dict(),
                "stateful_history": history,
                "artifact_inventory": inventory,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError("MVSS-Net fail-closed inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
