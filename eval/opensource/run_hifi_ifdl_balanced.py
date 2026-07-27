#!/usr/bin/env python3
"""Run the frozen official HiFi-IFDL general checkpoint on Balanced250.

The audited Mouse-v1 runner remains immutable.  This v2 adapter reuses only
its frozen inference primitives and binds them to the independent Balanced250
release.

HiFi-IFDL natively supports both benchmark tasks.  Its fine 14-class head
provides T1 for all 1,775 cache images.  Its hypersphere-distance map provides
T2 only for real images and the three local-insertion conditions (1,025
images).  Full-frame conditional edits are explicitly T2-not-applicable; the
transient dense output is discarded and is never scored against a placement
box or an invented target.

Every dataset, source, asset, environment, adapter, and strict CPU model-load
gate completes before accelerator configuration.  The CPU preflight performs
no model forward and computes no Balanced250 score.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource import run_hifi_ifdl as legacy
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
from eval.opensource.hifi_ifdl_metrics import binary_distance_metrics_strict


RUN_MANIFEST_SCHEMA = "hifi_ifdl_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "hifi_ifdl_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "hifi_ifdl_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "hifi_ifdl_balanced_cpu_preflight_v2"

MODEL_NAME = legacy.MODEL_NAME
MODEL_SLUG = legacy.MODEL_SLUG
MODEL_ARCHITECTURE = (
    "HRNet_W18_small_v2_plus_nonlocal_partial_convolution_localizer_"
    "and_hierarchical_3_5_7_14_class_heads"
)
MODEL_GIT_ORIGIN = "https://github.com/CHELSEA234/HiFi_IFDL.git"
MODEL_TREE = "f5dbb144329a048c2c03f102d441c3ef37a89a3f"
CHECKPOINT_ID = "official_general_release_identifier_750001"
PREPROCESS_PROFILE = "official_imageio_rgb_stretch256_bicubic_float32_divide255"

MODEL_SEED = 42
CLASSIFICATION_THRESHOLD = legacy.CLASSIFICATION_THRESHOLD
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD = legacy.MASK_THRESHOLD
MASK_THRESHOLD_OPERATOR = ">="
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/hifi_ifdl")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/hifi_ifdl")
DEFAULT_FORMAL_RUN_ID = "hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727"
DEFAULT_SMOKE_RUN_ID_A = "hifi_ifdl_general750001_balanced250_v1_smoke5x7_a_r2_20260727"
DEFAULT_SMOKE_RUN_ID_B = "hifi_ifdl_general750001_balanced250_v1_smoke5x7_b_r2_20260727"
DEFAULT_SMOKE_LIMIT = 5

EXPECTED_VENV_ROOT = Path("/root/.cache/claimforge/venvs/hifi-ifdl-0ca70d6")
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_BYTES = 207
EXPECTED_PYVENV_SHA256 = (
    "d39bed10d9ddf2b3da444ff8a53eb83adb6b4d93a8b7f169224eee65cea0d8bd"
)
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/hifi-ifdl-balanced-v2-empty"
)
EXPECTED_PACKAGES = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "imageio": "2.37.0",
    "yacs": "0.1.8",
    "scikit-learn": "1.5.2",
    "scipy": "1.16.0",
}

SOURCE_BOUND_FILES: dict[str, tuple[int, str]] = {
    "README.md": (
        7_265,
        "62fa6fa400f72797cb93c9104fe3781064f04c95fc3b72711e79690d536b6135",
    ),
    "LICENSE": (
        1_065,
        "b01d7140e1f323024b0db35e0db18ba7cd3fd3380abbec57aec55e6141864e2f",
    ),
    "HiFi_Net.py": (
        4_774,
        "353212467d2658284c91bd7ffb036599fea62bfe16f3c9855c082c42a0f0c088",
    ),
    "models/NLCDetection_api.py": (
        11_914,
        "9af51775587b6d48e8d6c4e49844ab2b679856440fcadf51702683252c2ae441",
    ),
    "models/NLCDetection_pconv.py": (
        12_017,
        "4e5f1cc3a71ec35c791ac2aceb76841f6564103364934ba80062059d6eb17333",
    ),
    "models/seg_hrnet.py": (
        22_858,
        "f6cf64c1febbb7e9cb383332987928b6eb4acaa3f81068411d5de90bc86c1d9a",
    ),
    "models/seg_hrnet_config.py": (
        1_752,
        "5ad6992a5ecc56a612245aa1bdfd5a4f8336be907c48cc6f86979f04235e4013",
    ),
    "models/LaPlacianMs.py": (
        2_370,
        "32d37e3a98919dcbe5e80d433f254b58ed6818e092dad4ef07417caba7d185cc",
    ),
    "models/GaussianSmoothing.py": (
        2_768,
        "bb6dedd7c955ddedea72d8e2f14258eb073e785520a6886516365f8b42312cef",
    ),
    "utils/custom_loss.py": (
        6_491,
        "8973bc2930ff17e833f0d288f369e1acdec5d2720548f1a9a98627b3f322882b",
    ),
    "utils/utils.py": (
        12_308,
        "0c46a3bcf008f6cb6157cd9511237a0efef2e463ab8463edd6c0e09021c63ff2",
    ),
    "utils/load_data.py": (
        13_687,
        "be1a704ffd3b81fc5b832c7cae09cc82e08e908268f8df739b03851f18a263f9",
    ),
}

CHECKPOINT_AUDIT: dict[str, dict[str, Any]] = {
    "feature_extractor": {
        "state_keys": 699,
        "state_elements": 6_379_824,
        "dtype_counts": {"torch.float32": 583, "torch.int64": 116},
        "ordered_keys_sha256": (
            "227d8a3bc85aa24104ad30ac4ccc256ca7b9ded504f7123e359adf449270dcb8"
        ),
        "tensor_schema_sha256": (
            "2b654516946debffd6dfddcc14a8fdae858bbf8680e1f8dc9092f79e7e6b02b6"
        ),
    },
    "hierarchical_localizer_classifier": {
        "state_keys": 66,
        "state_elements": 529_260,
        "dtype_counts": {"torch.float32": 62, "torch.int64": 4},
        "ordered_keys_sha256": (
            "15671527ed17eefeae640437da3900c6e79e7bb90b32b3414dfbef79656d176b"
        ),
        "tensor_schema_sha256": (
            "3bfa849976307ecc1dcf0b4e9f62baf0860d7609d707c2c3d06bc39b2481a41e"
        ),
    },
}

INITIALIZATION_AUDIT: dict[str, Any] = {
    "outer_type": "collections.OrderedDict",
    "state_keys": 874,
    "state_elements": 3_956_231,
    "dtype_counts": {"torch.float32": 729, "torch.int64": 145},
    "ordered_keys_sha256": (
        "554376c2bb3bb3452c98eaa57cfaee662809d9972d2331fb984658e2057d7740"
    ),
    "tensor_schema_sha256": (
        "39775f005431bf795638f2787dce30a516ac9de39ba12828456eeea0b1b354e6"
    ),
}

CENTER_TENSOR_SHA256 = (
    "ff38144416f5ae87e3cbf698f2207c7c1a74587b48cba3d18592486f29d30779"
)
RADIUS_TENSOR_SHA256 = (
    "4f727fb7e289540762ed7bcd686ed1c63a95f38c186a522e0942c7ca2971a14d"
)

EXPECTED_MODEL_PARAMETERS = 6_890_320
EXPECTED_MODEL_BUFFERS = 18_764
EXPECTED_MODEL_MODULES = 505
EXPECTED_COMPONENT_MODULES = {
    "feature_extractor": 437,
    "hierarchical_localizer_classifier": 68,
}
EXPECTED_TRAINABLE_PARAMETERS = {
    "feature_extractor": 6_361_208,
    "hierarchical_localizer_classifier": 528_923,
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_hifi_ifdl_balanced.py",
    "eval/opensource/run_hifi_ifdl.py",
    "eval/opensource/hifi_ifdl_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/common.py",
)

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
    "model_score_map": {
        "source": (
            "torch.nn.PairwiseDistance_p2_eps1e-6_between_18d_embedding_"
            "and_released_authentic_center"
        ),
        "shape": [256, 256],
        "dtype": "float32",
        "range": "nonnegative_unbounded",
        "probability": False,
    },
    "native_score_map": {
        "source": "bilinear_restore_raw_distance_align_corners_false",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": "nonnegative_unbounded",
        "saved_for": "successful_T2_applicable_inputs_only",
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
    "fullframe_dense_output": {
        "role": "transient_diagnostic_only",
        "saved": False,
        "scored": False,
        "promoted_to_image_score": False,
    },
}

TASK_SCOPE: dict[str, Any] = {
    "primary_task": "T1_image_detection_and_T2_localization",
    "valid_for_t1": True,
    "valid_for_t2": True,
    "fullframe_t2_not_applicable": True,
    "native_dense_output": True,
    "separate_image_classification_head": True,
    "native_image_score": (
        "one_minus_softmax_probability_of_authentic_class_in_fine_14class_head"
    ),
    "map_statistic_promoted_to_t1": False,
}

LICENSE_RECORD: dict[str, Any] = {
    "project_code": {
        "spdx": "MIT",
        "path": "LICENSE",
        "sha256": SOURCE_BOUND_FILES["LICENSE"][1],
        "commercial_use_permission": True,
        "redistribution_permission": True,
    },
    "official_checkpoint_bundle": {
        "provider": "official_author_google_drive",
        "separate_license_or_terms_found": False,
        "project_code_license_extended_to_weights": False,
        "commercial_use_clearance_established": False,
        "status": "not_separately_stated_by_release",
    },
    "limitations": {
        "trained_data_rights_not_audited": True,
        "dependency_compliance_not_legal_advice": True,
        "benchmark_use_does_not_establish_product_clearance": True,
    },
}

ARTIFACT_CONTRACT: dict[str, Any] = {
    "storage": "local_gitignored_outputs",
    "scope": "successful_T2_applicable_inputs_only",
    "fullframe_artifacts": "none_T2_not_applicable",
    "embedding_model_256": {
        "format": "NumPy .npy, allow_pickle=False",
        "directory": "embeddings_model_256",
        "shape": [legacy.EMBEDDING_CHANNELS, 256, 256],
        "dtype": "float32",
    },
    "distance_map_model_256": {
        "format": "NumPy .npy, allow_pickle=False",
        "directory": "distance_maps_model_256",
        "shape": [256, 256],
        "dtype": "float32",
    },
    "distance_map_native": {
        "format": "NumPy .npy, allow_pickle=False",
        "directory": "distance_maps_native",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
    },
    "native_binary_mask": {
        "format": "PNG",
        "directory": "masks_native",
        "mode": "L",
        "values": [0, 255],
    },
}

ARTIFACT_DIRECTORIES = (
    "embeddings_model_256",
    "distance_maps_model_256",
    "distance_maps_native",
    "masks_native",
)
NPY_HEADER_BYTES = 128
PNG_CONSERVATIVE_OVERHEAD_BYTES = 4_096
MIN_DISK_RESERVE_BYTES = 2_000_000_000
SIMPLEX_SUM_ABS_TOLERANCE = float(2 * np.finfo(np.float32).eps)
# A 14-way float32 softmax has unit roundoff u=eps/2.  Its reduction budget
# 14u is approximately 7*eps; round that up to the next binary integer,
# 8*eps, for this pre-device NumPy-vs-recorded-device CUDA sanity check only.
# Frozen A/B and fresh recorded-device replays remain bit-exact (atol=0).
STATIC_CPU_SOFTMAX_ABS_TOLERANCE = float(8 * np.finfo(np.float32).eps)

RESOURCE_EXPECTATION: dict[str, Any] = {
    "observed_mouse_v1_peak_cuda_memory_bytes": 341_163_008,
    "observed_mouse_v1_median_forward_latency_ms": 10.44,
    "observed_mouse_v1_artifact_bytes_for_550_images": 6_278_551_521,
    "balanced250_t2_raw_artifact_projection_bytes": 11_704_252_960,
    "formal_runner_projection_minutes": [5, 15],
    "fresh_replay_projection_minutes": [2, 8],
    "disk_note": (
        "the formal runner saves raw dense artifacts only for 1,025 "
        "T2-applicable inputs and requires an additional 2,000,000,000-byte "
        "free-space reserve"
    ),
}

HIERARCHY_SPECS = (
    ("out0_coarse_3class", 3),
    ("out1_5class", 5),
    ("out2_7class", 7),
    ("out3_fine_14class", 14),
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


def _require_exact_path(
    requested: Path,
    expected: Path,
    label: str,
) -> Path:
    _reject_symlink_components(requested, label)
    _reject_symlink_components(expected, f"expected {label}")
    resolved = requested.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected}")
    return resolved


def _fingerprint(value: Mapping[str, Any] | Sequence[Any]) -> str:
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


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local file that determines runner/result semantics."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        candidate = repo_root / relative
        _reject_symlink_components(
            candidate,
            f"HiFi-IFDL adapter source {relative}",
        )
        path = candidate.resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe HiFi-IFDL adapter source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_artifact_ignore(repo_root: Path) -> dict[str, Any]:
    """Prove raw HiFi-IFDL artifacts cannot enter the Git result commit."""

    probe = (
        "outputs/opensource/hifi_ifdl/_contract_probe/"
        "embeddings_model_256/sample.npy"
    )
    try:
        evidence = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "check-ignore",
                "-v",
                "--",
                probe,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "HiFi-IFDL raw artifact root is not covered by .gitignore"
        ) from error
    if not evidence or not evidence.endswith(f"\t{probe}"):
        raise ValueError("HiFi-IFDL git-ignore evidence changed")
    value = {
        "probe": probe,
        "git_check_ignore_evidence": evidence,
        "ignored": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_environment() -> dict[str, Any]:
    """Fail unless the exact audited, bytecode-isolated venv is active."""

    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    if executable != EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError(
            "HiFi-IFDL must run with the pinned interpreter "
            f"{EXPECTED_PYTHON_EXECUTABLE}, got {executable}"
        )
    if prefix != EXPECTED_VENV_ROOT:
        raise ValueError(
            f"HiFi-IFDL venv prefix changed: {prefix} != {EXPECTED_VENV_ROOT}"
        )
    if platform.python_version() != "3.12.3":
        raise ValueError("HiFi-IFDL Python version changed")
    pyvenv_path = prefix / "pyvenv.cfg"
    if (
        not pyvenv_path.is_file()
        or pyvenv_path.is_symlink()
        or pyvenv_path.stat().st_size != EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv_path) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("HiFi-IFDL pyvenv.cfg changed")
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
        raise ValueError(f"HiFi-IFDL package environment changed: {changed}")

    expected_prefix = FROZEN_PYTHONPYCACHEPREFIX.resolve()
    actual_prefix = (
        Path(sys.pycache_prefix).resolve()
        if isinstance(sys.pycache_prefix, str)
        else None
    )
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or sys.dont_write_bytecode is not True
        or actual_prefix != expected_prefix
        or not expected_prefix.is_dir()
        or expected_prefix.is_symlink()
        or any(expected_prefix.iterdir())
    ):
        raise ValueError(
            "HiFi-IFDL requires PYTHONDONTWRITEBYTECODE=1 and the frozen "
            "empty PYTHONPYCACHEPREFIX before interpreter startup"
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
        "PYTHONDONTWRITEBYTECODE": "1",
        "sys_dont_write_bytecode": True,
        "PYTHONPYCACHEPREFIX": str(expected_prefix),
        "sys_pycache_prefix": str(expected_prefix),
        "pycache_prefix_initially_empty": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_source(hifi_root: Path) -> dict[str, Any]:
    """Verify the exact official source and reject non-cache worktree drift."""

    _reject_symlink_components(hifi_root, "HiFi-IFDL source root")
    root = hifi_root.resolve()
    if (
        root != legacy.DEFAULT_HIFI_ROOT.resolve()
        or root.name != "HiFi_IFDL"
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise FileNotFoundError(f"missing/unsafe HiFi-IFDL source root: {root}")
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"HiFi-IFDL source commit changed: {commit} != "
            f"{legacy.MODEL_SOURCE_COMMIT}"
        )
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    if tree != MODEL_TREE:
        raise ValueError(f"HiFi-IFDL source tree changed: {tree}")
    origin = _git_value(root, "remote", "get-url", "origin")
    if origin != MODEL_GIT_ORIGIN:
        raise ValueError(f"HiFi-IFDL source origin changed: {origin}")
    status_text = _git_value(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status_text is None:
        raise ValueError("cannot inspect HiFi-IFDL source worktree")
    status = [line for line in status_text.splitlines() if line]
    cache_pattern = re.compile(r"^\?\? (?:[^/]+/)*__pycache__/[A-Za-z0-9_.-]+\.pyc$")
    non_cache = [line for line in status if cache_pattern.fullmatch(line) is None]
    if non_cache:
        raise ValueError(f"HiFi-IFDL source inventory drifted: {non_cache[:3]}")

    bindings: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in SOURCE_BOUND_FILES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing/unsafe HiFi-IFDL source-bound file: {path}"
            )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"HiFi-IFDL source file size changed: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"HiFi-IFDL source file hash changed: {relative}")
        if _git_value(root, "ls-files", "--error-unmatch", relative) != relative:
            raise ValueError(f"HiFi-IFDL source file is not git tracked: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "git_tracked": True,
        }
    value = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tree": tree,
        "origin": origin,
        "tracked_and_non_cache_untracked_clean": True,
        "untracked_bytecode_caches_ignored": len(status),
        "bytecode_cache_execution": False,
        "loader": "verified_source_with_empty_external_pycache_prefix",
        "source_bound_files": bindings,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _asset(
    path: Path,
    *,
    label: str,
    expected_path: Path,
    expected_bytes: int,
    expected_sha256: str,
    provider: str,
) -> dict[str, Any]:
    _reject_symlink_components(path, label)
    resolved = path.resolve()
    if resolved != expected_path.resolve():
        raise ValueError(f"{label} path changed: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {resolved}")
    if resolved.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size changed")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed")
    return {
        "path": str(resolved),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "provider": provider,
    }


def verify_assets(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    """Verify every initialization, task, center, and radius asset byte."""

    root = hifi_root.resolve()
    assets = {
        "initialization_weight": _asset(
            root / str(legacy.INITIALIZATION_WEIGHT["path"]),
            label="HiFi-IFDL HRNet initialization weight",
            expected_path=(
                legacy.DEFAULT_HIFI_ROOT / str(legacy.INITIALIZATION_WEIGHT["path"])
            ),
            expected_bytes=int(legacy.INITIALIZATION_WEIGHT["bytes"]),
            expected_sha256=str(legacy.INITIALIZATION_WEIGHT["sha256"]),
            provider="official_author_git_repository",
        ),
        "feature_extractor": _asset(
            hrnet_checkpoint,
            label="HiFi-IFDL general HRNet checkpoint 750001",
            expected_path=legacy.DEFAULT_HRNET_CHECKPOINT,
            expected_bytes=int(legacy.CHECKPOINTS["feature_extractor"]["bytes"]),
            expected_sha256=str(legacy.CHECKPOINTS["feature_extractor"]["sha256"]),
            provider="official_author_google_drive",
        ),
        "hierarchical_localizer_classifier": _asset(
            nlc_checkpoint,
            label="HiFi-IFDL general NLC checkpoint 750001",
            expected_path=legacy.DEFAULT_NLC_CHECKPOINT,
            expected_bytes=int(
                legacy.CHECKPOINTS["hierarchical_localizer_classifier"]["bytes"]
            ),
            expected_sha256=str(
                legacy.CHECKPOINTS["hierarchical_localizer_classifier"]["sha256"]
            ),
            provider="official_author_google_drive",
        ),
        "center_radius": _asset(
            root / str(legacy.CENTER_RADIUS["path"]),
            label="HiFi-IFDL released center/radius",
            expected_path=(
                legacy.DEFAULT_HIFI_ROOT / str(legacy.CENTER_RADIUS["path"])
            ),
            expected_bytes=int(legacy.CENTER_RADIUS["bytes"]),
            expected_sha256=str(legacy.CENTER_RADIUS["sha256"]),
            provider="official_author_git_repository",
        ),
    }
    value = {
        "released_identifier": "750001",
        "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "assets": assets,
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


def _audit_task_checkpoint(
    path: Path,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    import torch

    unsafe = tuple(sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path)))
    if unsafe:
        raise ValueError(f"HiFi-IFDL task checkpoint has unsafe globals: {unsafe}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or list(payload) != [
        "model",
        "optimizer",
    ]:
        raise ValueError("HiFi-IFDL task checkpoint outer schema changed")
    state = payload.get("model")
    if type(state).__name__ != "OrderedDict" or not isinstance(
        state,
        Mapping,
    ):
        raise ValueError("HiFi-IFDL task state is not an OrderedDict")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("HiFi-IFDL task state has a non-string key")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("HiFi-IFDL task state has a non-tensor value")
    dtype_counts = Counter(str(tensor.dtype) for tensor in state.values())
    elements = sum(int(tensor.numel()) for tensor in state.values())
    ordered_keys_sha256 = hashlib.sha256("\n".join(state).encode("utf-8")).hexdigest()
    tensor_schema_sha256 = _fingerprint(_checkpoint_tensor_schema(state))
    if (
        len(state) != int(expected["state_keys"])
        or elements != int(expected["state_elements"])
        or dict(sorted(dtype_counts.items())) != expected["dtype_counts"]
        or ordered_keys_sha256 != expected["ordered_keys_sha256"]
        or tensor_schema_sha256 != expected["tensor_schema_sha256"]
    ):
        raise ValueError("HiFi-IFDL task checkpoint tensor inventory changed")
    if any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("HiFi-IFDL task checkpoint has non-finite values")

    optimizer = payload.get("optimizer")
    if (
        not isinstance(optimizer, dict)
        or list(optimizer) != ["state", "param_groups"]
        or not isinstance(optimizer["state"], dict)
        or len(optimizer["state"]) != 393
        or not isinstance(optimizer["param_groups"], list)
        or len(optimizer["param_groups"]) != 1
        or not isinstance(optimizer["param_groups"][0], dict)
        or list(optimizer["param_groups"][0])
        != [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "params",
        ]
        or not isinstance(optimizer["param_groups"][0]["params"], list)
        or len(optimizer["param_groups"][0]["params"]) != 402
    ):
        raise ValueError("HiFi-IFDL optimizer provenance schema changed")
    value = {
        "outer_type": "builtins.dict",
        "top_level_keys": ["model", "optimizer"],
        "state_container": "collections.OrderedDict",
        "state_dict_tensors": len(state),
        "state_dict_elements": elements,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": ordered_keys_sha256,
        "tensor_schema_sha256": tensor_schema_sha256,
        "all_floating_tensors_finite": True,
        "weights_only": True,
        "map_location": "cpu",
        "unsafe_globals": [],
        "optimizer": {
            "container": "builtins.dict",
            "keys": ["state", "param_groups"],
            "state_entries": 393,
            "param_groups": 1,
            "registered_parameter_ids": 402,
            "loaded_into_model": False,
            "retained_for_provenance_only": True,
        },
    }
    return {**value, "contract_sha256": _fingerprint(value)}, state


def _audit_initialization_weight(path: Path) -> dict[str, Any]:
    import torch

    try:
        torch.serialization.get_unsafe_globals_in_checkpoint(path)
    except ValueError as error:
        scanner = {
            "status": "unsupported_legacy_serialization",
            "error_type": type(error).__name__,
            "weights_only_load_succeeded": True,
        }
    else:
        raise ValueError("HiFi-IFDL legacy initialization scanner behavior changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict" or not isinstance(
        state,
        Mapping,
    ):
        raise ValueError("HiFi-IFDL initialization state schema changed")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("HiFi-IFDL initialization has a non-string key")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("HiFi-IFDL initialization has a non-tensor value")
    dtype_counts = Counter(str(tensor.dtype) for tensor in state.values())
    elements = sum(int(tensor.numel()) for tensor in state.values())
    ordered_keys_sha256 = hashlib.sha256("\n".join(state).encode("utf-8")).hexdigest()
    tensor_schema_sha256 = _fingerprint(_checkpoint_tensor_schema(state))
    if (
        len(state) != int(INITIALIZATION_AUDIT["state_keys"])
        or elements != int(INITIALIZATION_AUDIT["state_elements"])
        or dict(sorted(dtype_counts.items())) != INITIALIZATION_AUDIT["dtype_counts"]
        or ordered_keys_sha256 != INITIALIZATION_AUDIT["ordered_keys_sha256"]
        or tensor_schema_sha256 != INITIALIZATION_AUDIT["tensor_schema_sha256"]
    ):
        raise ValueError("HiFi-IFDL initialization tensor inventory changed")
    if any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("HiFi-IFDL initialization has non-finite values")
    value = {
        **INITIALIZATION_AUDIT,
        "all_floating_tensors_finite": True,
        "map_location": "cpu",
        "weights_only": True,
        "unsafe_global_scanner": scanner,
        "loaded_into_balanced_model": False,
        "reason": (
            "the complete released general HRNet task checkpoint strictly "
            "replaces every parameter and buffer"
        ),
    }
    del state
    gc.collect()
    return {**value, "contract_sha256": _fingerprint(value)}


def _audit_center_radius(path: Path) -> tuple[dict[str, Any], Any, Any]:
    import torch

    try:
        torch.serialization.get_unsafe_globals_in_checkpoint(path)
    except ValueError as error:
        scanner = {
            "status": "unsupported_legacy_serialization",
            "error_type": type(error).__name__,
            "weights_only_load_succeeded": True,
        }
    else:
        raise ValueError("HiFi-IFDL center scanner behavior changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or list(payload) != [
        "center",
        "radius",
    ]:
        raise ValueError("HiFi-IFDL center/radius outer schema changed")
    center = payload["center"]
    radius = payload["radius"]
    if (
        not isinstance(center, torch.Tensor)
        or list(center.shape) != [legacy.EMBEDDING_CHANNELS]
        or center.dtype != torch.float32
        or not bool(torch.isfinite(center).all())
        or _array_sha256(center.detach().cpu().numpy()) != CENTER_TENSOR_SHA256
    ):
        raise ValueError("HiFi-IFDL center tensor changed")
    if (
        not isinstance(radius, torch.Tensor)
        or list(radius.shape) != []
        or radius.dtype != torch.float32
        or not bool(torch.isfinite(radius))
        or float(radius.item()) != float(legacy.CENTER_RADIUS["radius_value"])
        or _array_sha256(radius.detach().cpu().numpy()) != RADIUS_TENSOR_SHA256
    ):
        raise ValueError("HiFi-IFDL radius tensor changed")
    value = {
        "outer_type": "builtins.dict",
        "top_level_keys": ["center", "radius"],
        "center": {
            "shape": [legacy.EMBEDDING_CHANNELS],
            "dtype": "torch.float32",
            "tensor_sha256": CENTER_TENSOR_SHA256,
            "loaded_for_inference": True,
        },
        "radius": {
            "shape": [],
            "dtype": "torch.float32",
            "value": float(radius.item()),
            "tensor_sha256": RADIUS_TENSOR_SHA256,
            "loaded_for_provenance_validation_only": True,
        },
        "map_location": "cpu",
        "weights_only": True,
        "unsafe_global_scanner": scanner,
    }
    return (
        {**value, "contract_sha256": _fingerprint(value)},
        center.detach().clone(),
        radius.detach().clone(),
    )


class _CPUDataParallelEnvelope:
    """Namespace marker; the torch Module implementation is built lazily."""


def _cpu_envelope(module: Any) -> Any:
    """Create ``module.*`` state keys without DataParallel's CUDA probing."""

    import torch

    class Envelope(torch.nn.Module):
        def __init__(self, child: Any) -> None:
            super().__init__()
            self.module = child

    return Envelope(module)


def _construct_cpu_model_audit(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct and strictly load the full model on CPU without a forward."""

    import torch

    forbidden = [
        name for name in sys.modules if name == "models" or name.startswith("models.")
    ]
    if forbidden:
        raise RuntimeError(
            "HiFi-IFDL CPU preflight requires a fresh process without a "
            f"pre-imported top-level models namespace: {forbidden[:3]}"
        )
    root_text = str(hifi_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    raw_modules: dict[str, Any] = {}
    envelopes: dict[str, Any] = {}
    center = None
    radius = None
    try:
        with _working_directory(hifi_root):
            from models.NLCDetection_api import NLCDetection
            from models.seg_hrnet import HighResolutionNet
            from models.seg_hrnet_config import get_cfg_defaults

            with legacy._numpy_int_compatibility():
                feature_extractor = HighResolutionNet(get_cfg_defaults())
            with mock.patch.object(
                torch.Tensor,
                "cuda",
                new=lambda tensor, *args, **kwargs: tensor,
            ):
                hierarchical_head = NLCDetection()
        raw_modules = {
            "feature_extractor": feature_extractor,
            "hierarchical_localizer_classifier": hierarchical_head,
        }

        checkpoint_audits: dict[str, Any] = {}
        states: dict[str, Mapping[str, Any]] = {}
        for role, path in (
            ("feature_extractor", hrnet_checkpoint),
            ("hierarchical_localizer_classifier", nlc_checkpoint),
        ):
            audit, state = _audit_task_checkpoint(
                path,
                CHECKPOINT_AUDIT[role],
            )
            checkpoint_audits[role] = audit
            states[role] = state

        model_components: dict[str, Any] = {}
        for role, raw_module in raw_modules.items():
            envelope = _cpu_envelope(raw_module)
            incompatible = envelope.load_state_dict(
                states[role],
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"HiFi-IFDL {role} strict state load was incomplete")
            envelope.eval()
            if tuple(envelope.state_dict()) != tuple(states[role]):
                raise ValueError(f"HiFi-IFDL {role} model/checkpoint key order changed")
            parameters = sum(int(value.numel()) for value in raw_module.parameters())
            trainable = sum(
                int(value.numel())
                for value in raw_module.parameters()
                if value.requires_grad
            )
            buffers = sum(int(value.numel()) for value in raw_module.buffers())
            modules = sum(1 for _ in raw_module.modules())
            contract = legacy.CHECKPOINTS[role]
            if (
                parameters != int(contract["parameters"])
                or trainable != EXPECTED_TRAINABLE_PARAMETERS[role]
                or buffers != int(contract["buffers"])
                or modules != EXPECTED_COMPONENT_MODULES[role]
                or envelope.training
            ):
                raise ValueError(f"HiFi-IFDL {role} constructed inventory changed")
            model_components[role] = {
                "construction_device": "cpu",
                "strict_state_dict_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "state_key_order_matches_checkpoint": True,
                "eval_mode": True,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "buffer_elements": buffers,
                "module_count": modules,
            }
            envelopes[role] = envelope

        init_audit = _audit_initialization_weight(
            hifi_root / str(legacy.INITIALIZATION_WEIGHT["path"])
        )
        center_audit, center, radius = _audit_center_radius(
            hifi_root / str(legacy.CENTER_RADIUS["path"])
        )
        total_parameters = sum(
            value["parameters"] for value in model_components.values()
        )
        total_trainable = sum(
            value["trainable_parameters"] for value in model_components.values()
        )
        total_buffers = sum(
            value["buffer_elements"] for value in model_components.values()
        )
        total_modules = sum(
            value["module_count"] for value in model_components.values()
        )
        if (
            total_parameters != EXPECTED_MODEL_PARAMETERS
            or total_trainable != sum(EXPECTED_TRAINABLE_PARAMETERS.values())
            or total_buffers != EXPECTED_MODEL_BUFFERS
            or total_modules != EXPECTED_MODEL_MODULES
        ):
            raise ValueError("HiFi-IFDL complete model inventory changed")
        checkpoint_value = {
            "task_components": checkpoint_audits,
            "initialization_weight": init_audit,
            "center_radius": center_audit,
            "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        }
        model_value = {
            "construction_device": "cpu",
            "components": model_components,
            "parameter_count": total_parameters,
            "trainable_parameter_count": total_trainable,
            "buffer_elements": total_buffers,
            "module_count": total_modules,
            "forward_performed": False,
            "constructor_cuda_calls_neutralized": (
                "two_unused_unregistered_split_tensors_only"
            ),
            "state_prefix_rewrite": False,
            "cpu_module_envelope_for_DataParallel_prefix": True,
        }
        return (
            {
                **checkpoint_value,
                "contract_sha256": _fingerprint(checkpoint_value),
            },
            {
                **model_value,
                "contract_sha256": _fingerprint(model_value),
            },
        )
    finally:
        envelopes.clear()
        raw_modules.clear()
        del center
        del radius
        gc.collect()


def run_cpu_preflight(
    *,
    repo_root: Path,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    """Run every fail-closed gate without initializing CUDA."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "HiFi-IFDL CPU preflight must start before CUDA initialization"
        )
    environment = verify_environment()
    source = verify_source(hifi_root)
    assets = verify_assets(
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    adapter_sources = adapter_source_contract(repo_root)
    artifact_ignore = verify_artifact_ignore(repo_root)
    checkpoint_audit, model_audit = _construct_cpu_model_audit(
        hifi_root=hifi_root.resolve(),
        hrnet_checkpoint=hrnet_checkpoint.resolve(),
        nlc_checkpoint=nlc_checkpoint.resolve(),
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("HiFi-IFDL CPU preflight initialized CUDA")
    value = {
        "schema_version": CPU_PREFLIGHT_SCHEMA,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "environment": environment,
        "source": source,
        "assets": assets,
        "adapter_sources": adapter_sources,
        "artifact_ignore": artifact_ignore,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "balanced250_forward_performed": False,
        "balanced250_score_computed": False,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    """Configure deterministic inference only after CPU preflight."""

    if device_text != "cpu" and (
        not device_text.startswith("cuda:")
        or not device_text[5:].isdigit()
        or str(int(device_text[5:])) != device_text[5:]
    ):
        raise ValueError("device must be exactly cpu or cuda:N")
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with frozen runtime")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "HiFi-IFDL accelerator was initialized before runtime " "configuration"
        )
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device index is unavailable")
        torch.cuda.set_device(device)
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    value: dict[str, Any] = {
        "device": str(device),
        "seed": MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        value["cuda"] = {
            "logical_device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
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
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1_775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
    ):
        raise ValueError("formal HiFi-IFDL Balanced250 selection drifted")
    if sum(_t2_semantics(row)[0] for row in selected) != FORMAL_T2_IMAGES:
        raise ValueError("formal HiFi-IFDL T2 coverage drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        isinstance(per_condition_limit, bool)
        or not isinstance(per_condition_limit, int)
        or per_condition_limit != DEFAULT_SMOKE_LIMIT
    ):
        raise ValueError(
            f"smoke per-condition-limit must be exactly " f"{DEFAULT_SMOKE_LIMIT}"
        )
    spec = SelectionSpec(
        capability=Capability.LOCAL_T1_T2,
        per_condition_limit=per_condition_limit,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if counts[condition] >= per_condition_limit:
            continue
        sample_id = str(panel_row["sample_id"])
        source = inputs_by_id.get(sample_id)
        if source is None or source.get("panel") is not True:
            raise ValueError("smoke panel has a dangling/non-panel input")
        selected.append(source)
        counts[condition] += 1
    expected = {condition: per_condition_limit for condition in BALANCED_CONDITIONS}
    if dict(counts) != expected:
        raise ValueError("smoke panel does not cover every condition")
    selected.sort(key=lambda row: int(row["rank"]))
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
            raise ValueError("formal mode accepts no selection limit")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode accepts no sample-id")
        return _smoke_selection(
            release,
            (
                DEFAULT_SMOKE_LIMIT
                if per_condition_limit is None
                else per_condition_limit
            ),
        )
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode accepts no per-condition-limit")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.LOCAL_T1_T2,
            sample_id=sample_id,
        )
        return spec, select_inputs(release, spec)
    raise ValueError(f"unsupported inference mode: {mode!r}")


def _t2_semantics(row: Mapping[str, Any]) -> tuple[bool, str]:
    condition = str(row["condition"])
    if condition == "real":
        return True, "all_zero_real_false_positive_area"
    if condition in {"local_mouse", "local_cat", "local_trash_can"}:
        return True, "exact_diff_local_insertion"
    if condition in {
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    }:
        return False, "not_applicable_fullframe"
    raise ValueError(f"unsupported HiFi-IFDL condition: {condition}")


def result_task_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    applicable, semantics = _t2_semantics(row)
    return {
        "primary_task": "T1_image_detection_and_T2_localization",
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "t2_target_semantics": semantics,
        "fullframe_t2_not_applicable": not applicable,
        "native_dense_output_present_during_forward": True,
        "dense_output_saved": applicable,
        "image_score_source": "native_fine_14class_head",
        "map_statistic_promoted_to_t1": False,
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    applicable, semantics = _t2_semantics(row)
    return {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "checkpoint_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "checkpoint_released_identifier": "750001",
        "task_scope": result_task_scope(row),
        "t2_applicable": applicable,
        "t2_target_semantics": semantics,
    }


def _preprocess_with_audit(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    tensor, native_size, audit = legacy.preprocess_image(path)
    width, height = native_size
    expected = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [width, height],
        "model_size": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "tensor_shape": [3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "tensor_dtype": "float32",
        "tensor_sha256": _array_sha256(tensor),
    }
    if audit != expected:
        raise ValueError("HiFi-IFDL official preprocess audit changed")
    if (
        tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or tensor.shape != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise ValueError("HiFi-IFDL preprocessed tensor contract changed")
    return tensor, native_size, audit


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32)
    shifted = logits - np.max(logits)
    exponent = np.exp(shifted, dtype=np.float32)
    return np.asarray(
        exponent / np.sum(exponent, dtype=np.float32),
        dtype=np.float32,
    )


def _score_payload(
    hierarchy_logits: Mapping[str, Any],
    fine_probabilities: np.ndarray,
) -> dict[str, Any]:
    if set(hierarchy_logits) != {name for name, _ in HIERARCHY_SPECS}:
        raise ValueError("HiFi-IFDL hierarchy logit names changed")
    hierarchy: dict[str, dict[str, Any]] = {}
    for name, classes in HIERARCHY_SPECS:
        logits = np.ascontiguousarray(
            hierarchy_logits[name],
            dtype=np.float32,
        )
        if logits.shape != (classes,) or not np.isfinite(logits).all():
            raise ValueError(f"HiFi-IFDL {name} classification vector changed")
        hierarchy[name] = {
            "values": [float(value) for value in logits],
            "shape": [classes],
            "dtype": "float32",
            "array_sha256": _array_sha256(logits),
        }
    probabilities = np.ascontiguousarray(
        fine_probabilities,
        dtype=np.float32,
    )
    if probabilities.shape != (14,) or not np.isfinite(probabilities).all():
        raise ValueError("HiFi-IFDL fine probability vector changed")
    if (
        float(probabilities.min()) < 0.0
        or float(probabilities.max()) > 1.0
        or not math.isclose(
            float(probabilities.sum(dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=SIMPLEX_SUM_ABS_TOLERANCE,
        )
    ):
        raise ValueError("HiFi-IFDL fine probabilities fall outside the simplex")
    fine_logits = np.asarray(
        hierarchy["out3_fine_14class"]["values"],
        dtype=np.float32,
    )
    cpu_probabilities = _stable_softmax(fine_logits)
    if not np.allclose(
        probabilities,
        cpu_probabilities,
        rtol=0.0,
        atol=STATIC_CPU_SOFTMAX_ABS_TOLERANCE,
    ):
        raise ValueError("HiFi-IFDL fine logits/probabilities disagree")
    score = float(np.float32(1.0) - probabilities[0])
    fine_index = int(np.argmax(probabilities))
    benchmark_binary = SCORE_SPEC.decision(score)
    official_binary = fine_index != 0
    return {
        "classification_hierarchy": hierarchy,
        "fine_probabilities": [float(value) for value in probabilities],
        "fine_probabilities_shape": [14],
        "fine_probabilities_dtype": "float32",
        "fine_probabilities_array_sha256": _array_sha256(probabilities),
        "ai_score": score,
        "score_semantics": ("one_minus_softmax_probability_fine_class_0_authentic"),
        "classification_decision": ("forged" if benchmark_binary else "authentic"),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (CLASSIFICATION_THRESHOLD_OPERATOR),
        "official_fine_class_index": fine_index,
        "official_fine_class_name": legacy.FINE_CLASS_NAMES[fine_index],
        "official_binary_decision": official_binary,
        "official_decision": ("forged" if official_binary else "authentic"),
        "official_decision_rule": ("argmax_fine_14class_index_not_equal_to_0"),
    }


def _validate_auxiliary_mask(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("HiFi-IFDL auxiliary mask record is not an object")
    expected_keys = {
        "shape",
        "dtype",
        "minimum",
        "maximum",
        "mean",
        "primary_output",
        "reason",
    }
    if set(value) != expected_keys:
        raise ValueError("HiFi-IFDL auxiliary mask key set changed")
    minimum = _finite_number(value.get("minimum"), "auxiliary minimum")
    maximum = _finite_number(value.get("maximum"), "auxiliary maximum")
    mean = _finite_number(value.get("mean"), "auxiliary mean")
    if (
        value.get("shape") != [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE]
        or value.get("dtype") != "float32"
        or minimum < 0.0
        or maximum > 1.0
        or minimum > mean
        or mean > maximum
        or value.get("primary_output") is not False
        or value.get("reason")
        != (
            "the official public localize API ignores this sigmoid mask "
            "and thresholds hypersphere distance instead"
        )
    ):
        raise ValueError("HiFi-IFDL auxiliary mask contract changed")
    return dict(value)


def artifact_paths(run_dir: Path, sample_id: str) -> dict[str, Path]:
    return {
        "embedding_model_256": (run_dir / "embeddings_model_256" / f"{sample_id}.npy"),
        "distance_model_256": (
            run_dir / "distance_maps_model_256" / f"{sample_id}.npy"
        ),
        "distance_native": (run_dir / "distance_maps_native" / f"{sample_id}.npy"),
        "native_mask": run_dir / "masks_native" / f"{sample_id}.png",
    }


def _array_artifact_record(
    *,
    path: Path,
    repo_root: Path,
    array: np.ndarray,
    semantics: str,
) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "array_sha256": _array_sha256(array),
        "semantics": semantics,
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    embedding: np.ndarray,
    distance_model: np.ndarray,
    distance_native: np.ndarray,
) -> dict[str, Any]:
    embedding_record = _array_artifact_record(
        path=paths["embedding_model_256"],
        repo_root=repo_root,
        array=embedding,
        semantics="official_18d_pixel_embedding",
    )
    model_record = _array_artifact_record(
        path=paths["distance_model_256"],
        repo_root=repo_root,
        array=distance_model,
        semantics=("raw_pairwise_distance_to_released_authentic_center_model_256"),
    )
    native_record = _array_artifact_record(
        path=paths["distance_native"],
        repo_root=repo_root,
        array=distance_native,
        semantics=("raw_distance_bilinear_align_corners_false_native_restore"),
    )
    mask_path = paths["native_mask"]
    return {
        "artifact_paths": {
            "embedding_model_256": embedding_record["path"],
            "distance_model_256": model_record["path"],
            "distance_native": native_record["path"],
            "native_mask": repo_relative(mask_path, repo_root),
        },
        "embedding_artifact": embedding_record,
        "distance_model_artifact": model_record,
        "distance_native_artifact": native_record,
        "score_map_path": native_record["path"],
        "score_map_sha256": native_record["sha256"],
        "score_map_bytes": native_record["bytes"],
        "score_map_shape": native_record["shape"],
        "score_map_dtype": native_record["dtype"],
        "score_map_array_sha256": native_record["array_sha256"],
        "score_map_semantics": native_record["semantics"],
        "mask_path": repo_relative(mask_path, repo_root),
        "mask_sha256": sha256_file(mask_path),
        "mask_bytes": mask_path.stat().st_size,
        "mask_shape": list(distance_native.shape),
        "mask_dtype": "uint8",
        "mask_semantics": ("raw_hypersphere_distance_greater_than_or_equal_to_2_3"),
        "dense_output_disposition": ("saved_and_scored_with_applicable_ground_truth"),
    }


def _not_applicable_artifact_fields() -> dict[str, Any]:
    return {
        "artifact_paths": None,
        "embedding_artifact": None,
        "distance_model_artifact": None,
        "distance_native_artifact": None,
        "score_map_path": None,
        "score_map_sha256": None,
        "score_map_bytes": None,
        "score_map_shape": None,
        "score_map_dtype": None,
        "score_map_array_sha256": None,
        "score_map_semantics": None,
        "mask_path": None,
        "mask_sha256": None,
        "mask_bytes": None,
        "mask_shape": None,
        "mask_dtype": None,
        "mask_semantics": None,
        "dense_output_disposition": (
            "discarded_transient_diagnostic_T2_not_applicable"
        ),
    }


def _localization_payload(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    model_map: np.ndarray,
    native_map: np.ndarray,
) -> dict[str, Any]:
    target_native = load_ground_truth(row, repo_root)
    if target_native is None:
        raise ValueError("HiFi-IFDL applicable T2 row has no ground truth")
    target_model = legacy.model_space_target(target_native)
    include_ap = str(row["condition"]) != "real"
    return {
        "model_256": binary_distance_metrics_strict(
            model_map,
            target_model,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
        "native": binary_distance_metrics_strict(
            native_map,
            target_native,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
    }


_ARTIFACT_KEYS = frozenset(
    {
        "artifact_paths",
        "embedding_artifact",
        "distance_model_artifact",
        "distance_native_artifact",
        "score_map_path",
        "score_map_sha256",
        "score_map_bytes",
        "score_map_shape",
        "score_map_dtype",
        "score_map_array_sha256",
        "score_map_semantics",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
        "dense_output_disposition",
    }
)
_OK_ONLY_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "preprocess",
        "classification_hierarchy",
        "fine_probabilities",
        "fine_probabilities_shape",
        "fine_probabilities_dtype",
        "fine_probabilities_array_sha256",
        "ai_score",
        "score_semantics",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "official_fine_class_index",
        "official_fine_class_name",
        "official_binary_decision",
        "official_decision",
        "official_decision_rule",
        "auxiliary_learned_mask",
        *_ARTIFACT_KEYS,
        "mask_threshold",
        "mask_threshold_operator",
        "localization",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_ERROR_ONLY_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "error_type",
        "error",
        "traceback",
    }
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _resolve_exact_artifact(
    *,
    repo_root: Path,
    artifact_root: Path,
    value: Any,
    expected: Path,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path is not a string")
    path = _anchored(Path(value), repo_root)
    _reject_symlink_components(path, label)
    if path != expected.resolve():
        raise ValueError(f"{label} path changed")
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    return path


def _load_npy_artifact(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_sha256: Any,
    expected_bytes: Any,
    expected_array_sha256: Any,
    nonnegative: bool,
    label: str,
) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} file SHA-256 changed")
    expected_file_bytes = (
        int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize + NPY_HEADER_BYTES
    )
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != expected_file_bytes
        or path.stat().st_size != expected_file_bytes
    ):
        raise ValueError(f"{label} file byte size changed")
    array = np.load(path, allow_pickle=False)
    if (
        not isinstance(array, np.ndarray)
        or array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or (nonnegative and float(array.min()) < 0.0)
    ):
        raise ValueError(f"{label} array contract changed")
    if _array_sha256(array) != expected_array_sha256:
        raise ValueError(f"{label} array SHA-256 changed")
    return array


def _validate_array_record(
    record: Any,
    *,
    repo_root: Path,
    artifact_root: Path,
    expected_path: Path,
    expected_shape: tuple[int, ...],
    expected_semantics: str,
    nonnegative: bool,
    label: str,
) -> np.ndarray:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "bytes",
        "shape",
        "dtype",
        "array_sha256",
        "semantics",
    }:
        raise ValueError(f"{label} metadata changed")
    if (
        record.get("shape") != list(expected_shape)
        or record.get("dtype") != "float32"
        or record.get("semantics") != expected_semantics
    ):
        raise ValueError(f"{label} metadata contract changed")
    path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        value=record.get("path"),
        expected=expected_path,
        label=label,
    )
    return _load_npy_artifact(
        path,
        expected_shape=expected_shape,
        expected_sha256=record.get("sha256"),
        expected_bytes=record.get("bytes"),
        expected_array_sha256=record.get("array_sha256"),
        nonnegative=nonnegative,
        label=label,
    )


def _load_binary_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_sha256: Any,
    expected_bytes: Any,
    native_map: np.ndarray,
) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError("HiFi-IFDL native mask SHA-256 changed")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError("HiFi-IFDL native mask byte size changed")
    with Image.open(path) as opened:
        if opened.mode != "L" or opened.size != (
            expected_shape[1],
            expected_shape[0],
        ):
            raise ValueError("HiFi-IFDL native mask image contract changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    if (
        pixels.shape != expected_shape
        or not np.isin(
            pixels,
            (0, 255),
        ).all()
    ):
        raise ValueError("HiFi-IFDL native mask values changed")
    expected = np.where(
        native_map >= MASK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)
    if not np.array_equal(pixels, expected):
        raise ValueError("HiFi-IFDL native mask threshold replay changed")
    return pixels


def _validate_preprocess(
    preprocess: Any,
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    recompute: bool,
) -> None:
    if not isinstance(preprocess, Mapping):
        raise ValueError("HiFi-IFDL preprocess record is not an object")
    expected_keys = {
        "decoder",
        "channel_order",
        "decoded_dtype",
        "native_size",
        "model_size",
        "geometry",
        "resize_interpolation",
        "input_crop",
        "input_reencode",
        "normalization",
        "tensor_shape",
        "tensor_dtype",
        "tensor_sha256",
    }
    width = int(row["width"])
    height = int(row["height"])
    if (
        set(preprocess) != expected_keys
        or preprocess.get("decoder") != "imageio.v2.imread"
        or preprocess.get("channel_order") != "RGB"
        or preprocess.get("decoded_dtype") != "uint8"
        or preprocess.get("native_size") != [width, height]
        or preprocess.get("model_size") != [256, 256]
        or preprocess.get("geometry")
        != "direct_stretch_without_aspect_ratio_preservation"
        or preprocess.get("resize_interpolation") != "Pillow.Image.Resampling.BICUBIC"
        or preprocess.get("input_crop") is not None
        or preprocess.get("input_reencode") is not False
        or preprocess.get("normalization") != "uint8_rgb_divide_255_float32"
        or preprocess.get("tensor_shape") != [3, 256, 256]
        or preprocess.get("tensor_dtype") != "float32"
        or not isinstance(preprocess.get("tensor_sha256"), str)
        or len(str(preprocess["tensor_sha256"])) != 64
    ):
        raise ValueError("HiFi-IFDL preprocess record changed")
    if recompute:
        path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _, _, expected = _preprocess_with_audit(path)
        if dict(preprocess) != expected:
            raise ValueError("HiFi-IFDL preprocess replay changed")


def _validate_score_payload(row: Mapping[str, Any], sample_id: str) -> None:
    hierarchy = row.get("classification_hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise ValueError("HiFi-IFDL hierarchy record is invalid")
    raw_logits: dict[str, list[float]] = {}
    for name, classes in HIERARCHY_SPECS:
        record = hierarchy.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "values",
            "shape",
            "dtype",
            "array_sha256",
        }:
            raise ValueError(f"HiFi-IFDL {name} record changed")
        values = np.asarray(record.get("values"), dtype=np.float32)
        if (
            values.shape != (classes,)
            or not np.isfinite(values).all()
            or record.get("shape") != [classes]
            or record.get("dtype") != "float32"
            or record.get("array_sha256") != _array_sha256(values)
        ):
            raise ValueError(f"HiFi-IFDL {name} payload changed")
        raw_logits[name] = [float(value) for value in values]
    probabilities = np.asarray(
        row.get("fine_probabilities"),
        dtype=np.float32,
    )
    expected = _score_payload(raw_logits, probabilities)
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"HiFi-IFDL {sample_id} score payload {key} changed")


def _validate_ok_artifacts(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(artifact_root, sample_id)
    applicable, _ = _t2_semantics(input_row)
    if not applicable:
        expected = _not_applicable_artifact_fields()
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"HiFi-IFDL full-frame row claims artifact field {key}"
                )
        if row.get("localization") is not None:
            raise ValueError("HiFi-IFDL full-frame row claims localization")
        return

    path_map = row.get("artifact_paths")
    if not isinstance(path_map, Mapping) or set(path_map) != {
        "embedding_model_256",
        "distance_model_256",
        "distance_native",
        "native_mask",
    }:
        raise ValueError("HiFi-IFDL artifact path map changed")
    embedding = _validate_array_record(
        row.get("embedding_artifact"),
        repo_root=repo_root,
        artifact_root=artifact_root,
        expected_path=paths["embedding_model_256"],
        expected_shape=(legacy.EMBEDDING_CHANNELS, 256, 256),
        expected_semantics="official_18d_pixel_embedding",
        nonnegative=False,
        label="HiFi-IFDL embedding",
    )
    model_map = _validate_array_record(
        row.get("distance_model_artifact"),
        repo_root=repo_root,
        artifact_root=artifact_root,
        expected_path=paths["distance_model_256"],
        expected_shape=(256, 256),
        expected_semantics=(
            "raw_pairwise_distance_to_released_authentic_center_model_256"
        ),
        nonnegative=True,
        label="HiFi-IFDL model distance map",
    )
    native_shape = (
        int(input_row["height"]),
        int(input_row["width"]),
    )
    native_map = _validate_array_record(
        row.get("distance_native_artifact"),
        repo_root=repo_root,
        artifact_root=artifact_root,
        expected_path=paths["distance_native"],
        expected_shape=native_shape,
        expected_semantics=("raw_distance_bilinear_align_corners_false_native_restore"),
        nonnegative=True,
        label="HiFi-IFDL native distance map",
    )
    del embedding
    expected_path_values = {
        "embedding_model_256": row["embedding_artifact"]["path"],
        "distance_model_256": row["distance_model_artifact"]["path"],
        "distance_native": row["distance_native_artifact"]["path"],
        "native_mask": row.get("mask_path"),
    }
    if dict(path_map) != expected_path_values:
        raise ValueError("HiFi-IFDL artifact path aliases changed")
    native_record = row["distance_native_artifact"]
    aliases = {
        "score_map_path": native_record["path"],
        "score_map_sha256": native_record["sha256"],
        "score_map_bytes": native_record["bytes"],
        "score_map_shape": native_record["shape"],
        "score_map_dtype": native_record["dtype"],
        "score_map_array_sha256": native_record["array_sha256"],
        "score_map_semantics": native_record["semantics"],
        "mask_shape": list(native_shape),
        "mask_dtype": "uint8",
        "mask_semantics": ("raw_hypersphere_distance_greater_than_or_equal_to_2_3"),
        "dense_output_disposition": ("saved_and_scored_with_applicable_ground_truth"),
    }
    for key, value in aliases.items():
        if row.get(key) != value:
            raise ValueError(f"HiFi-IFDL artifact alias {key} changed")
    mask_path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        value=row.get("mask_path"),
        expected=paths["native_mask"],
        label="HiFi-IFDL native binary mask",
    )
    _load_binary_mask(
        mask_path,
        expected_shape=native_shape,
        expected_sha256=row.get("mask_sha256"),
        expected_bytes=row.get("mask_bytes"),
        native_map=native_map,
    )
    expected_localization = _localization_payload(
        row=input_row,
        repo_root=repo_root,
        model_map=model_map,
        native_map=native_map,
    )
    if stable_json(row.get("localization")) != stable_json(expected_localization):
        raise ValueError("HiFi-IFDL localization replay changed")


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
        raise ValueError("HiFi-IFDL result status is invalid")
    expected = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    allowed = set(expected) | (
        set(_OK_ONLY_KEYS) if status == "ok" else set(_ERROR_ONLY_KEYS)
    )
    if set(attempt) != allowed:
        raise ValueError("HiFi-IFDL result key set changed")
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"HiFi-IFDL result identity field {key} changed")
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("HiFi-IFDL result completed_at is invalid")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
        ):
            raise ValueError("HiFi-IFDL error payload is invalid")
        return
    _validate_preprocess(
        attempt.get("preprocess"),
        row=input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    _validate_score_payload(attempt, str(input_row["sample_id"]))
    _validate_auxiliary_mask(attempt.get("auxiliary_learned_mask"))
    if (
        attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError("HiFi-IFDL mask threshold semantics changed")
    latency = _finite_number(attempt.get("latency_ms"), "latency_ms")
    peak = attempt.get("peak_cuda_memory_bytes")
    if latency < 0.0:
        raise ValueError("HiFi-IFDL latency is negative")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("HiFi-IFDL peak CUDA memory is invalid")
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
    expected_ids = {str(row["sample_id"]) for row in selected}
    histories: dict[str, list[str]] = {}
    for line_number, attempt in enumerate(attempts, start=1):
        sample_id = attempt.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError(
                f"HiFi-IFDL history row {line_number} has unexpected " "sample_id"
            )
        status = attempt.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"HiFi-IFDL history row {line_number} has invalid status")
        prior = histories.setdefault(sample_id, [])
        if "ok" in prior:
            raise ValueError(
                "HiFi-IFDL append-only history contains an attempt after " "success"
            )
        prior.append(str(status))
    return {
        "policy": ("zero_or_more_errors_then_at_most_one_terminal_success_per_id"),
        "physical_attempts": len(attempts),
        "ids_with_attempts": len(histories),
        "errors": sum(
            status == "error" for statuses in histories.values() for status in statuses
        ),
        "recovered_error_to_ok": sum(
            statuses[-1] == "ok" and "error" in statuses[:-1]
            for statuses in histories.values()
        ),
    }


def _prepare_artifact_root(artifact_root: Path) -> None:
    expected = set(ARTIFACT_DIRECTORIES)
    if artifact_root.exists():
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise ValueError(f"HiFi-IFDL artifact root is unsafe: {artifact_root}")
        for entry in artifact_root.iterdir():
            if entry.name not in expected or not entry.is_dir() or entry.is_symlink():
                raise ValueError(
                    "HiFi-IFDL artifact root has unexpected/unsafe entry: " f"{entry}"
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
            f"missing/unsafe HiFi-IFDL artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != expected_directories or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("HiFi-IFDL artifact root inventory mismatch")
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    successful_applicable = {
        sample_id
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok" and _t2_semantics(inputs_by_id[sample_id])[0]
    }
    expected = {
        directory: {f"{sample_id}.npy" for sample_id in successful_applicable}
        for directory in ARTIFACT_DIRECTORIES[:-1]
    }
    expected["masks_native"] = {
        f"{sample_id}.png" for sample_id in successful_applicable
    }
    counts: dict[str, int] = {}
    for directory_name, expected_names in expected.items():
        directory = artifact_root / directory_name
        children = list(directory.iterdir())
        if any(child.is_symlink() or not child.is_file() for child in children):
            raise ValueError(f"HiFi-IFDL {directory_name} has unsafe/non-file entries")
        actual_names = {child.name for child in children}
        if actual_names != expected_names:
            raise ValueError(
                f"HiFi-IFDL {directory_name} inventory mismatch: "
                f"missing={sorted(expected_names - actual_names)[:1]}, "
                f"extra={sorted(actual_names - expected_names)[:1]}"
            )
        counts[directory_name] = len(actual_names)
    return counts


def _required_artifact_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Return a conservative raw-artifact bound plus the fixed reserve."""

    applicable = [row for row in rows if _t2_semantics(row)[0]]
    if not applicable:
        return 0
    embedding_per_image = (
        legacy.EMBEDDING_CHANNELS
        * legacy.MODEL_INPUT_SIZE
        * legacy.MODEL_INPUT_SIZE
        * np.dtype(np.float32).itemsize
        + NPY_HEADER_BYTES
    )
    model_map_per_image = (
        legacy.MODEL_INPUT_SIZE
        * legacy.MODEL_INPUT_SIZE
        * np.dtype(np.float32).itemsize
        + NPY_HEADER_BYTES
    )
    native_maps = sum(
        int(row["width"]) * int(row["height"]) * np.dtype(np.float32).itemsize
        + NPY_HEADER_BYTES
        for row in applicable
    )
    native_masks = sum(
        int(row["width"]) * int(row["height"]) + PNG_CONSERVATIVE_OVERHEAD_BYTES
        for row in applicable
    )
    return (
        len(applicable) * (embedding_per_image + model_map_per_image)
        + native_maps
        + native_masks
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
            "insufficient disk for HiFi-IFDL raw artifacts: "
            f"required={required}, free={usage.free}"
        )
    return {
        "free_bytes_before_inference": int(usage.free),
        "conservative_pending_bytes_plus_reserve": int(required),
        "fixed_reserve_bytes": (MIN_DISK_RESERVE_BYTES if required else 0),
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
        raise ValueError(f"HiFi-IFDL run directory is unsafe: {run_dir}")
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
            "HiFi-IFDL run directory has unexpected entries: "
            f"{sorted(unexpected)[:1]}"
        )
    for entry in entries:
        _reject_symlink_components(
            entry,
            f"HiFi-IFDL run file {entry.name}",
        )
        if not entry.is_file() or entry.is_symlink():
            raise ValueError(f"HiFi-IFDL run entry is not a regular file: {entry}")
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
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repository": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_tree": MODEL_TREE,
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_bundle_sha256": (legacy.CHECKPOINT_BUNDLE_SHA256),
            "checkpoint_release": dict(legacy.CHECKPOINT_RELEASE),
            "checkpoint_components": {
                role: dict(contract) for role, contract in legacy.CHECKPOINTS.items()
            },
            "center_radius": dict(legacy.CENTER_RADIUS),
            "initialization_weight": dict(legacy.INITIALIZATION_WEIGHT),
            "fine_class_names": list(legacy.FINE_CLASS_NAMES),
            "variant": (
                "official_general_detection_and_localization_750001_" "not_retrained"
            ),
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "decode": "imageio.v2.imread",
            "channel_order": "RGB",
            "decoded_dtype": "uint8",
            "geometry": "direct_stretch_to_256x256",
            "resize": "Pillow_bicubic",
            "aspect_ratio_preserved": False,
            "input_crop": None,
            "input_reencode": False,
            "scale": "uint8_divide_255_float32",
            "normalization_mean_std": None,
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "batch_size": 1,
        },
        "inference": {
            "feature_extractor": "HighResolutionNet",
            "head": "NLCDetection",
            "classification_output": (
                "one_minus_float32_softmax_fine14_authentic_index_0"
            ),
            "official_classification_decision": ("argmax_fine14_index_not_equal_to_0"),
            "localization_output": ("PairwiseDistance_p2_eps1e-6_to_released_center"),
            "native_restore": ("torch_bilinear_raw_distance_align_corners_false"),
            "threshold_after_native_restore": True,
            "auxiliary_sigmoid_mask_is_primary": False,
            "test_time_augmentation": False,
            "ensemble": False,
            "autocast": False,
            "forward_passes_per_image": 1,
            "static_cpu_softmax_sanity": {
                "classes": 14,
                "reference": "numpy_float32_exp_sum_div_from_fine14_logits",
                "comparison": "recorded_device_torch_softmax_float32",
                "simplex_sum_absolute_tolerance": (SIMPLEX_SUM_ABS_TOLERANCE),
                "cross_device_absolute_tolerance": (STATIC_CPU_SOFTMAX_ABS_TOLERANCE),
                "roundoff_basis": (
                    "float32 unit roundoff u=eps/2; 14u is approximately "
                    "7eps; next binary integer bound is 8eps"
                ),
                "scope": "static_cross_device_sanity_only",
                "recorded_device_smoke_and_fresh_replay_tolerance": 0.0,
                "affects_score_decision_or_artifacts": False,
            },
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
        "assets": dict(cpu_preflight["assets"]),
        "environment": dict(cpu_preflight["environment"]),
        "artifact_ignore": dict(cpu_preflight["artifact_ignore"]),
        "checkpoint_audit": dict(cpu_preflight["checkpoint_audit"]),
        "model_audit": dict(cpu_preflight["model_audit"]),
        "license": LICENSE_RECORD,
        "resource_expectation": RESOURCE_EXPECTATION,
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
        "--hifi-root",
        type=Path,
        default=legacy.DEFAULT_HIFI_ROOT,
    )
    parser.add_argument(
        "--hrnet-checkpoint",
        type=Path,
        default=legacy.DEFAULT_HRNET_CHECKPOINT,
    )
    parser.add_argument(
        "--nlc-checkpoint",
        type=Path,
        default=legacy.DEFAULT_NLC_CHECKPOINT,
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
        raise ValueError("resume HiFi-IFDL manifest/config drifted")
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
        raise ValueError("resume HiFi-IFDL manifest key set changed")
    if _read_jsonl_strict(expected_inputs_path) != list(selected):
        raise ValueError("resume HiFi-IFDL expected inputs drifted")
    started_at = prior.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume HiFi-IFDL started_at is invalid")
    completed_at = prior.get("completed_at")
    if status == "running":
        if completed_at is not None:
            raise ValueError("resume running HiFi-IFDL manifest has completed_at")
    elif not isinstance(completed_at, str) or not completed_at:
        raise ValueError("resume finalized HiFi-IFDL completed_at is invalid")

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
    ):
        raise ValueError("resume HiFi-IFDL disk preflight changed")

    outputs = prior.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resume HiFi-IFDL outputs are missing")
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
        raise ValueError("resume HiFi-IFDL output key set changed")
    for key, value in expected_immutable["outputs"].items():
        if outputs.get(key) != value:
            raise ValueError(f"resume HiFi-IFDL output {key} changed")

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
                raise ValueError(
                    f"resume finalized HiFi-IFDL {relative_key} hash " "changed"
                )
        inventory = outputs.get("artifact_inventory")
        if (
            not isinstance(inventory, Mapping)
            or set(inventory) != set(ARTIFACT_DIRECTORIES)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in inventory.values()
            )
        ):
            raise ValueError("resume HiFi-IFDL artifact inventory changed")
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
            }
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in execution.values()
            )
        ):
            raise ValueError("resume HiFi-IFDL execution record changed")
    return prior, started_at


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_manifest_path = _require_exact_path(
        _unresolved_anchored(args.dataset_manifest, repo_root),
        repo_root / DEFAULT_DATASET_MANIFEST,
        "HiFi-IFDL dataset manifest",
    )
    hifi_root = _require_exact_path(
        _unresolved_anchored(args.hifi_root, repo_root),
        legacy.DEFAULT_HIFI_ROOT,
        "HiFi-IFDL source root",
    )
    hrnet_checkpoint = _require_exact_path(
        _unresolved_anchored(args.hrnet_checkpoint, repo_root),
        legacy.DEFAULT_HRNET_CHECKPOINT,
        "HiFi-IFDL HRNet checkpoint",
    )
    nlc_checkpoint = _require_exact_path(
        _unresolved_anchored(args.nlc_checkpoint, repo_root),
        legacy.DEFAULT_NLC_CHECKPOINT,
        "HiFi-IFDL NLC checkpoint",
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
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
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
    results_root = _require_exact_path(
        requested_results,
        repo_root / DEFAULT_RESULTS_DIR,
        "HiFi-IFDL results root",
    )
    requested_artifacts = _unresolved_anchored(
        args.artifacts_dir,
        repo_root,
    )
    artifacts_root = _require_exact_path(
        requested_artifacts,
        repo_root / DEFAULT_ARTIFACTS_DIR,
        "HiFi-IFDL artifacts root",
    )
    run_dir = _safe_child(
        results_root,
        run_id,
        "HiFi-IFDL run directory",
    )
    artifact_root = _safe_child(
        artifacts_root,
        run_id,
        "HiFi-IFDL artifact directory",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("HiFi-IFDL result and artifact directories must be disjoint")
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

    # The strict complete model-load gate intentionally precedes every CUDA
    # API that configures or initializes an accelerator.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    device, runtime = configure_runtime(args.device or "cuda:0")

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
        "t1_applicable_images": len(selected),
        "t2_applicable_images": sum(_t2_semantics(row)[0] for row in selected),
    }
    if prior_manifest is not None and prior_manifest.get("dataset") != dataset_record:
        raise ValueError("resume HiFi-IFDL dataset envelope changed")
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
    _validate_physical_attempt_history(selected, physical_before)
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_before:
        input_row = inputs_by_id[str(attempt["sample_id"])]
        _validate_runner_attempt(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=attempt.get("status") == "ok",
            recompute_preprocess=attempt.get("status") == "ok",
        )
    pending = [
        row
        for row in selected
        if latest_before.latest_by_sample_id.get(
            str(row["sample_id"]),
            {},
        ).get("status")
        != "ok"
    ]
    resume_skips = len(selected) - len(pending)
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

    import torch

    models = None
    center = None
    radius = None
    new_successes = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            models, center, radius, loaded_device = legacy.load_model(
                hifi_root=hifi_root,
                hrnet_checkpoint=hrnet_checkpoint,
                nlc_checkpoint=nlc_checkpoint,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("HiFi-IFDL loaded on an unexpected device")
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
                    raise ValueError("canonical HiFi-IFDL image dimensions changed")
                assert models is not None
                assert center is not None
                processed, peak_bytes, latency_ms = legacy.infer_one(
                    models,
                    center,
                    loaded_device,
                    tensor,
                    native_width=width,
                    native_height=height,
                )
                score_payload = _score_payload(
                    processed["hierarchy_logits"],
                    np.asarray(
                        processed["fine_probabilities"],
                        dtype=np.float32,
                    ),
                )
                if (
                    float(processed["score"]) != score_payload["ai_score"]
                    or bool(processed["benchmark_binary_decision"])
                    != (score_payload["classification_decision"] == "forged")
                    or int(processed["official_fine_class_index"])
                    != score_payload["official_fine_class_index"]
                    or str(processed["official_fine_class_name"])
                    != score_payload["official_fine_class_name"]
                    or bool(processed["official_binary_decision"])
                    != score_payload["official_binary_decision"]
                ):
                    raise ValueError(
                        "HiFi-IFDL legacy postprocess score contract changed"
                    )
                auxiliary = _validate_auxiliary_mask(
                    processed["auxiliary_learned_mask_stats"]
                )
                applicable, _ = _t2_semantics(input_row)
                localization: dict[str, Any] | None = None
                if applicable:
                    embedding = np.ascontiguousarray(
                        processed["embedding"],
                        dtype=np.float32,
                    )
                    distance_model = np.ascontiguousarray(
                        processed["distance_model_256"],
                        dtype=np.float32,
                    )
                    distance_native = np.ascontiguousarray(
                        processed["distance_native"],
                        dtype=np.float32,
                    )
                    legacy._atomic_save_npy(
                        paths["embedding_model_256"],
                        embedding,
                    )
                    legacy._atomic_save_npy(
                        paths["distance_model_256"],
                        distance_model,
                    )
                    legacy._atomic_save_npy(
                        paths["distance_native"],
                        distance_native,
                    )
                    legacy._atomic_save_mask(
                        paths["native_mask"],
                        distance_native >= MASK_THRESHOLD,
                    )
                    artifact_fields = _artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        embedding=embedding,
                        distance_model=distance_model,
                        distance_native=distance_native,
                    )
                    localization = _localization_payload(
                        row=input_row,
                        repo_root=repo_root,
                        model_map=distance_model,
                        native_map=distance_native,
                    )
                else:
                    artifact_fields = _not_applicable_artifact_fields()
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **score_payload,
                    "auxiliary_learned_mask": auxiliary,
                    **artifact_fields,
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
                    f"score={result['ai_score']:.9f} "
                    f"t2={'yes' if applicable else 'N/A'}",
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
                _validate_runner_attempt(
                    error_result,
                    input_row=input_row,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    run_manifest_fingerprint=fingerprint,
                    verify_artifacts=False,
                )
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
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del models
        del center
        del radius
        gc.collect()
        if device.type == "cuda":
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
        "summary_kind": ("runtime_coverage_and_artifact_inventory_only"),
        "scientific_metrics": None,
        "scientific_metrics_owner": ("analyze_hifi_ifdl_balanced.py"),
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
        "attempt_history": history,
        "artifact_inventory": inventory,
        "generated_at": utc_now(),
    }
    atomic_write_json(summary_path, summary)
    manifest["status"] = summary["status"]
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": new_successes,
        "resume_skips": resume_skips,
        "new_errors": new_errors,
        "physical_result_rows": len(physical_results),
        "latest_result_rows": len(latest.latest_by_sample_id),
        "superseded_attempts": latest.superseded_attempts,
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
                "attempt_history": history,
                "artifact_inventory": inventory,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError("HiFi-IFDL fail-closed inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
