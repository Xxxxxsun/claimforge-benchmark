#!/usr/bin/env python3
"""Run the frozen official PSCC-Net bundle on Balanced250.

The audited Mouse-v1 adapter remains unchanged.  This v2 orchestration layer
binds the exact official HRNet, progressive localization head, and independent
classification head to the independent Balanced250 release.

PSCC-Net natively supports both tasks.  T1 therefore covers all 1,775 cache
images.  T2 covers authentic images and the three local-insertion conditions
only; full-frame edits retain diagnostic probability maps but are explicitly
localization-not-applicable.

Every dataset, environment, source, asset, adapter, and strict CPU model-load
check completes before accelerator configuration.  The CPU gate performs no
model forward and computes no Balanced250 score.
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
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_psccnet as legacy
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
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
from eval.opensource.psccnet_metrics import binary_pixel_metrics_strict


RUN_MANIFEST_SCHEMA = "psccnet_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "psccnet_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "psccnet_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "psccnet_balanced_cpu_preflight_v2"

MODEL_NAME = legacy.MODEL_NAME
MODEL_SLUG = legacy.MODEL_SLUG
MODEL_ARCHITECTURE = (
    "HRNet_W18_small_v2_progressive_spatio_channel_correlation_"
    "plus_independent_detection_head"
)
MODEL_GIT_ORIGIN = "https://github.com/proteus1991/PSCC-Net.git"
CHECKPOINT_ID = "official_committed_synthetic_pretrained_bundle"
PREPROCESS_PROFILE = "official_imageio_native_rgb_float32_divide_255"

MODEL_SEED = 42
CLASSIFICATION_THRESHOLD = legacy.CLASSIFICATION_THRESHOLD
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD = legacy.MASK_THRESHOLD
MASK_THRESHOLD_OPERATOR = ">"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/psccnet")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/psccnet")
DEFAULT_FORMAL_RUN_ID = "psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727"
DEFAULT_SMOKE_RUN_ID_A = "psccnet_tcsvt2022_official_balanced250_v1_smoke5x7_a_20260727"
DEFAULT_SMOKE_RUN_ID_B = "psccnet_tcsvt2022_official_balanced250_v1_smoke5x7_b_20260727"
DEFAULT_SMOKE_LIMIT = 5

EXPECTED_VENV_ROOT = Path("/root/.cache/claimforge/venvs/psccnet")
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_BYTES = 197
EXPECTED_PYVENV_SHA256 = (
    "b5e3d281734d6b01868b2a8506f999a366bfccfc1fbaf1450d09c6ec538bd71b"
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
    ".gitignore": (
        1_799,
        "79b6f8054f8ef5e9e78c18174bf57caf29b11410166b9268d6923e87520eb88f",
    ),
    "README.md": (
        5_774,
        "a4db54f5b755c505d0df85115acc7ca8c3d22306a46cbbbd5d11d102d5b6d9de",
    ),
    "test.py": (
        3_083,
        legacy.SOURCE_FILES["test.py"],
    ),
    "models/NLCDetection.py": (
        4_798,
        legacy.SOURCE_FILES["models/NLCDetection.py"],
    ),
    "models/detection_head.py": (
        5_390,
        legacy.SOURCE_FILES["models/detection_head.py"],
    ),
    "models/seg_hrnet.py": (
        17_547,
        legacy.SOURCE_FILES["models/seg_hrnet.py"],
    ),
    "models/seg_hrnet_config.py": (
        1_710,
        legacy.SOURCE_FILES["models/seg_hrnet_config.py"],
    ),
    "utils/load_vdata.py": (
        1_650,
        legacy.SOURCE_FILES["utils/load_vdata.py"],
    ),
    "utils/config.py": (
        627,
        legacy.SOURCE_FILES["utils/config.py"],
    ),
    "utils/utils.py": (
        1_277,
        "63575d9425ea009bcf7723cc10b30bb4fb9ff700b23892ffe53ca68618983041",
    ),
    "LICENSE": (
        1_069,
        legacy.SOURCE_FILES["LICENSE"],
    ),
}

CHECKPOINT_AUDIT = {
    "feature_extractor": {
        "state_keys": 444,
        "state_elements": 2_037_538,
        "dtype_counts": {"torch.float32": 370, "torch.int64": 74},
        "ordered_keys_sha256": (
            "8ebb32a3efb65b46ed575b4c32f50f2452221fd3e957e08433f9a6793110d13e"
        ),
        "tensor_schema_sha256": (
            "1f4d35b3ec8b5db2fcba4d94db654feb44c50d0542ad9902756102cc393f26cc"
        ),
    },
    "localization_head": {
        "state_keys": 64,
        "state_elements": 719_824,
        "dtype_counts": {"torch.float32": 64},
        "ordered_keys_sha256": (
            "9228a0b5fa712c98b5a4490c531f38eaa745cf41beadcf8816ad82e059107326"
        ),
        "tensor_schema_sha256": (
            "45fe765d763d89545e7a8d6cd29cfeabda1081a9aa12a38e3ce8d1fcb73c5a5e"
        ),
    },
    "classification_head": {
        "state_keys": 128,
        "state_elements": 924_070,
        "dtype_counts": {"torch.float32": 108, "torch.int64": 20},
        "ordered_keys_sha256": (
            "b1cc03da04ea14b2f97c2571b6750540bcf1c7afc1fe98ff0b7e15bcf2241e92"
        ),
        "tensor_schema_sha256": (
            "92ae7a2847a01a5c5d7775c683f686be37d80791351c307a074a89664555d97e"
        ),
    },
}

INITIALIZATION_AUDIT = {
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

EXPECTED_MODEL_PARAMETERS = 3_667_942
EXPECTED_MODEL_BUFFERS = 13_490
EXPECTED_MODEL_MODULES = 391
EXPECTED_COMPONENT_MODULES = {
    "feature_extractor": 279,
    "localization_head": 41,
    "classification_head": 71,
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_psccnet_balanced.py",
    "eval/opensource/run_psccnet.py",
    "eval/opensource/psccnet_metrics.py",
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
    "progressive_probability_maps": {
        "source": "four_released_NLC_sigmoid_outputs",
        "shapes": [[256, 256], [128, 128], [64, 64], [32, 32]],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "primary": "stage_1_256x256",
        "saved_for": "all_successful_inputs",
    },
    "native_probability_map": {
        "source": ("stage_1_probability_bilinear_restore_align_corners_true"),
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "saved_for": "all_successful_inputs",
        "fullframe_role": "diagnostic_only",
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
    "separate_image_classification_head": True,
    "native_image_score": "softmax_class_1_of_independent_detection_head",
}

LICENSE_RECORD: dict[str, Any] = {
    "project_license": {
        "spdx": "MIT",
        "path": "LICENSE",
        "sha256": legacy.SOURCE_FILES["LICENSE"],
        "commercial_use_permission": True,
        "redistribution_permission": True,
    },
    "official_checkpoints": {
        "distribution": "committed_in_the_same_official_repository",
        "separate_terms_present": False,
        "treated_as_project_repository_material": True,
    },
    "limitations": {
        "trained_data_rights_not_audited": True,
        "dependency_compliance_not_legal_advice": True,
        "benchmark_use_does_not_establish_product_clearance": True,
    },
}

ARTIFACT_CONTRACT: dict[str, Any] = {
    "storage": "local_gitignored_outputs",
    "progressive_maps": {
        "format": "NumPy .npy, allow_pickle=False",
        "directories": [
            "progressive_mask1",
            "progressive_mask2",
            "progressive_mask3",
            "progressive_mask4",
        ],
        "shapes": [[256, 256], [128, 128], [64, 64], [32, 32]],
        "dtype": "float32",
        "inventory": "one_of_each_stage_per_successful_input",
    },
    "native_probability_map": {
        "format": "NumPy .npy, allow_pickle=False",
        "directory": "score_maps_native",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "inventory": "one_per_successful_input",
    },
    "native_binary_mask": {
        "format": "PNG",
        "directory": "masks_native",
        "mode": "L",
        "values": [0, 255],
        "inventory": "one_per_successful_T2_applicable_input",
    },
}

ARTIFACT_DIRECTORIES = (
    "progressive_mask1",
    "progressive_mask2",
    "progressive_mask3",
    "progressive_mask4",
    "score_maps_native",
    "masks_native",
)
PROGRESSIVE_SHAPES = ((256, 256), (128, 128), (64, 64), (32, 32))
NPY_HEADER_BYTES = 128
PNG_CONSERVATIVE_OVERHEAD_BYTES = 4_096
MIN_DISK_RESERVE_BYTES = 2_000_000_000
STATIC_CPU_SOFTMAX_ABS_TOLERANCE = float(2 * np.finfo(np.float32).eps)

RESOURCE_EXPECTATION: dict[str, Any] = {
    "observed_mouse_v1_peak_cuda_memory_bytes": 8_435_241_984,
    "observed_mouse_v1_median_forward_latency_ms": 149.085,
    "formal_forward_only_projection_minutes": 4.5,
    "formal_runner_with_artifact_io_projection_minutes": [8, 16],
    "fresh_replay_projection_minutes": [5, 10],
    "disk_note": (
        "formal float32 maps are approximately 12.75 GiB before filesystem "
        "overhead; the runner additionally requires a 2,000,000,000-byte reserve"
    ),
}


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


def _fingerprint(value: Mapping[str, Any]) -> str:
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


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local file that determines runner/result semantics."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        candidate = repo_root / relative
        _reject_symlink_components(
            candidate,
            f"PSCC-Net adapter source {relative}",
        )
        path = candidate.resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe PSCC-Net adapter source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_artifact_ignore(repo_root: Path) -> dict[str, Any]:
    """Prove raw PSCC-Net output paths are ignored by the repository."""

    probe = "outputs/opensource/psccnet/_contract_probe/artifact.npy"
    try:
        evidence = subprocess.check_output(
            ["git", "-C", str(repo_root), "check-ignore", "-v", "--", probe],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "PSCC-Net raw artifact root is not covered by .gitignore"
        ) from error
    if not evidence or not evidence.endswith(f"\t{probe}"):
        raise ValueError("PSCC-Net git-ignore evidence changed")
    value = {
        "probe": probe,
        "git_check_ignore_evidence": evidence,
        "ignored": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_environment() -> dict[str, Any]:
    """Fail unless the audited PSCC-Net virtual environment is active."""

    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    if executable != EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError(
            "PSCC-Net must run with the pinned interpreter "
            f"{EXPECTED_PYTHON_EXECUTABLE}, got {executable}"
        )
    if prefix != EXPECTED_VENV_ROOT:
        raise ValueError(
            f"PSCC-Net venv prefix changed: {prefix} != {EXPECTED_VENV_ROOT}"
        )
    if platform.python_version() != "3.12.3":
        raise ValueError("PSCC-Net Python version changed")
    pyvenv_path = prefix / "pyvenv.cfg"
    if (
        not pyvenv_path.is_file()
        or pyvenv_path.is_symlink()
        or pyvenv_path.stat().st_size != EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv_path) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("PSCC-Net pyvenv.cfg changed")
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
        raise ValueError(f"PSCC-Net package environment changed: {changed}")
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
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_source(psccnet_root: Path) -> dict[str, Any]:
    """Verify the exact clean official source checkout."""

    _reject_symlink_components(psccnet_root, "PSCC-Net source root")
    root = psccnet_root.resolve()
    if root.name != "PSCC-Net" or not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe PSCC-Net source root: {root}")
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"PSCC-Net source commit changed: {commit} != "
            f"{legacy.MODEL_SOURCE_COMMIT}"
        )
    origin = _git_value(root, "remote", "get-url", "origin")
    if origin != MODEL_GIT_ORIGIN:
        raise ValueError(f"PSCC-Net source origin changed: {origin}")
    status = _git_value(root, "status", "--short", "--untracked-files=all")
    if status is None:
        raise ValueError("cannot inspect PSCC-Net source worktree")
    if status:
        raise ValueError("PSCC-Net source worktree is dirty")

    bindings: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in SOURCE_BOUND_FILES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing/unsafe PSCC-Net source-bound file: {path}"
            )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"PSCC-Net source file size changed: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"PSCC-Net source file hash changed: {relative}")
        if _git_value(root, "ls-files", "--error-unmatch", relative) != relative:
            raise ValueError(f"PSCC-Net source file is not git tracked: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "git_tracked": True,
        }
    value = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "origin": origin,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bindings,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_assets(psccnet_root: Path) -> dict[str, Any]:
    """Verify all official initialization and task checkpoint bytes."""

    root = psccnet_root.resolve()
    assets: dict[str, dict[str, Any]] = {}
    contracts = {
        "initialization_weight": legacy.INITIALIZATION_WEIGHT,
        **legacy.CHECKPOINTS,
    }
    for role, contract in contracts.items():
        path = root / str(contract["path"])
        _reject_symlink_components(path, f"PSCC-Net {role}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe PSCC-Net {role}: {path}")
        if path.stat().st_size != int(contract["bytes"]):
            raise ValueError(f"PSCC-Net {role} byte size changed")
        if sha256_file(path) != str(contract["sha256"]):
            raise ValueError(f"PSCC-Net {role} SHA-256 changed")
        relative = str(contract["path"])
        if _git_value(root, "ls-files", "--error-unmatch", relative) != relative:
            raise ValueError(f"PSCC-Net {role} is not git tracked")
        assets[role] = {
            "path": str(path),
            "repository_path": relative,
            "bytes": int(contract["bytes"]),
            "sha256": str(contract["sha256"]),
            "git_tracked": True,
            "provider": "official_author_git_repository",
        }
    value = {
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


def _audit_state(
    path: Path,
    expected: Mapping[str, Any],
    *,
    scan_unsafe_globals: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    import torch

    unsafe: tuple[str, ...] | None
    if scan_unsafe_globals:
        unsafe = tuple(
            sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        )
        if unsafe:
            raise ValueError(f"PSCC-Net checkpoint has unsafe globals: {unsafe}")
    else:
        unsafe = None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict":
        raise ValueError("PSCC-Net checkpoint is not an OrderedDict")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("PSCC-Net checkpoint has a non-string key")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("PSCC-Net checkpoint has a non-tensor value")
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
        raise ValueError("PSCC-Net checkpoint tensor inventory changed")
    if any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("PSCC-Net checkpoint has non-finite values")
    keys = tuple(state)
    value = {
        "outer_type": "collections.OrderedDict",
        "state_dict_tensors": len(state),
        "state_dict_elements": elements,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": ordered_keys_sha256,
        "tensor_schema_sha256": tensor_schema_sha256,
        "all_floating_tensors_finite": True,
        "weights_only": True,
        "map_location": "cpu",
        "unsafe_globals": (
            list(unsafe) if unsafe is not None else "legacy_pickle_not_scanable"
        ),
    }
    del state
    gc.collect()
    return {**value, "contract_sha256": _fingerprint(value)}, keys


def _construct_cpu_model_audit(
    psccnet_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct and strictly load all modules on CPU without a forward."""

    import torch

    if str(psccnet_root) not in sys.path:
        sys.path.insert(0, str(psccnet_root))
    if not hasattr(np, "int"):
        setattr(np, "int", int)
    previous = Path.cwd()
    os.chdir(psccnet_root)
    try:
        from models.NLCDetection import NLCDetection
        from models.detection_head import DetectionHead
        from models.seg_hrnet import get_seg_model
        from models.seg_hrnet_config import get_hrnet_cfg

        config = get_hrnet_cfg()
        # The complete official task checkpoint replaces every HRNet state
        # entry.  Suppressing the constructor-only legacy CUDA pickle is the
        # only way to make the strict preflight CPU-only; its bytes and tensor
        # inventory are independently audited below.
        config.PRETRAINED = ""
        raw_modules = {
            "feature_extractor": get_seg_model(config),
            "localization_head": NLCDetection(
                {"crop_size": list(legacy.MODEL_CROP_SIZE)}
            ),
            "classification_head": DetectionHead(
                {"crop_size": list(legacy.MODEL_CROP_SIZE)}
            ),
        }
    finally:
        os.chdir(previous)

    checkpoint_audits: dict[str, Any] = {}
    model_components: dict[str, Any] = {}
    wrapped: dict[str, Any] = {}
    try:
        for role, raw_module in raw_modules.items():
            path = psccnet_root / str(legacy.CHECKPOINTS[role]["path"])
            audit, checkpoint_keys = _audit_state(
                path,
                CHECKPOINT_AUDIT[role],
                scan_unsafe_globals=True,
            )
            module = legacy._wrap_for_official_state(
                raw_module,
                torch.device("cpu"),
            )
            legacy._load_state(
                module=module,
                path=path,
                contract=legacy.CHECKPOINTS[role],
            )
            module.eval()
            model_keys = tuple(module.state_dict())
            if model_keys != checkpoint_keys:
                raise ValueError(f"PSCC-Net {role} model/checkpoint key order changed")
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
                or trainable != parameters
                or buffers != int(contract["buffers"])
                or modules != EXPECTED_COMPONENT_MODULES[role]
                or module.training
            ):
                raise ValueError(f"PSCC-Net {role} constructed inventory changed")
            checkpoint_audits[role] = audit
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
            wrapped[role] = module

        init_path = psccnet_root / str(legacy.INITIALIZATION_WEIGHT["path"])
        init_audit, _ = _audit_state(
            init_path,
            INITIALIZATION_AUDIT,
            scan_unsafe_globals=False,
        )
        total_parameters = sum(
            value["parameters"] for value in model_components.values()
        )
        total_buffers = sum(
            value["buffer_elements"] for value in model_components.values()
        )
        total_modules = sum(
            value["module_count"] for value in model_components.values()
        )
        if (
            total_parameters != EXPECTED_MODEL_PARAMETERS
            or total_buffers != EXPECTED_MODEL_BUFFERS
            or total_modules != EXPECTED_MODEL_MODULES
        ):
            raise ValueError("PSCC-Net complete model inventory changed")
        model_value = {
            "construction_device": "cpu",
            "constructor_initialization_weight_loaded": False,
            "constructor_initialization_weight_suppression_reason": (
                "legacy_cuda_pickle_is_fully_overwritten_by_complete_"
                "strict_task_checkpoint"
            ),
            "complete_task_checkpoint_replaces_constructor_state": True,
            "components": model_components,
            "parameter_count": total_parameters,
            "trainable_parameter_count": total_parameters,
            "buffer_elements": total_buffers,
            "module_count": total_modules,
            "forward_performed": False,
        }
        checkpoint_value = {
            "initialization_weight": init_audit,
            "task_components": checkpoint_audits,
            "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
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
        wrapped.clear()
        raw_modules.clear()
        gc.collect()


def run_cpu_preflight(
    *,
    repo_root: Path,
    psccnet_root: Path,
) -> dict[str, Any]:
    """Run every fail-closed gate without initializing CUDA."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "PSCC-Net CPU preflight must start before CUDA initialization"
        )
    environment = verify_environment()
    source = verify_source(psccnet_root)
    assets = verify_assets(psccnet_root)
    adapter_sources = adapter_source_contract(repo_root)
    artifact_ignore = verify_artifact_ignore(repo_root)
    checkpoint_audit, model_audit = _construct_cpu_model_audit(psccnet_root.resolve())
    if torch.cuda.is_initialized():
        raise RuntimeError("PSCC-Net CPU preflight initialized CUDA")
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
            "PSCC-Net accelerator was initialized before runtime configuration"
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
        raise ValueError("formal PSCC-Net Balanced250 selection drifted")
    if sum(_t2_semantics(row)[0] for row in selected) != FORMAL_T2_IMAGES:
        raise ValueError("formal PSCC-Net T2 coverage drifted")
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
            f"smoke per-condition-limit must be exactly {DEFAULT_SMOKE_LIMIT}"
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
    raise ValueError(f"unsupported PSCC-Net condition: {condition}")


def result_task_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    applicable, semantics = _t2_semantics(row)
    return {
        "primary_task": "T1_image_detection_and_T2_localization",
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "t2_target_semantics": semantics,
        "fullframe_t2_not_applicable": not applicable,
        "native_dense_output_present": True,
        "image_score_source": "independent_detection_head",
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
        "native_size": [width, height],
        "input_resize": "none",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255",
        "alpha_policy": "not_applicable",
        "tensor_shape": [3, height, width],
        "tensor_sha256": _array_sha256(tensor),
    }
    if audit != expected:
        raise ValueError("PSCC-Net official preprocess audit changed")
    if (
        tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or tensor.shape != (3, height, width)
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise ValueError("PSCC-Net preprocessed tensor contract changed")
    return tensor, native_size, audit


def _stable_softmax_class1(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    exponent = np.exp(shifted, dtype=np.float32)
    return np.asarray(exponent / np.sum(exponent, dtype=np.float32), dtype=np.float32)


def _score_payload(
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
    if logits.shape != (2,) or probabilities.shape != (2,):
        raise ValueError("PSCC-Net classification vector shape changed")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("PSCC-Net classification vector is non-finite")
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise ValueError("PSCC-Net class probabilities fall outside [0,1]")
    if not math.isclose(
        float(probabilities.sum(dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=STATIC_CPU_SOFTMAX_ABS_TOLERANCE,
    ):
        raise ValueError("PSCC-Net class probabilities do not sum to one")
    cpu_probabilities = _stable_softmax_class1(logits)
    if not np.allclose(
        probabilities,
        cpu_probabilities,
        rtol=0.0,
        atol=STATIC_CPU_SOFTMAX_ABS_TOLERANCE,
    ):
        raise ValueError("PSCC-Net logits/probabilities disagree")
    score = float(probabilities[1])
    decision = SCORE_SPEC.decision(score)
    return {
        "classification_logits": [float(value) for value in logits],
        "classification_logits_dtype": "float32",
        "classification_logits_sha256": _array_sha256(logits),
        "classification_probabilities": [float(value) for value in probabilities],
        "classification_probabilities_dtype": "float32",
        "classification_probabilities_sha256": _array_sha256(probabilities),
        "ai_score": score,
        "score_semantics": "softmax_probability_class_1_forged",
        "classification_decision": "forged" if decision else "authentic",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (CLASSIFICATION_THRESHOLD_OPERATOR),
    }


def artifact_paths(run_dir: Path, sample_id: str) -> dict[str, Path]:
    return {
        "progressive_mask1": run_dir / "progressive_mask1" / f"{sample_id}.npy",
        "progressive_mask2": run_dir / "progressive_mask2" / f"{sample_id}.npy",
        "progressive_mask3": run_dir / "progressive_mask3" / f"{sample_id}.npy",
        "progressive_mask4": run_dir / "progressive_mask4" / f"{sample_id}.npy",
        "native_probability": (run_dir / "score_maps_native" / f"{sample_id}.npy"),
        "native_mask": run_dir / "masks_native" / f"{sample_id}.png",
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    model_masks: Sequence[np.ndarray],
    native_map: np.ndarray,
    mask_path: Path | None,
) -> dict[str, Any]:
    progressive: list[dict[str, Any]] = []
    artifact_path_map: dict[str, str | None] = {}
    for stage, model_mask in enumerate(model_masks, start=1):
        key = f"progressive_mask{stage}"
        path = paths[key]
        relative = repo_relative(path, repo_root)
        progressive.append(
            {
                "stage": stage,
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": list(model_mask.shape),
                "dtype": str(model_mask.dtype),
                "semantics": (
                    "official_primary_NLC_sigmoid_probability"
                    if stage == 1
                    else "official_auxiliary_NLC_sigmoid_probability"
                ),
                "primary": stage == 1,
                "array_sha256": _array_sha256(model_mask),
            }
        )
        artifact_path_map[key] = relative
    native_relative = repo_relative(paths["native_probability"], repo_root)
    mask_relative = (
        repo_relative(mask_path, repo_root) if mask_path is not None else None
    )
    artifact_path_map["native_probability"] = native_relative
    artifact_path_map["native_mask"] = mask_relative
    return {
        "artifact_paths": artifact_path_map,
        "progressive_maps": progressive,
        "primary_model_score_map_path": progressive[0]["path"],
        "primary_model_score_map_sha256": progressive[0]["sha256"],
        "primary_model_score_map_bytes": progressive[0]["bytes"],
        "primary_model_score_map_shape": progressive[0]["shape"],
        "primary_model_score_map_dtype": "float32",
        "primary_model_score_map_semantics": (
            "official_primary_NLC_sigmoid_probability"
        ),
        "score_map_path": native_relative,
        "score_map_sha256": sha256_file(paths["native_probability"]),
        "score_map_bytes": paths["native_probability"].stat().st_size,
        "score_map_shape": list(native_map.shape),
        "score_map_dtype": str(native_map.dtype),
        "score_map_semantics": (
            "primary_probability_bilinear_align_corners_true_native_restore"
        ),
        "score_map_array_sha256": _array_sha256(native_map),
        "mask_path": mask_relative,
        "mask_sha256": (sha256_file(mask_path) if mask_path is not None else None),
        "mask_bytes": (mask_path.stat().st_size if mask_path is not None else None),
        "mask_shape": (list(native_map.shape) if mask_path is not None else None),
        "mask_dtype": "uint8" if mask_path is not None else None,
        "mask_semantics": (
            "strict_probability_greater_than_0_5" if mask_path is not None else None
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
        raise ValueError("PSCC-Net applicable T2 row has no ground truth")
    target_model = legacy._model_target(target_native)
    include_ap = str(row["condition"]) != "real"
    return {
        "model_256": binary_pixel_metrics_strict(
            model_map,
            target_model,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
        "native": binary_pixel_metrics_strict(
            native_map,
            target_native,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
    }


_OK_ONLY_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "preprocess",
        "classification_logits",
        "classification_logits_dtype",
        "classification_logits_sha256",
        "classification_probabilities",
        "classification_probabilities_dtype",
        "classification_probabilities_sha256",
        "ai_score",
        "score_semantics",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "artifact_paths",
        "progressive_maps",
        "primary_model_score_map_path",
        "primary_model_score_map_sha256",
        "primary_model_score_map_bytes",
        "primary_model_score_map_shape",
        "primary_model_score_map_dtype",
        "primary_model_score_map_semantics",
        "score_map_path",
        "score_map_sha256",
        "score_map_bytes",
        "score_map_shape",
        "score_map_dtype",
        "score_map_semantics",
        "score_map_array_sha256",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
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


def _load_probability_map(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_sha256: Any,
    expected_bytes: Any,
    expected_array_sha256: Any,
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
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
    ):
        raise ValueError(f"{label} array contract changed")
    if _array_sha256(array) != expected_array_sha256:
        raise ValueError(f"{label} array SHA-256 changed")
    return array


def _load_binary_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_sha256: Any,
    expected_bytes: Any,
    native_map: np.ndarray,
) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError("PSCC-Net native mask SHA-256 changed")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError("PSCC-Net native mask byte size changed")
    with Image.open(path) as opened:
        if opened.mode != "L" or opened.size != (
            expected_shape[1],
            expected_shape[0],
        ):
            raise ValueError("PSCC-Net native mask image contract changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    if pixels.shape != expected_shape or not np.isin(pixels, (0, 255)).all():
        raise ValueError("PSCC-Net native mask values changed")
    expected = np.where(
        native_map > MASK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)
    if not np.array_equal(pixels, expected):
        raise ValueError("PSCC-Net native mask threshold replay changed")
    return pixels


def _validate_preprocess(
    preprocess: Any,
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    recompute: bool,
) -> None:
    if not isinstance(preprocess, Mapping):
        raise ValueError("PSCC-Net preprocess record is not an object")
    expected_keys = {
        "decoder",
        "channel_order",
        "native_size",
        "input_resize",
        "input_crop",
        "input_reencode",
        "normalization",
        "alpha_policy",
        "tensor_shape",
        "tensor_sha256",
    }
    if set(preprocess) != expected_keys:
        raise ValueError("PSCC-Net preprocess key set changed")
    width = int(row["width"])
    height = int(row["height"])
    if (
        preprocess.get("decoder") != "imageio.v2.imread"
        or preprocess.get("channel_order") != "RGB"
        or preprocess.get("native_size") != [width, height]
        or preprocess.get("input_resize") != "none"
        or preprocess.get("input_crop") is not None
        or preprocess.get("input_reencode") is not False
        or preprocess.get("normalization") != "uint8_rgb_divide_255"
        or preprocess.get("alpha_policy") != "not_applicable"
        or preprocess.get("tensor_shape") != [3, height, width]
        or not isinstance(preprocess.get("tensor_sha256"), str)
        or len(str(preprocess["tensor_sha256"])) != 64
    ):
        raise ValueError("PSCC-Net preprocess record changed")
    if recompute:
        path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _, _, expected = _preprocess_with_audit(path)
        if dict(preprocess) != expected:
            raise ValueError("PSCC-Net preprocess replay changed")


def _validate_score_payload(row: Mapping[str, Any], sample_id: str) -> None:
    logits = np.asarray(row.get("classification_logits"), dtype=np.float32)
    probabilities = np.asarray(
        row.get("classification_probabilities"),
        dtype=np.float32,
    )
    expected = _score_payload(logits, probabilities)
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"PSCC-Net {sample_id} score payload {key} changed")


def _validate_ok_artifacts(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(artifact_root, sample_id)
    path_map = row.get("artifact_paths")
    progressive = row.get("progressive_maps")
    if (
        not isinstance(path_map, Mapping)
        or set(path_map)
        != {
            "progressive_mask1",
            "progressive_mask2",
            "progressive_mask3",
            "progressive_mask4",
            "native_probability",
            "native_mask",
        }
        or not isinstance(progressive, list)
        or len(progressive) != 4
    ):
        raise ValueError("PSCC-Net artifact manifest changed")

    model_maps: list[np.ndarray] = []
    for stage, (raw, expected_shape) in enumerate(
        zip(progressive, PROGRESSIVE_SHAPES, strict=True),
        start=1,
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("PSCC-Net progressive map record is invalid")
        expected_keys = {
            "stage",
            "path",
            "sha256",
            "bytes",
            "shape",
            "dtype",
            "semantics",
            "primary",
            "array_sha256",
        }
        if set(raw) != expected_keys:
            raise ValueError("PSCC-Net progressive map key set changed")
        key = f"progressive_mask{stage}"
        expected_semantics = (
            "official_primary_NLC_sigmoid_probability"
            if stage == 1
            else "official_auxiliary_NLC_sigmoid_probability"
        )
        if (
            raw.get("stage") != stage
            or raw.get("shape") != list(expected_shape)
            or raw.get("dtype") != "float32"
            or raw.get("semantics") != expected_semantics
            or raw.get("primary") is not (stage == 1)
            or path_map.get(key) != raw.get("path")
        ):
            raise ValueError("PSCC-Net progressive map metadata changed")
        path = _resolve_exact_artifact(
            repo_root=repo_root,
            artifact_root=artifact_root,
            value=raw.get("path"),
            expected=paths[key],
            label=f"PSCC-Net stage-{stage} map",
        )
        model_maps.append(
            _load_probability_map(
                path,
                expected_shape=expected_shape,
                expected_sha256=raw.get("sha256"),
                expected_bytes=raw.get("bytes"),
                expected_array_sha256=raw.get("array_sha256"),
                label=f"PSCC-Net stage-{stage} map",
            )
        )

    primary = progressive[0]
    primary_fields = {
        "primary_model_score_map_path": primary["path"],
        "primary_model_score_map_sha256": primary["sha256"],
        "primary_model_score_map_bytes": primary["bytes"],
        "primary_model_score_map_shape": primary["shape"],
        "primary_model_score_map_dtype": "float32",
        "primary_model_score_map_semantics": (
            "official_primary_NLC_sigmoid_probability"
        ),
    }
    for key, value in primary_fields.items():
        if row.get(key) != value:
            raise ValueError(f"PSCC-Net primary map alias {key} changed")

    native_path = _resolve_exact_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        value=row.get("score_map_path"),
        expected=paths["native_probability"],
        label="PSCC-Net native probability map",
    )
    if path_map.get("native_probability") != row.get("score_map_path"):
        raise ValueError("PSCC-Net native artifact path alias changed")
    native_shape = (int(input_row["height"]), int(input_row["width"]))
    if (
        row.get("score_map_shape") != list(native_shape)
        or row.get("score_map_dtype") != "float32"
        or row.get("score_map_semantics")
        != "primary_probability_bilinear_align_corners_true_native_restore"
    ):
        raise ValueError("PSCC-Net native map metadata changed")
    native_map = _load_probability_map(
        native_path,
        expected_shape=native_shape,
        expected_sha256=row.get("score_map_sha256"),
        expected_bytes=row.get("score_map_bytes"),
        expected_array_sha256=row.get("score_map_array_sha256"),
        label="PSCC-Net native probability map",
    )

    applicable, _ = _t2_semantics(input_row)
    if applicable:
        if (
            row.get("mask_shape") != list(native_shape)
            or row.get("mask_dtype") != "uint8"
            or row.get("mask_semantics") != "strict_probability_greater_than_0_5"
            or path_map.get("native_mask") != row.get("mask_path")
        ):
            raise ValueError("PSCC-Net native mask metadata changed")
        mask_path = _resolve_exact_artifact(
            repo_root=repo_root,
            artifact_root=artifact_root,
            value=row.get("mask_path"),
            expected=paths["native_mask"],
            label="PSCC-Net native binary mask",
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
            model_map=model_maps[0],
            native_map=native_map,
        )
        if stable_json(row.get("localization")) != stable_json(expected_localization):
            raise ValueError("PSCC-Net localization replay changed")
    else:
        for key in (
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_shape",
            "mask_dtype",
            "mask_semantics",
            "localization",
        ):
            if row.get(key) is not None:
                raise ValueError(f"PSCC-Net full-frame row claims T2 field {key}")
        if path_map.get("native_mask") is not None:
            raise ValueError("PSCC-Net full-frame row claims a native mask")


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
        raise ValueError("PSCC-Net result status is invalid")
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
        raise ValueError("PSCC-Net result key set changed")
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"PSCC-Net result identity field {key} changed")
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("PSCC-Net result completed_at is invalid")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
        ):
            raise ValueError("PSCC-Net error payload is invalid")
        return
    _validate_preprocess(
        attempt.get("preprocess"),
        row=input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    _validate_score_payload(attempt, str(input_row["sample_id"]))
    if (
        attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError("PSCC-Net mask threshold semantics changed")
    latency = _finite_number(attempt.get("latency_ms"), "latency_ms")
    peak = attempt.get("peak_cuda_memory_bytes")
    if latency < 0.0:
        raise ValueError("PSCC-Net latency is negative")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("PSCC-Net peak CUDA memory is invalid")
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
                f"PSCC-Net history row {line_number} has unexpected sample_id"
            )
        status = attempt.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"PSCC-Net history row {line_number} has invalid status")
        prior = histories.setdefault(sample_id, [])
        if "ok" in prior:
            raise ValueError(
                "PSCC-Net append-only history contains an attempt after success"
            )
        prior.append(str(status))
    return {
        "policy": "zero_or_more_errors_then_at_most_one_terminal_success_per_id",
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
            raise ValueError(f"PSCC-Net artifact root is unsafe: {artifact_root}")
        for entry in artifact_root.iterdir():
            if entry.name not in expected or not entry.is_dir() or entry.is_symlink():
                raise ValueError(
                    "PSCC-Net artifact root has unexpected/unsafe entry: " f"{entry}"
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
            f"missing/unsafe PSCC-Net artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != expected_directories or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("PSCC-Net artifact root inventory mismatch")
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    successful = {
        sample_id
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    expected: dict[str, set[str]] = {
        directory: {f"{sample_id}.npy" for sample_id in successful}
        for directory in ARTIFACT_DIRECTORIES[:-1]
    }
    expected["masks_native"] = {
        f"{sample_id}.png"
        for sample_id in successful
        if _t2_semantics(inputs_by_id[sample_id])[0]
    }
    counts: dict[str, int] = {}
    for directory_name, expected_names in expected.items():
        directory = artifact_root / directory_name
        children = list(directory.iterdir())
        if any(child.is_symlink() or not child.is_file() for child in children):
            raise ValueError(f"PSCC-Net {directory_name} has unsafe/non-file entries")
        actual_names = {child.name for child in children}
        if actual_names != expected_names:
            raise ValueError(
                f"PSCC-Net {directory_name} inventory mismatch: "
                f"missing={sorted(expected_names - actual_names)[:1]}, "
                f"extra={sorted(actual_names - expected_names)[:1]}"
            )
        counts[directory_name] = len(actual_names)
    return counts


def _required_artifact_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Return a conservative artifact bound including the fixed reserve."""

    if not rows:
        return 0
    progressive_per_image = sum(
        height * width * np.dtype(np.float32).itemsize + NPY_HEADER_BYTES
        for height, width in PROGRESSIVE_SHAPES
    )
    progressive = len(rows) * progressive_per_image
    native_maps = sum(
        int(row["width"]) * int(row["height"]) * np.dtype(np.float32).itemsize
        + NPY_HEADER_BYTES
        for row in rows
    )
    native_masks = sum(
        int(row["width"]) * int(row["height"]) + PNG_CONSERVATIVE_OVERHEAD_BYTES
        for row in rows
        if _t2_semantics(row)[0]
    )
    return progressive + native_maps + native_masks + MIN_DISK_RESERVE_BYTES


def _verify_disk_capacity(
    artifact_root: Path,
    pending: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    required = _required_artifact_bytes(pending)
    usage = shutil.disk_usage(artifact_root.parent)
    if usage.free < required:
        raise OSError(
            "insufficient disk for PSCC-Net raw artifacts: "
            f"required={required}, free={usage.free}"
        )
    return {
        "free_bytes_before_inference": int(usage.free),
        "conservative_pending_bytes_plus_reserve": int(required),
        "fixed_reserve_bytes": (MIN_DISK_RESERVE_BYTES if pending else 0),
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
        raise ValueError(f"PSCC-Net run directory is unsafe: {run_dir}")
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
            "PSCC-Net run directory has unexpected entries: "
            f"{sorted(unexpected)[:1]}"
        )
    for entry in entries:
        _reject_symlink_components(
            entry,
            f"PSCC-Net run file {entry.name}",
        )
        if not entry.is_file() or entry.is_symlink():
            raise ValueError(f"PSCC-Net run entry is not a regular file: {entry}")
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
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
            "checkpoint_components": {
                role: dict(contract) for role, contract in legacy.CHECKPOINTS.items()
            },
            "initialization_weight": dict(legacy.INITIALIZATION_WEIGHT),
            "training_manipulations": [
                "authentic",
                "splicing",
                "copy_move",
                "RFR_Net_object_removal_inpainting",
            ],
            "variant": "official_synthetic_pretrained_not_retrained",
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "decode": "imageio.v2.imread",
            "channel_order": "RGB",
            "rgba": "official_float32_white_background_composite",
            "input_resize": None,
            "input_crop": None,
            "input_reencode": False,
            "scale": "uint8_divide_255",
            "normalization_mean_std": None,
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "batch_size": 1,
        },
        "inference": {
            "feature_extractor": "HRNet_W18_small_v2",
            "localization_head": "NLCDetection",
            "classification_head": "DetectionHead",
            "progressive_output_shapes": [list(shape) for shape in PROGRESSIVE_SHAPES],
            "primary_localization_output": "progressive_mask1",
            "localization_outputs_are_already_sigmoid_probabilities": True,
            "second_sigmoid_applied": False,
            "native_restore": ("torch_bilinear_probability_align_corners_true"),
            "classification_output": (
                "float32_softmax_two_class_logits_positive_index_1"
            ),
            "test_time_augmentation": False,
            "ensemble": False,
            "autocast": False,
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
        "--psccnet-root",
        type=Path,
        default=legacy.DEFAULT_PSCCNET_ROOT,
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
        raise ValueError("resume PSCC-Net manifest/config drifted")
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
        raise ValueError("resume PSCC-Net manifest key set changed")
    if _read_jsonl_strict(expected_inputs_path) != list(selected):
        raise ValueError("resume PSCC-Net expected inputs drifted")
    started_at = prior.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume PSCC-Net started_at is invalid")
    completed_at = prior.get("completed_at")
    if status == "running":
        if completed_at is not None:
            raise ValueError("resume running PSCC-Net manifest has completed_at")
    elif not isinstance(completed_at, str) or not completed_at:
        raise ValueError("resume finalized PSCC-Net completed_at is invalid")

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
        raise ValueError("resume PSCC-Net disk preflight changed")

    outputs = prior.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resume PSCC-Net outputs are missing")
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
        raise ValueError("resume PSCC-Net output key set changed")
    for key, value in expected_immutable["outputs"].items():
        if outputs.get(key) != value:
            raise ValueError(f"resume PSCC-Net output {key} changed")

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
                    f"resume finalized PSCC-Net {relative_key} hash changed"
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
            raise ValueError("resume PSCC-Net artifact inventory changed")
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
            raise ValueError("resume PSCC-Net execution record changed")
    return prior, started_at


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    psccnet_root = _unresolved_anchored(args.psccnet_root, repo_root)

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
            psccnet_root=psccnet_root,
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

    requested_results = _unresolved_anchored(args.results_dir, repo_root)
    expected_results = repo_root / DEFAULT_RESULTS_DIR
    _reject_symlink_components(
        requested_results,
        "PSCC-Net results root",
    )
    _reject_symlink_components(
        expected_results,
        "expected PSCC-Net results root",
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
        "PSCC-Net artifacts root",
    )
    _reject_symlink_components(
        expected_artifacts,
        "expected PSCC-Net artifacts root",
    )
    artifacts_root = requested_artifacts.resolve()
    if artifacts_root != expected_artifacts.resolve():
        raise ValueError(f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}")
    run_dir = _safe_child(
        results_root,
        run_id,
        "PSCC-Net run directory",
    )
    artifact_root = _safe_child(
        artifacts_root,
        run_id,
        "PSCC-Net artifact directory",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("PSCC-Net result and artifact directories must be disjoint")
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
        psccnet_root=psccnet_root,
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
        "t2_applicable_images": sum(_t2_semantics(row)[0] for row in selected),
    }
    if prior_manifest is not None and prior_manifest.get("dataset") != dataset_record:
        raise ValueError("resume PSCC-Net dataset envelope changed")
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
    new_successes = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            models, loaded_device = legacy.load_model(
                psccnet_root=psccnet_root.resolve(),
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("PSCC-Net loaded on an unexpected device")
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
                    raise ValueError("canonical PSCC-Net image dimensions changed")
                assert models is not None
                (
                    model_masks,
                    native_map,
                    logits,
                    probabilities,
                    peak_bytes,
                    latency_ms,
                ) = legacy.infer_one(
                    models,
                    device,
                    tensor,
                    native_width=width,
                    native_height=height,
                )
                model_masks = [
                    np.ascontiguousarray(value, dtype=np.float32)
                    for value in model_masks
                ]
                native_map = np.ascontiguousarray(
                    native_map,
                    dtype=np.float32,
                )
                logits = np.ascontiguousarray(logits, dtype=np.float32)
                probabilities = np.ascontiguousarray(
                    probabilities,
                    dtype=np.float32,
                )
                for stage, model_mask in enumerate(model_masks, start=1):
                    legacy._atomic_save_npy(
                        paths[f"progressive_mask{stage}"],
                        model_mask,
                    )
                legacy._atomic_save_npy(
                    paths["native_probability"],
                    native_map,
                )
                applicable, _ = _t2_semantics(input_row)
                mask_path: Path | None = None
                localization: dict[str, Any] | None = None
                if applicable:
                    mask_path = paths["native_mask"]
                    legacy._atomic_save_mask(
                        mask_path,
                        native_map > MASK_THRESHOLD,
                    )
                    localization = _localization_payload(
                        row=input_row,
                        repo_root=repo_root,
                        model_map=model_masks[0],
                        native_map=native_map,
                    )
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **_score_payload(logits, probabilities),
                    **_artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        model_masks=model_masks,
                        native_map=native_map,
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
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_psccnet_balanced.py",
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
        raise RuntimeError("PSCC-Net fail-closed inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
