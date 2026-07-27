#!/usr/bin/env python3
"""Run the official IML-ViT CAT-protocol checkpoint on Balanced250.

IML-ViT is a native pixel localizer without an image-level classification
head.  This adapter therefore selects only the 1,025 T2-applicable inputs
(real275 plus local mouse/cat/trash-can250), never forwards the full-frame
conditions, and never promotes a dense-map statistic to T1.

The audited Mouse-v1 runner remains immutable.  This adapter reuses its
frozen preprocessing, strict official checkpoint load, forward, and
postprocessing primitives while binding them to the v3 Balanced250 result
and resume contracts. Official masks use strict ``> 0.5``; the separately
frozen shared Balanced250 reducer uses ``>= 0.5`` and does not assume the two
operators are equivalent.
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
import re
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_imlvit as legacy
from eval.opensource.balanced_run_contract import (
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    LOCALIZATION_CONDITIONS,
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
from eval.opensource.imlvit_metrics import binary_pixel_metrics_strict


RUN_MANIFEST_SCHEMA = "imlvit_balanced_run_manifest_v3"
RUN_CONFIG_SCHEMA = "imlvit_balanced_run_config_v3"
RUNTIME_SUMMARY_SCHEMA = "imlvit_balanced_runtime_summary_v3"
CPU_PREFLIGHT_SCHEMA = "imlvit_balanced_cpu_preflight_v3"

MODEL_NAME = legacy.MODEL_NAME
MODEL_SLUG = legacy.MODEL_SLUG
MODEL_ARCHITECTURE = (
    "ViT_B16_window_global_attention_simple_feature_pyramid_"
    "one_channel_localization_head"
)
MODEL_TREE = "cfae1470a71de9f146df3c13e994bea41d70624a"
MODEL_GIT_ORIGIN = "https://github.com/SunnyHaze/IML-ViT.git"
PREPROCESS_PROFILE = (
    "paper_section4.1_conditional_longestmaxsize1024_top_left_raw_zero_pad"
)

MODEL_SEED = 42
MASK_THRESHOLD = legacy.MASK_THRESHOLD
MASK_THRESHOLD_OPERATOR = ">"
SHARED_T2_THRESHOLD_OPERATOR = ">="
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/imlvit")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/imlvit")
DEFAULT_FORMAL_RUN_ID = "imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727"
DEFAULT_SMOKE_RUN_ID_A = "imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727"
DEFAULT_SMOKE_RUN_ID_B = "imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727"
DEFAULT_SMOKE_LIMIT = 5

EXPECTED_VENV_ROOT = Path("/root/.cache/claimforge/venvs/imlvit-07dd2be")
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_BYTES = 204
EXPECTED_PYVENV_SHA256 = (
    "f06e00dd6d89b8a185e71db56644088e103d74312b659c12d3299ce1e10d0e19"
)
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/imlvit-balanced-v3-empty"
)
EXPECTED_PACKAGES = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "timm": "1.0.15",
    "albumentations": "1.3.0",
    "opencv-python-headless": "4.10.0.84",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "scikit-learn": "1.5.2",
}

SOURCE_BOUND_FILES: dict[str, tuple[int, str]] = {
    "README.md": (
        12_078,
        "a5356aa04266719cb248723e07aca6db6fcc9d122ab2a08c20ce012d53793764",
    ),
    "Demo.ipynb": (
        1_042_227,
        "98eff570ed0b98ac6f4db99d01ae2b07fb55684efce030f7df334f80cfbac4ba",
    ),
    "iml_vit_model.py": (
        6_438,
        "6d3b4ba1749bbf6b188e68f9657fff961f8a42bce818042fc1ba064fd64b7048",
    ),
    "modules/decoderhead.py": (
        5_433,
        "a3a1ac5c16a16a17ae65567e6670eb97b92f7cb6c56ef665f8ed80316fa3bd69",
    ),
    "modules/window_attention_ViT.py": (
        42_533,
        "382eabf15045420c827c5ad4bd1e44fa32b254b3c1139d6040f32df6a6bcdbde",
    ),
    "utils/iml_transforms.py": (
        8_496,
        "3abfda4ed3777f93db55dc05922c7f745a3fcb72d952409a195f43e1b33b36f2",
    ),
    "utils/evaluation.py": (
        1_715,
        "ac8ada749ffc4acfaf47d67550be642c96291640a5b130971f25a6857a3962aa",
    ),
    "LICENSE": (
        1_068,
        "7bce5d24d372c0abbf618951988ee2dc072e60027c55615f0229bdab0dad73c3",
    ),
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_imlvit_balanced.py",
    "eval/opensource/run_imlvit.py",
    "eval/opensource/imlvit_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/common.py",
)

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
}
FORMAL_IMAGES = 1_025
SMOKE_COUNTS = {condition: DEFAULT_SMOKE_LIMIT for condition in LOCALIZATION_CONDITIONS}
SMOKE_IMAGES = 20

TASK_SCOPE: dict[str, Any] = {
    "primary_task": "T2_native_pixel_localization_only",
    "capability": "local_t2_only",
    "valid_for_t1": False,
    "valid_for_t2": True,
    "native_dense_output": True,
    "separate_image_classification_head": False,
    "map_statistic_promoted_to_t1": False,
    "fullframe_t1": "not_applicable",
    "fullframe_t2": "not_applicable",
}

T2_SPEC: dict[str, Any] = {
    "valid_conditions": list(LOCALIZATION_CONDITIONS),
    "not_selected_conditions": [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ],
    "score_map": {
        "source": (
            "sigmoid_once_after_bilinear_head_logit_restore_then_"
            "valid_content_crop_and_native_bilinear_restore"
        ),
        "space": "native_decoded_pixels",
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "native_binary_mask": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "role": "official_artifact_and_fresh_replay",
        "encoding": "PNG_L_0_or_255",
    },
    "shared_reducer": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": SHARED_T2_THRESHOLD_OPERATOR,
        "role": "frozen_cross_method_balanced250_metrics",
        "operator_equivalent_to_official": False,
        "non_equivalence_policy": (
            "allow_and_report_native_pixels_and_images_exactly_at_threshold"
        ),
    },
    "ground_truth": {
        "real": "all_zero_false_positive_area",
        "local": "exact_diff_local_insertion",
        "fullframe": "not_applicable_and_not_selected",
        "conditioning_box_is_not_ground_truth": True,
    },
    "fullframe_output": {
        "selected": False,
        "forward_performed": False,
        "artifact_saved": False,
        "t1_derived_from_map": False,
        "t2_scored": False,
    },
}

LICENSE_RECORD: dict[str, Any] = {
    "project_code": {
        "spdx": "MIT",
        "path": "LICENSE",
        "sha256": SOURCE_BOUND_FILES["LICENSE"][1],
        "scope": "project_repository_code_only",
        "commercial_use_permission": True,
        "redistribution_permission": True,
    },
    "official_checkpoint": {
        "provider": legacy.CHECKPOINT["provider"],
        "google_drive_file_id": legacy.CHECKPOINT["file_id"],
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
    "scope": "all_and_only_successful_selected_T2_inputs",
    "raw_logits_model_1024": {
        "directory": "raw_logits_model_1024",
        "format": "NumPy_npy_allow_pickle_false",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
    },
    "score_maps_model_1024": {
        "directory": "score_maps_model_1024",
        "format": "NumPy_npy_allow_pickle_false",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "score_maps_native": {
        "directory": "score_maps_native",
        "format": "NumPy_npy_allow_pickle_false",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "masks_native": {
        "directory": "masks_native",
        "format": "PNG",
        "mode": "L",
        "values": [0, 255],
    },
    "fullframe_artifacts": "none_not_selected",
}
ARTIFACT_DIRECTORIES = (
    "raw_logits_model_1024",
    "score_maps_model_1024",
    "score_maps_native",
    "masks_native",
)
NPY_HEADER_BYTES = 128
PNG_CONSERVATIVE_OVERHEAD_BYTES = 4_096
MIN_DISK_RESERVE_BYTES = 2_000_000_000

RESOURCE_EXPECTATION: dict[str, Any] = {
    "observed_mouse_v1_peak_cuda_memory_bytes": 2_921_058_304,
    "observed_mouse_v1_median_forward_latency_ms": 40.11,
    "observed_mouse_v1_artifact_bytes_for_550_images": 8_153_834_700,
    "balanced250_artifact_projection_bytes": 15_200_000_000,
    "formal_runner_projection_minutes": [8, 30],
    "fresh_replay_projection_minutes": [3, 12],
    "recommended_free_cuda_memory_bytes": 4_000_000_000,
}

_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_FORBIDDEN_T1_TOP_LEVEL = frozenset(
    {
        "score",
        "ai_score",
        "image_score",
        "score_source",
        "score_semantics",
        "decision",
        "image_decision",
        "ai_decision",
        "detection",
        "classification",
        "classification_threshold",
        "classification_decision",
        "classification_logits",
        "classification_probabilities",
        "class_probabilities",
        "t1_score",
        "t1_decision",
        "valid_for_t1_score",
        "auroc",
        "roc_auc",
        "average_precision",
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
    if not _RUN_ID.fullmatch(name):
        raise ValueError(f"{label} has an invalid run ID")
    _reject_symlink_components(root, f"{label} root")
    candidate = root / name
    _reject_symlink_components(candidate, label)
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its configured root") from error
    if resolved == resolved_root:
        raise ValueError(f"{label} must be below its configured root")
    return resolved


def _require_exact_path(requested: Path, expected: Path, label: str) -> Path:
    _reject_symlink_components(requested, label)
    _reject_symlink_components(expected, f"expected {label}")
    resolved = requested.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected}")
    return resolved


def _fingerprint(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _fingerprint([dict(row) for row in rows])


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _valid_run_id(value: Any) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError("run_id must match [a-z0-9][a-z0-9._-]{0,127}")
    return value


def verified_input_path(
    row: Mapping[str, Any],
    repo_root: Path,
) -> Path:
    value = row.get("canonical_path")
    sample_id = row.get("sample_id")
    if not isinstance(value, str) or not isinstance(sample_id, str):
        raise ValueError("canonical input identity is incomplete")
    path = Path(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.name != f"{sample_id}.jpg"
    ):
        raise ValueError("canonical input path is unsafe")
    unresolved = repo_root / path
    _reject_symlink_components(unresolved, f"canonical input {sample_id}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("canonical input escapes the repository") from error
    if not unresolved.is_file() or unresolved.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    expected_bytes = row.get("canonical_bytes")
    expected_sha256 = row.get("canonical_sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or candidate.stat().st_size != expected_bytes
        or not isinstance(expected_sha256, str)
        or sha256_file(candidate) != expected_sha256
    ):
        raise ValueError(f"canonical input bytes changed: {sample_id}")
    return candidate


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"JSON contains forbidden non-finite constant {value}")


def _load_json_object_strict(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ValueError(f"{path} line {line_number} lacks LF terminator")
            try:
                text = raw[:-1].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not UTF-8") from error
            if not text or text.strip() != text:
                raise ValueError(
                    f"{path} line {line_number} is empty or has outer whitespace"
                )
            value = json.loads(
                text,
                object_pairs_hook=_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            if stable_json(value) != text:
                raise ValueError(f"{path} line {line_number} is not canonical JSON")
            rows.append(value)
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
    root = repo_root.resolve()
    contract: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        _reject_symlink_components(path, f"adapter source {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing adapter source: {path}")
        contract[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return contract


def verify_artifact_ignore(repo_root: Path) -> dict[str, Any]:
    probe = "outputs/opensource/imlvit/.balanced_contract_probe/artifact.npy"
    process = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-v", probe],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or ".gitignore:" not in process.stdout:
        raise ValueError("IML-ViT artifact path is not covered by .gitignore")
    value = {
        "probe": probe,
        "ignored": True,
        "git_check_ignore_evidence": process.stdout.strip(),
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_environment() -> dict[str, Any]:
    executable = Path(sys.executable)
    if executable != EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError(
            f"IML-ViT interpreter must be exactly {EXPECTED_PYTHON_EXECUTABLE}"
        )
    if Path(sys.prefix).resolve() != EXPECTED_VENV_ROOT.resolve():
        raise ValueError("IML-ViT sys.prefix is not the frozen virtualenv")
    pyvenv = EXPECTED_VENV_ROOT / "pyvenv.cfg"
    if (
        not pyvenv.is_file()
        or pyvenv.is_symlink()
        or pyvenv.stat().st_size != EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("IML-ViT pyvenv.cfg binding changed")
    actual_packages = {name: _package_version(name) for name in EXPECTED_PACKAGES}
    if actual_packages != EXPECTED_PACKAGES:
        raise ValueError(f"IML-ViT package lock changed: {actual_packages!r}")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("PYTHONDONTWRITEBYTECODE must be exactly 1")
    if not sys.dont_write_bytecode:
        raise ValueError("Python bytecode writes are not disabled")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be exactly 0")
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(FROZEN_PYTHONPYCACHEPREFIX):
        raise ValueError("PYTHONPYCACHEPREFIX is not the frozen IML-ViT path")
    if sys.pycache_prefix is None or Path(sys.pycache_prefix) != (
        FROZEN_PYTHONPYCACHEPREFIX
    ):
        raise ValueError("sys.pycache_prefix is not frozen")
    if (
        not FROZEN_PYTHONPYCACHEPREFIX.is_dir()
        or FROZEN_PYTHONPYCACHEPREFIX.is_symlink()
        or any(FROZEN_PYTHONPYCACHEPREFIX.iterdir())
    ):
        raise ValueError("frozen IML-ViT pycache directory is not empty")
    value = {
        "python_executable": str(executable),
        "venv_root": str(EXPECTED_VENV_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "bytes": pyvenv.stat().st_size,
            "sha256": sha256_file(pyvenv),
        },
        "packages": actual_packages,
        "python_hash_seed": 0,
        "python_dont_write_bytecode": True,
        "python_pycache_prefix": str(FROZEN_PYTHONPYCACHEPREFIX),
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_source(imlvit_root: Path) -> dict[str, Any]:
    root = _require_exact_path(
        imlvit_root,
        legacy.DEFAULT_IMLVIT_ROOT,
        "IML-ViT source root",
    )
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    commit = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    origin = _git_value(root, "remote", "get-url", "origin")
    status = _git_value(root, "status", "--short", "--untracked-files=all")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("IML-ViT source commit changed")
    if tree != MODEL_TREE:
        raise ValueError("IML-ViT source tree changed")
    if origin != MODEL_GIT_ORIGIN:
        raise ValueError("IML-ViT source origin changed")
    if status:
        raise ValueError("IML-ViT source checkout is not completely clean")
    bound: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in SOURCE_BOUND_FILES.items():
        path = root / relative
        _reject_symlink_components(path, f"IML-ViT source {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"IML-ViT source binding changed: {relative}")
        bound[relative] = {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    value = {
        "repo_url": legacy.MODEL_REPO_URL,
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bound,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_assets(checkpoint_path: Path) -> dict[str, Any]:
    path = _require_exact_path(
        checkpoint_path,
        legacy.DEFAULT_CHECKPOINT,
        "IML-ViT checkpoint",
    )
    _reject_symlink_components(path, "IML-ViT checkpoint")
    checkpoint = legacy.CHECKPOINT
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(checkpoint["bytes"])
        or sha256_file(path) != checkpoint["sha256"]
    ):
        raise ValueError("official IML-ViT checkpoint binding changed")
    value = {
        "checkpoint": {
            **checkpoint,
            "path": str(path),
            "safe_weights_only_load": True,
            "strict_model_load": True,
            "schema_fallbacks": False,
            "prefix_rewrites": False,
        }
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _checkpoint_schema(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    rows: list[dict[str, Any]] = []
    dtype_counts: Counter[str] = Counter()
    total_elements = 0
    total_bytes = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise ValueError("IML-ViT state is not a string/tensor mapping")
        row = {
            "key": key,
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
            "elements": int(tensor.numel()),
        }
        rows.append(row)
        dtype_counts[str(tensor.dtype)] += 1
        total_elements += int(tensor.numel())
        total_bytes += int(tensor.numel()) * int(tensor.element_size())
    return {
        "state_keys": len(rows),
        "state_elements": total_elements,
        "tensor_bytes": total_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": _fingerprint([row["key"] for row in rows]),
        "tensor_schema_sha256": _fingerprint(rows),
    }


def _safe_checkpoint_audit(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict" or not isinstance(state, Mapping):
        raise ValueError("IML-ViT checkpoint is not the frozen raw OrderedDict")
    audit = _checkpoint_schema(state)
    checkpoint = legacy.CHECKPOINT
    expected = {
        "state_keys": int(checkpoint["state_keys"]),
        "state_elements": int(checkpoint["state_elements"]),
        "tensor_bytes": int(checkpoint["tensor_bytes"]),
        "dtype_counts": dict(checkpoint["state_dtypes"]),
    }
    for key, expected_value in expected.items():
        if audit[key] != expected_value:
            raise ValueError(f"IML-ViT checkpoint {key} changed")
    del state
    gc.collect()
    return {
        "load": "torch.load_weights_only_true",
        "outer_type": "collections.OrderedDict",
        **audit,
    }


def run_cpu_preflight(
    *,
    repo_root: Path,
    imlvit_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Strict-load all frozen assets on CPU without a model forward."""

    environment = verify_environment()
    source = verify_source(imlvit_root)
    assets = verify_assets(checkpoint_path)
    adapter_sources = adapter_source_contract(repo_root)
    artifact_ignore = verify_artifact_ignore(repo_root)

    import torch

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise RuntimeError("CUDA was initialized before IML-ViT CPU preflight")
    checkpoint_audit = _safe_checkpoint_audit(checkpoint_path)
    if torch.cuda.is_initialized():
        raise RuntimeError("IML-ViT checkpoint audit initialized CUDA")
    model, device = legacy.load_model(
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )
    if str(device) != "cpu":
        raise ValueError("IML-ViT CPU preflight loaded on a non-CPU device")
    parameters = sum(int(value.numel()) for value in model.parameters())
    buffers = sum(int(value.numel()) for value in model.buffers())
    modules = sum(1 for _ in model.modules())
    state_keys = len(model.state_dict())
    if parameters != int(legacy.CHECKPOINT["parameters"]):
        raise ValueError("IML-ViT parameter count changed")
    if buffers != int(legacy.CHECKPOINT["buffers"]):
        raise ValueError("IML-ViT buffer count changed")
    if state_keys != int(legacy.CHECKPOINT["state_keys"]):
        raise ValueError("strict-loaded IML-ViT state schema changed")
    if model.training:
        raise ValueError("strict-loaded IML-ViT model is not in eval mode")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("IML-ViT CPU preflight model has non-CPU parameters")
    model_audit = {
        "strict_load": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "parameter_count": parameters,
        "buffer_elements": buffers,
        "module_count": modules,
        "state_keys": state_keys,
        "device": "cpu",
        "eval_mode": True,
        "forward_performed": False,
    }
    del model
    gc.collect()
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after:
        raise RuntimeError("IML-ViT CPU preflight initialized CUDA")
    value: dict[str, Any] = {
        "schema_version": CPU_PREFLIGHT_SCHEMA,
        "environment": environment,
        "source": source,
        "assets": assets,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "adapter_sources": adapter_sources,
        "artifact_ignore": artifact_ignore,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "balanced250_forward_performed": False,
        "balanced250_score_computed": False,
        "t1_output_computed": False,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    if device_text != "cpu" and (
        not device_text.startswith("cuda:")
        or not device_text[5:].isdigit()
        or str(int(device_text[5:])) != device_text[5:]
    ):
        raise ValueError("device must be exactly cpu or cuda:N")
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with frozen runtime")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "IML-ViT accelerator was initialized before runtime configuration"
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
    spec = SelectionSpec(capability=Capability.LOCAL_T2_ONLY)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    expected = [
        row for row in release.inputs if row.get("condition") in LOCALIZATION_CONDITIONS
    ]
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != FORMAL_IMAGES
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in expected]
    ):
        raise ValueError("formal IML-ViT Balanced250 selection drifted")
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
        capability=Capability.LOCAL_T2_ONLY,
        per_condition_limit=per_condition_limit,
    )
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        len(selected) != SMOKE_IMAGES
        or dict(counts) != SMOKE_COUNTS
        or any(row.get("panel") is not True for row in selected)
        or any(str(row["condition"]).startswith("fullframe_") for row in selected)
    ):
        raise ValueError("IML-ViT smoke must be exactly 5x4 applicable rows")
    inputs_by_id = {str(row["sample_id"]): row for row in release.inputs}
    expected_ids: list[str] = []
    expected_counts: Counter[str] = Counter()
    for row in release.panel:
        condition = str(row["condition"])
        if condition not in LOCALIZATION_CONDITIONS:
            continue
        if expected_counts[condition] >= per_condition_limit:
            continue
        expected_ids.append(str(row["sample_id"]))
        expected_counts[condition] += 1
    expected_ids.sort(key=lambda sample_id: int(inputs_by_id[sample_id]["rank"]))
    if [str(row["sample_id"]) for row in selected] != expected_ids:
        raise ValueError("IML-ViT smoke panel-priority selection drifted")
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
            raise ValueError("formal mode accepts no selection options")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None or per_condition_limit is None:
            raise ValueError(
                "smoke mode requires --per-condition-limit 5 and no sample ID"
            )
        return _smoke_selection(release, per_condition_limit)
    if mode == "single":
        if per_condition_limit is not None or sample_id is None:
            raise ValueError(
                "single mode requires --sample-id and no per-condition limit"
            )
        spec = SelectionSpec(
            capability=Capability.LOCAL_T2_ONLY,
            sample_id=sample_id,
        )
        selected = select_inputs(release, spec)
        if len(selected) != 1:
            raise ValueError("single mode must select exactly one applicable row")
        return spec, selected
    raise ValueError(f"unsupported run mode: {mode!r}")


def _t2_semantics(row: Mapping[str, Any]) -> str:
    condition = str(row.get("condition"))
    kind = str(row.get("gt_mask_kind"))
    if condition == "real" and kind == "all_zero":
        return "all_zero_real_false_positive_area"
    if condition in FORMAL_COUNTS and condition != "real" and kind == "exact_diff":
        return "exact_diff_local_insertion"
    raise ValueError("selected IML-ViT row violates T2-only semantics")


def result_task_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "t2_target_semantics": _t2_semantics(row),
        "map_statistic_promoted_to_t1": False,
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    identity = build_result_identity(
        row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
    )
    return {
        **identity,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "valid_for_metrics": valid_for_metrics,
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "task_scope": result_task_scope(row),
    }


def _preprocess_with_audit(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], dict[str, Any]]:
    tensor, native_size, resized_size, metadata = legacy.preprocess_image(path)
    width, height = (int(value) for value in native_size)
    resized_width, resized_height = (int(value) for value in resized_size)
    if (
        tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or tensor.shape != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
    ):
        raise ValueError("IML-ViT preprocessed tensor contract changed")
    if (
        width <= 0
        or height <= 0
        or resized_width <= 0
        or resized_height <= 0
        or resized_width > legacy.MODEL_INPUT_SIZE
        or resized_height > legacy.MODEL_INPUT_SIZE
    ):
        raise ValueError("IML-ViT preprocessing dimensions are invalid")
    if metadata.get("native_size") != [width, height]:
        raise ValueError("IML-ViT native preprocessing size changed")
    if metadata.get("resized_content_size") != [
        resized_width,
        resized_height,
    ]:
        raise ValueError("IML-ViT resized preprocessing size changed")
    if metadata.get("model_canvas_size") != [
        legacy.MODEL_INPUT_SIZE,
        legacy.MODEL_INPUT_SIZE,
    ]:
        raise ValueError("IML-ViT model canvas changed")
    expected_policy = (
        "albumentations_longest_max_size_downscale_only"
        if max(width, height) > legacy.MODEL_INPUT_SIZE
        else "none_image_within_1024_limit"
    )
    if metadata.get("resize_policy") != expected_policy:
        raise ValueError("IML-ViT conditional resize policy changed")
    if metadata.get("tensor_sha256") != _array_sha256(tensor):
        raise ValueError("IML-ViT preprocessing tensor hash changed")
    return tensor, native_size, resized_size, metadata


def artifact_paths(artifact_root: Path, sample_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[0-9a-f]{24}", sample_id):
        raise ValueError("IML-ViT sample_id is not a frozen 24-hex ID")
    return {
        "raw_logits_model": (
            artifact_root / "raw_logits_model_1024" / f"{sample_id}.npy"
        ),
        "score_map_model": (
            artifact_root / "score_maps_model_1024" / f"{sample_id}.npy"
        ),
        "score_map_native": (artifact_root / "score_maps_native" / f"{sample_id}.npy"),
        "mask_native": artifact_root / "masks_native" / f"{sample_id}.png",
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    raw_logits_model: np.ndarray,
    score_map_model: np.ndarray,
    score_map_native: np.ndarray,
    resized_size: tuple[int, int],
) -> dict[str, Any]:
    return {
        "raw_logits_model_path": repo_relative(paths["raw_logits_model"], repo_root),
        "raw_logits_model_sha256": sha256_file(paths["raw_logits_model"]),
        "raw_logits_model_shape": list(raw_logits_model.shape),
        "raw_logits_model_dtype": str(raw_logits_model.dtype),
        "score_map_model_path": repo_relative(paths["score_map_model"], repo_root),
        "score_map_model_sha256": sha256_file(paths["score_map_model"]),
        "score_map_model_shape": list(score_map_model.shape),
        "score_map_model_dtype": str(score_map_model.dtype),
        "model_valid_content_size": [
            int(resized_size[0]),
            int(resized_size[1]),
        ],
        "score_map_path": repo_relative(paths["score_map_native"], repo_root),
        "score_map_sha256": sha256_file(paths["score_map_native"]),
        "score_map_shape": list(score_map_native.shape),
        "score_map_dtype": str(score_map_native.dtype),
        "mask_path": repo_relative(paths["mask_native"], repo_root),
        "mask_sha256": sha256_file(paths["mask_native"]),
        "mask_shape": list(score_map_native.shape),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _resolve_exact_artifact(
    value: Any,
    *,
    repo_root: Path,
    expected: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    pure = Path(value)
    if pure.is_absolute() or "\\" in value or ".." in pure.parts:
        raise ValueError(f"{label} path is absolute or traversing")
    unresolved = repo_root / pure
    _reject_symlink_components(unresolved, label)
    _reject_symlink_components(expected, f"expected {label}")
    candidate = unresolved.resolve()
    if candidate != expected.resolve():
        raise ValueError(f"{label} path is not canonical")
    if not unresolved.is_file() or unresolved.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_npy_artifact(
    path: Path,
    *,
    expected_sha256: Any,
    expected_shape: tuple[int, ...],
    label: str,
    probability: bool,
) -> np.ndarray:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} hash changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a safe NumPy artifact") from error
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.float32
        or array.shape != expected_shape
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} array contract changed")
    if probability and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError(f"{label} probability range changed")
    return array


def _validate_preprocess(
    metadata: Any,
    *,
    input_row: Mapping[str, Any],
) -> tuple[int, int]:
    if not isinstance(metadata, Mapping):
        raise ValueError("IML-ViT preprocess record is not an object")
    width = int(input_row["width"])
    height = int(input_row["height"])
    native = metadata.get("native_size")
    resized = metadata.get("resized_content_size")
    if native != [width, height]:
        raise ValueError("IML-ViT preprocess native size drifted")
    if (
        not isinstance(resized, list)
        or len(resized) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in resized
        )
    ):
        raise ValueError("IML-ViT preprocess resized size is invalid")
    resized_width, resized_height = (int(value) for value in resized)
    if not (
        0 < resized_width <= legacy.MODEL_INPUT_SIZE
        and 0 < resized_height <= legacy.MODEL_INPUT_SIZE
        and max(resized_width, resized_height) <= legacy.MODEL_INPUT_SIZE
    ):
        raise ValueError("IML-ViT preprocess resized size is out of range")
    expected_policy = (
        "albumentations_longest_max_size_downscale_only"
        if max(width, height) > legacy.MODEL_INPUT_SIZE
        else "none_image_within_1024_limit"
    )
    if metadata.get("resize_policy") != expected_policy:
        raise ValueError("IML-ViT preprocess resize policy drifted")
    digest = metadata.get("tensor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("IML-ViT preprocess tensor hash is invalid")
    return resized_width, resized_height


def _validate_ok_artifacts(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
    artifact_root: Path,
    recompute_preprocess: bool,
) -> None:
    sample_id = str(input_row["sample_id"])
    width = int(input_row["width"])
    height = int(input_row["height"])
    paths = artifact_paths(artifact_root, sample_id)
    raw_path = _resolve_exact_artifact(
        row.get("raw_logits_model_path"),
        repo_root=repo_root,
        expected=paths["raw_logits_model"],
        label="raw model logits",
    )
    model_score_path = _resolve_exact_artifact(
        row.get("score_map_model_path"),
        repo_root=repo_root,
        expected=paths["score_map_model"],
        label="model score map",
    )
    native_score_path = _resolve_exact_artifact(
        row.get("score_map_path"),
        repo_root=repo_root,
        expected=paths["score_map_native"],
        label="native score map",
    )
    mask_path = _resolve_exact_artifact(
        row.get("mask_path"),
        repo_root=repo_root,
        expected=paths["mask_native"],
        label="native mask",
    )
    model_shape = (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
    raw = _load_npy_artifact(
        raw_path,
        expected_sha256=row.get("raw_logits_model_sha256"),
        expected_shape=model_shape,
        label="raw model logits",
        probability=False,
    )
    model_score = _load_npy_artifact(
        model_score_path,
        expected_sha256=row.get("score_map_model_sha256"),
        expected_shape=model_shape,
        label="model score map",
        probability=True,
    )
    native_score = _load_npy_artifact(
        native_score_path,
        expected_sha256=row.get("score_map_sha256"),
        expected_shape=(height, width),
        label="native score map",
        probability=True,
    )
    if row.get("raw_logits_model_shape") != list(model_shape):
        raise ValueError("recorded raw-logit shape changed")
    if row.get("score_map_model_shape") != list(model_shape):
        raise ValueError("recorded model-score shape changed")
    if row.get("score_map_shape") != [height, width]:
        raise ValueError("recorded native-score shape changed")
    if row.get("mask_shape") != [height, width]:
        raise ValueError("recorded mask shape changed")
    for key in (
        "raw_logits_model_dtype",
        "score_map_model_dtype",
        "score_map_dtype",
    ):
        if row.get(key) != "float32":
            raise ValueError(f"recorded {key} changed")
    resized_width, resized_height = _validate_preprocess(
        row.get("preprocess"),
        input_row=input_row,
    )
    if row.get("model_valid_content_size") != [
        resized_width,
        resized_height,
    ]:
        raise ValueError("model valid-content size changed")
    if row.get("mask_threshold") != MASK_THRESHOLD:
        raise ValueError("IML-ViT mask threshold changed")
    if row.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR:
        raise ValueError("IML-ViT mask threshold operator changed")

    with Image.open(mask_path) as opened:
        if opened.mode != "L" or opened.size != (width, height):
            raise ValueError("IML-ViT native mask image contract changed")
        mask = np.asarray(opened, dtype=np.uint8)
    if set(np.unique(mask).tolist()) - {0, 255}:
        raise ValueError("IML-ViT native mask is not binary")
    expected_mask = np.where(native_score > MASK_THRESHOLD, np.uint8(255), np.uint8(0))
    if not np.array_equal(mask, expected_mask):
        raise ValueError("IML-ViT native mask/probability relation changed")
    if row.get("mask_sha256") != sha256_file(mask_path):
        raise ValueError("IML-ViT native mask hash changed")

    # This is an artifact relation, not a fresh model replay.  CPU sigmoid is
    # compared with the recorded model probability using the independently
    # frozen tolerance already justified by the Mouse audit.
    positive = raw >= np.float32(0.0)
    sigmoid = np.empty_like(raw)
    sigmoid[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-raw[positive], dtype=np.float32)
    )
    exponential = np.exp(raw[~positive], dtype=np.float32)
    sigmoid[~positive] = exponential / (np.float32(1.0) + exponential)
    if not np.allclose(
        sigmoid,
        model_score,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("IML-ViT raw-logit sigmoid artifact relation changed")

    target = load_ground_truth(input_row, repo_root)
    if target is None:
        raise ValueError("selected IML-ViT row has no applicable T2 target")
    target_native = np.asarray(target, dtype=bool)
    target_model = legacy.model_space_target(
        target_native,
        resized_width=resized_width,
        resized_height=resized_height,
    )
    expected_localization = {
        "model_1024": binary_pixel_metrics_strict(
            model_score[:resized_height, :resized_width],
            target_model,
            MASK_THRESHOLD,
            include_ap=str(input_row["condition"]) != "real",
        ),
        "native": binary_pixel_metrics_strict(
            native_score,
            target_native,
            MASK_THRESHOLD,
            include_ap=str(input_row["condition"]) != "real",
        ),
    }
    if stable_json(row.get("localization")) != stable_json(expected_localization):
        raise ValueError("IML-ViT recorded localization changed")

    if recompute_preprocess:
        image_path = verified_input_path(input_row, repo_root)
        tensor, _, _, metadata = _preprocess_with_audit(image_path)
        del tensor
        if stable_json(metadata) != stable_json(row.get("preprocess")):
            raise ValueError("IML-ViT preprocessing replay changed")


def _validate_runner_attempt(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    verify_artifacts: bool,
    recompute_preprocess: bool = False,
) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("IML-ViT result attempt is not an object")
    forbidden = sorted(_FORBIDDEN_T1_TOP_LEVEL.intersection(row))
    if forbidden:
        raise ValueError(f"IML-ViT result contains forbidden T1 fields: {forbidden}")
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError("IML-ViT result status is invalid")
    expected = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(f"IML-ViT result identity drifted at {key}")
    if row.get("valid_for_t1") is not False:
        raise ValueError("IML-ViT result cannot be valid for T1")
    if row.get("t2_applicable") is not True:
        raise ValueError("IML-ViT selected result must be T2-applicable")
    if status == "error":
        for key in (
            "raw_logits_model_path",
            "score_map_model_path",
            "score_map_path",
            "mask_path",
            "localization",
            "latency_ms",
            "peak_cuda_memory_bytes",
        ):
            if key in row:
                raise ValueError(f"IML-ViT error attempt contains {key}")
        if not isinstance(row.get("error_type"), str) or not isinstance(
            row.get("error"), str
        ):
            raise ValueError("IML-ViT error attempt lacks error metadata")
        return
    _finite_number(row.get("latency_ms"), "latency_ms")
    peak = row.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("peak_cuda_memory_bytes is invalid")
    if verify_artifacts:
        _validate_ok_artifacts(
            row,
            input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
            recompute_preprocess=recompute_preprocess,
        )


def _validate_physical_attempt_history(
    selected: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {str(row["sample_id"]) for row in selected}
    histories: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected:
            raise ValueError(f"result row {index} has unexpected sample_id")
        histories.setdefault(sample_id, []).append(str(row.get("status")))
    recovered = 0
    for sample_id, statuses in histories.items():
        if any(status not in ("ok", "error") for status in statuses):
            raise ValueError(f"invalid attempt status history for {sample_id}")
        if "ok" in statuses[:-1]:
            raise ValueError(f"attempt appended after success for {sample_id}")
        if statuses.count("ok") > 1:
            raise ValueError(f"multiple successful attempts for {sample_id}")
        recovered += int(statuses[-1] == "ok" and "error" in statuses[:-1])
    return {
        "physical_attempts": len(rows),
        "unique_sample_ids": len(histories),
        "superseded_attempts": len(rows) - len(histories),
        "recovered_error_to_ok": recovered,
        "success_is_terminal": True,
        "append_only": True,
    }


def _prepare_artifact_root(artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    for directory in ARTIFACT_DIRECTORIES:
        path = artifact_root / directory
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"invalid IML-ViT artifact directory: {path}")
        path.mkdir()


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ValueError("IML-ViT artifact root is missing or invalid")
    actual_directories = {
        path.name
        for path in artifact_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    non_directories = [
        path.name
        for path in artifact_root.iterdir()
        if not path.is_dir() or path.is_symlink()
    ]
    if actual_directories != set(ARTIFACT_DIRECTORIES) or non_directories:
        raise ValueError("IML-ViT artifact root inventory mismatch")
    successful = {
        str(row["sample_id"])
        for row in selected
        if latest_by_sample_id.get(str(row["sample_id"]), {}).get("status") == "ok"
    }
    expected = {
        "raw_logits_model_1024": {f"{sample_id}.npy" for sample_id in successful},
        "score_maps_model_1024": {f"{sample_id}.npy" for sample_id in successful},
        "score_maps_native": {f"{sample_id}.npy" for sample_id in successful},
        "masks_native": {f"{sample_id}.png" for sample_id in successful},
    }
    actual: dict[str, set[str]] = {}
    bytes_by_directory: dict[str, int] = {}
    for directory in ARTIFACT_DIRECTORIES:
        root = artifact_root / directory
        entries = list(root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError(f"IML-ViT {directory} contains a non-regular file")
        actual[directory] = {path.name for path in entries}
        bytes_by_directory[directory] = sum(path.stat().st_size for path in entries)
        if actual[directory] != expected[directory]:
            raise ValueError(f"IML-ViT {directory} inventory mismatch")
    value = {
        "successful_images": len(successful),
        "files": sum(len(names) for names in actual.values()),
        "files_by_directory": {
            directory: len(actual[directory]) for directory in ARTIFACT_DIRECTORIES
        },
        "bytes_by_directory": bytes_by_directory,
        "total_bytes": sum(bytes_by_directory.values()),
        "exact_inventory": True,
    }
    return {**value, "inventory_sha256": _fingerprint(value)}


def _required_artifact_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    model_pixels = legacy.MODEL_INPUT_SIZE * legacy.MODEL_INPUT_SIZE
    total = 0
    for row in rows:
        native_pixels = int(row["width"]) * int(row["height"])
        total += 2 * model_pixels * 4
        total += native_pixels * 4
        total += 3 * NPY_HEADER_BYTES
        total += native_pixels + PNG_CONSERVATIVE_OVERHEAD_BYTES
    return total


def _verify_disk_capacity(
    artifact_root: Path,
    pending: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = _required_artifact_bytes(pending)
    stat = os.statvfs(artifact_root)
    available = int(stat.f_bavail * stat.f_frsize)
    minimum = required + MIN_DISK_RESERVE_BYTES
    if available < minimum:
        raise RuntimeError(
            f"insufficient IML-ViT artifact space: {available} < {minimum}"
        )
    return {
        "pending_images": len(pending),
        "estimated_artifact_bytes": required,
        "reserve_bytes": MIN_DISK_RESERVE_BYTES,
        "required_available_bytes": minimum,
        "available_bytes": available,
        "passed": True,
    }


def _validate_run_directory_safety(run_dir: Path, *, resume: bool) -> None:
    _reject_symlink_components(run_dir, "IML-ViT run directory")
    if not run_dir.exists():
        return
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("IML-ViT run directory is not a regular directory")
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_audit.json",
        "metrics.json",
        "fresh_replay.json",
    }
    unexpected = {path.name for path in run_dir.iterdir()} - allowed
    if unexpected:
        raise ValueError(
            "IML-ViT run directory contains unexpected entries: "
            f"{sorted(unexpected)}"
        )
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"IML-ViT run directory is non-empty; pass --resume: {run_dir}"
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
        "dataset_contract": dict(dataset_contract),
        "selection": {
            "selected_images": len(selected),
            "selected_ids_sha256": _fingerprint(
                [str(row["sample_id"]) for row in selected]
            ),
            "selected_rows_sha256": _rows_sha256(selected),
            "counts_by_condition": dict(
                sorted(Counter(str(row["condition"]) for row in selected).items())
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repo_url": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_tree": MODEL_TREE,
            "variant": "official_CAT_TruFor_protocol_checkpoint_20231104",
            "checkpoint_filename_note": (
                "the author release filename says trufor; the README "
                "identifies it as the CAT-Net protocol checkpoint and notes "
                "that TruFor follows the same protocol"
            ),
            "checkpoint": dict(legacy.CHECKPOINT),
            "checkpoint_strict_load": True,
            "checkpoint_safe_weights_only_load": True,
            "license": LICENSE_RECORD,
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "Pillow.Image.open.convert_RGB",
            "channel_order": "RGB",
            "conditional_resize": (
                "if_max_dimension_gt_1024_albumentations_LongestMaxSize_"
                "cv2_INTER_LINEAR"
            ),
            "small_image_resize": "none",
            "canvas": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
            "placement": "top_left",
            "padding": "right_bottom_raw_RGB_zero_before_normalization",
            "normalization": {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "input_crop": None,
            "input_reencode": False,
        },
        "inference": {
            "raw_output": "one_channel_head_logits_at_256x256",
            "model_logit_restore": ("bilinear_to_1024x1024_align_corners_false"),
            "model_probability": "single_sigmoid_after_logit_restore",
            "native_restore": (
                "crop_right_bottom_padding_then_bilinear_probability_to_"
                "native_align_corners_false"
            ),
            "precision": "float32",
            "batch_size": 1,
            "mask_threshold": MASK_THRESHOLD,
            "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
            "t1_policy": "unsupported_no_derived_image_score",
        },
        "task_scope": TASK_SCOPE,
        "t2_spec": T2_SPEC,
        "score_spec": None,
        "artifact_contract": ARTIFACT_CONTRACT,
        "resource_expectation": RESOURCE_EXPECTATION,
        "cpu_preflight": dict(cpu_preflight),
        "runtime": dict(runtime),
        "adapter_sources": adapter_source_contract(repo_root),
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(expected_inputs_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_root": repo_relative(artifact_root, repo_root),
        },
    }


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.mode == "formal":
        expected = DEFAULT_FORMAL_RUN_ID
        if args.run_id not in (None, expected):
            raise ValueError(f"formal run ID is frozen at {expected}")
        return expected
    if args.mode == "smoke":
        if args.run_id not in (DEFAULT_SMOKE_RUN_ID_A, DEFAULT_SMOKE_RUN_ID_B):
            raise ValueError("smoke run ID must be the frozen A or B deterministic ID")
        return str(args.run_id)
    if args.mode == "single":
        if args.run_id is None:
            raise ValueError("single mode requires --run-id")
        if args.run_id in {
            DEFAULT_FORMAL_RUN_ID,
            DEFAULT_SMOKE_RUN_ID_A,
            DEFAULT_SMOKE_RUN_ID_B,
        }:
            raise ValueError("single mode cannot reuse a frozen run ID")
        return _valid_run_id(args.run_id)
    raise ValueError(f"run ID is unsupported for mode {args.mode!r}")


def _prior_manifest_started_at(
    *,
    manifest_path: Path,
    expected_immutable: Mapping[str, Any],
    expected_fingerprint: str,
    expected_inputs_path: Path,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    prior = _load_json_object_strict(manifest_path)
    if prior.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("resume IML-ViT manifest schema changed")
    if prior.get("fingerprint") != expected_fingerprint:
        raise ValueError("resume IML-ViT manifest fingerprint changed")
    if stable_json(prior.get("immutable")) != stable_json(dict(expected_immutable)):
        raise ValueError("resume IML-ViT immutable run config changed")
    started_at = prior.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume IML-ViT manifest started_at is invalid")
    rows = _read_jsonl_strict(expected_inputs_path)
    if stable_json(rows) != stable_json(list(selected)):
        raise ValueError("resume IML-ViT expected input rows changed")
    return prior, started_at


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "formal", "single"),
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--imlvit-root",
        type=Path,
        default=legacy.DEFAULT_IMLVIT_ROOT,
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
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    dataset_manifest_path = _require_exact_path(
        _unresolved_anchored(args.dataset_manifest, repo_root),
        repo_root / DEFAULT_DATASET_MANIFEST,
        "Balanced250 dataset manifest",
    )
    imlvit_root = _require_exact_path(
        args.imlvit_root,
        legacy.DEFAULT_IMLVIT_ROOT,
        "IML-ViT source root",
    )
    checkpoint_path = _require_exact_path(
        args.checkpoint,
        legacy.DEFAULT_CHECKPOINT,
        "IML-ViT checkpoint",
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
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
        )
        print(
            json.dumps(
                {
                    **report,
                    "dataset": {
                        "schema_version": release.schema_version,
                        "dataset_id": release.dataset_id,
                        "manifest_path": repo_relative(
                            dataset_manifest_path, repo_root
                        ),
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
        score_spec=None,
    )
    if (
        dataset_contract.capability.name != "local_t2_only"
        or dataset_contract.capability.valid_for_t1
        or not dataset_contract.capability.valid_for_t2
        or dataset_contract.score_spec is not None
    ):
        raise ValueError("IML-ViT dataset capability binding changed")

    results_root = _require_exact_path(
        _unresolved_anchored(args.results_dir, repo_root),
        repo_root / DEFAULT_RESULTS_DIR,
        "IML-ViT results root",
    )
    artifacts_root = _require_exact_path(
        _unresolved_anchored(args.artifacts_dir, repo_root),
        repo_root / DEFAULT_ARTIFACTS_DIR,
        "IML-ViT artifacts root",
    )
    run_dir = _safe_child(results_root, run_id, "IML-ViT run directory")
    artifact_root = _safe_child(artifacts_root, run_id, "IML-ViT artifact directory")
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("IML-ViT result and artifact roots must be disjoint")
    _validate_run_directory_safety(run_dir, resume=args.resume)
    if artifact_root.exists():
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise ValueError("IML-ViT artifact root is invalid")
        if any(artifact_root.iterdir()) and not args.resume:
            raise FileExistsError(
                f"artifact directory is non-empty; pass --resume: {artifact_root}"
            )

    # The full CPU strict-load gate intentionally precedes every accelerator
    # configuration call and every output-directory mutation.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    device, runtime = configure_runtime(args.device or "cuda:0")

    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
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
            manifest_path=manifest_path,
            expected_immutable=immutable,
            expected_fingerprint=fingerprint,
            expected_inputs_path=expected_path,
            selected=selected,
        )
    else:
        prior_manifest = None
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(expected_path, list(selected))
        started_at = utc_now()

    dataset_record = {
        "contract": dataset_contract.as_dict(),
        "manifest_path": repo_relative(dataset_manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t1_applicable_images": 0,
        "t2_applicable_images": len(selected),
        "fullframe_selected_images": 0,
    }
    if prior_manifest is not None and prior_manifest.get("dataset") != dataset_record:
        raise ValueError("resume IML-ViT dataset envelope changed")
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
        score_spec=None,
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
        if latest_before.latest_by_sample_id.get(str(row["sample_id"]), {}).get(
            "status"
        )
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

    model = None
    loaded_device = device
    new_successes = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            model, loaded_device = legacy.load_model(
                imlvit_root=imlvit_root,
                checkpoint_path=checkpoint_path,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("IML-ViT loaded on an unexpected device")
        for index, input_row in enumerate(pending, start=1):
            sample_id = str(input_row["sample_id"])
            paths = artifact_paths(artifact_root, sample_id)
            try:
                input_path = verified_input_path(input_row, repo_root)
                (
                    image,
                    native_size,
                    resized_size,
                    preprocess,
                ) = _preprocess_with_audit(input_path)
                width, height = native_size
                resized_width, resized_height = resized_size
                if (width, height) != (
                    int(input_row["width"]),
                    int(input_row["height"]),
                ):
                    raise ValueError("canonical IML-ViT dimensions changed")
                assert model is not None
                (
                    raw_logits,
                    score_map_model,
                    score_map_native,
                    peak_bytes,
                    latency_ms,
                ) = legacy.infer_one(
                    model,
                    loaded_device,
                    image,
                    native_width=width,
                    native_height=height,
                    resized_width=resized_width,
                    resized_height=resized_height,
                )
                raw_logits = np.ascontiguousarray(raw_logits, dtype=np.float32)
                score_map_model = np.ascontiguousarray(
                    score_map_model, dtype=np.float32
                )
                score_map_native = np.ascontiguousarray(
                    score_map_native, dtype=np.float32
                )
                target = load_ground_truth(input_row, repo_root)
                if target is None:
                    raise ValueError(
                        "IML-ViT selected input has no applicable T2 target"
                    )
                target_native = np.asarray(target, dtype=bool)
                target_model = legacy.model_space_target(
                    target_native,
                    resized_width=resized_width,
                    resized_height=resized_height,
                )
                include_ap = str(input_row["condition"]) != "real"
                localization = {
                    "model_1024": binary_pixel_metrics_strict(
                        score_map_model[:resized_height, :resized_width],
                        target_model,
                        MASK_THRESHOLD,
                        include_ap=include_ap,
                    ),
                    "native": binary_pixel_metrics_strict(
                        score_map_native,
                        target_native,
                        MASK_THRESHOLD,
                        include_ap=include_ap,
                    ),
                }
                legacy._atomic_save_npy(paths["raw_logits_model"], raw_logits)
                legacy._atomic_save_npy(paths["score_map_model"], score_map_model)
                legacy._atomic_save_npy(paths["score_map_native"], score_map_native)
                legacy._atomic_save_mask(
                    paths["mask_native"],
                    score_map_native > MASK_THRESHOLD,
                )
                result = {
                    **result_identity(
                        input_row,
                        run_id=run_id,
                        run_manifest_fingerprint=fingerprint,
                        valid_for_metrics=True,
                    ),
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **_artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        raw_logits_model=raw_logits,
                        score_map_model=score_map_model,
                        score_map_native=score_map_native,
                        resized_size=resized_size,
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
                    f"t2=yes latency_ms={float(latency_ms):.3f}",
                    flush=True,
                )
            except Exception as error:
                for path in paths.values():
                    path.unlink(missing_ok=True)
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
                new_errors += 1
                fatal_error = error
                print(
                    f"[{index}/{len(pending)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    physical_results = (
        _read_jsonl_strict(results_path) if results_path.is_file() else []
    )
    history = _validate_physical_attempt_history(selected, physical_results)
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=None,
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
        "scientific_metrics_owner": "analyze_imlvit_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": args.mode,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "score_spec": None,
        "task_scope": TASK_SCOPE,
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
            "results_sha256": (
                sha256_file(results_path) if results_path.is_file() else None
            ),
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
        raise RuntimeError("IML-ViT fail-closed inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
