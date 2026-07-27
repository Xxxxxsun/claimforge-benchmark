#!/usr/bin/env python3
"""Run the official CAT-Net v2 localization checkpoint on Balanced250.

This adapter is deliberately T2-only.  CAT-Net exposes a two-class dense
localization head, but no native image-level integrity head.  Consequently it
scores only the 1,025 applicable Balanced250 inputs (real275 plus the three
local-insertion panels) and never promotes a map mean, maximum, area, or other
map statistic to T1.

The frozen Mouse-v1 runner remains unchanged.  Its JPEG/DCT preprocessing,
strict official checkpoint load, forward, and postprocessing primitives are
reused without changing their scientific protocol.
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

from eval.opensource import run_catnet as legacy
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
from eval.opensource.maskclip_metrics import binary_pixel_metrics


RUN_MANIFEST_SCHEMA = "catnet_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "catnet_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "catnet_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "catnet_balanced_cpu_preflight_v2"

MODEL_NAME = "CAT-Net v2"
MODEL_SLUG = "catnet_v2_ijcv2022"
MODEL_ARCHITECTURE = "dual_stream_RGB_HRNet_plus_JPEG_DCT_HRNet"
MODEL_TREE = "ab0302cf11760ba1d07e0b419f20e9add0681112"
MODEL_GIT_ORIGIN = "https://github.com/mjkwon2021/CAT-Net.git"
PREPROCESS_PROFILE = "official_original_jpeg_rgb_y_dct_qtable_no_resize"

MODEL_SEED = 42
MASK_THRESHOLD = legacy.MASK_THRESHOLD
MASK_THRESHOLD_OPERATOR = ">="
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/catnet")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/catnet")
DEFAULT_FORMAL_RUN_ID = (
    "catnet_v2_ijcv2022_balanced250_v1_full1025_20260727"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "catnet_v2_ijcv2022_balanced250_v1_smoke5x4_a_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "catnet_v2_ijcv2022_balanced250_v1_smoke5x4_b_20260727"
)
DEFAULT_SMOKE_LIMIT = 5

EXPECTED_VENV_ROOT = Path("/root/.cache/claimforge/venvs/catnet-b50d391")
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_BYTES = 205
EXPECTED_PYVENV_SHA256 = (
    "68d85b45e93fcb149d73ff58c3c717fa79b6e490ec309bc23e710b136678aabc"
)
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/catnet-balanced-v2-empty"
)
EXPECTED_PACKAGES = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "jpegio": "0.2.8",
    "yacs": "0.1.8",
    "scikit-learn": "1.5.2",
    "scipy": "1.16.0",
}
EXPECTED_TORCH_CUDA_VERSION = "12.8"

SOURCE_BOUND_FILES: dict[str, tuple[int, str]] = {
    "README.md": (
        7_499,
        "9018b5cfe4863025d742643c0c53ef231b2d580f0a2fbee7f06d1b4831e009eb",
    ),
    "LICENSE of HRNet": (
        1_362,
        legacy.MODEL_LICENSE_SHA256,
    ),
    "experiments/CAT_full.yaml": (
        2_366,
        legacy.MODEL_CONFIG_SHA256,
    ),
    "lib/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "lib/config/__init__.py": (
        478,
        "9b0c0ebc3db2f39f4c8b2b4eda311bfeb1b977d5fab2a42bc1ea6f926cae28db",
    ),
    "lib/config/default.py": (
        2_598,
        "0ff4fa6e42bb4c72f030858365392d431fe29500dffaa109fdcf2639d4d080f7",
    ),
    "lib/config/models.py": (
        1_975,
        "5db1613e604ee89c7cedf46306f67a67e74cdc131287803ffe46ebdaa22a0823",
    ),
    "lib/models/__init__.py": (
        439,
        "ecb7b0de1012239797183bef9f40d94b3dc5d9f4bb0764295fb0548fa1c2612b",
    ),
    "lib/models/network_CAT.py": (
        23_160,
        legacy.MODEL_NETWORK_SHA256,
    ),
    "lib/models/network_DCT.py": (
        18_273,
        "fad8c894e462c311a1cb83e91348875eedbb66019eebc035a943940c9b25f1c3",
    ),
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_catnet_balanced.py",
    "eval/opensource/run_catnet.py",
    "eval/opensource/catnet_metrics.py",
    "eval/opensource/maskclip_metrics.py",
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
FORMAL_SELECTED_IDS_SHA256 = (
    "612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717"
)
FORMAL_SELECTED_ROWS_SHA256 = (
    "19ff584a5d073dd03cd31eaf0d22b105d079b2dd606ea535fbbcd39fb692b887"
)
SMOKE_SELECTED_IDS_SHA256 = (
    "3ce822824a5548f12ae0633520a19686048fd175f7add178334ab5c4fe7e78f4"
)
SMOKE_SELECTED_ROWS_SHA256 = (
    "7ec14339cad5c6e083f6b1fde56a965686d552ca0b9026eea975144ade7d1d6c"
)

T2_SPEC: dict[str, Any] = {
    "valid_conditions": list(LOCALIZATION_CONDITIONS),
    "not_selected_conditions": [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ],
    "score_map": {
        "source": (
            "official_two_channel_quarter_resolution_logits_bilinearly_"
            "restored_before_softmax_channel_1_then_native_crop"
        ),
        "space": "native_decoded_pixels",
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "native_binary_mask": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "encoding": "PNG_L_0_or_255",
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

LICENSE_RECORD: dict[str, Any] = {
    "catnet_project": {
        "project_wide_license_found": False,
        "classification": "source_available_research_release",
        "commercial_use_clearance_established": False,
        "redistribution_clearance_established": False,
    },
    "hrnet_component_notice": {
        "path": "LICENSE of HRNet",
        "sha256": legacy.MODEL_LICENSE_SHA256,
        "scope": "inherited_HRNet_component_only",
        "must_not_be_presented_as_CATNet_project_license": True,
    },
    "official_checkpoint": {
        "provider": "official_author_google_drive",
        "drive_file_id": legacy.CHECKPOINT_DRIVE_ID,
        "separate_license_or_terms_found": False,
        "commercial_use_clearance_established": False,
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
    "raw_logits_quarter": {
        "directory": "raw_logits_quarter",
        "format": "NumPy_npy_allow_pickle_false",
        "shape": "[2,ceil8_height/4,ceil8_width/4]",
        "dtype": "float32",
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
    "raw_logits_quarter",
    "score_maps_native",
    "masks_native",
)
NPY_HEADER_BYTES = 128
PNG_CONSERVATIVE_OVERHEAD_BYTES = 4_096
MIN_DISK_RESERVE_BYTES = 2_000_000_000

EXPECTED_MODEL_PARAMETERS = 114_263_810
RESOURCE_EXPECTATION: dict[str, Any] = {
    "observed_mouse_v1_peak_cuda_memory_bytes": 2_320_000_000,
    "observed_mouse_v1_median_forward_latency_ms": 59.03,
    "observed_mouse_v1_artifact_bytes_for_550_images": 3_980_000_000,
    "balanced250_artifact_projection_bytes": 7_416_000_000,
    "formal_runner_projection_minutes": [12, 35],
    "fresh_replay_projection_minutes": [4, 15],
    "recommended_free_cuda_memory_bytes": 3_500_000_000,
}

_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_T1_TOP_LEVEL = frozenset(
    {
        "score",
        "ai_score",
        "image_score",
        "decision",
        "classification",
        "classification_threshold",
        "t1_score",
        "valid_for_t1_score",
        "auroc",
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
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _valid_run_id(value: Any) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError("run_id must match [a-z0-9][a-z0-9._-]{0,127}")
    return value


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
                raise ValueError(
                    f"{path} line {line_number} is not UTF-8"
                ) from error
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
    probe = "outputs/opensource/catnet/.balanced_contract_probe/artifact.npy"
    process = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-v", probe],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or ".gitignore:" not in process.stdout:
        raise ValueError("CAT-Net artifact path is not covered by .gitignore")
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
            f"CAT-Net interpreter must be exactly {EXPECTED_PYTHON_EXECUTABLE}"
        )
    if Path(sys.prefix).resolve() != EXPECTED_VENV_ROOT.resolve():
        raise ValueError("CAT-Net sys.prefix is not the frozen virtualenv")
    pyvenv = EXPECTED_VENV_ROOT / "pyvenv.cfg"
    if (
        not pyvenv.is_file()
        or pyvenv.stat().st_size != EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("CAT-Net pyvenv.cfg binding changed")
    actual_packages = {
        name: _package_version(name) for name in EXPECTED_PACKAGES
    }
    if actual_packages != EXPECTED_PACKAGES:
        raise ValueError(
            f"CAT-Net package lock changed: {actual_packages!r}"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("PYTHONDONTWRITEBYTECODE must be exactly 1")
    if not sys.dont_write_bytecode:
        raise ValueError("Python bytecode writes are not disabled")
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(
        FROZEN_PYTHONPYCACHEPREFIX
    ):
        raise ValueError("PYTHONPYCACHEPREFIX is not the frozen CAT-Net path")
    if sys.pycache_prefix is None or Path(sys.pycache_prefix) != (
        FROZEN_PYTHONPYCACHEPREFIX
    ):
        raise ValueError("sys.pycache_prefix is not frozen")
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before environment preflight")
    if (
        torch.__version__ != EXPECTED_PACKAGES["torch"]
        or torch.version.cuda != EXPECTED_TORCH_CUDA_VERSION
    ):
        raise ValueError("CAT-Net torch build identity changed")
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
        "torch_cuda_version": torch.version.cuda,
        "python_dont_write_bytecode": True,
        "python_pycache_prefix": str(FROZEN_PYTHONPYCACHEPREFIX),
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def verify_source(catnet_root: Path) -> dict[str, Any]:
    root = _require_exact_path(
        catnet_root,
        legacy.DEFAULT_CATNET_ROOT,
        "CAT-Net source root",
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    commit = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    origin = _git_value(root, "remote", "get-url", "origin")
    status = _git_value(root, "status", "--short", "--untracked-files=all")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("CAT-Net source commit changed")
    if tree != MODEL_TREE:
        raise ValueError("CAT-Net source tree changed")
    if origin != MODEL_GIT_ORIGIN:
        raise ValueError("CAT-Net source origin changed")
    if status:
        raise ValueError("CAT-Net source checkout is not completely clean")
    bound: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in SOURCE_BOUND_FILES.items():
        path = root / relative
        _reject_symlink_components(path, f"CAT-Net source {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"CAT-Net source binding changed: {relative}")
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
        "CAT-Net checkpoint",
    )
    _reject_symlink_components(path, "CAT-Net checkpoint")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != legacy.CHECKPOINT_BYTES
        or sha256_file(path) != legacy.CHECKPOINT_SHA256
    ):
        raise ValueError("official CAT_full_v2 checkpoint binding changed")
    value = {
        "checkpoint": {
            "path": str(path),
            "filename": legacy.CHECKPOINT_FILENAME,
            "provider": "official_author_google_drive",
            "drive_file_id": legacy.CHECKPOINT_DRIVE_ID,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "epoch": legacy.CHECKPOINT_EPOCH,
            "state_keys": legacy.CHECKPOINT_STATE_KEYS,
            "safe_weights_only_load": True,
            "strict_model_load": True,
        }
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _checkpoint_schema(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    rows: list[dict[str, Any]] = []
    dtype_counts: Counter[str] = Counter()
    total_elements = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise ValueError("CAT-Net state_dict is not a string/tensor mapping")
        row = {
            "key": key,
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
            "elements": int(tensor.numel()),
        }
        rows.append(row)
        dtype_counts[str(tensor.dtype)] += 1
        total_elements += int(tensor.numel())
    return {
        "state_keys": len(rows),
        "state_elements": total_elements,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": _fingerprint([row["key"] for row in rows]),
        "tensor_schema_sha256": _fingerprint(rows),
    }


def _safe_checkpoint_audit(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    safe_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("CAT-Net checkpoint is not an object")
    if set(checkpoint) != {"best_p_mIoU", "epoch", "optimizer", "state_dict"}:
        raise ValueError("CAT-Net checkpoint top-level schema changed")
    if int(checkpoint.get("epoch", -1)) != legacy.CHECKPOINT_EPOCH:
        raise ValueError("CAT-Net checkpoint epoch changed")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("CAT-Net checkpoint state_dict is not an object")
    audit = _checkpoint_schema(state)
    if audit["state_keys"] != legacy.CHECKPOINT_STATE_KEYS:
        raise ValueError("CAT-Net checkpoint state-key count changed")
    for key, shape in (
        ("conv1.weight", [64, 3, 3, 3]),
        ("last_layer.3.weight", [2, 360, 1, 1]),
    ):
        tensor = state.get(key)
        if tensor is None or list(tensor.shape) != shape:
            raise ValueError(f"CAT-Net checkpoint tensor changed: {key}")
    del checkpoint, state
    gc.collect()
    return {
        "load": "torch.load_weights_only_true_with_minimal_numpy_safe_globals",
        "outer_type": "dict",
        "top_level_keys": [
            "best_p_mIoU",
            "epoch",
            "optimizer",
            "state_dict",
        ],
        "epoch": legacy.CHECKPOINT_EPOCH,
        **audit,
    }


def run_cpu_preflight(
    *,
    repo_root: Path,
    catnet_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Verify every frozen dependency and strict-load the full model on CPU.

    This gate performs no model forward, computes no Balanced250 score, and
    must leave CUDA uninitialized.
    """

    environment = verify_environment()
    source = verify_source(catnet_root)
    assets = verify_assets(checkpoint_path)
    adapter_sources = adapter_source_contract(repo_root)
    artifact_ignore = verify_artifact_ignore(repo_root)

    import torch

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise RuntimeError("CUDA was initialized before CAT-Net CPU preflight")
    checkpoint_audit = _safe_checkpoint_audit(checkpoint_path)
    if torch.cuda.is_initialized():
        raise RuntimeError("checkpoint audit initialized CUDA")
    model, device = legacy.load_model(
        catnet_root=catnet_root,
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )
    if str(device) != "cpu":
        raise ValueError("CAT-Net CPU preflight loaded on a non-CPU device")
    parameters = sum(int(value.numel()) for value in model.parameters())
    buffers = sum(int(value.numel()) for value in model.buffers())
    modules = sum(1 for _ in model.modules())
    state_keys = len(model.state_dict())
    if parameters != EXPECTED_MODEL_PARAMETERS:
        raise ValueError(
            f"CAT-Net parameter count changed: {parameters} "
            f"!= {EXPECTED_MODEL_PARAMETERS}"
        )
    if state_keys != legacy.CHECKPOINT_STATE_KEYS:
        raise ValueError("strict-loaded CAT-Net state schema changed")
    if model.training:
        raise ValueError("strict-loaded CAT-Net model is not in eval mode")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("CAT-Net CPU preflight model has non-CPU parameters")
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
        raise RuntimeError("CAT-Net CPU preflight initialized CUDA")
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
            "CAT-Net accelerator was initialized before runtime configuration"
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
        row
        for row in release.inputs
        if row.get("condition") in LOCALIZATION_CONDITIONS
    ]
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != FORMAL_IMAGES
        or _fingerprint([str(row["sample_id"]) for row in selected])
        != FORMAL_SELECTED_IDS_SHA256
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in expected]
    ):
        raise ValueError("formal CAT-Net Balanced250 selection drifted")
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
        or _fingerprint([str(row["sample_id"]) for row in selected])
        != SMOKE_SELECTED_IDS_SHA256
        or _rows_sha256(selected) != SMOKE_SELECTED_ROWS_SHA256
        or any(row.get("panel") is not True for row in selected)
        or any(
            str(row["condition"]).startswith("fullframe_") for row in selected
        )
    ):
        raise ValueError("CAT-Net smoke must be exactly 5x4 applicable rows")
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
        raise ValueError("CAT-Net smoke panel-priority selection drifted")
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
    raise ValueError("selected CAT-Net row violates T2-only semantics")


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
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "valid_for_metrics": valid_for_metrics,
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "task_scope": result_task_scope(row),
    }


def _preprocess_with_audit(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image, qtable, metadata = legacy.preprocess_jpeg(path)
    if image.dtype != np.float32 or not image.flags.c_contiguous:
        raise ValueError("CAT-Net preprocessed image is not contiguous float32")
    if qtable.dtype != np.float32 or not qtable.flags.c_contiguous:
        raise ValueError("CAT-Net qtable is not contiguous float32")
    width, height = (int(value) for value in metadata["native_size"])
    padded_width, padded_height = (
        int(value) for value in metadata["padded_size"]
    )
    if image.shape != (3 + legacy.DCT_BINS, padded_height, padded_width):
        raise ValueError("CAT-Net preprocessed image shape changed")
    if qtable.shape != (1, 8, 8):
        raise ValueError("CAT-Net qtable shape changed")
    if metadata["jpeg_sampling_factors"][:3] != [[1, 1], [1, 1], [1, 1]]:
        raise ValueError("CAT-Net requires canonical JPEG 4:4:4 sampling")
    if padded_width != math.ceil(width / 8) * 8 or padded_height != (
        math.ceil(height / 8) * 8
    ):
        raise ValueError("CAT-Net ceil-8 preprocessing changed")
    return image, qtable, metadata


def artifact_paths(artifact_root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "raw_logits": artifact_root / "raw_logits_quarter" / f"{sample_id}.npy",
        "score_map": artifact_root / "score_maps_native" / f"{sample_id}.npy",
        "mask": artifact_root / "masks_native" / f"{sample_id}.png",
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    raw_logits: np.ndarray,
    score_map: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "raw_logits_path": repo_relative(paths["raw_logits"], repo_root),
        "raw_logits_sha256": sha256_file(paths["raw_logits"]),
        "raw_logits_bytes": paths["raw_logits"].stat().st_size,
        "raw_logits_shape": list(raw_logits.shape),
        "raw_logits_dtype": str(raw_logits.dtype),
        "raw_logits_array_sha256": _array_sha256(raw_logits),
        "raw_logits_semantics": "official_two_channel_quarter_resolution_logits",
        "score_map_path": repo_relative(paths["score_map"], repo_root),
        "score_map_sha256": sha256_file(paths["score_map"]),
        "score_map_bytes": paths["score_map"].stat().st_size,
        "score_map_shape": list(score_map.shape),
        "score_map_dtype": str(score_map.dtype),
        "score_map_array_sha256": _array_sha256(score_map),
        "score_map_semantics": "native_probability_of_channel_1_tampered",
        "mask_path": repo_relative(paths["mask"], repo_root),
        "mask_sha256": sha256_file(paths["mask"]),
        "mask_bytes": paths["mask"].stat().st_size,
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "mask_array_sha256": _array_sha256(mask),
        "mask_semantics": "score_map_greater_than_or_equal_to_0_5",
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _resolve_exact_artifact(
    value: Any,
    expected: Path,
    *,
    repo_root: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    path = _anchored(Path(value), repo_root)
    _reject_symlink_components(path, label)
    if path != expected.resolve():
        raise ValueError(f"{label} path changed")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _load_npy_artifact(
    *,
    row: Mapping[str, Any],
    prefix: str,
    expected_path: Path,
    repo_root: Path,
    expected_shape: tuple[int, ...],
    bounded: bool,
) -> np.ndarray:
    path = _resolve_exact_artifact(
        row.get(f"{prefix}_path"),
        expected_path,
        repo_root=repo_root,
        label=prefix,
    )
    expected_sha = row.get(f"{prefix}_sha256")
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        raise ValueError(f"{prefix} SHA-256 is invalid")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{prefix} SHA-256 mismatch")
    if row.get(f"{prefix}_bytes") != path.stat().st_size:
        raise ValueError(f"{prefix} byte count mismatch")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != expected_shape or value.dtype != np.float32:
        raise ValueError(f"{prefix} shape/dtype mismatch")
    if row.get(f"{prefix}_shape") != list(expected_shape):
        raise ValueError(f"{prefix} shape metadata mismatch")
    if row.get(f"{prefix}_dtype") != "float32":
        raise ValueError(f"{prefix} dtype metadata mismatch")
    if not np.isfinite(value).all():
        raise ValueError(f"{prefix} contains non-finite values")
    if bounded and (
        float(np.min(value)) < 0.0 or float(np.max(value)) > 1.0
    ):
        raise ValueError(f"{prefix} falls outside [0,1]")
    if row.get(f"{prefix}_array_sha256") != _array_sha256(value):
        raise ValueError(f"{prefix} array SHA-256 mismatch")
    return value


def _validate_preprocess(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
    recompute: bool,
) -> None:
    value = row.get("preprocess")
    if not isinstance(value, Mapping):
        raise ValueError("result preprocess is not an object")
    width = int(input_row["width"])
    height = int(input_row["height"])
    padded_width = math.ceil(width / 8) * 8
    padded_height = math.ceil(height / 8) * 8
    expected = {
        "native_size": [width, height],
        "padded_size": [padded_width, padded_height],
        "padding": {
            "left": 0,
            "top": 0,
            "right": padded_width - width,
            "bottom": padded_height - height,
        },
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"CAT-Net preprocess {key} changed")
    if value.get("jpeg_sampling_factors")[:3] != [[1, 1], [1, 1], [1, 1]]:
        raise ValueError("CAT-Net preprocess sampling changed")
    for key in ("qtable_sha256", "dct_y_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"CAT-Net preprocess {key} is invalid")
        if row.get(key) != digest:
            raise ValueError(f"CAT-Net top-level {key} drifted")
    if recompute:
        image_path = _anchored(Path(str(input_row["canonical_path"])), repo_root)
        image, qtable, actual = _preprocess_with_audit(image_path)
        del image, qtable
        if stable_json(actual) != stable_json(dict(value)):
            raise ValueError("CAT-Net persisted preprocessing audit changed")


def _validate_ok_artifacts(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
    artifact_root: Path,
    recompute_preprocess: bool,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(artifact_root, sample_id)
    width = int(input_row["width"])
    height = int(input_row["height"])
    padded_width = math.ceil(width / 8) * 8
    padded_height = math.ceil(height / 8) * 8
    raw = _load_npy_artifact(
        row=row,
        prefix="raw_logits",
        expected_path=paths["raw_logits"],
        repo_root=repo_root,
        expected_shape=(2, padded_height // 4, padded_width // 4),
        bounded=False,
    )
    score_map = _load_npy_artifact(
        row=row,
        prefix="score_map",
        expected_path=paths["score_map"],
        repo_root=repo_root,
        expected_shape=(height, width),
        bounded=True,
    )
    mask_path = _resolve_exact_artifact(
        row.get("mask_path"),
        paths["mask"],
        repo_root=repo_root,
        label="mask",
    )
    expected_mask_sha = row.get("mask_sha256")
    if (
        not isinstance(expected_mask_sha, str)
        or not _SHA256.fullmatch(expected_mask_sha)
        or sha256_file(mask_path) != expected_mask_sha
    ):
        raise ValueError("mask SHA-256 mismatch")
    if row.get("mask_bytes") != mask_path.stat().st_size:
        raise ValueError("mask byte count mismatch")
    with Image.open(mask_path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError("CAT-Net mask format/mode changed")
        mask = np.asarray(opened, dtype=np.uint8)
    if (
        mask.shape != (height, width)
        or row.get("mask_shape") != [height, width]
        or row.get("mask_dtype") != "uint8"
        or not set(np.unique(mask).tolist()).issubset({0, 255})
    ):
        raise ValueError("CAT-Net mask schema changed")
    if row.get("mask_array_sha256") != _array_sha256(mask):
        raise ValueError("CAT-Net mask array SHA-256 mismatch")
    expected_mask = np.where(
        np.asarray(score_map) >= MASK_THRESHOLD,
        np.uint8(255),
        np.uint8(0),
    )
    if not np.array_equal(mask, expected_mask):
        raise ValueError("CAT-Net mask is not score_map >= 0.5")
    if row.get("raw_logits_semantics") != (
        "official_two_channel_quarter_resolution_logits"
    ):
        raise ValueError("CAT-Net raw-logit semantics changed")
    if row.get("score_map_semantics") != (
        "native_probability_of_channel_1_tampered"
    ):
        raise ValueError("CAT-Net score-map semantics changed")
    if row.get("mask_semantics") != (
        "score_map_greater_than_or_equal_to_0_5"
    ):
        raise ValueError("CAT-Net mask semantics changed")
    if row.get("mask_threshold") != MASK_THRESHOLD or row.get(
        "mask_threshold_operator"
    ) != MASK_THRESHOLD_OPERATOR:
        raise ValueError("CAT-Net mask threshold changed")
    _validate_preprocess(
        row,
        input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    target = load_ground_truth(input_row, repo_root)
    if target is None:
        raise ValueError("selected CAT-Net input has no applicable T2 target")
    expected_metrics = binary_pixel_metrics(
        np.asarray(score_map),
        np.asarray(target, dtype=bool),
        MASK_THRESHOLD,
        include_ap=str(input_row["condition"]) != "real",
    )
    localization = row.get("localization")
    if (
        not isinstance(localization, Mapping)
        or stable_json(localization.get("native")) != stable_json(expected_metrics)
    ):
        raise ValueError("CAT-Net localization metrics do not replay")
    del raw, score_map


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
        raise ValueError("CAT-Net result attempt is not an object")
    forbidden = sorted(_FORBIDDEN_T1_TOP_LEVEL.intersection(row))
    if forbidden:
        raise ValueError(f"CAT-Net result contains forbidden T1 fields: {forbidden}")
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError("CAT-Net result status is invalid")
    expected = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(f"CAT-Net result identity drifted at {key}")
    if row.get("valid_for_t1") is not False:
        raise ValueError("CAT-Net result cannot be valid for T1")
    if row.get("t2_applicable") is not True:
        raise ValueError("CAT-Net selected result must be T2-applicable")
    if status == "error":
        for key in (
            "raw_logits_path",
            "score_map_path",
            "mask_path",
            "localization",
            "latency_ms",
            "peak_cuda_memory_bytes",
        ):
            if key in row:
                raise ValueError(f"CAT-Net error attempt contains {key}")
        if not isinstance(row.get("error_type"), str) or not isinstance(
            row.get("error"), str
        ):
            raise ValueError("CAT-Net error attempt lacks error metadata")
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
            raise ValueError(f"invalid CAT-Net artifact directory: {path}")
        path.mkdir()


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ValueError("CAT-Net artifact root is missing or invalid")
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
        raise ValueError("CAT-Net artifact root inventory mismatch")
    successful = {
        str(row["sample_id"])
        for row in selected
        if latest_by_sample_id.get(str(row["sample_id"]), {}).get("status")
        == "ok"
    }
    expected = {
        "raw_logits_quarter": {f"{sample_id}.npy" for sample_id in successful},
        "score_maps_native": {f"{sample_id}.npy" for sample_id in successful},
        "masks_native": {f"{sample_id}.png" for sample_id in successful},
    }
    actual: dict[str, set[str]] = {}
    bytes_by_directory: dict[str, int] = {}
    for directory in ARTIFACT_DIRECTORIES:
        root = artifact_root / directory
        entries = list(root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError(f"CAT-Net {directory} contains a non-regular file")
        actual[directory] = {path.name for path in entries}
        bytes_by_directory[directory] = sum(
            path.stat().st_size for path in entries
        )
        if actual[directory] != expected[directory]:
            raise ValueError(f"CAT-Net {directory} inventory mismatch")
    value = {
        "successful_images": len(successful),
        "files": sum(len(names) for names in actual.values()),
        "files_by_directory": {
            directory: len(actual[directory])
            for directory in ARTIFACT_DIRECTORIES
        },
        "bytes_by_directory": bytes_by_directory,
        "total_bytes": sum(bytes_by_directory.values()),
        "exact_inventory": True,
    }
    return {**value, "inventory_sha256": _fingerprint(value)}


def _required_artifact_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        width = int(row["width"])
        height = int(row["height"])
        padded_width = math.ceil(width / 8) * 8
        padded_height = math.ceil(height / 8) * 8
        total += 2 * (padded_height // 4) * (padded_width // 4) * 4
        total += height * width * 4
        total += 2 * NPY_HEADER_BYTES + height * width
        total += PNG_CONSERVATIVE_OVERHEAD_BYTES
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
            f"insufficient CAT-Net artifact space: {available} < {minimum}"
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
    _reject_symlink_components(run_dir, "CAT-Net run directory")
    if not run_dir.exists():
        return
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("CAT-Net run directory is not a regular directory")
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
            f"CAT-Net run directory contains unexpected entries: "
            f"{sorted(unexpected)}"
        )
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"CAT-Net run directory is non-empty; pass --resume: {run_dir}"
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
                sorted(
                    Counter(str(row["condition"]) for row in selected).items()
                )
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repo_url": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_tree": MODEL_TREE,
            "checkpoint_filename": legacy.CHECKPOINT_FILENAME,
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": legacy.CHECKPOINT_BYTES,
            "checkpoint_epoch": legacy.CHECKPOINT_EPOCH,
            "checkpoint_state_keys": legacy.CHECKPOINT_STATE_KEYS,
            "checkpoint_strict_load": True,
            "checkpoint_safe_weights_only_load": True,
            "license": LICENSE_RECORD,
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "input_source": "canonical_jpeg_original_bytes",
            "input_resize": "none",
            "input_crop": None,
            "input_reencode": False,
            "rgb_reader": "Pillow",
            "jpeg_reader": "jpegio",
            "jpeg_component": "luminance_y",
            "rgb_normalization": "(uint8_minus_127.5)_divide_127.5",
            "padding": "right_and_bottom_to_ceil8_rgb127.5_dct0",
            "dct_volume": "21_bins_abs_0_1_to_19_ge20",
            "qtable": "original_luminance_quantization_table",
        },
        "inference": {
            "raw_output": "two_channel_logits_at_quarter_resolution",
            "map_restore": (
                "bilinear_logits_to_padded_native_align_corners_false_then_"
                "softmax_channel_1_then_native_crop"
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
            "expected_inputs_path": repo_relative(
                expected_inputs_path, repo_root
            ),
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
            raise ValueError(
                "smoke run ID must be the frozen A or B deterministic ID"
            )
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
        raise ValueError("resume CAT-Net manifest schema changed")
    if prior.get("fingerprint") != expected_fingerprint:
        raise ValueError("resume CAT-Net manifest fingerprint changed")
    if stable_json(prior.get("immutable")) != stable_json(
        dict(expected_immutable)
    ):
        raise ValueError("resume CAT-Net immutable run config changed")
    started_at = prior.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume CAT-Net manifest started_at is invalid")
    rows = _read_jsonl_strict(expected_inputs_path)
    if stable_json(rows) != stable_json(list(selected)):
        raise ValueError("resume CAT-Net expected input rows changed")
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
        "--catnet-root",
        type=Path,
        default=legacy.DEFAULT_CATNET_ROOT,
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
    catnet_root = _require_exact_path(
        args.catnet_root,
        legacy.DEFAULT_CATNET_ROOT,
        "CAT-Net source root",
    )
    checkpoint_path = _require_exact_path(
        args.checkpoint,
        legacy.DEFAULT_CHECKPOINT,
        "CAT-Net checkpoint",
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
            raise ValueError(
                "preflight accepts no run/selection/resume/CUDA options"
            )
        report = run_cpu_preflight(
            repo_root=repo_root,
            catnet_root=catnet_root,
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
        raise ValueError("CAT-Net dataset capability binding changed")

    results_root = _require_exact_path(
        _unresolved_anchored(args.results_dir, repo_root),
        repo_root / DEFAULT_RESULTS_DIR,
        "CAT-Net results root",
    )
    artifacts_root = _require_exact_path(
        _unresolved_anchored(args.artifacts_dir, repo_root),
        repo_root / DEFAULT_ARTIFACTS_DIR,
        "CAT-Net artifacts root",
    )
    run_dir = _safe_child(results_root, run_id, "CAT-Net run directory")
    artifact_root = _safe_child(
        artifacts_root, run_id, "CAT-Net artifact directory"
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("CAT-Net result and artifact roots must be disjoint")
    _validate_run_directory_safety(run_dir, resume=args.resume)
    if artifact_root.exists():
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise ValueError("CAT-Net artifact root is invalid")
        if any(artifact_root.iterdir()) and not args.resume:
            raise FileExistsError(
                f"artifact directory is non-empty; pass --resume: {artifact_root}"
            )

    # The full CPU strict-load gate intentionally precedes every accelerator
    # configuration call and every output-directory mutation.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        catnet_root=catnet_root,
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
        raise ValueError("resume CAT-Net dataset envelope changed")
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

    physical_before = (
        _read_jsonl_strict(results_path) if results_path.is_file() else []
    )
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
        if latest_before.latest_by_sample_id.get(
            str(row["sample_id"]), {}
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

    model = None
    loaded_device = device
    new_successes = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            model, loaded_device = legacy.load_model(
                catnet_root=catnet_root,
                checkpoint_path=checkpoint_path,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("CAT-Net loaded on an unexpected device")
        for index, input_row in enumerate(pending, start=1):
            sample_id = str(input_row["sample_id"])
            paths = artifact_paths(artifact_root, sample_id)
            try:
                input_path = _anchored(
                    Path(str(input_row["canonical_path"])), repo_root
                )
                image, qtable, preprocess = _preprocess_with_audit(input_path)
                if preprocess["native_size"] != [
                    int(input_row["width"]),
                    int(input_row["height"]),
                ]:
                    raise ValueError("canonical CAT-Net dimensions changed")
                assert model is not None
                raw_logits, score_map, peak_bytes, latency_ms = legacy.infer_one(
                    model,
                    loaded_device,
                    image,
                    qtable,
                    preprocess,
                )
                raw_logits = np.ascontiguousarray(raw_logits, dtype=np.float32)
                score_map = np.ascontiguousarray(score_map, dtype=np.float32)
                mask = np.where(
                    score_map >= MASK_THRESHOLD,
                    np.uint8(255),
                    np.uint8(0),
                )
                legacy._atomic_save_npy(paths["raw_logits"], raw_logits)
                legacy._atomic_save_npy(paths["score_map"], score_map)
                legacy._atomic_save_mask(
                    paths["mask"], score_map >= MASK_THRESHOLD
                )
                target = load_ground_truth(input_row, repo_root)
                if target is None:
                    raise ValueError(
                        "CAT-Net selected input has no applicable T2 target"
                    )
                localization = {
                    "native": binary_pixel_metrics(
                        score_map,
                        np.asarray(target, dtype=bool),
                        MASK_THRESHOLD,
                        include_ap=str(input_row["condition"]) != "real",
                    )
                }
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
                    "qtable_sha256": preprocess["qtable_sha256"],
                    "dct_y_sha256": preprocess["dct_y_sha256"],
                    **_artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        raw_logits=raw_logits,
                        score_map=score_map,
                        mask=mask,
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
        "scientific_metrics_owner": "analyze_catnet_balanced.py",
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
        raise RuntimeError("CAT-Net fail-closed inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
