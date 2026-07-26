#!/usr/bin/env python3
"""Run pinned UniversalFakeDetect inference on the Balanced250 score cache.

This versioned v2 adapter leaves the legacy Mouse-v1 runner untouched.  It
freezes the official OpenAI CLIP ViT-L/14 backbone, the released
``Ours (L/14 + LC)`` linear head, and the repository's current native
CenterCrop(224) validation transform.  Every successful whole-image T1 result
persists the exact captured 768-dimensional float32 CLIP feature.  Scientific
metrics are intentionally owned by the separate Balanced250 analyzer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_universalfakedetect as legacy
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


RUN_MANIFEST_SCHEMA = "universalfakedetect_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "universalfakedetect_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "universalfakedetect_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "universalfakedetect_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/universalfakedetect")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/universalfakedetect")
DEFAULT_FORMAL_RUN_ID = (
    "universalfakedetect_clip_vit_l14_ours_lc_"
    "current_head_native_center_crop224_balanced250_v1_"
    "full1775_20260726"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "universalfakedetect_clip_vit_l14_ours_lc_"
    "current_head_native_center_crop224_balanced250_v1_"
    "smoke5x7_a_20260726"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "universalfakedetect_clip_vit_l14_ours_lc_"
    "current_head_native_center_crop224_balanced250_v1_"
    "smoke5x7_b_20260726"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/" "UniversalFakeDetect-76a0e3e60a8a"
)
DEFAULT_HEAD_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/"
    "universal-fake-detect-76a0e3e60a8a/fc_weights.pth"
)
DEFAULT_BACKBONE_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/openai-clip/ViT-L-14.pt"
)
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = 20260726
FROZEN_PROFILE = legacy.CURRENT_PROFILE
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 8 * 1024**3

# The clean, non-system-site-packages environment is part of the numerical
# contract.  Package versions are updated only together with the CPU golden.
FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/" "ufd-balanced-torch2.8.0/bin/python"
)
FROZEN_VENV_PREFIX = Path("/root/.cache/claimforge/venvs/ufd-balanced-torch2.8.0")
FROZEN_PYVENV_CONFIG_SHA256 = (
    "4126faa1b923a338127a0dc182f5df7f0dd0f53159c6f4568f56eb6582f22a8f"
)
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0+cu128",
    "torch_distribution": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "torchvision_distribution": "0.23.0+cu128",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "ftfy": "6.3.1",
    "regex": "2024.11.6",
    "setuptools": "75.8.0",
    "tqdm": "4.67.1",
}

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
    threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
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

PREPROCESS_CONTRACT = {
    "profile_id": FROZEN_PROFILE,
    "profile": legacy.PREPROCESS_PROFILES[FROZEN_PROFILE],
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "resize": None,
    "center_crop": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
    "tensor_dtype": "float32",
    "normalization_mean": list(legacy.CLIP_MEAN),
    "normalization_std": list(legacy.CLIP_STD),
    "official_forward_calls_per_image": 1,
    "feature_capture": "forward_hook_on_official_model.fc_input",
}
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_CONTRACT = {
    "feature": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "nbytes": (legacy.FEATURE_DIMENSION * np.dtype(np.float32).itemsize),
        "finite": True,
        "semantics": ("official_CLIP_encode_image_output_before_linear_head"),
        "allow_pickle": False,
        "exact_fc_and_sigmoid_replay": True,
    }
}

TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}

MODEL_CONTRACT = {
    "name": legacy.MODEL_NAME,
    "slug": legacy.MODEL_SLUG,
    "architecture": legacy.MODEL_ARCH,
    "repository": legacy.MODEL_REPO_URL,
    "paper": legacy.PAPER_URL,
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "released_head": legacy.HEAD_CHECKPOINT["released_name"],
    "head_checkpoint_sha256": legacy.HEAD_CHECKPOINT["sha256"],
    "backbone_provider": legacy.BACKBONE_CHECKPOINT["provider"],
    "backbone_checkpoint_sha256": legacy.BACKBONE_CHECKPOINT["sha256"],
    "asset_bundle_sha256": (
        "b57c7a8865336b82fe716dc7871006883417cf60f7b36ca8a9a5f925da009121"
    ),
    "license": legacy.LICENSE_RECORD,
}
EXPECTED_ASSET_BUNDLE_SHA256 = MODEL_CONTRACT["asset_bundle_sha256"]

SOURCE_CHECKPOINT_DRIFT = {
    "head_introduction_commit": legacy.HEAD_INTRO_COMMIT,
    "head_era_validate_sha256": legacy.HEAD_ERA_VALIDATE_SHA256,
    "resize_removal_commit": legacy.RESIZE_REMOVAL_COMMIT,
    "current_source_commit": legacy.MODEL_SOURCE_COMMIT,
    "frozen_resolution": "current_HEAD_native_center_crop224_only",
    "checkpoint_era_profile_executed": False,
}

ADAPTER_SOURCE_PATHS = (
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_universalfakedetect_balanced.py",
    "eval/opensource/analyze_universalfakedetect_balanced.py",
    "eval/opensource/run_universalfakedetect.py",
    "eval/opensource/analyze_universalfakedetect_run.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
    "eval/opensource/ufd_metrics.py",
)

IMMUTABLE_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "model",
        "source_checkpoint_drift",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "artifact_contract",
        "outputs",
    }
)

CPU_GOLDEN_SAMPLE_ID = "5f7535f0b957874982b1b080"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/" f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "f90c849192fd53e2e9560192d91b5b37a6162f80c14c862e24d37482784b8078"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "99e96c12b945768f247990c6fc521f95de1797ce68b0a5f30c889af3ccc472c0"
)
CPU_GOLDEN_CROP_RGB_SHA256 = (
    "105cbdcc566d3a48b85c5b34198c81dfd2a69a6ec79fe1005af1f7c36dc31dbe"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "bf9ce4ebbfd24f886d8bf70845b85d194ad942439275934e36b23284c22ca0cb"
)
# These stable-runtime values are deliberately isolated here.  They are
# finalized from two CPU full forwards in FROZEN_PYTHON_EXECUTABLE.
CPU_GOLDEN_FEATURE_ARRAY_SHA256 = (
    "6e320b71683f9a9a294d497c2a895ff437854e543ed2f4407ff4128b9296c29d"
)
CPU_GOLDEN_FEATURE_FILE_SHA256 = (
    "325da96b2f9d3e2d060bb8e19b49835b29e0190182d40070b2f58b3e0bcfb06d"
)
CPU_GOLDEN_RAW_LOGIT = -1.3850042819976807
CPU_GOLDEN_PROBABILITY = 0.20020648837089539
# A persisted feature is the exact input to the released float32 linear head.
# Resume therefore requires bit-exact scalar replay, rather than accepting the
# looser sigmoid tolerance used only to validate an isolated score payload.
HEAD_REPLAY_RAW_LOGIT_ABS_TOLERANCE = 0.0
HEAD_REPLAY_PROBABILITY_ABS_TOLERANCE = 0.0


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run-id must be one safe ASCII path component (max 160 chars)")
    return value


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local source participating in inference and audit."""

    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                "missing or unsafe UniversalFakeDetect Balanced adapter "
                f"source: {path}"
            )
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
        or any("pair_rank" in row for row in selected)
    ):
        raise ValueError("formal UniversalFakeDetect Balanced250 selection drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        isinstance(per_condition_limit, bool)
        or not isinstance(per_condition_limit, int)
        or not 1 <= per_condition_limit <= 250
    ):
        raise ValueError("smoke per-condition-limit must be in [1, 250]")
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=per_condition_limit,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if (
            condition in Capability.WHOLE_IMAGE_T1.conditions
            and counts[condition] < per_condition_limit
        ):
            sample_id = str(panel_row["sample_id"])
            source = inputs_by_id.get(sample_id)
            if source is None or source.get("panel") is not True:
                raise ValueError("smoke panel has a dangling/non-panel input")
            selected.append(source)
            counts[condition] += 1
    expected = {condition: per_condition_limit for condition in BALANCED_CONDITIONS}
    if dict(counts) != expected or any("pair_rank" in row for row in selected):
        raise ValueError("smoke panel does not cover the frozen T1 conditions")
    selected.sort(key=lambda row: int(row["rank"]))
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    """Create the exact formal, panel-smoke, or diagnostic selection."""

    if release.release_kind != "balanced250":
        raise ValueError("UniversalFakeDetect v2 requires a Balanced250 release")
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode does not accept input selectors")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode does not accept sample-id")
        return _smoke_selection(
            release,
            DEFAULT_SMOKE_LIMIT if per_condition_limit is None else per_condition_limit,
        )
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode does not accept per-condition-limit")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            sample_id=sample_id,
        )
        selected = select_inputs(release, spec)
        if len(selected) != 1 or "pair_rank" in selected[0]:
            raise ValueError("single selection drifted")
        return spec, selected
    raise ValueError(f"unsupported inference mode {mode!r}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_cublas_workspace() -> str:
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_CONFIG}, got {current!r}"
        )
    return CUBLAS_WORKSPACE_CONFIG


def _frozen_runtime_versions() -> dict[str, str]:
    import torch
    import torchvision

    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_distribution": str(_package_version("torch")),
        "torchvision": str(torchvision.__version__),
        "torchvision_distribution": str(_package_version("torchvision")),
        "numpy": str(np.__version__),
        "Pillow": str(_package_version("Pillow")),
        "ftfy": str(_package_version("ftfy")),
        "regex": str(_package_version("regex")),
        "setuptools": str(_package_version("setuptools")),
        "tqdm": str(_package_version("tqdm")),
    }
    if actual != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            "UniversalFakeDetect dedicated runtime version drifted: "
            f"expected {FROZEN_RUNTIME_VERSIONS}, got {actual}"
        )
    executable = Path(os.path.abspath(sys.executable))
    frozen_executable = Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
    if executable != frozen_executable:
        raise RuntimeError(
            "UniversalFakeDetect must run in its dedicated environment: "
            f"{FROZEN_PYTHON_EXECUTABLE}"
        )
    return actual


def _venv_contract() -> dict[str, Any]:
    prefix = Path(os.path.abspath(sys.prefix))
    base_prefix = Path(os.path.abspath(sys.base_prefix))
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    config_path = expected_prefix / "pyvenv.cfg"
    if (
        prefix != expected_prefix
        or base_prefix != Path("/usr")
        or config_path.is_symlink()
        or not config_path.is_file()
        or sha256_file(config_path) != FROZEN_PYVENV_CONFIG_SHA256
    ):
        raise RuntimeError(
            "UniversalFakeDetect clean virtual-environment contract drifted"
        )
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        normalized = key.strip()
        if separator != "=" or not normalized or normalized in values:
            raise RuntimeError("UniversalFakeDetect pyvenv.cfg is malformed")
        values[normalized] = value.strip()
    if values != {
        "home": "/usr/bin",
        "include-system-site-packages": "false",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": (
            "/usr/bin/python3.12 -m venv "
            "/root/.cache/claimforge/venvs/ufd-balanced-torch2.8.0"
        ),
    }:
        raise RuntimeError("UniversalFakeDetect pyvenv.cfg values drifted")
    return {
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "pyvenv_cfg_path": str(config_path),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": False,
    }


def configure_runtime(
    device_text: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the deterministic runtime and resolve an explicit device."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"UniversalFakeDetect seed must be exactly {DEFAULT_SEED}")
    cublas_workspace = _configure_cublas_workspace()
    import torch

    versions = _frozen_runtime_versions()
    venv = _venv_contract()
    if device_text == "cpu":
        device = torch.device("cpu")
    elif re.fullmatch(r"cuda:[0-9]+", device_text):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_text)
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device_text}")
        torch.cuda.set_device(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        if int(free_bytes) < MINIMUM_CUDA_FREE_BYTES:
            raise RuntimeError(
                f"{device} has only {int(free_bytes)} free bytes; "
                "UniversalFakeDetect requires at least "
                f"{MINIMUM_CUDA_FREE_BYTES}"
            )
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    runtime: dict[str, Any] = {
        "device": str(device),
        "python": {
            "implementation": platform.python_implementation(),
            "version": versions["python"],
            "executable": str(Path(os.path.abspath(sys.executable))),
        },
        "venv": venv,
        "platform": platform.platform(),
        "packages": {
            "torch": {
                "version": versions["torch"],
                "distribution_version": versions["torch_distribution"],
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
            },
            "torchvision": {
                "version": versions["torchvision"],
                "distribution_version": (versions["torchvision_distribution"]),
            },
            "numpy": versions["numpy"],
            "Pillow": versions["Pillow"],
            "ftfy": versions["ftfy"],
            "regex": versions["regex"],
            "setuptools": versions["setuptools"],
            "tqdm": versions["tqdm"],
        },
        "seed": seed,
        "preprocess_profile": FROZEN_PROFILE,
        "feature_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": cublas_workspace,
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "minimum_cuda_free_bytes": MINIMUM_CUDA_FREE_BYTES,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    validate_runtime_contract(runtime, label="configured runtime")
    return device, runtime


def validate_runtime_contract(
    value: Mapping[str, Any],
    *,
    label: str = "runtime",
) -> Mapping[str, Any]:
    """Validate persisted runtime evidence without changing global state."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    device = value.get("device")
    if not isinstance(device, str) or (
        device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None
    ):
        raise ValueError(f"{label}.device is unsupported")
    expected_keys = {
        "device",
        "python",
        "venv",
        "platform",
        "packages",
        "seed",
        "preprocess_profile",
        "feature_dtype",
        "batch_size",
        "autocast",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
        "float32_matmul_precision",
        "minimum_cuda_free_bytes",
    }
    if device.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = value.get("python")
    if not isinstance(python_record, Mapping) or set(python_record) != {
        "implementation",
        "version",
        "executable",
    }:
        raise ValueError(f"{label}.python key set changed")
    if (
        python_record.get("implementation") != "CPython"
        or python_record.get("version") != FROZEN_RUNTIME_VERSIONS["python"]
        or Path(os.path.abspath(str(python_record.get("executable"))))
        != Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
    ):
        raise ValueError(f"{label}.python dedicated runtime changed")
    venv = value.get("venv")
    if not isinstance(venv, Mapping) or set(venv) != {
        "prefix",
        "base_prefix",
        "pyvenv_cfg_path",
        "pyvenv_cfg_sha256",
        "include_system_site_packages",
    }:
        raise ValueError(f"{label}.venv key set changed")
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    if dict(venv) != {
        "prefix": str(expected_prefix),
        "base_prefix": "/usr",
        "pyvenv_cfg_path": str(expected_prefix / "pyvenv.cfg"),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": False,
    }:
        raise ValueError(f"{label}.venv clean-environment evidence changed")
    packages = value.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "ftfy",
        "regex",
        "setuptools",
        "tqdm",
    }:
        raise ValueError(f"{label}.packages key set changed")
    torch_record = packages.get("torch")
    torchvision_record = packages.get("torchvision")
    if (
        not isinstance(torch_record, Mapping)
        or set(torch_record)
        != {
            "version",
            "distribution_version",
            "cuda_runtime",
            "cudnn_version",
        }
        or not isinstance(torchvision_record, Mapping)
        or set(torchvision_record) != {"version", "distribution_version"}
    ):
        raise ValueError(f"{label}.package record key set changed")
    if (
        torch_record.get("version") != FROZEN_RUNTIME_VERSIONS["torch"]
        or torch_record.get("distribution_version")
        != FROZEN_RUNTIME_VERSIONS["torch_distribution"]
        or torch_record.get("cuda_runtime") != "12.8"
        or torchvision_record.get("version") != FROZEN_RUNTIME_VERSIONS["torchvision"]
        or torchvision_record.get("distribution_version")
        != FROZEN_RUNTIME_VERSIONS["torchvision_distribution"]
        or packages.get("numpy") != FROZEN_RUNTIME_VERSIONS["numpy"]
        or packages.get("Pillow") != FROZEN_RUNTIME_VERSIONS["Pillow"]
        or packages.get("ftfy") != FROZEN_RUNTIME_VERSIONS["ftfy"]
        or packages.get("regex") != FROZEN_RUNTIME_VERSIONS["regex"]
        or packages.get("setuptools") != FROZEN_RUNTIME_VERSIONS["setuptools"]
        or packages.get("tqdm") != FROZEN_RUNTIME_VERSIONS["tqdm"]
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    if not isinstance(value.get("platform"), str) or not value["platform"]:
        raise ValueError(f"{label}.platform is invalid")
    if (
        value.get("seed") != DEFAULT_SEED
        or value.get("preprocess_profile") != FROZEN_PROFILE
        or value.get("feature_dtype") != "float32"
        or value.get("batch_size") != 1
        or value.get("autocast") is not False
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("minimum_cuda_free_bytes") != MINIMUM_CUDA_FREE_BYTES
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = value.get("cudnn")
    if not isinstance(cudnn, Mapping) or set(cudnn) != {
        "enabled",
        "benchmark",
        "deterministic",
        "allow_tf32",
    }:
        raise ValueError(f"{label}.cudnn key set changed")
    if (
        type(cudnn.get("enabled")) is not bool
        or cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if device.startswith("cuda:"):
        cuda = value.get("cuda")
        if not isinstance(cuda, Mapping) or set(cuda) != {
            "runtime",
            "device_index",
            "device_name",
            "total_memory_bytes",
            "capability",
        }:
            raise ValueError(f"{label}.cuda key set changed")
        index = int(device.split(":", 1)[1])
        capability = cuda.get("capability")
        if (
            cuda.get("runtime") != "12.8"
            or cuda.get("device_index") != index
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{label}.cuda evidence changed")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    return path


def _expected_golden_preprocess() -> dict[str, Any]:
    return {
        "geometry": legacy.compute_preprocess_geometry(
            1800,
            1350,
            FROZEN_PROFILE,
        ),
        "decoded_rgb_sha256": CPU_GOLDEN_DECODED_RGB_SHA256,
        "crop_rgb_sha256": CPU_GOLDEN_CROP_RGB_SHA256,
        "crop_rgb_shape": [224, 224, 3],
        "crop_rgb_dtype": "uint8",
        "tensor_shape": [3, 224, 224],
        "tensor_dtype": "float32",
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
    }


def _validate_golden_forward(
    processed: Mapping[str, Any],
    feature: np.ndarray,
    *,
    label: str,
) -> tuple[bytes, str]:
    if (
        not isinstance(feature, np.ndarray)
        or feature.shape != (legacy.FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or not feature.flags.c_contiguous
        or not np.isfinite(feature).all()
    ):
        raise ValueError(f"{label} feature changed")
    expected_manual = {
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "probability": CPU_GOLDEN_PROBABILITY,
        "ai_score": CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
    }
    if dict(processed) != {
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "probability": CPU_GOLDEN_PROBABILITY,
        "ai_score": CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "manual_replay": expected_manual,
    }:
        raise ValueError(f"{label} score/full-forward evidence changed")
    array_sha = _array_sha256(feature)
    if array_sha != CPU_GOLDEN_FEATURE_ARRAY_SHA256:
        raise ValueError(f"{label} feature array SHA-256 changed")
    payload = _npy_bytes(feature)
    if hashlib.sha256(payload).hexdigest() != CPU_GOLDEN_FEATURE_FILE_SHA256:
        raise ValueError(f"{label} feature NPY SHA-256 changed")
    return payload, array_sha


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    head_checkpoint: Path,
    backbone_checkpoint: Path,
) -> dict[str, Any]:
    """Execute the pinned Balanced250 CPU golden twice, exactly."""

    import torch

    root = repo_root.resolve()
    image_path = root / CPU_GOLDEN_INPUT_PATH
    _require_regular_file(image_path, "CPU golden image")
    if sha256_file(image_path) != CPU_GOLDEN_IMAGE_SHA256:
        raise ValueError("CPU golden input SHA-256 changed")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was already initialized before the CPU golden forwards"
        )
    tensor, preprocess = legacy.preprocess_image(
        image_path,
        FROZEN_PROFILE,
    )
    if preprocess != _expected_golden_preprocess():
        raise ValueError("CPU golden preprocessing evidence changed")

    model = None
    try:
        model, loaded_device, load_audit = legacy.load_model(
            source_root=source_root.resolve(),
            head_checkpoint=head_checkpoint.resolve(),
            backbone_checkpoint=backbone_checkpoint.resolve(),
            device_name="cpu",
        )
        if str(device) != "cpu" or str(loaded_device) != "cpu":
            raise ValueError("CPU golden model did not load on CPU")
        first, first_feature, _first_peak, _first_latency = legacy.infer_one(
            model,
            loaded_device,
            tensor,
        )
        second, second_feature, _second_peak, _second_latency = legacy.infer_one(
            model,
            loaded_device,
            tensor,
        )
        first_bytes, first_array_sha = _validate_golden_forward(
            first,
            first_feature,
            label="CPU golden first forward",
        )
        second_bytes, second_array_sha = _validate_golden_forward(
            second,
            second_feature,
            label="CPU golden repeat forward",
        )
        if (
            first != second
            or not np.array_equal(first_feature, second_feature)
            or first_bytes != second_bytes
        ):
            raise ValueError("CPU golden repeated full forwards are not exact")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU golden model load/forwards initialized CUDA")
        source = load_audit.get("source")
        assets = load_audit.get("assets")
        if not isinstance(source, dict) or not isinstance(assets, dict):
            raise ValueError("CPU golden load audit lacks provenance")
        if assets.get("bundle_sha256") != EXPECTED_ASSET_BUNDLE_SHA256:
            raise ValueError("CPU golden asset bundle changed")
        golden = {
            "sample_id": CPU_GOLDEN_SAMPLE_ID,
            "input_path": CPU_GOLDEN_INPUT_PATH,
            "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
            "input_width": 1800,
            "input_height": 1350,
            "preprocess": preprocess,
            "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
            "feature_file_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "feature_file_bytes": len(first_bytes),
            "feature_array_sha256": first_array_sha,
            "feature_shape": [legacy.FEATURE_DIMENSION],
            "feature_dtype": "float32",
            "feature_nbytes": int(first_feature.nbytes),
            "raw_logit": first["raw_logit"],
            "probability": first["probability"],
            "ai_score": first["ai_score"],
            "classification_decision": first["classification_decision"],
            "full_image_forward": True,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "repeat_feature_file_sha256": hashlib.sha256(second_bytes).hexdigest(),
            "repeat_feature_file_bytes": len(second_bytes),
            "repeat_feature_array_sha256": second_array_sha,
            "repeat_raw_logit": second["raw_logit"],
            "repeat_probability": second["probability"],
            "repeat_ai_score": second["ai_score"],
            "repeat_classification_decision": second["classification_decision"],
            "repeat_full_image_forward": True,
            "repeat_model_forward_calls": 1,
            "repeat_fc_hook_calls": 1,
            "repeat_byte_exact": True,
        }
        return {
            "schema_version": CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "assets": assets,
            "model_load": load_audit,
            "runtime": runtime,
            "golden": golden,
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "cuda_initialized_before_cpu_model_load": False,
            "cuda_initialized_after_cpu_forwards": False,
            "dataset_manifest_loaded": False,
        }
    finally:
        if model is not None:
            del model
        gc.collect()


_FALSE_DECLARATIONS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULL_DECLARATIONS = frozenset(
    {
        "localization_output",
        "localisation_output",
        "dense_output",
        "score_map",
        "predicted_mask",
        "t2",
        "joint",
        "joint_score",
    }
)
_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "pair_rank",
        "localization",
        "localisation",
        "heatmap",
        "mask",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "pixel_f1",
        "iou",
        "miou",
        "dice",
        "mcc",
        "real_false_positive_area",
        "joint_metrics",
    }
)
_FORBIDDEN_CLAIM_PREFIXES = (
    "t2_",
    "pixel_",
    "localization_",
    "localisation_",
    "dense_",
    "heatmap_",
    "mask_",
    "predicted_mask",
    "score_map",
    "joint_",
)
_ALLOWED_DIAGNOSTIC_KEYS = frozenset({"pixel_center_mapping"})


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    """Reject pair-rank, localization, T2, dense, and joint-score claims."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(child, child_label)
                continue
            if normalized == "pair_rank":
                raise ValueError(f"{child_label} is an unsupported UFD claim")
            if normalized in _FALSE_DECLARATIONS:
                if child is not False:
                    raise ValueError(f"{child_label} is an unsupported UFD claim")
                continue
            if normalized in _NULL_DECLARATIONS:
                if child is not None:
                    raise ValueError(f"{child_label} is an unsupported UFD claim")
                continue
            if normalized in _FORBIDDEN_CLAIM_KEYS or normalized.startswith(
                _FORBIDDEN_CLAIM_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported UFD claim")
            _reject_unsupported_claims(child, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _reject_unsupported_claims(child, f"{label}[{index}]")


def _visibility_diagnostic(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Measure local exact-diff GT surviving the frozen center crop."""

    gt_kind = row.get("gt_mask_kind")
    geometry = legacy.compute_preprocess_geometry(
        int(row["width"]),
        int(row["height"]),
        FROZEN_PROFILE,
    )
    if gt_kind == "exact_diff":
        gt = legacy._validate_gt_row(row, repo_root)
        if gt is None:
            raise ValueError("local exact-diff input has no GT mask")
        edit_region = row.get("edit_region_xyxy")
        if (
            not isinstance(edit_region, list)
            or len(edit_region) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in edit_region
            )
        ):
            raise ValueError("local exact-diff input has invalid edit region")
        gt_diagnostic = legacy._gt_visibility(row, gt, geometry)
        box_diagnostic = legacy._edit_box_visibility(
            edit_region,
            list(geometry["effective_native_crop_xyxy"]),
        )
        return {
            "edit_visibility": gt_diagnostic["category"],
            "edit_visible_gt_fraction": gt_diagnostic["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": gt_diagnostic,
                "edit_box": box_diagnostic,
            },
        }
    if gt_kind not in ("all_zero", "not_applicable"):
        raise ValueError("Balanced250 input has unsupported GT kind")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": (
                "authentic_input_has_no_edit"
                if gt_kind == "all_zero"
                else "fullframe_condition_has_no_local_GT"
            ),
            "gt_mask_kind": gt_kind,
            "profile_id": FROZEN_PROFILE,
            "effective_native_crop_xyxy": list(geometry["effective_native_crop_xyxy"]),
        },
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    asset_bundle_sha256: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    """Build the exact UFD extension of the shared Balanced250 v2 identity."""

    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    if asset_bundle_sha256 != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("asset bundle SHA-256 changed")
    identity = build_result_identity(
        row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
    )
    input_path = _anchored(Path(str(row["canonical_path"])), repo_root)
    if repo_relative(input_path, repo_root) != identity["input_path"]:
        raise ValueError("Balanced250 input path is not repository-local")
    return {
        **identity,
        "valid_for_metrics": valid_for_metrics,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "model_arch": legacy.MODEL_ARCH,
        "model_source_commit": legacy.MODEL_SOURCE_COMMIT,
        "asset_bundle_sha256": asset_bundle_sha256,
        "preprocess_profile": FROZEN_PROFILE,
        "config_fingerprint": run_manifest_fingerprint,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        **_visibility_diagnostic(row, repo_root=repo_root),
    }


_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "clip_feature",
        "clip_feature_path",
        "clip_feature_sha256",
        "clip_feature_array_sha256",
        "clip_feature_shape",
        "clip_feature_dtype",
        "clip_feature_nbytes",
        "clip_feature_semantics",
        "artifact_paths",
        "raw_logit",
        "probability",
        "ai_score",
        "score",
        "score_semantics",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "classification",
        "t1",
        "manual_replay",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _safe_repo_file(value: Any, *, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a canonical repository path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} is not a canonical repository path")
    root = repo_root.resolve()
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    _require_regular_file(resolved, label)
    return resolved


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    raw = _finite_number(row.get("raw_logit"), f"{sample_id} raw_logit")
    probability = _finite_number(
        row.get("probability"),
        f"{sample_id} probability",
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{sample_id} probability is outside [0,1]")
    ai_score = _finite_number(row.get("ai_score"), f"{sample_id} ai_score")
    score = _finite_number(row.get("score"), f"{sample_id} score")
    # The official probability is produced and exactly replayed on the
    # recorded inference device.  Recomputing sigmoid from the scalar logit
    # on the host CPU is not a valid invariant: CUDA and CPU kernels may
    # differ by multiple float32 ULPs.  The authoritative same-device
    # feature -> released head -> sigmoid replay runs before resume skips and
    # again before finalization.
    if ai_score != probability or score != probability:
        raise ValueError(f"{sample_id} score orientation/aliases changed")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    expected_classification = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": "official_sigmoid_probability_higher_is_fake",
    }
    expected_t1 = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "policy": "official_UFD_CLIP_linear_probe_probability",
    }
    expected_manual = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
    }
    if (
        row.get("score_semantics") != "official_sigmoid_probability_higher_is_fake"
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold") != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or row.get("classification") != expected_classification
        or row.get("t1") != expected_t1
        or row.get("manual_replay") != expected_manual
    ):
        raise ValueError(f"{sample_id} score aliases/manual replay changed")


def _validate_feature_artifact(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    repo_root: Path,
    run_id: str,
) -> np.ndarray:
    feature = row.get("clip_feature")
    expected_keys = {
        "relative_path",
        "sha256",
        "file_bytes",
        "array_sha256",
        "dtype",
        "shape",
        "nbytes",
        "finite",
        "semantics",
    }
    if not isinstance(feature, Mapping) or set(feature) != expected_keys:
        raise ValueError(f"{sample_id} feature key set changed")
    expected_relative = (
        DEFAULT_ARTIFACTS_DIR / run_id / "clip_features" / f"{sample_id}.npy"
    ).as_posix()
    if feature.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} feature path is not canonical")
    path = _safe_repo_file(
        feature["relative_path"],
        repo_root=repo_root,
        label=f"{sample_id} CLIP feature",
    )
    payload = path.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    expected_nbytes = legacy.FEATURE_DIMENSION * np.dtype(np.float32).itemsize
    semantics = "official_CLIP_encode_image_output_before_linear_head"
    if (
        feature.get("sha256") != file_sha
        or feature.get("file_bytes") != len(payload)
        or feature.get("dtype") != "float32"
        or feature.get("shape") != [legacy.FEATURE_DIMENSION]
        or feature.get("nbytes") != expected_nbytes
        or feature.get("finite") is not True
        or feature.get("semantics") != semantics
    ):
        raise ValueError(f"{sample_id} feature metadata/hash changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} feature NPY is invalid") from error
    if (
        not isinstance(array, np.ndarray)
        or array.shape != (legacy.FEATURE_DIMENSION,)
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or array.nbytes != expected_nbytes
        or payload != _npy_bytes(array)
    ):
        raise ValueError(f"{sample_id} feature array changed")
    array_sha = _array_sha256(array)
    if feature.get("array_sha256") != array_sha:
        raise ValueError(f"{sample_id} feature array SHA-256 changed")
    aliases = {
        "clip_feature_path": expected_relative,
        "clip_feature_sha256": file_sha,
        "clip_feature_array_sha256": array_sha,
        "clip_feature_shape": [legacy.FEATURE_DIMENSION],
        "clip_feature_dtype": "float32",
        "clip_feature_nbytes": int(array.nbytes),
        "clip_feature_semantics": semantics,
        "artifact_paths": {"clip_feature_npy": expected_relative},
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} feature alias {key} changed")
    return array


def _validate_latest_feature_head_replay(
    *,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    head_state: Mapping[str, Any],
    device: Any,
) -> int:
    """Bind every latest successful feature to the audited released head."""

    import torch
    from torch.nn import functional as functional

    if (
        not isinstance(device, torch.device)
        or device.type not in ("cpu", "cuda")
        or (device.type == "cuda" and device.index is None)
        or (device.type == "cpu" and device.index is not None)
    ):
        raise ValueError("head replay requires an explicit configured torch device")
    if not isinstance(head_state, Mapping) or set(head_state) != {
        "weight",
        "bias",
    }:
        raise ValueError("audited linear head state key set changed")
    weight = head_state["weight"]
    bias = head_state["bias"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or tuple(weight.shape) != (1, legacy.FEATURE_DIMENSION)
        or tuple(bias.shape) != (1,)
        or weight.device.type != "cpu"
        or bias.device.type != "cpu"
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("audited linear head tensor schema changed")
    replay_weight = weight.detach().to(device=device)
    replay_bias = bias.detach().to(device=device)
    expected = sum(row.get("status") == "ok" for row in latest_by_sample_id.values())
    replayed = 0
    try:
        with torch.inference_mode():
            for sample_id, row in latest_by_sample_id.items():
                if row.get("status") != "ok":
                    continue
                feature_array = _validate_feature_artifact(
                    row,
                    sample_id=sample_id,
                    repo_root=repo_root,
                    run_id=run_id,
                )
                replay_feature = (
                    torch.from_numpy(feature_array).reshape(1, -1).to(device=device)
                )
                output = functional.linear(
                    replay_feature,
                    replay_weight,
                    replay_bias,
                )
                probability = torch.sigmoid(output)
                raw_value = float(output.reshape(()).item())
                probability_value = float(probability.reshape(()).item())
                raw_difference = abs(raw_value - float(row["raw_logit"]))
                probability_difference = abs(
                    probability_value - float(row["probability"])
                )
                if raw_difference > HEAD_REPLAY_RAW_LOGIT_ABS_TOLERANCE:
                    raise ValueError(
                        f"{sample_id} independent head replay logit mismatch"
                    )
                if probability_difference > HEAD_REPLAY_PROBABILITY_ABS_TOLERANCE:
                    raise ValueError(
                        f"{sample_id} independent head replay probability mismatch"
                    )
                replay_decision = probability_value > legacy.CLASSIFICATION_THRESHOLD
                if row.get("classification_decision") is not replay_decision:
                    raise ValueError(
                        f"{sample_id} independent head replay decision mismatch"
                    )
                replayed += 1
                del replay_feature, output, probability
    finally:
        del replay_weight, replay_bias
    if replayed != expected:
        raise ValueError("independent head replay coverage is incomplete")
    return replayed


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
) -> None:
    """Validate one append-only physical attempt and its feature."""

    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        asset_bundle_sha256=str(attempt.get("asset_bundle_sha256")),
        valid_for_metrics=status == "ok",
    )
    expected_keys = set(expected) | {"status", "completed_at"}
    if status == "ok":
        expected_keys |= _OK_RESULT_FIELDS
    else:
        expected_keys |= {"error_type", "error", "traceback"}
    if set(attempt) != expected_keys:
        raise ValueError(
            "result attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, expected_value in expected.items():
        if attempt.get(key) != expected_value:
            raise ValueError(f"result attempt field {key} drifted")
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("result attempt completed_at is invalid")
    _reject_unsupported_claims(attempt, "result attempt")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt["traceback"]
        ):
            raise ValueError("error result payload is invalid")
        return
    sample_id = str(input_row["sample_id"])
    _validate_score_payload(attempt, sample_id=sample_id)
    input_path = _safe_repo_file(
        str(input_row["canonical_path"]),
        repo_root=repo_root,
        label=f"{sample_id} canonical input",
    )
    _tensor, expected_preprocess = legacy.preprocess_image(
        input_path,
        FROZEN_PROFILE,
    )
    if attempt.get("preprocess") != expected_preprocess:
        raise ValueError(f"{sample_id} preprocessing record changed")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _finite_number(attempt.get(field), f"{sample_id} {field}") < 0.0:
            raise ValueError(f"{sample_id} {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{sample_id} peak memory is invalid")
    _validate_feature_artifact(
        attempt,
        sample_id=sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    run_dir: Path,
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    feature_dir: Path,
) -> dict[str, Any]:
    """Build the exact immutable config bound by the manifest fingerprint."""

    run_id = _valid_run_id(run_id)
    validate_runtime_contract(runtime, label="immutable runtime")
    if source.get("commit") != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("source audit commit changed")
    if assets.get("bundle_sha256") != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("asset bundle identity changed")
    immutable = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": MODEL_CONTRACT,
        "source_checkpoint_drift": SOURCE_CHECKPOINT_DRIFT,
        "preprocess": PREPROCESS_CONTRACT,
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "outputs": {
            "run_dir": repo_relative(run_dir, repo_root),
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "feature_dir": repo_relative(feature_dir, repo_root),
        },
    }
    if set(immutable) != IMMUTABLE_CONFIG_KEYS:
        raise AssertionError("internal immutable UFD config key set drifted")
    _reject_unsupported_claims(immutable, "immutable config")
    return immutable


def _ensure_repo_child(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    require_directory: bool = False,
) -> Path:
    """Resolve a repository child while rejecting every symlink component."""

    root = repo_root.resolve()
    raw = path if path.is_absolute() else root / path
    absolute = Path(os.path.abspath(raw))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if require_directory and not absolute.is_dir():
        raise FileNotFoundError(f"missing {label}: {absolute}")
    return absolute


def _prepare_output_directories(
    *,
    repo_root: Path,
    run_dir: Path,
    feature_root: Path,
    resume: bool,
) -> Path:
    run_dir = _ensure_repo_child(
        run_dir,
        repo_root=repo_root,
        label="UniversalFakeDetect run directory",
    )
    feature_root = _ensure_repo_child(
        feature_root,
        repo_root=repo_root,
        label="UniversalFakeDetect feature artifact root",
    )
    if (
        run_dir == feature_root
        or run_dir.is_relative_to(feature_root)
        or feature_root.is_relative_to(run_dir)
    ):
        raise ValueError("result and feature-artifact directories must be disjoint")
    feature_dir = feature_root / "clip_features"
    if not resume:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"run directory is non-empty; pass --resume: {run_dir}"
            )
        if feature_root.exists() and (
            not feature_root.is_dir() or any(feature_root.iterdir())
        ):
            raise FileExistsError(
                "feature artifact root is non-empty; pass --resume: " f"{feature_root}"
            )
    else:
        if not run_dir.is_dir() or not feature_dir.is_dir():
            raise FileNotFoundError(
                "resume requires the run and clip-feature directories"
            )
        root_entries = list(feature_root.iterdir())
        if (
            len(root_entries) != 1
            or root_entries[0].name != "clip_features"
            or root_entries[0].is_symlink()
            or not root_entries[0].is_dir()
        ):
            raise ValueError("resume feature artifact root inventory changed")
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_child(
        feature_dir,
        repo_root=repo_root,
        label="UniversalFakeDetect clip-feature directory",
        require_directory=True,
    )
    return feature_dir


def _validate_feature_inventory(
    *,
    feature_dir: Path,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
) -> int:
    feature_dir = _ensure_repo_child(
        feature_dir,
        repo_root=repo_root,
        label="UniversalFakeDetect clip-feature directory",
        require_directory=True,
    )
    entries = list(feature_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"feature inventory contains unsafe entry {entry.name}")
    expected = {
        f"{sample_id}.npy"
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "UniversalFakeDetect feature inventory mismatch: "
            f"missing={sorted(expected - actual)[:1]}, "
            f"extra={sorted(actual - expected)[:1]}"
        )
    for sample_id, row in latest_by_sample_id.items():
        if row.get("status") == "ok":
            _validate_feature_artifact(
                row,
                sample_id=sample_id,
                repo_root=repo_root,
                run_id=run_id,
            )
    return len(actual)


def _feature_record(
    *,
    feature: np.ndarray,
    feature_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(feature, np.ndarray)
        or feature.shape != (legacy.FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or not feature.flags.c_contiguous
        or not np.isfinite(feature).all()
    ):
        raise ValueError("official UFD feature array contract changed")
    payload = feature_path.read_bytes()
    if payload != _npy_bytes(feature):
        raise ValueError("persisted feature is not canonical NumPy bytes")
    return {
        "relative_path": repo_relative(feature_path, repo_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "array_sha256": _array_sha256(feature),
        "dtype": "float32",
        "shape": [legacy.FEATURE_DIMENSION],
        "nbytes": int(feature.nbytes),
        "finite": True,
        "semantics": ("official_CLIP_encode_image_output_before_linear_head"),
    }


def _build_ok_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    asset_bundle_sha256: str,
    feature_dir: Path,
    processed: Mapping[str, Any],
    feature: np.ndarray,
    preprocess: Mapping[str, Any],
    preprocess_latency_ms: float,
    latency_ms: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    feature_path = feature_dir / f"{sample_id}.npy"
    legacy._atomic_save_npy(feature_path, feature)
    feature_record = _feature_record(
        feature=feature,
        feature_path=feature_path,
        repo_root=repo_root,
    )
    raw = processed["raw_logit"]
    probability = processed["probability"]
    decision = processed["classification_decision"]
    relative = feature_record["relative_path"]
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            asset_bundle_sha256=asset_bundle_sha256,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "completed_at": utc_now(),
        "preprocess": dict(preprocess),
        "preprocess_latency_ms": float(preprocess_latency_ms),
        "clip_feature": feature_record,
        "clip_feature_path": relative,
        "clip_feature_sha256": feature_record["sha256"],
        "clip_feature_array_sha256": feature_record["array_sha256"],
        "clip_feature_shape": [legacy.FEATURE_DIMENSION],
        "clip_feature_dtype": "float32",
        "clip_feature_nbytes": feature_record["nbytes"],
        "clip_feature_semantics": feature_record["semantics"],
        "artifact_paths": {"clip_feature_npy": relative},
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": ("official_sigmoid_probability_higher_is_fake"),
        "classification_decision": decision,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
        "classification": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "decision": decision,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": (legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
            "semantics": ("official_sigmoid_probability_higher_is_fake"),
        },
        "t1": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "decision": decision,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": (legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
            "policy": "official_UFD_CLIP_linear_probe_probability",
        },
        "manual_replay": processed["manual_replay"],
        "latency_ms": float(latency_ms),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    return result


def _build_error_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    asset_bundle_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            asset_bundle_sha256=asset_bundle_sha256,
            valid_for_metrics=False,
        ),
        "status": "error",
        "completed_at": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    _require_regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row_label = f"{label}:{line_number}"
                if not line.endswith("\n"):
                    raise ValueError(f"{row_label} lacks final newline")
                if not line.strip():
                    raise ValueError(f"{row_label} is blank")
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(row, dict):
                    raise ValueError(f"{row_label} is not a JSON object")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


def _validate_preflight_report(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "assets",
        "model_load",
        "runtime",
        "golden",
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
        "dataset_manifest_loaded",
    }
    if set(report) != expected_keys:
        raise ValueError("CPU preflight report key set changed")
    if (
        report.get("schema_version") != CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("CPU preflight report/provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    load = report.get("model_load")
    if not isinstance(load, Mapping) or set(load) != {
        "source",
        "assets",
        "class_module",
        "class_name",
        "construction_api",
        "official_download_patch_calls",
        "urlopen_calls",
        "network_blocked",
        "clip_torch_load_fallback_blocked",
        "head_load",
        "feature_dimension",
        "visual_input_resolution",
        "head_parameters",
    }:
        raise ValueError("CPU preflight model-load key set changed")
    if (
        load.get("source") != source
        or load.get("assets") != assets
        or load.get("class_module") != "_claimforge_ufd_76a0e3e.models.clip_models"
        or load.get("class_name") != "CLIPModel"
        or load.get("construction_api") != "official models.get_model('CLIP:ViT-L/14')"
        or load.get("urlopen_calls") != 0
        or load.get("network_blocked") is not True
        or load.get("clip_torch_load_fallback_blocked") is not True
        or load.get("feature_dimension") != legacy.FEATURE_DIMENSION
        or load.get("visual_input_resolution") != legacy.MODEL_INPUT_SIZE
        or load.get("head_parameters") != legacy.FEATURE_DIMENSION + 1
    ):
        raise ValueError("CPU preflight model-load evidence changed")
    download_calls = load.get("official_download_patch_calls")
    if (
        not isinstance(download_calls, list)
        or len(download_calls) != 1
        or not isinstance(download_calls[0], Mapping)
        or set(download_calls[0]) != {"url", "requested_cache_root"}
        or download_calls[0].get("url") != legacy.BACKBONE_CHECKPOINT["official_url"]
        or not isinstance(download_calls[0].get("requested_cache_root"), str)
        or not download_calls[0]["requested_cache_root"]
        or load.get("head_load")
        != {
            "api": "torch.load",
            "weights_only": True,
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    ):
        raise ValueError("CPU preflight safe model-construction evidence changed")
    golden = report.get("golden")
    expected_golden_keys = {
        "sample_id",
        "input_path",
        "image_sha256",
        "input_width",
        "input_height",
        "preprocess",
        "tensor_sha256",
        "feature_file_sha256",
        "feature_file_bytes",
        "feature_array_sha256",
        "feature_shape",
        "feature_dtype",
        "feature_nbytes",
        "raw_logit",
        "probability",
        "ai_score",
        "classification_decision",
        "full_image_forward",
        "model_forward_calls",
        "fc_hook_calls",
        "repeat_feature_file_sha256",
        "repeat_feature_file_bytes",
        "repeat_feature_array_sha256",
        "repeat_raw_logit",
        "repeat_probability",
        "repeat_ai_score",
        "repeat_classification_decision",
        "repeat_full_image_forward",
        "repeat_model_forward_calls",
        "repeat_fc_hook_calls",
        "repeat_byte_exact",
    }
    if not isinstance(golden, Mapping) or set(golden) != expected_golden_keys:
        raise ValueError("CPU golden key set changed")
    expected_values = {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": _expected_golden_preprocess(),
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
        "feature_file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_file_bytes": 3200,
        "feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "feature_shape": [legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": 3072,
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "probability": CPU_GOLDEN_PROBABILITY,
        "ai_score": CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "full_image_forward": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "repeat_feature_file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        "repeat_feature_file_bytes": 3200,
        "repeat_feature_array_sha256": (CPU_GOLDEN_FEATURE_ARRAY_SHA256),
        "repeat_raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": CPU_GOLDEN_PROBABILITY,
        "repeat_ai_score": CPU_GOLDEN_PROBABILITY,
        "repeat_classification_decision": False,
        "repeat_full_image_forward": True,
        "repeat_model_forward_calls": 1,
        "repeat_fc_hook_calls": 1,
        "repeat_byte_exact": True,
    }
    if dict(golden) != expected_values:
        raise ValueError("CPU golden exact two-forward evidence changed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=DEFAULT_HEAD_CHECKPOINT,
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=DEFAULT_BACKBONE_CHECKPOINT,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
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
        help="explicit cpu or cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute one append-only UniversalFakeDetect Balanced250 run."""

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(Path(args.source_root), repo_root)
    head_checkpoint = _anchored(Path(args.head_checkpoint), repo_root)
    backbone_checkpoint = _anchored(
        Path(args.backbone_checkpoint),
        repo_root,
    )
    mode = str(args.mode)
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed != DEFAULT_SEED
    ):
        raise ValueError(f"UniversalFakeDetect seed must be exactly {DEFAULT_SEED}")
    if mode == "preflight":
        if (
            bool(args.resume)
            or bool(args.fail_fast)
            or args.run_id is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError("preflight accepts no run/selection/resume/CUDA options")
        report = run_cpu_preflight(
            repo_root=repo_root,
            source_root=source_root,
            head_checkpoint=head_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
        )
        source, assets, state = legacy._verify_asset_contract(
            source_root=source_root,
            head_checkpoint=head_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
        )
        del state
        _validate_preflight_report(
            report,
            source=source,
            assets=assets,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if mode != "formal" and args.run_id is None:
        raise ValueError("smoke and single modes require explicit --run-id")
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _anchored(
        Path(args.dataset_manifest),
        repo_root,
    )
    results_root = _ensure_repo_child(
        Path(args.results_dir),
        repo_root=repo_root,
        label="UniversalFakeDetect results root",
    )
    artifacts_root = _ensure_repo_child(
        Path(args.artifacts_dir),
        repo_root=repo_root,
        label="UniversalFakeDetect artifacts root",
    )
    if repo_relative(artifacts_root, repo_root) != DEFAULT_ARTIFACTS_DIR.as_posix():
        raise ValueError(f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}")
    run_dir = results_root / run_id
    feature_root = artifacts_root / run_id
    feature_dir = feature_root / "clip_features"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    selection_spec, selected = select_mode_inputs(
        release,
        mode=mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    )

    # The exact CPU gate deliberately precedes requested accelerator setup.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        head_checkpoint=head_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
    )
    source, assets, state = legacy._verify_asset_contract(
        source_root=source_root,
        head_checkpoint=head_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
    )
    _validate_preflight_report(
        cpu_preflight,
        source=source,
        assets=assets,
    )
    device, runtime = configure_runtime(device_text, seed=DEFAULT_SEED)
    adapter_sources = adapter_source_contract(repo_root)

    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        adapter_sources=adapter_sources,
        source=source,
        assets=assets,
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        run_dir=run_dir,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        feature_dir=feature_dir,
    )
    fingerprint = _fingerprint(immutable)
    feature_dir = _prepare_output_directories(
        repo_root=repo_root,
        run_dir=run_dir,
        feature_root=feature_root,
        resume=bool(args.resume),
    )

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )
        prior_manifest = _load_json_strict(manifest_path, "prior manifest")
        prior_status = prior_manifest.get("status")
        expected_top_level = {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "fingerprint",
            "immutable",
            "dataset",
            "outputs",
        }
        if prior_status in ("complete", "incomplete"):
            expected_top_level.add("execution")
        if (
            set(prior_manifest) != expected_top_level
            or prior_status not in ("running", "complete", "incomplete")
            or prior_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
            or prior_manifest.get("run_id") != run_id
            or prior_manifest.get("fingerprint") != fingerprint
            or prior_manifest.get("immutable") != immutable
        ):
            raise ValueError("resume run manifest fingerprint/config drifted")
        if _read_jsonl_strict(expected_path, "expected inputs") != selected:
            raise ValueError("resume expected-input snapshot drifted")
        expected_dataset = {
            "contract": dataset_contract.as_dict(),
            "manifest_path": repo_relative(
                dataset_manifest_path,
                repo_root,
            ),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(
                expected_path,
                repo_root,
            ),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        }
        if prior_manifest.get("dataset") != expected_dataset:
            raise ValueError("resume manifest dataset evidence drifted")
        prior_outputs = prior_manifest.get("outputs")
        if not isinstance(prior_outputs, Mapping):
            raise ValueError("resume manifest outputs are invalid")
        if prior_status == "running":
            if dict(prior_outputs) != immutable["outputs"]:
                raise ValueError("running resume output contract drifted")
        else:
            expected_output_keys = set(immutable["outputs"]) | {
                "results_sha256",
                "summary_sha256",
                "feature_files",
            }
            if (
                set(prior_outputs) != expected_output_keys
                or any(
                    prior_outputs.get(key) != value
                    for key, value in immutable["outputs"].items()
                )
                or not results_path.is_file()
                or prior_outputs.get("results_sha256") != sha256_file(results_path)
                or not summary_path.is_file()
                or prior_outputs.get("summary_sha256") != sha256_file(summary_path)
                or isinstance(prior_outputs.get("feature_files"), bool)
                or not isinstance(prior_outputs.get("feature_files"), int)
                or prior_outputs["feature_files"] < 0
            ):
                raise ValueError("finalized resume output evidence drifted")
        started_at = prior_manifest.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError("resume manifest started_at is invalid")
    else:
        atomic_write_jsonl(expected_path, selected)
        started_at = utc_now()

    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "fingerprint": fingerprint,
        "immutable": immutable,
        "dataset": {
            "contract": dataset_contract.as_dict(),
            "manifest_path": repo_relative(
                dataset_manifest_path,
                repo_root,
            ),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        },
        "outputs": dict(immutable["outputs"]),
    }

    physical_before = (
        _read_jsonl_strict(results_path, "prior physical results")
        if results_path.is_file()
        else []
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
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("prior result is outside the selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    validated_prior_feature_files = _validate_feature_inventory(
        feature_dir=feature_dir,
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    if (
        args.resume
        and prior_status in ("complete", "incomplete")
        and prior_outputs.get("feature_files") != validated_prior_feature_files
    ):
        raise ValueError("finalized resume feature count drifted")
    _validate_latest_feature_head_replay(
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        head_state=state,
        device=device,
    )
    # Do not mutate a resume manifest until all pre-existing history and
    # artifacts, including independent released-head replay, have passed.
    atomic_write_json(manifest_path, manifest)

    pending_ids = set(latest_before.pending_sample_ids(retry_errors=True))
    new_successes = 0
    resume_skips = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    model = None
    try:
        if pending_ids:
            model, loaded_device, load_audit = legacy.load_model(
                source_root=source_root,
                head_checkpoint=head_checkpoint,
                backbone_checkpoint=backbone_checkpoint,
                device_name=str(device),
            )
            if loaded_device != device:
                raise ValueError("loaded model device differs from runtime")
            if load_audit.get("source") != source or load_audit.get("assets") != assets:
                raise ValueError("loaded model provenance differs")
        else:
            loaded_device = device

        for index, input_row in enumerate(selected, start=1):
            sample_id = str(input_row["sample_id"])
            if sample_id not in pending_ids:
                resume_skips += 1
                print(
                    f"[{index}/{len(selected)}] resume {sample_id}",
                    flush=True,
                )
                continue
            feature_path = feature_dir / f"{sample_id}.npy"
            try:
                input_path = _safe_repo_file(
                    str(input_row["canonical_path"]),
                    repo_root=repo_root,
                    label=f"{sample_id} canonical input",
                )
                if sha256_file(input_path) != input_row["canonical_sha256"]:
                    raise ValueError(f"{sample_id} canonical input SHA-256 changed")
                preprocess_started = time.perf_counter()
                tensor, preprocess = legacy.preprocess_image(
                    input_path,
                    FROZEN_PROFILE,
                )
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if preprocess.get("geometry", {}).get("native_size") != [
                    int(input_row["width"]),
                    int(input_row["height"]),
                ]:
                    raise ValueError("preprocessed image dimensions changed")
                if model is None:
                    raise RuntimeError("UniversalFakeDetect model was not loaded")
                processed, feature, peak, latency = legacy.infer_one(
                    model,
                    loaded_device,
                    tensor,
                )
                result = _build_ok_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    asset_bundle_sha256=EXPECTED_ASSET_BUNDLE_SHA256,
                    feature_dir=feature_dir,
                    processed=processed,
                    feature=feature,
                    preprocess=preprocess,
                    preprocess_latency_ms=preprocess_latency_ms,
                    latency_ms=latency,
                    peak_cuda_memory_bytes=peak,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"ai_score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                if feature_path.exists():
                    if feature_path.is_symlink() or not feature_path.is_file():
                        raise ValueError(
                            f"unsafe failed feature path: {feature_path}"
                        ) from error
                    feature_path.unlink()
                result = _build_error_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    asset_bundle_sha256=EXPECTED_ASSET_BUNDLE_SHA256,
                    error=error,
                )
                append_jsonl(results_path, result)
                new_errors += 1
                print(
                    f"[{index}/{len(selected)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
                    break
            finally:
                gc.collect()
                if device.type == "cuda":
                    __import__("torch").cuda.empty_cache()
    finally:
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()

    physical_results = _read_jsonl_strict(
        results_path,
        "physical results",
    )
    for attempt in physical_results:
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("result is outside the selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    coverage = summarize_coverage(latest)
    feature_files = _validate_feature_inventory(
        feature_dir=feature_dir,
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    _validate_latest_feature_head_replay(
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        head_state=state,
        device=device,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": ("analyze_universalfakedetect_balanced.py"),
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": FROZEN_PROFILE,
        "score_spec": SCORE_SPEC.as_dict(),
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": coverage.as_dict(),
        "generated_at": utc_now(),
    }
    _reject_unsupported_claims(summary, "runtime summary")
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
            "feature_files": feature_files,
        }
    )
    _reject_unsupported_claims(manifest, "run manifest")
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "mode": mode,
                "coverage": coverage.as_dict(),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError(
            "UniversalFakeDetect fail-fast inference failed"
        ) from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
