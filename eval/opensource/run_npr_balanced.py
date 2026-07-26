#!/usr/bin/env python3
"""Run the pinned NPR detector on the Balanced250 whole-image score cache.

This v2 adapter leaves the legacy Mouse-v1 runner untouched.  It evaluates
the author-documented AIGCDetectBenchmark native-size compatibility
completion: the pinned GitHub ``ResNet.forward`` does not execute its
commented odd-dimension trim, while the pinned repository README instructs
users to add that trim and the pinned author Hugging Face Space implements
it.  At most the final bottom row and/or right column is therefore removed
before the otherwise unmodified eval-mode model is called.

Every successful T1 result records both the released float32 sigmoid score
and its raw logit, and persists the exact 512-dimensional float32 input to
``fc1``.  Resume and finalization replay feature -> released head -> sigmoid
on the recorded inference device; host-CPU sigmoid reconstruction is never a
validity gate.  NPR has no native dense output and is not valid for T2.
Scientific metrics belong to the separate Balanced250 analyzer.
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
import subprocess
import sys
import time
import traceback
import types
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_npr as legacy
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


RUN_MANIFEST_SCHEMA = "npr_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "npr_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "npr_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "npr_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/npr")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/npr")
DEFAULT_FORMAL_RUN_ID = (
    "npr_aigcdetect_progan4class_author_documented_native_even_trim_"
    "balanced250_v1_full1775_20260726"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "npr_aigcdetect_progan4class_author_documented_native_even_trim_"
    "balanced250_v1_smoke5x7_a_20260726"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "npr_aigcdetect_progan4class_author_documented_native_even_trim_"
    "balanced250_v1_smoke5x7_b_20260726"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_HF_SOURCE_ROOT = legacy.DEFAULT_HF_SOURCE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = legacy.MODEL_SEED
FROZEN_PROFILE = (
    "author_documented_aigcdetect_native_even_trim_completion"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 8 * 1024**3

FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0/bin/python"
)
FROZEN_VENV_PREFIX = Path(
    "/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0"
)
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/npr-balanced-torch2.8.0"
)
FROZEN_PYVENV_CONFIG_SHA256 = (
    "35470b7542154bebe1a55dac3c8760e7638711ff9b166285694e5156186acd06"
)
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0+cu128",
    "torch_distribution": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "torchvision_distribution": "0.23.0+cu128",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "setuptools": "75.8.0",
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
FORMAL_ODD_DIMENSION_COUNTS = {
    "odd_width": 81,
    "odd_height": 376,
    "both_odd": 13,
    "either_odd": 444,
}
EXPECTED_ODD_DIMENSION_IMAGES = 444

PREPROCESS_CONTRACT = {
    "profile_id": FROZEN_PROFILE,
    "profile": FROZEN_PROFILE,
    "status": "author_documented_compatibility_completion",
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "resize": None,
    "crop": None,
    "batch_size": 1,
    "tensor_dtype": "float32",
    "normalization_mean": list(legacy.IMAGE_MEAN),
    "normalization_std": list(legacy.IMAGE_STD),
    "odd_dimension_policy": (
        "remove_at_most_final_bottom_row_and_or_right_column_before_model"
    ),
    "trim_bottom_if_height_odd": True,
    "trim_right_if_width_odd": True,
    "odd_dimension_scope": "444_of_1775_formal_Balanced250_inputs",
    "github_live_forward_executes_trim": False,
    "github_readme_documents_trim": True,
    "pinned_author_hf_space_executes_trim": True,
    "paper_protocol_replication_claimed": False,
    "official_forward_calls_per_image": 1,
    "feature_capture": "forward_pre_hook_on_official_model.fc1_input",
}
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_CONTRACT = {
    "feature": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "nbytes": legacy.FEATURE_DIMENSION * np.dtype(np.float32).itemsize,
        "finite": True,
        "semantics": (
            "official_fc1_input_after_adaptive_global_average_pool"
        ),
        "allow_pickle": False,
        "exact_fc1_and_sigmoid_replay_on_recorded_device": True,
        "visibility": "local_only_gitignored_output",
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
    "hf_space": legacy.HF_SPACE_URL,
    "hf_space_commit": legacy.HF_SPACE_COMMIT,
    "checkpoint_id": legacy.CHECKPOINT["id"],
    "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
    "checkpoint_introduced_commit": legacy.CHECKPOINT["introduced_commit"],
    "asset_bundle_sha256": (
        "9c19c48e4a3a42f4628b89445e2a39fe564802efbfb8c93854aeae55dfa81b66"
    ),
    "model_mode": "eval",
    "model_mode_source": "pinned_GitHub_test.py",
    "npr_scale": 2.0 / 3.0,
    "feature_dimension": legacy.FEATURE_DIMENSION,
    "output": "one_float32_raw_logit_then_float32_sigmoid",
    "official_score": {
        "key": "ai_score",
        "value": "released_float32_sigmoid_of_raw_logit",
        "direction": "higher_means_fake",
        "threshold": 0.5,
        "threshold_operator": ">",
        "not_claimed": "calibrated_target_domain_probability",
    },
    "raw_logit_diagnostic": {
        "always_persisted": True,
        "always_reported_by_analyzer": True,
        "supplementary_not_primary": True,
        "never_replaces_released_threshold_policy": True,
        "motivation": "released_float32_sigmoid_saturation",
    },
    "license": legacy.LICENSE_RECORD,
}
EXPECTED_ASSET_BUNDLE_SHA256 = MODEL_CONTRACT["asset_bundle_sha256"]
EXPECTED_CHECKPOINT_SCHEMA_SHA256 = (
    "e60d79370c937aede4ff54ff57663207b6282f566c28caf56f6afd924af530d6"
)

SOURCE_COMPLETION_CONTRACT = {
    "pinned_github_live_resnet_trim_executed": False,
    "pinned_github_readme_trim_instruction": True,
    "pinned_author_hf_space_trim_executed": True,
    "completion": (
        "drop_only_final_bottom_row_when_height_odd_and_final_right_column_"
        "when_width_odd_before_unmodified_eval_mode_model"
    ),
    "terminology": (
        "author-documented deployment-corroborated compatibility completion"
    ),
    "not_claimed": [
        "pinned_GitHub_live_preprocessing_as_is",
        "paper_protocol_replication",
    ],
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_npr_balanced.py",
    "eval/opensource/analyze_npr_balanced.py",
    "eval/opensource/run_npr.py",
    "eval/opensource/analyze_npr_run.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
    "eval/opensource/npr_metrics.py",
)

IMMUTABLE_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "model",
        "source_completion",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "odd_dimension_counts",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "artifact_contract",
        "local_artifact_policy",
        "outputs",
    }
)

# The authoritative CPU golden deliberately has both dimensions odd.
CPU_GOLDEN_SAMPLE_ID = "7aeae0f17050bf766257b47d"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/" f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "21bfef64a1863cda43e122846c6cde1c40d97adcc33f813409f8204732e5093b"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "51e0da2279f209abe1b8349f0d215df0b2a989291c6ab02026b8b8b40e76a3e3"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "1ff28c34fdfdf89c8a684d99b71fbe78bf3e582ab1c5b0c9192b4ff18fd04640"
)
CPU_GOLDEN_RESIDUAL_SHA256 = (
    "84613a075de69e102fa62b71d99e7cca5f638774ae8b667928f91dcc6e8e9715"
)
CPU_GOLDEN_FEATURE_ARRAY_SHA256 = (
    "521a7bfbd00dbfee27d21271649c0d24b100842bd64b23f44dc422e9acddbbd4"
)
CPU_GOLDEN_FEATURE_FILE_SHA256 = (
    "6c3fc67b81c69bac75159dcfffd56d127393546a93fefca5d664c34c355bb8a4"
)
CPU_GOLDEN_RAW_LOGIT = -84.44386291503906
CPU_GOLDEN_PROBABILITY = 2.120783389925294e-37
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
    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or unsafe NPR Balanced source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def odd_dimension_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    odd_width = sum(int(row["width"]) % 2 == 1 for row in rows)
    odd_height = sum(int(row["height"]) % 2 == 1 for row in rows)
    both = sum(
        int(row["width"]) % 2 == 1 and int(row["height"]) % 2 == 1
        for row in rows
    )
    either = sum(
        int(row["width"]) % 2 == 1 or int(row["height"]) % 2 == 1
        for row in rows
    )
    return {
        "odd_width": odd_width,
        "odd_height": odd_height,
        "both_odd": both,
        "either_odd": either,
    }


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
        or odd_dimension_counts(selected) != FORMAL_ODD_DIMENSION_COUNTS
    ):
        raise ValueError("formal NPR Balanced250 selection drifted")
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
    if release.release_kind != "balanced250":
        raise ValueError("NPR v2 requires a Balanced250 release")
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


def _startup_isolation_contract() -> dict[str, Any]:
    expected_prefix = Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
    raw_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    actual_prefix = (
        Path(os.path.abspath(raw_prefix)) if isinstance(raw_prefix, str) else None
    )
    if (
        os.environ.get("PYTHONHASHSEED") != str(DEFAULT_SEED)
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or actual_prefix != expected_prefix
        or not expected_prefix.is_absolute()
        or expected_prefix.is_symlink()
        or (expected_prefix.exists() and not expected_prefix.is_dir())
        or (
            expected_prefix.is_dir()
            and any(expected_prefix.iterdir())
        )
        or sys.dont_write_bytecode is not True
        or Path(os.path.abspath(str(sys.pycache_prefix))) != expected_prefix
    ):
        raise RuntimeError(
            "NPR startup isolation requires PYTHONHASHSEED=100, "
            "PYTHONDONTWRITEBYTECODE=1, and an absolute empty "
            f"PYTHONPYCACHEPREFIX={FROZEN_PYTHONPYCACHEPREFIX}"
        )
    return {
        "PYTHONHASHSEED": str(DEFAULT_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(expected_prefix),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(expected_prefix),
        "pycache_prefix_initially_empty": True,
    }


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
        "scikit-learn": str(_package_version("scikit-learn")),
        "scipy": str(_package_version("scipy")),
        "joblib": str(_package_version("joblib")),
        "threadpoolctl": str(_package_version("threadpoolctl")),
        "setuptools": str(_package_version("setuptools")),
    }
    if actual != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            "NPR dedicated runtime version drifted: "
            f"expected {FROZEN_RUNTIME_VERSIONS}, got {actual}"
        )
    executable = Path(os.path.abspath(sys.executable))
    expected = Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
    if executable != expected:
        raise RuntimeError(
            f"NPR must run in its dedicated environment: {FROZEN_PYTHON_EXECUTABLE}"
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
        raise RuntimeError("NPR clean virtual-environment contract drifted")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        normalized = key.strip()
        if separator != "=" or not normalized or normalized in values:
            raise RuntimeError("NPR pyvenv.cfg is malformed")
        values[normalized] = value.strip()
    expected_values = {
        "home": "/usr/bin",
        "include-system-site-packages": "false",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": (
            "/usr/bin/python -m venv --upgrade "
            "/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0"
        ),
    }
    if values != expected_values:
        raise RuntimeError("NPR pyvenv.cfg values drifted")
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
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"NPR seed must be exactly {DEFAULT_SEED}")
    process_environment = _startup_isolation_contract()
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
                f"{device} has only {int(free_bytes)} free bytes; NPR requires "
                f"at least {MINIMUM_CUDA_FREE_BYTES}"
            )
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    # External NPR code is compiled from verified source bytes below.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.enabled = False
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
                "distribution_version": versions["torchvision_distribution"],
            },
            "numpy": versions["numpy"],
            "Pillow": versions["Pillow"],
            "scikit-learn": versions["scikit-learn"],
            "scipy": versions["scipy"],
            "joblib": versions["joblib"],
            "threadpoolctl": versions["threadpoolctl"],
            "setuptools": versions["setuptools"],
        },
        "seed": seed,
        "preprocess_profile": FROZEN_PROFILE,
        "inference_dtype": "float32",
        "feature_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "grad_enabled": False,
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
        "external_source_loader": "compile_verified_utf8_source_bytes_no_pyc",
        "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
        "process_environment": process_environment,
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
        "inference_dtype",
        "feature_dtype",
        "batch_size",
        "autocast",
        "grad_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
        "float32_matmul_precision",
        "minimum_cuda_free_bytes",
        "external_source_loader",
        "bytecode_writes_disabled",
        "process_environment",
    }
    if device.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = value.get("python")
    if not isinstance(python_record, Mapping) or dict(python_record) != {
        "implementation": "CPython",
        "version": FROZEN_RUNTIME_VERSIONS["python"],
        "executable": str(Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))),
    }:
        raise ValueError(f"{label}.python dedicated runtime changed")
    venv = value.get("venv")
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    if not isinstance(venv, Mapping) or dict(venv) != {
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
        "scikit-learn",
        "scipy",
        "joblib",
        "threadpoolctl",
        "setuptools",
    }:
        raise ValueError(f"{label}.packages key set changed")
    torch_record = packages.get("torch")
    torchvision_record = packages.get("torchvision")
    if (
        not isinstance(torch_record, Mapping)
        or set(torch_record)
        != {"version", "distribution_version", "cuda_runtime", "cudnn_version"}
        or not isinstance(torchvision_record, Mapping)
        or set(torchvision_record) != {"version", "distribution_version"}
        or torch_record.get("version") != FROZEN_RUNTIME_VERSIONS["torch"]
        or torch_record.get("distribution_version")
        != FROZEN_RUNTIME_VERSIONS["torch_distribution"]
        or torch_record.get("cuda_runtime") != "12.8"
        or torchvision_record.get("version")
        != FROZEN_RUNTIME_VERSIONS["torchvision"]
        or torchvision_record.get("distribution_version")
        != FROZEN_RUNTIME_VERSIONS["torchvision_distribution"]
        or any(
            packages.get(key) != FROZEN_RUNTIME_VERSIONS[key]
            for key in (
                "numpy",
                "Pillow",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "setuptools",
            )
        )
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    if not isinstance(value.get("platform"), str) or not value["platform"]:
        raise ValueError(f"{label}.platform is invalid")
    if (
        value.get("seed") != DEFAULT_SEED
        or value.get("preprocess_profile") != FROZEN_PROFILE
        or value.get("inference_dtype") != "float32"
        or value.get("feature_dtype") != "float32"
        or value.get("batch_size") != 1
        or value.get("autocast") is not False
        or value.get("grad_enabled") is not False
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("minimum_cuda_free_bytes") != MINIMUM_CUDA_FREE_BYTES
        or value.get("external_source_loader")
        != "compile_verified_utf8_source_bytes_no_pyc"
        or value.get("bytecode_writes_disabled") is not True
        or value.get("process_environment")
        != {
            "PYTHONHASHSEED": str(DEFAULT_SEED),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(
                Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
            ),
            "python_dont_write_bytecode": True,
            "sys_pycache_prefix": str(
                Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
            ),
            "pycache_prefix_initially_empty": True,
        }
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
        cudnn.get("enabled") is not False
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
        capability = cuda.get("capability")
        if (
            cuda.get("runtime") != "12.8"
            or cuda.get("device_index") != int(device.split(":", 1)[1])
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


def _git_status_lines(root: Path) -> list[str]:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot audit Git source inventory: {root}") from error
    return [line for line in completed.stdout.splitlines() if line]


def _validate_source_inventory(source_root: Path) -> dict[str, Any]:
    cache_pattern = re.compile(
        r"^\?\? (?:[^/]+/)*__pycache__/[A-Za-z0-9_.-]+\.pyc$"
    )
    status = _git_status_lines(source_root)
    non_cache = [line for line in status if cache_pattern.fullmatch(line) is None]
    if non_cache:
        raise ValueError(f"NPR source inventory drifted: {non_cache[:3]}")
    return {
        "tracked_and_non_cache_untracked_clean": True,
        "untracked_bytecode_caches_ignored": len(status),
        "bytecode_cache_execution": False,
        "loader": "compile_verified_utf8_source_bytes_no_pyc",
    }


def _import_verified_resnet(source_root: Path) -> Any:
    path = _require_regular_file(
        source_root / "networks" / "resnet.py",
        "NPR resnet source",
    )
    expected = legacy.SOURCE_FILES["networks/resnet.py"]
    if sha256_file(path) != expected:
        raise ValueError("NPR resnet source SHA-256 changed before compilation")
    payload = path.read_bytes()
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("NPR resnet source is not UTF-8") from error
    module_name = f"_claimforge_npr_verified_{legacy.MODEL_SOURCE_COMMIT[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    exec(code, module.__dict__)
    if module.__dict__.get("__cached__") is not None:
        raise ValueError("verified NPR source unexpectedly acquired bytecode cache")
    return module


def verify_assets(
    *,
    source_root: Path,
    checkpoint_path: Path,
    hf_source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], OrderedDict[str, Any], Any]:
    import torch

    if source_root.is_symlink() or not source_root.is_dir():
        raise FileNotFoundError(f"missing or unsafe NPR source-root: {source_root}")
    if hf_source_root.is_symlink() or not hf_source_root.is_dir():
        raise FileNotFoundError(
            f"missing or unsafe NPR Hugging Face source-root: {hf_source_root}"
        )
    source = legacy._verify_source_contract(source_root)
    source["inventory"] = _validate_source_inventory(source_root)
    hf_source = legacy._verify_hf_source_contract(hf_source_root)
    hf_status = _git_status_lines(hf_source_root)
    if hf_status:
        raise ValueError(f"NPR Hugging Face source inventory drifted: {hf_status[:3]}")
    hf_source["full_inventory_clean"] = True
    source["hf_space"] = hf_source
    if source.get("license_record") != legacy.LICENSE_RECORD:
        raise ValueError("NPR source license disclosure drifted")

    checkpoint_path = checkpoint_path.resolve()
    _require_regular_file(checkpoint_path, "NPR checkpoint")
    if (
        checkpoint_path.stat().st_size != int(legacy.CHECKPOINT["bytes"])
        or sha256_file(checkpoint_path) != legacy.CHECKPOINT["sha256"]
    ):
        raise ValueError("NPR checkpoint size/SHA-256 changed")
    unsafe = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    )
    if unsafe:
        raise ValueError(f"NPR checkpoint contains unsafe globals: {unsafe}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, OrderedDict):
        raise ValueError("NPR checkpoint is not the frozen flat OrderedDict")
    state = OrderedDict(payload)
    schema = legacy._checkpoint_schema(state)
    if (
        schema.get("entries") != legacy.CHECKPOINT["state_entries"]
        or schema.get("elements") != legacy.CHECKPOINT["state_elements"]
        or schema.get("items_sha256") != EXPECTED_CHECKPOINT_SCHEMA_SHA256
    ):
        raise ValueError("NPR checkpoint tensor schema changed")
    module = _import_verified_resnet(source_root)
    reference = module.resnet50(num_classes=1)
    if list(state) != list(reference.state_dict()):
        raise ValueError("NPR checkpoint key order does not match verified model")
    incompatible = reference.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("NPR strict checkpoint load reported incompatible keys")
    parameters = sum(int(value.numel()) for value in reference.parameters())
    if parameters != legacy.CHECKPOINT["trainable_parameters"]:
        raise ValueError("NPR model parameter count changed")
    del reference
    assets = {
        "checkpoint": {
            **legacy.CHECKPOINT,
            "path": str(checkpoint_path),
            "actual_bytes": checkpoint_path.stat().st_size,
            "actual_sha256": sha256_file(checkpoint_path),
            "serialization_safety": {
                "unsafe_globals": unsafe,
                "weights_only": True,
                "map_location": "cpu",
            },
            "schema": schema,
        },
        "excluded_release_assets": legacy.EXCLUDED_RELEASE_ASSETS,
        "bundle_sha256": EXPECTED_ASSET_BUNDLE_SHA256,
        "license": legacy.LICENSE_RECORD,
        "redistribution": False,
        "commercial_clearance_claimed": False,
    }
    recomputed_bundle = hashlib.sha256(
        stable_json(
            {
                "source_commit": legacy.MODEL_SOURCE_COMMIT,
                "source_files": legacy.SOURCE_FILES,
                "hf_space_commit": legacy.HF_SPACE_COMMIT,
                "hf_source_files": legacy.HF_SOURCE_FILES,
                "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
                "checkpoint_schema": schema,
            }
        ).encode("utf-8")
    ).hexdigest()
    if recomputed_bundle != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("NPR asset bundle identity changed")
    return source, assets, state, module


def effective_native_size(width: int, height: int) -> tuple[int, int]:
    return legacy.effective_native_size(width, height)


def _preprocess_image(path: Path) -> tuple[Any, dict[str, Any]]:
    tensor, legacy_audit = legacy.preprocess_image(path)
    if legacy_audit.get("profile") != legacy.PREPROCESS_PROFILE:
        raise ValueError("legacy NPR preprocessing profile changed")
    audit = dict(legacy_audit)
    audit["profile"] = FROZEN_PROFILE
    return tensor, audit


def _load_model(
    *,
    module: Any,
    state: OrderedDict[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if not isinstance(state, OrderedDict):
        raise ValueError("NPR checkpoint state is not an OrderedDict")
    model = module.resnet50(num_classes=1)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("NPR model strict load reported incompatible keys")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    if model.training or any(child.training for child in model.modules()):
        raise ValueError("NPR model did not enter recursive eval mode")
    audit = {
        "class_module": module.__name__,
        "class_name": type(model).__name__,
        "construction_api": "verified_source_bytes.resnet50(num_classes=1)",
        "source_loader": "compile_verified_utf8_source_bytes_no_pyc",
        "checkpoint_load": {
            "api": "torch.load",
            "weights_only": True,
            "map_location": "cpu",
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "model_mode": "eval",
        "feature_dimension": legacy.FEATURE_DIMENSION,
        "parameters": sum(int(value.numel()) for value in model.parameters()),
        "network_access": False,
    }
    return model, audit


def _infer_one(
    *,
    model: Any,
    tensor: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, int, float]:
    processed, feature = legacy._infer_one(
        model=model,
        tensor=tensor,
        device=device,
    )
    payload = dict(processed)
    latency = float(payload.pop("latency_ms"))
    peak_value = payload.pop("peak_cuda_memory_bytes")
    peak = 0 if peak_value is None else int(peak_value)
    return payload, feature, peak, latency


def _expected_golden_preprocess() -> dict[str, Any]:
    return {
        "profile": FROZEN_PROFILE,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
            "trim_last_row_if_height_odd",
            "trim_last_column_if_width_odd",
        ],
        "decoded_size": [1285, 1137],
        "decoded_rgb_shape": [1137, 1285, 3],
        "decoded_rgb_dtype": "uint8",
        "decoded_rgb_sha256": CPU_GOLDEN_DECODED_RGB_SHA256,
        "effective_size": [1284, 1136],
        "trim_bottom": 1,
        "trim_right": 1,
        "tensor_shape": [3, 1136, 1284],
        "tensor_dtype": "float32",
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
        "npr_residual_shape": [3, 1136, 1284],
        "npr_residual_dtype": "float32",
        "npr_residual_sha256": CPU_GOLDEN_RESIDUAL_SHA256,
        "npr_residual_stats": {
            "minimum": -3.834033489227295,
            "maximum": 3.939075469970703,
            "mean": -0.0007315525540249102,
            "mean_absolute": 0.08016720433814951,
            "l2": 391.39278433069876,
            "nonzero_elements": 2764595,
            "elements": 4375872,
        },
        "normalization": {
            "mean": list(legacy.IMAGE_MEAN),
            "std": list(legacy.IMAGE_STD),
        },
    }


def _expected_golden_processed() -> dict[str, Any]:
    decision = False
    return {
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "probability": CPU_GOLDEN_PROBABILITY,
        "ai_score": CPU_GOLDEN_PROBABILITY,
        "score": CPU_GOLDEN_PROBABILITY,
        "score_semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
        "classification_decision": decision,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": {
            "raw_logit": CPU_GOLDEN_RAW_LOGIT,
            "probability": CPU_GOLDEN_PROBABILITY,
            "ai_score": CPU_GOLDEN_PROBABILITY,
            "score": CPU_GOLDEN_PROBABILITY,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
            "decision": decision,
            "semantics": (
                "official_float32_sigmoid_probability_higher_is_fake"
            ),
        },
        "t1": {
            "raw_logit": CPU_GOLDEN_RAW_LOGIT,
            "probability": CPU_GOLDEN_PROBABILITY,
            "ai_score": CPU_GOLDEN_PROBABILITY,
            "score": CPU_GOLDEN_PROBABILITY,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
            "decision": decision,
            "policy": "official_NPR_AIGC_float32_sigmoid",
        },
        "manual_replay": {
            "raw_logit": CPU_GOLDEN_RAW_LOGIT,
            "probability": CPU_GOLDEN_PROBABILITY,
            "ai_score": CPU_GOLDEN_PROBABILITY,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def _validate_golden_forward(
    processed: Mapping[str, Any],
    feature: np.ndarray,
    *,
    label: str,
) -> tuple[bytes, str]:
    if dict(processed) != _expected_golden_processed():
        raise ValueError(f"{label} exact score/full-forward evidence changed")
    if (
        not isinstance(feature, np.ndarray)
        or feature.shape != (legacy.FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or not feature.flags.c_contiguous
        or not np.isfinite(feature).all()
        or feature.nbytes != 2048
    ):
        raise ValueError(f"{label} feature changed")
    array_sha = _array_sha256(feature)
    if array_sha != CPU_GOLDEN_FEATURE_ARRAY_SHA256:
        raise ValueError(f"{label} feature array SHA-256 changed")
    payload = _npy_bytes(feature)
    if (
        len(payload) != 2176
        or hashlib.sha256(payload).hexdigest() != CPU_GOLDEN_FEATURE_FILE_SHA256
    ):
        raise ValueError(f"{label} feature NPY bytes changed")
    return payload, array_sha


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    checkpoint_path: Path,
    hf_source_root: Path,
) -> dict[str, Any]:
    import torch

    root = repo_root.resolve()
    image_path = _require_regular_file(
        root / CPU_GOLDEN_INPUT_PATH,
        "NPR CPU golden image",
    )
    if sha256_file(image_path) != CPU_GOLDEN_IMAGE_SHA256:
        raise ValueError("NPR CPU golden input SHA-256 changed")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    initialized_before = bool(torch.cuda.is_initialized())
    if initialized_before:
        raise RuntimeError("CUDA was initialized before NPR CPU golden")
    source, assets, state, module = verify_assets(
        source_root=source_root.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        hf_source_root=hf_source_root.resolve(),
    )
    tensor, preprocess = _preprocess_image(image_path)
    if preprocess != _expected_golden_preprocess():
        raise ValueError("NPR CPU golden preprocessing evidence changed")
    model = None
    try:
        model, model_load = _load_model(
            module=module,
            state=state,
            device=device,
        )
        first, first_feature, _first_peak, _first_latency = _infer_one(
            model=model,
            tensor=tensor,
            device=device,
        )
        second, second_feature, _second_peak, _second_latency = _infer_one(
            model=model,
            tensor=tensor,
            device=device,
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
            raise ValueError("NPR CPU golden repeated forwards are not exact")
        initialized_after = bool(torch.cuda.is_initialized())
        if initialized_after:
            raise RuntimeError("NPR CPU golden initialized CUDA")
        golden = {
            "sample_id": CPU_GOLDEN_SAMPLE_ID,
            "input_path": CPU_GOLDEN_INPUT_PATH,
            "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
            "input_width": 1285,
            "input_height": 1137,
            "effective_width": 1284,
            "effective_height": 1136,
            "trim_bottom": 1,
            "trim_right": 1,
            "preprocess": preprocess,
            "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
            "npr_residual_sha256": CPU_GOLDEN_RESIDUAL_SHA256,
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
            "repeat_classification_decision": second[
                "classification_decision"
            ],
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
            "model_load": model_load,
            "runtime": runtime,
            "golden": golden,
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "cuda_initialized_before_cpu_model_load": initialized_before,
            "cuda_initialized_after_cpu_forwards": initialized_after,
            "dataset_manifest_loaded": False,
        }
    finally:
        if model is not None:
            del model
        del state, module
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
_ALLOWED_DIAGNOSTIC_KEYS = frozenset(
    {
        "pixel_center_mapping",
        "gt_mask_kind",
    }
)


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
                raise ValueError(f"{child_label} is an unsupported NPR claim")
            if normalized in _FALSE_DECLARATIONS:
                if child is not False:
                    raise ValueError(f"{child_label} is an unsupported NPR claim")
                continue
            if normalized in _NULL_DECLARATIONS:
                if child is not None:
                    raise ValueError(f"{child_label} is an unsupported NPR claim")
                continue
            if normalized in _FORBIDDEN_CLAIM_KEYS or normalized.startswith(
                _FORBIDDEN_CLAIM_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported NPR claim")
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
    """Measure exact local-edit GT surviving the final-row/column completion."""

    width = int(row["width"])
    height = int(row["height"])
    effective_width, effective_height = effective_native_size(width, height)
    geometry = {
        "profile_id": FROZEN_PROFILE,
        "effective_native_xyxy": [0, 0, effective_width, effective_height],
        "decoded_size": [width, height],
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "pixel_center_mapping": "identity_then_final_bottom_right_parity_trim",
    }
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        gt = legacy._load_gt_mask(row, repo_root)
        if gt is None:
            raise ValueError("local exact-diff input has no GT mask")
        total = int(np.count_nonzero(gt == 255))
        visible = int(
            np.count_nonzero(gt[:effective_height, :effective_width] == 255)
        )
        if total <= 0 or not 0 <= visible <= total:
            raise ValueError("local exact-diff visibility counts are invalid")
        fraction = visible / total
        category = (
            "none"
            if visible == 0
            else "full"
            if visible == total
            else "partial"
        )
        return {
            "edit_visibility": category,
            "edit_visible_gt_fraction": fraction,
            "edit_visibility_evidence": {
                **geometry,
                "basis": (
                    "exact_diff_mask_intersection_with_author_documented_"
                    "native_even_trim_completion"
                ),
                "gt_mask_kind": gt_kind,
                "total_positive_pixels": total,
                "visible_positive_pixel_centers": visible,
                "removed_positive_pixel_centers": total - visible,
            },
        }
    if gt_kind not in ("all_zero", "not_applicable"):
        raise ValueError("Balanced250 input has unsupported GT kind")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            **geometry,
            "basis": (
                "authentic_input_has_no_edit"
                if gt_kind == "all_zero"
                else "fullframe_condition_has_no_local_GT"
            ),
            "gt_mask_kind": gt_kind,
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
    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    if asset_bundle_sha256 != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("NPR asset bundle SHA-256 changed")
    identity = build_result_identity(
        row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
    )
    input_path = _anchored(Path(str(row["canonical_path"])), repo_root)
    if repo_relative(input_path, repo_root) != identity["input_path"]:
        raise ValueError("Balanced250 input path is not repository-local")
    result = {
        **identity,
        "valid_for_metrics": valid_for_metrics,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "model_arch": legacy.MODEL_ARCH,
        "model_source_commit": legacy.MODEL_SOURCE_COMMIT,
        "checkpoint_id": legacy.CHECKPOINT["id"],
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
    _reject_unsupported_claims(result, "result identity")
    return result


_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "npr_feature",
        "npr_feature_path",
        "npr_feature_sha256",
        "npr_feature_array_sha256",
        "npr_feature_shape",
        "npr_feature_dtype",
        "npr_feature_nbytes",
        "npr_feature_semantics",
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
    # Do not reconstruct sigmoid(raw) on the host CPU.  The persisted
    # probability is authoritative only after exact feature/head/sigmoid
    # replay on the recorded inference device.
    if ai_score != probability or score != probability:
        raise ValueError(f"{sample_id} score orientation/aliases changed")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
    }
    t1 = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "policy": "official_NPR_AIGC_float32_sigmoid",
    }
    manual = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
    }
    if (
        row.get("score_semantics")
        != "official_float32_sigmoid_probability_higher_is_fake"
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold")
        != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or row.get("classification") != classification
        or row.get("t1") != t1
        or row.get("manual_replay") != manual
    ):
        raise ValueError(f"{sample_id} score aliases/manual replay changed")


def _validate_feature_artifact(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    repo_root: Path,
    run_id: str,
) -> np.ndarray:
    feature = row.get("npr_feature")
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
        "visibility",
    }
    if not isinstance(feature, Mapping) or set(feature) != expected_keys:
        raise ValueError(f"{sample_id} NPR feature key set changed")
    expected_relative = (
        DEFAULT_ARTIFACTS_DIR / run_id / "features" / f"{sample_id}.npy"
    ).as_posix()
    if feature.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} NPR feature path is not canonical")
    path = _safe_repo_file(
        feature["relative_path"],
        repo_root=repo_root,
        label=f"{sample_id} NPR feature",
    )
    payload = path.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    expected_nbytes = legacy.FEATURE_DIMENSION * np.dtype(np.float32).itemsize
    semantics = "official_fc1_input_after_adaptive_global_average_pool"
    if (
        feature.get("sha256") != file_sha
        or feature.get("file_bytes") != len(payload)
        or feature.get("dtype") != "float32"
        or feature.get("shape") != [legacy.FEATURE_DIMENSION]
        or feature.get("nbytes") != expected_nbytes
        or feature.get("finite") is not True
        or feature.get("semantics") != semantics
        or feature.get("visibility") != "local_only_gitignored_output"
    ):
        raise ValueError(f"{sample_id} NPR feature metadata/hash changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} NPR feature NPY is invalid") from error
    if (
        not isinstance(array, np.ndarray)
        or array.shape != (legacy.FEATURE_DIMENSION,)
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or array.nbytes != expected_nbytes
        or payload != _npy_bytes(array)
    ):
        raise ValueError(f"{sample_id} NPR feature array changed")
    array_sha = _array_sha256(array)
    if feature.get("array_sha256") != array_sha:
        raise ValueError(f"{sample_id} NPR feature array SHA-256 changed")
    aliases = {
        "npr_feature_path": expected_relative,
        "npr_feature_sha256": file_sha,
        "npr_feature_array_sha256": array_sha,
        "npr_feature_shape": [legacy.FEATURE_DIMENSION],
        "npr_feature_dtype": "float32",
        "npr_feature_nbytes": int(array.nbytes),
        "npr_feature_semantics": semantics,
        "artifact_paths": {"npr_feature_npy": expected_relative},
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} NPR feature alias {key} changed")
    return array


def _validate_latest_feature_head_replay(
    *,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    checkpoint_state: Mapping[str, Any],
    device: Any,
) -> int:
    """Replay every latest successful feature on the recorded device."""

    import torch
    from torch.nn import functional as functional

    if (
        not isinstance(device, torch.device)
        or device.type not in ("cpu", "cuda")
        or (device.type == "cuda" and device.index is None)
        or (device.type == "cpu" and device.index is not None)
    ):
        raise ValueError("NPR head replay requires an explicit configured device")
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("NPR audited checkpoint state is not a mapping")
    weight = checkpoint_state.get("fc1.weight")
    bias = checkpoint_state.get("fc1.bias")
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
        raise ValueError("NPR audited fc1 tensor schema changed")
    replay_weight = weight.detach().to(device=device)
    replay_bias = bias.detach().to(device=device)
    expected = sum(row.get("status") == "ok" for row in latest_by_sample_id.values())
    replayed = 0
    try:
        with torch.inference_mode():
            for sample_id, row in latest_by_sample_id.items():
                if row.get("status") != "ok":
                    continue
                array = _validate_feature_artifact(
                    row,
                    sample_id=sample_id,
                    repo_root=repo_root,
                    run_id=run_id,
                )
                replay_feature = (
                    torch.from_numpy(array).reshape(1, -1).to(device=device)
                )
                output = functional.linear(
                    replay_feature,
                    replay_weight,
                    replay_bias,
                )
                probability = torch.sigmoid(output)
                raw_value = float(output.reshape(()).item())
                probability_value = float(probability.reshape(()).item())
                if (
                    abs(raw_value - float(row["raw_logit"]))
                    > HEAD_REPLAY_RAW_LOGIT_ABS_TOLERANCE
                ):
                    raise ValueError(
                        f"{sample_id} independent fc1 replay logit mismatch"
                    )
                if (
                    abs(probability_value - float(row["probability"]))
                    > HEAD_REPLAY_PROBABILITY_ABS_TOLERANCE
                ):
                    raise ValueError(
                        f"{sample_id} independent sigmoid replay mismatch"
                    )
                decision = probability_value > legacy.CLASSIFICATION_THRESHOLD
                if row.get("classification_decision") is not decision:
                    raise ValueError(
                        f"{sample_id} independent replay decision mismatch"
                    )
                replayed += 1
                del replay_feature, output, probability
    finally:
        del replay_weight, replay_bias
    if replayed != expected:
        raise ValueError("NPR independent head replay coverage is incomplete")
    return replayed


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("NPR result attempt has invalid status")
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
            "NPR result attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, expected_value in expected.items():
        if attempt.get(key) != expected_value:
            raise ValueError(f"NPR result attempt field {key} drifted")
    if not isinstance(attempt.get("completed_at"), str) or not attempt["completed_at"]:
        raise ValueError("NPR result attempt completed_at is invalid")
    _reject_unsupported_claims(attempt, "result attempt")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt["traceback"]
        ):
            raise ValueError("NPR error result payload is invalid")
        return
    sample_id = str(input_row["sample_id"])
    _validate_score_payload(attempt, sample_id=sample_id)
    input_path = _safe_repo_file(
        str(input_row["canonical_path"]),
        repo_root=repo_root,
        label=f"{sample_id} canonical input",
    )
    _tensor, expected_preprocess = _preprocess_image(input_path)
    if attempt.get("preprocess") != expected_preprocess:
        raise ValueError(f"{sample_id} NPR preprocessing record changed")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _finite_number(attempt.get(field), f"{sample_id} {field}") < 0.0:
            raise ValueError(f"{sample_id} {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{sample_id} peak CUDA memory is invalid")
    _validate_feature_artifact(
        attempt,
        sample_id=sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )


def _validate_physical_attempt_history(
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Allow error retries, but never a second/post-success physical attempt."""

    statuses_by_sample: dict[str, list[str]] = {}
    for attempt in attempts:
        sample_id = str(attempt.get("sample_id"))
        status = str(attempt.get("status"))
        statuses_by_sample.setdefault(sample_id, []).append(status)
    for sample_id, statuses in statuses_by_sample.items():
        successful = [index for index, status in enumerate(statuses) if status == "ok"]
        if len(successful) > 1:
            raise ValueError(
                f"duplicate successful physical attempts for {sample_id}"
            )
        if successful and successful[0] != len(statuses) - 1:
            raise ValueError(
                f"physical attempt exists after success for {sample_id}"
            )


def _local_artifact_policy(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    ignore_path = _require_regular_file(root / ".gitignore", ".gitignore")
    probe = (
        DEFAULT_ARTIFACTS_DIR
        / "_claimforge_ignore_probe"
        / "features"
        / "probe.npy"
    ).as_posix()
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", probe],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("NPR feature output is not covered by .gitignore") from error
    evidence = completed.stdout.strip()
    if (
        not evidence.startswith(".gitignore:")
        or "\t" not in evidence
        or not evidence.endswith(probe)
    ):
        raise ValueError("NPR git-ignore evidence changed")
    return {
        "visibility": "local_only",
        "artifact_root": DEFAULT_ARTIFACTS_DIR.as_posix(),
        "gitignored": True,
        "gitignore_path": ".gitignore",
        "gitignore_sha256": sha256_file(ignore_path),
        "git_check_ignore_probe": probe,
        "git_check_ignore_evidence": evidence,
        "publication": False,
        "checkpoint_redistribution": False,
        "commercial_clearance_claimed": False,
    }


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
    run_id = _valid_run_id(run_id)
    validate_runtime_contract(runtime, label="immutable runtime")
    if source.get("commit") != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("NPR source audit commit changed")
    hf_source = source.get("hf_space")
    if (
        not isinstance(hf_source, Mapping)
        or hf_source.get("commit") != legacy.HF_SPACE_COMMIT
    ):
        raise ValueError("NPR Hugging Face source audit commit changed")
    if (
        assets.get("bundle_sha256") != EXPECTED_ASSET_BUNDLE_SHA256
        or assets.get("license") != legacy.LICENSE_RECORD
        or assets.get("commercial_clearance_claimed") is not False
    ):
        raise ValueError("NPR asset/license identity changed")
    local_policy = _local_artifact_policy(repo_root)
    immutable = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": MODEL_CONTRACT,
        "source_completion": SOURCE_COMPLETION_CONTRACT,
        "preprocess": PREPROCESS_CONTRACT,
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "odd_dimension_counts": odd_dimension_counts(selected),
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "local_artifact_policy": local_policy,
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
        raise AssertionError("internal immutable NPR config key set drifted")
    _reject_unsupported_claims(immutable, "immutable config")
    return immutable


def _ensure_repo_child(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    require_directory: bool = False,
) -> Path:
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
        label="NPR run directory",
    )
    feature_root = _ensure_repo_child(
        feature_root,
        repo_root=repo_root,
        label="NPR feature artifact root",
    )
    if (
        run_dir == feature_root
        or run_dir.is_relative_to(feature_root)
        or feature_root.is_relative_to(run_dir)
    ):
        raise ValueError("NPR result and feature directories must be disjoint")
    feature_dir = feature_root / "features"
    if not resume:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"run directory is non-empty; pass --resume: {run_dir}"
            )
        if feature_root.exists() and (
            not feature_root.is_dir() or any(feature_root.iterdir())
        ):
            raise FileExistsError(
                f"feature artifact root is non-empty; pass --resume: {feature_root}"
            )
    else:
        if not run_dir.is_dir() or not feature_dir.is_dir():
            raise FileNotFoundError(
                "resume requires the NPR run and feature directories"
            )
        entries = list(feature_root.iterdir())
        if (
            len(entries) != 1
            or entries[0].name != "features"
            or entries[0].is_symlink()
            or not entries[0].is_dir()
        ):
            raise ValueError("resume NPR feature artifact inventory changed")
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_child(
        feature_dir,
        repo_root=repo_root,
        label="NPR feature directory",
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
        label="NPR feature directory",
        require_directory=True,
    )
    entries = list(feature_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"NPR feature inventory contains unsafe entry {entry.name}"
            )
    expected = {
        f"{sample_id}.npy"
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "NPR feature inventory mismatch: "
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
        raise ValueError("official NPR feature array contract changed")
    payload = feature_path.read_bytes()
    if payload != _npy_bytes(feature):
        raise ValueError("persisted NPR feature is not canonical NumPy bytes")
    return {
        "relative_path": repo_relative(feature_path, repo_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "array_sha256": _array_sha256(feature),
        "dtype": "float32",
        "shape": [legacy.FEATURE_DIMENSION],
        "nbytes": int(feature.nbytes),
        "finite": True,
        "semantics": (
            "official_fc1_input_after_adaptive_global_average_pool"
        ),
        "visibility": "local_only_gitignored_output",
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
    record = _feature_record(
        feature=feature,
        feature_path=feature_path,
        repo_root=repo_root,
    )
    raw = processed["raw_logit"]
    probability = processed["probability"]
    decision = processed["classification_decision"]
    relative = record["relative_path"]
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
        "npr_feature": record,
        "npr_feature_path": relative,
        "npr_feature_sha256": record["sha256"],
        "npr_feature_array_sha256": record["array_sha256"],
        "npr_feature_shape": [legacy.FEATURE_DIMENSION],
        "npr_feature_dtype": "float32",
        "npr_feature_nbytes": record["nbytes"],
        "npr_feature_semantics": record["semantics"],
        "artifact_paths": {"npr_feature_npy": relative},
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
        "classification_decision": decision,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": dict(processed["classification"]),
        "t1": dict(processed["t1"]),
        "manual_replay": dict(processed["manual_replay"]),
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


def _expected_golden_record() -> dict[str, Any]:
    return {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1285,
        "input_height": 1137,
        "effective_width": 1284,
        "effective_height": 1136,
        "trim_bottom": 1,
        "trim_right": 1,
        "preprocess": _expected_golden_preprocess(),
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
        "npr_residual_sha256": CPU_GOLDEN_RESIDUAL_SHA256,
        "feature_file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_file_bytes": 2176,
        "feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "feature_shape": [legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": 2048,
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "probability": CPU_GOLDEN_PROBABILITY,
        "ai_score": CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "full_image_forward": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "repeat_feature_file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        "repeat_feature_file_bytes": 2176,
        "repeat_feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "repeat_raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": CPU_GOLDEN_PROBABILITY,
        "repeat_ai_score": CPU_GOLDEN_PROBABILITY,
        "repeat_classification_decision": False,
        "repeat_full_image_forward": True,
        "repeat_model_forward_calls": 1,
        "repeat_fc_hook_calls": 1,
        "repeat_byte_exact": True,
    }


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
        raise ValueError("NPR CPU preflight report key set changed")
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
        raise ValueError("NPR CPU preflight report/provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("NPR CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    model_load = report.get("model_load")
    expected_model_load_keys = {
        "class_module",
        "class_name",
        "construction_api",
        "source_loader",
        "checkpoint_load",
        "model_mode",
        "feature_dimension",
        "parameters",
        "network_access",
    }
    if (
        not isinstance(model_load, Mapping)
        or set(model_load) != expected_model_load_keys
        or model_load.get("class_module")
        != f"_claimforge_npr_verified_{legacy.MODEL_SOURCE_COMMIT[:12]}"
        or model_load.get("class_name") != "ResNet"
        or model_load.get("construction_api")
        != "verified_source_bytes.resnet50(num_classes=1)"
        or model_load.get("source_loader")
        != "compile_verified_utf8_source_bytes_no_pyc"
        or model_load.get("checkpoint_load")
        != {
            "api": "torch.load",
            "weights_only": True,
            "map_location": "cpu",
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        or model_load.get("model_mode") != "eval"
        or model_load.get("feature_dimension") != legacy.FEATURE_DIMENSION
        or model_load.get("parameters")
        != legacy.CHECKPOINT["trainable_parameters"]
        or model_load.get("network_access") is not False
    ):
        raise ValueError("NPR CPU preflight model-load evidence changed")
    golden = report.get("golden")
    if not isinstance(golden, Mapping) or dict(golden) != _expected_golden_record():
        raise ValueError("NPR CPU golden exact two-forward evidence changed")


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
        "--hf-source-root",
        type=Path,
        default=DEFAULT_HF_SOURCE_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
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
    """Execute one append-only NPR Balanced250 run."""

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(Path(args.source_root), repo_root)
    hf_source_root = _anchored(Path(args.hf_source_root), repo_root)
    checkpoint_path = _anchored(Path(args.checkpoint), repo_root)
    mode = str(args.mode)
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed != DEFAULT_SEED
    ):
        raise ValueError(f"NPR seed must be exactly {DEFAULT_SEED}")
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
            checkpoint_path=checkpoint_path,
            hf_source_root=hf_source_root,
        )
        source, assets, state, module = verify_assets(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            hf_source_root=hf_source_root,
        )
        del state, module
        _validate_preflight_report(report, source=source, assets=assets)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if mode != "formal" and args.run_id is None:
        raise ValueError("smoke and single modes require explicit --run-id")
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _ensure_repo_child(
        _anchored(Path(args.dataset_manifest), repo_root),
        repo_root=repo_root,
        label="Balanced250 dataset manifest",
    )
    results_root = _ensure_repo_child(
        Path(args.results_dir),
        repo_root=repo_root,
        label="NPR results root",
    )
    artifacts_root = _ensure_repo_child(
        Path(args.artifacts_dir),
        repo_root=repo_root,
        label="NPR artifacts root",
    )
    if repo_relative(artifacts_root, repo_root) != DEFAULT_ARTIFACTS_DIR.as_posix():
        raise ValueError(f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}")
    run_dir = results_root / run_id
    feature_root = artifacts_root / run_id
    feature_dir = feature_root / "features"
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

    # CPU golden runs before requested accelerator configuration.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_source_root=hf_source_root,
    )
    source, assets, state, module = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_source_root=hf_source_root,
    )
    _validate_preflight_report(cpu_preflight, source=source, assets=assets)
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

    prior_status: Any = None
    prior_outputs: Mapping[str, Any] = {}
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
            raise ValueError("resume NPR run manifest fingerprint/config drifted")
        if _read_jsonl_strict(expected_path, "expected inputs") != selected:
            raise ValueError("resume NPR expected-input snapshot drifted")
        expected_dataset = {
            "contract": dataset_contract.as_dict(),
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        }
        if prior_manifest.get("dataset") != expected_dataset:
            raise ValueError("resume NPR manifest dataset evidence drifted")
        prior_outputs_value = prior_manifest.get("outputs")
        if not isinstance(prior_outputs_value, Mapping):
            raise ValueError("resume NPR manifest outputs are invalid")
        prior_outputs = prior_outputs_value
        if prior_status == "running":
            if dict(prior_outputs) != immutable["outputs"]:
                raise ValueError("running NPR resume output contract drifted")
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
                raise ValueError("finalized NPR resume output evidence drifted")
        started_at = prior_manifest.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError("resume NPR manifest started_at is invalid")
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
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
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
    _validate_physical_attempt_history(physical_before)
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
            raise ValueError("prior NPR result is outside the selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    prior_feature_files = _validate_feature_inventory(
        feature_dir=feature_dir,
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    if (
        args.resume
        and prior_status in ("complete", "incomplete")
        and prior_outputs.get("feature_files") != prior_feature_files
    ):
        raise ValueError("finalized NPR resume feature count drifted")
    _validate_latest_feature_head_replay(
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        checkpoint_state=state,
        device=device,
    )
    # Existing history and artifacts pass before the manifest is mutated.
    atomic_write_json(manifest_path, manifest)

    pending_ids = set(latest_before.pending_sample_ids(retry_errors=True))
    new_successes = 0
    resume_skips = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    model = None
    try:
        if pending_ids:
            model, model_load = _load_model(
                module=module,
                state=state,
                device=device,
            )
            if model_load != cpu_preflight["model_load"]:
                raise ValueError("NPR formal model-load contract differs from CPU gate")
        for index, input_row in enumerate(selected, start=1):
            sample_id = str(input_row["sample_id"])
            if sample_id not in pending_ids:
                resume_skips += 1
                print(f"[{index}/{len(selected)}] resume {sample_id}", flush=True)
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
                tensor, preprocess = _preprocess_image(input_path)
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if preprocess.get("decoded_size") != [
                    int(input_row["width"]),
                    int(input_row["height"]),
                ]:
                    raise ValueError("NPR preprocessed image dimensions changed")
                expected_width, expected_height = effective_native_size(
                    int(input_row["width"]),
                    int(input_row["height"]),
                )
                if preprocess.get("effective_size") != [
                    expected_width,
                    expected_height,
                ]:
                    raise ValueError("NPR effective native dimensions changed")
                if model is None:
                    raise RuntimeError("NPR model was not loaded")
                processed, feature, peak, latency = _infer_one(
                    model=model,
                    tensor=tensor,
                    device=device,
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
                    f"ai_score={result['ai_score']:.9g} "
                    f"raw_logit={result['raw_logit']:.9g}",
                    flush=True,
                )
            except Exception as error:
                if feature_path.exists():
                    if feature_path.is_symlink() or not feature_path.is_file():
                        raise ValueError(
                            f"unsafe failed NPR feature path: {feature_path}"
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

    physical_results = _read_jsonl_strict(results_path, "physical results")
    _validate_physical_attempt_history(physical_results)
    for attempt in physical_results:
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("NPR result is outside the selection")
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
    latest_has_error = any(
        row.get("status") == "error"
        for row in latest.latest_by_sample_id.values()
    )
    latest_all_ok = (
        len(latest.latest_by_sample_id) == len(selected)
        and all(
            row.get("status") == "ok"
            for row in latest.latest_by_sample_id.values()
        )
    )
    if coverage.is_complete is not latest_all_ok or (
        coverage.is_complete and latest_has_error
    ):
        raise ValueError("NPR latest-attempt completion invariant changed")
    feature_files = _validate_feature_inventory(
        feature_dir=feature_dir,
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    replayed_features = _validate_latest_feature_head_replay(
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        checkpoint_state=state,
        device=device,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_npr_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": FROZEN_PROFILE,
        "score_spec": SCORE_SPEC.as_dict(),
        "raw_logit_diagnostic": MODEL_CONTRACT["raw_logit_diagnostic"],
        "dataset_contract": dataset_contract.as_dict(),
        "odd_dimension_counts": odd_dimension_counts(selected),
        "coverage": coverage.as_dict(),
        "same_device_feature_head_sigmoid_replays": replayed_features,
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
        "same_device_feature_head_sigmoid_replays": replayed_features,
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
        raise RuntimeError("NPR fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
