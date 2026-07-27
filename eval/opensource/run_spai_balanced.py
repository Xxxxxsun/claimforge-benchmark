#!/usr/bin/env python3
"""Run the pinned official SPAI detector on Balanced250.

This v2 adapter leaves the legacy Mouse runner unchanged.  It executes the
released any-resolution SPAI whole-image classifier on the 1,775-image
Balanced250 score cache (or a frozen smoke/single selection), and stores three
auditable float32 arrays outside Git: per-patch frequency-restoration features,
the SCA/LayerNorm classifier feature, and the SCA attention weights.

SPAI is T1-only.  Its attention weights and the exact-difference receptive-field
visibility census are classifier/input diagnostics, not predicted masks,
pixelwise probabilities, or manipulation-localization (T2) outputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
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
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_spai as legacy
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


RUN_MANIFEST_SCHEMA = "spai_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "spai_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "spai_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "spai_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/spai")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/spai")
DEFAULT_FORMAL_RUN_ID = (
    "spai_any_resolution_spectral_balanced250_v1_full1775_20260726"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "spai_any_resolution_spectral_balanced250_v1_smoke5x7_a_20260726"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "spai_any_resolution_spectral_balanced250_v1_smoke5x7_b_20260726"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_GOLDEN_ROOT = legacy.DEFAULT_GOLDEN_ROOT
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = legacy.MODEL_SEED
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 12 * 1024**3

FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/spai/bin/python"
)
FROZEN_VENV_PREFIX = Path("/root/.cache/claimforge/venvs/spai")
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/spai-balanced-v2-empty"
)
FROZEN_PYVENV_CONFIG_SHA256 = (
    "506c01d6bc866a7500bde63a24b3f0c1fb3013df41051ad9a8bf7c42c85eb091"
)
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    **legacy.RUNTIME_VERSIONS,
    "setuptools": "79.0.1",
}
FROZEN_RUNTIME_MODULE_FILES = dict(legacy.RUNTIME_MODULE_FILES)

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
FORMAL_SELECTED_ROWS_SHA256 = (
    "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
)
FORMAL_SELECTED_IDS_SHA256 = (
    "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
)
SMOKE5X7_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

LOCAL_VISIBILITY_CENSUS = {
    "local_mouse": {
        "full": 221,
        "partial": 12,
        "none": 17,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.9074860638233333,
        "patch_modes": {"grid": 238, "five_crop": 12},
    },
    "local_cat": {
        "full": 176,
        "partial": 71,
        "none": 3,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.9255790726408315,
        "patch_modes": {"grid": 237, "five_crop": 13},
    },
    "local_trash_can": {
        "full": 75,
        "partial": 172,
        "none": 3,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.884741668388256,
        "patch_modes": {"grid": 238, "five_crop": 12},
    },
    "all_local": {
        "full": 472,
        "partial": 255,
        "none": 23,
        "total": 750,
        "mean_edit_visible_gt_fraction": 0.9059356016174737,
        "patch_modes": {"grid": 713, "five_crop": 37},
    },
}

PREPROCESS_CONTRACT = {
    "profile_id": legacy.PREPROCESS_PROFILE,
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "pad_if_needed": {
        "minimum_size": [legacy.PATCH_SIZE, legacy.PATCH_SIZE],
        "position": "center",
        "border_mode": "cv2.BORDER_REFLECT_101",
    },
    "resize": False,
    "crop": False,
    "test_augmentation": False,
    "normalization": "float32_uint8_times_float32_1_over_255",
    "patch_size": [legacy.PATCH_SIZE, legacy.PATCH_SIZE],
    "patch_stride": [legacy.PATCH_STRIDE, legacy.PATCH_STRIDE],
    "minimum_patches": legacy.MINIMUM_PATCHES,
    "fallback": "torchvision_five_crop_if_initial_grid_count_below_4",
    "remainder": "discard_nondivisible_right_and_bottom",
    "batch_size": 1,
}
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_CONTRACT = {
    "patch_features": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": ["effective_patch_count", legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "semantics": legacy.PATCH_FEATURE_SEMANTICS,
        "visibility": "local_only_gitignored_output",
    },
    "feature": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "semantics": legacy.FEATURE_SEMANTICS,
        "visibility": "local_only_gitignored_output",
    },
    "attention": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.ATTENTION_HEADS, "effective_patch_count"],
        "dtype": "float32",
        "semantics": legacy.ATTENTION_SEMANTICS,
        "valid_for_t2": False,
        "visibility": "local_only_gitignored_output",
    },
    "replay": (
        "patch_features_to_SCA_LayerNorm_complete_MLP_and_sigmoid; "
        "normalized_feature_compared_then_MLP_and_sigmoid; "
        "attention_compared_as_diagnostic_SCA_output_on_recorded_device"
    ),
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
    "checkpoint": {
        "id": legacy.CHECKPOINT["id"],
        "sha256": legacy.CHECKPOINT["sha256"],
        "bytes": legacy.CHECKPOINT["bytes"],
        "tensor_count": legacy.CHECKPOINT["tensor_count"],
        "state_elements": legacy.CHECKPOINT["state_elements"],
        "schema_items_sha256": legacy.CHECKPOINT["schema_items_sha256"],
        "loader": "torch.load(map_location=cpu, weights_only=True)",
        "safe_global_allowlist": ["yacs.config.CfgNode"],
    },
    "released_execution": {
        "config_minimum_patches": legacy.MINIMUM_PATCHES,
        "checkpoint_embedded_historical_minimum_patches": (
            legacy.CHECKPOINT["embedded_minimum_patches"]
        ),
        "paper_all_spectral_information_claim": True,
        "executable_discards_nondivisible_right_bottom_remainder": True,
        "project_attention_visualization_is_not_native_T2": True,
        "website_derivative_scores_are_not_executable_golden": True,
        "no_official_model_card_or_checkpoint_manifest": True,
    },
    "license": {
        "spai_release_record": legacy.LICENSE_RECORD,
        "commercial_clearance": "unresolved",
        "risk": "high",
        "reason": (
            "SPAI labels code and weights Apache-2.0, but acknowledges an "
            "MFM code/backbone basis whose upstream S-Lab License 1.0 is "
            "non-commercial absent contributor permission; the linked "
            "DMimageDetection training archive is nonprofit-only; and "
            "COCO/LSUN image rights are not a blanket commercial grant"
        ),
        "benchmark_does_not_redistribute_checkpoint": True,
    },
    "training_disclosure": {
        "paper": (
            "180k LDM-generated images and 180k real images from COCO/LSUN; "
            "frozen MFM ViT-B/16 backbone"
        ),
        "training_data_commercial_clearance": "not_established",
    },
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_spai_balanced.py",
    "eval/opensource/analyze_spai_balanced.py",
    "eval/opensource/run_spai.py",
    "eval/opensource/analyze_spai_run.py",
    "eval/opensource/spai_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
)
IMMUTABLE_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "model",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "selection_visibility_census",
        "formal_local_visibility_census",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "execution_device_golden",
        "artifact_contract",
        "local_artifact_policy",
        "outputs",
    }
)

CPU_GOLDEN_SAMPLE_ID = "143a80a0a7a34c757d67ff25"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/"
    f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "3625dba9fbe410d3e6f1ebaa2498cd4931072f60e8174d5b091a41d5289a710a"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "127c4c17dae8cdc304f09163d3e351571267f61aa8b9e96dfb3138bf02b2fba2"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "294a855f930cdafcbca4b46293084153051ad197d7b9e4d5576c0566e904dd84"
)
CPU_GOLDEN_PATCH_ARRAY_SHA256 = (
    "fc8c3ba429aac54a076d58438640cd760cb6ccbee4f150d6cb6fc177cbf831e1"
)
CPU_GOLDEN_PATCH_FILE_SHA256 = (
    "b88c025d0f183876e0c86a893aaf628fabdca358057d67dbb56df8dda896c807"
)
CPU_GOLDEN_FEATURE_ARRAY_SHA256 = (
    "cea88315feea8612d3f069298ec82f27449bdd16e9d41cd7b2fa6c7b2b72beda"
)
CPU_GOLDEN_FEATURE_FILE_SHA256 = (
    "e6c4bc48e2f20af307ec61dbf9f344fc70c659dc96656223bbd5ffd4a4ceb609"
)
CPU_GOLDEN_ATTENTION_ARRAY_SHA256 = (
    "85f0aec70d4f4cf80d367e4ab4a22f0395395e8b9329a1201fb821ae1877dad0"
)
CPU_GOLDEN_ATTENTION_FILE_SHA256 = (
    "306cc6e5a4a082325f898aef0bbd27a0be42a4e8dc291f61fc3e30d16462d397"
)
CPU_GOLDEN_RAW_LOGIT = -19.73525619506836
CPU_GOLDEN_PROBABILITY = 2.685883293551683e-09

CPU_OFFICIAL_GOLDEN_CASES = (
    {
        "relative_path": "midjourney-v6.1/224.png",
        "raw_logit": 0.9909074306488037,
        "probability": 0.7292671203613281,
        "patch_features_array_sha256": (
            "5e66f1d590047b626ecd673a2fd02b87a424084fa43ebc7ac82f8aeaf0b604fd"
        ),
        "feature_array_sha256": (
            "c54db66ab5c6b0aca55bb19c2b5d6a85ca9b02925d8dd089e5bef033172b1a03"
        ),
        "attention_array_sha256": (
            "74d0decaf373b15cfca8926c4ac571c3d5975a441d699c54a320754915bef73d"
        ),
    },
    {
        "relative_path": (
            "stable-diffusion-3/cfg_60/euler/steps_28/000001046_4.webp"
        ),
        "raw_logit": 1.6814380884170532,
        "probability": 0.8430948257446289,
        "patch_features_array_sha256": (
            "c74af428fbc58381af1bd9dff6e5330fc7b8715735c05012afab505618cf9ee8"
        ),
        "feature_array_sha256": (
            "cf6333aa789da7f2491777fcf4caf960d74982964cc2f0890b6fb4e819250939"
        ),
        "attention_array_sha256": (
            "948a3002a8ee55f49950b1671f7d0d9de3ff7694a5f9acc1f784ab1fe830726c"
        ),
    },
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _same_canonical_json(first: Any, second: Any) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    try:
        return stable_json(first) == stable_json(second)
    except (TypeError, ValueError):
        return False


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _expected_npy_file_bytes(shape: tuple[int, ...]) -> int:
    dtype = np.dtype(np.float32)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return len(header.getvalue()) + int(np.prod(shape)) * dtype.itemsize


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(
            "run-id must be one safe ASCII path component (max 160 chars)"
        )
    return value


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or unsafe SPAI source: {path}")
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
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("formal SPAI selection drifted")
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
    expected = {
        condition: per_condition_limit for condition in BALANCED_CONDITIONS
    }
    if dict(counts) != expected or any("pair_rank" in row for row in selected):
        raise ValueError("smoke panel does not cover all seven conditions")
    selected.sort(key=lambda row: int(row["rank"]))
    if (
        per_condition_limit == DEFAULT_SMOKE_LIMIT
        and selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != SMOKE5X7_SELECTED_IDS_SHA256
    ):
        raise ValueError("frozen SPAI 5x7 smoke selection drifted")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if release.release_kind != "balanced250":
        raise ValueError("SPAI v2 requires Balanced250")
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode does not accept input selectors")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode does not accept sample-id")
        return _smoke_selection(
            release,
            DEFAULT_SMOKE_LIMIT
            if per_condition_limit is None
            else per_condition_limit,
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


def _startup_isolation_contract() -> dict[str, Any]:
    expected_prefix = Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
    raw_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    actual_prefix = (
        Path(os.path.abspath(raw_prefix))
        if isinstance(raw_prefix, str)
        else None
    )
    if (
        os.environ.get("PYTHONHASHSEED") != str(DEFAULT_SEED)
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or actual_prefix != expected_prefix
        or expected_prefix.is_symlink()
        or (expected_prefix.exists() and not expected_prefix.is_dir())
        or (expected_prefix.is_dir() and any(expected_prefix.iterdir()))
        or sys.dont_write_bytecode is not True
        or Path(os.path.abspath(str(sys.pycache_prefix))) != expected_prefix
    ):
        raise RuntimeError(
            "SPAI startup isolation requires PYTHONHASHSEED=0, "
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


def _venv_contract() -> dict[str, Any]:
    prefix = Path(os.path.abspath(sys.prefix))
    base_prefix = Path(os.path.abspath(sys.base_prefix))
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    config_path = expected_prefix / "pyvenv.cfg"
    if (
        Path(os.path.abspath(sys.executable))
        != Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
        or prefix != expected_prefix
        or base_prefix != Path("/usr")
        or config_path.is_symlink()
        or not config_path.is_file()
        or sha256_file(config_path) != FROZEN_PYVENV_CONFIG_SHA256
    ):
        raise RuntimeError("SPAI dedicated virtual environment drifted")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        normalized = key.strip()
        if separator != "=" or not normalized or normalized in values:
            raise RuntimeError("SPAI pyvenv.cfg is malformed")
        values[normalized] = value.strip()
    expected = {
        "home": "/usr/bin",
        "include-system-site-packages": "true",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": (
            "/usr/bin/python -m venv --system-site-packages "
            "/root/.cache/claimforge/venvs/spai"
        ),
    }
    if values != expected:
        raise RuntimeError("SPAI pyvenv.cfg values drifted")
    return {
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "pyvenv_cfg_path": str(config_path),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }


def _runtime_versions_and_modules() -> tuple[
    dict[str, str],
    dict[str, dict[str, str]],
]:
    actual_versions = {
        "python": platform.python_version(),
        **{
            name: str(_package_version(name))
            for name in legacy.RUNTIME_VERSIONS
        },
        "setuptools": str(_package_version("setuptools")),
    }
    if actual_versions != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"SPAI runtime versions drifted: {actual_versions}"
        )
    modules: dict[str, dict[str, str]] = {}
    for name, expected_hash in FROZEN_RUNTIME_MODULE_FILES.items():
        module = importlib.import_module(name)
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            raise RuntimeError(f"SPAI runtime module has no file: {name}")
        path = Path(value).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected_hash
        ):
            raise RuntimeError(f"SPAI runtime module drifted: {name}")
        modules[name] = {
            "path": str(path),
            "sha256": expected_hash,
        }
    return actual_versions, modules


def configure_runtime(
    device_text: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the dedicated environment and official numerical settings."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"SPAI seed must be exactly {DEFAULT_SEED}")
    environment = _startup_isolation_contract()
    current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current_workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current_workspace != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG}"
        )
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    venv = _venv_contract()
    versions, modules = _runtime_versions_and_modules()
    import torch

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
                f"{device} has only {int(free_bytes)} free bytes; SPAI "
                f"requires at least {MINIMUM_CUDA_FREE_BYTES}"
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
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
        "versions": versions,
        "module_files": modules,
        "seed": seed,
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "inference_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "grad_enabled": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "minimum_cuda_free_bytes": MINIMUM_CUDA_FREE_BYTES,
        "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
        "process_environment": environment,
        "network_allowed": False,
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
    validate_runtime_contract(runtime)
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
        "versions",
        "module_files",
        "seed",
        "preprocess_profile",
        "inference_dtype",
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
        "bytecode_writes_disabled",
        "process_environment",
        "network_allowed",
    }
    if device.startswith("cuda:"):
        expected_keys.add("cuda")
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    expected_pycache = Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    if not _same_canonical_json(
        value.get("python"),
        {
        "implementation": "CPython",
        "version": FROZEN_RUNTIME_VERSIONS["python"],
        "executable": str(Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))),
        },
    ):
        raise ValueError(f"{label}.python changed")
    if not _same_canonical_json(
        value.get("venv"),
        {
        "prefix": str(expected_prefix),
        "base_prefix": "/usr",
        "pyvenv_cfg_path": str(expected_prefix / "pyvenv.cfg"),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
        },
    ):
        raise ValueError(f"{label}.venv changed")
    if not _same_canonical_json(
        value.get("versions"),
        FROZEN_RUNTIME_VERSIONS,
    ):
        raise ValueError(f"{label}.versions changed")
    modules = value.get("module_files")
    if (
        not isinstance(modules, Mapping)
        or set(modules) != set(FROZEN_RUNTIME_MODULE_FILES)
        or any(
            not isinstance(modules.get(name), Mapping)
            or set(modules[name]) != {"path", "sha256"}
            or modules[name].get("sha256") != digest
            or not isinstance(modules[name].get("path"), str)
            for name, digest in FROZEN_RUNTIME_MODULE_FILES.items()
        )
    ):
        raise ValueError(f"{label}.module files changed")
    expected_environment = {
        "PYTHONHASHSEED": str(DEFAULT_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(expected_pycache),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(expected_pycache),
        "pycache_prefix_initially_empty": True,
    }
    if (
        not isinstance(value.get("platform"), str)
        or not value["platform"]
        or not _same_canonical_json(value.get("seed"), DEFAULT_SEED)
        or value.get("preprocess_profile") != legacy.PREPROCESS_PROFILE
        or value.get("inference_dtype") != "float32"
        or not _same_canonical_json(value.get("batch_size"), 1)
        or value.get("autocast") is not False
        or value.get("grad_enabled") is not False
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or not _same_canonical_json(
            value.get("cudnn"),
            {
                "enabled": True,
                "benchmark": False,
                "deterministic": True,
                "allow_tf32": False,
            },
        )
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or not _same_canonical_json(
            value.get("minimum_cuda_free_bytes"),
            MINIMUM_CUDA_FREE_BYTES,
        )
        or value.get("bytecode_writes_disabled") is not True
        or not _same_canonical_json(
            value.get("process_environment"),
            expected_environment,
        )
        or value.get("network_allowed") is not False
    ):
        raise ValueError(f"{label} numerical contract changed")
    if device.startswith("cuda:"):
        cuda = value.get("cuda")
        if (
            not isinstance(cuda, Mapping)
            or set(cuda)
            != {
                "runtime",
                "device_index",
                "device_name",
                "total_memory_bytes",
                "capability",
            }
            or cuda.get("runtime") != "12.8"
            or not _same_canonical_json(
                cuda.get("device_index"),
                int(device.split(":", 1)[1]),
            )
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(cuda.get("capability"), list)
            or len(cuda["capability"]) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in cuda["capability"]
            )
        ):
            raise ValueError(f"{label}.cuda changed")
    return value


def verify_assets(
    *,
    source_root: Path,
    checkpoint_path: Path,
    golden_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source, assets, state = legacy.verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        golden_root=golden_root,
    )
    checkpoint = assets.get("checkpoint", {})
    schema = checkpoint.get("schema", {})
    if (
        source.get("commit") != legacy.MODEL_SOURCE_COMMIT
        or checkpoint.get("actual_sha256") != legacy.CHECKPOINT["sha256"]
        or checkpoint.get("actual_bytes") != legacy.CHECKPOINT["bytes"]
        or schema.get("items_sha256")
        != legacy.CHECKPOINT["schema_items_sha256"]
        or schema.get("tensor_count") != legacy.CHECKPOINT["tensor_count"]
        or schema.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or checkpoint.get("serialization_safety")
        != {
            "weights_only": True,
            "pickle_executed": False,
            "safe_global_allowlist": ["yacs.config.CfgNode"],
            "loader": "torch.load(map_location=cpu, weights_only=True)",
        }
    ):
        raise ValueError("SPAI source/checkpoint contract drifted")
    return source, assets, state


def _visibility_diagnostic(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    geometry = legacy.compute_patch_geometry(
        int(row["width"]),
        int(row["height"]),
    )
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        gt = legacy._load_gt_mask(row, repo_root)
        if gt is None:
            raise ValueError("local exact-diff input has no GT mask")
        evidence = legacy._gt_visibility(row, gt, geometry)
        return {
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "edit_visibility_evidence": evidence,
        }
    expected_kind = "all_zero" if row.get("condition") == "real" else "not_applicable"
    if gt_kind != expected_kind:
        raise ValueError("Balanced250 non-local GT semantics changed")
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
            "geometry": geometry,
        },
    }


def selection_visibility_census(
    selected: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = {}
    all_counts: Counter[str] = Counter()
    all_modes: Counter[str] = Counter()
    all_fractions: list[float] = []
    for condition in ("local_mouse", "local_cat", "local_trash_can"):
        rows = [row for row in selected if row.get("condition") == condition]
        counts: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        fractions: list[float] = []
        for row in rows:
            diagnostic = _visibility_diagnostic(row, repo_root=repo_root)
            category = str(diagnostic["edit_visibility"])
            fraction = float(diagnostic["edit_visible_gt_fraction"])
            mode = str(
                diagnostic["edit_visibility_evidence"]["geometry"]["patch_mode"]
            )
            if (
                category not in ("full", "partial", "none")
                or mode not in ("grid", "five_crop")
                or not math.isfinite(fraction)
                or not 0.0 <= fraction <= 1.0
            ):
                raise ValueError("SPAI local visibility diagnostic changed")
            counts[category] += 1
            modes[mode] += 1
            fractions.append(fraction)
            all_counts[category] += 1
            all_modes[mode] += 1
            all_fractions.append(fraction)
        by_condition[condition] = {
            "full": counts["full"],
            "partial": counts["partial"],
            "none": counts["none"],
            "total": len(rows),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
            "patch_modes": {
                "grid": modes["grid"],
                "five_crop": modes["five_crop"],
            },
        }
    result = {
        "by_condition": by_condition,
        "all_local": {
            "full": all_counts["full"],
            "partial": all_counts["partial"],
            "none": all_counts["none"],
            "total": len(all_fractions),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(all_fractions)) if all_fractions else None
            ),
            "patch_modes": {
                "grid": all_modes["grid"],
                "five_crop": all_modes["five_crop"],
            },
        },
        "not_applicable_images": len(selected) - len(all_fractions),
        "basis": (
            "exact_diff_positive_pixels_in_union_of_official_native_"
            "resolution_patch_receptive_fields"
        ),
        "role": "input_condition_stratum_not_model_localization",
    }
    formal = len(selected) == 1775
    if formal and (
        by_condition
        != {
            key: LOCAL_VISIBILITY_CENSUS[key]
            for key in ("local_mouse", "local_cat", "local_trash_can")
        }
        or result["all_local"] != LOCAL_VISIBILITY_CENSUS["all_local"]
        or result["not_applicable_images"] != 1025
    ):
        raise ValueError("formal SPAI visibility census drifted")
    return result


def _golden_artifact_record(array: np.ndarray) -> dict[str, Any]:
    payload = _npy_bytes(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "array_sha256": _array_sha256(array),
        "file_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_official_cpu_golden(
    *,
    model: Any,
    device: Any,
    golden_root: Path,
) -> dict[str, Any]:
    """Freeze CPU repeats without treating CUDA values as CPU bit-goldens."""

    if str(device) != "cpu":
        raise ValueError("official CPU golden requires torch.device('cpu')")
    cases: list[dict[str, Any]] = []
    for frozen, expected_cpu in zip(
        legacy.GOLDEN_CASES,
        CPU_OFFICIAL_GOLDEN_CASES,
        strict=True,
    ):
        if frozen["relative_path"] != expected_cpu["relative_path"]:
            raise AssertionError("SPAI CPU/CUDA official case ordering changed")
        path = golden_root / str(frozen["relative_path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != frozen["sha256"]
        ):
            raise ValueError("SPAI official CPU golden asset changed")
        image, preprocess = legacy.preprocess_image(path)
        if (
            preprocess["decoded_rgb_sha256"] != frozen["decoded_rgb_sha256"]
            or preprocess["tensor_sha256"] != frozen["tensor_sha256"]
            or preprocess["native_size"] != frozen["native_size"]
            or preprocess["geometry"]["effective_patch_count"]
            != frozen["patch_count"]
        ):
            raise ValueError("SPAI official CPU golden preprocessing changed")
        observed: list[dict[str, Any]] = []
        for _index in range(2):
            scoring, patch, feature, attention, peak, _latency = (
                legacy.infer_one(model, device, image)
            )
            observed.append(
                {
                    "raw_logit": scoring["raw_logit"],
                    "probability": scoring["probability"],
                    "patch_features_array_sha256": _array_sha256(patch),
                    "feature_array_sha256": _array_sha256(feature),
                    "attention_array_sha256": _array_sha256(attention),
                    "peak_cuda_memory_bytes": (
                        0 if peak is None else int(peak)
                    ),
                }
            )
        first, second = observed
        if first != second:
            raise ValueError("SPAI official CPU golden is not bit-repeatable")
        expected = {
            key: expected_cpu[key]
            for key in (
                "raw_logit",
                "probability",
                "patch_features_array_sha256",
                "feature_array_sha256",
                "attention_array_sha256",
            )
        }
        expected["peak_cuda_memory_bytes"] = 0
        if first != expected:
            raise ValueError("SPAI official CPU golden output drifted")
        cases.append(
            {
                "relative_path": frozen["relative_path"],
                "path": str(path.resolve()),
                "sha256": frozen["sha256"],
                "preprocess": preprocess,
                "cpu_observed_runs": observed,
                "cpu_bit_identical_repeats": True,
                "cuda_reference": {
                    "raw_logit": frozen["raw_logit"],
                    "probability": frozen["probability"],
                    "logit_absolute_difference_from_cpu": abs(
                        float(frozen["raw_logit"]) - first["raw_logit"]
                    ),
                    "probability_absolute_difference_from_cpu": abs(
                        float(frozen["probability"]) - first["probability"]
                    ),
                    "role": (
                        "released CUDA implementation regression; not a CPU "
                        "bit-equality acceptance value"
                    ),
                },
                "passed": True,
            }
        )
    return {
        "status": "passed",
        "device": "cpu",
        "runs_per_case": 2,
        "cpu_repeat_tolerance": 0.0,
        "cross_device_bit_equality_required": False,
        "cuda_reference_is_not_cpu_acceptance_gate": True,
        "cases": cases,
    }


def validate_execution_device_golden(
    *,
    model: Any,
    device: Any,
    golden_root: Path,
) -> dict[str, Any]:
    """Gate the configured inference device using its matching reference."""

    if getattr(device, "type", None) == "cuda":
        report = legacy.validate_official_golden(
            model=model,
            device=device,
            golden_root=golden_root,
        )
        return {
            "status": "passed",
            "device": str(device),
            "reference_device": "cuda",
            "gate": "released_CUDA_highest_no_TF32_implementation_regression",
            "cross_device_bit_equality_required": False,
            "report": report,
        }
    if getattr(device, "type", None) == "cpu":
        report = validate_official_cpu_golden(
            model=model,
            device=device,
            golden_root=golden_root,
        )
        return {
            "status": "passed",
            "device": "cpu",
            "reference_device": "cpu",
            "gate": "frozen_CPU_bit_repeat_regression",
            "cross_device_bit_equality_required": False,
            "report": report,
        }
    raise ValueError("SPAI execution golden received an unsupported device")


def _balanced_golden_record(
    *,
    model: Any,
    device: Any,
    repo_root: Path,
) -> dict[str, Any]:
    path = _anchored(Path(CPU_GOLDEN_INPUT_PATH), repo_root)
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != CPU_GOLDEN_IMAGE_SHA256
    ):
        raise ValueError("SPAI Balanced CPU golden input changed")
    records: list[dict[str, Any]] = []
    for _index in range(2):
        image, preprocess = legacy.preprocess_image(path)
        scoring, patch, feature, attention, peak, _latency = legacy.infer_one(
            model,
            device,
            image,
        )
        records.append(
            {
                "preprocess": preprocess,
                "patch_features": _golden_artifact_record(patch),
                "feature": _golden_artifact_record(feature),
                "attention": _golden_artifact_record(attention),
                "raw_logit": scoring["raw_logit"],
                "probability": scoring["probability"],
                "ai_score": scoring["ai_score"],
                "classification_decision": scoring[
                    "classification_decision"
                ],
                "manual_replay": scoring["manual_replay"],
                "peak_cuda_memory_bytes": 0 if peak is None else int(peak),
            }
        )
    first, second = records
    if first != second:
        raise ValueError("SPAI Balanced CPU golden forwards are not byte-exact")
    preprocess = first["preprocess"]
    expected_artifacts = {
        "patch_features": {
            "shape": [4, legacy.FEATURE_DIMENSION],
            "dtype": "float32",
            "nbytes": 4 * legacy.FEATURE_DIMENSION * 4,
            "array_sha256": CPU_GOLDEN_PATCH_ARRAY_SHA256,
            "file_bytes": 17664,
            "file_sha256": CPU_GOLDEN_PATCH_FILE_SHA256,
        },
        "feature": {
            "shape": [legacy.FEATURE_DIMENSION],
            "dtype": "float32",
            "nbytes": legacy.FEATURE_DIMENSION * 4,
            "array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
            "file_bytes": 4512,
            "file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        },
        "attention": {
            "shape": [legacy.ATTENTION_HEADS, 4],
            "dtype": "float32",
            "nbytes": legacy.ATTENTION_HEADS * 4 * 4,
            "array_sha256": CPU_GOLDEN_ATTENTION_ARRAY_SHA256,
            "file_bytes": 320,
            "file_sha256": CPU_GOLDEN_ATTENTION_FILE_SHA256,
        },
    }
    if (
        preprocess.get("decoded_rgb_sha256")
        != CPU_GOLDEN_DECODED_RGB_SHA256
        or preprocess.get("tensor_sha256") != CPU_GOLDEN_TENSOR_SHA256
        or preprocess.get("native_size") != [640, 640]
        or preprocess.get("geometry", {}).get("effective_patch_count") != 4
        or any(
            first.get(name) != expected
            for name, expected in expected_artifacts.items()
        )
        or first.get("raw_logit") != CPU_GOLDEN_RAW_LOGIT
        or first.get("probability") != CPU_GOLDEN_PROBABILITY
        or first.get("ai_score") != CPU_GOLDEN_PROBABILITY
        or first.get("classification_decision") is not False
        or first.get("peak_cuda_memory_bytes") != 0
    ):
        raise ValueError("SPAI Balanced CPU golden output drifted")
    manual = first.get("manual_replay")
    if not isinstance(manual, Mapping) or any(
        manual.get(key) is not True
        for key in (
            "official_attention_exact_match",
            "official_aggregated_exact_match",
            "official_feature_exact_match",
            "official_logit_exact_match",
            "official_probability_exact_match",
            "sca_replay",
            "norm_replay",
            "complete_mlp_replay",
        )
    ):
        raise ValueError("SPAI Balanced CPU golden replay drifted")
    return {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 640,
        "input_height": 640,
        **first,
        "repeat_patch_features_file_sha256": second["patch_features"][
            "file_sha256"
        ],
        "repeat_feature_file_sha256": second["feature"]["file_sha256"],
        "repeat_attention_file_sha256": second["attention"]["file_sha256"],
        "repeat_raw_logit": second["raw_logit"],
        "repeat_probability": second["probability"],
        "repeat_byte_exact": True,
    }


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    checkpoint_path: Path,
    golden_root: Path,
) -> dict[str, Any]:
    """Run source, restricted-load, preprocess, and two independent goldens."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before SPAI CPU preflight")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    if device.type != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("SPAI CPU preflight configured a non-CPU runtime")
    source, assets, state = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        golden_root=golden_root,
    )
    model = None
    try:
        preprocess_equivalence = (
            legacy.validate_official_preprocess_equivalence()
        )
        model, model_load = legacy.load_model(
            state=state,
            source_root=source_root,
            device=device,
        )
        official = validate_official_cpu_golden(
            model=model,
            device=device,
            golden_root=golden_root,
        )
        balanced = _balanced_golden_record(
            model=model,
            device=device,
            repo_root=repo_root,
        )
        if torch.cuda.is_initialized():
            raise RuntimeError("SPAI CPU preflight initialized CUDA")
        return {
            "schema_version": CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "assets": assets,
            "model_load": model_load,
            "runtime": runtime,
            "official_preprocess_equivalence": preprocess_equivalence,
            "official_golden": official,
            "balanced_golden": balanced,
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "cuda_initialized_before_cpu_model_load": False,
            "cuda_initialized_after_cpu_forwards": False,
            "dataset_manifest_loaded": False,
        }
    finally:
        if model is not None:
            del model
        del state
        gc.collect()


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
        "official_preprocess_equivalence",
        "official_golden",
        "balanced_golden",
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
        "dataset_manifest_loaded",
    }
    if set(report) != expected_keys:
        raise ValueError("SPAI CPU preflight key set changed")
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
        raise ValueError("SPAI CPU preflight provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("SPAI CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    load = report.get("model_load")
    if (
        not isinstance(load, Mapping)
        or load.get("load", {}).get("strict") is not True
        or load.get("load", {}).get("full_state_coverage") is not True
        or load.get("load", {}).get("loaded_tensor_exact_match") is not True
        or load.get("model", {}).get("state_tensors")
        != legacy.CHECKPOINT["tensor_count"]
        or load.get("model", {}).get("state_elements")
        != legacy.CHECKPOINT["state_elements"]
        or load.get("model", {}).get("feature_dimension")
        != legacy.FEATURE_DIMENSION
        or load.get("model", {}).get("attention_heads")
        != legacy.ATTENTION_HEADS
        or load.get("model", {}).get("eval") is not True
        or any(load.get("network", {}).get("attempts", {}).values())
    ):
        raise ValueError("SPAI CPU preflight strict model load changed")
    equivalence = report.get("official_preprocess_equivalence")
    if (
        not isinstance(equivalence, Mapping)
        or equivalence.get("status") != "passed"
        or not isinstance(equivalence.get("cases"), list)
        or len(equivalence["cases"]) != 2
        or any(case.get("exact_match") is not True for case in equivalence["cases"])
    ):
        raise ValueError("SPAI official preprocess equivalence changed")
    official = report.get("official_golden")
    if (
        not isinstance(official, Mapping)
        or official.get("status") != "passed"
        or official.get("device") != "cpu"
        or official.get("runs_per_case") != 2
        or official.get("cpu_repeat_tolerance") != 0.0
        or official.get("cross_device_bit_equality_required") is not False
        or official.get("cuda_reference_is_not_cpu_acceptance_gate")
        is not True
        or not isinstance(official.get("cases"), list)
        or len(official["cases"]) != len(legacy.GOLDEN_CASES)
        or any(case.get("passed") is not True for case in official["cases"])
    ):
        raise ValueError("SPAI official golden changed")
    balanced = report.get("balanced_golden")
    if (
        not isinstance(balanced, Mapping)
        or balanced.get("sample_id") != CPU_GOLDEN_SAMPLE_ID
        or balanced.get("image_sha256") != CPU_GOLDEN_IMAGE_SHA256
        or balanced.get("raw_logit") != CPU_GOLDEN_RAW_LOGIT
        or balanced.get("probability") != CPU_GOLDEN_PROBABILITY
        or balanced.get("repeat_byte_exact") is not True
        or balanced.get("patch_features", {}).get("array_sha256")
        != CPU_GOLDEN_PATCH_ARRAY_SHA256
        or balanced.get("feature", {}).get("array_sha256")
        != CPU_GOLDEN_FEATURE_ARRAY_SHA256
        or balanced.get("attention", {}).get("array_sha256")
        != CPU_GOLDEN_ATTENTION_ARRAY_SHA256
    ):
        raise ValueError("SPAI Balanced CPU golden changed")


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
        "joint_output",
        "joint_score",
        "s_joint",
    }
)
_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "pair_rank",
        "localization",
        "localisation",
        "heatmap",
        "attention_map",
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
    "attention_map",
    "attention_mask",
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
    {"gt_mask_kind", "mask_positive_pixels"}
)


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(child, child_label)
                continue
            if normalized == "pair_rank":
                raise ValueError(f"{child_label} is an unsupported SPAI claim")
            if normalized in _FALSE_DECLARATIONS:
                if child is not False:
                    raise ValueError(
                        f"{child_label} is an unsupported SPAI claim"
                    )
                continue
            if normalized in _NULL_DECLARATIONS:
                if child is not None:
                    raise ValueError(
                        f"{child_label} is an unsupported SPAI claim"
                    )
                continue
            if normalized in _FORBIDDEN_CLAIM_KEYS or normalized.startswith(
                _FORBIDDEN_CLAIM_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported SPAI claim")
            _reject_unsupported_claims(child, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _reject_unsupported_claims(child, f"{label}[{index}]")


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
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
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "config_fingerprint": run_manifest_fingerprint,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        **_visibility_diagnostic(row, repo_root=repo_root),
    }


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


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    return path


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
    return _require_regular_file(resolved, label)


def _artifact_spec(
    kind: str,
    *,
    patch_count: int,
) -> tuple[tuple[int, ...], str, str]:
    if kind == "patch_features":
        return (
            (patch_count, legacy.FEATURE_DIMENSION),
            legacy.PATCH_FEATURE_SEMANTICS,
            "spai_patch_features",
        )
    if kind == "feature":
        return (
            (legacy.FEATURE_DIMENSION,),
            legacy.FEATURE_SEMANTICS,
            "spai_feature",
        )
    if kind == "attention":
        return (
            (legacy.ATTENTION_HEADS, patch_count),
            legacy.ATTENTION_SEMANTICS,
            "spai_attention",
        )
    raise ValueError(f"unsupported SPAI artifact kind: {kind}")


def _artifact_relative_path(
    *,
    run_id: str,
    kind: str,
    sample_id: str,
) -> str:
    return (
        DEFAULT_ARTIFACTS_DIR / run_id / kind / f"{sample_id}.npy"
    ).as_posix()


def _artifact_record(
    *,
    array: np.ndarray,
    path: Path,
    semantics: str,
    repo_root: Path,
) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload != _npy_bytes(array):
        raise ValueError("persisted SPAI artifact is not canonical NumPy bytes")
    return {
        "relative_path": repo_relative(path, repo_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "array_sha256": _array_sha256(array),
        "dtype": "float32",
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "finite": True,
        "semantics": semantics,
        "allow_pickle": False,
    }


def _persist_artifacts(
    *,
    artifact_root: Path,
    sample_id: str,
    patch: np.ndarray,
    feature: np.ndarray,
    attention: np.ndarray,
    repo_root: Path,
) -> dict[str, Any]:
    patch_count = int(patch.shape[0]) if patch.ndim == 2 else -1
    arrays = {
        "patch_features": patch,
        "feature": feature,
        "attention": attention,
    }
    records: dict[str, dict[str, Any]] = {}
    for kind, array in arrays.items():
        shape, semantics, _prefix = _artifact_spec(
            kind,
            patch_count=patch_count,
        )
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != np.float32
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"official SPAI {kind} artifact changed")
        path = artifact_root / kind / f"{sample_id}.npy"
        legacy._atomic_save_npy(path, array)
        records[kind] = _artifact_record(
            array=array,
            path=path,
            semantics=semantics,
            repo_root=repo_root,
        )
    patch_record = records["patch_features"]
    feature_record = records["feature"]
    attention_record = records["attention"]
    return {
        "spai_patch_features": patch_record,
        "spai_feature": feature_record,
        "spai_attention": attention_record,
        "spai_patch_features_path": patch_record["relative_path"],
        "spai_patch_features_sha256": patch_record["sha256"],
        "spai_patch_features_array_sha256": patch_record["array_sha256"],
        "spai_patch_features_shape": patch_record["shape"],
        "spai_patch_features_dtype": "float32",
        "spai_patch_features_nbytes": patch_record["nbytes"],
        "spai_patch_features_semantics": legacy.PATCH_FEATURE_SEMANTICS,
        "spai_feature_path": feature_record["relative_path"],
        "spai_feature_sha256": feature_record["sha256"],
        "spai_feature_array_sha256": feature_record["array_sha256"],
        "spai_feature_shape": feature_record["shape"],
        "spai_feature_dtype": "float32",
        "spai_feature_nbytes": feature_record["nbytes"],
        "spai_feature_semantics": legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": feature_record["array_sha256"],
        "spai_attention_path": attention_record["relative_path"],
        "spai_attention_sha256": attention_record["sha256"],
        "spai_attention_array_sha256": attention_record["array_sha256"],
        "spai_attention_shape": attention_record["shape"],
        "spai_attention_dtype": "float32",
        "spai_attention_nbytes": attention_record["nbytes"],
        "spai_attention_semantics": legacy.ATTENTION_SEMANTICS,
        "attention_is_diagnostic_not_t2": True,
        "artifact_paths": {
            "spai_patch_features_npy": patch_record["relative_path"],
            "spai_feature_npy": feature_record["relative_path"],
            "spai_attention_npy": attention_record["relative_path"],
        },
    }


def _validate_artifact(
    row: Mapping[str, Any],
    *,
    kind: str,
    sample_id: str,
    patch_count: int,
    repo_root: Path,
    run_id: str,
) -> np.ndarray:
    shape, semantics, prefix = _artifact_spec(kind, patch_count=patch_count)
    record = row.get(prefix)
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
        "allow_pickle",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ValueError(f"{sample_id} {kind} artifact key set changed")
    expected_relative = _artifact_relative_path(
        run_id=run_id,
        kind=kind,
        sample_id=sample_id,
    )
    if record.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} {kind} artifact path changed")
    path = _safe_repo_file(
        expected_relative,
        repo_root=repo_root,
        label=f"{sample_id} SPAI {kind}",
    )
    expected_nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    expected_file_bytes = _expected_npy_file_bytes(shape)
    if (
        type(record.get("file_bytes")) is not int
        or record.get("file_bytes") != expected_file_bytes
        or type(record.get("nbytes")) is not int
        or record.get("nbytes") != expected_nbytes
        or path.stat().st_size != expected_file_bytes
    ):
        raise ValueError(f"{sample_id} {kind} metadata/hash changed")
    payload = path.read_bytes()
    if (
        record.get("sha256") != hashlib.sha256(payload).hexdigest()
        or record.get("file_bytes") != len(payload)
        or record.get("dtype") != "float32"
        or record.get("shape") != list(shape)
        or record.get("nbytes") != expected_nbytes
        or record.get("finite") is not True
        or record.get("semantics") != semantics
        or record.get("allow_pickle") is not False
    ):
        raise ValueError(f"{sample_id} {kind} metadata/hash changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} {kind} NPY is invalid") from error
    if (
        not isinstance(array, np.ndarray)
        or array.shape != shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or array.nbytes != expected_nbytes
        or payload != _npy_bytes(array)
        or record.get("array_sha256") != _array_sha256(array)
    ):
        raise ValueError(f"{sample_id} {kind} array changed")
    aliases = {
        f"{prefix}_path": expected_relative,
        f"{prefix}_sha256": record["sha256"],
        f"{prefix}_array_sha256": record["array_sha256"],
        f"{prefix}_shape": list(shape),
        f"{prefix}_dtype": "float32",
        f"{prefix}_nbytes": expected_nbytes,
        f"{prefix}_semantics": semantics,
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} {kind} alias {key} changed")
    return array


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> tuple[float, float, bool]:
    if (
        type(row.get("raw_logit")) is not float
        or type(row.get("probability")) is not float
        or type(row.get("ai_score")) is not float
        or type(row.get("score")) is not float
        or type(row.get("classification_threshold")) is not float
    ):
        raise ValueError(f"{sample_id} score/replay payload changed")
    raw = _finite_number(row.get("raw_logit"), f"{sample_id} raw_logit")
    probability = _finite_number(
        row.get("probability"),
        f"{sample_id} probability",
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{sample_id} probability is outside [0,1]")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": legacy.SCORE_SEMANTICS,
    }
    t1 = {key: value for key, value in classification.items() if key != "semantics"}
    t1["policy"] = legacy.T1_POLICY
    observed_classification = row.get("classification")
    observed_t1 = row.get("t1")
    replay = row.get("manual_replay")
    if (
        row.get("ai_score") != probability
        or row.get("score") != probability
        or row.get("score_semantics") != legacy.SCORE_SEMANTICS
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold")
        != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or not isinstance(observed_classification, Mapping)
        or not _same_canonical_json(observed_classification, classification)
        or any(
            type(observed_classification.get(key)) is not float
            for key in (
                "raw_logit",
                "probability",
                "ai_score",
                "score",
                "threshold",
            )
        )
        or type(observed_classification.get("decision")) is not bool
        or not isinstance(observed_t1, Mapping)
        or not _same_canonical_json(observed_t1, t1)
        or any(
            type(observed_t1.get(key)) is not float
            for key in (
                "raw_logit",
                "probability",
                "ai_score",
                "score",
                "threshold",
            )
        )
        or type(observed_t1.get("decision")) is not bool
        or not isinstance(replay, Mapping)
        or set(replay)
        != {
            "raw_logit",
            "probability",
            "ai_score",
            "classification_decision",
            "official_attention_exact_match",
            "official_aggregated_exact_match",
            "official_feature_exact_match",
            "official_logit_exact_match",
            "official_probability_exact_match",
            "sca_replay",
            "norm_replay",
            "complete_mlp_replay",
            "model_forward_calls",
            "to_kv_hook_calls",
            "attention_hook_calls",
            "norm_hook_calls",
        }
        or replay.get("raw_logit") != raw
        or replay.get("probability") != probability
        or replay.get("ai_score") != probability
        or replay.get("classification_decision") is not decision
        or any(
            type(replay.get(key)) is not float
            for key in ("raw_logit", "probability", "ai_score")
        )
        or type(replay.get("classification_decision")) is not bool
        or any(
            replay.get(key) is not True
            for key in (
                "official_attention_exact_match",
                "official_aggregated_exact_match",
                "official_feature_exact_match",
                "official_logit_exact_match",
                "official_probability_exact_match",
                "sca_replay",
                "norm_replay",
                "complete_mlp_replay",
            )
        )
        or any(
            type(replay.get(key)) is not int or replay.get(key) != 1
            for key in (
                "model_forward_calls",
                "to_kv_hook_calls",
                "attention_hook_calls",
                "norm_hook_calls",
            )
        )
    ):
        raise ValueError(f"{sample_id} score/replay payload changed")
    return raw, probability, decision


_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "spai_patch_features",
        "spai_feature",
        "spai_attention",
        "spai_patch_features_path",
        "spai_patch_features_sha256",
        "spai_patch_features_array_sha256",
        "spai_patch_features_shape",
        "spai_patch_features_dtype",
        "spai_patch_features_nbytes",
        "spai_patch_features_semantics",
        "spai_feature_path",
        "spai_feature_sha256",
        "spai_feature_array_sha256",
        "spai_feature_shape",
        "spai_feature_dtype",
        "spai_feature_nbytes",
        "spai_feature_semantics",
        "feature_array_sha256",
        "spai_attention_path",
        "spai_attention_sha256",
        "spai_attention_array_sha256",
        "spai_attention_shape",
        "spai_attention_dtype",
        "spai_attention_nbytes",
        "spai_attention_semantics",
        "attention_is_diagnostic_not_t2",
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


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    expected_keys = set(expected) | {"status", "completed_at"}
    expected_keys |= (
        _OK_RESULT_FIELDS
        if status == "ok"
        else {"error_type", "error", "traceback"}
    )
    if set(attempt) != expected_keys:
        raise ValueError(
            "result attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, expected_value in expected.items():
        if not _same_canonical_json(attempt.get(key), expected_value):
            raise ValueError(f"result attempt field {key} drifted")
    if (
        not isinstance(attempt.get("completed_at"), str)
        or not attempt["completed_at"]
    ):
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
        return None
    sample_id = str(input_row["sample_id"])
    _validate_score_payload(attempt, sample_id=sample_id)
    input_path = _safe_repo_file(
        str(input_row["canonical_path"]),
        repo_root=repo_root,
        label=f"{sample_id} canonical input",
    )
    _image, expected_preprocess = legacy.preprocess_image(input_path)
    if not _same_canonical_json(
        attempt.get("preprocess"),
        expected_preprocess,
    ):
        raise ValueError(f"{sample_id} preprocessing record changed")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _finite_number(attempt.get(field), f"{sample_id} {field}") < 0.0:
            raise ValueError(f"{sample_id} {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{sample_id} peak memory is invalid")
    patch_count = int(
        expected_preprocess["geometry"]["effective_patch_count"]
    )
    patch = _validate_artifact(
        attempt,
        kind="patch_features",
        sample_id=sample_id,
        patch_count=patch_count,
        repo_root=repo_root,
        run_id=run_id,
    )
    feature = _validate_artifact(
        attempt,
        kind="feature",
        sample_id=sample_id,
        patch_count=patch_count,
        repo_root=repo_root,
        run_id=run_id,
    )
    attention = _validate_artifact(
        attempt,
        kind="attention",
        sample_id=sample_id,
        patch_count=patch_count,
        repo_root=repo_root,
        run_id=run_id,
    )
    expected_paths = {
        "spai_patch_features_npy": attempt["spai_patch_features_path"],
        "spai_feature_npy": attempt["spai_feature_path"],
        "spai_attention_npy": attempt["spai_attention_path"],
    }
    if (
        attempt.get("artifact_paths") != expected_paths
        or attempt.get("feature_array_sha256")
        != attempt.get("spai_feature_array_sha256")
        or attempt.get("attention_is_diagnostic_not_t2") is not True
    ):
        raise ValueError(f"{sample_id} SPAI artifact aliases changed")
    return patch, feature, attention


def _replay_artifacts(
    *,
    row: Mapping[str, Any],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    model: Any,
    device: Any,
) -> None:
    import torch

    patch, feature, attention = arrays
    raw, probability, decision = _validate_score_payload(
        row,
        sample_id=str(row["sample_id"]),
    )
    with torch.inference_mode():
        scoring, _ = legacy.replay_sca_norm_head(
            model=model,
            patch_features=torch.from_numpy(patch).to(device),
            official_output=torch.tensor(
                [[raw]],
                dtype=torch.float32,
                device=device,
            ),
            expected_attention=(
                torch.from_numpy(attention)
                .reshape(1, legacy.ATTENTION_HEADS, 1, patch.shape[0])
                .to(device)
            ),
            expected_feature=(
                torch.from_numpy(feature).reshape(1, -1).to(device)
            ),
        )
    if (
        scoring["raw_logit"] != raw
        or scoring["probability"] != probability
        or scoring["classification_decision"] is not decision
    ):
        raise ValueError(
            f"{row['sample_id']} same-device SCA/norm/head replay mismatch"
        )


def _validate_latest_artifact_replay(
    *,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    inputs_by_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    model: Any,
    device: Any,
) -> int:
    replayed = 0
    for sample_id, row in latest_by_sample_id.items():
        input_row = inputs_by_id.get(sample_id)
        if input_row is None:
            raise ValueError("latest SPAI result is outside selection")
        arrays = _validate_runner_attempt(
            row,
            input_row=input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        if row.get("status") == "ok":
            if arrays is None:
                raise AssertionError("successful SPAI row lost artifacts")
            _replay_artifacts(
                row=row,
                arrays=arrays,
                model=model,
                device=device,
            )
            replayed += 1
    expected = sum(
        row.get("status") == "ok"
        for row in latest_by_sample_id.values()
    )
    if replayed != expected:
        raise ValueError("SPAI artifact replay coverage is incomplete")
    return replayed


def _validate_physical_attempt_history(
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    statuses: dict[str, list[str]] = {}
    for attempt in attempts:
        sample_id = str(attempt.get("sample_id"))
        statuses.setdefault(sample_id, []).append(str(attempt.get("status")))
    for sample_id, values in statuses.items():
        successful = [
            index for index, status in enumerate(values) if status == "ok"
        ]
        if len(successful) > 1:
            raise ValueError(
                f"duplicate successful physical attempts for {sample_id}"
            )
        if successful and successful[0] != len(values) - 1:
            raise ValueError(
                f"physical attempt exists after success for {sample_id}"
            )


def _local_artifact_policy(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    ignore_path = _require_regular_file(root / ".gitignore", ".gitignore")
    probe = (
        DEFAULT_ARTIFACTS_DIR
        / "_claimforge_ignore_probe"
        / "patch_features"
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
        raise ValueError("SPAI artifact output is not gitignored") from error
    evidence = completed.stdout.strip()
    if (
        not evidence.startswith(".gitignore:")
        or "\t" not in evidence
        or not evidence.endswith(probe)
    ):
        raise ValueError("SPAI git-ignore evidence changed")
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
    selection_visibility: Mapping[str, Any],
    adapter_sources: Mapping[str, Any],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    execution_device_golden: Mapping[str, Any],
    run_dir: Path,
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    run_id = _valid_run_id(run_id)
    validate_runtime_contract(runtime, label="immutable runtime")
    if (
        source.get("commit") != legacy.MODEL_SOURCE_COMMIT
        or assets.get("checkpoint", {}).get("actual_sha256")
        != legacy.CHECKPOINT["sha256"]
    ):
        raise ValueError("SPAI immutable source/assets changed")
    immutable = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": MODEL_CONTRACT,
        "preprocess": PREPROCESS_CONTRACT,
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "selection_visibility_census": dict(selection_visibility),
        "formal_local_visibility_census": LOCAL_VISIBILITY_CENSUS,
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_dataset_and_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "execution_device_golden": {
            "performed_after_explicit_device_configuration_before_scoring": True,
            "cross_device_bit_equality_required": False,
            "report": dict(execution_device_golden),
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "local_artifact_policy": _local_artifact_policy(repo_root),
        "outputs": {
            "run_dir": repo_relative(run_dir, repo_root),
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_root": repo_relative(artifact_root, repo_root),
            "patch_features_dir": repo_relative(
                artifact_root / "patch_features",
                repo_root,
            ),
            "features_dir": repo_relative(
                artifact_root / "feature",
                repo_root,
            ),
            "attention_dir": repo_relative(
                artifact_root / "attention",
                repo_root,
            ),
        },
    }
    if set(immutable) != IMMUTABLE_CONFIG_KEYS:
        raise AssertionError("internal immutable SPAI config key set drifted")
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


def _validate_run_dir_inventory(
    run_dir: Path,
    *,
    allow_missing_results: bool,
) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FileNotFoundError("missing or unsafe SPAI run directory")
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
        "balanced250_metrics.json",
        "independent_audit.json",
    }
    entries = list(run_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("SPAI run directory contains a non-regular entry")
    names = {entry.name for entry in entries}
    if not names <= allowed:
        raise ValueError(
            f"SPAI run directory has unexpected entries: "
            f"{sorted(names - allowed)}"
        )
    required = {"manifest.json", "expected_inputs.jsonl"}
    if not allow_missing_results:
        required |= {"results.jsonl", "summary.json"}
    if not required <= names:
        raise FileNotFoundError(
            f"SPAI run directory is missing: {sorted(required - names)}"
        )


def _prepare_output_directories(
    *,
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
    resume: bool,
) -> None:
    run_dir = _ensure_repo_child(
        run_dir,
        repo_root=repo_root,
        label="SPAI run directory",
    )
    artifact_root = _ensure_repo_child(
        artifact_root,
        repo_root=repo_root,
        label="SPAI artifact root",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("SPAI result and artifact directories must be disjoint")
    expected_dirs = {"patch_features", "feature", "attention"}
    if not resume:
        if run_dir.exists() and (
            not run_dir.is_dir() or any(run_dir.iterdir())
        ):
            raise FileExistsError(
                f"SPAI run directory is non-empty; pass --resume: {run_dir}"
            )
        if artifact_root.exists() and (
            not artifact_root.is_dir() or any(artifact_root.iterdir())
        ):
            raise FileExistsError(
                "SPAI artifact root is non-empty; pass --resume: "
                f"{artifact_root}"
            )
    else:
        if not run_dir.is_dir() or not artifact_root.is_dir():
            raise FileNotFoundError(
                "SPAI resume requires run and artifact directories"
            )
        root_entries = list(artifact_root.iterdir())
        if (
            {entry.name for entry in root_entries} != expected_dirs
            or any(entry.is_symlink() or not entry.is_dir() for entry in root_entries)
        ):
            raise ValueError("SPAI resume artifact-root inventory changed")
        _validate_run_dir_inventory(
            run_dir,
            allow_missing_results=True,
        )
        finalized_analysis = {
            name
            for name in ("balanced250_metrics.json", "independent_audit.json")
            if (run_dir / name).is_file()
        }
        if finalized_analysis:
            raise ValueError(
                "SPAI resume is forbidden after analyzer outputs exist; use "
                f"a new run ID: {sorted(finalized_analysis)}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected_dirs):
        directory = artifact_root / name
        directory.mkdir(parents=True, exist_ok=True)
        _ensure_repo_child(
            directory,
            repo_root=repo_root,
            label=f"SPAI {name} directory",
            require_directory=True,
        )


def _validate_artifact_inventory(
    *,
    artifact_root: Path,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> int:
    expected_ids = {
        sample_id
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    for kind in ("patch_features", "feature", "attention"):
        directory = artifact_root / kind
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"missing SPAI {kind} directory")
        entries = list(directory.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError(f"SPAI {kind} inventory has an unsafe entry")
        actual = {entry.name for entry in entries}
        expected = {f"{sample_id}.npy" for sample_id in expected_ids}
        if actual != expected:
            raise ValueError(
                f"SPAI {kind} inventory mismatch: "
                f"missing={sorted(expected - actual)[:1]}, "
                f"extra={sorted(actual - expected)[:1]}"
            )
    return len(expected_ids) * 3


def _build_ok_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    artifact_root: Path,
    processed: Mapping[str, Any],
    patch: np.ndarray,
    feature: np.ndarray,
    attention: np.ndarray,
    preprocess: Mapping[str, Any],
    preprocess_latency_ms: float,
    latency_ms: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    artifacts = _persist_artifacts(
        artifact_root=artifact_root,
        sample_id=sample_id,
        patch=patch,
        feature=feature,
        attention=attention,
        repo_root=repo_root,
    )
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "completed_at": utc_now(),
        "preprocess": dict(preprocess),
        "preprocess_latency_ms": float(preprocess_latency_ms),
        **artifacts,
        **{
            key: processed[key]
            for key in (
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
            )
        },
        "latency_ms": float(latency_ms),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
    }
    return result


def _build_error_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        **result_identity(
            input_row,
            repo_root=repo_root,
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
                    raise ValueError(f"{row_label} is not an object")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


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
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=DEFAULT_GOLDEN_ROOT,
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


def _validate_prior_manifest(
    *,
    manifest: Mapping[str, Any],
    run_id: str,
    fingerprint: str,
    immutable: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    expected_path: Path,
    dataset_contract: Mapping[str, Any],
    dataset_manifest_path: Path,
    manifest_sha256: str,
    repo_root: Path,
    results_path: Path,
    summary_path: Path,
) -> tuple[str, str, Mapping[str, Any]]:
    status = manifest.get("status")
    expected_keys = {
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
    if status in ("complete", "incomplete"):
        expected_keys.add("execution")
    if (
        set(manifest) != expected_keys
        or status not in ("running", "complete", "incomplete")
        or manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("fingerprint") != fingerprint
        or not _same_canonical_json(manifest.get("immutable"), immutable)
    ):
        raise ValueError("SPAI resume manifest fingerprint/config drifted")
    if not _same_canonical_json(
        _read_jsonl_strict(expected_path, "expected inputs"),
        list(selected),
    ):
        raise ValueError("SPAI resume expected-input snapshot drifted")
    expected_dataset = {
        "contract": dict(dataset_contract),
        "manifest_path": repo_relative(dataset_manifest_path, repo_root),
        "manifest_sha256": manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    if not _same_canonical_json(manifest.get("dataset"), expected_dataset):
        raise ValueError("SPAI resume dataset evidence drifted")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("SPAI resume outputs are invalid")
    immutable_outputs = immutable["outputs"]
    if status == "running":
        if not _same_canonical_json(dict(outputs), immutable_outputs):
            raise ValueError("SPAI running resume output contract drifted")
    else:
        expected_output_keys = set(immutable_outputs) | {
            "results_sha256",
            "summary_sha256",
            "artifact_files",
        }
        if (
            set(outputs) != expected_output_keys
            or any(
                outputs.get(key) != value
                for key, value in immutable_outputs.items()
            )
            or not results_path.is_file()
            or outputs.get("results_sha256") != sha256_file(results_path)
            or not summary_path.is_file()
            or outputs.get("summary_sha256") != sha256_file(summary_path)
            or isinstance(outputs.get("artifact_files"), bool)
            or not isinstance(outputs.get("artifact_files"), int)
            or outputs["artifact_files"] < 0
        ):
            raise ValueError("SPAI finalized resume output evidence drifted")
    started_at = manifest.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("SPAI resume started_at is invalid")
    return str(status), started_at, outputs


def run(args: argparse.Namespace) -> int:
    """Execute one append-only SPAI Balanced250 run."""

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(Path(args.source_root), repo_root)
    checkpoint_path = _anchored(Path(args.checkpoint), repo_root)
    golden_root = _anchored(Path(args.golden_root), repo_root)
    mode = str(args.mode)
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed != DEFAULT_SEED
    ):
        raise ValueError(f"SPAI seed must be exactly {DEFAULT_SEED}")
    if mode == "preflight":
        if (
            bool(args.resume)
            or bool(args.fail_fast)
            or args.run_id is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError(
                "SPAI preflight accepts no run/selection/resume/CUDA options"
            )
        report = run_cpu_preflight(
            repo_root=repo_root,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            golden_root=golden_root,
        )
        source, assets, state = verify_assets(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            golden_root=golden_root,
        )
        del state
        _validate_preflight_report(report, source=source, assets=assets)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if mode != "formal" and args.run_id is None:
        raise ValueError("SPAI smoke and single modes require --run-id")
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _anchored(
        Path(args.dataset_manifest),
        repo_root,
    )
    results_root = _ensure_repo_child(
        Path(args.results_dir),
        repo_root=repo_root,
        label="SPAI results root",
    )
    artifacts_root = _ensure_repo_child(
        Path(args.artifacts_dir),
        repo_root=repo_root,
        label="SPAI artifacts root",
    )
    if (
        repo_relative(artifacts_root, repo_root)
        != DEFAULT_ARTIFACTS_DIR.as_posix()
    ):
        raise ValueError(
            f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}"
        )
    run_dir = results_root / run_id
    artifact_root = artifacts_root / run_id
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    # This CPU gate intentionally precedes dataset loading and accelerator setup.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        golden_root=golden_root,
    )
    source, assets, state = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        golden_root=golden_root,
    )
    _validate_preflight_report(cpu_preflight, source=source, assets=assets)

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
    visibility = selection_visibility_census(
        selected,
        repo_root=repo_root,
    )
    device, runtime = configure_runtime(device_text, seed=DEFAULT_SEED)
    gate_model = None
    try:
        gate_model, gate_model_load = legacy.load_model(
            state=state,
            source_root=source_root,
            device=device,
        )
        if gate_model_load != cpu_preflight["model_load"]:
            raise ValueError(
                "SPAI execution-device model load differs from CPU gate"
            )
        execution_device_golden = validate_execution_device_golden(
            model=gate_model,
            device=device,
            golden_root=golden_root,
        )
    finally:
        if gate_model is not None:
            del gate_model
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()
    adapter_sources = adapter_source_contract(repo_root)
    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        selection_visibility=visibility,
        adapter_sources=adapter_sources,
        source=source,
        assets=assets,
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        execution_device_golden=execution_device_golden,
        run_dir=run_dir,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    fingerprint = _fingerprint(immutable)
    _prepare_output_directories(
        repo_root=repo_root,
        run_dir=run_dir,
        artifact_root=artifact_root,
        resume=bool(args.resume),
    )

    prior_status: str | None = None
    prior_outputs: Mapping[str, Any] = {}
    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "SPAI resume requires manifest and expected inputs"
            )
        prior_manifest = _load_json_strict(manifest_path, "prior manifest")
        prior_status, started_at, prior_outputs = _validate_prior_manifest(
            manifest=prior_manifest,
            run_id=run_id,
            fingerprint=fingerprint,
            immutable=immutable,
            selected=selected,
            expected_path=expected_path,
            dataset_contract=dataset_contract.as_dict(),
            dataset_manifest_path=dataset_manifest_path,
            manifest_sha256=release.manifest_sha256,
            repo_root=repo_root,
            results_path=results_path,
            summary_path=summary_path,
        )
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
            raise ValueError("prior SPAI result is outside selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    prior_artifact_files = _validate_artifact_inventory(
        artifact_root=artifact_root,
        latest_by_sample_id=latest_before.latest_by_sample_id,
    )
    if (
        args.resume
        and prior_status in ("complete", "incomplete")
        and prior_outputs.get("artifact_files") != prior_artifact_files
    ):
        raise ValueError("SPAI finalized resume artifact count drifted")

    pending_ids = set(latest_before.pending_sample_ids(retry_errors=True))
    successful_before = sum(
        row.get("status") == "ok"
        for row in latest_before.latest_by_sample_id.values()
    )
    model = None
    model_load: Mapping[str, Any] | None = None
    if pending_ids or successful_before:
        model, model_load = legacy.load_model(
            state=state,
            source_root=source_root,
            device=device,
        )
        if model_load != cpu_preflight["model_load"]:
            raise ValueError("SPAI inference model load differs from CPU gate")
    replayed_before = 0
    if successful_before:
        if model is None:
            raise AssertionError("SPAI resume replay model is missing")
        replayed_before = _validate_latest_artifact_replay(
            latest_by_sample_id=latest_before.latest_by_sample_id,
            inputs_by_id=inputs_by_id,
            repo_root=repo_root,
            run_id=run_id,
            fingerprint=fingerprint,
            model=model,
            device=device,
        )
    # No resume state is mutated before all old rows/artifacts replay exactly.
    atomic_write_json(manifest_path, manifest)

    new_successes = 0
    resume_skips = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        for index, input_row in enumerate(selected, start=1):
            sample_id = str(input_row["sample_id"])
            if sample_id not in pending_ids:
                resume_skips += 1
                print(
                    f"[{index}/{len(selected)}] resume {sample_id}",
                    flush=True,
                )
                continue
            artifact_paths = [
                artifact_root / kind / f"{sample_id}.npy"
                for kind in ("patch_features", "feature", "attention")
            ]
            try:
                input_path = _safe_repo_file(
                    str(input_row["canonical_path"]),
                    repo_root=repo_root,
                    label=f"{sample_id} canonical input",
                )
                if sha256_file(input_path) != input_row["canonical_sha256"]:
                    raise ValueError(
                        f"{sample_id} canonical input SHA-256 changed"
                    )
                preprocess_started = time.perf_counter()
                image, preprocess = legacy.preprocess_image(input_path)
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if preprocess.get("native_size") != [
                    int(input_row["width"]),
                    int(input_row["height"]),
                ]:
                    raise ValueError("SPAI preprocessed dimensions changed")
                if model is None:
                    raise RuntimeError("SPAI inference model was not loaded")
                processed, patch, feature, attention, peak, latency = (
                    legacy.infer_one(model, device, image)
                )
                result = _build_ok_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    artifact_root=artifact_root,
                    processed=processed,
                    patch=patch,
                    feature=feature,
                    attention=attention,
                    preprocess=preprocess,
                    preprocess_latency_ms=preprocess_latency_ms,
                    latency_ms=latency,
                    peak_cuda_memory_bytes=0 if peak is None else int(peak),
                )
                arrays = _validate_runner_attempt(
                    result,
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    run_manifest_fingerprint=fingerprint,
                )
                if arrays is None:
                    raise AssertionError("SPAI success validation lost arrays")
                _replay_artifacts(
                    row=result,
                    arrays=arrays,
                    model=model,
                    device=device,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"ai_score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                for artifact_path in artifact_paths:
                    if artifact_path.exists():
                        if (
                            artifact_path.is_symlink()
                            or not artifact_path.is_file()
                        ):
                            raise ValueError(
                                f"unsafe failed SPAI artifact: {artifact_path}"
                            ) from error
                        artifact_path.unlink()
                result = _build_error_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    error=error,
                )
                _validate_runner_attempt(
                    result,
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    run_manifest_fingerprint=fingerprint,
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
            raise ValueError("SPAI result is outside selection")
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
    artifact_files = _validate_artifact_inventory(
        artifact_root=artifact_root,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    successful_after = sum(
        row.get("status") == "ok"
        for row in latest.latest_by_sample_id.values()
    )
    replayed_after = 0
    replay_model = None
    try:
        if successful_after:
            replay_model, replay_load = legacy.load_model(
                state=state,
                source_root=source_root,
                device=device,
            )
            if replay_load != cpu_preflight["model_load"]:
                raise ValueError("SPAI final replay model differs from CPU gate")
            replayed_after = _validate_latest_artifact_replay(
                latest_by_sample_id=latest.latest_by_sample_id,
                inputs_by_id=inputs_by_id,
                repo_root=repo_root,
                run_id=run_id,
                fingerprint=fingerprint,
                model=replay_model,
                device=device,
            )
    finally:
        if replay_model is not None:
            del replay_model
        del state
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()

    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_spai_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "score_spec": SCORE_SPEC.as_dict(),
        "dataset_contract": dataset_contract.as_dict(),
        "selection_visibility_census": visibility,
        "same_device_artifact_replays_before_execution": replayed_before,
        "same_device_artifact_replays_final": replayed_after,
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
        "same_device_artifact_replays_before_execution": replayed_before,
        "same_device_artifact_replays_final": replayed_after,
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": artifact_files,
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
        raise RuntimeError("SPAI fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
