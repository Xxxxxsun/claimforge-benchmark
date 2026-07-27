#!/usr/bin/env python3
"""Run the pinned official B-Free DINO2reg4 detector on Balanced250.

This v2 orchestration layer leaves the audited Mouse-v1 implementation
unchanged.  It executes B-Free's official native-resolution, five-crop
whole-image classifier on the 1,775-image Balanced250 score cache (or a
frozen smoke/single selection), while persisting the exact five classifier
features and five crop logits in a local-only, gitignored NPZ artifact.

B-Free is T1-only.  Crop visibility is an input-condition diagnostic, never a
predicted mask, localization output, or joint T1/T2 score.  Scientific metrics
belong to ``analyze_bfree_balanced.py`` and the shared Balanced250 metrics.
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
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_bfree as legacy
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


RUN_MANIFEST_SCHEMA = "bfree_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "bfree_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "bfree_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "bfree_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/bfree")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/bfree")
DEFAULT_FORMAL_RUN_ID = "bfree_dino2reg4_balanced250_v1_full1775_20260727"
DEFAULT_SMOKE_RUN_ID_A = (
    "bfree_dino2reg4_balanced250_v1_smoke5x7_a_r3_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "bfree_dino2reg4_balanced250_v1_smoke5x7_b_r3_20260727"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_WEIGHTS_DIR = legacy.DEFAULT_WEIGHTS_DIR
DEFAULT_WEIGHTS_ZIP = legacy.DEFAULT_WEIGHTS_ZIP
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = legacy.MODEL_SEED
FROZEN_PYTHONHASHSEED = "0"
FROZEN_PROFILE = legacy.PREPROCESS_PROFILE
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 2 * 1024**3

FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/bfree/bin/python"
)
FROZEN_VENV_PREFIX = Path("/root/.cache/claimforge/venvs/bfree")
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/bfree-balanced-v2-empty"
)
FROZEN_PYVENV_CONFIG_SHA256 = (
    "1ee492ad073827f75ebf74bf270e554ee23a28ee44756d616218d4bd6e40c6cc"
)
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torch_distribution": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "torchvision_distribution": "0.23.0.dev20250627+cu128",
    "timm": "1.0.12",
    "transformers": "4.43.4",
    "safetensors": "0.5.2",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "PyYAML": "6.0.2",
    "scikit-learn": "1.6.1",
    "scipy": "1.16.0",
    "joblib": "1.4.2",
    "threadpoolctl": "3.5.0",
    "setuptools": "79.0.1",
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
        "full": 159,
        "partial": 32,
        "none": 59,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.6928341953878898,
        "replicate_wrap": 23,
        "distinct_crop_starts": {"1": 23, "3": 1, "5": 226},
    },
    "local_cat": {
        "full": 111,
        "partial": 118,
        "none": 21,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.75827158637198,
        "replicate_wrap": 24,
        "distinct_crop_starts": {"1": 24, "3": 1, "5": 225},
    },
    "local_trash_can": {
        "full": 47,
        "partial": 195,
        "none": 8,
        "total": 250,
        "mean_edit_visible_gt_fraction": 0.7096941433813817,
        "replicate_wrap": 21,
        "distinct_crop_starts": {"1": 21, "3": 1, "5": 228},
    },
    "all_local": {
        "full": 317,
        "partial": 345,
        "none": 88,
        "total": 750,
        "mean_edit_visible_gt_fraction": 0.7202666417137505,
        "replicate_wrap": 68,
        "distinct_crop_starts": {"1": 68, "3": 3, "5": 679},
    },
}
FORMAL_GEOMETRY_CENSUS = {
    "replicate_wrap_by_condition": {
        "real": 26,
        "local_mouse": 23,
        "local_cat": 24,
        "local_trash_can": 21,
        "fullframe_mouse": 25,
        "fullframe_cat": 23,
        "fullframe_trash_can": 23,
    },
    "replicate_wrap_total": 165,
    "distinct_crop_starts_all": {"1": 165, "3": 7, "5": 1603},
    "distinct_crop_starts_by_condition": {
        "real": {"1": 26, "3": 1, "5": 248},
        "local_mouse": {"1": 23, "3": 1, "5": 226},
        "local_cat": {"1": 24, "3": 1, "5": 225},
        "local_trash_can": {"1": 21, "3": 1, "5": 228},
        "fullframe_mouse": {"1": 25, "3": 1, "5": 224},
        "fullframe_cat": {"1": 23, "3": 1, "5": 226},
        "fullframe_trash_can": {"1": 23, "3": 1, "5": 226},
    },
}

PREPROCESS_CONTRACT = {
    "profile_id": FROZEN_PROFILE,
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "resize": False,
    "tensor": "torchvision_ToTensor_uint8_div_255_float32",
    "normalization_mean": list(legacy.IMAGE_MEAN),
    "normalization_std": list(legacy.IMAGE_STD),
    "patch_projection_kernel": legacy.PATCH_STRIDE,
    "patch_projection_stride": legacy.PATCH_STRIDE,
    "right_bottom_remainder": "discarded",
    "replicate_wrap": (
        "if_either_grid_dimension_below_36_repeat_both_then_truncate_to_36"
    ),
    "crop_size_pixels": legacy.CROP_SIZE,
    "crop_count": legacy.CROP_COUNT,
    "crop_order": [
        "center",
        "top_left",
        "bottom_left",
        "bottom_right",
        "top_right",
    ],
    "batch_size": 1,
}
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_FILE_BYTES = 15_904
ARTIFACT_CONTRACT = {
    "format": "NumPy NPZ, allow_pickle=False, ZIP_STORED",
    "keys": ["features", "crop_logits"],
    "file_bytes": ARTIFACT_FILE_BYTES,
    "zip_members": {
        "features.npy": {
            "compress_type": zipfile.ZIP_STORED,
            "file_size": 15_488,
            "compress_size": 15_488,
        },
        "crop_logits.npy": {
            "compress_type": zipfile.ZIP_STORED,
            "file_size": 148,
            "compress_size": 148,
        },
    },
    "features": {
        "shape": [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "nbytes": legacy.CROP_COUNT * legacy.FEATURE_DIMENSION * 4,
        "semantics": legacy.FEATURE_SEMANTICS,
    },
    "crop_logits": {
        "shape": [legacy.CROP_COUNT],
        "dtype": "float32",
        "nbytes": legacy.CROP_COUNT * 4,
        "semantics": "five_official_crop_raw_logits_in_official_crop_order",
    },
    "finite": True,
    "exact_same_device_head_replay": True,
    "visibility": "local_only_gitignored_output",
}

TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}

ASSET_IDENTITY = {
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "source_files": legacy.SOURCE_FILES,
    "official_zip": legacy.OFFICIAL_ZIP,
    "config": legacy.CONFIG,
    "checkpoint": legacy.CHECKPOINT,
}
EXPECTED_ASSET_BUNDLE_SHA256 = (
    "58859ff170ba42edd9c13bfcbc0094513de227d7001e5a261f7c37dd69db8349"
)
MODEL_CONTRACT = {
    "name": legacy.MODEL_NAME,
    "slug": legacy.MODEL_SLUG,
    "architecture": legacy.MODEL_ARCH,
    "repository": legacy.MODEL_REPO_URL,
    "paper": legacy.PAPER_URL,
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "checkpoint_id": legacy.CHECKPOINT["id"],
    "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
    "checkpoint_schema_sha256": legacy.CHECKPOINT["schema_sha256"],
    "asset_bundle_sha256": EXPECTED_ASSET_BUNDLE_SHA256,
    "construction": "official_Wrapper5crops_strict_full_state",
    "feature_shape": [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
    "score": {
        "semantics": legacy.SCORE_SEMANTICS,
        "direction": "higher_means_fake",
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
    },
    "license": legacy.LICENSE_RECORD,
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_bfree_balanced.py",
    "eval/opensource/analyze_bfree_balanced.py",
    "eval/opensource/run_bfree.py",
    "eval/opensource/analyze_bfree_run.py",
    "eval/opensource/bfree_metrics.py",
    "eval/opensource/whole_image_metrics.py",
    "eval/opensource/maskclip_metrics.py",
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
        "formal_geometry_census",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "execution_model_load",
        "execution_official_golden",
        "artifact_contract",
        "local_artifact_policy",
        "outputs",
    }
)

CPU_GOLDEN_SAMPLE_ID = "2c80d38ac19c2d3b76950996"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/" f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "12607f3cdada1480038f3d506146cdc1fa0c1c50034afda5e3a5f175433e716b"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "5a4747a6e3a8313f8c9ec3dde2504bb53184666276d7e54dc5fab53ca0e7194b"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "bf55e6ebe26e1da9ad303753a289830de9bda761766ea8bbbb4d4ad5cb938d2e"
)
CPU_GOLDEN_FEATURE_ARRAY_SHA256 = (
    "c08c6452aabec2e9a842ea68ab2fbc91ef4612c033317bcc5a3060f0e67f73fc"
)
CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256 = (
    "0372c0cf2fed737fe8e93bbcb6f561fecefd9723c9ba9b2451da580b51290ae0"
)
CPU_GOLDEN_ARTIFACT_SHA256 = (
    "ad50cd98e2d66fe773d2f598385e0054ff56b5f5cfe86beb5ecfa09ff6e9b61d"
)
CPU_GOLDEN_RAW_LOGIT = -4.131394863128662
CPU_GOLDEN_FAKE_PROBABILITY = 0.01580660045146942
CPU_GOLDEN_CROP_LOGITS = [
    -6.979626655578613,
    -5.850933074951172,
    -3.1853511333465576,
    -3.072199583053589,
    -1.5688656568527222,
]


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _same_json_type_and_value(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_same_json(left: Any, right: Any, label: str) -> None:
    if not _same_json_type_and_value(left, right):
        raise ValueError(f"{label} changed")


def _require_json_native_round_trip(value: Any, label: str) -> None:
    """Require a value to survive canonical JSON without type coercion."""

    try:
        decoded = json.loads(
            stable_json(value),
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical finite JSON") from error
    if not _same_json_type_and_value(decoded, value):
        raise ValueError(f"{label} contains non-native JSON values")


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _npz_bytes(features: np.ndarray, crop_logits: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.savez(
        handle,
        features=np.ascontiguousarray(features, dtype=np.float32),
        crop_logits=np.ascontiguousarray(crop_logits, dtype=np.float32),
    )
    return handle.getvalue()


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
            raise FileNotFoundError(
                f"missing or unsafe B-Free Balanced source: {path}"
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
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("formal B-Free Balanced250 selection drifted")
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
        raise ValueError("frozen B-Free 5x7 smoke selection drifted")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if release.release_kind != "balanced250":
        raise ValueError("B-Free v2 requires the Balanced250 release")
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
            raise ValueError("single B-Free selection drifted")
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
        Path(os.path.abspath(raw_prefix))
        if isinstance(raw_prefix, str)
        else None
    )
    if (
        os.environ.get("PYTHONHASHSEED") != FROZEN_PYTHONHASHSEED
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("NO_ALBUMENTATIONS_UPDATE") != "1"
        or actual_prefix != expected_prefix
        or not expected_prefix.is_absolute()
        or expected_prefix.is_symlink()
        or (expected_prefix.exists() and not expected_prefix.is_dir())
        or (expected_prefix.is_dir() and any(expected_prefix.iterdir()))
        or sys.dont_write_bytecode is not True
        or Path(os.path.abspath(str(sys.pycache_prefix))) != expected_prefix
    ):
        raise RuntimeError(
            "B-Free startup isolation requires "
            f"PYTHONHASHSEED={FROZEN_PYTHONHASHSEED}, "
            "PYTHONDONTWRITEBYTECODE=1, NO_ALBUMENTATIONS_UPDATE=1, "
            "and an absolute empty "
            f"PYTHONPYCACHEPREFIX={FROZEN_PYTHONPYCACHEPREFIX}"
        )
    return {
        "PYTHONHASHSEED": FROZEN_PYTHONHASHSEED,
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
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
        **{
            name: str(_package_version(name))
            for name in (
                "timm",
                "transformers",
                "safetensors",
                "Pillow",
                "PyYAML",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "setuptools",
            )
        },
        "numpy": str(np.__version__),
    }
    if actual != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            "B-Free dedicated runtime version drifted: "
            f"expected {FROZEN_RUNTIME_VERSIONS}, got {actual}"
        )
    executable = Path(os.path.abspath(sys.executable))
    frozen = Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
    if executable != frozen:
        raise RuntimeError(
            f"B-Free must run in its frozen environment: {frozen}"
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
        raise RuntimeError("B-Free virtual-environment contract drifted")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        normalized = key.strip()
        if separator != "=" or not normalized or normalized in values:
            raise RuntimeError("B-Free pyvenv.cfg is malformed")
        values[normalized] = value.strip()
    if values != {
        "home": "/usr/bin",
        "include-system-site-packages": "true",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": (
            "/usr/bin/python -m venv --system-site-packages "
            "/root/.cache/claimforge/venvs/bfree"
        ),
    }:
        raise RuntimeError("B-Free pyvenv.cfg values drifted")
    return {
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "pyvenv_cfg_path": str(config_path),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }


def configure_runtime(
    device_text: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the dedicated environment and legacy numerical runtime."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"B-Free seed must be exactly {DEFAULT_SEED}")
    process_environment = _startup_isolation_contract()
    _configure_cublas_workspace()
    import torch

    versions = _frozen_runtime_versions()
    venv = _venv_contract()
    if device_text == "cpu":
        requested = "cpu"
    elif re.fullmatch(r"cuda:[0-9]+", device_text):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        candidate = torch.device(device_text)
        if candidate.index is None or candidate.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device_text}")
        torch.cuda.set_device(candidate)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(candidate)
        if int(free_bytes) < MINIMUM_CUDA_FREE_BYTES:
            raise RuntimeError(
                f"{candidate} has only {int(free_bytes)} free bytes; "
                f"B-Free requires at least {MINIMUM_CUDA_FREE_BYTES}"
            )
        requested = device_text
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    device, legacy_runtime = legacy.configure_runtime(requested)
    if str(device) != requested:
        raise RuntimeError("legacy B-Free runtime resolved a different device")
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
            **{
                key: versions[key]
                for key in (
                    "timm",
                    "transformers",
                    "safetensors",
                    "numpy",
                    "Pillow",
                    "PyYAML",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                    "setuptools",
                )
            },
        },
        "seed": seed,
        "preprocess_profile": FROZEN_PROFILE,
        "inference_dtype": "float32",
        "feature_dtype": "float32",
        "crop_logit_dtype": "float32",
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
            "allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "minimum_cuda_free_bytes": MINIMUM_CUDA_FREE_BYTES,
        "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
        "process_environment": process_environment,
        # torch.__version__ may be a ``str`` subclass.  Normalize the
        # complete legacy evidence now so a JSON manifest round-trip cannot
        # make a valid run impossible to resume.
        "legacy_runtime": json.loads(stable_json(dict(legacy_runtime))),
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
        "crop_logit_dtype",
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
        "legacy_runtime",
    }
    if str(device).startswith("cuda:"):
        expected_keys.add("cuda")
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    expected_executable = str(
        Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))
    )
    if not _same_json_type_and_value(
        value.get("python"),
        {
            "implementation": "CPython",
            "version": FROZEN_RUNTIME_VERSIONS["python"],
            "executable": expected_executable,
        },
    ):
        raise ValueError(f"{label}.python dedicated runtime changed")
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    if not _same_json_type_and_value(
        value.get("venv"),
        {
            "prefix": str(expected_prefix),
            "base_prefix": "/usr",
            "pyvenv_cfg_path": str(expected_prefix / "pyvenv.cfg"),
            "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
            "include_system_site_packages": True,
        },
    ):
        raise ValueError(f"{label}.venv evidence changed")
    packages = value.get("packages")
    expected_package_keys = {
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "safetensors",
        "numpy",
        "Pillow",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "joblib",
        "threadpoolctl",
        "setuptools",
    }
    if not isinstance(packages, Mapping) or set(packages) != expected_package_keys:
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
            for key in expected_package_keys - {"torch", "torchvision"}
        )
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    expected_environment = {
        "PYTHONHASHSEED": FROZEN_PYTHONHASHSEED,
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPYCACHEPREFIX": str(
            Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
        ),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(
            Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
        ),
        "pycache_prefix_initially_empty": True,
    }
    if (
        not isinstance(value.get("platform"), str)
        or not value["platform"]
        or value.get("seed") != DEFAULT_SEED
        or value.get("preprocess_profile") != FROZEN_PROFILE
        or value.get("inference_dtype") != "float32"
        or value.get("feature_dtype") != "float32"
        or value.get("crop_logit_dtype") != "float32"
        or type(value.get("batch_size")) is not int
        or value["batch_size"] != 1
        or value.get("autocast") is not False
        or value.get("grad_enabled") is not False
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("minimum_cuda_free_bytes") != MINIMUM_CUDA_FREE_BYTES
        or value.get("bytecode_writes_disabled") is not True
        or not _same_json_type_and_value(
            value.get("process_environment"),
            expected_environment,
        )
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = value.get("cudnn")
    if not _same_json_type_and_value(
        cudnn,
        {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    legacy_runtime = value.get("legacy_runtime")
    if (
        not isinstance(legacy_runtime, Mapping)
        or legacy_runtime.get("device") != device
        or legacy_runtime.get("seed") != DEFAULT_SEED
        or legacy_runtime.get("dtype") != "float32"
        or legacy_runtime.get("autocast") is not False
        or legacy_runtime.get("deterministic_algorithms") is not True
        or legacy_runtime.get("network_allowed") is not False
    ):
        raise ValueError(f"{label}.legacy runtime changed")
    if str(device).startswith("cuda:"):
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
            or cuda.get("device_index") != int(str(device).split(":", 1)[1])
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{label}.cuda evidence changed")
    return value


def _asset_bundle_sha256() -> str:
    return _fingerprint(ASSET_IDENTITY)


def verify_assets(
    *,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    source, assets, state = legacy.verify_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    schema = assets.get("checkpoint", {}).get("schema", {})
    bundle = _asset_bundle_sha256()
    if (
        bundle != EXPECTED_ASSET_BUNDLE_SHA256
        or source.get("commit") != legacy.MODEL_SOURCE_COMMIT
        or assets.get("zip", {}).get("verified_sha256")
        != legacy.OFFICIAL_ZIP["sha256"]
        or assets.get("config", {}).get("sha256") != legacy.CONFIG["sha256"]
        or assets.get("checkpoint", {}).get("sha256")
        != legacy.CHECKPOINT["sha256"]
        or schema.get("schema_sha256") != legacy.CHECKPOINT["schema_sha256"]
        or schema.get("tensor_count") != legacy.CHECKPOINT["tensor_count"]
        or schema.get("state_elements") != legacy.CHECKPOINT["state_elements"]
    ):
        raise ValueError("B-Free source/asset contract drifted")
    enriched = dict(assets)
    enriched["bundle_sha256"] = bundle
    return source, enriched, state


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
        raise ValueError("B-Free Balanced CPU golden input changed")
    records: list[dict[str, Any]] = []
    for _index in range(2):
        image, preprocess = legacy.preprocess_image(path)
        scoring, features, crop_logits, peak, _latency = legacy.infer_one(
            model,
            device,
            image,
        )
        payload = _npz_bytes(features, crop_logits)
        records.append(
            {
                "preprocess": preprocess,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_bytes": len(payload),
                "feature_array_sha256": _array_sha256(features),
                "crop_logits_array_sha256": _array_sha256(crop_logits),
                "raw_logit": scoring["raw_logit"],
                "ai_score": scoring["ai_score"],
                "fake_probability": scoring["fake_probability"],
                "crop_logits": scoring["crop_logits"],
                "classification_decision": scoring[
                    "classification_decision"
                ],
                "model_forward_calls": scoring["manual_replay"][
                    "model_forward_calls"
                ],
                "classifier_hook_calls": scoring["manual_replay"][
                    "classifier_hook_calls"
                ],
                "peak_cuda_memory_bytes": 0 if peak is None else int(peak),
            }
        )
    first, second = records
    if not _same_json_type_and_value(first, second):
        raise ValueError("B-Free Balanced CPU golden is not bit-exact")
    preprocess = first["preprocess"]
    expected = {
        "artifact_sha256": CPU_GOLDEN_ARTIFACT_SHA256,
        "artifact_bytes": ARTIFACT_FILE_BYTES,
        "feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "crop_logits_array_sha256": CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256,
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "ai_score": CPU_GOLDEN_RAW_LOGIT,
        "fake_probability": CPU_GOLDEN_FAKE_PROBABILITY,
        "crop_logits": CPU_GOLDEN_CROP_LOGITS,
        "classification_decision": False,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
        "peak_cuda_memory_bytes": 0,
    }
    if (
        preprocess.get("decoded_rgb_sha256")
        != CPU_GOLDEN_DECODED_RGB_SHA256
        or preprocess.get("tensor_sha256") != CPU_GOLDEN_TENSOR_SHA256
        or preprocess.get("tensor_shape") != [3, 1350, 1800]
        or preprocess.get("profile") != FROZEN_PROFILE
        or any(
            not _same_json_type_and_value(first.get(key), value)
            for key, value in expected.items()
        )
    ):
        raise ValueError("B-Free Balanced CPU golden output changed")
    return {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": preprocess,
        **expected,
        "repeat_artifact_sha256": second["artifact_sha256"],
        "repeat_feature_array_sha256": second["feature_array_sha256"],
        "repeat_crop_logits_array_sha256": second[
            "crop_logits_array_sha256"
        ],
        "repeat_raw_logit": second["raw_logit"],
        "repeat_ai_score": second["ai_score"],
        "repeat_fake_probability": second["fake_probability"],
        "repeat_crop_logits": second["crop_logits"],
        "repeat_classification_decision": second[
            "classification_decision"
        ],
        "repeat_model_forward_calls": second["model_forward_calls"],
        "repeat_classifier_hook_calls": second["classifier_hook_calls"],
        "repeat_byte_exact": True,
    }


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
) -> dict[str, Any]:
    """Run source, asset, official four-demo, and Balanced CPU gates."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before B-Free CPU preflight")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    if device.type != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("B-Free CPU preflight configured a non-CPU runtime")
    source, assets, state = verify_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    model = None
    try:
        model, model_load = legacy.load_model(
            state=state,
            source_root=source_root,
            device=device,
        )
        official = legacy.validate_official_golden(
            model=model,
            device=device,
            source_root=source_root,
        )
        balanced = _balanced_golden_record(
            model=model,
            device=device,
            repo_root=repo_root,
        )
        if torch.cuda.is_initialized():
            raise RuntimeError("B-Free CPU preflight initialized CUDA")
        return {
            "schema_version": CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "assets": assets,
            "model_load": model_load,
            "runtime": runtime,
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
        gc.collect()


_FALSE_DECLARATIONS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULL_DECLARATIONS = frozenset(
    {
        "localization_output",
        "localisation_output",
    }
)
_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "pair_rank",
        "t2",
        "joint",
        "joint_score",
        "localization",
        "localisation",
        "dense_output",
        "heatmap",
        "mask",
        "score_map",
        "predicted_mask",
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
        "gt_mask_kind",
        "used_native_pixel_rule",
        "visible_positive_pixels",
        "positive_pixels",
    }
)


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    """Reject pair/T2/joint claims at every nesting depth."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(child, child_label)
                continue
            if normalized == "pair_rank":
                raise ValueError(f"{child_label} is an unsupported B-Free claim")
            if normalized in _FALSE_DECLARATIONS:
                if child is not False:
                    raise ValueError(
                        f"{child_label} is an unsupported B-Free claim"
                    )
                continue
            if normalized in _NULL_DECLARATIONS:
                if child is not None:
                    raise ValueError(
                        f"{child_label} is an unsupported B-Free claim"
                    )
                continue
            if normalized in _FORBIDDEN_CLAIM_KEYS or normalized.startswith(
                _FORBIDDEN_CLAIM_PREFIXES
            ):
                raise ValueError(
                    f"{child_label} is an unsupported B-Free claim"
                )
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
    """Measure exact local GT within the official five-crop receptive union."""

    geometry = legacy.compute_preprocess_geometry(
        int(row["width"]),
        int(row["height"]),
    )
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        gt = load_ground_truth(row, repo_root)
        if gt is None or gt.dtype != np.bool_:
            raise ValueError("local exact-diff input has no boolean GT mask")
        y, x = np.nonzero(gt)
        total = int(x.size)
        if total <= 0:
            raise ValueError("local exact-diff mask is empty")
        visible_mask = np.zeros(total, dtype=bool)
        for left, top, right, bottom in geometry[
            "used_native_rectangles_xyxy"
        ]:
            visible_mask |= (
                (x >= int(left))
                & (x < int(right))
                & (y >= int(top))
                & (y < int(bottom))
            )
        visible = int(np.count_nonzero(visible_mask))
        fraction = visible / total
        category = (
            "none" if visible == 0 else "full" if visible == total else "partial"
        )
        return {
            "edit_visibility": category,
            "edit_visible_gt_fraction": fraction,
            "edit_visibility_evidence": {
                "basis": (
                    "exact_diff_positive_pixels_intersecting_union_of_"
                    "official_five_patch_crop_receptive_fields"
                ),
                "role": "input_condition_stratum_not_model_localization",
                "gt_mask_kind": "exact_diff",
                "positive_pixels": total,
                "visible_positive_pixels": visible,
                "geometry": geometry,
            },
        }
    expected_kind = (
        "all_zero" if row.get("condition") == "real" else "not_applicable"
    )
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
            "role": "input_condition_stratum_not_model_localization",
            "gt_mask_kind": gt_kind,
            "geometry": geometry,
        },
    }


def selection_visibility_census(
    selected: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Return visibility and crop-geometry census for one exact selection."""

    by_condition: dict[str, dict[str, Any]] = {}
    all_counts: Counter[str] = Counter()
    all_fractions: list[float] = []
    all_local_wrap = 0
    all_local_starts: Counter[int] = Counter()
    wrap_by_condition: Counter[str] = Counter()
    starts_by_condition: dict[str, Counter[int]] = {
        condition: Counter() for condition in BALANCED_CONDITIONS
    }
    for row in selected:
        condition = str(row["condition"])
        geometry = legacy.compute_preprocess_geometry(
            int(row["width"]),
            int(row["height"]),
        )
        wrap_by_condition[condition] += int(
            bool(geometry["replicate_wrap_applied"])
        )
        starts_by_condition[condition][
            int(geometry["distinct_crop_starts"])
        ] += 1
    for condition in ("local_mouse", "local_cat", "local_trash_can"):
        rows = [row for row in selected if row.get("condition") == condition]
        counts: Counter[str] = Counter()
        fractions: list[float] = []
        wrapped = 0
        starts: Counter[int] = Counter()
        for row in rows:
            diagnostic = _visibility_diagnostic(row, repo_root=repo_root)
            category = str(diagnostic["edit_visibility"])
            fraction = float(diagnostic["edit_visible_gt_fraction"])
            geometry = diagnostic["edit_visibility_evidence"]["geometry"]
            if (
                category not in ("full", "partial", "none")
                or not math.isfinite(fraction)
                or not 0.0 <= fraction <= 1.0
            ):
                raise ValueError("B-Free local visibility diagnostic changed")
            counts[category] += 1
            fractions.append(fraction)
            wrapped += int(bool(geometry["replicate_wrap_applied"]))
            starts[int(geometry["distinct_crop_starts"])] += 1
            all_counts[category] += 1
            all_fractions.append(fraction)
            all_local_wrap += int(bool(geometry["replicate_wrap_applied"]))
            all_local_starts[int(geometry["distinct_crop_starts"])] += 1
        by_condition[condition] = {
            "full": counts["full"],
            "partial": counts["partial"],
            "none": counts["none"],
            "total": len(rows),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
            "replicate_wrap": wrapped,
            "distinct_crop_starts": {
                str(key): value for key, value in sorted(starts.items())
            },
        }
    result = {
        "profile_id": FROZEN_PROFILE,
        "basis": (
            "exact_diff_positive_pixels_intersecting_union_of_official_"
            "five_patch_crop_receptive_fields"
        ),
        "role": "input_condition_stratum_not_model_localization",
        "by_condition": by_condition,
        "all_local": {
            "full": all_counts["full"],
            "partial": all_counts["partial"],
            "none": all_counts["none"],
            "total": len(all_fractions),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(all_fractions)) if all_fractions else None
            ),
            "replicate_wrap": all_local_wrap,
            "distinct_crop_starts": {
                str(key): value
                for key, value in sorted(all_local_starts.items())
            },
        },
        "not_applicable_images": sum(
            row.get("gt_mask_kind") != "exact_diff" for row in selected
        ),
        "replicate_wrap_by_condition": {
            condition: wrap_by_condition[condition]
            for condition in BALANCED_CONDITIONS
        },
        "replicate_wrap_total": sum(wrap_by_condition.values()),
        "distinct_crop_starts_by_condition": {
            condition: {
                str(key): value
                for key, value in sorted(
                    starts_by_condition[condition].items()
                )
            }
            for condition in BALANCED_CONDITIONS
        },
        "distinct_crop_starts_all": {
            str(key): value
            for key, value in sorted(
                sum(
                    (starts_by_condition[c] for c in BALANCED_CONDITIONS),
                    Counter(),
                ).items()
            )
        },
    }
    counts = Counter(str(row["condition"]) for row in selected)
    if dict(counts) == FORMAL_COUNTS:
        actual = {**by_condition, "all_local": result["all_local"]}
        for condition, expected in LOCAL_VISIBILITY_CENSUS.items():
            observed = actual[condition]
            for key in (
                "full",
                "partial",
                "none",
                "total",
                "replicate_wrap",
                "distinct_crop_starts",
            ):
                if observed[key] != expected[key]:
                    raise ValueError(
                        "formal B-Free local visibility census drifted"
                    )
            if not math.isclose(
                float(observed["mean_edit_visible_gt_fraction"]),
                float(expected["mean_edit_visible_gt_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("formal B-Free visibility mean drifted")
        observed_geometry = {
            key: result[key] for key in FORMAL_GEOMETRY_CENSUS
        }
        if observed_geometry != FORMAL_GEOMETRY_CENSUS:
            raise ValueError("formal B-Free crop geometry census drifted")
        if result["not_applicable_images"] != 1025:
            raise ValueError("formal B-Free non-local count drifted")
    return result


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    asset_bundle_sha256: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    """Build the B-Free extension of the shared v2 result identity."""

    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    if asset_bundle_sha256 != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("B-Free asset bundle SHA-256 changed")
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
        "checkpoint_id": legacy.CHECKPOINT["id"],
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "checkpoint_schema_sha256": legacy.CHECKPOINT["schema_sha256"],
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
        "bfree_artifact",
        "bfree_artifact_path",
        "bfree_artifact_sha256",
        "feature_array_sha256",
        "crop_logits_array_sha256",
        "feature_shape",
        "feature_dtype",
        "feature_nbytes",
        "feature_semantics",
        "crop_logits_shape",
        "crop_logits_dtype",
        "crop_logits_nbytes",
        "crop_logits_semantics",
        "artifact_paths",
        "raw_logit",
        "ai_score",
        "score",
        "fake_probability",
        "crop_logits",
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
_ERROR_NULL_FIELDS = _OK_RESULT_FIELDS


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


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    raw = _finite_number(row.get("raw_logit"), f"{sample_id} raw_logit")
    ai_score = _finite_number(row.get("ai_score"), f"{sample_id} ai_score")
    score = _finite_number(row.get("score"), f"{sample_id} score")
    if ai_score != raw or score != raw:
        raise ValueError(f"{sample_id} raw-logit aliases changed")
    probability = _finite_number(
        row.get("fake_probability"),
        f"{sample_id} fake_probability",
    )
    expected_probability = legacy.bfree_fake_probability_float32(raw)
    if (
        probability != expected_probability
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError(f"{sample_id} diagnostic sigmoid changed")
    crop_values = row.get("crop_logits")
    if (
        not isinstance(crop_values, list)
        or len(crop_values) != legacy.CROP_COUNT
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in crop_values
        )
    ):
        raise ValueError(f"{sample_id} crop logits changed")
    decision = raw > legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw,
        "ai_score": raw,
        "fake_probability": probability,
        "decision": decision,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": legacy.SCORE_SEMANTICS,
    }
    t1 = {
        key: value for key, value in classification.items() if key != "semantics"
    }
    t1["policy"] = legacy.T1_POLICY
    manual = {
        "crop_logits": [float(value) for value in crop_values],
        "raw_logit": raw,
        "ai_score": raw,
        "official_crop_logits_exact_match": True,
        "official_mean_exact_match": True,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
    }
    if (
        row.get("score_semantics") != legacy.SCORE_SEMANTICS
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold")
        != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or not _same_json_type_and_value(
            row.get("classification"),
            classification,
        )
        or not _same_json_type_and_value(row.get("t1"), t1)
        or not _same_json_type_and_value(
            row.get("manual_replay"),
            manual,
        )
    ):
        raise ValueError(
            f"{sample_id} score semantics/strict decision/replay changed"
        )


def _validate_npz_structure(payload: bytes, *, sample_id: str) -> None:
    if len(payload) != ARTIFACT_FILE_BYTES:
        raise ValueError(f"{sample_id} B-Free NPZ byte size changed")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            members = {
                info.filename: {
                    "compress_type": info.compress_type,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                }
                for info in infos
            }
            if (
                [info.filename for info in infos]
                != ["features.npy", "crop_logits.npy"]
                or members != ARTIFACT_CONTRACT["zip_members"]
                or any(
                    Path(info.filename).is_absolute()
                    or ".." in Path(info.filename).parts
                    or info.is_dir()
                    for info in infos
                )
            ):
                raise ValueError(
                    f"{sample_id} B-Free NPZ members changed"
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{sample_id} B-Free NPZ is invalid") from error


def _validate_artifact(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    repo_root: Path,
    run_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    artifact = row.get("bfree_artifact")
    expected_keys = {
        "relative_path",
        "sha256",
        "file_bytes",
        "feature_array_sha256",
        "crop_logits_array_sha256",
        "feature_shape",
        "crop_logits_shape",
        "dtype",
        "feature_nbytes",
        "crop_logits_nbytes",
        "finite",
        "feature_semantics",
        "crop_logits_semantics",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_keys:
        raise ValueError(f"{sample_id} B-Free artifact key set changed")
    expected_relative = (
        DEFAULT_ARTIFACTS_DIR
        / run_id
        / "bfree_artifacts"
        / f"{sample_id}.npz"
    ).as_posix()
    if artifact.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} B-Free artifact path changed")
    path = _safe_repo_file(
        expected_relative,
        repo_root=repo_root,
        label=f"{sample_id} B-Free NPZ",
    )
    payload = path.read_bytes()
    _validate_npz_structure(payload, sample_id=sample_id)
    file_sha = hashlib.sha256(payload).hexdigest()
    expected_feature_nbytes = legacy.CROP_COUNT * legacy.FEATURE_DIMENSION * 4
    expected_crop_nbytes = legacy.CROP_COUNT * 4
    if (
        artifact.get("sha256") != file_sha
        or artifact.get("file_bytes") != ARTIFACT_FILE_BYTES
        or not _same_json_type_and_value(
            artifact.get("feature_shape"),
            [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
        )
        or not _same_json_type_and_value(
            artifact.get("crop_logits_shape"),
            [legacy.CROP_COUNT],
        )
        or artifact.get("dtype") != "float32"
        or artifact.get("feature_nbytes") != expected_feature_nbytes
        or artifact.get("crop_logits_nbytes") != expected_crop_nbytes
        or artifact.get("finite") is not True
        or artifact.get("feature_semantics") != legacy.FEATURE_SEMANTICS
        or artifact.get("crop_logits_semantics")
        != "five_official_crop_raw_logits_in_official_crop_order"
    ):
        raise ValueError(f"{sample_id} B-Free artifact metadata changed")
    features, crop_logits = legacy._load_artifact(path)
    if payload != _npz_bytes(features, crop_logits):
        raise ValueError(f"{sample_id} B-Free NPZ bytes are not canonical")
    feature_sha = _array_sha256(features)
    crop_sha = _array_sha256(crop_logits)
    if (
        artifact.get("feature_array_sha256") != feature_sha
        or artifact.get("crop_logits_array_sha256") != crop_sha
    ):
        raise ValueError(f"{sample_id} B-Free artifact array hash changed")
    aliases = {
        "bfree_artifact_path": expected_relative,
        "bfree_artifact_sha256": file_sha,
        "feature_array_sha256": feature_sha,
        "crop_logits_array_sha256": crop_sha,
        "feature_shape": [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": expected_feature_nbytes,
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "crop_logits_shape": [legacy.CROP_COUNT],
        "crop_logits_dtype": "float32",
        "crop_logits_nbytes": expected_crop_nbytes,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
        "artifact_paths": {"bfree_npz": expected_relative},
    }
    for key, expected in aliases.items():
        if not _same_json_type_and_value(row.get(key), expected):
            raise ValueError(f"{sample_id} B-Free alias {key} changed")
    if not _same_json_type_and_value(
        row.get("crop_logits"),
        [float(value) for value in crop_logits.tolist()],
    ):
        raise ValueError(f"{sample_id} artifact crop-logit alias changed")
    return features, crop_logits


def _validate_latest_artifact_head_replay(
    *,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    state: Mapping[str, Any],
    device: Any,
) -> int:
    """Replay all successful [5,768] artifacts through the pinned head."""

    import torch
    from torch.nn import functional

    if (
        not isinstance(device, torch.device)
        or device.type not in ("cpu", "cuda")
        or (device.type == "cuda" and device.index is None)
        or (device.type == "cpu" and device.index is not None)
    ):
        raise ValueError("B-Free head replay needs an explicit device")
    weight = state.get("head.weight")
    bias = state.get("head.bias")
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
        raise ValueError("pinned B-Free head schema changed")
    replay_weight = weight.detach().to(device=device)
    replay_bias = bias.detach().to(device=device)
    expected = sum(
        row.get("status") == "ok" for row in latest_by_sample_id.values()
    )
    replayed = 0
    try:
        with torch.inference_mode():
            for sample_id, row in latest_by_sample_id.items():
                if row.get("status") != "ok":
                    continue
                features, crop_logits = _validate_artifact(
                    row,
                    sample_id=sample_id,
                    repo_root=repo_root,
                    run_id=run_id,
                )
                feature_tensor = torch.from_numpy(features).to(device=device)
                replay_crop = functional.linear(
                    feature_tensor,
                    replay_weight,
                    replay_bias,
                ).reshape(legacy.CROP_COUNT)
                expected_crop = torch.from_numpy(crop_logits).to(device=device)
                replay_mean = replay_crop.mean()
                raw = float(replay_mean.item())
                if (
                    not torch.equal(replay_crop, expected_crop)
                    or raw != float(row["raw_logit"])
                    or row.get("classification_decision")
                    is not (raw > legacy.CLASSIFICATION_THRESHOLD)
                ):
                    raise ValueError(
                        f"{sample_id} exact same-device B-Free head replay mismatch"
                    )
                replayed += 1
                del feature_tensor, replay_crop, expected_crop, replay_mean
    finally:
        del replay_weight, replay_bias
    if replayed != expected:
        raise ValueError("B-Free artifact replay coverage is incomplete")
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
        expected_keys |= _ERROR_NULL_FIELDS | {
            "error_type",
            "error",
            "traceback",
        }
    if set(attempt) != expected_keys:
        raise ValueError(
            "result attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, expected_value in expected.items():
        _require_same_json(
            attempt.get(key),
            expected_value,
            f"result attempt field {key}",
        )
    if (
        not isinstance(attempt.get("completed_at"), str)
        or not attempt["completed_at"]
    ):
        raise ValueError("result attempt completed_at is invalid")
    _reject_unsupported_claims(attempt, "result attempt")
    if status == "error":
        if (
            any(attempt.get(key) is not None for key in _ERROR_NULL_FIELDS)
            or
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
    _image, expected_preprocess = legacy.preprocess_image(input_path)
    if not _same_json_type_and_value(
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
    _validate_artifact(
        attempt,
        sample_id=sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )


def _validate_physical_attempt_history(
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Allow error retries, while rejecting duplicates/attempts after success."""

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
        / "bfree_artifacts"
        / "probe.npz"
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
        raise ValueError("B-Free artifact output is not gitignored") from error
    evidence = completed.stdout.strip()
    if (
        not evidence.startswith(".gitignore:")
        or "\t" not in evidence
        or not evidence.endswith(probe)
    ):
        raise ValueError("B-Free git-ignore evidence changed")
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
    execution_model_load: Mapping[str, Any],
    execution_official_golden: Mapping[str, Any],
    run_dir: Path,
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    artifact_dir: Path,
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
        "formal_geometry_census": FORMAL_GEOMETRY_CENSUS,
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_dataset_manifest_load": True,
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "execution_model_load": dict(execution_model_load),
        "execution_official_golden": dict(execution_official_golden),
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
            "artifact_dir": repo_relative(artifact_dir, repo_root),
        },
    }
    if set(immutable) != IMMUTABLE_CONFIG_KEYS:
        raise AssertionError("internal immutable B-Free config key set drifted")
    _reject_unsupported_claims(immutable, "immutable config")
    _require_json_native_round_trip(immutable, "immutable config")
    return immutable


def _ensure_repo_child(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    require_directory: bool = False,
) -> Path:
    """Resolve a repository child and reject every symlink component."""

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
        raise FileNotFoundError("missing or unsafe B-Free run directory")
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
        raise ValueError("B-Free run directory contains a non-regular entry")
    names = {entry.name for entry in entries}
    if not names <= allowed:
        raise ValueError(
            "B-Free run directory contains unexpected entries: "
            f"{sorted(names - allowed)}"
        )
    required = {"manifest.json", "expected_inputs.jsonl"}
    if not allow_missing_results:
        required |= {"results.jsonl", "summary.json"}
    if not required <= names:
        raise FileNotFoundError(
            f"B-Free run directory is missing: {sorted(required - names)}"
        )


def _prepare_output_directories(
    *,
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
    resume: bool,
) -> Path:
    run_dir = _ensure_repo_child(
        run_dir,
        repo_root=repo_root,
        label="B-Free run directory",
    )
    artifact_root = _ensure_repo_child(
        artifact_root,
        repo_root=repo_root,
        label="B-Free artifact root",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("B-Free result and artifact directories must be disjoint")
    artifact_dir = artifact_root / "bfree_artifacts"
    if not resume:
        if run_dir.exists() and (
            not run_dir.is_dir() or any(run_dir.iterdir())
        ):
            raise FileExistsError(
                f"run directory is non-empty; pass --resume: {run_dir}"
            )
        if artifact_root.exists() and (
            not artifact_root.is_dir() or any(artifact_root.iterdir())
        ):
            raise FileExistsError(
                "artifact root is non-empty; pass --resume: "
                f"{artifact_root}"
            )
    else:
        if not run_dir.is_dir() or not artifact_dir.is_dir():
            raise FileNotFoundError(
                "resume requires B-Free run and artifact directories"
            )
        root_entries = list(artifact_root.iterdir())
        if (
            len(root_entries) != 1
            or root_entries[0].name != "bfree_artifacts"
            or root_entries[0].is_symlink()
            or not root_entries[0].is_dir()
        ):
            raise ValueError("resume B-Free artifact-root inventory changed")
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
                "resume is forbidden after analyzer outputs exist; use a new "
                f"run ID: {sorted(finalized_analysis)}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_child(
        artifact_dir,
        repo_root=repo_root,
        label="B-Free artifact directory",
        require_directory=True,
    )
    return artifact_dir


def _validate_artifact_inventory(
    *,
    artifact_dir: Path,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
) -> int:
    artifact_dir = _ensure_repo_child(
        artifact_dir,
        repo_root=repo_root,
        label="B-Free artifact directory",
        require_directory=True,
    )
    entries = list(artifact_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("B-Free artifact inventory contains an unsafe entry")
    expected = {
        f"{sample_id}.npz"
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "B-Free artifact inventory mismatch: "
            f"missing={sorted(expected - actual)[:1]}, "
            f"extra={sorted(actual - expected)[:1]}"
        )
    for sample_id, row in latest_by_sample_id.items():
        if row.get("status") == "ok":
            _validate_artifact(
                row,
                sample_id=sample_id,
                repo_root=repo_root,
                run_id=run_id,
            )
    return len(actual)


def _artifact_record(
    *,
    features: np.ndarray,
    crop_logits: np.ndarray,
    artifact_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(features, np.ndarray)
        or features.shape != (legacy.CROP_COUNT, legacy.FEATURE_DIMENSION)
        or features.dtype != np.float32
        or not features.flags.c_contiguous
        or not np.isfinite(features).all()
        or not isinstance(crop_logits, np.ndarray)
        or crop_logits.shape != (legacy.CROP_COUNT,)
        or crop_logits.dtype != np.float32
        or not crop_logits.flags.c_contiguous
        or not np.isfinite(crop_logits).all()
    ):
        raise ValueError("official B-Free artifact arrays changed")
    payload = artifact_path.read_bytes()
    _validate_npz_structure(payload, sample_id=artifact_path.stem)
    if payload != _npz_bytes(features, crop_logits):
        raise ValueError("persisted B-Free NPZ is not canonical")
    return {
        "relative_path": repo_relative(artifact_path, repo_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "feature_array_sha256": _array_sha256(features),
        "crop_logits_array_sha256": _array_sha256(crop_logits),
        "feature_shape": [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
        "crop_logits_shape": [legacy.CROP_COUNT],
        "dtype": "float32",
        "feature_nbytes": int(features.nbytes),
        "crop_logits_nbytes": int(crop_logits.nbytes),
        "finite": True,
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
    }


def _build_ok_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    asset_bundle_sha256: str,
    artifact_dir: Path,
    scoring: Mapping[str, Any],
    features: np.ndarray,
    crop_logits: np.ndarray,
    preprocess: Mapping[str, Any],
    preprocess_latency_ms: float,
    latency_ms: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    artifact_path = artifact_dir / f"{sample_id}.npz"
    if artifact_path.is_symlink() or artifact_path.exists():
        raise FileExistsError(
            f"B-Free artifact already exists before append: {artifact_path}"
        )
    artifact_written = False
    try:
        legacy._atomic_save_artifact(artifact_path, features, crop_logits)
        artifact_written = True
        record = _artifact_record(
            features=features,
            crop_logits=crop_logits,
            artifact_path=artifact_path,
            repo_root=repo_root,
        )
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
            "bfree_artifact": record,
            "bfree_artifact_path": relative,
            "bfree_artifact_sha256": record["sha256"],
            "feature_array_sha256": record["feature_array_sha256"],
            "crop_logits_array_sha256": record[
                "crop_logits_array_sha256"
            ],
            "feature_shape": [legacy.CROP_COUNT, legacy.FEATURE_DIMENSION],
            "feature_dtype": "float32",
            "feature_nbytes": record["feature_nbytes"],
            "feature_semantics": legacy.FEATURE_SEMANTICS,
            "crop_logits_shape": [legacy.CROP_COUNT],
            "crop_logits_dtype": "float32",
            "crop_logits_nbytes": record["crop_logits_nbytes"],
            "crop_logits_semantics": record["crop_logits_semantics"],
            "artifact_paths": {"bfree_npz": relative},
            **{
                key: scoring[key]
                for key in (
                    "raw_logit",
                    "ai_score",
                    "score",
                    "fake_probability",
                    "crop_logits",
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
        _validate_runner_attempt(
            result,
            input_row=input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        return result
    except BaseException:
        # A JSONL row is the append-only commit point.  No artifact may
        # survive if construction or fail-closed self-validation fails first.
        if artifact_written or artifact_path.is_symlink():
            if artifact_path.is_symlink() or artifact_path.is_file():
                artifact_path.unlink()
            elif artifact_path.exists():
                raise RuntimeError(
                    f"cannot safely clean failed B-Free artifact: {artifact_path}"
                )
        raise


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
        **{key: None for key in sorted(_ERROR_NULL_FIELDS)},
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
                    raise ValueError(f"{row_label} is not an object")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


def _validate_model_load(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "construction",
        "load",
        "network",
        "model_mode",
        "requires_grad",
    }:
        raise ValueError(f"{label} key set changed")
    construction = value.get("construction")
    load = value.get("load")
    network = value.get("network")
    if (
        not isinstance(construction, Mapping)
        or construction.get("architecture")
        != str(legacy.CONFIG["parsed"]["arch"])
        or construction.get("timm_architecture") != legacy.MODEL_ARCH
        or construction.get("pretrained") is not False
        or construction.get("wrapper") != "Wrapper5crops"
        or construction.get("patch_size") != legacy.PATCH_STRIDE
        or construction.get("crop_size") != legacy.CROP_SIZE
        or construction.get("crop_count") != legacy.CROP_COUNT
        or construction.get("feature_dimension") != legacy.FEATURE_DIMENSION
        or construction.get("register_tokens") != 4
        or construction.get("global_pool") != "token"
        or not isinstance(load, Mapping)
        or load.get("strict_full_state_load") is not True
        or not _same_json_type_and_value(load.get("missing_keys"), [])
        or not _same_json_type_and_value(load.get("unexpected_keys"), [])
        or load.get("loaded_tensor_count")
        != legacy.CHECKPOINT["tensor_count"]
        or load.get("loaded_state_elements")
        != legacy.CHECKPOINT["state_elements"]
        or not isinstance(network, Mapping)
        or network.get("allowed") is not False
        or not _same_json_type_and_value(
            network.get("attempts"),
            {
                "urllib_urlopen": 0,
                "socket_create_connection": 0,
                "socket_connect": 0,
            },
        )
        or value.get("model_mode") != "eval"
        or value.get("requires_grad") is not False
    ):
        raise ValueError(f"{label} strict model evidence changed")


def _validate_official_golden(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("status") != "passed"
        or value.get("source") != "official_code_demo_images_results.csv"
        or value.get("score") != legacy.SCORE_SEMANTICS
        or value.get("absolute_tolerance") != legacy.GOLDEN_ABS_TOLERANCE
        or value.get("runtime_regression_absolute_tolerance")
        != legacy.GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE
        or type(value.get("mouse_model_scores_computed")) is not int
        or value["mouse_model_scores_computed"] != 0
        or not isinstance(value.get("cases"), list)
        or len(value["cases"]) != len(legacy.GOLDEN_CASES)
    ):
        raise ValueError(f"{label} envelope changed")
    for observed, frozen in zip(
        value["cases"],
        legacy.GOLDEN_CASES,
        strict=True,
    ):
        if (
            observed.get("filename") != frozen["filename"]
            or observed.get("sha256") != frozen["sha256"]
            or observed.get("label") != frozen["label"]
            or observed.get("published_raw_logit")
            != frozen["published_raw_logit"]
            or observed.get("decoded_rgb_sha256")
            != frozen["decoded_rgb_sha256"]
            or observed.get("tensor_sha256") != frozen["tensor_sha256"]
            or observed.get("feature_array_sha256") is None
            or observed.get("crop_logits_array_sha256") is None
            or observed.get("repeat_bit_identical") is not True
            or observed.get("passed") is not True
            or _finite_number(
                observed.get("absolute_difference_from_published"),
                f"{label} published difference",
            )
            > legacy.GOLDEN_ABS_TOLERANCE
            or _finite_number(
                observed.get("absolute_difference_from_runtime_reference"),
                f"{label} runtime difference",
            )
            > legacy.GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE
        ):
            raise ValueError(f"{label} case changed")


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
        "official_golden",
        "balanced_golden",
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
        "dataset_manifest_loaded",
    }
    if set(report) != expected_keys:
        raise ValueError("B-Free CPU preflight report key set changed")
    if (
        report.get("schema_version") != CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or not _same_json_type_and_value(report.get("source"), source)
        or not _same_json_type_and_value(report.get("assets"), assets)
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("B-Free CPU preflight provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("B-Free CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    _validate_model_load(report.get("model_load"), label="CPU model load")
    _validate_official_golden(
        report.get("official_golden"),
        label="CPU official four-demo golden",
    )
    golden = report.get("balanced_golden")
    expected_scalar = {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "artifact_sha256": CPU_GOLDEN_ARTIFACT_SHA256,
        "artifact_bytes": ARTIFACT_FILE_BYTES,
        "feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "crop_logits_array_sha256": CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256,
        "raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "ai_score": CPU_GOLDEN_RAW_LOGIT,
        "fake_probability": CPU_GOLDEN_FAKE_PROBABILITY,
        "crop_logits": CPU_GOLDEN_CROP_LOGITS,
        "classification_decision": False,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
        "peak_cuda_memory_bytes": 0,
        "repeat_artifact_sha256": CPU_GOLDEN_ARTIFACT_SHA256,
        "repeat_feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "repeat_crop_logits_array_sha256": (
            CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256
        ),
        "repeat_raw_logit": CPU_GOLDEN_RAW_LOGIT,
        "repeat_ai_score": CPU_GOLDEN_RAW_LOGIT,
        "repeat_fake_probability": CPU_GOLDEN_FAKE_PROBABILITY,
        "repeat_crop_logits": CPU_GOLDEN_CROP_LOGITS,
        "repeat_classification_decision": False,
        "repeat_model_forward_calls": 1,
        "repeat_classifier_hook_calls": 1,
        "repeat_byte_exact": True,
    }
    if (
        not isinstance(golden, Mapping)
        or set(golden) != set(expected_scalar) | {"preprocess"}
        or any(
            not _same_json_type_and_value(golden.get(key), value)
            for key, value in expected_scalar.items()
        )
    ):
        raise ValueError("B-Free Balanced CPU golden scalar evidence changed")
    preprocess = golden.get("preprocess")
    if (
        not isinstance(preprocess, Mapping)
        or preprocess.get("decoded_rgb_sha256")
        != CPU_GOLDEN_DECODED_RGB_SHA256
        or preprocess.get("tensor_sha256") != CPU_GOLDEN_TENSOR_SHA256
        or preprocess.get("tensor_shape") != [3, 1350, 1800]
        or preprocess.get("profile") != FROZEN_PROFILE
    ):
        raise ValueError("B-Free Balanced CPU golden preprocess changed")


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
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
    )
    parser.add_argument(
        "--weights-zip",
        type=Path,
        default=DEFAULT_WEIGHTS_ZIP,
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
    parser.add_argument(
        "--mode",
        choices=("preflight", "formal", "smoke", "single"),
        default="formal",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--sample-id")
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute one append-only B-Free Balanced250 run."""

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(Path(args.source_root), repo_root)
    weights_dir = _anchored(Path(args.weights_dir), repo_root)
    weights_zip = _anchored(Path(args.weights_zip), repo_root)
    mode = str(args.mode)
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed != DEFAULT_SEED
    ):
        raise ValueError(f"B-Free seed must be exactly {DEFAULT_SEED}")
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
                "preflight accepts no run/selection/resume/CUDA options"
            )
        report = run_cpu_preflight(
            repo_root=repo_root,
            source_root=source_root,
            weights_dir=weights_dir,
            weights_zip=weights_zip,
        )
        source, assets, state = verify_assets(
            source_root=source_root,
            weights_dir=weights_dir,
            weights_zip=weights_zip,
        )
        del state
        _validate_preflight_report(report, source=source, assets=assets)
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
        label="B-Free results root",
    )
    artifacts_root = _ensure_repo_child(
        Path(args.artifacts_dir),
        repo_root=repo_root,
        label="B-Free artifacts root",
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
    artifact_dir = artifact_root / "bfree_artifacts"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    # P0 ordering invariant: this full CPU gate runs before both the release
    # manifest is opened and any accelerator runtime is configured.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    source, assets, state = verify_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    _validate_preflight_report(
        cpu_preflight,
        source=source,
        assets=assets,
    )

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
    model, execution_model_load = legacy.load_model(
        state=state,
        source_root=source_root,
        device=device,
    )
    _validate_model_load(
        execution_model_load,
        label="execution model load",
    )
    _require_same_json(
        execution_model_load,
        cpu_preflight["model_load"],
        "execution vs CPU model load",
    )
    execution_official_golden = legacy.validate_official_golden(
        model=model,
        device=device,
        source_root=source_root,
    )
    _validate_official_golden(
        execution_official_golden,
        label="execution official four-demo golden",
    )
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
        execution_model_load=execution_model_load,
        execution_official_golden=execution_official_golden,
        run_dir=run_dir,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_dir=artifact_dir,
    )
    fingerprint = _fingerprint(immutable)
    artifact_dir = _prepare_output_directories(
        repo_root=repo_root,
        run_dir=run_dir,
        artifact_root=artifact_root,
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
        ):
            raise ValueError("resume manifest envelope drifted")
        _require_same_json(
            prior_manifest.get("immutable"),
            immutable,
            "resume immutable manifest",
        )
        expected_rows = _read_jsonl_strict(
            expected_path,
            "expected inputs",
        )
        _require_same_json(
            expected_rows,
            selected,
            "resume expected-input snapshot",
        )
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
        _require_same_json(
            prior_manifest.get("dataset"),
            expected_dataset,
            "resume dataset evidence",
        )
        outputs_value = prior_manifest.get("outputs")
        if not isinstance(outputs_value, Mapping):
            raise ValueError("resume manifest outputs are invalid")
        prior_outputs = outputs_value
        if prior_status == "running":
            _require_same_json(
                dict(prior_outputs),
                immutable["outputs"],
                "running resume output contract",
            )
        else:
            expected_output_keys = set(immutable["outputs"]) | {
                "results_sha256",
                "summary_sha256",
                "artifact_files",
            }
            if (
                set(prior_outputs) != expected_output_keys
                or not results_path.is_file()
                or prior_outputs.get("results_sha256")
                != sha256_file(results_path)
                or not summary_path.is_file()
                or prior_outputs.get("summary_sha256")
                != sha256_file(summary_path)
                or isinstance(prior_outputs.get("artifact_files"), bool)
                or not isinstance(prior_outputs.get("artifact_files"), int)
                or prior_outputs["artifact_files"] < 0
            ):
                raise ValueError("finalized resume output evidence drifted")
            for key, expected_value in immutable["outputs"].items():
                _require_same_json(
                    prior_outputs.get(key),
                    expected_value,
                    f"finalized resume output {key}",
                )
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
            "expected_inputs_path": repo_relative(
                expected_path,
                repo_root,
            ),
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
            raise ValueError("prior B-Free result is outside selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    prior_artifact_files = _validate_artifact_inventory(
        artifact_dir=artifact_dir,
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    if (
        args.resume
        and prior_status in ("complete", "incomplete")
        and prior_outputs.get("artifact_files") != prior_artifact_files
    ):
        raise ValueError("finalized resume artifact count drifted")
    _validate_latest_artifact_head_replay(
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        state=state,
        device=device,
    )
    # Old history/artifacts must be fully replayed before resume mutates state.
    atomic_write_json(manifest_path, manifest)

    pending_ids = set(latest_before.pending_sample_ids(retry_errors=True))
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
            artifact_path = artifact_dir / f"{sample_id}.npz"
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
                if preprocess.get("geometry", {}).get("native_size") != [
                    int(input_row["width"]),
                    int(input_row["height"]),
                ]:
                    raise ValueError("B-Free preprocessed dimensions changed")
                scoring, features, crop_logits, peak, latency = legacy.infer_one(
                    model,
                    device,
                    image,
                )
                result = _build_ok_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    asset_bundle_sha256=EXPECTED_ASSET_BUNDLE_SHA256,
                    artifact_dir=artifact_dir,
                    scoring=scoring,
                    features=features,
                    crop_logits=crop_logits,
                    preprocess=preprocess,
                    preprocess_latency_ms=preprocess_latency_ms,
                    latency_ms=latency,
                    peak_cuda_memory_bytes=0 if peak is None else int(peak),
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"raw_logit={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                if artifact_path.exists():
                    if artifact_path.is_symlink() or not artifact_path.is_file():
                        raise ValueError(
                            f"unsafe failed artifact path: {artifact_path}"
                        ) from error
                    artifact_path.unlink()
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
        del model
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()

    physical_results = _read_jsonl_strict(results_path, "physical results")
    _validate_physical_attempt_history(physical_results)
    for attempt in physical_results:
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("B-Free result is outside selection")
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
        artifact_dir=artifact_dir,
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    replayed_artifacts = _validate_latest_artifact_head_replay(
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
        state=state,
        device=device,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_bfree_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": FROZEN_PROFILE,
        "score_spec": SCORE_SPEC.as_dict(),
        "dataset_contract": dataset_contract.as_dict(),
        "selection_visibility_census": visibility,
        "same_device_artifact_head_replays": replayed_artifacts,
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
        "same_device_artifact_head_replays": replayed_artifacts,
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
        raise RuntimeError("B-Free fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
