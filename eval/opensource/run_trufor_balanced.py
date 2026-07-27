#!/usr/bin/env python3
"""Run the frozen official TruFor checkpoint on Balanced250.

The Mouse-v1 adapter remains unchanged.  This v2 orchestration layer binds the
official phase-3 checkpoint to the independent 1,775-image Balanced250 release,
records TruFor's complete released outputs, and applies the benchmark's native
T1+T2 contract:

* every condition receives the released image-level detection score;
* the native forged-probability and TCP reliability maps are retained for
  every successful input;
* only authentic and local-insertion inputs receive a threshold mask and T2
  metrics; and
* full-frame edits are explicitly T2-not-applicable.

All dataset, environment, source, license, archive, checkpoint, and strict
CPU-model-load checks run before accelerator configuration.  The CPU gate
performs no model forward and computes no Balanced250 score.
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

from eval.opensource import run_trufor as legacy
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
from eval.opensource.maskclip_metrics import binary_pixel_metrics


RUN_MANIFEST_SCHEMA = "trufor_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "trufor_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "trufor_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "trufor_balanced_cpu_preflight_v2"

MODEL_NAME = "TruFor"
MODEL_SLUG = "trufor_cvpr2023"
MODEL_ARCHITECTURE = (
    "RGB_NoiseprintPP_CMX_SegFormerB2_localization_confidence_detection"
)
CHECKPOINT_ID = "trufor_ph3_epoch81"
PREPROCESS_PROFILE = "official_trufor_native_rgb_float32_divide_256"

MODEL_SEED = 42
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">="
MASK_THRESHOLD = 0.5
MASK_THRESHOLD_OPERATOR = ">="

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/balanced250_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/trufor")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/trufor")
DEFAULT_FORMAL_RUN_ID = (
    "trufor_cvpr2023_balanced250_v1_full1775_20260727"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "trufor_cvpr2023_balanced250_v1_smoke5x7_a_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "trufor_cvpr2023_balanced250_v1_smoke5x7_b_20260727"
)
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_ARCHIVE = Path(
    "/root/.cache/claimforge/checkpoints/trufor/TruFor_weights.zip"
)
EXPECTED_VENV_ROOT = Path(
    "/root/.cache/claimforge/venvs/trufor-ae54475"
)
EXPECTED_PYTHON_EXECUTABLE = EXPECTED_VENV_ROOT / "bin/python"
EXPECTED_PYVENV_SHA256 = (
    "031b3a83d9970ae0167b24ae8ece3e31007082e16db51b061be6df9df2276d6c"
)

CHECKPOINT_BYTES = 281_496_429
CHECKPOINT_OUTER_KEYS = (
    "epoch",
    "best_value",
    "best_key",
    "state_dict",
    "optimizer",
)
CHECKPOINT_STATE_KEYS = 952
CHECKPOINT_STATE_ELEMENTS = 68_705_510
CHECKPOINT_FLOAT32_TENSORS = 927
CHECKPOINT_INT64_TENSORS = 25
CHECKPOINT_ORDERED_KEYS_SHA256 = (
    "7bf9cb83a052ca0b70c3c7957111a732e83e4592e4ca563bcf21605a065e6d84"
)
CHECKPOINT_TENSOR_SCHEMA_SHA256 = (
    "8a9ebd68344360a8117337c0d2e65b7b8ef82a0040503b1a7ac3cdf7a784f9cd"
)
CHECKPOINT_UNSAFE_GLOBALS = (
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
)
ARCHIVE_BYTES = 260_878_690

EXPECTED_MODEL_PARAMETERS = 68_697_421
EXPECTED_TRAINABLE_PARAMETERS = 1_578_242
EXPECTED_MODEL_BUFFERS = 8_089
EXPECTED_MODEL_MODULES = 872

MIN_DISK_RESERVE_BYTES = 2_000_000_000
NPY_HEADER_BYTES = 128

EXPECTED_PACKAGES = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "timm": "0.5.4",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "scikit-learn": "1.5.2",
    "scipy": "1.16.0",
    "yacs": "0.1.8",
    "PyYAML": "6.0.2",
    "matplotlib": "3.10.0",
    "setuptools": "79.0.1",
}

TRUFOR_SOURCE_FILES: dict[str, tuple[int, str]] = {
    "__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "lib/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "lib/config/__init__.py": (
        69,
        "81a4b734c2f951313dc863aa859e7cbce05460fe0f26ebcab2d175ebe3e4e65e",
    ),
    "lib/config/default.py": (
        3_317,
        "885d9c09fea7686ca347c05cc3a7cdf85d45e7494f058068fd9b7e3af6f3ff75",
    ),
    "lib/utils.py": (
        10_375,
        "6f2dcecb50d457f3116e30878c969d9f68e51f028f61cfe92bd606a121796b8e",
    ),
    "lib/models/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "lib/models/DnCNN.py": (
        5_442,
        "3736b7c625d03b85accea77c26a41ea2da956dae07d70182532e7df9b74ae882",
    ),
    "lib/models/cmx/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "lib/models/cmx/builder_np_conf.py": (
        10_608,
        "b31b02a96417b424228f0301ac653a44d257e522c3ddede414d3c95209d80ded",
    ),
    "lib/models/cmx/init_func.py": (
        2_273,
        "20dad47877caaf5ee485d49b6618c5c3de68261ef0090c586eb409e605f976a4",
    ),
    "lib/models/cmx/net_utils.py": (
        8_089,
        "171239507b49f00026ca46db88cbb101f8c24f47dd7a93b1b5a7f919deb8ed21",
    ),
    "lib/models/cmx/layer_utils.py": (
        1_388,
        "4d75874f1c793b6db0893f734853cd074d8efa1cafc3c33519360c8357005a2e",
    ),
    "lib/models/cmx/encoders/dual_segformer.py": (
        24_910,
        "faf1d8483a559d25d151c58fbf171e52b504c88ba4c05235d15dc96467a7ac60",
    ),
    "lib/models/cmx/decoders/MLPDecoder.py": (
        3_060,
        "172efeeb155591f9857f888263c513b9f8d03bb41c4ebf64a74eb15e8b250a91",
    ),
}

SOURCE_BOUND_ASSETS: dict[str, tuple[int, str]] = {
    "lib/config/trufor_ph3.yaml": (
        1_124,
        legacy.MODEL_CONFIG_SHA256,
    ),
    "LICENSE.txt": (
        1_837,
        legacy.MODEL_LICENSE_SHA256,
    ),
    "LICENSE_CMX.txt": (
        1_086,
        "687ff4d0ea13200541df7359799a08c2db626094ecacbb8a3fe63ffad177f2a1",
    ),
}

ADAPTER_SOURCE_PATHS = (
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_trufor_balanced.py",
    "eval/opensource/run_trufor.py",
    "eval/opensource/trufor_metrics.py",
    "eval/opensource/maskclip_metrics.py",
    "eval/opensource/balanced250_localization_metrics.py",
    "eval/opensource/balanced250_metrics.py",
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
    "native_probability_map": {
        "source": "softmax_localization_logits_channel_1",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "saved_for": "all_successful_inputs",
        "fullframe_role": "diagnostic_only",
    },
    "native_reliability_map": {
        "source": "sigmoid_TCP_confidence_logit",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "saved_for": "all_successful_inputs",
        "used_for_primary_metrics": False,
        "must_not_be_multiplied_into_probability_map": True,
    },
    "native_binary_mask": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "encoding": "PNG_L_0_or_255",
        "saved_for": "t2_applicable_inputs_only",
    },
    "ground_truth": {
        "real": "all_zero_false_positive_area_only",
        "local": "exact_diff",
        "fullframe": "not_applicable",
        "fullframe_conditioning_box_is_not_ground_truth": True,
    },
}

TASK_SCOPE: dict[str, Any] = {
    "primary_task": "T1_whole_image_AIGC_detection_and_T2_localization",
    "valid_for_t1": True,
    "valid_for_t2": True,
    "fullframe_t2_not_applicable": True,
    "native_dense_output": True,
}

LICENSE_RECORD: dict[str, Any] = {
    "overall": {
        "identifier": "TruFor_custom_informational_nonprofit_only",
        "path": "LICENSE.txt",
        "sha256": legacy.MODEL_LICENSE_SHA256,
        "informational_and_nonprofit_only": True,
        "unauthorized_industrial_or_profit_use_prohibited": True,
        "commercial_use_requires_separate_authorization": True,
        "notice_and_attribution_required": True,
    },
    "cmx_component": {
        "identifier": "MIT",
        "path": "LICENSE_CMX.txt",
        "sha256": SOURCE_BOUND_ASSETS["LICENSE_CMX.txt"][1],
        "does_not_override_overall_trufor_restriction": True,
    },
}

ARTIFACT_CONTRACT: dict[str, Any] = {
    "score_map_native": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "softmax_localization_logits_channel_1_forged_probability",
        "inventory": "one_per_successful_input",
    },
    "reliability_map_native": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "sigmoid_TCP_localization_reliability_not_anomaly",
        "inventory": "one_per_successful_input",
        "used_for_primary_metrics": False,
    },
    "mask_native": {
        "format": "PNG",
        "mode": "L",
        "values": [0, 255],
        "shape": "native_height_by_native_width",
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "inventory": "one_per_successful_t2_applicable_input",
    },
}

_OK_ONLY_KEYS = frozenset(
    {
        "preprocess",
        "raw_detection_logit",
        "raw_outputs",
        "class_probabilities",
        "ai_score",
        "probability",
        "score",
        "score_margin",
        "score_semantics",
        "calibrated_probability",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "score_map_native_path",
        "score_map_native_sha256",
        "score_map_native_bytes",
        "score_map_native_array_sha256",
        "score_map_native_shape",
        "score_map_native_dtype",
        "score_map_native_semantics",
        "reliability_map_native_path",
        "reliability_map_native_sha256",
        "reliability_map_native_bytes",
        "reliability_map_native_array_sha256",
        "reliability_map_native_shape",
        "reliability_map_native_dtype",
        "reliability_map_native_semantics",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_array_sha256",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
        "mask_threshold",
        "mask_threshold_operator",
        "artifact_paths",
        "localization",
        "reliability",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _unresolved_anchored(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject an existing symlink at any component before resolving a path."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_child(root: Path, name: str, label: str) -> Path:
    """Resolve one non-symlink child and prove it remains below its root."""

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
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_run_id(value: Any) -> str:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_.-"
    )
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(character not in allowed for character in value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(
            "run-id must be one safe ASCII path component (max 160 chars)"
        )
    return value


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
                    raise ValueError(
                        f"{path}:{line_number} lacks terminating newline"
                    )
                if not line.strip():
                    raise ValueError(f"{path}:{line_number} is blank")
                value = json.loads(
                    line,
                    object_pairs_hook=_without_duplicate_keys,
                    parse_constant=_reject_nonfinite_json,
                )
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                if line != f"{stable_json(value)}\n":
                    raise ValueError(
                        f"{path}:{line_number} is not canonical JSONL"
                    )
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
    """Hash all local sources bound into the runner/audit contract."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        candidate = repo_root / relative
        _reject_symlink_components(
            candidate,
            f"TruFor adapter source {relative}",
        )
        path = candidate.resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing/unsafe TruFor adapter source: {path}"
            )
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_environment() -> dict[str, Any]:
    """Fail unless the audited TruFor virtual environment is active."""

    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    if executable != EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError(
            "TruFor must run with the pinned interpreter "
            f"{EXPECTED_PYTHON_EXECUTABLE}, got {executable}"
        )
    if prefix != EXPECTED_VENV_ROOT:
        raise ValueError(
            f"TruFor venv prefix changed: {prefix} != {EXPECTED_VENV_ROOT}"
        )
    if platform.python_version() != "3.12.3":
        raise ValueError("TruFor Python version changed")
    pyvenv_path = prefix / "pyvenv.cfg"
    if (
        not pyvenv_path.is_file()
        or pyvenv_path.is_symlink()
        or sha256_file(pyvenv_path) != EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("TruFor pyvenv.cfg changed")
    versions = {
        name: _package_version(name)
        for name in EXPECTED_PACKAGES
    }
    if versions != EXPECTED_PACKAGES:
        changed = {
            name: {
                "expected": EXPECTED_PACKAGES[name],
                "actual": versions.get(name),
            }
            for name in EXPECTED_PACKAGES
            if versions.get(name) != EXPECTED_PACKAGES[name]
        }
        raise ValueError(f"TruFor package environment changed: {changed}")
    return {
        "python_executable": str(executable),
        "python_prefix": str(prefix),
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv_path),
            "bytes": pyvenv_path.stat().st_size,
            "sha256": EXPECTED_PYVENV_SHA256,
            "include_system_site_packages": True,
        },
        "packages": versions,
    }


def verify_source(trufor_root: Path) -> dict[str, Any]:
    """Verify the exact clean official source tree and forward-path files."""

    _reject_symlink_components(trufor_root, "TruFor source root")
    root = trufor_root.resolve()
    if (
        root.name != "TruFor_train_test"
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise FileNotFoundError(f"missing/unsafe TruFor source root: {root}")
    repository = root.parent
    commit = legacy._git_value(repository, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"TruFor source commit changed: {commit} != "
            f"{legacy.MODEL_SOURCE_COMMIT}"
        )
    status = legacy._git_value(
        repository,
        "status",
        "--short",
        "--untracked-files=all",
    )
    if status is None:
        raise ValueError("cannot inspect TruFor source worktree")
    if status:
        raise ValueError("TruFor source worktree is dirty")

    bindings: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha256) in {
        **TRUFOR_SOURCE_FILES,
        **SOURCE_BOUND_ASSETS,
    }.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing/unsafe TruFor source-bound file: {path}"
            )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"TruFor source file size changed: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"TruFor source file hash changed: {relative}")
        tracked = legacy._git_value(
            repository,
            "ls-files",
            "--error-unmatch",
            f"TruFor_train_test/{relative}",
        )
        if tracked is None:
            raise ValueError(f"TruFor source file is not tracked: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "git_tracked": True,
        }
    return {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "git_root": str(repository),
        "commit": commit,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bindings,
    }


def _asset(
    path: Path,
    *,
    label: str,
    expected_name: str,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    _reject_symlink_components(path, label)
    resolved = path.resolve()
    if resolved.name != expected_name:
        raise ValueError(f"{label} filename changed")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {resolved}")
    if resolved.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size changed")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed")
    return {
        "path": str(resolved),
        "filename": expected_name,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
    }


def verify_assets(
    *,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    """Verify the official archive, extracted checkpoint, config, and licenses."""

    archive = _asset(
        archive_path,
        label="TruFor official weight archive",
        expected_name="TruFor_weights.zip",
        expected_bytes=ARCHIVE_BYTES,
        expected_sha256=legacy.CHECKPOINT_ZIP_SHA256,
    )
    if _md5_file(archive_path.resolve()) != legacy.CHECKPOINT_ZIP_MD5:
        raise ValueError("TruFor official weight archive MD5 changed")
    archive.update(
        {
            "url": legacy.CHECKPOINT_URL,
            "published_md5": legacy.CHECKPOINT_ZIP_MD5,
            "members": [
                "weights/",
                "weights/trufor.pth.tar",
            ],
            "inner_checkpoint_uncompressed_bytes": CHECKPOINT_BYTES,
        }
    )
    checkpoint = _asset(
        checkpoint_path,
        label="TruFor phase-3 checkpoint",
        expected_name="trufor.pth.tar",
        expected_bytes=CHECKPOINT_BYTES,
        expected_sha256=legacy.CHECKPOINT_SHA256,
    )
    checkpoint.update(
        {
            "id": CHECKPOINT_ID,
            "epoch": legacy.CHECKPOINT_EPOCH,
            "weights_only": True,
            "strict_model_load": True,
        }
    )
    config_path = trufor_root.resolve() / "lib/config/trufor_ph3.yaml"
    license_path = trufor_root.resolve() / "LICENSE.txt"
    cmx_license_path = trufor_root.resolve() / "LICENSE_CMX.txt"
    return {
        "archive": archive,
        "checkpoint": checkpoint,
        "configuration": {
            "path": str(config_path),
            "bytes": SOURCE_BOUND_ASSETS[
                "lib/config/trufor_ph3.yaml"
            ][0],
            "sha256": legacy.MODEL_CONFIG_SHA256,
            "constructor_pretrained": None,
            "noiseprint_weights": None,
            "constructor_external_weight_files_used": False,
        },
        "licenses": {
            "overall": {
                **LICENSE_RECORD["overall"],
                "absolute_path": str(license_path),
                "bytes": SOURCE_BOUND_ASSETS["LICENSE.txt"][0],
            },
            "cmx_component": {
                **LICENSE_RECORD["cmx_component"],
                "absolute_path": str(cmx_license_path),
                "bytes": SOURCE_BOUND_ASSETS["LICENSE_CMX.txt"][0],
            },
        },
    }


def _checkpoint_tensor_schema(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in state.items()
    ]


def _load_checkpoint_state(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Safely load and audit the pinned CPU checkpoint."""

    import torch

    unsafe = tuple(
        sorted(
            torch.serialization.get_unsafe_globals_in_checkpoint(
                checkpoint_path
            )
        )
    )
    if unsafe != tuple(sorted(CHECKPOINT_UNSAFE_GLOBALS)):
        raise ValueError(f"TruFor checkpoint unsafe globals changed: {unsafe}")
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
    if type(checkpoint) is not dict:
        raise ValueError("TruFor checkpoint outer object changed")
    if tuple(checkpoint) != CHECKPOINT_OUTER_KEYS:
        raise ValueError("TruFor checkpoint outer keys/order changed")
    if (
        checkpoint.get("epoch") != legacy.CHECKPOINT_EPOCH
        or checkpoint.get("best_key") != "loss"
        or checkpoint.get("best_value") != 0.14553078470670194
    ):
        raise ValueError("TruFor checkpoint training metadata changed")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "state",
        "param_groups",
    }:
        raise ValueError("TruFor checkpoint optimizer schema changed")
    state = checkpoint.get("state_dict")
    if (
        state is None
        or type(state).__name__ != "OrderedDict"
        or len(state) != CHECKPOINT_STATE_KEYS
    ):
        raise ValueError("TruFor checkpoint state_dict schema changed")
    if any(not isinstance(name, str) for name in state):
        raise ValueError("TruFor checkpoint contains a non-string tensor key")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("TruFor checkpoint state_dict contains non-tensors")

    ordered_keys_sha256 = hashlib.sha256(
        "\n".join(state).encode("utf-8")
    ).hexdigest()
    tensor_schema = _checkpoint_tensor_schema(state)
    tensor_schema_sha256 = _fingerprint({"tensors": tensor_schema}["tensors"])
    dtype_counts = Counter(str(tensor.dtype) for tensor in state.values())
    elements = sum(int(tensor.numel()) for tensor in state.values())
    if ordered_keys_sha256 != CHECKPOINT_ORDERED_KEYS_SHA256:
        raise ValueError("TruFor checkpoint ordered key hash changed")
    if tensor_schema_sha256 != CHECKPOINT_TENSOR_SCHEMA_SHA256:
        raise ValueError("TruFor checkpoint tensor schema hash changed")
    if dtype_counts != {
        "torch.float32": CHECKPOINT_FLOAT32_TENSORS,
        "torch.int64": CHECKPOINT_INT64_TENSORS,
    }:
        raise ValueError("TruFor checkpoint dtype inventory changed")
    if elements != CHECKPOINT_STATE_ELEMENTS:
        raise ValueError("TruFor checkpoint tensor element count changed")
    for name, tensor in state.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"TruFor checkpoint tensor contains non-finite values: {name}"
            )
    audit = {
        "outer_type": "builtins.dict",
        "outer_keys": list(CHECKPOINT_OUTER_KEYS),
        "epoch": legacy.CHECKPOINT_EPOCH,
        "best_key": "loss",
        "best_value": 0.14553078470670194,
        "state_dict_type": "collections.OrderedDict",
        "state_dict_tensors": CHECKPOINT_STATE_KEYS,
        "state_dict_elements": CHECKPOINT_STATE_ELEMENTS,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": ordered_keys_sha256,
        "tensor_schema_sha256": tensor_schema_sha256,
        "all_floating_tensors_finite": True,
        "static_unsafe_globals": list(unsafe),
        "safe_globals_allowlist": [
            "numpy.core.multiarray.scalar",
            "numpy.dtype",
            "numpy.dtype[float64]_class",
        ],
        "weights_only": True,
        "map_location": "cpu",
    }
    del checkpoint, optimizer
    return audit, state


def _build_cpu_model_audit(
    *,
    trufor_root: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct and strictly load the official model on CPU, with no forward."""

    import torch

    checkpoint_audit, state = _load_checkpoint_state(checkpoint_path)
    torch.manual_seed(MODEL_SEED)
    if str(trufor_root) not in sys.path:
        sys.path.insert(0, str(trufor_root))
    with legacy._working_directory(trufor_root):
        from lib.config import config as base_config
        from lib.utils import get_model

        config = base_config.clone()
        config.defrost()
        config.merge_from_file(
            str(trufor_root / "lib/config/trufor_ph3.yaml")
        )
        if (
            config.MODEL.NAME != "detconfcmx"
            or config.MODEL.PRETRAINED is not None
            or config.MODEL.EXTRA.NP_WEIGHTS is not None
            or config.MODEL.EXTRA.BACKBONE != "mit_b2"
            or config.MODEL.EXTRA.DECODER != "MLPDecoder"
            or config.MODEL.EXTRA.DETECTION != "confpool"
            or bool(config.CUDNN.ENABLED)
            or bool(config.CUDNN.BENCHMARK)
            or bool(config.CUDNN.DETERMINISTIC)
        ):
            raise ValueError("TruFor phase-3 model/config contract changed")
        config.freeze()
        model = get_model(config)
        incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("TruFor strict CPU state load was incompatible")
    if list(model.state_dict()) != list(state):
        raise ValueError("TruFor model/checkpoint state ordering changed")
    model.eval()
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    trainable_count = sum(
        int(value.numel())
        for value in model.parameters()
        if value.requires_grad
    )
    buffer_count = sum(int(value.numel()) for value in model.buffers())
    module_count = sum(1 for _ in model.modules())
    if (
        parameter_count != EXPECTED_MODEL_PARAMETERS
        or trainable_count != EXPECTED_TRAINABLE_PARAMETERS
        or buffer_count != EXPECTED_MODEL_BUFFERS
        or module_count != EXPECTED_MODEL_MODULES
    ):
        raise ValueError("TruFor constructed model inventory changed")
    model_audit = {
        "construction_device": "cpu",
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "eval_mode": model.training is False,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "buffers": buffer_count,
        "modules": module_count,
        "constructor_external_weight_files_used": False,
        "model_forwards": 0,
    }
    del incompatible, model, state
    gc.collect()
    return checkpoint_audit, model_audit


def run_cpu_preflight(
    *,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    """Run all CPU-only provenance and strict-load gates."""

    import torch

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise RuntimeError(
            "TruFor CPU preflight started after CUDA initialization"
        )
    environment = verify_environment()
    source = verify_source(trufor_root)
    assets = verify_assets(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    checkpoint_audit, model_audit = _build_cpu_model_audit(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
    )
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after:
        raise RuntimeError("TruFor CPU preflight initialized CUDA")
    return {
        "schema_version": CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "environment": environment,
        "source": source,
        "assets": assets,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "license": LICENSE_RECORD,
        "accelerator_model_forwards": 0,
        "balanced250_model_scores_computed": 0,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
    }


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    """Freeze the historically audited official FP32 numerical path."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "TruFor accelerator was initialized before runtime configuration"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise ValueError(
            "historical TruFor runtime requires CUBLAS_WORKSPACE_CONFIG unset"
        )
    if device_text == "cpu":
        device = torch.device("cpu")
    elif (
        device_text.startswith("cuda:")
        and device_text[5:].isdigit()
        and str(int(device_text[5:])) == device_text[5:]
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_text)
        torch.cuda.set_device(device)
    else:
        raise ValueError("device must be 'cpu' or an explicit canonical 'cuda:N'")

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    runtime: dict[str, Any] = {
        "device": str(device),
        "seed": MODEL_SEED,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": None,
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
            "source": "official_trufor_ph3_config",
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_initialized_before_configuration": False,
        "cuda_initialized_after_configuration": bool(
            torch.cuda.is_initialized()
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    return device, runtime


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.LOCAL_T1_T2)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    t2_images = sum(_t2_semantics(row)[0] for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1_775
        or t2_images != FORMAL_T2_IMAGES
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
    ):
        raise ValueError("formal TruFor Balanced250 selection drifted")
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
    inputs_by_id = {
        str(row["sample_id"]): row
        for row in release.inputs
    }
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if (
            condition in BALANCED_CONDITIONS
            and counts[condition] < per_condition_limit
        ):
            sample_id = str(panel_row["sample_id"])
            source = inputs_by_id.get(sample_id)
            if source is None or source.get("panel") is not True:
                raise ValueError(
                    "TruFor smoke panel contains dangling/non-panel input"
                )
            selected.append(source)
            counts[condition] += 1
    expected = {
        condition: per_condition_limit
        for condition in BALANCED_CONDITIONS
    }
    if dict(counts) != expected:
        raise ValueError("TruFor smoke panel does not cover every condition")
    selected.sort(key=lambda row: int(row["rank"]))
    if len(selected) != 35:
        raise ValueError("TruFor standard smoke must select exactly 35 inputs")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    """Select exact formal, fixed 5x7 smoke, or one debugging input."""

    if release.release_kind != "balanced250":
        raise ValueError("TruFor v2 requires a Balanced250 release")
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
        "score_map_output_role": (
            "t2_and_diagnostic" if applicable else "diagnostic_only"
        ),
        "reliability_map_output_role": "diagnostic_only",
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
    with Image.open(input_path) as opened:
        rgb = np.ascontiguousarray(
            np.asarray(opened.convert("RGB"), dtype=np.uint8)
        )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("TruFor RGB decode produced an invalid array")
    height, width = rgb.shape[:2]
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1),
        dtype=np.float32,
    )
    tensor /= np.float32(256.0)
    if (
        tensor.shape != (3, height, width)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 255.0 / 256.0
    ):
        raise ValueError("TruFor preprocessing produced an invalid tensor")
    audit = {
        "profile": PREPROCESS_PROFILE,
        "decode": "PIL_convert_RGB",
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(rgb.shape),
        "decoded_rgb_dtype": "uint8",
        "decoded_rgb_array_sha256": _array_sha256(rgb),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_array_sha256": _array_sha256(tensor),
        "input_scale": "float32_divide_by_256",
        "input_scale_divisor": 256.0,
        "input_resize": None,
        "input_crop": None,
        "network_map_upsample": (
            "bilinear_align_corners_false_to_native_input_size"
        ),
        "post_network_map_restore": None,
    }
    return tensor, (width, height), audit


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _score_payload(score: float, detection_logit: float) -> dict[str, Any]:
    score_value = float(score)
    logit_value = float(detection_logit)
    if (
        not math.isfinite(score_value)
        or not math.isfinite(logit_value)
        or not 0.0 <= score_value <= 1.0
        or not math.isclose(
            score_value,
            _stable_sigmoid(logit_value),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise ValueError("TruFor image detection score/logit is invalid")
    return {
        "raw_detection_logit": logit_value,
        "raw_outputs": {
            "binary_forged_logit": logit_value,
        },
        "class_probabilities": {
            "real": 1.0 - score_value,
            "forged": score_value,
        },
        "ai_score": score_value,
        "probability": score_value,
        "score": score_value,
        "score_margin": logit_value,
        "score_semantics": "sigmoid_binary_logit_probability_of_forged",
        "calibrated_probability": False,
        "classification_decision": (
            score_value >= CLASSIFICATION_THRESHOLD
        ),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
    }


def _reliability_payload(reliability: np.ndarray) -> dict[str, Any]:
    return {
        "semantics": "TCP_localization_reliability_not_forged_probability",
        "used_for_primary_metrics": False,
        "multiplied_into_score_map": False,
        "min": float(np.min(reliability)),
        "mean": float(np.mean(reliability)),
        "median": float(np.median(reliability)),
        "p05": float(np.quantile(reliability, 0.05)),
        "p95": float(np.quantile(reliability, 0.95)),
        "max": float(np.max(reliability)),
    }


def artifact_paths(run_dir: Path, sample_id: str) -> dict[str, Path]:
    return {
        "score": run_dir / "score_maps_native" / f"{sample_id}.npy",
        "reliability": (
            run_dir / "reliability_maps_native" / f"{sample_id}.npy"
        ),
        "mask": run_dir / "masks_native" / f"{sample_id}.png",
    }


def _artifact_fields(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    score_map: np.ndarray,
    reliability_map: np.ndarray,
    mask_path: Path | None,
) -> dict[str, Any]:
    score_relative = repo_relative(paths["score"], repo_root)
    reliability_relative = repo_relative(paths["reliability"], repo_root)
    fields: dict[str, Any] = {
        "score_map_native_path": score_relative,
        "score_map_native_sha256": sha256_file(paths["score"]),
        "score_map_native_bytes": paths["score"].stat().st_size,
        "score_map_native_array_sha256": _array_sha256(score_map),
        "score_map_native_shape": list(score_map.shape),
        "score_map_native_dtype": str(score_map.dtype),
        "score_map_native_semantics": (
            "softmax_localization_logits_channel_1_forged_probability"
        ),
        "reliability_map_native_path": reliability_relative,
        "reliability_map_native_sha256": sha256_file(paths["reliability"]),
        "reliability_map_native_bytes": paths["reliability"].stat().st_size,
        "reliability_map_native_array_sha256": _array_sha256(
            reliability_map
        ),
        "reliability_map_native_shape": list(reliability_map.shape),
        "reliability_map_native_dtype": str(reliability_map.dtype),
        "reliability_map_native_semantics": (
            "sigmoid_TCP_localization_reliability_not_anomaly"
        ),
    }
    if mask_path is None:
        fields.update(
            {
                "mask_path": None,
                "mask_sha256": None,
                "mask_bytes": None,
                "mask_array_sha256": None,
                "mask_shape": None,
                "mask_dtype": None,
                "mask_semantics": None,
                "artifact_paths": {
                    "score_map_native": score_relative,
                    "reliability_map_native": reliability_relative,
                    "mask_native": None,
                },
            }
        )
    else:
        mask_pixels = np.where(
            score_map >= MASK_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)
        mask_relative = repo_relative(mask_path, repo_root)
        fields.update(
            {
                "mask_path": mask_relative,
                "mask_sha256": sha256_file(mask_path),
                "mask_bytes": mask_path.stat().st_size,
                "mask_array_sha256": _array_sha256(mask_pixels),
                "mask_shape": list(mask_pixels.shape),
                "mask_dtype": "uint8",
                "mask_semantics": (
                    "native_probability_map_ge_0_5_encoded_L_0_or_255"
                ),
                "artifact_paths": {
                    "score_map_native": score_relative,
                    "reliability_map_native": reliability_relative,
                    "mask_native": mask_relative,
                },
            }
        )
    return fields


def _localization_payload(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    score_map: np.ndarray,
) -> dict[str, Any]:
    target = load_ground_truth(row, repo_root)
    if target is None:
        raise ValueError("T2-applicable TruFor input has no ground truth")
    if target.shape != score_map.shape:
        raise ValueError("TruFor GT/native score-map dimensions differ")
    return {
        "native": binary_pixel_metrics(
            score_map,
            target,
            MASK_THRESHOLD,
            include_ap=row.get("gt_mask_kind") == "exact_diff",
        )
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_score_payload(row: Mapping[str, Any], sample_id: str) -> None:
    score = _finite_number(row.get("ai_score"), f"{sample_id} ai_score")
    logit = _finite_number(
        row.get("raw_detection_logit"),
        f"{sample_id} raw detection logit",
    )
    probabilities = row.get("class_probabilities")
    raw_outputs = row.get("raw_outputs")
    if (
        not 0.0 <= score <= 1.0
        or not math.isclose(
            score,
            _stable_sigmoid(logit),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not isinstance(probabilities, Mapping)
        or dict(probabilities) != {
            "real": 1.0 - score,
            "forged": score,
        }
        or not isinstance(raw_outputs, Mapping)
        or dict(raw_outputs) != {"binary_forged_logit": logit}
        or row.get("probability") != score
        or row.get("score") != score
        or row.get("score_margin") != logit
        or row.get("score_semantics")
        != "sigmoid_binary_logit_probability_of_forged"
        or row.get("calibrated_probability") is not False
        or row.get("classification_decision")
        is not (score >= CLASSIFICATION_THRESHOLD)
        or row.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} T1 score contract changed")


def _resolve_exact_artifact(
    value: Any,
    *,
    repo_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    path = _anchored(Path(value), repo_root)
    if path != expected_path.resolve():
        raise ValueError(f"{label} path is not canonical")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/non-regular {label}: {path}")
    return path


def _load_probability_map(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_path: Path,
    expected_shape: tuple[int, int],
    expected_semantics: str,
    repo_root: Path,
    sample_id: str,
) -> np.ndarray:
    path = _resolve_exact_artifact(
        row.get(f"{prefix}_path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=f"{sample_id} {prefix}",
    )
    if (
        row.get(f"{prefix}_sha256") != sha256_file(path)
        or row.get(f"{prefix}_bytes") != path.stat().st_size
        or path.stat().st_size
        != int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        + NPY_HEADER_BYTES
    ):
        raise ValueError(f"{sample_id} {prefix} file metadata changed")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
        or row.get(f"{prefix}_array_sha256") != _array_sha256(array)
        or row.get(f"{prefix}_shape") != list(expected_shape)
        or row.get(f"{prefix}_dtype") != "float32"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"{sample_id} {prefix} array contract changed")
    return np.asarray(array)


def _validate_preprocess(
    result: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    recompute: bool,
) -> None:
    sample_id = str(input_row["sample_id"])
    preprocess = result.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ValueError(f"{sample_id} preprocess audit is missing")
    width = int(input_row["width"])
    height = int(input_row["height"])
    expected_static = {
        "profile": PREPROCESS_PROFILE,
        "decode": "PIL_convert_RGB",
        "decoded_size": [width, height],
        "decoded_rgb_shape": [height, width, 3],
        "decoded_rgb_dtype": "uint8",
        "tensor_shape": [3, height, width],
        "tensor_dtype": "float32",
        "input_scale": "float32_divide_by_256",
        "input_scale_divisor": 256.0,
        "input_resize": None,
        "input_crop": None,
        "network_map_upsample": (
            "bilinear_align_corners_false_to_native_input_size"
        ),
        "post_network_map_restore": None,
    }
    for key, value in expected_static.items():
        if preprocess.get(key) != value:
            raise ValueError(f"{sample_id} preprocess {key} changed")
    for key in ("decoded_rgb_array_sha256", "tensor_array_sha256"):
        digest = preprocess.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(f"{sample_id} preprocess {key} is invalid")
    if set(preprocess) != set(expected_static) | {
        "decoded_rgb_array_sha256",
        "tensor_array_sha256",
    }:
        raise ValueError(f"{sample_id} preprocess audit key set changed")
    if recompute:
        input_path = _anchored(
            Path(str(input_row["canonical_path"])),
            repo_root,
        )
        _, _, expected = _preprocess_with_audit(input_path)
        if dict(preprocess) != expected:
            raise ValueError(f"{sample_id} preprocessing replay changed")


def _validate_reliability_payload(
    result: Mapping[str, Any],
    *,
    reliability_map: np.ndarray,
    sample_id: str,
) -> None:
    expected = _reliability_payload(reliability_map)
    if result.get("reliability") != expected:
        raise ValueError(f"{sample_id} reliability summary changed")


def _validate_ok_artifacts(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(artifact_root, sample_id)
    expected_shape = (
        int(input_row["height"]),
        int(input_row["width"]),
    )
    score_map = _load_probability_map(
        attempt,
        prefix="score_map_native",
        expected_path=paths["score"],
        expected_shape=expected_shape,
        expected_semantics=(
            "softmax_localization_logits_channel_1_forged_probability"
        ),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    reliability_map = _load_probability_map(
        attempt,
        prefix="reliability_map_native",
        expected_path=paths["reliability"],
        expected_shape=expected_shape,
        expected_semantics=(
            "sigmoid_TCP_localization_reliability_not_anomaly"
        ),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    _validate_reliability_payload(
        attempt,
        reliability_map=reliability_map,
        sample_id=sample_id,
    )
    score_relative = repo_relative(paths["score"], repo_root)
    reliability_relative = repo_relative(paths["reliability"], repo_root)
    applicable, _ = _t2_semantics(input_row)
    expected_artifact_paths = {
        "score_map_native": score_relative,
        "reliability_map_native": reliability_relative,
        "mask_native": (
            repo_relative(paths["mask"], repo_root)
            if applicable
            else None
        ),
    }
    if attempt.get("artifact_paths") != expected_artifact_paths:
        raise ValueError(f"{sample_id} artifact path mapping changed")
    if not applicable:
        null_fields = {
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_array_sha256",
            "mask_shape",
            "mask_dtype",
            "mask_semantics",
            "localization",
        }
        if any(attempt.get(field) is not None for field in null_fields):
            raise ValueError(f"{sample_id} fullframe result claims T2 output")
        return

    mask_path = _resolve_exact_artifact(
        attempt.get("mask_path"),
        repo_root=repo_root,
        expected_path=paths["mask"],
        label=f"{sample_id} native mask",
    )
    if (
        attempt.get("mask_sha256") != sha256_file(mask_path)
        or attempt.get("mask_bytes") != mask_path.stat().st_size
        or attempt.get("mask_shape") != list(expected_shape)
        or attempt.get("mask_dtype") != "uint8"
        or attempt.get("mask_semantics")
        != "native_probability_map_ge_0_5_encoded_L_0_or_255"
        or attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} native mask metadata changed")
    with Image.open(mask_path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError(f"{sample_id} native mask encoding changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    if (
        pixels.shape != expected_shape
        or not np.isin(pixels, (0, 255)).all()
        or attempt.get("mask_array_sha256") != _array_sha256(pixels)
        or not np.array_equal(pixels == 255, score_map >= MASK_THRESHOLD)
    ):
        raise ValueError(f"{sample_id} native threshold mask changed")
    expected_localization = _localization_payload(
        row=input_row,
        repo_root=repo_root,
        score_map=score_map,
    )
    if attempt.get("localization") != expected_localization:
        raise ValueError(f"{sample_id} localization metrics changed")


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
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    common_keys = set(expected) | {"status", "completed_at"}
    expected_keys = (
        common_keys | _OK_ONLY_KEYS
        if status == "ok"
        else common_keys | {"error_type", "error", "traceback"}
    )
    actual_keys = set(attempt)
    if actual_keys != expected_keys:
        raise ValueError(
            "result attempt key set changed: "
            f"missing={sorted(expected_keys - actual_keys)[:1]}, "
            f"extra={sorted(actual_keys - expected_keys)[:1]}"
        )
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"result attempt field {key} drifted")
    sample_id = str(input_row["sample_id"])
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError(f"{sample_id} completed_at is invalid")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt.get("error_type")
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt.get("traceback")
        ):
            raise ValueError(f"error result {sample_id} payload is invalid")
        return
    _validate_score_payload(attempt, sample_id)
    _validate_preprocess(
        attempt,
        input_row=input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    latency = _finite_number(
        attempt.get("latency_ms"),
        f"{sample_id} latency_ms",
    )
    if latency < 0.0:
        raise ValueError(f"{sample_id} latency is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{sample_id} peak memory is invalid")
    if (
        attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} mask threshold changed")
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
    """Allow only zero-or-more errors followed by at most one success."""

    expected_ids = {
        str(row["sample_id"])
        for row in selected
    }
    histories: dict[str, list[str]] = {}
    for line_number, attempt in enumerate(attempts, start=1):
        sample_id = attempt.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError(
                f"result history row {line_number} has unexpected sample_id"
            )
        status = attempt.get("status")
        if status not in ("error", "ok"):
            raise ValueError(
                f"result history row {line_number} has invalid status"
            )
        history = histories.setdefault(sample_id, [])
        if "ok" in history:
            raise ValueError(
                "result history appends an attempt after success: "
                f"{sample_id}"
            )
        history.append(str(status))
    return {
        "policy": "zero_or_more_errors_then_at_most_one_terminal_ok",
        "physical_attempts": len(attempts),
        "sample_histories": len(histories),
        "recovered_error_to_ok": sum(
            statuses[-1] == "ok" and "error" in statuses[:-1]
            for statuses in histories.values()
        ),
    }


def _prepare_artifact_root(artifact_root: Path) -> None:
    expected_directories = {
        "score_maps_native",
        "reliability_maps_native",
        "masks_native",
    }
    if artifact_root.exists():
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise ValueError(
                f"TruFor artifact root is unsafe: {artifact_root}"
            )
        for entry in artifact_root.iterdir():
            if (
                entry.name not in expected_directories
                or not entry.is_dir()
                or entry.is_symlink()
            ):
                raise ValueError(
                    "TruFor artifact root contains unexpected/unsafe entry: "
                    f"{entry}"
                )
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected_directories):
        (artifact_root / name).mkdir(exist_ok=True)


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    expected_directories = {
        "score_maps_native",
        "reliability_maps_native",
        "masks_native",
    }
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe TruFor artifact root: {artifact_root}"
        )
    root_entries = list(artifact_root.iterdir())
    actual_directories = {entry.name for entry in root_entries}
    if (
        actual_directories != expected_directories
        or any(
            not entry.is_dir() or entry.is_symlink()
            for entry in root_entries
        )
    ):
        raise ValueError("TruFor artifact root inventory mismatch")
    inputs_by_id = {
        str(row["sample_id"]): row
        for row in selected
    }
    successful = {
        sample_id
        for sample_id, result in latest_by_sample_id.items()
        if result.get("status") == "ok"
    }
    expected = {
        "score_maps_native": {
            f"{sample_id}.npy"
            for sample_id in successful
        },
        "reliability_maps_native": {
            f"{sample_id}.npy"
            for sample_id in successful
        },
        "masks_native": {
            f"{sample_id}.png"
            for sample_id in successful
            if _t2_semantics(inputs_by_id[sample_id])[0]
        },
    }
    counts: dict[str, int] = {}
    for directory_name, expected_names in expected.items():
        directory = artifact_root / directory_name
        entries = list(directory.iterdir())
        if any(
            entry.is_symlink() or not entry.is_file()
            for entry in entries
        ):
            raise ValueError(
                f"TruFor {directory_name} contains unsafe/non-file entries"
            )
        actual_names = {entry.name for entry in entries}
        if actual_names != expected_names:
            raise ValueError(
                f"TruFor {directory_name} inventory mismatch: "
                f"missing={sorted(expected_names - actual_names)[:1]}, "
                f"extra={sorted(actual_names - expected_names)[:1]}"
            )
        counts[directory_name] = len(actual_names)
    return counts


def _required_artifact_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    if not rows:
        return 0
    map_bytes = sum(
        2
        * (
            int(row["width"])
            * int(row["height"])
            * np.dtype(np.float32).itemsize
            + NPY_HEADER_BYTES
        )
        for row in rows
    )
    return map_bytes + MIN_DISK_RESERVE_BYTES


def _verify_disk_capacity(
    artifact_root: Path,
    pending: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    required = _required_artifact_bytes(pending)
    usage = shutil.disk_usage(artifact_root.parent)
    if usage.free < required:
        raise OSError(
            "insufficient disk for complete TruFor raw outputs: "
            f"required={required}, free={usage.free}"
        )
    return {
        "free_bytes_before_inference": int(usage.free),
        "estimated_pending_map_bytes_plus_reserve": int(required),
        "fixed_reserve_bytes": MIN_DISK_RESERVE_BYTES,
    }


def _validate_prior_finalized_output_hashes(
    prior_manifest: Mapping[str, Any],
    *,
    results_path: Path,
    summary_path: Path,
) -> None:
    """Bind a finalized resume to the exact prior result and summary bytes."""

    status = prior_manifest.get("status")
    if status == "running":
        return
    if status not in ("complete", "incomplete"):
        raise ValueError("resume prior manifest has an invalid status")
    outputs = prior_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resume prior manifest outputs are missing")
    for path, key, label in (
        (results_path, "results_sha256", "results"),
        (summary_path, "summary_sha256", "summary"),
    ):
        expected = outputs.get(key)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected
            )
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"resume prior {label} SHA-256 changed")


def _validate_prior_manifest_envelope(
    prior_manifest: Mapping[str, Any],
    *,
    expected_dataset: Mapping[str, Any],
    expected_base_outputs: Mapping[str, Any],
) -> None:
    """Validate every mutable outer-manifest field before a resume."""

    status = prior_manifest.get("status")
    if status not in ("running", "complete", "incomplete"):
        raise ValueError("resume prior manifest has an invalid status")
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
    expected_keys = (
        base_keys
        if status == "running"
        else base_keys | {"execution"}
    )
    if set(prior_manifest) != expected_keys:
        raise ValueError("resume prior manifest key set changed")
    started_at = prior_manifest.get("started_at")
    completed_at = prior_manifest.get("completed_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume prior manifest started_at changed")
    if status == "running":
        if completed_at is not None:
            raise ValueError("running prior manifest has completed_at")
    elif not isinstance(completed_at, str) or not completed_at:
        raise ValueError("finalized prior manifest completed_at changed")
    if prior_manifest.get("dataset") != dict(expected_dataset):
        raise ValueError("resume prior manifest dataset binding changed")

    disk = prior_manifest.get("disk_preflight")
    if (
        not isinstance(disk, Mapping)
        or set(disk)
        != {
            "free_bytes_before_inference",
            "estimated_pending_map_bytes_plus_reserve",
            "fixed_reserve_bytes",
        }
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in disk.values()
        )
        or disk.get("fixed_reserve_bytes") != MIN_DISK_RESERVE_BYTES
    ):
        raise ValueError("resume prior disk preflight changed")

    outputs = prior_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resume prior manifest outputs changed")
    expected_output_keys = set(expected_base_outputs)
    if status != "running":
        expected_output_keys |= {
            "results_sha256",
            "summary_sha256",
            "artifact_inventory",
        }
    if set(outputs) != expected_output_keys:
        raise ValueError("resume prior manifest output key set changed")
    for key, value in expected_base_outputs.items():
        if outputs.get(key) != value:
            raise ValueError(f"resume prior manifest output {key} changed")
    if status == "running":
        return
    for key in ("results_sha256", "summary_sha256"):
        digest = outputs.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(f"resume prior manifest {key} is invalid")
    inventory = outputs.get("artifact_inventory")
    if (
        not isinstance(inventory, Mapping)
        or set(inventory)
        != {
            "score_maps_native",
            "reliability_maps_native",
            "masks_native",
        }
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in inventory.values()
        )
    ):
        raise ValueError("resume prior manifest artifact inventory changed")
    execution = prior_manifest.get("execution")
    execution_keys = {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
    }
    if (
        not isinstance(execution, Mapping)
        or set(execution) != execution_keys
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in execution.values()
        )
        or execution["physical_result_rows"]
        < execution["latest_result_rows"]
    ):
        raise ValueError("resume prior manifest execution changed")


def _validate_run_directory_safety(
    run_dir: Path,
    *,
    resume: bool,
) -> None:
    """Restrict a run directory to regular, non-symlink contract files."""

    if not run_dir.exists():
        if resume:
            raise FileNotFoundError(
                f"resume run directory is missing: {run_dir}"
            )
        return
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"TruFor run directory is unsafe: {run_dir}")
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
    }
    entries = list(run_dir.iterdir())
    unexpected = {
        entry.name
        for entry in entries
        if entry.name not in allowed
    }
    if unexpected:
        raise ValueError(
            "TruFor run directory contains unexpected entries: "
            f"{sorted(unexpected)[:1]}"
        )
    for entry in entries:
        _reject_symlink_components(
            entry,
            f"TruFor run file {entry.name}",
        )
        if not entry.is_file() or entry.is_symlink():
            raise ValueError(
                f"TruFor run entry is not a regular file: {entry}"
            )
    if resume:
        names = {entry.name for entry in entries}
        required = {"manifest.json", "expected_inputs.jsonl"}
        if not required.issubset(names):
            raise FileNotFoundError(
                "resume requires regular manifest.json and "
                "expected_inputs.jsonl"
            )


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
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
        "adapter_sources": dict(adapter_sources),
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repository": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "positive_class_index": 1,
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "decode": "PIL_convert_RGB",
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "input_scale_divisor": 256.0,
            "input_resize": None,
            "input_crop": None,
            "network_map_upsample": (
                "bilinear_align_corners_false_to_native_input_size"
            ),
            "batch_size": 1,
            "autocast": False,
        },
        "score_spec": SCORE_SPEC.as_dict(),
        "t2_spec": T2_SPEC,
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"])
            for row in selected
        ),
        "source": dict(cpu_preflight["source"]),
        "assets": dict(cpu_preflight["assets"]),
        "environment": dict(cpu_preflight["environment"]),
        "checkpoint_audit": dict(cpu_preflight["checkpoint_audit"]),
        "model_audit": dict(cpu_preflight["model_audit"]),
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
            "score_maps_native_dir": repo_relative(
                artifact_root / "score_maps_native",
                repo_root,
            ),
            "reliability_maps_native_dir": repo_relative(
                artifact_root / "reliability_maps_native",
                repo_root,
            ),
            "masks_native_dir": repo_relative(
                artifact_root / "masks_native",
                repo_root,
            ),
        },
    }


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.mode == "formal":
        run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
        if run_id != DEFAULT_FORMAL_RUN_ID:
            raise ValueError(
                f"formal run-id must be exactly {DEFAULT_FORMAL_RUN_ID}"
            )
        return run_id
    if args.mode == "smoke":
        if args.run_id is None:
            raise ValueError("smoke mode requires an explicit A/B --run-id")
        run_id = _valid_run_id(args.run_id)
        if run_id not in {
            DEFAULT_SMOKE_RUN_ID_A,
            DEFAULT_SMOKE_RUN_ID_B,
        }:
            raise ValueError("smoke run-id must be the frozen A or B ID")
        return run_id
    if args.mode == "single":
        if args.run_id is None:
            raise ValueError("single mode requires an explicit --run-id")
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
        "--trufor-root",
        type=Path,
        default=legacy.DEFAULT_TRUFOR_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=legacy.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
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
        help="explicit cpu or cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    trufor_root = _unresolved_anchored(args.trufor_root, repo_root)
    checkpoint_path = _unresolved_anchored(args.checkpoint, repo_root)
    archive_path = _unresolved_anchored(args.archive, repo_root)

    # The canonical release itself is part of the CPU preflight boundary.
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
        cpu_preflight = run_cpu_preflight(
            trufor_root=trufor_root,
            checkpoint_path=checkpoint_path,
            archive_path=archive_path,
        )
        report = {
            **cpu_preflight,
            "dataset": {
                "dataset_id": release.dataset_id,
                "manifest_sha256": release.manifest_sha256,
                "verified_images": len(release.inputs),
                "verify_files": True,
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
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

    requested_results_root = _unresolved_anchored(
        args.results_dir,
        repo_root,
    )
    expected_results_candidate = repo_root / DEFAULT_RESULTS_DIR
    _reject_symlink_components(
        requested_results_root,
        "TruFor results root",
    )
    _reject_symlink_components(
        expected_results_candidate,
        "expected TruFor results root",
    )
    results_root = requested_results_root.resolve()
    expected_results_root = expected_results_candidate.resolve()
    if results_root != expected_results_root:
        raise ValueError(
            f"--results-dir must be exactly {DEFAULT_RESULTS_DIR}"
        )
    requested_artifacts_root = _unresolved_anchored(
        args.artifacts_dir,
        repo_root,
    )
    expected_artifacts_candidate = repo_root / DEFAULT_ARTIFACTS_DIR
    _reject_symlink_components(
        requested_artifacts_root,
        "TruFor artifacts root",
    )
    _reject_symlink_components(
        expected_artifacts_candidate,
        "expected TruFor artifacts root",
    )
    artifacts_root = requested_artifacts_root.resolve()
    expected_artifacts_root = expected_artifacts_candidate.resolve()
    if artifacts_root != expected_artifacts_root:
        raise ValueError(
            f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}"
        )
    run_dir = _safe_child(results_root, run_id, "TruFor run directory")
    artifact_root = _safe_child(
        artifacts_root,
        run_id,
        "TruFor artifact directory",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError(
            "TruFor result and artifact directories must be disjoint"
        )
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    _validate_run_directory_safety(run_dir, resume=args.resume)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    if (
        artifact_root.exists()
        and any(artifact_root.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            "artifact directory is non-empty; pass --resume: "
            f"{artifact_root}"
        )

    # This strict CPU model-load gate deliberately precedes all accelerator
    # selection/configuration.
    cpu_preflight = run_cpu_preflight(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    adapter_sources = adapter_source_contract(repo_root)
    device_text = args.device or "cuda:0"
    device, runtime = configure_runtime(device_text)
    import torch

    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=args.mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        adapter_sources=adapter_sources,
        cpu_preflight=cpu_preflight,
        runtime=runtime,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    fingerprint = _fingerprint(immutable)

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )
        prior_manifest = _load_json_object_strict(manifest_path)
        if (
            prior_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
            or prior_manifest.get("run_id") != run_id
            or prior_manifest.get("fingerprint") != fingerprint
            or prior_manifest.get("immutable") != immutable
        ):
            raise ValueError(
                "resume run manifest fingerprint/config drifted"
            )
        if _read_jsonl_strict(expected_path) != selected:
            raise ValueError("resume expected input snapshot drifted")
        started_at = prior_manifest.get("started_at")
    else:
        prior_manifest = None
        atomic_write_jsonl(expected_path, selected)
        started_at = utc_now()
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("run manifest started_at is invalid")

    dataset_record = {
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
        "t2_applicable_images": sum(
            _t2_semantics(row)[0]
            for row in selected
        ),
    }
    if prior_manifest is not None:
        _validate_prior_manifest_envelope(
            prior_manifest,
            expected_dataset=dataset_record,
            expected_base_outputs=immutable["outputs"],
        )

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
    if prior_manifest is not None:
        # A finalized resume is bound to its prior result/summary bytes before
        # any directory creation or manifest rewrite can occur.
        _validate_prior_finalized_output_hashes(
            prior_manifest,
            results_path=results_path,
            summary_path=summary_path,
        )

    physical_before = (
        _read_jsonl_strict(results_path)
        if results_path.is_file()
        else []
    )
    _validate_physical_attempt_history(selected, physical_before)
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {
        str(row["sample_id"]): row
        for row in selected
    }
    for attempt in physical_before:
        input_row = inputs_by_id[str(attempt["sample_id"])]
        _validate_runner_attempt(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=False,
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
    for row in selected:
        sample_id = str(row["sample_id"])
        prior = latest_before.latest_by_sample_id.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
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
    if prior_manifest is None:
        _prepare_artifact_root(artifact_root)
    validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest_before.latest_by_sample_id,
    )
    disk_audit = _verify_disk_capacity(artifact_root, pending)
    # Resume stays non-mutating until every reusable row and artifact passes.
    manifest["disk_preflight"] = disk_audit
    atomic_write_json(manifest_path, manifest)

    model = None
    new_successes = 0
    resume_skips = len(selected) - len(pending)
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            model, loaded_device = legacy.load_model(
                trufor_root=trufor_root,
                checkpoint_path=checkpoint_path,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("TruFor loaded on an unexpected device")
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
                tensor, (width, height), preprocess = (
                    _preprocess_with_audit(input_path)
                )
                if (width, height) != (
                    int(input_row["width"]),
                    int(input_row["height"]),
                ):
                    raise ValueError(
                        "canonical image dimensions changed"
                    )
                assert model is not None
                (
                    score,
                    detection_logit,
                    raw_score_map,
                    raw_reliability_map,
                    peak_bytes,
                    latency_ms,
                ) = legacy.infer_one(model, device, tensor)
                score_map = np.ascontiguousarray(
                    raw_score_map,
                    dtype=np.float32,
                )
                reliability_map = np.ascontiguousarray(
                    raw_reliability_map,
                    dtype=np.float32,
                )
                expected_shape = (height, width)
                for label, value in (
                    ("score", score_map),
                    ("reliability", reliability_map),
                ):
                    if (
                        value.shape != expected_shape
                        or value.dtype != np.float32
                        or not value.flags.c_contiguous
                        or not np.isfinite(value).all()
                        or float(value.min()) < 0.0
                        or float(value.max()) > 1.0
                    ):
                        raise ValueError(
                            f"TruFor native {label} map is invalid"
                        )
                legacy._atomic_save_npy(paths["score"], score_map)
                legacy._atomic_save_npy(
                    paths["reliability"],
                    reliability_map,
                )
                applicable, _ = _t2_semantics(input_row)
                mask_path: Path | None = None
                localization: dict[str, Any] | None = None
                if applicable:
                    mask_path = paths["mask"]
                    legacy._atomic_save_mask(
                        mask_path,
                        score_map >= MASK_THRESHOLD,
                    )
                    localization = _localization_payload(
                        row=input_row,
                        repo_root=repo_root,
                        score_map=score_map,
                    )
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **_score_payload(score, detection_logit),
                    **_artifact_fields(
                        repo_root=repo_root,
                        paths=paths,
                        score_map=score_map,
                        reliability_map=reliability_map,
                        mask_path=mask_path,
                    ),
                    "mask_threshold": MASK_THRESHOLD,
                    "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
                    "localization": localization,
                    "reliability": _reliability_payload(
                        reliability_map
                    ),
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
                    f"[{index}/{len(pending)}] error "
                    f"{sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
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
        _read_jsonl_strict(results_path)
        if results_path.is_file()
        else []
    )
    _validate_physical_attempt_history(selected, physical_results)
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
            # Every new successful row was read back before append, and every
            # reusable latest row was replay-validated before inference.  The
            # final pass rechecks the complete append-only identity/history
            # without repeating billion-pixel AP work a third time.
            verify_artifacts=False,
        )
    coverage = summarize_coverage(latest)
    inventories = validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_trufor_balanced.py",
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
        "artifact_inventory": inventories,
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
            "artifact_inventory": inventories,
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
                "artifact_inventory": inventories,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError(
            "TruFor fail-fast inference failed"
        ) from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
