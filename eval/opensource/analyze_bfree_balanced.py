#!/usr/bin/env python3
"""Fail-closed audit and full replay for B-Free on Balanced250.

The analyzer treats the runner manifest, append-only JSONL, and one NPZ
artifact per successful image as untrusted evidence.  It independently pins
the released B-Free source and checkpoint, rebuilds the exact Balanced250
selection, validates the complete artifact inventory, replays the released
linear head from every persisted ``[5, 768]`` feature array, recomputes only
the shared Balanced250 T1 metrics, and by default freshly replays every one of
the 1,775 canonical JPEGs through the complete official five-crop model.

B-Free is a whole-image classifier.  Its five crop logits and features, and
the score-blind exact-difference crop-visibility census, are diagnostics for
T1 execution.  They are not dense predictions or T2 localization outputs.
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
import re
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import analyze_bfree_run as legacy_audit
from eval.opensource import run_bfree as legacy
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
from eval.opensource.balanced_run_contract import (
    RunDatasetContract,
    ScoreSpec,
    build_run_dataset_contract,
    index_latest_attempts,
    require_complete_coverage,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    Capability,
    CanonicalRelease,
    SelectionSpec,
    load_canonical_release,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "bfree_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "bfree_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
EXPECTED_RUN_MANIFEST_SCHEMA = "bfree_balanced_run_manifest_v2"
EXPECTED_RUN_CONFIG_SCHEMA = "bfree_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = "bfree_balanced_runtime_summary_v2"
EXPECTED_CPU_PREFLIGHT_SCHEMA = "bfree_balanced_cpu_preflight_v1"
EXPECTED_RUN_DATASET_CONTRACT_SCHEMA = (
    "opensource_run_dataset_contract_v2"
)

DEFAULT_RESULTS_DIR = Path("results/opensource/bfree")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/bfree")
DEFAULT_RUN_ID = (
    "bfree_dino2reg4_balanced250_v1_full1775_20260727"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "bfree_dino2reg4_balanced250_v1_smoke5x7_a_r3_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "bfree_dino2reg4_balanced250_v1_smoke5x7_b_r3_20260727"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/b-free-c6a9f898"
)
DEFAULT_WEIGHTS_DIR = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4"
)
DEFAULT_WEIGHTS_ZIP = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4.zip"
)

FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
EXPECTED_RUNTIME_SEED = 20260725
FEATURE_SHAPE = (5, 768)
CROP_LOGITS_SHAPE = (5,)
FEATURE_DTYPE = np.dtype(np.float32)
FEATURE_NBYTES = 5 * 768 * FEATURE_DTYPE.itemsize
CROP_LOGITS_NBYTES = 5 * FEATURE_DTYPE.itemsize
NPZ_FILE_BYTES = 15_904
NPZ_MEMBER_BYTES = {
    "features.npy": 15_488,
    "crop_logits.npy": 148,
}
RAW_LOGIT_ABS_TOLERANCE = 1e-6
FEATURE_ABS_TOLERANCE = 0.0
CROP_LOGIT_ABS_TOLERANCE = 1e-6
PROBABILITY_ABS_TOLERANCE = 1e-7
CROSS_DEVICE_MEAN_ABS_TOLERANCE = 2e-6
PERSISTED_HEAD_ABS_TOLERANCE = 0.0

EXPECTED_SOURCE_COMMIT = (
    "c6a9f898782fb466b29af01f21960b67415afb0e"
)
EXPECTED_ZIP_SHA256 = (
    "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f"
)
EXPECTED_ZIP_BYTES = 321_653_488
EXPECTED_CONFIG_SHA256 = (
    "1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40"
)
EXPECTED_CONFIG_BYTES = 153
EXPECTED_CHECKPOINT_SHA256 = (
    "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
)
EXPECTED_CHECKPOINT_BYTES = 346_171_370
EXPECTED_CHECKPOINT_TENSORS = 177
EXPECTED_CHECKPOINT_ELEMENTS = 86_526_721
EXPECTED_CHECKPOINT_SCHEMA_SHA256 = (
    "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524"
)
EXPECTED_ASSET_BUNDLE_SHA256 = (
    "58859ff170ba42edd9c13bfcbc0094513de227d7001e5a261f7c37dd69db8349"
)
EXPECTED_FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/bfree/bin/python"
)
EXPECTED_FROZEN_VENV_PREFIX = Path(
    "/root/.cache/claimforge/venvs/bfree"
)
EXPECTED_FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/bfree-balanced-v2-empty"
)
EXPECTED_FROZEN_PYVENV_CONFIG_SHA256 = (
    "1ee492ad073827f75ebf74bf270e554ee23a28ee44756d616218d4bd6e40c6cc"
)
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "timm": "1.0.12",
    "transformers": "4.43.4",
    "safetensors": "0.5.2",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "PyYAML": "6.0.2",
    "scipy": "1.16.0",
    "scikit-learn": "1.6.1",
    "joblib": "1.4.2",
    "threadpoolctl": "3.5.0",
    "setuptools": "79.0.1",
}
EXPECTED_TORCH_DISTRIBUTION_VERSION = "2.8.0.dev20250627+cu128"
EXPECTED_TORCHVISION_DISTRIBUTION_VERSION = "0.23.0.dev20250627+cu128"
EXPECTED_MODULE_HASHES = {
    "torch": "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d",
    "torchvision": "ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73",
    "timm": "d1b91a5531a3481e81661a86f05c3a3c5726197238ec174db78720f9e03f41d0",
    "transformers": "936288235b9270f7d6d8c85709892f74f1407d5cfd6d223b0539fe4bbf34b28f",
    "numpy": "6ae17b070c0f70a8e3cad89a510a256942e5a1f37ea5feb120cec167ed2a6236",
    "PIL": "7c95303c6848f3f99c07c8cd583fa1530ecc88c2725a0a955ff9c5b73223d59b",
    "yaml": "377e52d351cc7ac1537b469144c5a43e3d0f6bc2046c7a44f452bb72be4176dc",
    "scipy": "a72412a5c62442876e073a01da1910c84b7214d1382380b71a873b470420dd68",
    "sklearn": "7a2a7d742b4681503a9bfac50fcaf69706f86069deb3eb287408f1f32721e1fc",
}
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

EXPECTED_MODEL_CONTRACT = {
    "name": "B-Free",
    "slug": "bfree_dino2reg4",
    "architecture": "vit_base_patch14_reg4_dinov2.lvd142m",
    "repository": "https://github.com/grip-unina/B-Free",
    "paper": "https://arxiv.org/abs/2412.17671",
    "source_commit": EXPECTED_SOURCE_COMMIT,
    "checkpoint_id": (
        "official_grip_unina_BFREE_dino2reg4_model_epoch_best"
    ),
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "checkpoint_schema_sha256": EXPECTED_CHECKPOINT_SCHEMA_SHA256,
    "asset_bundle_sha256": EXPECTED_ASSET_BUNDLE_SHA256,
    "construction": "official_Wrapper5crops_strict_full_state",
    "feature_shape": [5, 768],
    "score": {
        "semantics": "official_float32_mean_of_five_crop_raw_logits",
        "direction": "higher_means_fake",
        "threshold": 0.0,
        "threshold_operator": ">",
    },
    "license": {
        "code_and_weights": {
            "license": "GRIP_UNINA_nonprofit_research_only",
            "license_file_sha256": (
                "cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27"
            ),
            "commercial_use": False,
        },
        "benchmark_role": "research_evaluation_only",
    },
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

EXPECTED_ADAPTER_SOURCE_PATHS = (
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

EXPECTED_OK_RESULT_FIELDS = frozenset(
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

_RUN_IDENTITY_FIELDS = frozenset(
    {"run_id", "run_manifest_fingerprint", "config_fingerprint"}
)
_SMOKE_IGNORED_FIELDS = frozenset(
    {
        *_RUN_IDENTITY_FIELDS,
        "completed_at",
        "preprocess_latency_ms",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_ARTIFACT_VOLATILE_FIELDS = frozenset(
    {"relative_path", "sha256", "feature_array_sha256", "crop_logits_array_sha256"}
)
_FALSE_SCOPE_KEYS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULL_SCOPE_KEYS = frozenset(
    {
        "localization_output",
        "localisation_output",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "pair_rank",
        "t2",
        "joint",
        "joint_output",
        "joint_score",
        "s_joint",
        "localization",
        "localisation",
        "dense_output",
        "score_map",
        "predicted_mask",
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
_FORBIDDEN_PREFIXES = (
    "t2_",
    "pixel_",
    "localization_",
    "localisation_",
    "dense_",
    "heatmap_",
    "attention_map",
    "mask_",
    "predicted_mask",
    "score_map",
    "joint_",
)
_ALLOWED_DIAGNOSTIC_KEYS = frozenset(
    {
        "edit_visibility_evidence",
        "patch_visibility",
        "gt_mask_kind",
        "mask_positive_pixels",
    }
)


@dataclass(frozen=True)
class BFreeArtifact:
    """One fully validated canonical B-Free NPZ artifact."""

    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    feature_array_sha256: str
    crop_logits_array_sha256: str
    features: np.ndarray
    crop_logits: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    """All independently validated evidence for one formal or smoke run."""

    run_id: str
    fingerprint: str
    mode: str
    run_dir: Path
    manifest_path: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    immutable: dict[str, Any]
    release: CanonicalRelease
    selected: tuple[dict[str, Any], ...]
    contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]
    artifact_dir: Path
    artifacts: Mapping[str, BFreeArtifact]
    evidence_snapshot: Mapping[str, str]


def _runner() -> Any:
    return importlib.import_module("eval.opensource.run_bfree_balanced")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _json_loads(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} is not a non-empty safe string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return result


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _same_json_type_and_value(actual: Any, expected: Any) -> bool:
    """Canonical JSON equality that never lets bool impersonate int."""

    return type(actual) is type(expected) and stable_json(actual) == stable_json(
        expected
    )


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"missing regular {label}: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return _require_mapping(_json_loads(text, label), label)


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
                row = _require_mapping(_json_loads(line, row_label), row_label)
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return rows


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode()
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


def _safe_repo_path(
    value: Any,
    *,
    repo_root: Path,
    label: str,
    require_file: bool = True,
) -> Path:
    relative = _require_string(value, label)
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} is absolute, non-canonical, or traversing")
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
        raise ValueError(f"{label} escapes repository root") from error
    if require_file:
        _require_regular_file(resolved, label)
    return resolved


def _safe_absolute_dir(path: Path, *, root: Path, label: str) -> Path:
    resolved_root = root.resolve()
    raw = path if path.is_absolute() else resolved_root / path
    candidate = Path(os.path.abspath(raw))
    try:
        relative = candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed root") from error
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if not candidate.is_dir():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    return candidate


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value) is None
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run ID is not one safe ASCII path component")
    return value


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    raw = results_dir if results_dir.is_absolute() else root / results_dir
    result = Path(os.path.abspath(raw))
    try:
        relative = result.relative_to(root)
    except ValueError as error:
        raise ValueError("results root escapes repository") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("results root contains a symlink component")
    return result


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    safe_id = _valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / safe_id).resolve()
    if candidate == root or candidate.parent != root:
        raise ValueError("run directory escapes results root")
    if (root / safe_id).is_symlink():
        raise ValueError("run directory is a symlink")
    return candidate


def _reject_nonfinite_numbers(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_numbers(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _reject_nonfinite_numbers(nested, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        raise ValueError(f"{label} is not finite")


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    """Reject pair-rank and every invented dense, T2, or joint claim."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(nested, child)
            elif normalized in _FALSE_SCOPE_KEYS:
                if nested is not False:
                    raise ValueError(f"{child} must be false")
            elif normalized in _NULL_SCOPE_KEYS:
                if nested is not None:
                    raise ValueError(f"{child} must be null")
            elif normalized in _FORBIDDEN_KEYS or any(
                normalized.startswith(prefix)
                for prefix in _FORBIDDEN_PREFIXES
            ):
                raise ValueError(f"{child} invents unsupported B-Free T2 data")
            else:
                _reject_unsupported_claims(nested, child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _reject_unsupported_claims(nested, f"{label}[{index}]")


def _score_spec() -> ScoreSpec:
    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=0.0,
        threshold_operator=">",
    )


def _sigmoid_float32(raw_logit: float) -> float:
    value = np.float32(raw_logit)
    return float(
        np.float32(1.0)
        / (np.float32(1.0) + np.exp(np.float32(-value), dtype=np.float32))
    )


def _cpu_float32_crop_mean_sanity(crop_logits: np.ndarray) -> float:
    """Compute a CPU FP32 sanity mean without claiming CUDA bit equality."""

    crops = np.ascontiguousarray(crop_logits, dtype=np.float32)
    if crops.shape != CROP_LOGITS_SHAPE or not np.isfinite(crops).all():
        raise ValueError("B-Free crop logits for CPU sanity are malformed")
    import torch

    cuda_was_initialized = bool(torch.cuda.is_initialized())
    tensor = torch.from_numpy(crops)
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError("B-Free score sanity reduction is not CPU FP32")
    with torch.inference_mode():
        mean = float(torch.mean(tensor, dtype=torch.float32).item())
    if bool(torch.cuda.is_initialized()) is not cuda_was_initialized:
        raise RuntimeError("CPU B-Free score sanity reduction initialized CUDA")
    if not math.isfinite(mean):
        raise ValueError("B-Free CPU crop-mean sanity value is not finite")
    return mean


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    expected_crop_logits: np.ndarray | None = None,
    expected_raw_logit: float | None = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    raw = _require_finite(row.get("raw_logit"), f"{sample_id}.raw_logit")
    for alias in ("ai_score", "score"):
        value = _require_finite(row.get(alias), f"{sample_id}.{alias}")
        if value != raw:
            raise ValueError(f"{sample_id} raw score aliases differ")
    crop_values = _require_list(row.get("crop_logits"), f"{sample_id}.crop_logits")
    if len(crop_values) != CROP_LOGITS_SHAPE[0]:
        raise ValueError(f"{sample_id} crop-logit count changed")
    crop_logits = np.asarray(
        [
            _require_finite(value, f"{sample_id}.crop_logits[{index}]")
            for index, value in enumerate(crop_values)
        ],
        dtype=np.float32,
    )
    if crop_logits.shape != CROP_LOGITS_SHAPE or not np.isfinite(
        crop_logits
    ).all():
        raise ValueError(f"{sample_id} crop logits are malformed")
    cpu_mean = _cpu_float32_crop_mean_sanity(crop_logits)
    if not math.isclose(
        raw,
        cpu_mean,
        rel_tol=0.0,
        abs_tol=CROSS_DEVICE_MEAN_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"{sample_id} raw logit fails the frozen cross-device "
            "FP32 five-crop mean sanity tolerance"
        )
    if expected_raw_logit is not None:
        replayed_raw = _require_finite(
            expected_raw_logit, f"{sample_id}.replayed_raw_logit"
        )
        if not math.isclose(
            raw, replayed_raw, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"{sample_id} raw logit differs from recorded-device replay"
            )
    if expected_crop_logits is not None:
        expected = np.ascontiguousarray(expected_crop_logits, dtype=np.float32)
        if expected.shape != CROP_LOGITS_SHAPE or not np.allclose(
            crop_logits, expected, rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"{sample_id} crop logits differ from replay")
    probability = _sigmoid_float32(raw)
    recorded_probability = _require_finite(
        row.get("fake_probability"), f"{sample_id}.fake_probability"
    )
    if not math.isclose(
        recorded_probability,
        probability,
        rel_tol=0.0,
        abs_tol=PROBABILITY_ABS_TOLERANCE,
    ):
        raise ValueError(f"{sample_id} diagnostic probability changed")
    decision = raw > 0.0
    if (
        row.get("score_semantics") != legacy.SCORE_SEMANTICS
        or row.get("classification_decision") is not decision
        or not _same_json_type_and_value(
            row.get("classification_threshold"), 0.0
        )
        or row.get("classification_threshold_operator") != ">"
    ):
        raise ValueError(f"{sample_id} released score policy changed")
    expected_classification = {
        "raw_logit": raw,
        "ai_score": raw,
        "fake_probability": recorded_probability,
        "decision": decision,
        "threshold": 0.0,
        "threshold_operator": ">",
        "semantics": legacy.SCORE_SEMANTICS,
    }
    classification = _require_mapping(
        row.get("classification"), f"{sample_id}.classification"
    )
    if stable_json(classification) != stable_json(expected_classification):
        raise ValueError(f"{sample_id} nested classification changed")
    expected_t1 = {
        key: value
        for key, value in expected_classification.items()
        if key != "semantics"
    }
    expected_t1["policy"] = legacy.T1_POLICY
    if stable_json(row.get("t1")) != stable_json(expected_t1):
        raise ValueError(f"{sample_id} nested T1 record changed")
    manual = _require_mapping(
        row.get("manual_replay"), f"{sample_id}.manual_replay"
    )
    required_manual = {
        "crop_logits": crop_logits.tolist(),
        "raw_logit": raw,
        "ai_score": raw,
        "official_crop_logits_exact_match": True,
        "official_mean_exact_match": True,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
    }
    if stable_json(manual) != stable_json(required_manual):
        raise ValueError(f"{sample_id} manual head replay changed")
    return {
        "raw_logit": raw,
        "fake_probability": recorded_probability,
        "crop_logits": crop_logits,
        "decision": decision,
    }


def _independent_visibility_diagnostic(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    width = _require_nonnegative_int(row.get("width"), "input width")
    height = _require_nonnegative_int(row.get("height"), "input height")
    if width == 0 or height == 0:
        raise ValueError("input dimensions must be positive")
    geometry = legacy_audit.compute_preprocess_geometry(width, height)
    condition = row.get("condition")
    if condition in {"local_mouse", "local_cat", "local_trash_can"}:
        mask = legacy_audit._load_gt_mask(
            row,
            repo_root=repo_root,
            expected_shape=(height, width),
        )
        core = legacy_audit._visibility_from_exact_gt(mask, geometry)
        return {
            "edit_visibility": core["edit_visibility"],
            "edit_visible_gt_fraction": core["edit_visible_gt_fraction"],
            "edit_visibility_evidence": {
                "basis": (
                    "exact_diff_positive_pixels_intersecting_union_of_"
                    "official_five_patch_crop_receptive_fields"
                ),
                "role": "input_condition_stratum_not_model_localization",
                "gt_mask_kind": "exact_diff",
                "positive_pixels": int(np.count_nonzero(mask)),
                "visible_positive_pixels": core[
                    "edit_visibility_evidence"
                ]["visible_gt_positive_pixels"],
                "geometry": geometry,
            },
        }
    if condition not in {
        "real",
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    }:
        raise ValueError("unsupported Balanced250 condition")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": (
                "authentic_input_has_no_edit"
                if condition == "real"
                else "fullframe_condition_has_no_local_GT"
            ),
            "role": "input_condition_stratum_not_model_localization",
            "gt_mask_kind": (
                "all_zero" if condition == "real" else "not_applicable"
            ),
            "geometry": geometry,
        },
    }


def _independent_visibility_census(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    local_conditions = ("local_mouse", "local_cat", "local_trash_can")
    wrap_by_condition: Counter[str] = Counter()
    starts_by_condition = {
        condition: Counter() for condition in BALANCED_CONDITIONS
    }
    for row in rows:
        condition = str(row["condition"])
        geometry = legacy_audit.compute_preprocess_geometry(
            int(row["width"]), int(row["height"])
        )
        wrap_by_condition[condition] += int(
            bool(geometry["replicate_wrap_applied"])
        )
        starts_by_condition[condition][
            int(geometry["distinct_crop_starts"])
        ] += 1
    by_condition: dict[str, dict[str, Any]] = {}
    all_counts: Counter[str] = Counter()
    all_fractions: list[float] = []
    all_wrap = 0
    all_starts: Counter[int] = Counter()
    for condition in local_conditions:
        condition_rows = [
            row for row in rows if row.get("condition") == condition
        ]
        counts: Counter[str] = Counter()
        fractions: list[float] = []
        wrapped = 0
        starts: Counter[int] = Counter()
        for row in condition_rows:
            diagnostic = _independent_visibility_diagnostic(
                row, repo_root=repo_root
            )
            category = str(diagnostic["edit_visibility"])
            fraction = float(diagnostic["edit_visible_gt_fraction"])
            geometry = diagnostic["edit_visibility_evidence"]["geometry"]
            counts[category] += 1
            fractions.append(fraction)
            wrapped += int(bool(geometry["replicate_wrap_applied"]))
            starts[int(geometry["distinct_crop_starts"])] += 1
            all_counts[category] += 1
            all_fractions.append(fraction)
            all_wrap += int(bool(geometry["replicate_wrap_applied"]))
            all_starts[int(geometry["distinct_crop_starts"])] += 1
        by_condition[condition] = {
            "full": counts["full"],
            "partial": counts["partial"],
            "none": counts["none"],
            "total": len(condition_rows),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
            "replicate_wrap": wrapped,
            "distinct_crop_starts": {
                str(key): value for key, value in sorted(starts.items())
            },
        }
    return {
        "profile_id": legacy.PREPROCESS_PROFILE,
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
            "replicate_wrap": all_wrap,
            "distinct_crop_starts": {
                str(key): value
                for key, value in sorted(all_starts.items())
            },
        },
        "not_applicable_images": sum(
            row.get("gt_mask_kind") != "exact_diff" for row in rows
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


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, tuple[dict[str, Any], ...]]:
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = tuple(dict(row) for row in release.inputs)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
        or len(selected) != FORMAL_IMAGES
        or dict(counts) != FORMAL_COUNTS
        or any("pair_rank" in row for row in selected)
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("formal B-Free Balanced250 selection drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, tuple[dict[str, Any], ...]]:
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=SMOKE_PER_CONDITION,
    )
    inputs = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if counts[condition] >= SMOKE_PER_CONDITION:
            continue
        source = inputs.get(str(panel_row["sample_id"]))
        if source is None or source.get("panel") is not True:
            raise ValueError("smoke panel references a non-panel input")
        selected.append(dict(source))
        counts[condition] += 1
    selected.sort(key=lambda row: int(row["rank"]))
    expected_counts = {
        condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS
    }
    if (
        len(selected) != SMOKE_IMAGES
        or dict(counts) != expected_counts
        or any("pair_rank" in row for row in selected)
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != SMOKE5X7_SELECTED_IDS_SHA256
    ):
        raise ValueError("B-Free frozen 5x7 smoke selection drifted")
    return spec, tuple(selected)


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    dataset_contract = _require_mapping(
        immutable.get("dataset_contract"), "immutable.dataset_contract"
    )
    if set(dataset_contract) != {
        "schema_version",
        "release",
        "ledgers",
        "capability",
        "selection",
        "score_spec",
    }:
        raise ValueError("immutable dataset contract v2 key set changed")
    if (
        dataset_contract.get("schema_version")
        != EXPECTED_RUN_DATASET_CONTRACT_SCHEMA
    ):
        raise ValueError("immutable dataset contract schema changed")
    release_record = _require_mapping(
        dataset_contract.get("release"),
        "immutable.dataset_contract.release",
    )
    if set(release_record) != {
        "schema_version",
        "release_kind",
        "dataset_id",
        "manifest_path",
        "manifest_sha256",
        "contract_sha256",
    }:
        raise ValueError("immutable dataset release binding key set changed")
    if (
        release_record.get("schema_version") != BALANCED_SCHEMA
        or release_record.get("release_kind") != "balanced250"
        or release_record.get("dataset_id") != BALANCED_DATASET_ID
    ):
        raise ValueError("immutable dataset release identity changed")
    recorded_manifest_sha256 = _require_sha256(
        release_record.get("manifest_sha256"),
        "immutable dataset manifest SHA-256",
    )
    recorded_contract_sha256 = _require_sha256(
        release_record.get("contract_sha256"),
        "immutable dataset contract SHA-256",
    )
    manifest_path = _safe_repo_path(
        release_record.get("manifest_path"),
        repo_root=repo_root,
        label="Balanced250 dataset manifest",
    )
    release = load_canonical_release(
        repo_root, manifest_path, verify_files=True
    )
    if (
        release.manifest_sha256 != recorded_manifest_sha256
        or release.contract_sha256 != recorded_contract_sha256
    ):
        raise ValueError("immutable dataset release hashes changed")
    if expected_mode == "formal":
        spec, selected = _formal_selection(release)
    elif expected_mode == "smoke":
        spec, selected = _smoke_selection(release)
    else:
        raise ValueError("analyzer only accepts formal or frozen smoke runs")
    rebuilt = build_run_dataset_contract(
        release, spec, selected, score_spec=_score_spec()
    )
    if stable_json(rebuilt.as_dict()) != stable_json(dataset_contract):
        raise ValueError("immutable dataset contract is not independently reproducible")
    return release, selected, rebuilt


def _artifact_mapping(
    row: Mapping[str, Any], *, sample_id: str
) -> dict[str, Any]:
    artifact = _require_mapping(
        row.get("bfree_artifact"), f"{sample_id}.bfree_artifact"
    )
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
    if set(artifact) != expected_keys:
        raise ValueError(f"{sample_id} B-Free artifact key set changed")
    return artifact


def _load_npz_artifact(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    repo_root: Path,
    run_id: str,
) -> BFreeArtifact:
    artifact = _artifact_mapping(row, sample_id=sample_id)
    expected_relative = (
        DEFAULT_ARTIFACTS_DIR
        / run_id
        / "bfree_artifacts"
        / f"{sample_id}.npz"
    ).as_posix()
    if artifact.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} artifact path is not canonical")
    path = _safe_repo_path(
        expected_relative,
        repo_root=repo_root,
        label=f"{sample_id} B-Free artifact",
    )
    stat = path.stat()
    if stat.st_size != NPZ_FILE_BYTES:
        raise ValueError(f"{sample_id} NPZ byte size changed")
    file_sha = sha256_file(path)
    if (
        artifact.get("sha256") != file_sha
        or artifact.get("file_bytes") != NPZ_FILE_BYTES
    ):
        raise ValueError(f"{sample_id} NPZ file metadata changed")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if (
                [info.filename for info in infos]
                != ["features.npy", "crop_logits.npy"]
                or any(
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 0x1
                    or info.file_size != NPZ_MEMBER_BYTES[info.filename]
                    or info.compress_size != info.file_size
                    for info in infos
                )
            ):
                raise ValueError(f"{sample_id} NPZ member contract changed")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"{sample_id} NPZ container is malformed") from error
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ["features", "crop_logits"]:
                raise ValueError(f"{sample_id} NPZ keys/order changed")
            features = np.ascontiguousarray(payload["features"])
            crop_logits = np.ascontiguousarray(payload["crop_logits"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(sample_id):
            raise
        raise ValueError(f"{sample_id} NPZ arrays are unsafe/malformed") from error
    if (
        features.shape != FEATURE_SHAPE
        or crop_logits.shape != CROP_LOGITS_SHAPE
        or features.dtype != FEATURE_DTYPE
        or crop_logits.dtype != FEATURE_DTYPE
        or features.nbytes != FEATURE_NBYTES
        or crop_logits.nbytes != CROP_LOGITS_NBYTES
        or not features.flags.c_contiguous
        or not crop_logits.flags.c_contiguous
        or not np.isfinite(features).all()
        or not np.isfinite(crop_logits).all()
    ):
        raise ValueError(f"{sample_id} NPZ arrays violate shape/dtype/finite contract")
    feature_sha = _array_sha256(features)
    crop_sha = _array_sha256(crop_logits)
    if path.read_bytes() != _npz_bytes(features, crop_logits):
        raise ValueError(f"{sample_id} NPZ bytes are not canonical")
    expected_metadata = {
        "relative_path": expected_relative,
        "sha256": file_sha,
        "file_bytes": NPZ_FILE_BYTES,
        "feature_array_sha256": feature_sha,
        "crop_logits_array_sha256": crop_sha,
        "feature_shape": [5, 768],
        "crop_logits_shape": [5],
        "dtype": "float32",
        "feature_nbytes": FEATURE_NBYTES,
        "crop_logits_nbytes": CROP_LOGITS_NBYTES,
        "finite": True,
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
    }
    if stable_json(artifact) != stable_json(expected_metadata):
        raise ValueError(f"{sample_id} NPZ metadata/content hashes changed")
    aliases = {
        "bfree_artifact_path": expected_relative,
        "bfree_artifact_sha256": file_sha,
        "feature_array_sha256": feature_sha,
        "crop_logits_array_sha256": crop_sha,
        "feature_shape": [5, 768],
        "feature_dtype": "float32",
        "feature_nbytes": FEATURE_NBYTES,
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "crop_logits_shape": [5],
        "crop_logits_dtype": "float32",
        "crop_logits_nbytes": CROP_LOGITS_NBYTES,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
        "artifact_paths": {"bfree_npz": expected_relative},
    }
    for key, expected in aliases.items():
        if stable_json(row.get(key)) != stable_json(expected):
            raise ValueError(f"{sample_id} artifact alias {key} changed")
    score = _validate_score_payload(
        row,
        sample_id=sample_id,
        expected_crop_logits=crop_logits,
        tolerance=0.0,
    )
    if not math.isclose(
        _cpu_float32_crop_mean_sanity(crop_logits),
        score["raw_logit"],
        rel_tol=0.0,
        abs_tol=CROSS_DEVICE_MEAN_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"{sample_id} artifact crop logits fail cross-device mean sanity"
        )
    return BFreeArtifact(
        sample_id=sample_id,
        path=path,
        file_sha256=file_sha,
        file_bytes=NPZ_FILE_BYTES,
        feature_array_sha256=feature_sha,
        crop_logits_array_sha256=crop_sha,
        features=features,
        crop_logits=crop_logits,
    )


def validate_artifact_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
) -> dict[str, BFreeArtifact]:
    expected_dir = (
        repo_root.resolve()
        / DEFAULT_ARTIFACTS_DIR
        / run_id
        / "bfree_artifacts"
    ).resolve()
    if artifact_dir.resolve() != expected_dir:
        raise ValueError("B-Free artifact directory is not canonical")
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("B-Free artifact directory is missing or a symlink")
    artifacts: dict[str, BFreeArtifact] = {}
    for row in latest_results:
        sample_id = _require_string(row.get("sample_id"), "artifact sample_id")
        if row.get("status") != "ok":
            raise ValueError("artifact inventory requires successful latest rows")
        artifact = _load_npz_artifact(
            row,
            sample_id=sample_id,
            repo_root=repo_root,
            run_id=run_id,
        )
        if sample_id in artifacts or artifact.path in {
            value.path for value in artifacts.values()
        }:
            raise ValueError("B-Free artifact identity/path is reused")
        artifacts[sample_id] = artifact
    physical: set[Path] = set()
    for entry in artifact_dir.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".npz":
            raise ValueError("B-Free artifact inventory contains an extra entry")
        physical.add(entry.resolve())
    expected_paths = {artifact.path for artifact in artifacts.values()}
    if physical != expected_paths:
        raise ValueError("B-Free artifact inventory coverage changed")
    return artifacts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, BFreeArtifact],
) -> str:
    rows = [
        {
            "sample_id": sample_id,
            "relative_path": artifact.path.as_posix(),
            "file_sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "feature_array_sha256": artifact.feature_array_sha256,
            "crop_logits_array_sha256": artifact.crop_logits_array_sha256,
        }
        for sample_id, artifact in sorted(artifacts.items())
    ]
    return hashlib.sha256(stable_json(rows).encode()).hexdigest()


def _assert_runner_contract_exports() -> Any:
    """Check runner exports against analyzer-owned method constants."""

    runner = _runner()
    expected_scalars = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_FORMAL_RUN_ID": DEFAULT_RUN_ID,
        "DEFAULT_SMOKE_RUN_ID_A": DEFAULT_SMOKE_RUN_ID_A,
        "DEFAULT_SMOKE_RUN_ID_B": DEFAULT_SMOKE_RUN_ID_B,
        "FORMAL_SELECTED_ROWS_SHA256": FORMAL_SELECTED_ROWS_SHA256,
        "FORMAL_SELECTED_IDS_SHA256": FORMAL_SELECTED_IDS_SHA256,
        "SMOKE5X7_SELECTED_IDS_SHA256": SMOKE5X7_SELECTED_IDS_SHA256,
        "ARTIFACT_FILE_BYTES": NPZ_FILE_BYTES,
        "CPU_GOLDEN_SAMPLE_ID": "2c80d38ac19c2d3b76950996",
        "CPU_GOLDEN_IMAGE_SHA256": (
            "12607f3cdada1480038f3d506146cdc1fa0c1c50034afda5e3a5f175433e716b"
        ),
        "CPU_GOLDEN_DECODED_RGB_SHA256": (
            "5a4747a6e3a8313f8c9ec3dde2504bb53184666276d7e54dc5fab53ca0e7194b"
        ),
        "CPU_GOLDEN_TENSOR_SHA256": (
            "bf55e6ebe26e1da9ad303753a289830de9bda761766ea8bbbb4d4ad5cb938d2e"
        ),
        "CPU_GOLDEN_FEATURE_ARRAY_SHA256": (
            "c08c6452aabec2e9a842ea68ab2fbc91ef4612c033317bcc5a3060f0e67f73fc"
        ),
        "CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256": (
            "0372c0cf2fed737fe8e93bbcb6f561fecefd9723c9ba9b2451da580b51290ae0"
        ),
        "CPU_GOLDEN_ARTIFACT_SHA256": (
            "ad50cd98e2d66fe773d2f598385e0054ff56b5f5cfe86beb5ecfa09ff6e9b61d"
        ),
        "CPU_GOLDEN_RAW_LOGIT": -4.131394863128662,
        "CPU_GOLDEN_FAKE_PROBABILITY": 0.01580660045146942,
        "EXPECTED_ASSET_BUNDLE_SHA256": EXPECTED_ASSET_BUNDLE_SHA256,
    }
    missing = [name for name in expected_scalars if not hasattr(runner, name)]
    missing.extend(
        name
        for name in (
            "_OK_RESULT_FIELDS",
            "_ERROR_NULL_FIELDS",
            "LOCAL_VISIBILITY_CENSUS",
            "FORMAL_GEOMETRY_CENSUS",
            "MODEL_CONTRACT",
        )
        if not hasattr(runner, name)
    )
    if missing:
        raise ValueError(f"B-Free runner lacks frozen export {missing[0]}")
    for name, expected in expected_scalars.items():
        if not _same_json_type_and_value(getattr(runner, name), expected):
            raise ValueError(f"B-Free runner frozen export {name} drifted")
    if not _same_json_type_and_value(
        runner.SCORE_SPEC.as_dict(), _score_spec().as_dict()
    ):
        raise ValueError("B-Free runner score contract drifted")
    if tuple(runner.ADAPTER_SOURCE_PATHS) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("B-Free runner adapter source inventory drifted")
    if (
        set(runner._OK_RESULT_FIELDS) != EXPECTED_OK_RESULT_FIELDS
        or set(runner._ERROR_NULL_FIELDS) != EXPECTED_OK_RESULT_FIELDS
    ):
        raise ValueError("B-Free runner result-field schema drifted")
    if set(runner.IMMUTABLE_CONFIG_KEYS) != {
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
    }:
        raise ValueError("B-Free runner immutable config keys drifted")
    if not _same_json_type_and_value(
        runner.LOCAL_VISIBILITY_CENSUS, LOCAL_VISIBILITY_CENSUS
    ):
        raise ValueError("B-Free runner local visibility census drifted")
    if not _same_json_type_and_value(
        runner.FORMAL_GEOMETRY_CENSUS, FORMAL_GEOMETRY_CENSUS
    ):
        raise ValueError("B-Free runner geometry census drifted")
    if not _same_json_type_and_value(
        runner.MODEL_CONTRACT, EXPECTED_MODEL_CONTRACT
    ):
        raise ValueError("B-Free runner model contract drifted")
    expected_preprocess_contract = {
            "profile_id": legacy.PREPROCESS_PROFILE,
            "decoder": "Pillow.Image.open.convert_RGB",
            "exif_transpose": False,
            "icc_conversion": False,
            "resize": False,
            "tensor": "torchvision_ToTensor_uint8_div_255_float32",
            "normalization_mean": list(legacy.IMAGE_MEAN),
            "normalization_std": list(legacy.IMAGE_STD),
            "patch_projection_kernel": 14,
            "patch_projection_stride": 14,
            "right_bottom_remainder": "discarded",
            "replicate_wrap": (
                "if_either_grid_dimension_below_36_repeat_both_then_truncate_to_36"
            ),
            "crop_size_pixels": 504,
            "crop_count": 5,
            "crop_order": [
                "center",
                "top_left",
                "bottom_left",
                "bottom_right",
                "top_right",
            ],
            "batch_size": 1,
        }
    if not _same_json_type_and_value(
        runner.FROZEN_PREPROCESS_CONTRACT,
        expected_preprocess_contract,
    ):
        raise ValueError("B-Free runner preprocess contract drifted")
    expected_artifact_contract = {
        "format": "NumPy NPZ, allow_pickle=False, ZIP_STORED",
        "keys": ["features", "crop_logits"],
        "file_bytes": NPZ_FILE_BYTES,
        "zip_members": {
            "features.npy": {
                "compress_type": zipfile.ZIP_STORED,
                "file_size": NPZ_MEMBER_BYTES["features.npy"],
                "compress_size": NPZ_MEMBER_BYTES["features.npy"],
            },
            "crop_logits.npy": {
                "compress_type": zipfile.ZIP_STORED,
                "file_size": NPZ_MEMBER_BYTES["crop_logits.npy"],
                "compress_size": NPZ_MEMBER_BYTES["crop_logits.npy"],
            },
        },
        "features": {
            "shape": [5, 768],
            "dtype": "float32",
            "nbytes": FEATURE_NBYTES,
            "semantics": legacy.FEATURE_SEMANTICS,
        },
        "crop_logits": {
            "shape": [5],
            "dtype": "float32",
            "nbytes": CROP_LOGITS_NBYTES,
            "semantics": (
                "five_official_crop_raw_logits_in_official_crop_order"
            ),
        },
        "finite": True,
        "exact_same_device_head_replay": True,
        "visibility": "local_only_gitignored_output",
    }
    if not _same_json_type_and_value(
        runner.ARTIFACT_CONTRACT, expected_artifact_contract
    ):
        raise ValueError("B-Free runner artifact contract drifted")
    expected_task_scope = {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }
    if not _same_json_type_and_value(runner.TASK_SCOPE, expected_task_scope):
        raise ValueError("B-Free runner task boundary drifted")
    return runner


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    records = _require_mapping(value, "immutable.adapter_sources")
    if tuple(records) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("adapter source ordering/inventory changed")
    for relative in EXPECTED_ADAPTER_SOURCE_PATHS:
        record = _require_mapping(records.get(relative), f"adapter {relative}")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source record changed: {relative}")
        path = _safe_repo_path(
            record.get("path"),
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        if (
            record.get("path") != relative
            or isinstance(record.get("bytes"), bool)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"adapter source hash/size changed: {relative}")


def _validate_runtime_contract(value: Any, *, label: str) -> dict[str, Any]:
    """Validate runner schema plus analyzer-owned environment pins."""

    runner = _assert_runner_contract_exports()
    runtime = _require_mapping(value, label)
    runner.validate_runtime_contract(runtime, label=label)
    executable = _require_mapping(runtime.get("python"), f"{label}.python")
    if (
        executable.get("executable")
        != EXPECTED_FROZEN_PYTHON_EXECUTABLE.as_posix()
        or executable.get("version") != EXPECTED_RUNTIME_VERSIONS["python"]
    ):
        raise ValueError(f"{label} frozen Python runtime changed")
    venv = _require_mapping(runtime.get("venv"), f"{label}.venv")
    if (
        venv.get("prefix") != EXPECTED_FROZEN_VENV_PREFIX.as_posix()
        or venv.get("pyvenv_cfg_sha256")
        != EXPECTED_FROZEN_PYVENV_CONFIG_SHA256
        or venv.get("include_system_site_packages") is not True
    ):
        raise ValueError(f"{label} frozen venv evidence changed")
    packages = _require_mapping(runtime.get("packages"), f"{label}.packages")
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        if distribution in {"python"}:
            continue
        actual = packages.get(distribution)
        if distribution == "torch":
            actual = _require_mapping(actual, f"{label}.packages.torch").get(
                "version"
            )
        elif distribution == "torchvision":
            actual = _require_mapping(
                actual, f"{label}.packages.torchvision"
            ).get("version")
        if actual != expected:
            raise ValueError(f"{label} package {distribution} changed")
    torch_package = _require_mapping(
        packages.get("torch"), f"{label}.packages.torch"
    )
    torchvision_package = _require_mapping(
        packages.get("torchvision"), f"{label}.packages.torchvision"
    )
    if (
        torch_package.get("distribution_version")
        != EXPECTED_TORCH_DISTRIBUTION_VERSION
        or torchvision_package.get("distribution_version")
        != EXPECTED_TORCHVISION_DISTRIBUTION_VERSION
    ):
        raise ValueError(f"{label} torch distribution versions changed")
    expected_environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPYCACHEPREFIX": (
            EXPECTED_FROZEN_PYTHONPYCACHEPREFIX.as_posix()
        ),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": (
            EXPECTED_FROZEN_PYTHONPYCACHEPREFIX.as_posix()
        ),
        "pycache_prefix_initially_empty": True,
    }
    if not _same_json_type_and_value(
        runtime.get("process_environment"), expected_environment
    ):
        raise ValueError(f"{label} startup isolation changed")
    _validate_frozen_module_hashes()
    if (
        not _same_json_type_and_value(
            runtime.get("seed"), EXPECTED_RUNTIME_SEED
        )
        or runtime.get("preprocess_profile") != legacy.PREPROCESS_PROFILE
        or runtime.get("inference_dtype") != "float32"
        or runtime.get("feature_dtype") != "float32"
        or not _same_json_type_and_value(runtime.get("batch_size"), 1)
        or runtime.get("autocast") is not False
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("matmul_allow_tf32") is not False
        or runtime.get("cublas_workspace_config")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ValueError(f"{label} deterministic runtime changed")
    if not _same_json_type_and_value(
        runtime.get("cudnn"),
        {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
    ):
        raise ValueError(f"{label} cuDNN runtime changed")
    return runtime


def _validate_frozen_module_hashes() -> dict[str, dict[str, str]]:
    """Independently pin import entry points omitted from runner evidence."""

    evidence: dict[str, dict[str, str]] = {}
    for name, expected_hash in EXPECTED_MODULE_HASHES.items():
        module = importlib.import_module(name)
        path_value = getattr(module, "__file__", None)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"B-Free module {name} lacks a source path")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"B-Free module {name} path is unsafe")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"B-Free module {name} hash changed")
        evidence[name] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
        }
    return evidence


def _verify_source_assets(
    *,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
    recorded_source: Mapping[str, Any],
    recorded_assets: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    """Verify assets via independent legacy audit and compare runner evidence."""

    import torch

    source = legacy_audit._verify_source_tree(source_root)
    assets, state = legacy_audit._verify_assets(
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        torch_module=torch,
    )
    if (
        source["commit"] != EXPECTED_SOURCE_COMMIT
        or sha256_file(weights_zip) != EXPECTED_ZIP_SHA256
        or weights_zip.stat().st_size != EXPECTED_ZIP_BYTES
        or sha256_file(weights_dir / "config.yaml") != EXPECTED_CONFIG_SHA256
        or (weights_dir / "config.yaml").stat().st_size
        != EXPECTED_CONFIG_BYTES
        or sha256_file(weights_dir / "model_epoch_best.pth")
        != EXPECTED_CHECKPOINT_SHA256
        or (weights_dir / "model_epoch_best.pth").stat().st_size
        != EXPECTED_CHECKPOINT_BYTES
        or assets["checkpoint"]["tensor_count"]
        != EXPECTED_CHECKPOINT_TENSORS
        or assets["checkpoint"]["state_elements"]
        != EXPECTED_CHECKPOINT_ELEMENTS
        or assets["checkpoint"]["schema_items_sha256"]
        != EXPECTED_CHECKPOINT_SCHEMA_SHA256
    ):
        raise ValueError("independent B-Free source/asset pins changed")
    if (
        recorded_source.get("commit") != EXPECTED_SOURCE_COMMIT
        or Path(str(recorded_source.get("root"))).resolve()
        != source_root.resolve()
    ):
        raise ValueError("recorded B-Free source differs from independent source")
    recorded_checkpoint = _require_mapping(
        recorded_assets.get("checkpoint"), "immutable.assets.checkpoint"
    )
    recorded_zip = _require_mapping(
        recorded_assets.get("zip"), "immutable.assets.zip"
    )
    recorded_config = _require_mapping(
        recorded_assets.get("config"), "immutable.assets.config"
    )
    if (
        recorded_zip.get("sha256") != EXPECTED_ZIP_SHA256
        and recorded_zip.get("verified_sha256") != EXPECTED_ZIP_SHA256
    ):
        raise ValueError("recorded B-Free ZIP identity changed")
    if recorded_config.get("sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("recorded B-Free config identity changed")
    if (
        recorded_checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or recorded_checkpoint.get("schema", {}).get("tensor_count")
        != EXPECTED_CHECKPOINT_TENSORS
        or recorded_checkpoint.get("schema", {}).get("state_elements")
        != EXPECTED_CHECKPOINT_ELEMENTS
    ):
        raise ValueError("recorded B-Free checkpoint identity changed")
    return source, assets, state


def _validate_cpu_preflight(
    value: Any,
    *,
    repo_root: Path,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    runner = _assert_runner_contract_exports()
    wrapper = _require_mapping(value, "immutable.cpu_preflight")
    if set(wrapper) != {
        "performed_before_dataset_manifest_load",
        "performed_before_accelerator_configuration",
        "report",
    } or (
        wrapper.get("performed_before_dataset_manifest_load") is not True
        or wrapper.get("performed_before_accelerator_configuration") is not True
    ):
        raise ValueError("B-Free CPU preflight ordering evidence changed")
    report = _require_mapping(
        wrapper.get("report"), "immutable.cpu_preflight.report"
    )
    runner._validate_preflight_report(
        report,
        source=source,
        assets=assets,
    )
    if (
        report.get("schema_version") != EXPECTED_CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or not _same_json_type_and_value(report.get("source"), source)
        or not _same_json_type_and_value(report.get("assets"), assets)
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("B-Free CPU preflight provenance/order changed")
    runtime = _validate_runtime_contract(
        report.get("runtime"), label="CPU preflight runtime"
    )
    if runtime.get("device") != "cpu":
        raise ValueError("B-Free CPU preflight did not use CPU")
    official = _require_mapping(
        report.get("official_golden"), "CPU preflight official golden"
    )
    cases = _require_list(official.get("cases"), "official golden cases")
    if (
        official.get("status") != "passed"
        or len(cases) != 4
        or official.get("absolute_tolerance")
        != legacy.GOLDEN_ABS_TOLERANCE
        or official.get("runtime_regression_absolute_tolerance")
        != legacy.GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE
        or any(case.get("passed") is not True for case in cases)
    ):
        raise ValueError("B-Free official CPU golden changed")
    balanced = _require_mapping(
        report.get("balanced_golden"), "CPU preflight balanced golden"
    )
    golden_path = _safe_repo_path(
        runner.CPU_GOLDEN_INPUT_PATH,
        repo_root=repo_root,
        label="B-Free Balanced CPU golden input",
    )
    prepared = legacy_audit.preprocess_image(
        golden_path, torch_module=__import__("torch")
    )
    expected = {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": prepared.audit,
        "artifact_sha256": runner.CPU_GOLDEN_ARTIFACT_SHA256,
        "artifact_bytes": NPZ_FILE_BYTES,
        "feature_array_sha256": runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "crop_logits_array_sha256": (
            runner.CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256
        ),
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "ai_score": runner.CPU_GOLDEN_RAW_LOGIT,
        "fake_probability": runner.CPU_GOLDEN_FAKE_PROBABILITY,
        "crop_logits": runner.CPU_GOLDEN_CROP_LOGITS,
        "classification_decision": False,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
        "peak_cuda_memory_bytes": 0,
        "repeat_artifact_sha256": runner.CPU_GOLDEN_ARTIFACT_SHA256,
        "repeat_feature_array_sha256": (
            runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
        ),
        "repeat_crop_logits_array_sha256": (
            runner.CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256
        ),
        "repeat_raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "repeat_ai_score": runner.CPU_GOLDEN_RAW_LOGIT,
        "repeat_fake_probability": runner.CPU_GOLDEN_FAKE_PROBABILITY,
        "repeat_crop_logits": runner.CPU_GOLDEN_CROP_LOGITS,
        "repeat_classification_decision": False,
        "repeat_model_forward_calls": 1,
        "repeat_classifier_hook_calls": 1,
        "repeat_byte_exact": True,
    }
    if stable_json(balanced) != stable_json(expected):
        raise ValueError("B-Free Balanced CPU golden changed")


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, dict[str, Any]]:
    runner = _assert_runner_contract_exports()
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
        "execution",
    }
    if set(manifest) != expected_keys:
        raise ValueError("B-Free run manifest key set changed")
    if (
        manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != _valid_run_id(run_id)
        or manifest.get("status") != "complete"
    ):
        raise ValueError("analyzer requires an exact complete B-Free run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest.immutable")
    if set(immutable) != set(runner.IMMUTABLE_CONFIG_KEYS):
        raise ValueError("B-Free immutable config key set changed")
    if (
        immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
    ):
        raise ValueError("B-Free immutable run identity changed")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"), "manifest.fingerprint"
    )
    expected_fingerprint = hashlib.sha256(
        stable_json(immutable).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("B-Free manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"), repo_root=repo_root
    )
    if any(
        not _same_json_type_and_value(actual, expected)
        for actual, expected in (
            (immutable.get("model"), EXPECTED_MODEL_CONTRACT),
            (
                immutable.get("preprocess"),
                runner.FROZEN_PREPROCESS_CONTRACT,
            ),
            (
                immutable.get("artifact_contract"),
                runner.ARTIFACT_CONTRACT,
            ),
            (immutable.get("task_scope"), runner.TASK_SCOPE),
            (immutable.get("score_spec"), _score_spec().as_dict()),
            (
                immutable.get("formal_local_visibility_census"),
                runner.LOCAL_VISIBILITY_CENSUS,
            ),
            (
                immutable.get("formal_geometry_census"),
                runner.FORMAL_GEOMETRY_CENSUS,
            ),
        )
    ):
        raise ValueError("B-Free immutable method contract changed")
    source = _require_mapping(immutable.get("source"), "immutable.source")
    assets = _require_mapping(immutable.get("assets"), "immutable.assets")
    if (
        source.get("commit") != EXPECTED_SOURCE_COMMIT
        or assets.get("bundle_sha256") != EXPECTED_ASSET_BUNDLE_SHA256
    ):
        raise ValueError("B-Free immutable source/asset identity changed")
    _validate_runtime_contract(
        immutable.get("runtime"), label="immutable.runtime"
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        repo_root=repo_root,
        source=source,
        assets=assets,
    )
    runner._validate_model_load(
        immutable.get("execution_model_load"),
        label="execution-device model load",
    )
    runner._validate_official_golden(
        immutable.get("execution_official_golden"),
        label="execution-device official four-demo golden",
    )
    expected_policy = runner._local_artifact_policy(repo_root)
    if not _same_json_type_and_value(
        immutable.get("local_artifact_policy"), expected_policy
    ):
        raise ValueError("B-Free local artifact policy changed")
    outputs = _require_mapping(immutable.get("outputs"), "immutable.outputs")
    if set(outputs) != {
        "run_dir",
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "artifact_dir",
    }:
        raise ValueError("B-Free immutable output contract changed")
    for field, value in outputs.items():
        _safe_repo_path(
            value,
            repo_root=repo_root,
            label=f"immutable.outputs.{field}",
            require_file=False,
        )
    _reject_unsupported_claims(manifest, "manifest")
    _reject_nonfinite_numbers(manifest, "manifest")
    return fingerprint, immutable


def _validate_dataset_artifacts(
    *,
    manifest: Mapping[str, Any],
    repo_root: Path,
    release: CanonicalRelease,
    selected: Sequence[Mapping[str, Any]],
    contract: RunDatasetContract,
    expected_path: Path,
) -> None:
    expected_rows = _read_jsonl_strict(expected_path, "expected inputs")
    if stable_json(expected_rows) != stable_json(list(selected)):
        raise ValueError("expected-input snapshot is not the exact selection")
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    expected = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    if stable_json(dataset) != stable_json(expected):
        raise ValueError("B-Free manifest dataset evidence changed")


def _validate_physical_attempts(
    *,
    physical: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    runner = _assert_runner_contract_exports()
    runner._validate_physical_attempt_history(physical)
    inputs = {str(row["sample_id"]): row for row in selected}
    for index, row in enumerate(physical):
        sample_id = _require_string(
            row.get("sample_id"), f"physical result {index}.sample_id"
        )
        input_row = inputs.get(sample_id)
        if input_row is None:
            raise ValueError(f"physical result {index} is unexpected")
        runner._validate_runner_attempt(
            row,
            input_row=input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        expected_visibility = _independent_visibility_diagnostic(
            input_row, repo_root=repo_root
        )
        for key, expected in expected_visibility.items():
            if stable_json(row.get(key)) != stable_json(expected):
                raise ValueError(f"{sample_id} visibility evidence changed")
        if row.get("preprocess_profile") != legacy.PREPROCESS_PROFILE:
            raise ValueError(f"{sample_id} preprocess profile changed")
        _reject_unsupported_claims(row, f"physical result {index}")
        _reject_nonfinite_numbers(row, f"physical result {index}")
        if row.get("status") == "ok":
            _validate_score_payload(row, sample_id=sample_id)
        else:
            if row.get("valid_for_metrics") is not False:
                raise ValueError(f"{sample_id} error result validity changed")
            for key in EXPECTED_OK_RESULT_FIELDS:
                if key not in row or row[key] is not None:
                    raise ValueError(f"{sample_id} error field {key} must be null")


def _latest_in_selection_order(
    *,
    selected: Sequence[Mapping[str, Any]],
    physical: Sequence[Mapping[str, Any]],
    run_id: str,
    fingerprint: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    latest = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=_score_spec(),
    )
    coverage = summarize_coverage(latest)
    require_complete_coverage(coverage)
    ordered = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])])
        for row in selected
    )
    return ordered, coverage.as_dict()


def _validate_execution(
    *,
    manifest: Mapping[str, Any],
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(
        manifest.get("execution"), "manifest.execution"
    )
    expected_keys = {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
        "same_device_artifact_head_replays",
    }
    if set(execution) != expected_keys:
        raise ValueError("B-Free execution key set changed")
    for key, value in execution.items():
        _require_nonnegative_int(value, f"execution.{key}")
    if (
        execution["physical_result_rows"] != physical_rows
        or execution["latest_result_rows"] != latest_rows
        or execution["superseded_attempts"] != physical_rows - latest_rows
        or execution["same_device_artifact_head_replays"] != latest_rows
        or execution["new_errors"] != 0
        or execution["new_successes"] + execution["resume_skips"]
        != selected_images
    ):
        raise ValueError("B-Free execution accounting changed")


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    visibility: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_bfree_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "score_spec": _score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
        "selection_visibility_census": dict(visibility),
        "same_device_artifact_head_replays": coverage["valid_images"],
        "coverage": dict(coverage),
    }
    if set(summary) != {*required, "generated_at"}:
        raise ValueError("B-Free runtime summary key set changed")
    for key, expected in required.items():
        if stable_json(summary.get(key)) != stable_json(expected):
            raise ValueError(f"B-Free runtime summary {key} changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    _reject_unsupported_claims(summary, "summary")
    _reject_nonfinite_numbers(summary, "summary")


def _capture_evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    artifacts: Mapping[str, BFreeArtifact],
    primary_snapshot: Mapping[str, str] | None = None,
) -> dict[str, str]:
    current = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if primary_snapshot is not None and current != dict(primary_snapshot):
        raise ValueError("B-Free run evidence changed during validation")
    for artifact in artifacts.values():
        if (
            artifact.path.is_symlink()
            or not artifact.path.is_file()
            or artifact.path.stat().st_size != artifact.file_bytes
            or sha256_file(artifact.path) != artifact.file_sha256
        ):
            raise ValueError("B-Free NPZ evidence changed during validation")
    return {
        **current,
        "artifact_inventory_sha256": _artifact_inventory_sha256(artifacts),
    }


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    mode: str,
) -> RunBundle:
    _assert_runner_contract_exports()
    root = repo_root.resolve()
    results_root = _resolve_results_root(results_dir, root)
    run_dir = _resolve_run_dir(results_root, run_id)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for label, path in (
        ("manifest", manifest_path),
        ("results", results_path),
        ("expected inputs", expected_path),
        ("summary", summary_path),
    ):
        _require_regular_file(path, f"B-Free {label}")
    primary = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    manifest = _load_json(manifest_path, f"{mode} manifest")
    fingerprint, immutable = _validate_manifest(
        manifest=manifest,
        repo_root=root,
        run_id=run_id,
        expected_mode=mode,
    )
    release, selected, contract = _rebuild_contract(
        repo_root=root,
        immutable=immutable,
        expected_mode=mode,
    )
    visibility = _independent_visibility_census(
        selected, repo_root=root
    )
    if stable_json(immutable.get("selection_visibility_census")) != stable_json(
        visibility
    ):
        raise ValueError("B-Free selection visibility census changed")
    _validate_dataset_artifacts(
        manifest=manifest,
        repo_root=root,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
    )
    physical = tuple(
        _read_jsonl_strict(results_path, f"{mode} physical results")
    )
    _validate_physical_attempts(
        physical=physical,
        selected=selected,
        repo_root=root,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    latest, coverage = _latest_in_selection_order(
        selected=selected,
        physical=physical,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    _validate_execution(
        manifest=manifest,
        selected_images=len(selected),
        physical_rows=len(physical),
        latest_rows=len(latest),
    )
    if mode == "smoke" and len(physical) != SMOKE_IMAGES:
        raise ValueError("B-Free smoke must have one attempt per image")
    immutable_outputs = _require_mapping(
        immutable.get("outputs"), "immutable.outputs"
    )
    artifact_dir = _safe_repo_path(
        immutable_outputs.get("artifact_dir"),
        repo_root=root,
        label="B-Free artifact directory",
        require_file=False,
    )
    canonical_artifact_dir = (
        root / DEFAULT_ARTIFACTS_DIR / run_id / "bfree_artifacts"
    ).resolve()
    if artifact_dir != canonical_artifact_dir:
        raise ValueError("B-Free artifact directory is not canonical")
    outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    expected_outputs = {
        "run_dir": repo_relative(run_dir, root),
        "results_path": repo_relative(results_path, root),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "summary_sha256": sha256_file(summary_path),
        "artifact_dir": repo_relative(artifact_dir, root),
        "artifact_files": len(selected),
    }
    if stable_json(outputs) != stable_json(expected_outputs):
        raise ValueError("B-Free manifest output evidence changed")
    summary = _load_json(summary_path, f"{mode} summary")
    _validate_summary(
        summary=summary,
        mode=mode,
        run_id=run_id,
        fingerprint=fingerprint,
        contract=contract,
        visibility=visibility,
        coverage=coverage,
    )
    artifacts = validate_artifact_inventory(
        latest_results=latest,
        repo_root=root,
        artifact_dir=artifact_dir,
        run_id=run_id,
    )
    snapshot = _capture_evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        artifacts=artifacts,
        primary_snapshot=primary,
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        mode=mode,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest=manifest,
        immutable=immutable,
        release=release,
        selected=selected,
        contract=contract,
        physical_results=physical,
        latest_results=latest,
        coverage=coverage,
        artifact_dir=artifact_dir,
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *, repo_root: Path, results_dir: Path, run_id: str
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        mode="formal",
    )


def load_smoke_run(
    *, repo_root: Path, results_dir: Path, run_id: str
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        mode="smoke",
    )


def _verify_bundle_unchanged(
    bundle: RunBundle, *, repo_root: Path
) -> None:
    expected = dict(bundle.evidence_snapshot)
    for key, path in (
        ("manifest_sha256", bundle.manifest_path),
        ("results_sha256", bundle.results_path),
        ("expected_inputs_sha256", bundle.expected_path),
        ("summary_sha256", bundle.summary_path),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected[key]
        ):
            raise ValueError(f"B-Free evidence changed after validation: {key}")
    artifacts = validate_artifact_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        artifact_dir=bundle.artifact_dir,
        run_id=bundle.run_id,
    )
    if (
        _artifact_inventory_sha256(artifacts)
        != expected["artifact_inventory_sha256"]
    ):
        raise ValueError("B-Free artifact evidence changed after validation")
    release, selected, contract = _rebuild_contract(
        repo_root=repo_root,
        immutable=bundle.immutable,
        expected_mode=bundle.mode,
    )
    if (
        stable_json(selected) != stable_json(bundle.selected)
        or stable_json(contract.as_dict())
        != stable_json(bundle.contract.as_dict())
        or release.manifest_sha256 != bundle.release.manifest_sha256
    ):
        raise ValueError("B-Free canonical selection changed after validation")


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "B-Free Balanced250 metrics require iterations=1000 "
            "and seed=20260726"
        )
    metrics = summarize_balanced250_t1(
        bundle.release.inputs,
        bundle.release.panel,
        bundle.release.source_pairs,
        bundle.latest_results,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        iterations=iterations,
        seed=seed,
    )
    if (
        metrics.get("schema_version") != METRICS_SCHEMA_VERSION
        or metrics.get("coverage", {}).get("is_complete") is not True
    ):
        raise ValueError("shared Balanced250 T1 metrics are incomplete")
    return metrics


def _configure_exact_recorded_runtime(
    *,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
    label: str,
) -> tuple[Any, dict[str, Any]]:
    if device_text != recorded_runtime.get("device"):
        raise ValueError(f"{label} device differs from recorded runtime")
    runner = _assert_runner_contract_exports()
    configured = runner.configure_runtime(
        device_text, seed=EXPECTED_RUNTIME_SEED
    )
    if (
        not isinstance(configured, tuple)
        or len(configured) != 2
        or not isinstance(configured[1], Mapping)
    ):
        raise ValueError("B-Free runtime constructor contract changed")
    device, runtime = configured
    current = _validate_runtime_contract(runtime, label=f"{label} current runtime")
    if stable_json(current) != stable_json(recorded_runtime):
        raise ValueError(f"{label} current runtime differs from recorded runtime")
    return device, dict(current)


def _build_independent_model(
    *,
    source_root: Path,
    state: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    model, evidence = legacy_audit._build_and_load_model(
        source_root=source_root,
        state=state,
        torch_module=torch,
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, evidence


def replay_persisted_head(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, BFreeArtifact],
    model: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    runtime = legacy_audit.ReplayRuntime(
        torch=torch,
        device=device,
        evidence={"device": str(device)},
    )
    maximum_crop = 0.0
    maximum_mean = 0.0
    decisions = 0
    for row in latest_results:
        sample_id = _require_string(row.get("sample_id"), "head replay sample")
        artifact = artifacts.get(sample_id)
        if artifact is None:
            raise ValueError(f"missing B-Free artifact for {sample_id}")
        replay_logits, replay_mean = legacy_audit._replay_artifact_head(
            artifact.features, model, runtime
        )
        crop_difference = float(
            np.max(
                np.abs(
                    replay_logits.astype(np.float64)
                    - artifact.crop_logits.astype(np.float64)
                ),
                initial=0.0,
            )
        )
        mean_difference = abs(
            replay_mean
            - _require_finite(row.get("raw_logit"), f"{sample_id}.raw_logit")
        )
        maximum_crop = max(maximum_crop, crop_difference)
        maximum_mean = max(maximum_mean, mean_difference)
        if (
            crop_difference > PERSISTED_HEAD_ABS_TOLERANCE
            or mean_difference > PERSISTED_HEAD_ABS_TOLERANCE
        ):
            raise ValueError(f"{sample_id} persisted head replay changed")
        score = _validate_score_payload(
            row,
            sample_id=sample_id,
            expected_crop_logits=replay_logits,
            expected_raw_logit=replay_mean,
            tolerance=PERSISTED_HEAD_ABS_TOLERANCE,
        )
        if score["decision"] is not (replay_mean > 0.0):
            raise ValueError(f"{sample_id} persisted head decision changed")
        decisions += 1
    return {
        "artifacts_replayed": len(latest_results),
        "linear_head_replays": len(latest_results),
        "strict_decisions_replayed": decisions,
        "maximum_crop_logit_absolute_difference": maximum_crop,
        "maximum_raw_logit_absolute_difference": maximum_mean,
        "crop_logit_absolute_tolerance": PERSISTED_HEAD_ABS_TOLERANCE,
        "raw_logit_absolute_tolerance": PERSISTED_HEAD_ABS_TOLERANCE,
        "static_cross_device_mean_sanity_absolute_tolerance": (
            CROSS_DEVICE_MEAN_ABS_TOLERANCE
        ),
        "all_passed": True,
    }


def _canonical_input_path(
    row: Mapping[str, Any], *, repo_root: Path
) -> Path:
    path = _safe_repo_path(
        row.get("canonical_path"),
        repo_root=repo_root,
        label=f"{row.get('sample_id')} canonical input",
    )
    if sha256_file(path) != row.get("canonical_sha256"):
        raise ValueError("canonical input SHA-256 changed before fresh replay")
    with Image.open(path) as opened:
        if (
            opened.size != (int(row["width"]), int(row["height"]))
            or opened.mode != "RGB"
        ):
            raise ValueError("canonical input decode contract changed")
    return path


def replay_model(
    bundle: RunBundle,
    *,
    repo_root: Path,
    source_root: Path,
    model: Any,
    device: Any,
) -> dict[str, Any]:
    """Freshly replay all 1,775 canonical JPEGs through official B-Free."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh B-Free replay requires the full 1,775 selection")
    import torch

    runtime = legacy_audit.ReplayRuntime(
        torch=torch,
        device=device,
        evidence={"device": str(device)},
    )
    maximum_feature = 0.0
    maximum_crop = 0.0
    maximum_raw = 0.0
    wrap_images = 0
    for input_row, result_row in zip(
        bundle.selected, bundle.latest_results, strict=True
    ):
        sample_id = str(input_row["sample_id"])
        if result_row.get("sample_id") != sample_id:
            raise ValueError("fresh replay order/identity changed")
        path = _canonical_input_path(input_row, repo_root=repo_root)
        prepared = legacy_audit.preprocess_image(
            path, torch_module=torch
        )
        if stable_json(result_row.get("preprocess")) != stable_json(
            prepared.audit
        ):
            raise ValueError(f"{sample_id} preprocessing evidence changed")
        geometry = prepared.audit["geometry"]
        wrap_images += int(bool(geometry["replicate_wrap_applied"]))
        forward = legacy_audit._forward_with_evidence(
            model, prepared.tensor, runtime
        )
        artifact = bundle.artifacts[sample_id]
        feature_difference = float(
            np.max(
                np.abs(
                    forward.features.astype(np.float64)
                    - artifact.features.astype(np.float64)
                ),
                initial=0.0,
            )
        )
        crop_difference = float(
            np.max(
                np.abs(
                    forward.crop_logits.astype(np.float64)
                    - artifact.crop_logits.astype(np.float64)
                ),
                initial=0.0,
            )
        )
        raw_difference = abs(
            forward.raw_logit
            - _require_finite(
                result_row.get("raw_logit"), f"{sample_id}.raw_logit"
            )
        )
        maximum_feature = max(maximum_feature, feature_difference)
        maximum_crop = max(maximum_crop, crop_difference)
        maximum_raw = max(maximum_raw, raw_difference)
        if (
            feature_difference > FEATURE_ABS_TOLERANCE
            or crop_difference > CROP_LOGIT_ABS_TOLERANCE
            or raw_difference > RAW_LOGIT_ABS_TOLERANCE
        ):
            raise ValueError(f"{sample_id} fresh full-model replay changed")
        _validate_score_payload(
            result_row,
            sample_id=sample_id,
            expected_crop_logits=forward.crop_logits,
            expected_raw_logit=forward.raw_logit,
            tolerance=RAW_LOGIT_ABS_TOLERANCE,
        )
    if wrap_images != 165:
        raise ValueError("B-Free fresh replay wrap census changed")
    return {
        "selected_images_freshly_reopened": FORMAL_IMAGES,
        "selected_images_freshly_preprocessed": FORMAL_IMAGES,
        "complete_model_forward_passes": FORMAL_IMAGES,
        "features_compared": FORMAL_IMAGES,
        "crop_logits_compared": FORMAL_IMAGES,
        "raw_logits_compared": FORMAL_IMAGES,
        "wrap_images": wrap_images,
        "maximum_feature_absolute_difference": maximum_feature,
        "maximum_crop_logit_absolute_difference": maximum_crop,
        "maximum_raw_logit_absolute_difference": maximum_raw,
        "feature_absolute_tolerance": FEATURE_ABS_TOLERANCE,
        "crop_logit_absolute_tolerance": CROP_LOGIT_ABS_TOLERANCE,
        "raw_logit_absolute_tolerance": RAW_LOGIT_ABS_TOLERANCE,
        "all_passed": True,
        "scope": {
            "T1_whole_image_AIGC_detection": True,
            "T2_localization": False,
            "joint_score": False,
        },
    }


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing = _SMOKE_IGNORED_FIELDS - set(row)
    if missing:
        raise ValueError(f"{sample_id} lacks runtime field {sorted(missing)[0]}")
    projection = {
        key: value
        for key, value in row.items()
        if key not in _SMOKE_IGNORED_FIELDS
    }
    artifact = _artifact_mapping(row, sample_id=sample_id)
    projection["bfree_artifact"] = {
        key: value
        for key, value in artifact.items()
        if key not in _ARTIFACT_VOLATILE_FIELDS
    }
    for key in (
        "bfree_artifact_path",
        "bfree_artifact_sha256",
        "feature_array_sha256",
        "crop_logits_array_sha256",
        "artifact_paths",
    ):
        projection.pop(key, None)
    return projection


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_artifacts: Mapping[str, BFreeArtifact],
    replay_artifacts: Mapping[str, BFreeArtifact],
) -> dict[str, Any]:
    def unique(
        rows: Sequence[Mapping[str, Any]], label: str
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"), f"{label}[{index}].sample_id"
            )
            if sample_id in result:
                raise ValueError(f"{label} repeats {sample_id}")
            result[sample_id] = row
        if not result:
            raise ValueError(f"{label} is empty")
        return result

    reference = unique(reference_rows, "reference")
    replay = unique(replay_rows, "replay")
    if (
        set(reference) != set(replay)
        or set(reference_artifacts) != set(reference)
        or set(replay_artifacts) != set(replay)
    ):
        raise ValueError("B-Free smoke coverage differs")
    for sample_id in sorted(reference):
        if stable_json(
            _exact_smoke_projection(reference[sample_id])
        ) != stable_json(_exact_smoke_projection(replay[sample_id])):
            raise ValueError(f"B-Free smoke result differs for {sample_id}")
        left = reference_artifacts[sample_id]
        right = replay_artifacts[sample_id]
        if (
            left.path.read_bytes() != right.path.read_bytes()
            or left.file_sha256 != right.file_sha256
            or left.file_bytes != right.file_bytes
            or left.feature_array_sha256 != right.feature_array_sha256
            or left.crop_logits_array_sha256
            != right.crop_logits_array_sha256
            or not np.array_equal(left.features, right.features)
            or not np.array_equal(left.crop_logits, right.crop_logits)
        ):
            raise ValueError(f"B-Free smoke NPZ bytes/arrays differ for {sample_id}")
    return {
        "images_compared": len(reference),
        "ignored_result_fields": sorted(_SMOKE_IGNORED_FIELDS),
        "ignored_artifact_metadata_fields": sorted(
            _ARTIFACT_VOLATILE_FIELDS
        ),
        "exact_computational_projection": True,
        "npz_file_bytes_exact": True,
        "feature_arrays_exact": True,
        "crop_logit_arrays_exact": True,
        "maximum_raw_logit_absolute_difference": 0.0,
        "maximum_feature_absolute_difference": 0.0,
        "maximum_crop_logit_absolute_difference": 0.0,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any]
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    if set(immutable) != set(runner.IMMUTABLE_CONFIG_KEYS):
        raise ValueError("B-Free smoke immutable key set changed")
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def _analysis_runtime_contract() -> dict[str, Any]:
    return {
        "python": {
            "version": ".".join(
                str(value)
                for value in __import__("sys").version_info[:3]
            ),
            "executable": str(
                Path(os.path.abspath(__import__("sys").executable))
            ),
        },
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "analyzer": "eval/opensource/analyze_bfree_balanced.py",
    }


def _json_artifact_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _verify_json_artifact(
    path: Path, *, expected_sha256: str, label: str
) -> None:
    _reject_output_symlink_components(path, label=label)
    if (
        not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} changed after write")


def _write_json_verified(path: Path, value: Any, *, label: str) -> None:
    expected = _json_artifact_sha256(value)
    _reject_output_symlink_components(path, label=label)
    atomic_write_json(path, value)
    _verify_json_artifact(path, expected_sha256=expected, label=label)


def _lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    """Normalize without resolving symlinks, preserving evidence for rejection."""

    raw = path
    if not raw.is_absolute():
        raw = (base if base is not None else Path.cwd()) / raw
    return Path(os.path.abspath(raw))


def _reject_output_symlink_components(path: Path, *, label: str) -> None:
    """Reject a symlink in any existing component of an output path."""

    absolute = _lexical_absolute(path)
    if not absolute.is_absolute():
        raise ValueError(f"{label} output path is not absolute")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} output path contains a symlink")
    if absolute.exists() and not absolute.is_file():
        raise ValueError(f"{label} output path is not a regular file")


def _validate_formal_output_scope(
    *,
    repo_root: Path,
    run_dir: Path,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
) -> dict[str, Path | None]:
    """Authorize only the two canonical report files in the current run."""

    expected = {
        "metrics": _lexical_absolute(
            run_dir / "balanced250_metrics.json", base=repo_root
        ),
        "audit": _lexical_absolute(
            run_dir / "independent_audit.json", base=repo_root
        ),
    }
    requested = {
        "metrics": metrics_output_path,
        "audit": audit_output_path,
    }
    authorized: dict[str, Path | None] = {}
    for name, path in requested.items():
        if path is None:
            authorized[name] = None
            continue
        candidate = _lexical_absolute(path, base=repo_root)
        if candidate != expected[name]:
            raise ValueError(
                f"B-Free formal {name} output must be the canonical "
                f"current-run file {expected[name]}"
            )
        _reject_output_symlink_components(
            candidate, label=f"formal {name}"
        )
        authorized[name] = candidate
    return authorized


def _smoke_comparison_default_path(
    *,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    reference = _valid_run_id(reference_run_id)
    replay = _valid_run_id(replay_run_id)
    fingerprint = hashlib.sha256(
        stable_json([reference, replay]).encode()
    ).hexdigest()
    return (
        _lexical_absolute(results_dir)
        / "_reports"
        / f"{SMOKE_COMPARISON_SCHEMA_VERSION}_{fingerprint}.json"
    )


def _validate_smoke_comparison_output_scope(
    *,
    output_path: Path | None,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path | None:
    """Authorize the canonical comparison or a JSON file under `_reports/`."""

    if output_path is None:
        return None
    results_root = _lexical_absolute(results_dir)
    candidate = _lexical_absolute(output_path)
    default = _smoke_comparison_default_path(
        results_dir=results_root,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    reports_root = results_root / "_reports"
    try:
        report_relative = candidate.relative_to(reports_root)
    except ValueError:
        report_relative = None
    in_reports = (
        report_relative is not None
        and len(report_relative.parts) >= 1
        and candidate.suffix == ".json"
        and candidate.name not in {".json", "..json"}
    )
    if candidate != default and not in_reports:
        raise ValueError(
            "B-Free smoke comparison output must be the canonical default "
            "or a JSON file under the results _reports directory"
        )
    _reject_output_symlink_components(
        candidate, label="smoke comparison"
    )
    return candidate


def _validate_output_targets(
    outputs: Mapping[str, Path | None],
    *,
    protected_files: Sequence[Path],
    protected_dirs: Sequence[Path],
) -> None:
    resolved: dict[str, Path] = {}
    for name, path in outputs.items():
        if path is None:
            continue
        candidate = _lexical_absolute(Path(path))
        _reject_output_symlink_components(
            candidate, label=f"analysis {name}"
        )
        resolved[name] = candidate.resolve()
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("B-Free analysis output paths collide")
    protected = {Path(path).resolve() for path in protected_files}
    directories = tuple(Path(path).resolve() for path in protected_dirs)
    for name, output in resolved.items():
        if output in protected or any(
            output == directory or directory in output.parents
            for directory in directories
        ):
            raise ValueError(
                f"B-Free analysis output {name} would overwrite evidence"
            )


def _bundle_protected_files(
    bundle: RunBundle, *, repo_root: Path
) -> tuple[Path, ...]:
    files = [
        bundle.manifest_path,
        bundle.results_path,
        bundle.expected_path,
        bundle.summary_path,
        *(repo_root / relative for relative in EXPECTED_ADAPTER_SOURCE_PATHS),
    ]
    assets = _require_mapping(bundle.immutable.get("assets"), "immutable.assets")
    for key in ("zip", "config", "checkpoint"):
        record = _require_mapping(assets.get(key), f"assets.{key}")
        path = record.get("path")
        if isinstance(path, str) and path:
            files.append(Path(path))
    source = _require_mapping(bundle.immutable.get("source"), "immutable.source")
    source_files = source.get("files")
    if isinstance(source_files, Mapping):
        for record in source_files.values():
            if isinstance(record, Mapping) and isinstance(
                record.get("path"), str
            ):
                files.append(Path(str(record["path"])))
    return tuple(files)


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if reference_run_id == replay_run_id:
        raise ValueError("smoke comparison requires distinct run IDs")
    reference = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=reference_run_id,
    )
    replay = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=replay_run_id,
    )
    output_path = _validate_smoke_comparison_output_scope(
        output_path=output_path,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    _validate_output_targets(
        {"comparison": output_path},
        protected_files=(
            *_bundle_protected_files(reference, repo_root=repo_root),
            *_bundle_protected_files(replay, repo_root=repo_root),
        ),
        protected_dirs=(
            reference.artifact_dir,
            replay.artifact_dir,
            source_root,
            weights_dir,
            reference.release.manifest_path.parent,
            replay.release.manifest_path.parent,
        ),
    )
    if (
        stable_json(_smoke_immutable_projection(reference.immutable))
        != stable_json(_smoke_immutable_projection(replay.immutable))
        or stable_json(reference.selected) != stable_json(replay.selected)
        or len(reference.selected) != SMOKE_IMAGES
    ):
        raise ValueError("B-Free smoke immutable config/selection differs")
    recorded_runtime = _validate_runtime_contract(
        reference.immutable.get("runtime"), label="smoke recorded runtime"
    )
    if not _same_json_type_and_value(
        replay.immutable.get("runtime"), recorded_runtime
    ):
        raise ValueError("B-Free smoke runtimes differ")
    _source, _assets, state = _verify_source_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        recorded_source=_require_mapping(
            reference.immutable.get("source"), "smoke source"
        ),
        recorded_assets=_require_mapping(
            reference.immutable.get("assets"), "smoke assets"
        ),
    )
    device, current_runtime = _configure_exact_recorded_runtime(
        device_text=str(recorded_runtime["device"]),
        recorded_runtime=recorded_runtime,
        label="smoke head replay",
    )
    model, model_load = _build_independent_model(
        source_root=source_root, state=state, device=device
    )
    try:
        reference_head = replay_persisted_head(
            latest_results=reference.latest_results,
            artifacts=reference.artifacts,
            model=model,
            device=device,
        )
        replay_head = replay_persisted_head(
            latest_results=replay.latest_results,
            artifacts=replay.artifacts,
            model=model,
            device=device,
        )
        comparison = compare_computational_results(
            reference_rows=reference.latest_results,
            replay_rows=replay.latest_results,
            reference_artifacts=reference.artifacts,
            replay_artifacts=replay.artifacts,
        )
    finally:
        del model, state
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            __import__("torch").cuda.empty_cache()
    _verify_bundle_unchanged(reference, repo_root=repo_root)
    _verify_bundle_unchanged(replay, repo_root=repo_root)
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_smoke_comparison_passed",
        "compared_at": utc_now(),
        "analysis_runtime": _analysis_runtime_contract(),
        "recorded_runtime": current_runtime,
        "independent_model_load": model_load,
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            **dict(reference.evidence_snapshot),
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            **dict(replay.evidence_snapshot),
        },
        "selection": reference.contract.selection.as_dict(),
        "reference_persisted_head_replay": reference_head,
        "replay_persisted_head_replay": replay_head,
        "comparison": comparison,
        "evidence_reverified_after_comparison": True,
    }
    if output_path is not None:
        _write_json_verified(output_path, report, label="smoke comparison")
    return report


def _verify_source_assets_unchanged(
    *, source_root: Path, weights_dir: Path, weights_zip: Path
) -> None:
    source = legacy_audit._verify_source_tree(source_root)
    if (
        source.get("commit") != EXPECTED_SOURCE_COMMIT
        or sha256_file(weights_zip) != EXPECTED_ZIP_SHA256
        or sha256_file(weights_dir / "config.yaml") != EXPECTED_CONFIG_SHA256
        or sha256_file(weights_dir / "model_epoch_best.pth")
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("B-Free source/assets changed during audit")


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
    device_text: str,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
    replay: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    bundle = load_formal_run(
        repo_root=repo_root, results_dir=results_dir, run_id=run_id
    )
    recorded_source = _require_mapping(
        bundle.immutable.get("source"), "immutable.source"
    )
    recorded_assets = _require_mapping(
        bundle.immutable.get("assets"), "immutable.assets"
    )
    if (
        source_root.resolve() != Path(str(recorded_source["root"])).resolve()
        or weights_dir.resolve()
        != Path(str(recorded_assets["config"]["path"])).resolve().parent
        or weights_zip.resolve()
        != Path(str(recorded_assets["zip"]["path"])).resolve()
    ):
        raise ValueError("analysis asset paths differ from immutable manifest")
    authorized_outputs = _validate_formal_output_scope(
        repo_root=repo_root,
        run_dir=bundle.run_dir,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    metrics_output_path = authorized_outputs["metrics"]
    audit_output_path = authorized_outputs["audit"]
    _validate_output_targets(
        {"metrics": metrics_output_path, "audit": audit_output_path},
        protected_files=_bundle_protected_files(bundle, repo_root=repo_root),
        protected_dirs=(
            bundle.artifact_dir,
            source_root,
            weights_dir,
            bundle.release.manifest_path.parent,
        ),
    )
    recorded_runtime = _validate_runtime_contract(
        bundle.immutable.get("runtime"), label="immutable.runtime"
    )
    device, current_runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="formal audit",
    )
    source_evidence, asset_evidence, state = _verify_source_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        recorded_source=recorded_source,
        recorded_assets=recorded_assets,
    )
    model, model_load = _build_independent_model(
        source_root=source_root, state=state, device=device
    )
    try:
        runtime = legacy_audit.ReplayRuntime(
            torch=__import__("torch"),
            device=device,
            evidence={"device": str(device)},
        )
        official_golden = legacy_audit._audit_official_golden(
            source_root=source_root,
            model=model,
            runtime=runtime,
            recorded=_require_mapping(
                bundle.immutable.get("execution_official_golden"),
                "execution official golden",
            ),
        )
        head_replay = replay_persisted_head(
            latest_results=bundle.latest_results,
            artifacts=bundle.artifacts,
            model=model,
            device=device,
        )
        metrics = recompute_metrics(
            bundle, iterations=iterations, seed=seed
        )
        fresh_replay = (
            replay_model(
                bundle,
                repo_root=repo_root,
                source_root=source_root,
                model=model,
                device=device,
            )
            if replay
            else None
        )
    finally:
        del model, state
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            __import__("torch").cuda.empty_cache()
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    _verify_source_assets_unchanged(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    metrics_sha256 = _json_artifact_sha256(metrics)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "replay_audit_passed" if replay else "artifact_audit_passed"
        ),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": len(bundle.selected),
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": bundle.coverage,
        "artifact_files": len(bundle.artifacts),
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "analysis_runtime": _analysis_runtime_contract(),
        "recorded_runtime_reproduced": current_runtime,
        "independent_source": source_evidence,
        "independent_assets": asset_evidence,
        "independent_model_load": model_load,
        "independent_official_golden": official_golden,
        "persisted_artifact_head_replay": head_replay,
        "fresh_model_replay": fresh_replay,
        "method_boundary": {
            "method": "B-Free official BFREE_dino2reg4",
            "architecture": legacy.MODEL_ARCH,
            "preprocess_profile": legacy.PREPROCESS_PROFILE,
            "released_checkpoint_evaluated": True,
            "valid_for_t1": True,
            "valid_for_t2": False,
            "fullframe_t2": "not_applicable",
            "fullframe_data_kind": "conditional_full_frame_edit_not_fully_synthetic",
            "license": legacy.LICENSE_RECORD,
            "commercial_clearance": False,
        },
        "contract_checks": {
            "exact_formal_whole_image_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "pair_rank_rejected": True,
            "t2_joint_dense_claims_rejected": True,
            "npz_inventory_bytes_members_hash_shape_dtype_finite_validated": True,
            "persisted_features_to_five_logits_to_fp32_mean_replayed": True,
            "persisted_recorded_device_head_replay_absolute_tolerance": (
                PERSISTED_HEAD_ABS_TOLERANCE
            ),
            "static_cpu_mean_is_cross_device_sanity_not_exact_replay": True,
            "static_cross_device_mean_sanity_absolute_tolerance": (
                CROSS_DEVICE_MEAN_ABS_TOLERANCE
            ),
            "raw_logit_primary_sigmoid_diagnostic_only": True,
            "shared_balanced250_metrics_only": True,
            "run_canonical_source_asset_evidence_reverified_after_replay": True,
        },
        "artifacts": {
            **dict(bundle.evidence_snapshot),
            "metrics_sha256": metrics_sha256,
        },
    }
    # No analysis output is published until every replay and final evidence
    # recheck above succeeds.
    authorized_outputs = _validate_formal_output_scope(
        repo_root=repo_root,
        run_dir=bundle.run_dir,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    metrics_output_path = authorized_outputs["metrics"]
    audit_output_path = authorized_outputs["audit"]
    if metrics_output_path is not None:
        _write_json_verified(
            metrics_output_path, metrics, label="Balanced250 metrics"
        )
    if audit_output_path is not None:
        _write_json_verified(
            audit_output_path, audit, label="B-Free independent audit"
        )
    if metrics_output_path is not None:
        _verify_json_artifact(
            metrics_output_path,
            expected_sha256=metrics_sha256,
            label="Balanced250 metrics",
        )
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_smoke_comparison_output(
    *,
    requested_output: Path | None,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    candidate = (
        requested_output
        if requested_output is not None
        else _smoke_comparison_default_path(
            results_dir=results_dir,
            reference_run_id=reference_run_id,
            replay_run_id=replay_run_id,
        )
    )
    authorized = _validate_smoke_comparison_output_scope(
        output_path=candidate,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    if authorized is None:
        raise AssertionError("smoke comparison output authorization failed")
    return authorized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--weights-zip", type=Path, default=DEFAULT_WEIGHTS_ZIP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--compare-smoke-run-id")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _assert_runner_contract_exports()
    repo_root = args.repo_root.resolve()
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    run_id = _valid_run_id(args.run_id)
    source_root = _anchored(args.source_root, repo_root)
    weights_dir = _anchored(args.weights_dir, repo_root)
    weights_zip = _anchored(args.weights_zip, repo_root)
    if args.compare_smoke_run_id is not None:
        compare_id = _valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
        ):
            raise ValueError(
                "smoke comparison cannot combine with formal audit options"
            )
        output = _resolve_smoke_comparison_output(
            requested_output=(
                _lexical_absolute(args.comparison_output, base=repo_root)
                if args.comparison_output is not None
                else None
            ),
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
        )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
            source_root=source_root,
            weights_dir=weights_dir,
            weights_zip=weights_zip,
            output_path=output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError("--comparison-output requires --compare-smoke-run-id")
    run_dir = _resolve_run_dir(results_dir, run_id)
    metrics_output = (
        _lexical_absolute(args.metrics_output, base=repo_root)
        if args.metrics_output is not None
        else run_dir / "balanced250_metrics.json"
    )
    audit_output = (
        _lexical_absolute(args.audit_output, base=repo_root)
        if args.audit_output is not None
        else run_dir / "independent_audit.json"
    )
    report = analyze(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        device_text=args.device,
        metrics_output_path=metrics_output,
        audit_output_path=audit_output,
        replay=not args.skip_model_replay,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
