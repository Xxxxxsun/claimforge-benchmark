#!/usr/bin/env python3
"""Audit and fully replay Community Forensics High-res 384 on Balanced250.

The formal adapter persists the exact 384-dimensional float32 input to the
released ViT classifier head for every image.  This analyzer treats the run
as untrusted evidence: it
rebuilds the exact 1,775-image selection, validates every append-only attempt,
checks the complete feature inventory, independently replays the released
linear head, recomputes the shared Balanced250 T1 metrics, and by default
performs a fresh image-to-feature replay of all canonical JPEGs.

Only the released Resize(short side=440)+CenterCrop(384) test profile is
accepted.
Community Forensics is an image-level detector, so pair-rank, localization,
pixel-mask, T2, dense-output, and joint-score claims are rejected.  Crop
visibility is an explanatory diagnostic derived independently from canonical
exact-diff GT; it is not a predicted localization output.
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
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import analyze_community_forensics_run as legacy_audit
from eval.opensource import run_community_forensics as legacy
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
    CanonicalRelease,
    load_canonical_release,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "community_forensics_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = (
    "community_forensics_balanced_smoke_comparison_v2"
)
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
EXPECTED_RUN_MANIFEST_SCHEMA = (
    "community_forensics_balanced_run_manifest_v2"
)
EXPECTED_RUN_CONFIG_SCHEMA = "community_forensics_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = (
    "community_forensics_balanced_runtime_summary_v2"
)
EXPECTED_CPU_PREFLIGHT_SCHEMA = (
    "community_forensics_balanced_cpu_preflight_v1"
)

DEFAULT_RESULTS_DIR = Path("results/opensource/community_forensics")
DEFAULT_RUN_ID = (
    "community_forensics_highres_vit_s16_384_"
    "balanced250_v1_full1775_20260726"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/Community-Forensics-ee5b71d4"
)
DEFAULT_MODEL_ROOT = Path(
    "/root/.cache/claimforge/models/community_forensics/"
    "commfor-model-384-6076002b"
)
DEFAULT_PROCESSOR_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "commfor-data-preprocessor-3540a3f0"
)

FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
EXPECTED_RUNTIME_SEED = 100
FEATURE_DIMENSION = 384
FEATURE_DTYPE = np.dtype(np.float32)
FEATURE_NBYTES = FEATURE_DIMENSION * FEATURE_DTYPE.itemsize
RAW_LOGIT_ABS_TOLERANCE = 0.0
PROBABILITY_ABS_TOLERANCE = 0.0
FEATURE_ABS_TOLERANCE = 0.0
FROZEN_PROFILE = legacy.PREPROCESS_PROFILE
FEATURE_SEMANTICS = legacy.FEATURE_SEMANTICS

EXPECTED_ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_community_forensics_balanced.py",
    "eval/opensource/analyze_community_forensics_balanced.py",
    "eval/opensource/run_community_forensics.py",
    "eval/opensource/analyze_community_forensics_run.py",
    "eval/opensource/community_forensics_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
)
EXPECTED_IMMUTABLE_CONFIG_KEYS = frozenset(
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
        "artifact_contract",
        "local_artifact_policy",
        "outputs",
    }
)
EXPECTED_FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/"
    "community-forensics-balanced-nightly20250627/bin/python"
)
EXPECTED_FROZEN_VENV_PREFIX = Path(
    "/root/.cache/claimforge/venvs/"
    "community-forensics-balanced-nightly20250627"
)
EXPECTED_FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/"
    "community-forensics-balanced-nightly20250627-v2-empty"
)
EXPECTED_FROZEN_PYVENV_CONFIG_SHA256 = (
    "7a40b0582b3525537e9e005348ceec3a23259899af45afc367014c7acbdf91f4"
)
EXPECTED_FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torch_distribution": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "torchvision_distribution": "0.23.0.dev20250627+cu128",
    "numpy": "2.2.6",
    "Pillow": "11.1.0",
    "timm": "1.0.15",
    "safetensors": "0.5.2",
    "scikit-learn": "1.5.2",
    "scipy": "1.16.0",
    "joblib": "1.4.2",
    "threadpoolctl": "3.5.0",
    "setuptools": "79.0.1",
}
EXPECTED_ANALYSIS_PACKAGE_VERSIONS = {
    "scipy": "1.16.0",
    "scikit-learn": "1.5.2",
}
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_MINIMUM_CUDA_FREE_BYTES = 8 * 1024**3
CPU_GOLDEN_SAMPLE_ID = "2c80d38ac19c2d3b76950996"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/"
    f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "12607f3cdada1480038f3d506146cdc1fa0c1c50034afda5e3a5f175433e716b"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "5a4747a6e3a8313f8c9ec3dde2504bb53184666276d7e54dc5fab53ca0e7194b"
)
CPU_GOLDEN_RESIZED_RGB_SHA256 = (
    "dd85a9c31b4e7248da5857f6928e64cd8955ffa8e984d19600620e5c42321fb7"
)
CPU_GOLDEN_CROP_RGB_SHA256 = (
    "eb7c1b6c7c527f8bdbacaf6bd3957cea3577bd0e2029cd69b253042aa4e1328a"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "9540fe65ec48c8a1e6ecafb0ede0c96888c076eda1ba046117f3b2830f4a881e"
)
# The exact CPU model values are asserted against the runner's frozen
# exports.  Keeping the image/preprocessing anchors independent here catches
# a runner and analyzer drifting together before model inference.
EXPECTED_PREPROCESS_CONTRACT = {
    "profile_id": FROZEN_PROFILE,
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "resize": {
        "kind": "torchvision_Resize_integer_short_side",
        "short_side": legacy.RESIZE_SHORT_SIDE,
        "preserve_aspect_ratio": True,
        "interpolation": "PIL_BILINEAR",
        "antialias": True,
    },
    "center_crop": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
    "tensor_dtype": "float32",
    "normalization_mean": list(legacy.IMAGE_MEAN),
    "normalization_std": list(legacy.IMAGE_STD),
    "batch_size": 1,
    "official_forward_calls_per_image": 1,
    "feature_capture": "forward_hook_on_model.vit.head_input",
}
EXPECTED_ARTIFACT_CONTRACT = {
    "feature": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [FEATURE_DIMENSION],
        "dtype": "float32",
        "nbytes": FEATURE_NBYTES,
        "finite": True,
        "semantics": FEATURE_SEMANTICS,
        "allow_pickle": False,
        "exact_head_and_sigmoid_replay_on_recorded_device": True,
        "visibility": "local_only_gitignored_output",
    }
}
EXPECTED_TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}
EXPECTED_MODEL_CONTRACT = {
    "name": legacy.MODEL_NAME,
    "slug": legacy.MODEL_SLUG,
    "architecture": legacy.MODEL_ARCH,
    "repository": legacy.MODEL_REPO_URL,
    "paper": legacy.PAPER_URL,
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "eval_single_commit": legacy.EVAL_SINGLE_COMMIT,
    "model_repository": legacy.MODEL_HF_REPO,
    "model_revision": legacy.MODEL_HF_REVISION,
    "processor_repository": legacy.PROCESSOR_HF_REPO,
    "processor_revision": legacy.PROCESSOR_HF_REVISION,
    "checkpoint_id": legacy.CHECKPOINT["id"],
    "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
    "checkpoint_schema_sha256": (
        "e855838996a924c0c49bc95c60ad5365c7502a556eb228523699751ec865df60"
    ),
    "asset_bundle_sha256": (
        "810a7592a82f09cbf638985e9c59eed9ebd2c3ff28ebab97f348bfd3c69b7fb3"
    ),
    "construction": (
        "timm_create_model_pretrained_false_replace_head_then_strict_full_state"
    ),
    "feature_dimension": FEATURE_DIMENSION,
    "score": {
        "semantics": legacy.SCORE_SEMANTICS,
        "direction": "higher_means_fake",
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
    },
    "license": legacy.LICENSE_RECORD,
}

_RUN_IDENTITY_FIELDS = frozenset(
    {"run_id", "run_manifest_fingerprint", "config_fingerprint"}
)
_SMOKE_ROW_IGNORED_FIELDS = frozenset(
    {
        *_RUN_IDENTITY_FIELDS,
        "completed_at",
        "preprocess_latency_ms",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_FEATURE_VOLATILE_FIELDS = frozenset(
    {"relative_path", "sha256", "array_sha256"}
)
_FALSE_DECLARATIONS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULLABLE_DECLARATIONS = frozenset(
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
_FORBIDDEN_EXACT_KEYS = frozenset(
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
_FORBIDDEN_PREFIXES = (
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


@dataclass(frozen=True)
class FeatureArtifact:
    """One validated lossless Community Forensics feature."""

    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: str
    array: np.ndarray


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
    feature_dir: Path
    features: Mapping[str, FeatureArtifact]
    evidence_snapshot: Mapping[str, str]


def _runner() -> Any:
    return importlib.import_module(
        "eval.opensource.run_community_forensics_balanced"
    )


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
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
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


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
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


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    raw = results_dir if results_dir.is_absolute() else root / results_dir
    absolute = Path(os.path.abspath(raw))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError("results root escapes repository") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("results root contains a symlink component")
    return absolute


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    run_id = _runner()._valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / run_id).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("resolved run directory escapes results root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("resolved run directory contains a symlink")
    if candidate == root:
        raise ValueError("run directory must be below results root")
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


def _reject_unsupported_claims(value: Any, label: str) -> None:
    """Reject every T2, localization, pair-rank, dense, and joint claim."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(nested, child)
            elif normalized in _FALSE_DECLARATIONS:
                if nested is not False:
                    raise ValueError(
                        f"{child} is an unsupported Community Forensics claim"
                    )
            elif normalized in _NULLABLE_DECLARATIONS:
                if nested is not None:
                    raise ValueError(
                        f"{child} is an unsupported Community Forensics claim"
                    )
            elif normalized in _FORBIDDEN_EXACT_KEYS or normalized.startswith(
                _FORBIDDEN_PREFIXES
            ):
                raise ValueError(
                    f"{child} is an unsupported Community Forensics claim"
                )
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
        fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
        threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
    )


def _assert_runner_contract_exports() -> Any:
    runner = _runner()
    expected = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_SEED": EXPECTED_RUNTIME_SEED,
        "FROZEN_PROFILE": FROZEN_PROFILE,
        "FROZEN_PYTHON_EXECUTABLE": EXPECTED_FROZEN_PYTHON_EXECUTABLE,
        "FROZEN_VENV_PREFIX": EXPECTED_FROZEN_VENV_PREFIX,
        "FROZEN_PYTHONPYCACHEPREFIX": (
            EXPECTED_FROZEN_PYTHONPYCACHEPREFIX
        ),
        "FROZEN_PYVENV_CONFIG_SHA256": (
            EXPECTED_FROZEN_PYVENV_CONFIG_SHA256
        ),
        "FROZEN_RUNTIME_VERSIONS": EXPECTED_FROZEN_RUNTIME_VERSIONS,
        "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "MINIMUM_CUDA_FREE_BYTES": EXPECTED_MINIMUM_CUDA_FREE_BYTES,
        "CPU_GOLDEN_SAMPLE_ID": CPU_GOLDEN_SAMPLE_ID,
        "CPU_GOLDEN_INPUT_PATH": CPU_GOLDEN_INPUT_PATH,
        "CPU_GOLDEN_IMAGE_SHA256": CPU_GOLDEN_IMAGE_SHA256,
        "CPU_GOLDEN_DECODED_RGB_SHA256": (
            CPU_GOLDEN_DECODED_RGB_SHA256
        ),
        "CPU_GOLDEN_RESIZED_RGB_SHA256": CPU_GOLDEN_RESIZED_RGB_SHA256,
        "CPU_GOLDEN_CROP_RGB_SHA256": CPU_GOLDEN_CROP_RGB_SHA256,
        "CPU_GOLDEN_TENSOR_SHA256": CPU_GOLDEN_TENSOR_SHA256,
        "PREPROCESS_CONTRACT": EXPECTED_PREPROCESS_CONTRACT,
        "MODEL_CONTRACT": EXPECTED_MODEL_CONTRACT,
        "TASK_SCOPE": EXPECTED_TASK_SCOPE,
        "ARTIFACT_CONTRACT": EXPECTED_ARTIFACT_CONTRACT,
    }
    for name, expected_value in expected.items():
        if getattr(runner, name, None) != expected_value:
            raise ValueError(f"runner export {name} differs from analyzer pin")
    if tuple(getattr(runner, "ADAPTER_SOURCE_PATHS", ())) != (
        EXPECTED_ADAPTER_SOURCE_PATHS
    ):
        raise ValueError("runner adapter source inventory differs from analyzer pin")
    if frozenset(getattr(runner, "IMMUTABLE_CONFIG_KEYS", ())) != (
        EXPECTED_IMMUTABLE_CONFIG_KEYS
    ):
        raise ValueError("runner immutable key set differs from analyzer pin")
    required_callables = (
        "configure_runtime",
        "validate_runtime_contract",
        "select_mode_inputs",
        "selection_visibility_census",
        "_valid_run_id",
        "_validate_runner_attempt",
        "_validate_physical_attempt_history",
        "_validate_preflight_report",
        "_local_artifact_policy",
    )
    for name in required_callables:
        if not callable(getattr(runner, name, None)):
            raise ValueError(f"runner lacks mandatory contract function {name}")
    return runner


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    _assert_runner_contract_exports()
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected = set(EXPECTED_ADAPTER_SOURCE_PATHS)
    if set(sources) != expected:
        raise ValueError("immutable.adapter_sources exact inventory changed")
    for relative in sorted(expected):
        record = _require_mapping(sources[relative], f"adapter {relative}")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} key set changed")
        if record.get("path") != relative:
            raise ValueError(f"adapter source {relative} path mismatch")
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"adapter source {relative} hash/size mismatch")


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    runner = _assert_runner_contract_exports()
    raw_contract = _require_mapping(
        immutable.get("dataset_contract"), "immutable.dataset_contract"
    )
    release_record = _require_mapping(
        raw_contract.get("release"), "dataset contract release"
    )
    manifest_path = _safe_repo_path(
        release_record.get("manifest_path"),
        repo_root=repo_root,
        label="dataset manifest",
    )
    release = load_canonical_release(repo_root, manifest_path, verify_files=True)
    if expected_mode == "formal":
        per_condition_limit = None
    elif expected_mode == "smoke":
        selection = _require_mapping(
            raw_contract.get("selection"), "dataset contract selection"
        )
        spec = _require_mapping(
            selection.get("spec"), "dataset contract selection spec"
        )
        per_condition_limit = spec.get("per_condition_limit")
        if per_condition_limit != SMOKE_PER_CONDITION:
            raise ValueError(
                "Community Forensics smoke must select five images per condition"
            )
    else:
        raise ValueError(f"unsupported analyzer mode {expected_mode!r}")
    spec, selected_value = runner.select_mode_inputs(
        release,
        mode=expected_mode,
        per_condition_limit=per_condition_limit,
        sample_id=None,
    )
    selected = tuple(selected_value)
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=_score_spec(),
    )
    if rebuilt.as_dict() != raw_contract:
        raise ValueError("immutable dataset contract does not rebuild exactly")
    if rebuilt.capability.as_dict() != {
        "name": "whole_image_t1",
        "conditions": list(BALANCED_CONDITIONS),
        "valid_for_t1": True,
        "valid_for_t2": False,
    }:
        raise ValueError("dataset contract is not exact WHOLE_IMAGE_T1")
    expected_count = FORMAL_IMAGES if expected_mode == "formal" else SMOKE_IMAGES
    if len(selected) != expected_count:
        raise ValueError(
            f"{expected_mode} selection has {len(selected)} images, "
            f"expected {expected_count}"
        )
    counts = Counter(str(row["condition"]) for row in selected)
    expected_counts = (
        {
            "real": 275,
            **{condition: 250 for condition in BALANCED_CONDITIONS[1:]},
        }
        if expected_mode == "formal"
        else {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
    )
    if dict(counts) != expected_counts:
        raise ValueError(f"{expected_mode} condition counts changed")
    if immutable.get("mode") != expected_mode:
        raise ValueError(f"analyzer requires immutable.mode={expected_mode}")
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score spec changed")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("immutable selected-row SHA-256 changed")
    if immutable.get("selected_ids_sha256") != selected_ids_sha256(
        str(row["sample_id"]) for row in selected
    ):
        raise ValueError("immutable selected-ID SHA-256 changed")
    visibility = _independent_selection_visibility_census(
        selected,
        repo_root=repo_root,
    )
    if immutable.get("selection_visibility_census") != visibility:
        raise ValueError(
            "immutable selection crop-visibility census does not rebuild"
        )
    runner_visibility = runner.selection_visibility_census(
        selected,
        repo_root=repo_root,
    )
    if runner_visibility != visibility:
        raise ValueError(
            "runner and independent crop-visibility censuses differ"
        )
    return release, selected, rebuilt


def _current_geometry(width: int, height: int) -> dict[str, Any]:
    """Independently reproduce Resize(440)+CenterCrop(384) geometry."""

    # This helper lives in the legacy *analyzer*, not the inference runner.
    # It independently implements torchvision's integer resize and crop
    # rounding and has regression coverage for wide, tall, and small inputs.
    geometry = legacy_audit.compute_preprocess_geometry(width, height)
    if geometry.get("profile_id") != FROZEN_PROFILE:
        raise ValueError("independent Community Forensics profile drifted")
    return geometry


def _intersection_xyxy(
    first: Sequence[float],
    second: Sequence[float],
) -> list[float] | None:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return (
        [float(left), float(top), float(right), float(bottom)]
        if right > left and bottom > top
        else None
    )


def _independent_visibility_diagnostic(
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recompute official-crop visibility directly from canonical GT."""

    width = int(input_row["width"])
    height = int(input_row["height"])
    geometry = _current_geometry(width, height)
    gt_kind = input_row.get("gt_mask_kind")
    if gt_kind in ("all_zero", "not_applicable"):
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
                "effective_native_crop_xyxy": list(
                    geometry["effective_native_crop_xyxy"]
                ),
            },
        }
    if gt_kind != "exact_diff":
        raise ValueError("input has unsupported visibility GT kind")
    sample_id = str(input_row["sample_id"])
    mask_path = _safe_repo_path(
        input_row.get("gt_mask_path"),
        repo_root=repo_root,
        label=f"{sample_id} exact-diff GT",
    )
    if sha256_file(mask_path) != input_row.get("gt_mask_sha256"):
        raise ValueError(f"{sample_id} exact-diff GT hash changed")
    with Image.open(mask_path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError(f"{sample_id} exact-diff GT encoding changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    if pixels.shape != (height, width) or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} exact-diff GT pixels changed")
    positive_y, positive_x = np.nonzero(pixels == 255)
    total = int(positive_x.size)
    if total <= 0 or total != input_row.get("gt_positive_pixels"):
        raise ValueError(f"{sample_id} exact-diff positive count changed")
    resized_width, resized_height = geometry["resize"]["destination_size"]
    start_x, start_y = geometry["center_crop"]["start_xy"]
    end_x, end_y = geometry["center_crop"]["end_xy"]
    destination_x = (
        (positive_x.astype(np.float64) + 0.5)
        * float(resized_width)
        / float(width)
        - 0.5
    )
    destination_y = (
        (positive_y.astype(np.float64) + 0.5)
        * float(resized_height)
        / float(height)
        - 0.5
    )
    visible_mask = (
        (destination_x >= start_x)
        & (destination_x < end_x)
        & (destination_y >= start_y)
        & (destination_y < end_y)
    )
    visible = int(np.count_nonzero(visible_mask))
    fraction = visible / total
    category = "none" if visible == 0 else "full" if visible == total else "partial"
    gt_evidence = {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": sample_id,
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_through_"
            "official_resize440_and_center_crop384"
        ),
        "profile_id": FROZEN_PROFILE,
        "formula": geometry["pixel_center_mapping"],
    }
    edit_region = input_row.get("edit_region_xyxy")
    if (
        not isinstance(edit_region, list)
        or len(edit_region) != 4
        or any(type(value) is not int for value in edit_region)
    ):
        raise ValueError(f"{sample_id} edit region changed")
    box = [float(value) for value in edit_region]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{sample_id} edit region has non-positive area")
    native_crop = list(geometry["effective_native_crop_xyxy"])
    intersection = _intersection_xyxy(box, native_crop)
    area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    box_fraction = min(1.0, max(0.0, visible_area / area))
    box_category = (
        "none"
        if box_fraction == 0.0
        else "full"
        if math.isclose(box_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else "partial"
    )
    return {
        "edit_visibility": category,
        "edit_visible_gt_fraction": fraction,
        "edit_visibility_evidence": {
            "gt": gt_evidence,
            "edit_box": {
                "edit_region_xyxy": edit_region,
                "effective_native_crop_xyxy": native_crop,
                "intersection_xyxy": intersection,
                "edit_area": area,
                "visible_area": visible_area,
                "visible_fraction": box_fraction,
                "category": box_category,
                "basis": (
                    "continuous_edit_box_area_intersection_with_"
                    "effective_native_crop"
                ),
            },
        },
    }


def _independent_selection_visibility_census(
    selected: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Aggregate crop visibility without calling the Balanced runner helper."""

    by_condition: dict[str, dict[str, Any]] = {}
    all_counts: Counter[str] = Counter()
    all_fractions: list[float] = []
    local_conditions = ("local_mouse", "local_cat", "local_trash_can")
    for condition in local_conditions:
        counts: Counter[str] = Counter()
        fractions: list[float] = []
        for row in selected:
            if row.get("condition") != condition:
                continue
            diagnostic = _independent_visibility_diagnostic(
                row,
                repo_root=repo_root,
            )
            category = str(diagnostic["edit_visibility"])
            fraction = float(diagnostic["edit_visible_gt_fraction"])
            if category not in ("full", "partial", "none"):
                raise ValueError("independent local visibility category changed")
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("independent local visibility fraction is invalid")
            counts[category] += 1
            fractions.append(fraction)
            all_counts[category] += 1
            all_fractions.append(fraction)
        by_condition[condition] = {
            "full": counts["full"],
            "partial": counts["partial"],
            "none": counts["none"],
            "total": len(fractions),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
        }
    return {
        "profile_id": FROZEN_PROFILE,
        "role": "input_condition_diagnostic_not_model_localization",
        "by_condition": by_condition,
        "all_local": {
            "full": all_counts["full"],
            "partial": all_counts["partial"],
            "none": all_counts["none"],
            "total": len(all_fractions),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(all_fractions)) if all_fractions else None
            ),
        },
        "not_applicable_images": sum(
            row.get("gt_mask_kind") != "exact_diff" for row in selected
        ),
    }


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    raw = _require_finite(row.get("raw_logit"), f"{sample_id} raw_logit")
    probability = _require_finite(
        row.get("probability"), f"{sample_id} probability"
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{sample_id} probability is outside [0,1]")
    # Do not rebuild the CUDA-produced sigmoid on the host CPU: different
    # float32 kernels are not required to agree bit-for-bit (or within a
    # fixed absolute tolerance).  replay_linear_head below is the
    # authoritative exact check and uses the run's recorded device/runtime.
    ai_score = _require_finite(row.get("ai_score"), f"{sample_id} ai_score")
    if row.get("score") != ai_score or ai_score != probability:
        raise ValueError(f"{sample_id} score aliases differ")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    if row.get("classification_decision") is not decision:
        raise ValueError(f"{sample_id} strict decision changed")
    if (
        row.get("classification_threshold")
        != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} fixed threshold changed")
    if (
        row.get("score_semantics")
        != legacy.SCORE_SEMANTICS
    ):
        raise ValueError(f"{sample_id} score semantics changed")
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
    if row.get("classification") != classification:
        raise ValueError(f"{sample_id} classification aliases changed")
    t1 = dict(classification)
    t1.pop("semantics")
    t1["policy"] = legacy.T1_POLICY
    if row.get("t1") != t1:
        raise ValueError(f"{sample_id} T1 aliases changed")
    manual = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
    }
    if row.get("manual_replay") != manual:
        raise ValueError(f"{sample_id} manual head replay changed")


def _validate_source_contract(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "immutable.source")
    root = Path(_require_string(source.get("root"), "immutable.source.root"))
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != Path(os.path.abspath(root))
    ):
        raise ValueError("immutable source root is missing, relative, or a symlink")
    for relative in legacy.SOURCE_FILES:
        source_file = root / relative
        if (
            source_file.is_symlink()
            or not source_file.is_file()
            or source_file.resolve() != Path(os.path.abspath(source_file))
        ):
            raise ValueError(f"immutable source file is unsafe: {relative}")
    pins = legacy_audit._load_runner_pins()
    actual = legacy_audit._verify_source_tree(root, pins=pins)
    if source != actual:
        raise ValueError("immutable.source differs from frozen source audit")
    return source


def _validate_assets_contract(
    value: Any,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    assets = _require_mapping(value, "immutable.assets")
    checkpoint = _require_mapping(
        assets.get("checkpoint"), "immutable.assets.checkpoint"
    )
    model_repository = _require_mapping(
        assets.get("model_repository"),
        "immutable.assets.model_repository",
    )
    processor = _require_mapping(
        assets.get("processor"), "immutable.assets.processor"
    )
    checkpoint_path = Path(
        _require_string(
            checkpoint.get("path"), "immutable.assets.checkpoint.path"
        )
    )
    model_root = Path(
        _require_string(
            model_repository.get("root"),
            "immutable.assets.model_repository.root",
        )
    )
    processor_root = Path(
        _require_string(
            processor.get("root"), "immutable.assets.processor.root"
        )
    )
    if (
        not checkpoint_path.is_absolute()
        or not model_root.is_absolute()
        or not processor_root.is_absolute()
        or checkpoint_path.is_symlink()
        or model_root.is_symlink()
        or processor_root.is_symlink()
        or not checkpoint_path.is_file()
        or not model_root.is_dir()
        or not processor_root.is_dir()
        or checkpoint_path.resolve() != Path(os.path.abspath(checkpoint_path))
        or model_root.resolve() != Path(os.path.abspath(model_root))
        or processor_root.resolve() != Path(os.path.abspath(processor_root))
        or checkpoint_path.parent != model_root
    ):
        raise ValueError("immutable asset path is relative or a symlink")
    import torch

    pins = legacy_audit._load_runner_pins()
    actual_assets_source = legacy_audit._verify_source_tree(
        Path(str(source["root"])), pins=pins
    )
    _state, actual_assets = legacy_audit._verify_assets(
        model_root=model_root,
        processor_root=processor_root,
        pins=pins,
        torch_module=torch,
    )
    if actual_assets_source != source:
        raise ValueError("source audit changed during asset verification")
    if assets != actual_assets:
        raise ValueError("immutable.assets differs from frozen asset audit")
    return assets


def _validate_runtime_contract(value: Any, *, label: str) -> dict[str, Any]:
    """Validate exact runner schema plus independent numerical/runtime pins."""

    runtime = _require_mapping(value, label)
    _reject_nonfinite_numbers(runtime, label)
    runner = _assert_runner_contract_exports()
    validated = runner.validate_runtime_contract(runtime, label=label)
    if validated is not None and dict(validated) != runtime:
        raise ValueError(f"{label} runner validator changed runtime evidence")

    device = _require_string(runtime.get("device"), f"{label}.device")
    if device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError(f"{label}.device is unsupported")
    python_record = _require_mapping(runtime.get("python"), f"{label}.python")
    if set(python_record) != {"implementation", "version", "executable"}:
        raise ValueError(f"{label}.python key set changed")
    if (
        python_record.get("implementation") != "CPython"
        or python_record.get("version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["python"]
        or not Path(str(python_record.get("executable"))).is_absolute()
        or str(python_record.get("executable"))
        != EXPECTED_FROZEN_PYTHON_EXECUTABLE.as_posix()
    ):
        raise ValueError(f"{label}.python dedicated runtime changed")
    venv = _require_mapping(runtime.get("venv"), f"{label}.venv")
    expected_prefix = EXPECTED_FROZEN_VENV_PREFIX.as_posix()
    expected_venv = {
        "prefix": expected_prefix,
        "base_prefix": "/usr",
        "pyvenv_cfg_path": f"{expected_prefix}/pyvenv.cfg",
        "pyvenv_cfg_sha256": EXPECTED_FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }
    if venv != expected_venv:
        raise ValueError(f"{label}.venv frozen-environment evidence changed")
    packages = _require_mapping(runtime.get("packages"), f"{label}.packages")
    expected_package_keys = {
        "torch",
        "torchvision",
        "timm",
        "safetensors",
        "numpy",
        "Pillow",
        "scikit-learn",
        "scipy",
        "joblib",
        "threadpoolctl",
        "setuptools",
    }
    if set(packages) != expected_package_keys:
        raise ValueError(f"{label}.packages exact key set changed")
    torch_record = _require_mapping(
        packages.get("torch"), f"{label}.packages.torch"
    )
    torchvision_record = _require_mapping(
        packages.get("torchvision"), f"{label}.packages.torchvision"
    )
    if set(torch_record) != {
        "version",
        "distribution_version",
        "cuda_runtime",
        "cudnn_version",
    } or set(torchvision_record) != {"version", "distribution_version"}:
        raise ValueError(f"{label}.package record key set changed")
    cudnn_version = torch_record.get("cudnn_version")
    if (
        torch_record.get("version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torch"]
        or torch_record.get("distribution_version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torch_distribution"]
        or torch_record.get("cuda_runtime") != "12.8"
        or (
            cudnn_version is not None
            and (
                isinstance(cudnn_version, bool)
                or not isinstance(cudnn_version, int)
                or cudnn_version <= 0
            )
        )
        or torchvision_record.get("version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torchvision"]
        or torchvision_record.get("distribution_version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torchvision_distribution"]
        or packages.get("numpy")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["numpy"]
        or packages.get("Pillow")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["Pillow"]
        or packages.get("timm")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["timm"]
        or packages.get("safetensors")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["safetensors"]
        or packages.get("joblib")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["joblib"]
        or packages.get("threadpoolctl")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["threadpoolctl"]
        or packages.get("setuptools")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["setuptools"]
        or packages.get("scipy")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["scipy"]
        or packages.get("scikit-learn")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["scikit-learn"]
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    if not isinstance(runtime.get("platform"), str) or not runtime["platform"]:
        raise ValueError(f"{label}.platform is invalid")
    expected_pycache = EXPECTED_FROZEN_PYTHONPYCACHEPREFIX.as_posix()
    expected_process_environment = {
        "PYTHONHASHSEED": str(EXPECTED_RUNTIME_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": expected_pycache,
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": expected_pycache,
        "pycache_prefix_initially_empty": True,
    }
    if (
        runtime.get("seed") != EXPECTED_RUNTIME_SEED
        or runtime.get("preprocess_profile") != FROZEN_PROFILE
        or runtime.get("inference_dtype") != "float32"
        or runtime.get("feature_dtype") != "float32"
        or runtime.get("batch_size") != 1
        or runtime.get("autocast") is not False
        or runtime.get("grad_enabled") is not False
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cublas_workspace_config")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("matmul_allow_tf32") is not False
        or runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("minimum_cuda_free_bytes")
        != EXPECTED_MINIMUM_CUDA_FREE_BYTES
        or runtime.get("bytecode_writes_disabled") is not True
        or runtime.get("process_environment")
        != expected_process_environment
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    if set(cudnn) != {
        "enabled",
        "benchmark",
        "deterministic",
        "allow_tf32",
    } or (
        cudnn.get("enabled") is not False
        or cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if device.startswith("cuda:"):
        cuda = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        if set(cuda) != {
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
    return runtime


def _validate_cpu_preflight(
    value: Any,
    *,
    repo_root: Path,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    """Independently validate the pre-accelerator double CPU full forward."""

    runner = _assert_runner_contract_exports()
    wrapper = _require_mapping(value, "immutable.cpu_preflight")
    if set(wrapper) != {
        "performed_before_accelerator_configuration",
        "report",
    } or wrapper.get("performed_before_accelerator_configuration") is not True:
        raise ValueError("CPU preflight ordering evidence changed")
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
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
    ):
        raise ValueError("CPU preflight report/provenance changed")
    runtime = _validate_runtime_contract(
        report.get("runtime"), label="CPU preflight runtime"
    )
    if runtime.get("device") != "cpu":
        raise ValueError("CPU preflight runtime is not CPU")
    model_load = _require_mapping(
        report.get("model_load"), "CPU preflight model_load"
    )
    if (
        model_load.get("construction", {}).get("architecture")
        != legacy.MODEL_ARCH
        or model_load.get("construction", {}).get("pretrained") is not False
        or model_load.get("load", {}).get("format") != "safetensors"
        or model_load.get("load", {}).get("strict") is not True
        or model_load.get("load", {}).get("full_state_coverage") is not True
        or model_load.get("network", {}).get("allowed") is not False
        or model_load.get("feature_dimension") != FEATURE_DIMENSION
    ):
        raise ValueError("CPU preflight safe model-load evidence changed")
    official = _require_mapping(
        report.get("official_golden"), "CPU preflight official_golden"
    )
    if official.get("status") != "passed" or len(
        _require_list(official.get("cases"), "official golden cases")
    ) != 5:
        raise ValueError("CPU preflight official five-image gate changed")
    golden = _require_mapping(
        report.get("balanced_golden"), "CPU preflight balanced_golden"
    )
    image_path = _safe_repo_path(
        CPU_GOLDEN_INPUT_PATH,
        repo_root=repo_root,
        label="CPU golden input",
    )
    if sha256_file(image_path) != CPU_GOLDEN_IMAGE_SHA256:
        raise ValueError("CPU golden input SHA-256 changed")
    import torch

    independent = legacy_audit.preprocess_image(
        image_path,
        torch_module=torch,
    )
    preprocess = independent.audit
    expected_preprocess_subset = {
        "profile": FROZEN_PROFILE,
        "geometry": _current_geometry(1800, 1350),
        "decoded_rgb_sha256": CPU_GOLDEN_DECODED_RGB_SHA256,
        "resized_rgb_sha256": CPU_GOLDEN_RESIZED_RGB_SHA256,
        "crop_rgb_sha256": CPU_GOLDEN_CROP_RGB_SHA256,
        "crop_rgb_shape": [384, 384, 3],
        "crop_rgb_dtype": "uint8",
        "tensor_shape": [3, 384, 384],
        "tensor_dtype": "float32",
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
    }
    if any(
        preprocess.get(key) != expected
        for key, expected in expected_preprocess_subset.items()
    ):
        raise ValueError("CPU golden exact preprocess evidence changed")
    runner_pins = {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "feature_file_sha256": runner.CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_array_sha256": runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
    }
    if any(golden.get(key) != expected for key, expected in runner_pins.items()):
        raise ValueError("CPU balanced golden frozen values changed")
    if (
        golden.get("preprocess") != preprocess
        or golden.get("feature_shape") != [FEATURE_DIMENSION]
        or golden.get("feature_dtype") != "float32"
        or golden.get("feature_nbytes") != FEATURE_NBYTES
        or golden.get("model_forward_calls") != 1
        or golden.get("classifier_hook_calls") != 1
        or golden.get("repeat_feature_file_sha256")
        != runner.CPU_GOLDEN_FEATURE_FILE_SHA256
        or golden.get("repeat_feature_array_sha256")
        != runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
        or golden.get("repeat_raw_logit") != runner.CPU_GOLDEN_RAW_LOGIT
        or golden.get("repeat_probability") != runner.CPU_GOLDEN_PROBABILITY
        or golden.get("repeat_byte_exact") is not True
    ):
        raise ValueError("CPU balanced golden double-forward evidence changed")


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, dict[str, Any]]:
    runner = _assert_runner_contract_exports()
    run_id = runner._valid_run_id(run_id)
    expected_manifest_keys = {
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
    if set(manifest) != expected_manifest_keys:
        raise ValueError("run manifest key set changed")
    if manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported Community Forensics Balanced run manifest")
    if manifest.get("run_id") != run_id or manifest.get("status") != "complete":
        raise ValueError("analyzer requires the exact complete run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest immutable")
    if set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS:
        raise ValueError("manifest immutable key set changed")
    if immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA:
        raise ValueError("immutable schema_version changed")
    if (
        immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
    ):
        raise ValueError("immutable run identity/mode mismatch")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint")
    if fingerprint != hashlib.sha256(stable_json(immutable).encode()).hexdigest():
        raise ValueError("manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"), repo_root=repo_root
    )
    if immutable.get("preprocess") != EXPECTED_PREPROCESS_CONTRACT:
        raise ValueError("immutable preprocess contract changed")
    if immutable.get("model") != EXPECTED_MODEL_CONTRACT:
        raise ValueError("immutable model contract changed")
    if immutable.get("task_scope") != EXPECTED_TASK_SCOPE:
        raise ValueError("immutable task scope changed")
    if immutable.get("artifact_contract") != EXPECTED_ARTIFACT_CONTRACT:
        raise ValueError("immutable feature artifact contract changed")
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score spec changed")
    if (
        immutable.get("formal_local_visibility_census")
        != runner.LOCAL_VISIBILITY_CENSUS
    ):
        raise ValueError("immutable formal crop-visibility census changed")
    expected_local_policy = runner._local_artifact_policy(repo_root)
    if immutable.get("local_artifact_policy") != expected_local_policy:
        raise ValueError("immutable local feature-artifact policy changed")
    source = _validate_source_contract(immutable.get("source"))
    _validate_assets_contract(immutable.get("assets"), source=source)
    _validate_runtime_contract(
        immutable.get("runtime"), label="immutable.runtime"
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        repo_root=repo_root,
        source=source,
        assets=immutable["assets"],
    )
    outputs = _require_mapping(immutable.get("outputs"), "immutable.outputs")
    if set(outputs) != {
        "run_dir",
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "feature_dir",
    }:
        raise ValueError("immutable outputs key set changed")
    for field, path_value in outputs.items():
        if field.endswith(("_path", "_dir")) or field == "run_dir":
            _safe_repo_path(
                path_value,
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
    if expected_rows != list(selected):
        raise ValueError("expected-input snapshot is not the exact selection")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    expected = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    if set(dataset) != set(expected):
        raise ValueError("manifest dataset key set changed")
    for key, expected_value in expected.items():
        if dataset.get(key) != expected_value:
            raise ValueError(f"manifest dataset {key} mismatch")


def _validate_execution(
    *,
    manifest: Mapping[str, Any],
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    expected_keys = {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
        "same_device_feature_head_replays",
    }
    if set(execution) != expected_keys:
        raise ValueError("manifest execution key set changed")
    for key in execution:
        _require_nonnegative_int(execution[key], f"execution.{key}")
    expected = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
        "same_device_feature_head_replays": latest_rows,
        "new_errors": 0,
    }
    for key, expected_value in expected.items():
        if execution.get(key) != expected_value:
            raise ValueError(f"manifest execution {key} mismatch")
    if execution["new_successes"] + execution["resume_skips"] != selected_images:
        raise ValueError("manifest successful work accounting mismatch")


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
            row.get("sample_id"), f"physical result {index} sample_id"
        )
        expected = inputs.get(sample_id)
        if expected is None:
            raise ValueError(f"physical result {index} has unexpected sample_id")
        runner._validate_runner_attempt(
            row,
            input_row=expected,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        if row.get("preprocess_profile") != FROZEN_PROFILE:
            raise ValueError(f"{sample_id} preprocessing profile changed")
        visibility = _independent_visibility_diagnostic(
            expected, repo_root=repo_root
        )
        for key, expected_value in visibility.items():
            if row.get(key) != expected_value:
                raise ValueError(f"{sample_id} crop visibility changed")
        _reject_unsupported_claims(row, f"physical result {index}")
        _reject_nonfinite_numbers(row, f"physical result {index}")
        if row.get("status") == "ok":
            _validate_score_payload(row, sample_id=sample_id)


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


def _feature_mapping(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> dict[str, Any]:
    feature = _require_mapping(row.get("commfor_feature"), f"{sample_id} feature")
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
    if set(feature) != expected_keys:
        raise ValueError(f"{sample_id} feature key set changed")
    return feature


def _feature_artifact(
    *,
    row: Mapping[str, Any],
    sample_id: str,
    repo_root: Path,
    feature_dir: Path,
) -> FeatureArtifact:
    feature = _feature_mapping(row, sample_id=sample_id)
    relative_path = _require_string(
        feature.get("relative_path"), f"{sample_id} feature path"
    )
    path = _safe_repo_path(
        relative_path,
        repo_root=repo_root,
        label=f"{sample_id} feature path",
    )
    expected_path = (feature_dir / f"{sample_id}.npy").resolve()
    if path != expected_path:
        raise ValueError(f"{sample_id} feature path is not canonical")
    file_sha = _require_sha256(
        feature.get("sha256"), f"{sample_id} feature SHA-256"
    )
    if sha256_file(path) != file_sha:
        raise ValueError(f"{sample_id} feature artifact hash mismatch")
    expected_file_bytes = len(
        _npy_bytes(np.zeros((FEATURE_DIMENSION,), dtype=FEATURE_DTYPE))
    )
    if (
        feature.get("file_bytes") != path.stat().st_size
        or path.stat().st_size != expected_file_bytes
        or feature.get("dtype") != "float32"
        or feature.get("shape") != [FEATURE_DIMENSION]
        or feature.get("nbytes") != FEATURE_NBYTES
        or feature.get("finite") is not True
        or feature.get("semantics") != FEATURE_SEMANTICS
    ):
        raise ValueError(f"{sample_id} feature metadata changed")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} feature is not a safe NumPy array") from error
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.shape != (FEATURE_DIMENSION,)
        or loaded.dtype != FEATURE_DTYPE
        or not loaded.flags.c_contiguous
        or not np.isfinite(loaded).all()
        or loaded.nbytes != FEATURE_NBYTES
    ):
        raise ValueError(f"{sample_id} feature array is invalid")
    array = np.ascontiguousarray(loaded)
    if path.read_bytes() != _npy_bytes(array):
        raise ValueError(f"{sample_id} feature NPY bytes are non-canonical")
    array_sha = _array_sha256(array)
    if feature.get("array_sha256") != array_sha:
        raise ValueError(f"{sample_id} feature array SHA differs")
    aliases = {
        "commfor_feature_path": relative_path,
        "commfor_feature_sha256": file_sha,
        "commfor_feature_array_sha256": array_sha,
        "commfor_feature_shape": [FEATURE_DIMENSION],
        "commfor_feature_dtype": "float32",
        "commfor_feature_nbytes": FEATURE_NBYTES,
        "commfor_feature_semantics": FEATURE_SEMANTICS,
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} feature alias {key} differs")
    if row.get("artifact_paths") != {"commfor_feature_npy": relative_path}:
        raise ValueError(f"{sample_id} feature artifact-path alias differs")
    return FeatureArtifact(
        sample_id=sample_id,
        path=path,
        file_sha256=file_sha,
        file_bytes=path.stat().st_size,
        array_sha256=array_sha,
        array=array,
    )


def validate_feature_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    feature_dir: Path,
) -> dict[str, FeatureArtifact]:
    feature_dir = _safe_absolute_dir(
        feature_dir, root=repo_root, label="feature directory"
    )
    ids = [
        _require_string(row.get("sample_id"), "latest result sample_id")
        for row in latest_results
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("latest results contain duplicate sample_id")
    entries = list(feature_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("feature inventory contains a non-regular entry")
    expected_names = {f"{sample_id}.npy" for sample_id in ids}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            "feature inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    result: dict[str, FeatureArtifact] = {}
    for row in latest_results:
        sample_id = str(row["sample_id"])
        _validate_score_payload(row, sample_id=sample_id)
        result[sample_id] = _feature_artifact(
            row=row,
            sample_id=sample_id,
            repo_root=repo_root,
            feature_dir=feature_dir,
        )
    return result


def replay_linear_head(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    features: Mapping[str, FeatureArtifact],
    model_root: Path,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay stored features with the head from the full safetensors state."""

    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="independent classifier-head replay",
    )
    import torch
    from torch.nn import functional as functional

    model_root = Path(model_root)
    checkpoint = model_root / str(legacy.CHECKPOINT["filename"])
    if (
        not model_root.is_absolute()
        or model_root.is_symlink()
        or not model_root.is_dir()
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.parent.resolve() != model_root.resolve()
    ):
        raise ValueError("model root or full checkpoint is missing or unsafe")
    if (
        checkpoint.stat().st_size != int(legacy.CHECKPOINT["bytes"])
        or sha256_file(checkpoint) != legacy.CHECKPOINT["sha256"]
    ):
        raise ValueError("full Community Forensics checkpoint changed")
    state, schema = legacy_audit._checkpoint_schema(
        checkpoint,
        torch_module=torch,
    )
    if (
        schema.get("items_sha256")
        != EXPECTED_MODEL_CONTRACT["checkpoint_schema_sha256"]
    ):
        raise ValueError("full checkpoint tensor schema changed")
    weight = state.get("vit.head.weight")
    bias = state.get("vit.head.bias")
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or tuple(weight.shape) != (1, FEATURE_DIMENSION)
        or tuple(bias.shape) != (1,)
        or weight.device.type != "cpu"
        or bias.device.type != "cpu"
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("pinned Community Forensics classifier head changed")
    replay_weight = weight.detach().to(device=device)
    replay_bias = bias.detach().to(device=device)
    max_raw, max_probability = 0.0, 0.0
    replayed = 0
    try:
        with torch.inference_mode():
            for row in latest_results:
                sample_id = str(row["sample_id"])
                artifact = features.get(sample_id)
                if artifact is None:
                    raise ValueError(
                        f"missing feature for classifier replay {sample_id}"
                    )
                feature = (
                    torch.from_numpy(artifact.array)
                    .reshape(1, -1)
                    .to(device=device)
                )
                output = functional.linear(
                    feature,
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
                max_raw = max(max_raw, raw_difference)
                max_probability = max(
                    max_probability,
                    probability_difference,
                )
                if raw_difference > RAW_LOGIT_ABS_TOLERANCE:
                    raise ValueError(
                        f"{sample_id} independent classifier logit mismatch"
                    )
                if probability_difference > PROBABILITY_ABS_TOLERANCE:
                    raise ValueError(
                        f"{sample_id} independent classifier probability mismatch"
                    )
                if (
                    (probability_value > legacy.CLASSIFICATION_THRESHOLD)
                    is not row.get("classification_decision")
                ):
                    raise ValueError(
                        f"{sample_id} independent classifier decision mismatch"
                    )
                replayed += 1
                del feature, output, probability
    finally:
        del replay_weight, replay_bias
    if replayed != len(latest_results):
        raise ValueError("independent classifier replay coverage is incomplete")
    return {
        "status": "independent_classifier_head_replay_passed",
        "features_replayed": replayed,
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "checkpoint_schema_sha256": schema["items_sha256"],
        "weight_key": "vit.head.weight",
        "bias_key": "vit.head.bias",
        "device": device_text,
        "runtime": runtime,
        "recorded_runtime_exact_match": True,
        "raw_logit_abs_tolerance": RAW_LOGIT_ABS_TOLERANCE,
        "probability_abs_tolerance": PROBABILITY_ABS_TOLERANCE,
        "max_raw_logit_abs_difference": max_raw,
        "max_probability_abs_difference": max_probability,
    }


def _feature_inventory_sha256(
    features: Mapping[str, FeatureArtifact],
) -> str:
    rows = [
        {
            "sample_id": sample_id,
            "path": artifact.path.resolve().as_posix(),
            "file_sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "array_sha256": artifact.array_sha256,
        }
        for sample_id, artifact in sorted(features.items())
    ]
    return hashlib.sha256(stable_json(rows).encode()).hexdigest()


def _capture_evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    features: Mapping[str, FeatureArtifact],
    primary_snapshot: Mapping[str, str] | None = None,
) -> dict[str, str]:
    current = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if primary_snapshot is not None and current != dict(primary_snapshot):
        raise ValueError("run evidence changed while it was being validated")
    for artifact in features.values():
        if (
            artifact.path.is_symlink()
            or not artifact.path.is_file()
            or artifact.path.stat().st_size != artifact.file_bytes
            or sha256_file(artifact.path) != artifact.file_sha256
        ):
            raise ValueError("feature evidence changed while it was being validated")
    return {
        **current,
        "feature_inventory_sha256": _feature_inventory_sha256(features),
    }


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    bundle_mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    selection_visibility: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": (
            "analyze_community_forensics_balanced.py"
        ),
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": bundle_mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": FROZEN_PROFILE,
        "score_spec": _score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
        "selection_visibility_census": dict(selection_visibility),
        "same_device_feature_head_replays": coverage["valid_images"],
        "coverage": dict(coverage),
    }
    if set(summary) != {*required, "generated_at"}:
        raise ValueError("stored run summary key set changed")
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"stored run summary {key} mismatch")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    _reject_unsupported_claims(summary, "summary")
    _reject_nonfinite_numbers(summary, "summary")


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    mode: str,
) -> RunBundle:
    runner = _assert_runner_contract_exports()
    root = repo_root.resolve()
    run_id = runner._valid_run_id(run_id)
    results_root = _resolve_results_root(results_dir, root)
    run_dir = _resolve_run_dir(results_root, run_id)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    primary_snapshot = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    manifest = _load_json(manifest_path, f"{mode} run manifest")
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
        raise ValueError("smoke requires exactly one attempt per selected image")
    for row in latest:
        if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
            raise ValueError("latest result coverage is not successful")
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    immutable_outputs = _require_mapping(
        immutable.get("outputs"), "immutable outputs"
    )
    feature_dir = _safe_repo_path(
        immutable_outputs.get("feature_dir"),
        repo_root=root,
        label="immutable feature directory",
        require_file=False,
    )
    canonical_feature_dir = (
        root
        / "outputs"
        / "opensource"
        / "community_forensics"
        / run_id
        / "commfor_features"
    ).resolve()
    if feature_dir != canonical_feature_dir:
        raise ValueError("immutable feature directory is not canonical")
    expected_outputs = {
        "run_dir": repo_relative(run_dir, root),
        "results_path": repo_relative(results_path, root),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "summary_sha256": sha256_file(summary_path),
        "feature_dir": repo_relative(feature_dir, root),
        "feature_files": len(selected),
    }
    if set(outputs) != set(expected_outputs):
        raise ValueError("manifest outputs key set changed")
    for key, expected in expected_outputs.items():
        if outputs.get(key) != expected:
            raise ValueError(f"manifest outputs {key} mismatch")
    for key, expected in immutable_outputs.items():
        if key not in outputs or outputs[key] != expected:
            raise ValueError(f"immutable/manifest output {key} differs")
    summary = _load_json(summary_path, f"{mode} run summary")
    _validate_summary(
        summary=summary,
        bundle_mode=mode,
        run_id=run_id,
        fingerprint=fingerprint,
        contract=contract,
        selection_visibility=_require_mapping(
            immutable.get("selection_visibility_census"),
            "immutable.selection_visibility_census",
        ),
        coverage=coverage,
    )
    features = validate_feature_inventory(
        latest_results=latest,
        repo_root=root,
        feature_dir=feature_dir,
    )
    evidence_snapshot = _capture_evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        features=features,
        primary_snapshot=primary_snapshot,
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
        feature_dir=feature_dir,
        features=features,
        evidence_snapshot=evidence_snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        mode="formal",
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        mode="smoke",
    )


def _verify_bundle_unchanged(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> None:
    """Post-operation TOCTOU revalidate run, features, and canonical inputs."""

    expected = dict(bundle.evidence_snapshot)
    if set(expected) != {
        "manifest_sha256",
        "results_sha256",
        "expected_inputs_sha256",
        "summary_sha256",
        "feature_inventory_sha256",
    }:
        raise ValueError("run evidence snapshot key set changed")
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
            raise ValueError(f"run evidence changed after validation: {key}")
    current_manifest = _load_json(
        bundle.manifest_path, f"{bundle.mode} manifest recheck"
    )
    fingerprint, immutable = _validate_manifest(
        manifest=current_manifest,
        repo_root=repo_root,
        run_id=bundle.run_id,
        expected_mode=bundle.mode,
    )
    if (
        fingerprint != bundle.fingerprint
        or immutable != bundle.immutable
        or current_manifest != bundle.manifest
    ):
        raise ValueError("run manifest changed after validation")
    _release, selected, contract = _rebuild_contract(
        repo_root=repo_root,
        immutable=bundle.immutable,
        expected_mode=bundle.mode,
    )
    if selected != bundle.selected or contract.as_dict() != bundle.contract.as_dict():
        raise ValueError("canonical release selection changed after validation")
    features = validate_feature_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        feature_dir=bundle.feature_dir,
    )
    if _feature_inventory_sha256(features) != expected["feature_inventory_sha256"]:
        raise ValueError("feature evidence changed after validation")


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "Community Forensics Balanced250 metrics require "
            "iterations=1000 seed=20260726"
        )
    metrics = summarize_balanced250_t1(
        bundle.release.inputs,
        bundle.release.panel,
        bundle.release.source_pairs,
        bundle.latest_results,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("shared Balanced250 metrics schema changed")
    if metrics.get("coverage", {}).get("is_complete") is not True:
        raise ValueError("formal Balanced250 metrics are incomplete")
    return metrics


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing = _SMOKE_ROW_IGNORED_FIELDS - set(row)
    if missing:
        raise ValueError(
            f"{sample_id} lacks ignored runtime field {sorted(missing)[0]}"
        )
    result = {
        key: value
        for key, value in row.items()
        if key not in _SMOKE_ROW_IGNORED_FIELDS
    }
    feature = _feature_mapping(row, sample_id=sample_id)
    result["commfor_feature"] = {
        key: value
        for key, value in feature.items()
        if key not in _FEATURE_VOLATILE_FIELDS
    }
    for key in (
        "commfor_feature_path",
        "commfor_feature_sha256",
        "commfor_feature_array_sha256",
    ):
        result.pop(key, None)
    result.pop("artifact_paths", None)
    return result


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_features: Mapping[str, FeatureArtifact],
    replay_features: Mapping[str, FeatureArtifact],
) -> dict[str, Any]:
    def unique(
        rows: Sequence[Mapping[str, Any]], label: str
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"), f"{label} row {index} sample_id"
            )
            if sample_id in result:
                raise ValueError(f"{label} has duplicate sample_id {sample_id}")
            result[sample_id] = row
        if not result:
            raise ValueError(f"{label} is empty")
        return result

    reference = unique(reference_rows, "reference")
    replay = unique(replay_rows, "replay")
    if set(reference) != set(replay):
        raise ValueError("smoke result coverage differs")
    if set(reference_features) != set(reference):
        raise ValueError("reference feature coverage differs")
    if set(replay_features) != set(replay):
        raise ValueError("replay feature coverage differs")
    max_raw, max_probability, max_feature = 0.0, 0.0, 0.0
    for sample_id in sorted(reference):
        left = reference[sample_id]
        right = replay[sample_id]
        left_projection = _exact_smoke_projection(left)
        right_projection = _exact_smoke_projection(right)
        if left_projection != right_projection:
            differing = sorted(
                {
                    *(set(left_projection) ^ set(right_projection)),
                    *(
                        key
                        for key in set(left_projection) & set(right_projection)
                        if left_projection[key] != right_projection[key]
                    ),
                }
            )
            raise ValueError(
                f"smoke result {sample_id} computational projection differs "
                f"at {differing[:3]}"
            )
        max_raw = max(
            max_raw, abs(float(left["raw_logit"]) - float(right["raw_logit"]))
        )
        max_probability = max(
            max_probability,
            abs(float(left["probability"]) - float(right["probability"])),
        )
        left_artifact = reference_features[sample_id]
        right_artifact = replay_features[sample_id]
        if (
            left_artifact.path.read_bytes() != right_artifact.path.read_bytes()
            or left_artifact.file_sha256 != right_artifact.file_sha256
            or left_artifact.file_bytes != right_artifact.file_bytes
            or left_artifact.array_sha256 != right_artifact.array_sha256
        ):
            raise ValueError(f"smoke feature {sample_id} bytes differ")
        if not np.array_equal(left_artifact.array, right_artifact.array):
            raise ValueError(f"smoke feature {sample_id} values differ")
        max_feature = max(
            max_feature,
            float(
                np.max(
                    np.abs(
                        left_artifact.array.astype(np.float64)
                        - right_artifact.array.astype(np.float64)
                    )
                )
            ),
        )
    if any(value != 0.0 for value in (max_raw, max_probability, max_feature)):
        raise ValueError("smoke comparison is not bit-exact")
    return {
        "images_compared": len(reference),
        "ignored_row_fields": sorted(_SMOKE_ROW_IGNORED_FIELDS),
        "ignored_feature_metadata_fields": sorted(_FEATURE_VOLATILE_FIELDS),
        "exact_computational_projection": True,
        "feature_file_bytes_exact": True,
        "feature_array_exact": True,
        "max_raw_logit_abs_difference": max_raw,
        "max_probability_abs_difference": max_probability,
        "max_ai_score_abs_difference": max_probability,
        "max_feature_abs_difference": max_feature,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_runner_contract_exports()
    if set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS:
        raise ValueError("smoke immutable config key set changed")
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def _configure_frozen_runtime(
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    if not isinstance(device_text, str) or (
        device_text != "cpu"
        and re.fullmatch(r"cuda:[0-9]+", device_text) is None
    ):
        raise ValueError("runtime device must be 'cpu' or an explicit 'cuda:N'")
    runner = _assert_runner_contract_exports()
    configured = runner.configure_runtime(
        device_text, seed=EXPECTED_RUNTIME_SEED
    )
    if (
        not isinstance(configured, tuple)
        or len(configured) != 2
        or not isinstance(configured[1], Mapping)
    ):
        raise ValueError("runner configure_runtime return contract changed")
    device = configured[0]
    runtime = dict(configured[1])
    _validate_runtime_contract(runtime, label="current analysis runtime")
    if runtime.get("device") != device_text:
        raise ValueError("configured runtime silently changed the requested device")
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_text == "cpu":
        if device_type != "cpu" or device_index is not None:
            raise ValueError("configured device object differs from requested CPU")
    elif (
        device_type != "cuda"
        or device_index != int(device_text.split(":", 1)[1])
    ):
        raise ValueError(
            "configured device object differs from requested CUDA device"
        )
    return device, runtime


def _configure_exact_recorded_runtime(
    *,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
    label: str,
) -> tuple[Any, dict[str, Any]]:
    recorded = _validate_runtime_contract(recorded_runtime, label=label)
    if device_text != recorded.get("device"):
        raise ValueError(f"{label} device differs from immutable runtime")
    device, runtime = _configure_frozen_runtime(device_text)
    if runtime != recorded:
        raise ValueError(f"{label} current runtime differs from immutable runtime")
    return device, runtime


def _actual_runtime_contract(device_text: str) -> dict[str, Any]:
    _device, runtime = _configure_frozen_runtime(device_text)
    return runtime


def _analysis_runtime_contract() -> dict[str, Any]:
    """Bind metric-only packages without changing the inference manifest."""

    inference_runtime = _actual_runtime_contract("cpu")
    actual = {
        name: importlib.metadata.version(name)
        for name in EXPECTED_ANALYSIS_PACKAGE_VERSIONS
    }
    if actual != EXPECTED_ANALYSIS_PACKAGE_VERSIONS:
        raise ValueError(
            "analysis package versions changed: "
            f"{actual} != {EXPECTED_ANALYSIS_PACKAGE_VERSIONS}"
        )
    return {
        "schema_version": (
            "community_forensics_balanced_analysis_runtime_v1"
        ),
        "inference_runtime": inference_runtime,
        "analysis_packages": actual,
    }


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    output_path: Path | None,
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    reference_run_id = runner._valid_run_id(reference_run_id)
    replay_run_id = runner._valid_run_id(replay_run_id)
    if reference_run_id == replay_run_id:
        raise ValueError("smoke comparison requires two distinct run IDs")
    analysis_runtime = _analysis_runtime_contract()
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
    _validate_output_targets(
        {"comparison": output_path},
        protected_files=(
            *_bundle_protected_files(reference, repo_root=repo_root),
            *_bundle_protected_files(replay, repo_root=repo_root),
        ),
        protected_dirs=(
            reference.feature_dir,
            replay.feature_dir,
            Path(str(reference.immutable["source"]["root"])),
            Path(str(replay.immutable["source"]["root"])),
            Path(
                str(
                    reference.immutable["assets"]["model_repository"]["root"]
                )
            ),
            Path(
                str(replay.immutable["assets"]["model_repository"]["root"])
            ),
            Path(str(reference.immutable["assets"]["processor"]["root"])),
            Path(str(replay.immutable["assets"]["processor"]["root"])),
            reference.release.manifest_path.parent,
            replay.release.manifest_path.parent,
        ),
    )
    if _smoke_immutable_projection(
        reference.immutable
    ) != _smoke_immutable_projection(replay.immutable):
        raise ValueError(
            "smoke immutable computational/runtime configurations differ"
        )
    reference_runtime = _validate_runtime_contract(
        reference.immutable.get("runtime"),
        label="reference immutable.runtime",
    )
    replay_runtime = _validate_runtime_contract(
        replay.immutable.get("runtime"),
        label="replay immutable.runtime",
    )
    reference_device = str(reference_runtime["device"])
    replay_device = str(replay_runtime["device"])
    if (
        reference_runtime != replay_runtime
        or reference_device != replay_device
    ):
        raise ValueError("smoke runs do not share the exact recorded runtime")
    if reference.selected != replay.selected or len(reference.selected) != SMOKE_IMAGES:
        raise ValueError("smoke runs do not use the exact 35-image selection")
    reference_head_replay = replay_linear_head(
        latest_results=reference.latest_results,
        features=reference.features,
        model_root=Path(
            str(
                reference.immutable["assets"]["model_repository"]["root"]
            )
        ),
        device_text=reference_device,
        recorded_runtime=reference_runtime,
    )
    replay_head_replay = replay_linear_head(
        latest_results=replay.latest_results,
        features=replay.features,
        model_root=Path(
            str(replay.immutable["assets"]["model_repository"]["root"])
        ),
        device_text=replay_device,
        recorded_runtime=replay_runtime,
    )
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_features=reference.features,
        replay_features=replay.features,
    )
    _verify_bundle_unchanged(reference, repo_root=repo_root)
    _verify_bundle_unchanged(replay, repo_root=repo_root)
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_smoke_comparison_passed",
        "compared_at": utc_now(),
        "analysis_runtime": analysis_runtime,
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
        "reference_independent_linear_head_replay": reference_head_replay,
        "replay_independent_linear_head_replay": replay_head_replay,
        "comparison": comparison,
        "immutable_computational_runtime_config_exact": True,
        "evidence_reverified_after_comparison": True,
    }
    if output_path is not None:
        _write_json_verified(output_path, report, label="comparison output")
    return report


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    model_root: Path,
    processor_root: Path,
    device_text: str,
) -> dict[str, Any]:
    """Freshly replay every canonical JPEG through the released ViT model."""

    if len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh replay requires the full 1,775-image selection")
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"), "immutable.runtime"
    )
    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="fresh full-model replay",
    )
    source = _require_mapping(bundle.immutable.get("source"), "immutable.source")
    assets = _require_mapping(bundle.immutable.get("assets"), "immutable.assets")
    if source_root.resolve() != Path(str(source["root"])).resolve():
        raise ValueError("fresh replay source path differs from manifest")
    model_repository = _require_mapping(
        assets.get("model_repository"),
        "immutable.assets.model_repository",
    )
    processor = _require_mapping(
        assets.get("processor"),
        "immutable.assets.processor",
    )
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    if model_root.resolve() != Path(str(model_repository["root"])).resolve():
        raise ValueError("fresh replay model root differs from manifest")
    if processor_root.resolve() != Path(str(processor["root"])).resolve():
        raise ValueError("fresh replay processor root differs from manifest")
    expected_checkpoint = model_root / str(legacy.CHECKPOINT["filename"])
    if expected_checkpoint.resolve() != Path(
        str(checkpoint["path"])
    ).resolve():
        raise ValueError("fresh replay checkpoint path differs from manifest")

    import torch
    from torch.nn import functional

    pins = legacy_audit._load_runner_pins()
    replay_source = legacy_audit._verify_source_tree(
        source_root,
        pins=pins,
    )
    state, replay_assets = legacy_audit._verify_assets(
        model_root=model_root,
        processor_root=processor_root,
        pins=pins,
        torch_module=torch,
    )
    if replay_source != source:
        raise ValueError("fresh replay source audit differs from manifest")
    if replay_assets != assets:
        raise ValueError("fresh replay asset audit differs from manifest")
    replay_runtime = legacy_audit.ReplayRuntime(
        torch=torch,
        device=device,
        evidence=runtime,
    )
    model = None
    replayed = 0
    max_raw, max_probability, max_feature = 0.0, 0.0, 0.0
    model_load: Mapping[str, Any] | None = None
    try:
        model, model_load = legacy_audit._construct_model(
            state=state,
            runtime=replay_runtime,
        )
        for expected, row in zip(
            bundle.selected, bundle.latest_results, strict=True
        ):
            sample_id = str(expected["sample_id"])
            input_path = _safe_repo_path(
                expected.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                label=f"{sample_id} canonical input",
            )
            if sha256_file(input_path) != expected.get("canonical_sha256"):
                raise ValueError(f"{sample_id} canonical input hash changed")
            preprocessed = legacy_audit.preprocess_image(
                input_path,
                torch_module=torch,
            )
            if preprocessed.audit.get("geometry") != _current_geometry(
                int(expected["width"]), int(expected["height"])
            ):
                raise ValueError(f"{sample_id} independent geometry changed")
            if row.get("preprocess") != preprocessed.audit:
                raise ValueError(f"{sample_id} preprocessing record changed")
            tensor = preprocessed.tensor.unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )
            captured: list[Any] = []

            def capture_classifier(
                _module: Any,
                inputs: tuple[Any, ...],
                _output: Any,
            ) -> None:
                if len(inputs) != 1:
                    raise RuntimeError(
                        "Community Forensics classifier input changed"
                    )
                captured.append(inputs[0].detach().clone())

            hook = model.vit.head.register_forward_hook(capture_classifier)
            try:
                with torch.inference_mode():
                    official_output = model(tensor)
            finally:
                hook.remove()
            if (
                len(captured) != 1
                or tuple(captured[0].shape) != (1, FEATURE_DIMENSION)
                or captured[0].dtype != torch.float32
                or tuple(official_output.shape) != (1, 1)
                or official_output.dtype != torch.float32
            ):
                raise ValueError(
                    f"{sample_id} fresh model/head tensor contract changed"
                )
            with torch.inference_mode():
                manual_output = functional.linear(
                    captured[0],
                    model.vit.head.weight,
                    model.vit.head.bias,
                )
                official_probability = torch.sigmoid(official_output)
                manual_probability = torch.sigmoid(manual_output)
            if (
                not torch.equal(manual_output, official_output)
                or not torch.equal(
                    manual_probability,
                    official_probability,
                )
            ):
                raise ValueError(
                    f"{sample_id} fresh classifier replay differs"
                )
            feature = np.ascontiguousarray(
                captured[0]
                .reshape(FEATURE_DIMENSION)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if not np.array_equal(feature, bundle.features[sample_id].array):
                difference = float(
                    np.max(
                        np.abs(
                            feature.astype(np.float64)
                            - bundle.features[sample_id].array.astype(np.float64)
                        )
                    )
                )
                raise ValueError(
                    f"{sample_id} fresh feature mismatch (max abs {difference})"
                )
            feature_difference = float(
                np.max(
                    np.abs(
                        feature.astype(np.float64)
                        - bundle.features[sample_id].array.astype(np.float64)
                    )
                )
            )
            raw_difference = abs(
                float(official_output.reshape(()).item())
                - float(row["raw_logit"])
            )
            probability_difference = abs(
                float(official_probability.reshape(()).item())
                - float(row["probability"])
            )
            max_feature = max(max_feature, feature_difference)
            max_raw = max(max_raw, raw_difference)
            max_probability = max(max_probability, probability_difference)
            if raw_difference > RAW_LOGIT_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} fresh raw-logit replay mismatch")
            if probability_difference > PROBABILITY_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} fresh probability replay mismatch")
            decision = (
                float(official_probability.reshape(()).item())
                > legacy.CLASSIFICATION_THRESHOLD
            )
            if (
                decision is not row.get("classification_decision")
            ):
                raise ValueError(f"{sample_id} fresh decision replay mismatch")
            replayed += 1
            del (
                tensor,
                captured,
                official_output,
                manual_output,
                official_probability,
                manual_probability,
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if replayed != FORMAL_IMAGES:
        raise ValueError("fresh full-image replay coverage is incomplete")
    if model_load is None:
        raise ValueError("fresh model was not constructed")
    return {
        "status": "fresh_full_image_to_feature_replay_passed",
        "images_replayed": replayed,
        "full_image_forward_per_input": True,
        "full_model_replay": True,
        "classifier_head_only_replay": False,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "asset_bundle_sha256": assets["bundle_sha256"],
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "checkpoint_schema_sha256": checkpoint["schema"]["items_sha256"],
        "preprocess_profile": FROZEN_PROFILE,
        "runtime": runtime,
        "independent_model_load": dict(model_load),
        "raw_logit_abs_tolerance": RAW_LOGIT_ABS_TOLERANCE,
        "probability_abs_tolerance": PROBABILITY_ABS_TOLERANCE,
        "feature_comparison": "numpy.array_equal",
        "max_raw_logit_abs_difference": max_raw,
        "max_probability_abs_difference": max_probability,
        "max_ai_score_abs_difference": max_probability,
        "max_feature_abs_difference": max_feature,
    }


def _json_artifact_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _verify_json_artifact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} changed after write")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} changed after write")


def _write_json_verified(path: Path, value: Any, *, label: str) -> None:
    expected_sha256 = _json_artifact_sha256(value)
    atomic_write_json(path, value)
    _verify_json_artifact(
        path, expected_sha256=expected_sha256, label=label
    )


def _validate_output_targets(
    outputs: Mapping[str, Path | None],
    *,
    protected_files: Sequence[Path],
    protected_dirs: Sequence[Path],
) -> None:
    resolved_outputs = {
        name: Path(path).resolve()
        for name, path in outputs.items()
        if path is not None
    }
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise ValueError("analysis output collision: report paths must be distinct")
    files = {Path(path).resolve() for path in protected_files}
    directories = tuple(Path(path).resolve() for path in protected_dirs)
    for name, output in resolved_outputs.items():
        if output in files or any(
            output == directory or directory in output.parents
            for directory in directories
        ):
            raise ValueError(
                f"analysis output collision: {name} would overwrite run evidence"
            )


def _bundle_protected_files(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> tuple[Path, ...]:
    """Enumerate immutable files that an analysis report may never replace."""

    files = [
        bundle.manifest_path,
        bundle.results_path,
        bundle.expected_path,
        bundle.summary_path,
        *(repo_root / relative for relative in EXPECTED_ADAPTER_SOURCE_PATHS),
    ]
    source = _require_mapping(bundle.immutable.get("source"), "immutable.source")
    assets = _require_mapping(bundle.immutable.get("assets"), "immutable.assets")
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    files.append(Path(_require_string(checkpoint.get("path"), "checkpoint path")))
    for section_name in ("model_repository", "processor"):
        section = _require_mapping(
            assets.get(section_name),
            f"immutable.assets.{section_name}",
        )
        records = _require_mapping(
            section.get("files"),
            f"immutable.assets.{section_name}.files",
        )
        for record in records.values():
            mapping = _require_mapping(record, f"{section_name} file")
            files.append(
                Path(_require_string(mapping.get("path"), "asset file path"))
            )
    source_files = _require_mapping(
        source.get("source_files"), "immutable.source.source_files"
    )
    for record in source_files.values():
        mapping = _require_mapping(record, "immutable source file")
        files.append(Path(_require_string(mapping.get("path"), "source file path")))
    # ``eval_single`` evidence is made of immutable Git blobs
    # (git_object/bytes/sha256), not checked-out filesystem paths.  Callers
    # protect the entire pinned source repository as a directory, so those
    # blob records must not be reinterpreted as output-collision paths.
    eval_single = _require_mapping(
        source.get("eval_single"),
        "immutable.source.eval_single",
    )
    eval_single_files = _require_mapping(
        eval_single.get("files"),
        "immutable.source.eval_single.files",
    )
    for record in eval_single_files.values():
        mapping = _require_mapping(record, "eval_single source blob")
        if set(mapping) != {"git_object", "bytes", "sha256"}:
            raise ValueError("eval_single source blob key set changed")
    return tuple(files)


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    model_root: Path,
    processor_root: Path,
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
    source_record = _require_mapping(
        bundle.immutable.get("source"), "immutable.source"
    )
    asset_record = _require_mapping(
        bundle.immutable.get("assets"), "immutable.assets"
    )
    if source_root.resolve() != Path(str(source_record["root"])).resolve():
        raise ValueError("analysis source path differs from manifest")
    if model_root.resolve() != Path(
        str(asset_record["model_repository"]["root"])
    ).resolve():
        raise ValueError("analysis model root differs from manifest")
    if processor_root.resolve() != Path(
        str(asset_record["processor"]["root"])
    ).resolve():
        raise ValueError("analysis processor root differs from manifest")
    recorded_runtime = _validate_runtime_contract(
        bundle.immutable.get("runtime"), label="immutable.runtime"
    )
    if device_text != recorded_runtime.get("device"):
        raise ValueError("analysis device differs from immutable runtime")
    _validate_output_targets(
        {"metrics": metrics_output_path, "audit": audit_output_path},
        protected_files=_bundle_protected_files(bundle, repo_root=repo_root),
        protected_dirs=(
            bundle.feature_dir,
            source_root,
            model_root,
            processor_root,
            bundle.release.manifest_path.parent,
        ),
    )
    analysis_runtime = _analysis_runtime_contract()
    head_replay = replay_linear_head(
        latest_results=bundle.latest_results,
        features=bundle.features,
        model_root=model_root,
        device_text=device_text,
        recorded_runtime=recorded_runtime,
    )
    metrics = recompute_metrics(bundle, iterations=iterations, seed=seed)
    metrics_sha256 = _json_artifact_sha256(metrics)
    if metrics_output_path is not None:
        _write_json_verified(
            metrics_output_path, metrics, label="metrics output"
        )
    replay_report = (
        replay_model(
            bundle,
            source_root=source_root,
            model_root=model_root,
            processor_root=processor_root,
            device_text=device_text,
        )
        if replay
        else None
    )
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    if metrics_output_path is not None:
        _verify_json_artifact(
            metrics_output_path,
            expected_sha256=metrics_sha256,
            label="metrics output",
        )
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
        "feature_files": len(bundle.features),
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "analysis_runtime": analysis_runtime,
        "independent_linear_head_replay": head_replay,
        "fresh_model_replay": replay_report,
        "method_boundary": {
            "method": "Community Forensics released High-res ViT-S/16 384",
            "architecture": legacy.MODEL_ARCH,
            "preprocess_profile": FROZEN_PROFILE,
            "released_checkpoint_evaluated": True,
            "paper_vs_executable_discrepancy": {
                "paper_prose": (
                    "describes a contrastive-language-image pretraining family"
                ),
                "released_executable": legacy.MODEL_ARCH,
                "operational_choice": (
                    "released_code_and_strict_full_checkpoint"
                ),
                "pretraining_identity_claimed": False,
            },
            "valid_for_t1": True,
            "valid_for_t2": False,
            "fullframe_t2": "not_applicable",
            "license": legacy.LICENSE_RECORD,
            "commercial_clearance": (
                "training_data_commercial_lineage_not_established"
            ),
        },
        "contract_checks": {
            "exact_formal_whole_image_t1_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "released_highres_profile_only": True,
            "pair_rank_rejected": True,
            "t2_joint_dense_claims_rejected": True,
            "source_assets_runtime_adapter_hashes_validated": True,
            "current_metrics_runtime_validated_before_recomputation": True,
            "run_and_canonical_evidence_reverified_after_replay": True,
            "feature_inventory_bytes_sha_array_shape_dtype_finite_validated": True,
            "score_direction_threshold_sigmoid_and_classifier_replay_validated": True,
            "shared_balanced250_metrics_only": True,
        },
        "artifacts": {
            **dict(bundle.evidence_snapshot),
            "metrics_sha256": metrics_sha256,
        },
    }
    if audit_output_path is not None:
        _write_json_verified(audit_output_path, audit, label="audit output")
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_smoke_comparison_output(
    *,
    requested_output: Path | None,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    if requested_output is not None:
        return _anchored(requested_output, repo_root)
    ordered_run_ids = [reference_run_id, replay_run_id]
    fingerprint = hashlib.sha256(
        stable_json(ordered_run_ids).encode("utf-8")
    ).hexdigest()
    basename = (
        f"{SMOKE_COMPARISON_SCHEMA_VERSION}_{fingerprint}.json"
    )
    return results_dir / basename


def _build_parser() -> argparse.ArgumentParser:
    runner = _runner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=getattr(runner, "DEFAULT_RESULTS_DIR", DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--run-id",
        default=getattr(runner, "DEFAULT_FORMAL_RUN_ID", DEFAULT_RUN_ID),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=getattr(runner, "DEFAULT_SOURCE_ROOT", DEFAULT_SOURCE_ROOT),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=getattr(runner, "DEFAULT_MODEL_ROOT", DEFAULT_MODEL_ROOT),
    )
    parser.add_argument(
        "--processor-root",
        type=Path,
        default=getattr(
            runner,
            "DEFAULT_PROCESSOR_ROOT",
            DEFAULT_PROCESSOR_ROOT,
        ),
    )
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
    runner = _assert_runner_contract_exports()
    repo_root = args.repo_root.resolve()
    run_id = runner._valid_run_id(args.run_id)
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    run_dir = _resolve_run_dir(results_dir, run_id)
    if args.compare_smoke_run_id is not None:
        compare_run_id = runner._valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
        ):
            raise ValueError(
                "smoke comparison cannot combine with formal audit options"
            )
        output = _resolve_smoke_comparison_output(
            requested_output=args.comparison_output,
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_run_id,
        )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_run_id,
            output_path=output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError("--comparison-output requires --compare-smoke-run-id")
    metrics_output = (
        _anchored(args.metrics_output, repo_root)
        if args.metrics_output is not None
        else run_dir / "balanced250_metrics.json"
    )
    audit_output = (
        _anchored(args.audit_output, repo_root)
        if args.audit_output is not None
        else run_dir / "independent_audit.json"
    )
    report = analyze(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
        source_root=_anchored(args.source_root, repo_root),
        model_root=_anchored(args.model_root, repo_root),
        processor_root=_anchored(args.processor_root, repo_root),
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
