#!/usr/bin/env python3
"""Independent audit and Balanced250 scoring for frozen NPR runs.

This module intentionally reimplements the critical validation and replay path
instead of trusting runner-produced summaries.  The official detector output is
the float32 sigmoid probability.  Raw logits are reported only in a separate,
explicitly non-replacement diagnostic artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_npr as legacy
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
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
    Capability,
    SelectionSpec,
    load_canonical_release,
    select_inputs,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "npr_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "npr_balanced_smoke_comparison_v2"
RAW_DIAGNOSTIC_SCHEMA_VERSION = "npr_balanced_raw_logit_diagnostic_v1"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"

DEFAULT_RESULTS_DIR = Path("results/opensource/npr")
DEFAULT_RUN_ID = (
    "npr_aigcdetect_progan4class_author_documented_native_even_trim_"
    "balanced250_v1_full1775_20260726"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_HF_SOURCE_ROOT = legacy.DEFAULT_HF_SOURCE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT

FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726

FEATURE_DIMENSION = legacy.FEATURE_DIMENSION
FEATURE_DTYPE = np.dtype(np.float32)
FEATURE_NBYTES = FEATURE_DIMENSION * FEATURE_DTYPE.itemsize
PREPROCESS_PROFILE = (
    "author_documented_aigcdetect_native_even_trim_completion"
)
RAW_LOGIT_ABS_TOLERANCE = 1e-5
PROBABILITY_ABS_TOLERANCE = 1e-7

EXPECTED_RUN_MANIFEST_SCHEMA = "npr_balanced_run_manifest_v2"
EXPECTED_RUN_CONFIG_SCHEMA = "npr_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = "npr_balanced_runtime_summary_v2"
EXPECTED_CPU_PREFLIGHT_SCHEMA = "npr_balanced_cpu_preflight_v1"
EXPECTED_RUNTIME_SEED = legacy.MODEL_SEED
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0/bin/python"
)
EXPECTED_FROZEN_PYCACHE_PREFIX = Path(
    "/root/.cache/claimforge/pycache/npr-balanced-torch2.8.0"
)
EXPECTED_FROZEN_RUNTIME_VERSIONS = {
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
EXPECTED_PYVENV_CFG_SHA256 = (
    "35470b7542154bebe1a55dac3c8760e7638711ff9b166285694e5156186acd06"
)
EXPECTED_ASSET_BUNDLE_SHA256 = (
    "9c19c48e4a3a42f4628b89445e2a39fe564802efbfb8c93854aeae55dfa81b66"
)
CPU_GOLDEN_SAMPLE_ID = "7aeae0f17050bf766257b47d"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/"
    f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
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

EXPECTED_ADAPTER_SOURCE_PATHS = (
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

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}

EXPECTED_LOCAL_VISIBILITY = {
    "local_mouse": {"full": 250, "partial": 0, "none": 0},
    "local_cat": {"full": 239, "partial": 11, "none": 0},
    "local_trash_can": {"full": 211, "partial": 39, "none": 0},
}
FORMAL_SELECTED_IDS_SHA256 = (
    "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
)
SMOKE_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

_RUN_IDENTITY_FIELDS = frozenset(
    {"run_id", "run_manifest_fingerprint", "config_fingerprint"}
)
_SMOKE_VOLATILE_FIELDS = frozenset(
    {
        *_RUN_IDENTITY_FIELDS,
        "completed_at",
        "latency_ms",
        "preprocess_latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_FEATURE_VOLATILE_FIELDS = frozenset(
    {"relative_path", "path", "sha256", "file_sha256", "array_sha256"}
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
_INPUT_VISIBILITY_EVIDENCE_KEY = "edit_visibility_evidence"
_INPUT_PIXEL_CENTER_MAPPING = (
    "identity_then_final_bottom_right_parity_trim"
)
_INPUT_GT_MASK_KINDS = frozenset(
    {"all_zero", "not_applicable", "exact_diff"}
)


@dataclass(frozen=True)
class FeatureArtifact:
    """One independently validated lossless NPR penultimate feature."""

    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: str
    array: np.ndarray


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent NPR preprocessing materialization."""

    tensor: Any
    residual: Any
    decoded_rgb: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class RunBundle:
    """All validated inputs and artifacts belonging to one frozen run."""

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
    """Import the Balanced250 runner lazily to avoid import-time GPU effects."""

    return importlib.import_module("eval.opensource.run_npr_balanced")


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
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} is not boolean")
    return value


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


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def _json_artifact_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _safe_external_file(path: Path, label: str) -> Path:
    candidate = Path(os.path.abspath(path))
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {candidate}")
    parent = candidate.parent
    if parent.is_symlink():
        raise ValueError(f"{label} parent is a symlink")
    return candidate


def _safe_external_dir(path: Path, label: str) -> Path:
    candidate = Path(os.path.abspath(path))
    if candidate.is_symlink() or not candidate.is_dir():
        raise FileNotFoundError(f"missing or unsafe {label}: {candidate}")
    return candidate


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    result = (
        results_dir.resolve()
        if results_dir.is_absolute()
        else (root / results_dir).resolve()
    )
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError("results root escapes repository") from error
    return result


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    runner = _runner()
    valid = runner._valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / valid).resolve()
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


def _reject_input_visibility_evidence(value: Any, label: str) -> None:
    """Allow only the two claim-like fields in the runner's input evidence."""

    evidence = _require_mapping(value, label)
    for raw_key, nested in evidence.items():
        key = str(raw_key)
        normalized = key.lower()
        child_label = f"{label}.{key}"
        if key == "pixel_center_mapping":
            if nested != _INPUT_PIXEL_CENTER_MAPPING:
                raise ValueError(f"{child_label} is an unsupported NPR claim")
        elif key == "gt_mask_kind":
            if (
                not isinstance(nested, str)
                or nested not in _INPUT_GT_MASK_KINDS
            ):
                raise ValueError(f"{child_label} is an unsupported NPR claim")
        else:
            # The exception is deliberately direct-child-only. A model payload
            # cannot hide a pixel/localization claim below an evidence wrapper.
            if normalized in {
                _INPUT_VISIBILITY_EVIDENCE_KEY,
                "pixel_center_mapping",
                "gt_mask_kind",
            }:
                raise ValueError(f"{child_label} is an unsupported NPR claim")
            _reject_unsupported_claims({key: nested}, label)


def _reject_unsupported_claims(value: Any, label: str) -> None:
    """Reject pair, localization, T2, and joint-score claims recursively."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if key == _INPUT_VISIBILITY_EVIDENCE_KEY:
                _reject_input_visibility_evidence(nested, child_label)
            elif normalized == _INPUT_VISIBILITY_EVIDENCE_KEY:
                raise ValueError(f"{child_label} is an unsupported NPR claim")
            elif normalized in _FALSE_DECLARATIONS:
                if nested is not False:
                    raise ValueError(f"{child_label} is an unsupported NPR claim")
            elif normalized in _NULLABLE_DECLARATIONS:
                if nested is not None:
                    raise ValueError(f"{child_label} is an unsupported NPR claim")
            elif normalized in _FORBIDDEN_EXACT_KEYS or normalized.startswith(
                _FORBIDDEN_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported NPR claim")
            else:
                _reject_unsupported_claims(nested, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _reject_unsupported_claims(nested, f"{label}[{index}]")


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
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


def _score_spec() -> ScoreSpec:
    """The paper/main-result score contract; this remains authoritative."""

    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
        threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
    )


def _raw_logit_score_spec() -> ScoreSpec:
    """Preregistered diagnostic only, never the official detector result."""

    return ScoreSpec(
        key="raw_logit",
        direction="higher_means_fake",
        fixed_threshold=0.0,
        threshold_operator=">",
    )


def effective_native_size(width: int, height: int) -> tuple[int, int]:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 1
        or height <= 1
    ):
        raise ValueError("NPR input dimensions must be integers above one")
    return width - width % 2, height - height % 2


def _tensor_sha256(tensor: Any) -> str:
    return _array_sha256(tensor.detach().cpu().contiguous().numpy())


def preprocess_image(
    path: Path,
    *,
    torch_module: ModuleType,
) -> PreprocessedImage:
    """Independently implement RGB normalization and the NPR residual.

    The final bottom/right even-size trim is the benchmark's frozen parity
    completion.  It is corroborated by the pinned HF Space, but is not claimed
    as an executed transform in the pinned upstream GitHub test forward.
    """

    functional = torch_module.nn.functional
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.asarray(rgb, dtype=np.uint8).copy()
    if decoded.shape != (height, width, 3):
        raise ValueError(f"decoded RGB shape is invalid: {decoded.shape}")
    channel_first = np.ascontiguousarray(decoded.transpose(2, 0, 1))
    tensor = torch_module.from_numpy(channel_first).to(
        dtype=torch_module.float32
    )
    tensor = tensor.div(255.0)
    mean = torch_module.tensor(
        legacy.IMAGE_MEAN, dtype=torch_module.float32
    )[:, None, None]
    std = torch_module.tensor(
        legacy.IMAGE_STD, dtype=torch_module.float32
    )[:, None, None]
    tensor = tensor.sub(mean).div(std)
    effective_width, effective_height = effective_native_size(width, height)
    tensor = tensor[:, :effective_height, :effective_width].contiguous()
    down = functional.interpolate(
        tensor.unsqueeze(0),
        scale_factor=0.5,
        mode="nearest",
        recompute_scale_factor=True,
    )
    reconstructed = functional.interpolate(
        down,
        scale_factor=2.0,
        mode="nearest",
        recompute_scale_factor=True,
    )
    residual = (tensor.unsqueeze(0) - reconstructed).squeeze(0).contiguous()
    residual64 = residual.to(dtype=torch_module.float64)
    residual_stats = {
        "minimum": float(residual.min().item()),
        "maximum": float(residual.max().item()),
        "mean": float(residual64.mean().item()),
        "mean_absolute": float(residual64.abs().mean().item()),
        "l2": float(torch_module.linalg.vector_norm(residual64).item()),
        "nonzero_elements": int(torch_module.count_nonzero(residual).item()),
        "elements": int(residual.numel()),
    }
    audit = {
        "profile": PREPROCESS_PROFILE,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
            "trim_last_row_if_height_odd",
            "trim_last_column_if_width_odd",
        ],
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.detach().cpu().numpy().dtype),
        "tensor_sha256": _tensor_sha256(tensor),
        "npr_residual_shape": list(residual.shape),
        "npr_residual_dtype": str(
            residual.detach().cpu().numpy().dtype
        ),
        "npr_residual_sha256": _tensor_sha256(residual),
        "npr_residual_stats": residual_stats,
        "normalization": {
            "mean": list(legacy.IMAGE_MEAN),
            "std": list(legacy.IMAGE_STD),
        },
    }
    return PreprocessedImage(
        tensor=tensor,
        residual=residual,
        decoded_rgb=decoded,
        audit=audit,
    )


def _load_exact_gt(
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> np.ndarray:
    sample_id = _require_string(input_row.get("sample_id"), "sample_id")
    if input_row.get("gt_mask_kind") != "exact_diff":
        raise ValueError(f"{sample_id} does not have exact-diff GT")
    width = int(input_row["width"])
    height = int(input_row["height"])
    path = _safe_repo_path(
        input_row.get("gt_mask_path"),
        repo_root=repo_root,
        label=f"{sample_id} exact-diff GT",
    )
    if sha256_file(path) != input_row.get("gt_mask_sha256"):
        raise ValueError(f"{sample_id} exact-diff GT hash changed")
    with Image.open(path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError(f"{sample_id} exact-diff GT encoding changed")
        pixels = np.asarray(opened, dtype=np.uint8).copy()
    if pixels.shape != (height, width) or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} exact-diff GT pixels changed")
    positives = int(np.count_nonzero(pixels == 255))
    if positives <= 0 or positives != input_row.get("gt_positive_pixels"):
        raise ValueError(f"{sample_id} exact-diff positive count changed")
    return pixels


def _independent_visibility_diagnostic(
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recompute only input visibility; this is not model localization."""

    sample_id = _require_string(input_row.get("sample_id"), "sample_id")
    width = int(input_row["width"])
    height = int(input_row["height"])
    effective_width, effective_height = effective_native_size(width, height)
    geometry = {
        "profile_id": PREPROCESS_PROFILE,
        "effective_native_xyxy": [0, 0, effective_width, effective_height],
        "decoded_size": [width, height],
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "pixel_center_mapping": _INPUT_PIXEL_CENTER_MAPPING,
    }
    gt_kind = input_row.get("gt_mask_kind")
    if gt_kind == "all_zero":
        if (
            input_row.get("kind") != "real"
            or input_row.get("label") != 0
            or input_row.get("gt_mask_path") is not None
            or input_row.get("gt_mask_sha256") is not None
            or input_row.get("gt_positive_pixels") != 0
        ):
            raise ValueError(f"{sample_id} real GT contract changed")
        return {
            "edit_visibility": "not_applicable",
            "edit_visible_gt_fraction": None,
            "edit_visibility_evidence": {
                **geometry,
                "gt_mask_kind": "all_zero",
                "basis": "authentic_input_has_no_edit",
            },
        }
    if gt_kind == "not_applicable":
        if (
            input_row.get("kind") != "forged"
            or input_row.get("label") != 1
            or input_row.get("condition_family")
            != "full_frame_conditional_edit"
            or input_row.get("gt_mask_path") is not None
            or input_row.get("gt_mask_sha256") is not None
            or input_row.get("gt_positive_pixels") is not None
        ):
            raise ValueError(f"{sample_id} full-frame GT contract changed")
        return {
            "edit_visibility": "not_applicable",
            "edit_visible_gt_fraction": None,
            "edit_visibility_evidence": {
                **geometry,
                "gt_mask_kind": "not_applicable",
                "basis": "fullframe_condition_has_no_local_GT",
            },
        }
    if gt_kind != "exact_diff":
        raise ValueError(f"{sample_id} has unsupported visibility GT kind")
    mask = _load_exact_gt(input_row, repo_root=repo_root)
    total = int(np.count_nonzero(mask == 255))
    visible = int(
        np.count_nonzero(mask[:effective_height, :effective_width] == 255)
    )
    fraction = visible / total
    category = (
        "none" if visible == 0 else "full" if visible == total else "partial"
    )
    return {
        "edit_visibility": category,
        "edit_visible_gt_fraction": fraction,
            "edit_visibility_evidence": {
                **geometry,
                "gt_mask_kind": "exact_diff",
                "total_positive_pixels": total,
                "visible_positive_pixel_centers": visible,
                "removed_positive_pixel_centers": total - visible,
                "basis": (
                    "exact_diff_mask_intersection_with_author_documented_"
                    "native_even_trim_completion"
                ),
            },
    }


def _validate_formal_visibility_census(
    selected: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, dict[str, int]]:
    """Lock the known Balanced250 local-edit visibility census."""

    census: dict[str, Counter[str]] = {
        condition: Counter() for condition in EXPECTED_LOCAL_VISIBILITY
    }
    for row in selected:
        condition = str(row.get("condition"))
        diagnostic = _independent_visibility_diagnostic(
            row, repo_root=repo_root
        )
        if condition in census:
            census[condition][str(diagnostic["edit_visibility"])] += 1
        elif condition == "real":
            if row.get("gt_mask_kind") != "all_zero":
                raise ValueError("real condition GT kind changed")
        elif condition.startswith("fullframe_"):
            if row.get("gt_mask_kind") != "not_applicable":
                raise ValueError("full-frame condition GT kind changed")
        else:
            raise ValueError(f"unexpected condition {condition!r}")
    materialized = {
        condition: {
            category: counts.get(category, 0)
            for category in ("full", "partial", "none")
        }
        for condition, counts in census.items()
    }
    if materialized != EXPECTED_LOCAL_VISIBILITY:
        raise ValueError(
            "NPR Balanced250 native-even visibility census changed: "
            f"{materialized!r}"
        )
    return materialized


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    """Validate stored aliases without performing a cross-device sigmoid."""

    raw_logit = _require_finite(row.get("raw_logit"), f"{sample_id} raw_logit")
    probability = _require_finite(
        row.get("ai_score"), f"{sample_id} ai_score"
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{sample_id} ai_score falls outside [0, 1]")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    aliases = {
        "probability": probability,
        "score": probability,
        "classification_decision": decision,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "score_semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} scoring alias {key} changed")
    expected_classification = {
        "raw_logit": raw_logit,
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
    if row.get("classification") != expected_classification:
        raise ValueError(f"{sample_id} classification aliases changed")
    expected_t1 = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "policy": "official_NPR_AIGC_float32_sigmoid",
    }
    if row.get("t1") != expected_t1:
        raise ValueError(f"{sample_id} T1 aliases changed")
    manual = _require_mapping(
        row.get("manual_replay"), f"{sample_id} manual_replay"
    )
    required_manual = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
    }
    if manual != required_manual:
        raise ValueError(f"{sample_id} manual replay contract changed")
    # Deliberately absent: sigmoid(raw_logit) on the analyzer's current CPU.
    # CUDA/CPU sigmoid can differ by ULPs.  The blocking tail replay below
    # executes on the exact runtime/device recorded in the manifest.


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Recompute the authoritative official-probability Balanced250 result."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("official metrics require the formal 1,775-image run")
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "NPR Balanced250 metrics require iterations=1000 seed=20260726"
        )
    if bundle.contract.score_spec != _score_spec():
        raise ValueError("formal run does not use the official probability score")
    summary = summarize_balanced250_t1(
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
    if summary.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("shared Balanced250 metrics schema changed")
    if summary.get("coverage", {}).get("is_complete") is not True:
        raise ValueError("official Balanced250 metrics are incomplete")
    if summary.get("score_contract") != {
        "score_key": "ai_score",
        "direction": "higher_is_forged",
        "fixed_threshold": 0.5,
        "fixed_threshold_operator": ">",
        "tpr_at_fpr_5_percent_threshold_operator": ">",
        "target_fpr": 0.05,
        "fpr_quantile_method": "higher",
    }:
        raise ValueError("official probability metric score contract changed")
    return summary


def _boundary_and_saturation_diagnostic(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[str, Counter[str]] = {}
    zero_ids: list[str] = []
    one_ids: list[str] = []
    half_ids: list[str] = []
    raw_positive_rounded_nonpositive_ids: list[str] = []
    raw_nonpositive_probability_positive_ids: list[str] = []
    probabilities: list[float] = []
    logits: list[float] = []
    for row in rows:
        sample_id = _require_string(row.get("sample_id"), "result sample_id")
        condition = _require_string(row.get("condition"), "result condition")
        raw_logit = _require_finite(
            row.get("raw_logit"), f"{sample_id} raw_logit"
        )
        probability = _require_finite(
            row.get("ai_score"), f"{sample_id} ai_score"
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{sample_id} ai_score is outside [0, 1]")
        probabilities.append(probability)
        logits.append(raw_logit)
        counts = by_condition.setdefault(condition, Counter())
        if probability == 0.0:
            zero_ids.append(sample_id)
            counts["exact_zero_probability"] += 1
        if probability == 1.0:
            one_ids.append(sample_id)
            counts["exact_one_probability"] += 1
        if probability == 0.5:
            half_ids.append(sample_id)
            counts["exact_half_probability"] += 1
        probability_decision = probability > 0.5
        raw_decision = raw_logit > 0.0
        if raw_decision and not probability_decision:
            raw_positive_rounded_nonpositive_ids.append(sample_id)
            counts["raw_positive_probability_not_above_half"] += 1
        if probability_decision and not raw_decision:
            raw_nonpositive_probability_positive_ids.append(sample_id)
            counts["raw_nonpositive_probability_above_half"] += 1
    expected_conditions = set(FORMAL_COUNTS)
    if set(by_condition) != expected_conditions:
        raise ValueError("diagnostic result condition coverage changed")

    def id_evidence(values: Sequence[str]) -> dict[str, Any]:
        return {
            "count": len(values),
            "ordered_sample_ids_sha256": (
                selected_ids_sha256(values) if values else None
            ),
        }

    return {
        "scope": "same_formal_1775_rows_as_official_probability_metrics",
        "official_decision_authority": {
            "score_key": "ai_score",
            "threshold": 0.5,
            "operator": ">",
        },
        "raw_zero_diagnostic": {
            "score_key": "raw_logit",
            "threshold": 0.0,
            "operator": ">",
            "never_replaces_official_probability_decision": True,
        },
        "probability_saturation": {
            "exact_zero": id_evidence(zero_ids),
            "exact_one": id_evidence(one_ids),
            "exact_half": id_evidence(half_ids),
            "unique_values": len(set(probabilities)),
            "minimum": min(probabilities),
            "maximum": max(probabilities),
        },
        "raw_logit_range": {
            "minimum": min(logits),
            "maximum": max(logits),
        },
        "decision_boundary_disagreements": {
            "raw_positive_probability_not_above_half": id_evidence(
                raw_positive_rounded_nonpositive_ids
            ),
            "raw_nonpositive_probability_above_half": id_evidence(
                raw_nonpositive_probability_positive_ids
            ),
        },
        "by_condition": {
            condition: {
                key: counts.get(key, 0)
                for key in (
                    "exact_zero_probability",
                    "exact_one_probability",
                    "exact_half_probability",
                    "raw_positive_probability_not_above_half",
                    "raw_nonpositive_probability_above_half",
                )
            }
            for condition, counts in sorted(by_condition.items())
        },
    }


def recompute_raw_logit_diagnostic(
    bundle: RunBundle,
    *,
    official_metrics: Mapping[str, Any] | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the preregistered raw-logit sensitivity analysis.

    It intentionally calls the same shared Balanced250 implementation, with
    the same release, exact 1,775 rows, seed, iterations, and cluster bootstrap.
    Only the score field and mathematically corresponding zero threshold differ.
    """

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("raw-logit diagnostic requires the formal 1,775 rows")
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "NPR raw-logit diagnostic requires iterations=1000 seed=20260726"
        )
    official = (
        dict(official_metrics)
        if official_metrics is not None
        else recompute_metrics(bundle, iterations=iterations, seed=seed)
    )
    raw_contract = dataclasses.replace(
        bundle.contract, score_spec=_raw_logit_score_spec()
    )
    if dataclasses.replace(
        raw_contract, score_spec=bundle.contract.score_spec
    ) != bundle.contract:
        raise ValueError("raw diagnostic changed more than the score spec")
    raw_metrics = summarize_balanced250_t1(
        bundle.release.inputs,
        bundle.release.panel,
        bundle.release.source_pairs,
        bundle.latest_results,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=raw_contract,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    if raw_metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("raw shared Balanced250 metrics schema changed")
    expected_raw_score_contract = {
        "score_key": "raw_logit",
        "direction": "higher_is_forged",
        "fixed_threshold": 0.0,
        "fixed_threshold_operator": ">",
        "tpr_at_fpr_5_percent_threshold_operator": ">",
        "target_fpr": 0.05,
        "fpr_quantile_method": "higher",
    }
    if raw_metrics.get("score_contract") != expected_raw_score_contract:
        raise ValueError("raw diagnostic score contract changed")
    for key in (
        "dataset_schema_version",
        "dataset_id",
        "run_id",
        "run_manifest_fingerprint",
        "bootstrap",
        "coverage",
    ):
        if raw_metrics.get(key) != official.get(key):
            raise ValueError(
                f"raw and official summaries do not share exact {key}"
            )
    selected_ids = [str(row["sample_id"]) for row in bundle.selected]
    selection_sha = selected_ids_sha256(selected_ids)
    if (
        bundle.contract.selection.selected_images != FORMAL_IMAGES
        or bundle.contract.selection.selected_ids_sha256 != selection_sha
        or raw_contract.selection.selected_ids_sha256 != selection_sha
        or raw_contract.selection.selected_images != FORMAL_IMAGES
    ):
        raise ValueError("official/raw diagnostic selection binding changed")
    return {
        "schema_version": RAW_DIAGNOSTIC_SCHEMA_VERSION,
        "status": "preregistered_diagnostic_complete",
        "role": (
            "raw_logit_sensitivity_only_not_the_official_NPR_probability_result"
        ),
        "official_probability_result_remains_primary": True,
        "must_not_replace_official_fixed_threshold_result": True,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "same_selection_and_bootstrap_proof": {
            "selected_images": FORMAL_IMAGES,
            "selected_ids_sha256": selection_sha,
            "selected_rows_sha256": _rows_sha256(bundle.selected),
            "official_dataset_contract_sha256": official[
                "run_dataset_contract_sha256"
            ],
            "raw_diagnostic_dataset_contract_sha256": raw_metrics[
                "run_dataset_contract_sha256"
            ],
            "only_contract_difference": "score_spec",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "bootstrap": raw_metrics["bootstrap"],
            "coverage": raw_metrics["coverage"],
        },
        "probability_saturation_and_boundary": (
            _boundary_and_saturation_diagnostic(bundle.latest_results)
        ),
        "metrics": raw_metrics,
    }


def _verify_source_tree(source_root: Path) -> dict[str, Any]:
    """Independently verify the pinned upstream GitHub source tree."""

    root = _safe_external_dir(source_root, "NPR source root")
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("NPR source commit changed")
    tracked_dirty = _git_value(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_dirty:
        raise ValueError("NPR tracked source tree is dirty")
    source_files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in legacy.SOURCE_FILES.items():
        path = _safe_external_file(root / relative, f"NPR source {relative}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"NPR source {relative} hash changed")
        source_files[relative] = {
            "path": str(path.resolve()),
            "sha256": expected_sha,
        }
    bundled = _safe_external_file(
        root / str(legacy.CHECKPOINT["repo_relative_path"]),
        "repository-bundled NPR checkpoint",
    )
    if (
        sha256_file(bundled) != legacy.CHECKPOINT["sha256"]
        or bundled.stat().st_size != legacy.CHECKPOINT["bytes"]
    ):
        raise ValueError("repository-bundled NPR checkpoint changed")
    history = _git_value(
        root,
        "log",
        "--format=%H",
        "--",
        str(legacy.CHECKPOINT["repo_relative_path"]),
    )
    if history != legacy.CHECKPOINT["introduced_commit"]:
        raise ValueError("NPR checkpoint introduction history changed")
    present_licenses = [
        name
        for name in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE")
        if (root / name).exists()
    ]
    if present_licenses:
        raise ValueError("frozen NPR no-license finding changed")
    readme = (root / "README.md").read_text(encoding="utf-8")
    resnet = (root / "networks/resnet.py").read_text(encoding="utf-8")
    if (
        "To deal with images of odd sizes" not in readme
        or "if w%2 == 1 : x = x[:,:,:-1,:]" not in readme
        or "if h%2 == 1 : x = x[:,:,:,:-1]" not in readme
    ):
        raise ValueError("NPR README parity-completion evidence changed")
    uncommented_trim = [
        line
        for line in resnet.splitlines()
        if ("w%2" in line or "h%2" in line)
        and not line.lstrip().startswith("#")
    ]
    if uncommented_trim:
        raise ValueError("pinned GitHub live forward parity behavior changed")
    try:
        status_output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot audit NPR full source inventory") from error
    status_lines = [
        line for line in status_output.splitlines() if line
    ]
    cache_pattern = re.compile(
        r"^\?\? (?:[^/]+/)*__pycache__/[A-Za-z0-9_.-]+\.pyc$"
    )
    non_cache = [
        line for line in status_lines if cache_pattern.fullmatch(line) is None
    ]
    if non_cache:
        raise ValueError(f"NPR source inventory drifted: {non_cache[:3]}")
    result = {
        "repo_url": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": source_files,
        "checkpoint_history": {
            "path": str(bundled),
            "introduced_commit": history,
        },
        "root_license_files": present_licenses,
        "license_record": legacy.LICENSE_RECORD,
    }
    result["inventory"] = {
        "tracked_and_non_cache_untracked_clean": True,
        "untracked_bytecode_caches_ignored": len(status_lines),
        "bytecode_cache_execution": False,
        "loader": "compile_verified_utf8_source_bytes_no_pyc",
    }
    return result


def _verify_hf_source_tree(hf_source_root: Path) -> dict[str, Any]:
    """Verify author deployment evidence without treating it as executable."""

    root = _safe_external_dir(hf_source_root, "NPR HF source root")
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != legacy.HF_SPACE_COMMIT:
        raise ValueError("NPR HF Space commit changed")
    tracked_dirty = _git_value(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_dirty:
        raise ValueError("NPR HF Space tracked source tree is dirty")
    source_files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in legacy.HF_SOURCE_FILES.items():
        path = _safe_external_file(
            root / relative, f"NPR HF source {relative}"
        )
        if sha256_file(path) != expected_sha:
            raise ValueError(f"NPR HF source {relative} hash changed")
        source_files[relative] = {
            "path": str(path.resolve()),
            "sha256": expected_sha,
        }
    app_text = (root / "app.py").read_text(encoding="utf-8")
    evidence = (
        "model_epoch_last_3090.pth",
        "transforms.ToTensor()",
        "transforms.Normalize(mean=[0.485, 0.456, 0.406]",
        "if w%2 == 1: img = img[:, :, :-1,:  ]",
        "if h%2 == 1: img = img[:, :, :  ,:-1]",
        "NPR = img - interpolate(img, 0.5)",
        "x.sigmoid()",
    )
    missing = [item for item in evidence if item not in app_text]
    if missing:
        raise ValueError(f"NPR HF Space evidence changed: missing {missing}")
    if "NPRmodel.eval()" in app_text:
        raise ValueError("frozen HF missing-eval finding changed")
    try:
        status_output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot audit NPR HF full source inventory") from error
    if status_output.strip():
        raise ValueError("NPR HF source inventory is not fully clean")
    result = {
        "space_url": legacy.HF_SPACE_URL,
        "root": str(root),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": source_files,
        "role": (
            "corroborating checkpoint and native-size preprocessing only; "
            "official GitHub test.py eval-mode remains the inference contract"
        ),
        "deployment_mode_defect": {
            "calls_model_eval": False,
            "impact": (
                "BatchNorm remains in train mode with batch-one statistics; "
                "the output is batch/input-composition dependent, mutates "
                "running buffers, and materially differs from checkpoint "
                "eval semantics; it is not reproduced or used as a "
                "sensitivity condition"
            ),
        },
        "supply_chain_note": (
            "original app downloads a mutable GitHub-main pickle checkpoint; "
            "this adapter instead hashes and weights-only loads the pinned "
            "GitHub checkpoint"
        ),
    }
    result["full_inventory_clean"] = True
    return result


def _checkpoint_schema(
    state: Mapping[str, Any],
    *,
    torch_module: ModuleType,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    elements = 0
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(
            value, torch_module.Tensor
        ):
            raise ValueError("NPR checkpoint must map string keys to tensors")
        if value.is_complex():
            raise ValueError(f"NPR checkpoint tensor {key} is complex")
        if value.is_floating_point() and not torch_module.isfinite(
            value
        ).all().item():
            raise ValueError(f"NPR checkpoint tensor {key} is not finite")
        elements += int(value.numel())
        items.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    return {
        "container": f"{type(state).__module__}.{type(state).__name__}",
        "entries": len(items),
        "elements": elements,
        "items_sha256": _json_sha256(items),
    }


def _compile_official_resnet(source_root: Path) -> ModuleType:
    """Load verified source bytes without creating/importing a ``.pyc``."""

    path = _safe_external_file(
        source_root / "networks/resnet.py", "official NPR resnet source"
    )
    if sha256_file(path) != legacy.SOURCE_FILES["networks/resnet.py"]:
        raise ValueError("official NPR resnet source hash changed")
    source = path.read_text(encoding="utf-8")
    module_name = f"_claimforge_npr_audit_{legacy.MODEL_SOURCE_COMMIT[:12]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def verify_assets(
    *,
    source_root: Path,
    hf_source_root: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], ModuleType]:
    """Independently verify source, checkpoint safety/schema, and HF evidence."""

    import torch

    source = _verify_source_tree(source_root)
    hf_source = _verify_hf_source_tree(hf_source_root)
    checkpoint = _safe_external_file(checkpoint_path, "NPR checkpoint")
    if (
        sha256_file(checkpoint) != legacy.CHECKPOINT["sha256"]
        or checkpoint.stat().st_size != legacy.CHECKPOINT["bytes"]
    ):
        raise ValueError("explicit NPR checkpoint changed")
    unsafe = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    )
    if unsafe:
        raise ValueError(f"NPR checkpoint contains unsafe globals: {unsafe}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if type(payload).__name__ != "OrderedDict":
        raise ValueError("NPR checkpoint is not the frozen flat OrderedDict")
    state = payload.copy()
    schema = _checkpoint_schema(state, torch_module=torch)
    if (
        schema["entries"] != legacy.CHECKPOINT["state_entries"]
        or schema["elements"] != legacy.CHECKPOINT["state_elements"]
    ):
        raise ValueError("NPR checkpoint state schema changed")
    module = _compile_official_resnet(Path(source["root"]))
    reference = module.resnet50(num_classes=1)
    if list(state) != list(reference.state_dict()):
        raise ValueError("NPR checkpoint key order differs from model")
    reference.load_state_dict(state, strict=True)
    parameters = sum(
        int(parameter.numel()) for parameter in reference.parameters()
    )
    if parameters != legacy.CHECKPOINT["trainable_parameters"]:
        raise ValueError("NPR trainable parameter count changed")
    del reference
    assets = {
        "checkpoint": {
            **legacy.CHECKPOINT,
            "path": str(checkpoint),
            "actual_bytes": checkpoint.stat().st_size,
            "actual_sha256": sha256_file(checkpoint),
            "serialization_safety": {
                "unsafe_globals": unsafe,
                "weights_only": True,
                "map_location": "cpu",
            },
            "schema": schema,
        },
        "excluded_release_assets": legacy.EXCLUDED_RELEASE_ASSETS,
        "bundle_sha256": _json_sha256(
            {
                "source_commit": legacy.MODEL_SOURCE_COMMIT,
                "source_files": legacy.SOURCE_FILES,
                "hf_space_commit": legacy.HF_SPACE_COMMIT,
                "hf_source_files": legacy.HF_SOURCE_FILES,
                "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
                "checkpoint_schema": schema,
            }
        ),
        "license": legacy.LICENSE_RECORD,
        "redistribution": False,
        "commercial_clearance_claimed": False,
    }
    if assets["bundle_sha256"] != EXPECTED_ASSET_BUNDLE_SHA256:
        raise ValueError("NPR source/checkpoint bundle SHA changed")
    source["hf_space"] = hf_source
    return source, assets, state, module


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


def _feature_mapping(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> dict[str, Any]:
    feature = _require_mapping(
        row.get("npr_feature"), f"{sample_id} NPR feature"
    )
    required = {
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
    if set(feature) != required:
        raise ValueError(f"{sample_id} feature key set changed")
    return feature


def _validate_feature_aliases(
    row: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    sample_id: str,
    array_sha256: str,
) -> None:
    aliases = {
        "npr_feature_path": feature["relative_path"],
        "npr_feature_sha256": feature["sha256"],
        "npr_feature_array_sha256": array_sha256,
        "npr_feature_shape": [FEATURE_DIMENSION],
        "npr_feature_dtype": "float32",
        "npr_feature_nbytes": FEATURE_NBYTES,
        "npr_feature_semantics": (
            "official_fc1_input_after_adaptive_global_average_pool"
        ),
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} feature alias {key} differs")
    if feature["array_sha256"] != array_sha256:
        raise ValueError(f"{sample_id} feature array SHA differs")
    artifact_paths = _require_mapping(
        row.get("artifact_paths"), f"{sample_id} artifact_paths"
    )
    if artifact_paths != {"npr_feature_npy": feature["relative_path"]}:
        raise ValueError(f"{sample_id} artifact path alias differs")


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
    if (
        feature.get("file_bytes") != path.stat().st_size
        or path.stat().st_size != 2176
    ):
        raise ValueError(f"{sample_id} feature file byte-size changed")
    if (
        feature.get("dtype") != "float32"
        or feature.get("shape") != [FEATURE_DIMENSION]
        or feature.get("nbytes") != FEATURE_NBYTES
        or feature.get("finite") is not True
        or feature.get("semantics")
        != "official_fc1_input_after_adaptive_global_average_pool"
        or feature.get("visibility") != "local_only_gitignored_output"
    ):
        raise ValueError(f"{sample_id} feature metadata changed")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{sample_id} feature is not a safe NumPy array"
        ) from error
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
    _validate_feature_aliases(
        row, feature, sample_id=sample_id, array_sha256=array_sha
    )
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
        feature_dir, root=repo_root, label="NPR feature directory"
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


def _feature_inventory_sha256(
    features: Mapping[str, FeatureArtifact],
) -> str:
    records = [
        {
            "sample_id": sample_id,
            "file_sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "array_sha256": artifact.array_sha256,
        }
        for sample_id, artifact in sorted(features.items())
    ]
    return _json_sha256(records)


def _validate_runtime_contract(value: Any, *, label: str) -> dict[str, Any]:
    """Pin the dedicated runtime before trusting numerical replay."""

    runtime = _require_mapping(value, label)
    _reject_nonfinite_numbers(runtime, label)
    device_text = _require_string(runtime.get("device"), f"{label}.device")
    if device_text != "cpu" and re.fullmatch(r"cuda:[0-9]+", device_text) is None:
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
    if device_text.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = _require_mapping(runtime.get("python"), f"{label}.python")
    expected_python = {
        "implementation": "CPython",
        "version": EXPECTED_FROZEN_RUNTIME_VERSIONS["python"],
        "executable": str(
            Path(os.path.abspath(EXPECTED_FROZEN_PYTHON_EXECUTABLE))
        ),
    }
    if python_record != expected_python:
        raise ValueError(f"{label}.python dedicated runtime changed")
    packages = _require_mapping(runtime.get("packages"), f"{label}.packages")
    expected_package_keys = {
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "scikit-learn",
        "scipy",
        "joblib",
        "threadpoolctl",
        "setuptools",
    }
    if set(packages) != expected_package_keys:
        raise ValueError(f"{label}.packages key set changed")
    torch_record = _require_mapping(
        packages.get("torch"), f"{label}.packages.torch"
    )
    if set(torch_record) != {
        "version",
        "distribution_version",
        "cuda_runtime",
        "cudnn_version",
    }:
        raise ValueError(f"{label}.packages torch key set changed")
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
    ):
        raise ValueError(f"{label}.packages torch changed")
    torchvision_record = _require_mapping(
        packages.get("torchvision"), f"{label}.packages.torchvision"
    )
    if (
        set(torchvision_record) != {"version", "distribution_version"}
        or torchvision_record.get("version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torchvision"]
        or torchvision_record.get("distribution_version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["torchvision_distribution"]
    ):
        raise ValueError(f"{label}.packages torchvision changed")
    expected_packages = {
        "numpy": EXPECTED_FROZEN_RUNTIME_VERSIONS["numpy"],
        "Pillow": EXPECTED_FROZEN_RUNTIME_VERSIONS["Pillow"],
        "scikit-learn": EXPECTED_FROZEN_RUNTIME_VERSIONS["scikit-learn"],
        "scipy": EXPECTED_FROZEN_RUNTIME_VERSIONS["scipy"],
        "joblib": EXPECTED_FROZEN_RUNTIME_VERSIONS["joblib"],
        "threadpoolctl": EXPECTED_FROZEN_RUNTIME_VERSIONS["threadpoolctl"],
        "setuptools": EXPECTED_FROZEN_RUNTIME_VERSIONS["setuptools"],
    }
    for name, expected in expected_packages.items():
        if packages.get(name) != expected:
            raise ValueError(f"{label}.packages {name} changed")
    venv = _require_mapping(runtime.get("venv"), f"{label}.venv")
    pyvenv_path = Path(
        _require_string(venv.get("pyvenv_cfg_path"), f"{label}.venv path")
    )
    expected_cfg = EXPECTED_FROZEN_PYTHON_EXECUTABLE.parent.parent / "pyvenv.cfg"
    expected_venv = {
        "prefix": str(expected_cfg.parent.resolve()),
        "base_prefix": "/usr",
        "pyvenv_cfg_path": str(expected_cfg.resolve()),
        "pyvenv_cfg_sha256": EXPECTED_PYVENV_CFG_SHA256,
        "include_system_site_packages": False,
    }
    if (
        venv != expected_venv
        or pyvenv_path.resolve() != expected_cfg.resolve()
        or sha256_file(_safe_external_file(pyvenv_path, "pyvenv.cfg"))
        != EXPECTED_PYVENV_CFG_SHA256
    ):
        raise ValueError(f"{label}.venv isolation changed")
    if (
        runtime.get("seed") != EXPECTED_RUNTIME_SEED
        or runtime.get("preprocess_profile") != PREPROCESS_PROFILE
        or runtime.get("batch_size") != 1
        or runtime.get("inference_dtype") != "float32"
        or runtime.get("feature_dtype") != "float32"
        or runtime.get("autocast") is not False
        or runtime.get("grad_enabled") is not False
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cublas_workspace_config")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("matmul_allow_tf32") is not False
        or runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("minimum_cuda_free_bytes") != 8 * 1024**3
        or runtime.get("external_source_loader")
        != "compile_verified_utf8_source_bytes_no_pyc"
        or runtime.get("bytecode_writes_disabled") is not True
        or runtime.get("process_environment")
        != {
            "PYTHONHASHSEED": str(EXPECTED_RUNTIME_SEED),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(
                EXPECTED_FROZEN_PYCACHE_PREFIX.resolve()
            ),
            "python_dont_write_bytecode": True,
            "sys_pycache_prefix": str(
                EXPECTED_FROZEN_PYCACHE_PREFIX.resolve()
            ),
            "pycache_prefix_initially_empty": True,
        }
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    if (
        cudnn.get("enabled") is not False
        or cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if device_text.startswith("cuda:"):
        cuda = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        if (
            set(cuda)
            != {
                "runtime",
                "device_index",
                "device_name",
                "total_memory_bytes",
                "capability",
            }
            or cuda.get("runtime") != "12.8"
            or cuda.get("device_index") != int(device_text.split(":")[1])
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(cuda.get("capability"), list)
            or len(cuda["capability"]) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in cuda["capability"]
            )
        ):
            raise ValueError(f"{label}.cuda evidence changed")
    runner = _runner()
    validated = runner.validate_runtime_contract(runtime, label=label)
    if validated is not None and dict(validated) != runtime:
        raise ValueError(f"{label} runner validator changed runtime evidence")
    return runtime


def _actual_runtime_contract(
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    runner = _runner()
    configured = runner.configure_runtime(
        device_text, seed=EXPECTED_RUNTIME_SEED
    )
    if (
        not isinstance(configured, tuple)
        or len(configured) != 2
        or not isinstance(configured[1], Mapping)
    ):
        raise ValueError("runner configure_runtime return contract changed")
    device, runtime_value = configured
    runtime = _validate_runtime_contract(
        dict(runtime_value), label="current analysis runtime"
    )
    return device, runtime


@contextlib.contextmanager
def _loaded_model(
    *,
    module: ModuleType,
    state: Mapping[str, Any],
    torch_module: ModuleType,
    device: Any,
) -> Iterator[Any]:
    model = module.resnet50(num_classes=1)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch_module.float32)
    model.eval()
    try:
        yield model
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch_module.cuda.empty_cache()


def _forward_with_feature(
    *,
    model: Any,
    tensor: Any,
    torch_module: ModuleType,
    device: Any,
) -> tuple[Any, Any, Any]:
    captured: list[Any] = []

    def capture(_module: Any, arguments: tuple[Any, ...]) -> None:
        if len(arguments) != 1:
            raise ValueError("NPR fc1 hook arguments changed")
        captured.append(arguments[0].detach())

    hook = model.fc1.register_forward_pre_hook(capture)
    try:
        with torch_module.inference_mode():
            output = model(
                tensor.unsqueeze(0).to(
                    device=device,
                    dtype=torch_module.float32,
                    non_blocking=False,
                )
            )
    finally:
        hook.remove()
    if list(output.shape) != [1, 1]:
        raise ValueError("NPR output shape changed")
    if len(captured) != 1 or list(captured[0].shape) != [1, FEATURE_DIMENSION]:
        raise ValueError("NPR fc1 feature hook changed")
    with torch_module.inference_mode():
        tail = torch_module.nn.functional.linear(
            captured[0], model.fc1.weight, model.fc1.bias
        )
        probability = torch_module.sigmoid(output)
        tail_probability = torch_module.sigmoid(tail)
    if not torch_module.equal(output, tail):
        raise ValueError("same-device NPR feature-to-fc1 replay is not exact")
    if not torch_module.equal(probability, tail_probability):
        raise ValueError("same-device NPR sigmoid replay is not exact")
    return output, probability, captured[0]


def replay_feature_head(
    bundle: RunBundle,
    *,
    source_root: Path,
    hf_source_root: Path,
    checkpoint_path: Path,
    device_text: str,
) -> dict[str, Any]:
    """Replay every persisted feature through fc1+sigmoid on recorded device."""

    device, runtime = _actual_runtime_contract(device_text)
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"), "immutable.runtime"
    )
    if runtime != recorded_runtime:
        raise ValueError("feature replay runtime differs from manifest")
    source, assets, state, module = verify_assets(
        source_root=source_root,
        hf_source_root=hf_source_root,
        checkpoint_path=checkpoint_path,
    )
    if source != bundle.immutable.get("source"):
        raise ValueError("feature replay source differs from manifest")
    if assets != bundle.immutable.get("assets"):
        raise ValueError("feature replay assets differ from manifest")
    import torch

    replayed = 0
    with _loaded_model(
        module=module,
        state=state,
        torch_module=torch,
        device=device,
    ) as model:
        for row in bundle.latest_results:
            sample_id = str(row["sample_id"])
            feature_array = bundle.features[sample_id].array
            feature = torch.from_numpy(feature_array).reshape(1, -1).to(
                device=device, dtype=torch.float32
            )
            with torch.inference_mode():
                output = torch.nn.functional.linear(
                    feature, model.fc1.weight, model.fc1.bias
                )
                probability = torch.sigmoid(output)
            raw = float(output.reshape(()).item())
            score = float(probability.reshape(()).item())
            if raw != float(row["raw_logit"]):
                raise ValueError(f"{sample_id} exact same-device raw replay failed")
            if score != float(row["ai_score"]):
                raise ValueError(
                    f"{sample_id} exact same-device sigmoid replay failed"
                )
            replayed += 1
    if replayed != len(bundle.selected):
        raise ValueError("feature-head replay coverage is incomplete")
    return {
        "status": "same_recorded_device_feature_head_replay_passed",
        "images_replayed": replayed,
        "device": str(device),
        "feature_to_fc1_comparison": "exact_float32_scalar",
        "sigmoid_comparison": "exact_float32_scalar",
        "cross_device_sigmoid_gate": False,
        "runtime": runtime,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "asset_bundle_sha256": assets["bundle_sha256"],
    }


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    hf_source_root: Path,
    checkpoint_path: Path,
    device_text: str,
) -> dict[str, Any]:
    """Freshly replay all 1,775 JPEGs through the complete NPR model."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh replay requires the full formal selection")
    device, runtime = _actual_runtime_contract(device_text)
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"), "immutable.runtime"
    )
    if runtime != recorded_runtime:
        raise ValueError("fresh replay runtime differs from manifest")
    source, assets, state, module = verify_assets(
        source_root=source_root,
        hf_source_root=hf_source_root,
        checkpoint_path=checkpoint_path,
    )
    if source != bundle.immutable.get("source"):
        raise ValueError("fresh replay source differs from manifest")
    if assets != bundle.immutable.get("assets"):
        raise ValueError("fresh replay assets differ from manifest")
    import torch

    replayed = 0
    max_feature_difference = 0.0
    max_raw_difference = 0.0
    max_probability_difference = 0.0
    with _loaded_model(
        module=module,
        state=state,
        torch_module=torch,
        device=device,
    ) as model:
        for expected, row in zip(
            bundle.selected, bundle.latest_results, strict=True
        ):
            sample_id = str(expected["sample_id"])
            path = _safe_repo_path(
                expected.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                label=f"{sample_id} canonical input",
            )
            if sha256_file(path) != expected.get("canonical_sha256"):
                raise ValueError(f"{sample_id} canonical input hash changed")
            prepared = preprocess_image(path, torch_module=torch)
            if row.get("preprocess_profile") != PREPROCESS_PROFILE:
                raise ValueError(f"{sample_id} preprocess profile changed")
            if row.get("preprocess") != prepared.audit:
                raise ValueError(f"{sample_id} preprocess evidence changed")
            output, probability, feature_device = _forward_with_feature(
                model=model,
                tensor=prepared.tensor,
                torch_module=torch,
                device=device,
            )
            fresh_feature = np.ascontiguousarray(
                feature_device.squeeze(0).detach().cpu().numpy(),
                dtype=np.float32,
            )
            stored_feature = bundle.features[sample_id].array
            difference = float(
                np.max(np.abs(fresh_feature - stored_feature))
            )
            max_feature_difference = max(max_feature_difference, difference)
            if not np.array_equal(fresh_feature, stored_feature):
                raise ValueError(
                    f"{sample_id} fresh full-model feature replay failed"
                )
            raw = float(output.reshape(()).item())
            score = float(probability.reshape(()).item())
            raw_difference = abs(raw - float(row["raw_logit"]))
            probability_difference = abs(score - float(row["ai_score"]))
            max_raw_difference = max(max_raw_difference, raw_difference)
            max_probability_difference = max(
                max_probability_difference, probability_difference
            )
            if raw != float(row["raw_logit"]):
                raise ValueError(
                    f"{sample_id} fresh same-device raw replay failed"
                )
            if score != float(row["ai_score"]):
                raise ValueError(
                    f"{sample_id} fresh same-device sigmoid replay failed"
                )
            replayed += 1
            del prepared, output, probability, feature_device
    if replayed != FORMAL_IMAGES:
        raise ValueError("fresh full-model replay coverage is incomplete")
    return {
        "status": "fresh_full_image_replay_passed",
        "images_replayed": replayed,
        "full_image_forward_per_input": True,
        "feature_tail_only_replay": False,
        "device": str(device),
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "asset_bundle_sha256": assets["bundle_sha256"],
        "runtime": runtime,
        "feature_comparison": "numpy.array_equal",
        "raw_logit_comparison": "exact_same_device_float32_scalar",
        "probability_comparison": "exact_same_device_float32_scalar",
        "cross_device_sigmoid_gate": False,
        "max_feature_abs_difference": max_feature_difference,
        "max_raw_logit_abs_difference": max_raw_difference,
        "max_probability_abs_difference": max_probability_difference,
    }


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing = set(_SMOKE_VOLATILE_FIELDS - set(row))
    # ``config_fingerprint`` is a legacy alias and may be absent in v2.
    missing.discard("config_fingerprint")
    if missing:
        raise ValueError(
            f"{sample_id} lacks volatile field {sorted(missing)[0]}"
        )
    result = {
        key: value
        for key, value in row.items()
        if key not in _SMOKE_VOLATILE_FIELDS
    }
    feature = _feature_mapping(row, sample_id=sample_id)
    result["npr_feature"] = {
        key: value
        for key, value in feature.items()
        if key not in _FEATURE_VOLATILE_FIELDS
    }
    for key in (
        "npr_feature_path",
        "npr_feature_sha256",
        "npr_feature_array_sha256",
        "artifact_paths",
    ):
        result.pop(key, None)
    return result


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_features: Mapping[str, FeatureArtifact],
    replay_features: Mapping[str, FeatureArtifact],
) -> dict[str, Any]:
    """Demand exact A/B determinism for rows and persisted feature bytes."""

    def unique(
        rows: Sequence[Mapping[str, Any]], label: str
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"), f"{label} row {index} sample_id"
            )
            if sample_id in result:
                raise ValueError(f"{label} has duplicate {sample_id}")
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
    max_raw = 0.0
    max_probability = 0.0
    max_feature = 0.0
    for sample_id in sorted(reference):
        left = reference[sample_id]
        right = replay[sample_id]
        if _exact_smoke_projection(left) != _exact_smoke_projection(right):
            raise ValueError(
                f"smoke result {sample_id} computational projection differs"
            )
        raw_difference = abs(
            float(left["raw_logit"]) - float(right["raw_logit"])
        )
        probability_difference = abs(
            float(left["ai_score"]) - float(right["ai_score"])
        )
        max_raw = max(max_raw, raw_difference)
        max_probability = max(max_probability, probability_difference)
        left_artifact = reference_features[sample_id]
        right_artifact = replay_features[sample_id]
        if (
            left_artifact.path.read_bytes()
            != right_artifact.path.read_bytes()
            or left_artifact.file_sha256 != right_artifact.file_sha256
            or left_artifact.file_bytes != right_artifact.file_bytes
            or left_artifact.array_sha256 != right_artifact.array_sha256
        ):
            raise ValueError(f"smoke feature {sample_id} bytes differ")
        if not np.array_equal(left_artifact.array, right_artifact.array):
            raise ValueError(f"smoke feature {sample_id} values differ")
        feature_difference = float(
            np.max(np.abs(left_artifact.array - right_artifact.array))
        )
        max_feature = max(max_feature, feature_difference)
    if any(value != 0.0 for value in (max_raw, max_probability, max_feature)):
        raise ValueError("NPR smoke comparison is not bit-exact")
    return {
        "images_compared": len(reference),
        "ignored_row_fields": sorted(_SMOKE_VOLATILE_FIELDS),
        "ignored_feature_metadata_fields": sorted(_FEATURE_VOLATILE_FIELDS),
        "exact_computational_projection": True,
        "feature_file_bytes_exact": True,
        "feature_array_exact": True,
        "raw_logit_abs_tolerance": 0.0,
        "probability_abs_tolerance": 0.0,
        "max_raw_logit_abs_difference": max_raw,
        "max_probability_abs_difference": max_probability,
        "max_feature_abs_difference": max_feature,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    """Rebuild the exact selection directly from the canonical release."""

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
    release = load_canonical_release(
        repo_root, manifest_path, verify_files=True
    )
    if expected_mode == "formal":
        spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    elif expected_mode == "smoke":
        spec = SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            per_condition_limit=SMOKE_PER_CONDITION,
        )
    else:
        raise ValueError(f"unsupported analyzer mode {expected_mode!r}")
    selected = tuple(select_inputs(release, spec))
    rebuilt = build_run_dataset_contract(
        release, spec, selected, score_spec=_score_spec()
    )
    if rebuilt.as_dict() != raw_contract:
        raise ValueError("immutable dataset contract does not rebuild exactly")
    expected_count = FORMAL_IMAGES if expected_mode == "formal" else SMOKE_IMAGES
    if len(selected) != expected_count:
        raise ValueError(f"{expected_mode} selection count changed")
    counts = Counter(str(row["condition"]) for row in selected)
    expected_counts = (
        FORMAL_COUNTS
        if expected_mode == "formal"
        else {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
    )
    if dict(counts) != expected_counts:
        raise ValueError(f"{expected_mode} condition counts changed")
    ids_sha = selected_ids_sha256(
        str(row["sample_id"]) for row in selected
    )
    expected_ids_sha = (
        FORMAL_SELECTED_IDS_SHA256
        if expected_mode == "formal"
        else SMOKE_SELECTED_IDS_SHA256
    )
    if ids_sha != expected_ids_sha:
        raise ValueError(f"{expected_mode} selected IDs changed")
    if (
        immutable.get("mode") != expected_mode
        or immutable.get("score_spec") != _score_spec().as_dict()
        or immutable.get("selected_rows_sha256") != _rows_sha256(selected)
        or immutable.get("selected_ids_sha256") != ids_sha
    ):
        raise ValueError(f"{expected_mode} immutable selection binding changed")
    odd_counts = {
        "odd_width": sum(int(row["width"]) % 2 == 1 for row in selected),
        "odd_height": sum(int(row["height"]) % 2 == 1 for row in selected),
        "both_odd": sum(
            int(row["width"]) % 2 == 1 and int(row["height"]) % 2 == 1
            for row in selected
        ),
        "either_odd": sum(
            int(row["width"]) % 2 == 1 or int(row["height"]) % 2 == 1
            for row in selected
        ),
    }
    if immutable.get("odd_dimension_counts") != odd_counts:
        raise ValueError(f"{expected_mode} odd-dimension binding changed")
    if expected_mode == "formal":
        _validate_formal_visibility_census(selected, repo_root=repo_root)
    return release, selected, rebuilt


def _assert_runner_contract_exports() -> Any:
    runner = _runner()
    expected_scalars = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_SEED": EXPECTED_RUNTIME_SEED,
        "SCORE_SPEC": _score_spec(),
        "FORMAL_COUNTS": FORMAL_COUNTS,
    }
    for name, expected in expected_scalars.items():
        if getattr(runner, name, None) != expected:
            raise ValueError(f"runner export {name} differs from analyzer pin")
    if tuple(getattr(runner, "ADAPTER_SOURCE_PATHS", ())) != (
        EXPECTED_ADAPTER_SOURCE_PATHS
    ):
        raise ValueError("runner adapter source inventory differs")
    preprocess = getattr(runner, "PREPROCESS_CONTRACT", None)
    expected_preprocess = {
        "profile_id": PREPROCESS_PROFILE,
        "profile": PREPROCESS_PROFILE,
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
    if not isinstance(preprocess, Mapping) or dict(preprocess) != (
        expected_preprocess
    ):
        raise ValueError("runner preprocess provenance boundary differs")
    task_scope = getattr(runner, "TASK_SCOPE", None)
    if task_scope != {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }:
        raise ValueError("runner task scope differs")
    artifact = getattr(runner, "ARTIFACT_CONTRACT", None)
    feature = artifact.get("feature") if isinstance(artifact, Mapping) else None
    if (
        not isinstance(feature, Mapping)
        or feature.get("shape") != [FEATURE_DIMENSION]
        or feature.get("dtype") != "float32"
        or feature.get("nbytes") != FEATURE_NBYTES
        or feature.get("finite") is not True
        or feature.get("semantics")
        != "official_fc1_input_after_adaptive_global_average_pool"
    ):
        raise ValueError("runner feature artifact contract differs")
    required_immutable = {
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
    immutable_keys = frozenset(getattr(runner, "IMMUTABLE_CONFIG_KEYS", ()))
    if not required_immutable.issubset(immutable_keys):
        raise ValueError("runner immutable key set lacks required bindings")
    for name in (
        "select_mode_inputs",
        "configure_runtime",
        "validate_runtime_contract",
        "run_cpu_preflight",
        "result_identity",
        "build_immutable_run_config",
    ):
        if not callable(getattr(runner, name, None)):
            raise ValueError(f"runner lacks mandatory function {name}")
    return runner


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    _assert_runner_contract_exports()
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected = set(EXPECTED_ADAPTER_SOURCE_PATHS)
    if set(sources) != expected:
        raise ValueError("immutable adapter source key set changed")
    for relative in sorted(expected):
        record = _require_mapping(
            sources[relative], f"adapter source {relative}"
        )
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} key set changed")
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        if (
            record.get("path") != relative
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"adapter source {relative} evidence changed")


def _expected_local_artifact_policy(repo_root: Path) -> dict[str, Any]:
    """Reconstruct the runner's repository-local, gitignored feature policy."""

    root = repo_root.resolve()
    gitignore = _safe_repo_path(
        ".gitignore",
        repo_root=root,
        label=".gitignore",
    )
    probe = (
        Path("outputs/opensource/npr")
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
        raise ValueError(
            "NPR feature output is not covered by .gitignore"
        ) from error
    evidence = completed.stdout.strip()
    if (
        not evidence.startswith(".gitignore:")
        or "\t" not in evidence
        or not evidence.endswith(probe)
    ):
        raise ValueError("NPR git-ignore evidence changed")
    return {
        "visibility": "local_only",
        "artifact_root": "outputs/opensource/npr",
        "gitignored": True,
        "gitignore_path": ".gitignore",
        "gitignore_sha256": sha256_file(gitignore),
        "git_check_ignore_probe": probe,
        "git_check_ignore_evidence": evidence,
        "publication": False,
        "checkpoint_redistribution": False,
        "commercial_clearance_claimed": False,
    }


def _validate_cpu_preflight(
    value: Any,
    *,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    preflight = _require_mapping(value, "immutable.cpu_preflight")
    if set(preflight) != {
        "performed_before_accelerator_configuration",
        "report",
    }:
        raise ValueError("CPU preflight binding key set changed")
    if preflight["performed_before_accelerator_configuration"] is not True:
        raise ValueError("CPU preflight ordering evidence changed")
    report = _require_mapping(
        preflight.get("report"), "immutable.cpu_preflight.report"
    )
    expected_report_keys = {
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
    if (
        set(report) != expected_report_keys
        or report.get("schema_version") != EXPECTED_CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("CPU preflight provenance changed")
    runtime = _validate_runtime_contract(
        report.get("runtime"), label="CPU preflight runtime"
    )
    if runtime.get("device") != "cpu":
        raise ValueError("CPU preflight runtime is not CPU")
    model_load = _require_mapping(
        report.get("model_load"), "CPU preflight model load"
    )
    expected_model_load = {
        "class_module": (
            f"_claimforge_npr_verified_{legacy.MODEL_SOURCE_COMMIT[:12]}"
        ),
        "class_name": "ResNet",
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
        "feature_dimension": FEATURE_DIMENSION,
        "parameters": legacy.CHECKPOINT["trainable_parameters"],
        "network_access": False,
    }
    if model_load != expected_model_load:
        raise ValueError("CPU preflight model-load evidence changed")
    golden = _require_mapping(report.get("golden"), "CPU preflight golden")
    critical = {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1285,
        "input_height": 1137,
        "effective_width": 1284,
        "effective_height": 1136,
        "trim_bottom": 1,
        "trim_right": 1,
        "tensor_sha256": CPU_GOLDEN_TENSOR_SHA256,
        "npr_residual_sha256": CPU_GOLDEN_RESIDUAL_SHA256,
        "feature_file_sha256": CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_file_bytes": 2176,
        "feature_array_sha256": CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "feature_shape": [FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": FEATURE_NBYTES,
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
    if set(golden) != {*critical, "preprocess"}:
        raise ValueError("CPU golden key set changed")
    for key, expected in critical.items():
        actual = golden.get(key)
        if type(expected) is bool:
            matches = actual is expected
        elif type(expected) is int:
            matches = type(actual) is int and actual == expected
        elif type(expected) is float:
            matches = type(actual) is float and actual == expected
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"CPU golden {key} changed")
    preprocess = _require_mapping(
        golden.get("preprocess"), "CPU golden preprocess"
    )
    expected_preprocess = {
        "profile": PREPROCESS_PROFILE,
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
    if preprocess != expected_preprocess:
        raise ValueError("CPU golden preprocess evidence changed")
    if (
        report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
    ):
        raise ValueError("CPU golden initialized CUDA")
    runner = _runner()
    validator = getattr(runner, "_validate_preflight_report", None)
    if callable(validator):
        validator(report, source=source, assets=assets)


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, dict[str, Any]]:
    runner = _assert_runner_contract_exports()
    run_id = runner._valid_run_id(run_id)
    required_manifest_keys = {
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
    if set(manifest) != required_manifest_keys:
        raise ValueError("NPR run manifest key set changed")
    if (
        manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("NPR run manifest identity/status changed")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest immutable")
    expected_keys = frozenset(runner.IMMUTABLE_CONFIG_KEYS)
    if set(immutable) != expected_keys:
        raise ValueError("manifest immutable key set changed")
    if (
        immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
    ):
        raise ValueError("manifest immutable identity/mode changed")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"), "manifest fingerprint"
    )
    if fingerprint != _json_sha256(immutable):
        raise ValueError("manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"), repo_root=repo_root
    )
    if immutable.get("model") != runner.MODEL_CONTRACT:
        raise ValueError("immutable model contract changed")
    model = _require_mapping(immutable.get("model"), "immutable.model")
    model_text = stable_json(model)
    for expected in (
        legacy.MODEL_NAME,
        legacy.MODEL_SLUG,
        legacy.MODEL_ARCH,
        legacy.MODEL_REPO_URL,
        legacy.MODEL_SOURCE_COMMIT,
    ):
        if str(expected) not in model_text:
            raise ValueError("immutable model provenance is incomplete")
    if immutable.get("source_completion") != runner.SOURCE_COMPLETION_CONTRACT:
        raise ValueError("immutable source-completion boundary changed")
    if immutable.get("preprocess") != runner.PREPROCESS_CONTRACT:
        raise ValueError("immutable preprocess contract changed")
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score contract changed")
    if immutable.get("task_scope") != runner.TASK_SCOPE:
        raise ValueError("immutable T1-only task scope changed")
    if immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT:
        raise ValueError("immutable feature artifact contract changed")
    if immutable.get("local_artifact_policy") != (
        _expected_local_artifact_policy(repo_root)
    ):
        raise ValueError("immutable local artifact policy changed")
    source_record = _require_mapping(
        immutable.get("source"), "immutable.source"
    )
    assets_record = _require_mapping(
        immutable.get("assets"), "immutable.assets"
    )
    source_root = Path(
        _require_string(source_record.get("root"), "immutable.source.root")
    )
    hf_record = _require_mapping(
        source_record.get("hf_space"), "immutable.source.hf_space"
    )
    hf_source_root = Path(
        _require_string(hf_record.get("root"), "immutable HF root")
    )
    checkpoint_record = _require_mapping(
        assets_record.get("checkpoint"), "immutable.assets.checkpoint"
    )
    checkpoint_path = Path(
        _require_string(
            checkpoint_record.get("path"), "immutable checkpoint path"
        )
    )
    actual_source, actual_assets, _state, _module = verify_assets(
        source_root=source_root,
        hf_source_root=hf_source_root,
        checkpoint_path=checkpoint_path,
    )
    if source_record != actual_source or assets_record != actual_assets:
        raise ValueError("immutable source/assets differ from verified bytes")
    _validate_runtime_contract(
        immutable.get("runtime"), label="immutable.runtime"
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        source=actual_source,
        assets=actual_assets,
    )
    outputs = _require_mapping(
        immutable.get("outputs"), "immutable.outputs"
    )
    if set(outputs) != {
        "run_dir",
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "feature_dir",
    }:
        raise ValueError("immutable output key set changed")
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
    if dataset != expected:
        raise ValueError("manifest dataset evidence changed")


def _validate_execution(
    *,
    manifest: Mapping[str, Any],
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    required = {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
        "same_device_feature_head_sigmoid_replays",
    }
    if set(execution) != required:
        raise ValueError("manifest execution key set changed")
    for key in required:
        _require_nonnegative_int(execution[key], f"execution.{key}")
    if (
        execution["physical_result_rows"] != physical_rows
        or execution["latest_result_rows"] != latest_rows
        or execution["superseded_attempts"] != physical_rows - latest_rows
        or execution["new_errors"] != 0
        or execution["new_successes"] + execution["resume_skips"]
        != selected_images
        or execution["same_device_feature_head_sigmoid_replays"]
        != latest_rows
    ):
        raise ValueError("manifest execution accounting changed")


def _validate_physical_attempts(
    *,
    physical: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    _validate_physical_attempt_history(physical)
    runner = _runner()
    inputs = {str(row["sample_id"]): row for row in selected}
    for index, row in enumerate(physical):
        sample_id = _require_string(
            row.get("sample_id"), f"physical result {index} sample_id"
        )
        expected = inputs.get(sample_id)
        if expected is None:
            raise ValueError(f"physical result {index} has unknown sample_id")
        runner._validate_runner_attempt(
            row,
            input_row=expected,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        visibility = _independent_visibility_diagnostic(
            expected, repo_root=repo_root
        )
        for key, expected_value in visibility.items():
            if row.get(key) != expected_value:
                raise ValueError(f"{sample_id} input visibility changed")
        _reject_unsupported_claims(row, f"physical result {index}")
        _reject_nonfinite_numbers(row, f"physical result {index}")
        if row.get("status") == "ok":
            _validate_score_payload(row, sample_id=sample_id)


def _validate_physical_attempt_history(
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Independently allow error retries only until the first success."""

    statuses_by_sample: dict[str, list[str]] = {}
    for index, attempt in enumerate(attempts):
        sample_id = _require_string(
            attempt.get("sample_id"),
            f"physical result {index} sample_id",
        )
        status = _require_string(
            attempt.get("status"),
            f"physical result {index} status",
        )
        if status not in ("ok", "error"):
            raise ValueError(f"physical result {index} status is invalid")
        statuses_by_sample.setdefault(sample_id, []).append(status)
    for sample_id, statuses in statuses_by_sample.items():
        successes = [
            index for index, status in enumerate(statuses) if status == "ok"
        ]
        if len(successes) > 1:
            raise ValueError(
                f"duplicate successful physical attempts for {sample_id}"
            )
        if successes and successes[0] != len(statuses) - 1:
            raise ValueError(
                f"physical attempt exists after success for {sample_id}"
            )


def _latest_in_selection_order(
    *,
    selected: Sequence[Mapping[str, Any]],
    physical: Sequence[Mapping[str, Any]],
    run_id: str,
    fingerprint: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    indexed = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=_score_spec(),
    )
    coverage_value = summarize_coverage(indexed)
    require_complete_coverage(coverage_value)
    latest = tuple(
        dict(indexed.latest_by_sample_id[str(row["sample_id"])])
        for row in selected
    )
    if len(latest) != len(selected) or any(
        row.get("status") != "ok" or row.get("valid_for_metrics") is not True
        for row in latest
    ):
        raise ValueError("latest result coverage is not completely successful")
    return latest, coverage_value.as_dict()


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    bundle_mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    coverage: Mapping[str, Any],
    odd_dimension_counts: Mapping[str, Any],
    raw_logit_diagnostic: Mapping[str, Any],
    expected_replays: int,
) -> None:
    required = {
        "schema_version": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_npr_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": bundle_mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "score_spec": _score_spec().as_dict(),
        "raw_logit_diagnostic": dict(raw_logit_diagnostic),
        "dataset_contract": contract.as_dict(),
        "odd_dimension_counts": dict(odd_dimension_counts),
        "coverage": dict(coverage),
        "same_device_feature_head_sigmoid_replays": expected_replays,
    }
    if set(summary) != {*required, "generated_at"}:
        raise ValueError("stored runtime summary key set changed")
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"stored runtime summary {key} changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    _reject_unsupported_claims(summary, "summary")
    _reject_nonfinite_numbers(summary, "summary")


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
        raise ValueError("run evidence changed while being validated")
    for artifact in features.values():
        if (
            artifact.path.is_symlink()
            or not artifact.path.is_file()
            or artifact.path.stat().st_size != artifact.file_bytes
            or sha256_file(artifact.path) != artifact.file_sha256
        ):
            raise ValueError("feature evidence changed while being validated")
    return {
        **current,
        "feature_inventory_sha256": _feature_inventory_sha256(features),
    }


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
        "manifest_sha256": sha256_file(
            _require_regular_file(manifest_path, "run manifest")
        ),
        "results_sha256": sha256_file(
            _require_regular_file(results_path, "run results")
        ),
        "expected_inputs_sha256": sha256_file(
            _require_regular_file(expected_path, "expected inputs")
        ),
        "summary_sha256": sha256_file(
            _require_regular_file(summary_path, "runtime summary")
        ),
    }
    manifest = _load_json(manifest_path, f"{mode} run manifest")
    fingerprint, immutable = _validate_manifest(
        manifest=manifest,
        repo_root=root,
        run_id=run_id,
        expected_mode=mode,
    )
    release, selected, contract = _rebuild_contract(
        repo_root=root, immutable=immutable, expected_mode=mode
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
        raise ValueError("smoke requires one physical attempt per input")
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
        root / "outputs" / "opensource" / "npr" / run_id / "features"
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
    if outputs != expected_outputs:
        raise ValueError("manifest output evidence changed")
    for key, expected in immutable_outputs.items():
        if outputs.get(key) != expected:
            raise ValueError(f"immutable/manifest output {key} differs")
    summary = _load_json(summary_path, f"{mode} runtime summary")
    _validate_summary(
        summary=summary,
        bundle_mode=mode,
        run_id=run_id,
        fingerprint=fingerprint,
        contract=contract,
        coverage=coverage,
        odd_dimension_counts=_require_mapping(
            immutable.get("odd_dimension_counts"),
            "immutable odd-dimension counts",
        ),
        raw_logit_diagnostic=_require_mapping(
            _require_mapping(immutable.get("model"), "immutable model").get(
                "raw_logit_diagnostic"
            ),
            "immutable raw-logit diagnostic",
        ),
        expected_replays=len(latest),
    )
    features = validate_feature_inventory(
        latest_results=latest,
        repo_root=root,
        feature_dir=feature_dir,
    )
    evidence = _capture_evidence_snapshot(
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
        evidence_snapshot=evidence,
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
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> None:
    expected = dict(bundle.evidence_snapshot)
    required = {
        "manifest_sha256",
        "results_sha256",
        "expected_inputs_sha256",
        "summary_sha256",
        "feature_inventory_sha256",
    }
    if set(expected) != required:
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
    manifest = _load_json(
        bundle.manifest_path, f"{bundle.mode} manifest recheck"
    )
    fingerprint, immutable = _validate_manifest(
        manifest=manifest,
        repo_root=repo_root,
        run_id=bundle.run_id,
        expected_mode=bundle.mode,
    )
    if (
        fingerprint != bundle.fingerprint
        or immutable != bundle.immutable
        or manifest != bundle.manifest
    ):
        raise ValueError("run manifest changed after validation")
    _release, selected, contract = _rebuild_contract(
        repo_root=repo_root,
        immutable=bundle.immutable,
        expected_mode=bundle.mode,
    )
    if (
        selected != bundle.selected
        or contract.as_dict() != bundle.contract.as_dict()
    ):
        raise ValueError("canonical release selection changed after validation")
    features = validate_feature_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        feature_dir=bundle.feature_dir,
    )
    if _feature_inventory_sha256(features) != expected[
        "feature_inventory_sha256"
    ]:
        raise ValueError("feature evidence changed after validation")


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate_output_targets(
    outputs: Mapping[str, Path | None],
    *,
    repo_root: Path,
    protected_files: Sequence[Path],
    protected_dirs: Sequence[Path],
) -> None:
    targets: dict[str, Path] = {}
    root = repo_root.resolve()
    for label, raw in outputs.items():
        if raw is None:
            continue
        target = _anchored(raw, root)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} output escapes repository") from error
        current = root
        for part in target.relative_to(root).parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ValueError(f"{label} output parent contains a symlink")
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"{label} output target is unsafe")
        targets[label] = target
    if len(set(targets.values())) != len(targets):
        raise ValueError("analysis output paths collide")
    protected = {path.resolve() for path in protected_files}
    for label, target in targets.items():
        if target in protected:
            raise ValueError(f"{label} output would overwrite run evidence")
        for directory in protected_dirs:
            try:
                target.relative_to(directory.resolve())
            except ValueError:
                continue
            raise ValueError(f"{label} output falls inside artifact inventory")


def _write_json_verified(path: Path, value: Any, *, label: str) -> None:
    expected = _json_artifact_sha256(value)
    atomic_write_json(path, value)
    loaded = _load_json(path, label)
    if loaded != value or sha256_file(path) != expected:
        raise ValueError(f"{label} write verification failed")


def _verify_json_artifact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    _load_json(path, label)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} changed after writing")


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
    _device, analysis_runtime = _actual_runtime_contract("cpu")
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
        repo_root=repo_root,
        protected_files=(
            reference.manifest_path,
            reference.results_path,
            reference.expected_path,
            reference.summary_path,
            replay.manifest_path,
            replay.results_path,
            replay.expected_path,
            replay.summary_path,
        ),
        protected_dirs=(reference.feature_dir, replay.feature_dir),
    )
    if _smoke_immutable_projection(
        reference.immutable
    ) != _smoke_immutable_projection(replay.immutable):
        raise ValueError("smoke computational/runtime configurations differ")
    if reference.selected != replay.selected:
        raise ValueError("smoke runs do not use the same exact selection")
    if (
        len(reference.selected) != SMOKE_IMAGES
        or reference.contract.selection.selected_ids_sha256
        != SMOKE_SELECTED_IDS_SHA256
    ):
        raise ValueError("smoke selection is not the frozen 35 images")
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
        "comparison": comparison,
        "immutable_computational_runtime_config_exact": True,
        "evidence_reverified_after_comparison": True,
    }
    if output_path is not None:
        _write_json_verified(output_path, report, label="smoke comparison")
    return report


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    hf_source_root: Path,
    checkpoint_path: Path,
    device_text: str,
    metrics_output_path: Path | None,
    raw_diagnostic_output_path: Path | None,
    audit_output_path: Path | None,
    replay: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=run_id,
    )
    _validate_output_targets(
        {
            "official metrics": metrics_output_path,
            "raw diagnostic": raw_diagnostic_output_path,
            "audit": audit_output_path,
        },
        repo_root=repo_root,
        protected_files=(
            bundle.manifest_path,
            bundle.results_path,
            bundle.expected_path,
            bundle.summary_path,
        ),
        protected_dirs=(bundle.feature_dir,),
    )
    _device, analysis_runtime = _actual_runtime_contract("cpu")
    official_metrics = recompute_metrics(
        bundle, iterations=iterations, seed=seed
    )
    raw_diagnostic = recompute_raw_logit_diagnostic(
        bundle,
        official_metrics=official_metrics,
        iterations=iterations,
        seed=seed,
    )
    metrics_sha = _json_artifact_sha256(official_metrics)
    raw_sha = _json_artifact_sha256(raw_diagnostic)
    if metrics_output_path is not None:
        _write_json_verified(
            metrics_output_path,
            official_metrics,
            label="official probability metrics",
        )
    if raw_diagnostic_output_path is not None:
        _write_json_verified(
            raw_diagnostic_output_path,
            raw_diagnostic,
            label="raw-logit diagnostic",
        )
    feature_head_report = (
        replay_feature_head(
            bundle,
            source_root=source_root,
            hf_source_root=hf_source_root,
            checkpoint_path=checkpoint_path,
            device_text=device_text,
        )
        if replay
        else None
    )
    fresh_replay_report = (
        replay_model(
            bundle,
            source_root=source_root,
            hf_source_root=hf_source_root,
            checkpoint_path=checkpoint_path,
            device_text=device_text,
        )
        if replay
        else None
    )
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    if metrics_output_path is not None:
        _verify_json_artifact(
            metrics_output_path,
            expected_sha256=metrics_sha,
            label="official probability metrics",
        )
    if raw_diagnostic_output_path is not None:
        _verify_json_artifact(
            raw_diagnostic_output_path,
            expected_sha256=raw_sha,
            label="raw-logit diagnostic",
        )
    visibility = _validate_formal_visibility_census(
        bundle.selected, repo_root=repo_root
    )
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "replay_audit_passed" if replay else "artifact_audit_passed"
        ),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": FORMAL_IMAGES,
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": bundle.coverage,
        "feature_files": len(bundle.features),
        "official_metrics": {
            "role": "paper_main_result",
            "score_key": "ai_score",
            "fixed_threshold": 0.5,
            "fixed_threshold_operator": ">",
            "schema_version": official_metrics["schema_version"],
            "bootstrap": official_metrics["bootstrap"],
            "sha256": metrics_sha,
        },
        "raw_logit_diagnostic": {
            "role": "preregistered_sensitivity_only_non_replacement",
            "official_probability_result_remains_primary": True,
            "schema_version": raw_diagnostic["schema_version"],
            "sha256": raw_sha,
            "same_selection_and_bootstrap_proof": raw_diagnostic[
                "same_selection_and_bootstrap_proof"
            ],
        },
        "analysis_runtime": analysis_runtime,
        "same_device_feature_head_replay": feature_head_report,
        "fresh_model_replay": fresh_replay_report,
        "input_visibility_diagnostic": {
            "scope": "local_exact_diff_input_pixels_only",
            "not_model_localization": True,
            "by_local_condition": visibility,
            "pooled": {"full": 700, "partial": 50, "none": 0},
        },
        "method_boundary": {
            "method": "NPR released ProGAN-4class sigmoid detector",
            "paper_protocol_parity_claimed": False,
            "preprocess_profile": PREPROCESS_PROFILE,
            "parity_completion": {
                "github_live_forward_executes_odd_trim": False,
                "github_readme_documents_odd_trim": True,
                "author_hf_space_executes_odd_trim": True,
                "hf_space_is_executable_reference": False,
                "hf_space_exclusion_reason": "missing_model_eval",
                "balanced250_odd_width_images": 81,
                "balanced250_odd_height_images": 376,
                "balanced250_both_odd_images": 13,
                "balanced250_odd_union_images": 444,
            },
            "valid_for_t1": True,
            "valid_for_t2": False,
            "fullframe_t2": "not_applicable",
            "license": legacy.LICENSE_RECORD,
        },
        "contract_checks": {
            "exact_formal_whole_image_t1_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "pair_rank_rejected": True,
            "t2_joint_dense_claims_rejected": True,
            "source_hf_checkpoint_runtime_adapter_hashes_validated": True,
            "run_and_canonical_evidence_reverified_after_replay": True,
            "feature_inventory_bytes_sha_array_shape_dtype_finite_validated": True,
            "static_cross_device_sigmoid_gate_used": False,
            "official_probability_metrics_use_shared_balanced250_only": True,
            "raw_diagnostic_uses_same_shared_balanced250_release_rows_bootstrap": True,
        },
        "artifacts": {
            **dict(bundle.evidence_snapshot),
            "official_metrics_sha256": metrics_sha,
            "raw_logit_diagnostic_sha256": raw_sha,
        },
    }
    if audit_output_path is not None:
        _write_json_verified(audit_output_path, audit, label="NPR audit output")
    return audit


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
        "--hf-source-root",
        type=Path,
        default=getattr(
            runner, "DEFAULT_HF_SOURCE_ROOT", DEFAULT_HF_SOURCE_ROOT
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=getattr(runner, "DEFAULT_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--compare-smoke-run-id")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--raw-diagnostic-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser


def _short_comparison_name(reference: str, replay: str) -> str:
    digest = hashlib.sha256(
        stable_json([reference, replay]).encode()
    ).hexdigest()[:20]
    return f"npr_smoke_comparison_v2_{digest}.json"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runner = _assert_runner_contract_exports()
    repo_root = args.repo_root.resolve()
    run_id = runner._valid_run_id(args.run_id)
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    run_dir = _resolve_run_dir(results_dir, run_id)
    if args.compare_smoke_run_id is not None:
        comparison_id = runner._valid_run_id(args.compare_smoke_run_id)
        if any(
            value is not None
            for value in (
                args.metrics_output,
                args.raw_diagnostic_output,
                args.audit_output,
            )
        ) or args.skip_model_replay:
            raise ValueError(
                "smoke comparison cannot combine with formal audit options"
            )
        output = (
            _anchored(args.comparison_output, repo_root)
            if args.comparison_output is not None
            else results_dir / _short_comparison_name(run_id, comparison_id)
        )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=comparison_id,
            output_path=output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError(
            "--comparison-output requires --compare-smoke-run-id"
        )
    metrics_output = (
        _anchored(args.metrics_output, repo_root)
        if args.metrics_output is not None
        else run_dir / "balanced250_metrics.json"
    )
    raw_output = (
        _anchored(args.raw_diagnostic_output, repo_root)
        if args.raw_diagnostic_output is not None
        else run_dir / "npr_raw_logit_diagnostic.json"
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
        hf_source_root=_anchored(args.hf_source_root, repo_root),
        checkpoint_path=_anchored(args.checkpoint, repo_root),
        device_text=args.device,
        metrics_output_path=metrics_output,
        raw_diagnostic_output_path=raw_output,
        audit_output_path=audit_output,
        replay=not args.skip_model_replay,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
