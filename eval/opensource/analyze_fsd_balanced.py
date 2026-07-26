#!/usr/bin/env python3
"""Independently audit and fully replay an FSD Balanced250 run.

The FSD runner owns inference and append-only result recording.  This module
independently rebuilds the frozen Balanced250 selection, validates every
physical attempt and every persisted 960-dimensional descriptor, recomputes
the shared T1 statistics, compares the two deterministic smoke runs, and (by
default) performs a fresh image-to-score replay of all 1,775 canonical JPEGs.

FSD v1.2 is a whole-image detector.  This analyzer consequently rejects
``pair_rank`` and any dense, T2, or joint-score claim.  Full-frame conditions
remain conditional edits with T2 not applicable; they are not relabelled as
fully synthetic images.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
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

from eval.opensource import run_fsd as legacy
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
    load_canonical_release,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "fsd_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "fsd_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
DEFAULT_RESULTS_DIR = Path("results/opensource/fsd")
DEFAULT_RUN_ID = (
    "fsd_v1_2_0_official_balanced250_v1_full1775_20260726"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_WEIGHTS_DIR = Path(
    "/root/.cache/claimforge/checkpoints/fsd-v1.2.0"
)
FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
RAW_LIKELIHOOD_ABS_TOLERANCE = 1e-9
ZSCORE_ABS_TOLERANCE = 1e-12
AI_SCORE_ABS_TOLERANCE = 1e-12
DESCRIPTOR_DIMENSION = 960
DESCRIPTOR_DTYPE = np.dtype(np.float64)
DESCRIPTOR_NBYTES = DESCRIPTOR_DIMENSION * DESCRIPTOR_DTYPE.itemsize
CPU_GOLDEN_SAMPLE_ID = "5f7535f0b957874982b1b080"
CPU_GOLDEN_IMAGE_SHA256 = (
    "f90c849192fd53e2e9560192d91b5b37a6162f80c14c862e24d37482784b8078"
)
CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256 = (
    "96ee62ffc9e5efd54070f1dd182f3d474305d7508218aad84c2ad9d4690478e1"
)
CPU_GOLDEN_DESCRIPTOR_FILE_SHA256 = (
    "233a1645b1d93d6c97e540c7e7c2f022d948ee861eb316d4e71db2a032fca842"
)
CPU_GOLDEN_RAW_LIKELIHOOD = -289.2140144870369
CPU_GOLDEN_RELEASED_Z_SCORE = -0.34977523419069584
CPU_GOLDEN_AI_SCORE = 0.34977523419069584
EXPECTED_RUN_MANIFEST_SCHEMA = "fsd_balanced_run_manifest_v2"
EXPECTED_RUN_CONFIG_SCHEMA = "fsd_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = "fsd_balanced_runtime_summary_v2"
EXPECTED_CPU_PREFLIGHT_SCHEMA = "fsd_balanced_cpu_preflight_v1"
EXPECTED_RUNTIME_SEED = 20260726
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_MINIMUM_CUDA_FREE_BYTES = 12 * 1024**3
EXPECTED_FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.10.0+cu128",
    "torch_distribution": "2.10.0",
    "numpy": "2.4.3",
    "Pillow": "12.1.1",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
}
EXPECTED_FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/fsd-v1.2.0/bin/python"
)
EXPECTED_ADAPTER_SOURCE_PATHS = (
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_fsd_balanced.py",
    "eval/opensource/analyze_fsd_balanced.py",
    "eval/opensource/run_fsd.py",
    "eval/opensource/analyze_fsd_run.py",
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
        "source_tag_drift",
        "paper_release_drift",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "source",
        "weights",
        "runtime",
        "cpu_preflight",
        "artifact_contract",
        "outputs",
    }
)
EXPECTED_MODEL_CONTRACT = {
    "name": legacy.MODEL_NAME,
    "slug": legacy.MODEL_SLUG,
    "repository": legacy.MODEL_REPO_URL,
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "release_tag": legacy.RELEASE_TAG,
    "license": {
        "spdx": legacy.LICENSE_SPDX,
        "commercial_use": False,
        "share_alike": True,
    },
    "evaluation_claim": legacy.PAPER_RELEASE_DRIFT["evaluation_claim"],
}
EXPECTED_TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}
EXPECTED_PREPROCESS_CONTRACT = {
    "decoder": "Pillow.Image.open_then_convert_L",
    "exif_transpose": False,
    "icc_conversion": False,
    "grayscale_dtype": "uint8",
    "grayscale_range": [0, 255],
    "fre": {
        "kernel_size": 15,
        "output_channels": 8,
        "padding": 7,
        "border_each_side": 7,
    },
    "resize": {
        "rule": "post_FRE_trim_short_side_to_1024",
        "mode": "torch.nn.functional.interpolate_bilinear",
        "rounding": "python_round",
        "align_corners": False,
        "antialias": False,
    },
    "center_crop": {
        "rule": "center_crop_at_most_1024x1024",
        "size": [1024, 1024],
    },
    "scales": {
        "count": 3,
        "mode": "torch_bilinear",
        "align_corners": False,
        "antialias": False,
    },
    "descriptor": {
        "shape": [legacy.FSD_DIMENSION],
        "dtype": "float64",
        "neighborhood": [legacy.FSD_NEIGHBORHOOD, legacy.FSD_NEIGHBORHOOD],
        "solver": "float64_KKT_constrained_least_squares",
        "lambda_regularization": 1e-5,
    },
}
EXPECTED_ARTIFACT_CONTRACT = {
    "descriptor": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.FSD_DIMENSION],
        "dtype": "float64",
        "nbytes": DESCRIPTOR_NBYTES,
        "finite": True,
        "semantics": "official_compute_fsd_before_released_transforms",
        "allow_pickle": False,
    }
}

_RUN_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
    }
)
_SMOKE_ROW_IGNORED_FIELDS = frozenset(
    {
        *_RUN_IDENTITY_FIELDS,
        "completed_at",
        "latency_ms",
        "preprocess_latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_DESCRIPTOR_VOLATILE_FIELDS = frozenset(
    {
        "relative_path",
        "path",
        "sha256",
        "file_sha256",
        "array_sha256",
    }
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
class DescriptorArtifact:
    """One validated, lossless FSD descriptor artifact."""

    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: str
    array: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    """All independently validated artifacts for one formal or smoke run."""

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
    descriptor_dir: Path
    descriptors: Mapping[str, DescriptorArtifact]
    evidence_snapshot: Mapping[str, str]


def _runner() -> Any:
    """Import the parallel Balanced250 runner only when it is needed."""

    return importlib.import_module("eval.opensource.run_fsd_balanced")


def _assert_runner_contract_exports() -> Any:
    """Prove the live runner still exposes the independently pinned contract."""

    runner = _runner()
    expected = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_SEED": EXPECTED_RUNTIME_SEED,
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
    if not callable(getattr(runner, "configure_runtime", None)) or not callable(
        getattr(runner, "validate_runtime_contract", None)
    ):
        raise ValueError("runner lacks mandatory frozen runtime functions")
    return runner


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{label} contains NUL")
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
                row = _require_mapping(
                    _json_loads(line, row_label),
                    row_label,
                )
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
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"missing regular {label}: {path}")
    return path


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
        current = current / part
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
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if not candidate.is_dir():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    return candidate


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    result = (
        results_dir.resolve()
        if results_dir.is_absolute()
        else (repo_root.resolve() / results_dir).resolve()
    )
    try:
        result.relative_to(repo_root.resolve())
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
        current = current / part
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
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_nonfinite_numbers(nested, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        raise ValueError(f"{label} is not finite")


def _reject_unsupported_claims(value: Any, label: str) -> None:
    """Reject pair, localization, T2, and joint-score claims recursively."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(nested, child_label)
            elif normalized in _FALSE_DECLARATIONS:
                if nested is not False:
                    raise ValueError(
                        f"{child_label} is an unsupported FSD claim"
                    )
            elif normalized in _NULLABLE_DECLARATIONS:
                if nested is not None:
                    raise ValueError(
                        f"{child_label} is an unsupported FSD claim"
                    )
            elif normalized in _FORBIDDEN_EXACT_KEYS or normalized.startswith(
                _FORBIDDEN_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported FSD claim")
            else:
                _reject_unsupported_claims(nested, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_unsupported_claims(nested, f"{label}[{index}]")


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    _assert_runner_contract_exports()
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected = set(EXPECTED_ADAPTER_SOURCE_PATHS)
    if set(sources) != expected:
        raise ValueError(
            "immutable.adapter_sources key set mismatch: "
            f"missing={sorted(expected - set(sources))[:3]}, "
            f"extra={sorted(set(sources) - expected)[:3]}"
        )
    for relative in sorted(expected):
        record = _require_mapping(
            sources[relative],
            f"adapter source {relative}",
        )
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} key set changed")
        if record.get("path") != relative:
            raise ValueError(f"adapter source {relative} path mismatch")
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"adapter source {relative} byte-size mismatch")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"adapter source {relative} SHA-256 mismatch")


def _score_spec() -> ScoreSpec:
    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=legacy.AI_SCORE_THRESHOLD,
        threshold_operator=legacy.THRESHOLD_OPERATOR,
    )


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    runner = _runner()
    raw_contract = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_record = _require_mapping(
        raw_contract.get("release"),
        "dataset contract release",
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
            raw_contract.get("selection"),
            "dataset contract selection",
        )
        spec_record = _require_mapping(
            selection.get("spec"),
            "dataset contract selection spec",
        )
        per_condition_limit = spec_record.get("per_condition_limit")
        if per_condition_limit != SMOKE_PER_CONDITION:
            raise ValueError("FSD smoke must select five images per condition")
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
            f"{expected_mode} selection has {len(selected)} images; "
            f"expected exactly {expected_count} images"
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
    return release, selected, rebuilt


def _validate_model_contract(value: Any) -> dict[str, Any]:
    model = _require_mapping(value, "immutable.model")
    if model != EXPECTED_MODEL_CONTRACT:
        raise ValueError("immutable.model exact contract changed")
    return model


def _validate_source_contract(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "immutable.source")
    root = Path(_require_string(source.get("root"), "immutable.source.root"))
    if not root.is_absolute():
        raise ValueError("immutable.source.root is not absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("immutable.source.root is missing or a symlink")
    actual = legacy._verify_repository_contract(root)
    if source != actual:
        raise ValueError("immutable.source differs from frozen source audit")
    return source


def _validate_weights_contract(value: Any) -> dict[str, Any]:
    weights = _require_mapping(value, "immutable.weights")
    root = Path(
        _require_string(weights.get("weights_dir"), "immutable.weights.weights_dir")
    )
    if not root.is_absolute():
        raise ValueError("immutable.weights.weights_dir is not absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("immutable weights directory is missing or a symlink")
    entries = list(root.iterdir())
    expected_names = set(legacy.WEIGHT_FILES)
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("immutable weights directory inventory changed")
    actual = legacy._verify_weights_contract(root)
    if weights != actual:
        raise ValueError("immutable.weights differs from frozen weight audit")
    return weights


def _validate_runtime_contract(value: Any, *, label: str) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    _reject_nonfinite_numbers(runtime, label)
    runner = _assert_runner_contract_exports()
    device_text = _require_string(runtime.get("device"), f"{label}.device")
    if device_text != "cpu" and re.fullmatch(
        r"cuda:[0-9]+",
        device_text,
    ) is None:
        raise ValueError(f"{label}.device is unsupported")
    expected_keys = {
        "device",
        "python",
        "platform",
        "packages",
        "seed",
        "descriptor_dtype",
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
    if device_text.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = _require_mapping(runtime.get("python"), f"{label}.python")
    if set(python_record) != {"implementation", "version", "executable"}:
        raise ValueError(f"{label}.python key set changed")
    if (
        python_record.get("implementation") != "CPython"
        or python_record.get("version")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["python"]
        or Path(str(python_record.get("executable"))).resolve()
        != EXPECTED_FROZEN_PYTHON_EXECUTABLE.resolve()
    ):
        raise ValueError(f"{label}.python dedicated runtime changed")
    packages = _require_mapping(runtime.get("packages"), f"{label}.packages")
    if set(packages) != {
        "torch",
        "numpy",
        "Pillow",
        "scipy",
        "scikit-learn",
    }:
        raise ValueError(f"{label}.packages key set changed")
    torch_record = _require_mapping(
        packages.get("torch"),
        f"{label}.packages.torch",
    )
    if set(torch_record) != {
        "version",
        "distribution_version",
        "cuda_runtime",
        "cudnn_version",
    }:
        raise ValueError(f"{label}.packages.torch key set changed")
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
        or packages.get("numpy")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["numpy"]
        or packages.get("Pillow")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["Pillow"]
        or packages.get("scipy")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["scipy"]
        or packages.get("scikit-learn")
        != EXPECTED_FROZEN_RUNTIME_VERSIONS["scikit-learn"]
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    if not isinstance(runtime.get("platform"), str) or not runtime["platform"]:
        raise ValueError(f"{label}.platform is invalid")
    if (
        runtime.get("seed") != EXPECTED_RUNTIME_SEED
        or runtime.get("descriptor_dtype") != "float64"
        or runtime.get("batch_size") != 1
        or runtime.get("autocast") is not False
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cublas_workspace_config")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("matmul_allow_tf32") is not False
        or runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("minimum_cuda_free_bytes")
        != EXPECTED_MINIMUM_CUDA_FREE_BYTES
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    if set(cudnn) != {
        "enabled",
        "benchmark",
        "deterministic",
        "allow_tf32",
    } or (
        type(cudnn.get("enabled")) is not bool
        or cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if device_text.startswith("cuda:"):
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
            or cuda.get("device_index")
            != int(device_text.split(":", 1)[1])
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
    validated = runner.validate_runtime_contract(runtime, label=label)
    if validated is not None and dict(validated) != runtime:
        raise ValueError(f"{label} runner validator changed the runtime record")
    return runtime


def _validate_preprocess_contract(value: Any) -> dict[str, Any]:
    preprocess = _require_mapping(value, "immutable.preprocess")
    _assert_runner_contract_exports()
    if preprocess != EXPECTED_PREPROCESS_CONTRACT:
        raise ValueError("immutable official preprocess contract changed")
    return preprocess


def _validate_task_scope_contract(value: Any) -> dict[str, Any]:
    task_scope = _require_mapping(value, "immutable.task_scope")
    if task_scope != EXPECTED_TASK_SCOPE:
        raise ValueError("immutable task_scope changed")
    _reject_unsupported_claims(task_scope, "immutable.task_scope")
    return task_scope


def _validate_artifact_contract(value: Any) -> dict[str, Any]:
    artifact_contract = _require_mapping(
        value,
        "immutable.artifact_contract",
    )
    if artifact_contract != EXPECTED_ARTIFACT_CONTRACT:
        raise ValueError("immutable descriptor contract changed")
    return artifact_contract


def _validate_cpu_preflight(
    value: Any,
    *,
    source: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> None:
    preflight = _require_mapping(value, "immutable.cpu_preflight")
    if set(preflight) != {
        "performed_before_accelerator_configuration",
        "report",
    }:
        raise ValueError("immutable.cpu_preflight key set changed")
    if preflight.get("performed_before_accelerator_configuration") is not True:
        raise ValueError("CPU preflight ordering evidence changed")
    report = _require_mapping(
        preflight.get("report"),
        "immutable.cpu_preflight.report",
    )
    report_keys = {
        "schema_version",
        "status",
        "source",
        "weights",
        "runtime",
        "golden",
        "cuda_used",
        "cuda_tensor_operations",
        "dataset_manifest_loaded",
    }
    if set(report) != report_keys:
        raise ValueError("CPU preflight report key set changed")
    if (
        report.get("schema_version") != EXPECTED_CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("weights") != weights
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("CPU preflight report/provenance changed")
    runtime = _validate_runtime_contract(
        report.get("runtime"),
        label="CPU preflight runtime",
    )
    if runtime.get("device") != "cpu":
        raise ValueError("CPU preflight runtime is not CPU")
    golden = _require_mapping(report.get("golden"), "CPU preflight golden")
    expected = {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": (
            "outputs/opensource/balanced250_v1/images/"
            f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
        ),
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": legacy.compute_preprocess_geometry(1800, 1350),
        "descriptor_file_sha256": CPU_GOLDEN_DESCRIPTOR_FILE_SHA256,
        "descriptor_file_bytes": 7808,
        "descriptor_array_sha256": CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256,
        "descriptor_shape": [DESCRIPTOR_DIMENSION],
        "descriptor_dtype": "float64",
        "descriptor_nbytes": DESCRIPTOR_NBYTES,
        "raw_likelihood": CPU_GOLDEN_RAW_LIKELIHOOD,
        "released_z_score": CPU_GOLDEN_RELEASED_Z_SCORE,
        "ai_score": CPU_GOLDEN_AI_SCORE,
        "classification_decision": False,
        "released_is_fake": False,
        "full_image_forward": True,
        "compute_fsd_calls": 1,
        "repeat_descriptor_file_sha256": (
            CPU_GOLDEN_DESCRIPTOR_FILE_SHA256
        ),
        "repeat_descriptor_file_bytes": 7808,
        "repeat_descriptor_array_sha256": (
            CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256
        ),
        "repeat_raw_likelihood": CPU_GOLDEN_RAW_LIKELIHOOD,
        "repeat_released_z_score": CPU_GOLDEN_RELEASED_Z_SCORE,
        "repeat_ai_score": CPU_GOLDEN_AI_SCORE,
        "repeat_classification_decision": False,
        "repeat_released_is_fake": False,
        "repeat_full_image_forward": True,
        "repeat_compute_fsd_calls": 1,
        "repeat_byte_exact": True,
    }
    if set(golden) != set(expected):
        raise ValueError("CPU golden key set changed")
    for key, expected_value in expected.items():
        actual = golden.get(key)
        if type(expected_value) is bool:
            matches = actual is expected_value
        elif type(expected_value) is int:
            matches = type(actual) is int and actual == expected_value
        elif type(expected_value) is float:
            matches = type(actual) is float and actual == expected_value
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(f"CPU golden {key} changed")


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, dict[str, Any]]:
    runner = _assert_runner_contract_exports()
    run_id = runner._valid_run_id(run_id)
    manifest_keys = {
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
    if set(manifest) != manifest_keys:
        raise ValueError("run manifest key set changed")
    if manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported FSD Balanced run manifest")
    if manifest.get("run_id") != run_id:
        raise ValueError("manifest run_id mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("analyzer requires manifest status complete")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest immutable")
    if set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS:
        raise ValueError("manifest immutable key set changed")
    if immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA:
        raise ValueError("immutable schema_version changed")
    if immutable.get("run_id") != run_id:
        raise ValueError("immutable run_id mismatch")
    if immutable.get("mode") != expected_mode:
        raise ValueError(f"analyzer requires immutable.mode={expected_mode}")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint")
    expected_fingerprint = hashlib.sha256(
        stable_json(immutable).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    _validate_model_contract(immutable.get("model"))
    if immutable.get("source_tag_drift") != legacy.SOURCE_TAG_DRIFT:
        raise ValueError("immutable source/tag drift contract changed")
    if immutable.get("paper_release_drift") != legacy.PAPER_RELEASE_DRIFT:
        raise ValueError("immutable paper/release boundary changed")
    _validate_preprocess_contract(immutable.get("preprocess"))
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score_spec changed")
    _validate_task_scope_contract(immutable.get("task_scope"))
    source = _validate_source_contract(immutable.get("source"))
    weights = _validate_weights_contract(immutable.get("weights"))
    _validate_runtime_contract(immutable.get("runtime"), label="immutable.runtime")
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        source=source,
        weights=weights,
    )
    _validate_artifact_contract(immutable.get("artifact_contract"))
    outputs = _require_mapping(immutable.get("outputs"), "immutable.outputs")
    if set(outputs) != {
        "run_dir",
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "descriptor_dir",
    }:
        raise ValueError("immutable.outputs key set changed")
    for field in outputs:
        _safe_repo_path(
            outputs[field],
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
    }
    if set(execution) != expected_keys:
        raise ValueError("manifest execution key set changed")
    for key in execution:
        _require_nonnegative_int(execution[key], f"execution.{key}")
    expected = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
        "new_errors": 0,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise ValueError(f"manifest execution {key} mismatch")
    if execution["new_successes"] + execution["resume_skips"] != selected_images:
        raise ValueError("manifest successful work accounting mismatch")


def _validate_score_payload(row: Mapping[str, Any], *, sample_id: str) -> None:
    raw = _require_finite(
        row.get("raw_likelihood"),
        f"{sample_id} raw_likelihood",
    )
    z_score = _require_finite(
        row.get("released_z_score"),
        f"{sample_id} released_z_score",
    )
    ai_score = _require_finite(row.get("ai_score"), f"{sample_id} ai_score")
    if row.get("score") != ai_score:
        raise ValueError(f"{sample_id} score alias differs")
    expected_z = (raw - legacy.TRAIN_MEAN) / legacy.TRAIN_STD
    if z_score != expected_z:
        raise ValueError(f"{sample_id} released calibration changed")
    if ai_score != -z_score:
        raise ValueError(f"{sample_id} ai_score is not exact -z")
    decision = ai_score > legacy.AI_SCORE_THRESHOLD
    released_decision = z_score < legacy.RELEASED_Z_THRESHOLD
    if decision != released_decision:
        raise ValueError(f"{sample_id} score orientation changed")
    if row.get("classification_decision") is not decision:
        raise ValueError(f"{sample_id} strict decision changed")
    if (
        row.get("classification_threshold") != legacy.AI_SCORE_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} fixed threshold changed")
    if (
        row.get("released_threshold") != legacy.RELEASED_Z_THRESHOLD
        or row.get("released_threshold_operator")
        != legacy.RELEASED_THRESHOLD_OPERATOR
        or row.get("released_is_fake") is not released_decision
    ):
        raise ValueError(f"{sample_id} released decision fields changed")
    if row.get("score_semantics") != "negative_released_FSD_z_score":
        raise ValueError(f"{sample_id} score semantics changed")
    classification = _require_mapping(
        row.get("classification"),
        f"{sample_id} classification",
    )
    classification_expected = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": decision,
        "threshold": legacy.AI_SCORE_THRESHOLD,
        "threshold_operator": legacy.THRESHOLD_OPERATOR,
        "semantics": "higher_is_more_AI_negative_released_z",
    }
    if classification != classification_expected:
        raise ValueError(f"{sample_id} classification aliases changed")
    t1 = _require_mapping(row.get("t1"), f"{sample_id} t1")
    expected_t1 = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": decision,
        "threshold": legacy.AI_SCORE_THRESHOLD,
        "threshold_operator": legacy.THRESHOLD_OPERATOR,
        "policy": "released_FSD_whole_image_score_sign_inverted",
    }
    if t1 != expected_t1:
        raise ValueError(f"{sample_id} t1 aliases changed")
    manual = _require_mapping(
        row.get("manual_replay"),
        f"{sample_id} manual_replay",
    )
    expected_manual = {
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "ai_score": ai_score,
        "released_is_fake": released_decision,
        "classification_decision": decision,
        "official_raw_exact_match": True,
        "official_z_exact_match": True,
        "compute_fsd_calls": 1,
    }
    if manual != expected_manual:
        raise ValueError(f"{sample_id} manual tail replay changed")


def _independent_visibility_diagnostic(
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Independently recompute FSD crop visibility from canonical GT pixels."""

    width = int(input_row["width"])
    height = int(input_row["height"])
    preprocess = legacy.compute_preprocess_geometry(width, height)
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
                "effective_native_crop_xyxy": list(
                    preprocess["effective_native_crop_xyxy"]
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
    if pixels.shape != (height, width) or not np.isin(
        pixels,
        (0, 255),
    ).all():
        raise ValueError(f"{sample_id} exact-diff GT pixels changed")
    positive_y, positive_x = np.nonzero(pixels == 255)
    total = int(positive_x.size)
    if total <= 0 or total != input_row.get("gt_positive_pixels"):
        raise ValueError(f"{sample_id} exact-diff positive count changed")
    resize = preprocess["resize"]
    crop = preprocess["center_crop"]
    residual_width, residual_height = resize["source_size"]
    resized_width, resized_height = resize["destination_size"]
    start_x, start_y = crop["start_xy"]
    crop_width, crop_height = crop["size"]
    destination_x = (
        (positive_x.astype(np.float64) - legacy.FRE_BORDER + 0.5)
        * resized_width
        / residual_width
        - 0.5
    )
    destination_y = (
        (positive_y.astype(np.float64) - legacy.FRE_BORDER + 0.5)
        * resized_height
        / residual_height
        - 0.5
    )
    visible = (
        (destination_x >= start_x)
        & (destination_x < start_x + crop_width)
        & (destination_y >= start_y)
        & (destination_y < start_y + crop_height)
    )
    visible_count = int(np.count_nonzero(visible))
    fraction = visible_count / total
    category = (
        "none"
        if visible_count == 0
        else "full"
        if visible_count == total
        else "partial"
    )
    gt_evidence = {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible_count,
        "forged_sample_id": sample_id,
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_after_FRE_trim_"
            "with_align_corners_false_formula_into_official_center_crop"
        ),
        "formula": (
            "d=(native_index-7+0.5)*resized_size/(native_size-14)-0.5; "
            "visible iff crop_start <= d < crop_start+1024"
        ),
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
    crop_box = list(preprocess["effective_native_crop_xyxy"])
    x0, y0 = max(box[0], crop_box[0]), max(box[1], crop_box[1])
    x1, y1 = min(box[2], crop_box[2]), min(box[3], crop_box[3])
    intersection = [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None
    edit_area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    box_fraction = min(1.0, max(0.0, visible_area / edit_area))
    box_category = (
        "none"
        if box_fraction == 0.0
        else "full"
        if math.isclose(box_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else "partial"
    )
    box_evidence = {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": crop_box,
        "intersection_xyxy": intersection,
        "edit_area": edit_area,
        "visible_area": visible_area,
        "visible_fraction": box_fraction,
        "category": box_category,
        "basis": (
            "continuous_edit_box_area_intersection_with_effective_native_crop"
        ),
    }
    return {
        "edit_visibility": category,
        "edit_visible_gt_fraction": fraction,
        "edit_visibility_evidence": {
            "gt": gt_evidence,
            "edit_box": box_evidence,
        },
    }


def _validate_physical_attempts(
    *,
    physical: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    runner = _runner()
    inputs = {str(row["sample_id"]): row for row in selected}
    for index, row in enumerate(physical):
        sample_id = _require_string(
            row.get("sample_id"),
            f"physical result {index} sample_id",
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
        expected_visibility = _independent_visibility_diagnostic(
            expected,
            repo_root=repo_root,
        )
        for key, expected_value in expected_visibility.items():
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


def _descriptor_mapping(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> dict[str, Any]:
    descriptor = _require_mapping(
        row.get("descriptor"),
        f"{sample_id} descriptor",
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
    }
    if set(descriptor) != required:
        raise ValueError(f"{sample_id} descriptor key set changed")
    return descriptor


def _validate_descriptor_aliases(
    row: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    sample_id: str,
    array_sha256: str,
) -> None:
    aliases = {
        "raw_descriptor_path": descriptor["relative_path"],
        "raw_descriptor_sha256": descriptor["sha256"],
        "raw_descriptor_array_sha256": array_sha256,
        "raw_descriptor_shape": [DESCRIPTOR_DIMENSION],
        "raw_descriptor_dtype": "float64",
        "raw_descriptor_nbytes": DESCRIPTOR_NBYTES,
        "raw_descriptor_semantics": (
            "official_compute_fsd_before_released_transforms"
        ),
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} descriptor alias {key} differs")
    if descriptor["array_sha256"] != array_sha256:
        raise ValueError(f"{sample_id} descriptor array SHA differs")
    artifact_paths = _require_mapping(
        row.get("artifact_paths"),
        f"{sample_id} artifact_paths",
    )
    if artifact_paths != {
        "raw_descriptor_npy": descriptor["relative_path"]
    }:
        raise ValueError(f"{sample_id} artifact path alias differs")


def _descriptor_artifact(
    *,
    row: Mapping[str, Any],
    sample_id: str,
    repo_root: Path,
    descriptor_dir: Path,
) -> DescriptorArtifact:
    descriptor = _descriptor_mapping(row, sample_id=sample_id)
    relative_path = _require_string(
        descriptor.get("relative_path"),
        f"{sample_id} descriptor path",
    )
    path = _safe_repo_path(
        relative_path,
        repo_root=repo_root,
        label=f"{sample_id} descriptor path",
    )
    expected_path = (descriptor_dir / f"{sample_id}.npy").resolve()
    if path != expected_path:
        raise ValueError(f"{sample_id} descriptor path is not canonical")
    file_sha = _require_sha256(
        descriptor.get("sha256"),
        f"{sample_id} descriptor SHA-256",
    )
    if sha256_file(path) != file_sha:
        raise ValueError(f"{sample_id} descriptor artifact hash mismatch")
    if (
        descriptor.get("file_bytes") != path.stat().st_size
        or path.stat().st_size != 7808
    ):
        raise ValueError(f"{sample_id} descriptor file byte-size changed")
    if (
        descriptor.get("dtype") != "float64"
        or descriptor.get("shape") != [DESCRIPTOR_DIMENSION]
        or descriptor.get("nbytes") != DESCRIPTOR_NBYTES
        or descriptor.get("finite") is not True
        or descriptor.get("semantics")
        != "official_compute_fsd_before_released_transforms"
    ):
        raise ValueError(f"{sample_id} descriptor metadata changed")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{sample_id} descriptor is not a safe NumPy array"
        ) from error
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.shape != (DESCRIPTOR_DIMENSION,)
        or loaded.dtype != DESCRIPTOR_DTYPE
        or not loaded.flags.c_contiguous
        or not np.isfinite(loaded).all()
        or loaded.nbytes != DESCRIPTOR_NBYTES
    ):
        raise ValueError(f"{sample_id} descriptor array is invalid")
    array = np.ascontiguousarray(loaded)
    if path.read_bytes() != _npy_bytes(array):
        raise ValueError(f"{sample_id} descriptor NPY bytes are non-canonical")
    array_sha = _array_sha256(array)
    _validate_descriptor_aliases(
        row,
        descriptor,
        sample_id=sample_id,
        array_sha256=array_sha,
    )
    return DescriptorArtifact(
        sample_id=sample_id,
        path=path,
        file_sha256=file_sha,
        file_bytes=path.stat().st_size,
        array_sha256=array_sha,
        array=array,
    )


def validate_descriptor_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    descriptor_dir: Path,
) -> dict[str, DescriptorArtifact]:
    descriptor_dir = _safe_absolute_dir(
        descriptor_dir,
        root=repo_root,
        label="descriptor directory",
    )
    ids = [
        _require_string(row.get("sample_id"), "latest result sample_id")
        for row in latest_results
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("latest results contain duplicate sample_id")
    entries = list(descriptor_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"descriptor inventory has non-regular entry {entry.name}"
            )
    expected_names = {f"{sample_id}.npy" for sample_id in ids}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            "descriptor inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    result: dict[str, DescriptorArtifact] = {}
    for row in latest_results:
        sample_id = str(row["sample_id"])
        _validate_score_payload(row, sample_id=sample_id)
        result[sample_id] = _descriptor_artifact(
            row=row,
            sample_id=sample_id,
            repo_root=repo_root,
            descriptor_dir=descriptor_dir,
        )
    return result


def _descriptor_inventory_sha256(
    descriptors: Mapping[str, DescriptorArtifact],
) -> str:
    rows = [
        {
            "sample_id": sample_id,
            "path": artifact.path.resolve().as_posix(),
            "file_sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "array_sha256": artifact.array_sha256,
        }
        for sample_id, artifact in sorted(descriptors.items())
    ]
    return hashlib.sha256(stable_json(rows).encode("utf-8")).hexdigest()


def _capture_evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    descriptors: Mapping[str, DescriptorArtifact],
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
    for artifact in descriptors.values():
        if (
            artifact.path.is_symlink()
            or not artifact.path.is_file()
            or artifact.path.stat().st_size != artifact.file_bytes
            or sha256_file(artifact.path) != artifact.file_sha256
        ):
            raise ValueError(
                "descriptor evidence changed while it was being validated"
            )
    return {
        **current,
        "descriptor_inventory_sha256": _descriptor_inventory_sha256(
            descriptors
        ),
    }


def _verify_bundle_unchanged(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> None:
    """Revalidate every persisted and canonical input after long analysis."""

    expected = dict(bundle.evidence_snapshot)
    if set(expected) != {
        "manifest_sha256",
        "results_sha256",
        "expected_inputs_sha256",
        "summary_sha256",
        "descriptor_inventory_sha256",
    }:
        raise ValueError("run evidence snapshot key set changed")
    for key, path in (
        ("manifest_sha256", bundle.manifest_path),
        ("results_sha256", bundle.results_path),
        ("expected_inputs_sha256", bundle.expected_path),
        ("summary_sha256", bundle.summary_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"run evidence changed after validation: {key}")
        if sha256_file(path) != expected[key]:
            raise ValueError(f"run evidence changed after validation: {key}")

    current_manifest = _load_json(
        bundle.manifest_path,
        f"{bundle.mode} run manifest recheck",
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
    descriptors = validate_descriptor_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        descriptor_dir=bundle.descriptor_dir,
    )
    if (
        _descriptor_inventory_sha256(descriptors)
        != expected["descriptor_inventory_sha256"]
    ):
        raise ValueError("descriptor evidence changed after validation")


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    bundle_mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    coverage: Mapping[str, Any],
) -> None:
    _assert_runner_contract_exports()
    required = {
        "schema_version": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_fsd_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": bundle_mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "score_spec": _score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
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
    runner = _runner()
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
        immutable.get("outputs"),
        "immutable outputs",
    )
    descriptor_dir = _safe_repo_path(
        immutable_outputs.get("descriptor_dir"),
        repo_root=root,
        label="immutable descriptor directory",
        require_file=False,
    )
    canonical_descriptor_dir = (
        root
        / "outputs"
        / "opensource"
        / "fsd"
        / run_id
        / "raw_descriptors"
    ).resolve()
    if descriptor_dir != canonical_descriptor_dir:
        raise ValueError("immutable descriptor directory is not canonical")
    expected_outputs = {
        "run_dir": repo_relative(run_dir, root),
        "results_path": repo_relative(results_path, root),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "summary_sha256": sha256_file(summary_path),
        "descriptor_dir": repo_relative(descriptor_dir, root),
        "descriptor_files": len(selected),
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
        coverage=coverage,
    )
    descriptors = validate_descriptor_inventory(
        latest_results=latest,
        repo_root=root,
        descriptor_dir=descriptor_dir,
    )
    evidence_snapshot = _capture_evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        descriptors=descriptors,
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
        descriptor_dir=descriptor_dir,
        descriptors=descriptors,
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


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "FSD Balanced250 metrics require iterations=1000 seed=20260726"
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
    descriptor = _descriptor_mapping(row, sample_id=sample_id)
    result["descriptor"] = {
        key: value
        for key, value in descriptor.items()
        if key not in _DESCRIPTOR_VOLATILE_FIELDS
    }
    for key in (
        "raw_descriptor_path",
        "raw_descriptor_sha256",
        "raw_descriptor_array_sha256",
    ):
        result.pop(key, None)
    if "artifact_paths" in result:
        result.pop("artifact_paths")
    return result


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_descriptors: Mapping[str, DescriptorArtifact],
    replay_descriptors: Mapping[str, DescriptorArtifact],
) -> dict[str, Any]:
    def unique(
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"),
                f"{label} row {index} sample_id",
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
    if set(reference_descriptors) != set(reference):
        raise ValueError("reference descriptor coverage differs")
    if set(replay_descriptors) != set(replay):
        raise ValueError("replay descriptor coverage differs")
    max_raw = 0.0
    max_z = 0.0
    max_ai = 0.0
    max_descriptor = 0.0
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
        for key, accumulator in (
            ("raw_likelihood", "raw"),
            ("released_z_score", "z"),
            ("ai_score", "ai"),
        ):
            difference = abs(float(left[key]) - float(right[key]))
            if accumulator == "raw":
                max_raw = max(max_raw, difference)
            elif accumulator == "z":
                max_z = max(max_z, difference)
            else:
                max_ai = max(max_ai, difference)
        left_artifact = reference_descriptors[sample_id]
        right_artifact = replay_descriptors[sample_id]
        if (
            left_artifact.path.read_bytes()
            != right_artifact.path.read_bytes()
            or left_artifact.file_sha256 != right_artifact.file_sha256
            or left_artifact.file_bytes != right_artifact.file_bytes
        ):
            raise ValueError(f"smoke descriptor {sample_id} bytes differ")
        if not np.array_equal(left_artifact.array, right_artifact.array):
            raise ValueError(f"smoke descriptor {sample_id} values differ")
        difference = float(
            np.max(np.abs(left_artifact.array - right_artifact.array))
        )
        max_descriptor = max(max_descriptor, difference)
    if any(value != 0.0 for value in (max_raw, max_z, max_ai, max_descriptor)):
        raise ValueError("smoke comparison is not bit-exact")
    return {
        "images_compared": len(reference),
        "ignored_row_fields": sorted(_SMOKE_ROW_IGNORED_FIELDS),
        "ignored_descriptor_metadata_fields": sorted(
            _DESCRIPTOR_VOLATILE_FIELDS
        ),
        "exact_computational_projection": True,
        "descriptor_file_bytes_exact": True,
        "descriptor_array_exact": True,
        "max_raw_likelihood_abs_difference": max_raw,
        "max_released_z_score_abs_difference": max_z,
        "max_ai_score_abs_difference": max_ai,
        "max_descriptor_abs_difference": max_descriptor,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    if set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS:
        raise ValueError("smoke immutable config key set changed")
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    output_path: Path | None,
) -> dict[str, Any]:
    runner = _runner()
    reference_run_id = runner._valid_run_id(reference_run_id)
    replay_run_id = runner._valid_run_id(replay_run_id)
    if reference_run_id == replay_run_id:
        raise ValueError("smoke comparison requires two distinct run IDs")
    analysis_runtime = _actual_runtime_contract("cpu")
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
            reference.manifest_path,
            reference.results_path,
            reference.expected_path,
            reference.summary_path,
            replay.manifest_path,
            replay.results_path,
            replay.expected_path,
            replay.summary_path,
        ),
        protected_dirs=(
            reference.descriptor_dir,
            replay.descriptor_dir,
        ),
    )
    if _smoke_immutable_projection(
        reference.immutable
    ) != _smoke_immutable_projection(replay.immutable):
        raise ValueError(
            "smoke immutable computational/runtime configurations differ"
        )
    if reference.selected != replay.selected:
        raise ValueError("smoke runs do not use the same exact selection")
    if len(reference.selected) != SMOKE_IMAGES:
        raise ValueError("smoke selection is not 35 images")
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_descriptors=reference.descriptors,
        replay_descriptors=replay.descriptors,
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
            "manifest_sha256": reference.evidence_snapshot[
                "manifest_sha256"
            ],
            "results_sha256": reference.evidence_snapshot[
                "results_sha256"
            ],
            "expected_inputs_sha256": reference.evidence_snapshot[
                "expected_inputs_sha256"
            ],
            "summary_sha256": reference.evidence_snapshot["summary_sha256"],
            "descriptor_inventory_sha256": reference.evidence_snapshot[
                "descriptor_inventory_sha256"
            ],
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            "manifest_sha256": replay.evidence_snapshot["manifest_sha256"],
            "results_sha256": replay.evidence_snapshot["results_sha256"],
            "expected_inputs_sha256": replay.evidence_snapshot[
                "expected_inputs_sha256"
            ],
            "summary_sha256": replay.evidence_snapshot["summary_sha256"],
            "descriptor_inventory_sha256": replay.evidence_snapshot[
                "descriptor_inventory_sha256"
            ],
        },
        "selection": reference.contract.selection.as_dict(),
        "comparison": comparison,
        "immutable_computational_runtime_config_exact": True,
        "evidence_reverified_after_comparison": True,
    }
    if output_path is not None:
        _write_json_verified(output_path, report, label="comparison output")
    return report


def _actual_runtime_contract(device_text: str) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    configured = runner.configure_runtime(
        device_text,
        seed=EXPECTED_RUNTIME_SEED,
    )
    if (
        not isinstance(configured, tuple)
        or len(configured) != 2
        or not isinstance(configured[1], Mapping)
    ):
        raise ValueError("runner configure_runtime return contract changed")
    runtime = dict(configured[1])
    _validate_runtime_contract(runtime, label="current analysis runtime")
    return runtime


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    weights_dir: Path,
    device_text: str,
) -> dict[str, Any]:
    """Freshly replay every canonical JPEG through the complete FSD model."""

    if len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh replay requires the full 1,775-image selection")
    runtime = _actual_runtime_contract(device_text)
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"),
        "immutable.runtime",
    )
    if runtime != recorded_runtime:
        raise ValueError("fresh replay runtime differs from manifest")
    detector = None
    device = None
    replayed = 0
    max_raw = 0.0
    max_z = 0.0
    max_ai = 0.0
    max_descriptor = 0.0
    try:
        detector, device, load_audit = legacy.load_detector(
            source_root=source_root,
            weights_dir=weights_dir,
            device_name=device_text,
        )
        if load_audit.get("source") != bundle.immutable.get("source"):
            raise ValueError("fresh replay source differs from manifest")
        if load_audit.get("weights") != bundle.immutable.get("weights"):
            raise ValueError("fresh replay weights differ from manifest")
        for expected, row in zip(
            bundle.selected,
            bundle.latest_results,
            strict=True,
        ):
            sample_id = str(expected["sample_id"])
            input_path = _safe_repo_path(
                expected.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                label=f"{sample_id} canonical input",
            )
            if sha256_file(input_path) != expected.get("canonical_sha256"):
                raise ValueError(f"{sample_id} canonical input hash changed")
            preprocess = legacy.compute_preprocess_geometry(
                int(expected["width"]),
                int(expected["height"]),
            )
            if row.get("preprocess") != preprocess:
                raise ValueError(f"{sample_id} preprocessing record changed")
            processed, descriptor, _peak, _latency = legacy.infer_one(
                detector,
                device,
                input_path,
            )
            if set(processed) != {
                "raw_likelihood",
                "released_z_score",
                "ai_score",
                "released_is_fake",
                "classification_decision",
                "manual_replay",
            }:
                raise ValueError(f"{sample_id} fresh score key set changed")
            fresh_raw = _require_finite(
                processed.get("raw_likelihood"),
                f"{sample_id} fresh raw likelihood",
            )
            fresh_z = _require_finite(
                processed.get("released_z_score"),
                f"{sample_id} fresh z-score",
            )
            fresh_ai = _require_finite(
                processed.get("ai_score"),
                f"{sample_id} fresh AI score",
            )
            stored_descriptor = bundle.descriptors[sample_id].array
            if not np.array_equal(descriptor, stored_descriptor):
                difference = float(
                    np.max(np.abs(descriptor - stored_descriptor))
                )
                raise ValueError(
                    f"{sample_id} fresh descriptor mismatch "
                    f"(max abs {difference})"
                )
            descriptor_difference = float(
                np.max(np.abs(descriptor - stored_descriptor))
            )
            raw_difference = abs(
                fresh_raw
                - _require_finite(
                    row.get("raw_likelihood"),
                    f"{sample_id} stored raw likelihood",
                )
            )
            z_difference = abs(
                fresh_z
                - _require_finite(
                    row.get("released_z_score"),
                    f"{sample_id} stored z-score",
                )
            )
            ai_difference = abs(
                fresh_ai
                - _require_finite(
                    row.get("ai_score"),
                    f"{sample_id} stored AI score",
                )
            )
            max_descriptor = max(max_descriptor, descriptor_difference)
            max_raw = max(max_raw, raw_difference)
            max_z = max(max_z, z_difference)
            max_ai = max(max_ai, ai_difference)
            if raw_difference > RAW_LIKELIHOOD_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} raw likelihood replay mismatch")
            if z_difference > ZSCORE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} z-score replay mismatch")
            if ai_difference > AI_SCORE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} AI score replay mismatch")
            if (
                processed["classification_decision"]
                is not row.get("classification_decision")
                or processed["released_is_fake"]
                is not row.get("released_is_fake")
            ):
                raise ValueError(f"{sample_id} decision replay mismatch")
            replayed += 1
    finally:
        if detector is not None:
            del detector
        gc.collect()
        if device is not None and getattr(device, "type", None) == "cuda":
            __import__("torch").cuda.empty_cache()
    if replayed != FORMAL_IMAGES:
        raise ValueError("fresh full-image replay coverage is incomplete")
    return {
        "status": "fresh_full_image_replay_passed",
        "images_replayed": replayed,
        "full_image_forward_per_input": True,
        "descriptor_tail_only_replay": False,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "weights_bundle_sha256": bundle.immutable["weights"]["bundle_sha256"],
        "runtime": runtime,
        "raw_likelihood_abs_tolerance": RAW_LIKELIHOOD_ABS_TOLERANCE,
        "released_z_score_abs_tolerance": ZSCORE_ABS_TOLERANCE,
        "ai_score_abs_tolerance": AI_SCORE_ABS_TOLERANCE,
        "descriptor_comparison": "numpy.array_equal",
        "max_raw_likelihood_abs_difference": max_raw,
        "max_released_z_score_abs_difference": max_z,
        "max_ai_score_abs_difference": max_ai,
        "max_descriptor_abs_difference": max_descriptor,
    }


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    weights_dir: Path,
    device_text: str,
    metrics_output_path: Path | None,
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
            "metrics": metrics_output_path,
            "audit": audit_output_path,
        },
        protected_files=(
            bundle.manifest_path,
            bundle.results_path,
            bundle.expected_path,
            bundle.summary_path,
        ),
        protected_dirs=(bundle.descriptor_dir,),
    )
    analysis_runtime = _actual_runtime_contract("cpu")
    metrics = recompute_metrics(bundle, iterations=iterations, seed=seed)
    metrics_sha256 = _json_artifact_sha256(metrics)
    if metrics_output_path is not None:
        _write_json_verified(
            metrics_output_path,
            metrics,
            label="metrics output",
        )
    replay_report = (
        replay_model(
            bundle,
            source_root=source_root,
            weights_dir=weights_dir,
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
        "descriptor_files": len(bundle.descriptors),
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "analysis_runtime": analysis_runtime,
        "fresh_model_replay": replay_report,
        "method_boundary": {
            "method": "FSD official v1.2 inference release",
            "paper_protocol_parity_claimed": False,
            "valid_for_t1": True,
            "valid_for_t2": False,
            "fullframe_t2": "not_applicable",
            "license": {
                "spdx": legacy.LICENSE_SPDX,
                "commercial_use": False,
                "share_alike": True,
            },
        },
        "contract_checks": {
            "exact_formal_whole_image_t1_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "pair_rank_rejected": True,
            "t2_joint_dense_claims_rejected": True,
            "source_weights_runtime_adapter_hashes_validated": True,
            "current_metrics_runtime_validated_before_recomputation": True,
            "run_and_canonical_evidence_reverified_after_replay": True,
            "descriptor_inventory_bytes_sha_array_shape_dtype_finite_validated": True,
            "score_calibration_direction_threshold_and_tail_replay_validated": True,
            "shared_balanced250_metrics_only": True,
        },
        "artifacts": {
            "manifest_sha256": bundle.evidence_snapshot["manifest_sha256"],
            "results_sha256": bundle.evidence_snapshot["results_sha256"],
            "expected_inputs_sha256": bundle.evidence_snapshot[
                "expected_inputs_sha256"
            ],
            "summary_sha256": bundle.evidence_snapshot["summary_sha256"],
            "descriptor_inventory_sha256": bundle.evidence_snapshot[
                "descriptor_inventory_sha256"
            ],
            "metrics_sha256": metrics_sha256,
        },
    }
    if audit_output_path is not None:
        _write_json_verified(audit_output_path, audit, label="audit output")
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _json_artifact_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        path,
        expected_sha256=expected_sha256,
        label=label,
    )


def _validate_output_targets(
    outputs: Mapping[str, Path | None],
    *,
    protected_files: Sequence[Path],
    protected_dirs: Sequence[Path],
) -> None:
    """Reject aliases between reports and immutable run evidence."""

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
        "--weights-dir",
        type=Path,
        default=getattr(runner, "DEFAULT_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--compare-smoke-run-id")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runner = _runner()
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
        output = (
            _anchored(args.comparison_output, repo_root)
            if args.comparison_output is not None
            else results_dir
            / f"{run_id}__vs__{compare_run_id}_comparison.json"
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
        raise ValueError(
            "--comparison-output requires --compare-smoke-run-id"
        )
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
        weights_dir=_anchored(args.weights_dir, repo_root),
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
