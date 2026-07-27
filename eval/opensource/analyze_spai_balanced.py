#!/usr/bin/env python3
"""Fail-closed audit and replay for SPAI on Balanced250.

The analyzer treats the runner manifest, append-only JSONL, and three NumPy
arrays per successful image as untrusted evidence.  It rebuilds the exact
formal or smoke selection, validates provenance and runtime pins, replays
persisted patch features through SCA/LayerNorm/the complete MLP, recomputes
the shared Balanced250 T1 metrics, and can freshly replay all 1,775 canonical
JPEGs through the complete FFT -> ViT -> SRS -> SCA -> MLP executable.

SPAI is a whole-image detector.  Its attention weights and exact-difference
patch-visibility census are classifier/input diagnostics only; they are not
T2 localization, predicted masks, or pixel probabilities.
"""

from __future__ import annotations

import argparse
import copy
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

from eval.opensource import analyze_spai_run as legacy_audit
from eval.opensource import run_spai as legacy
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


AUDIT_SCHEMA_VERSION = "spai_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "spai_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
EXPECTED_RUN_MANIFEST_SCHEMA = "spai_balanced_run_manifest_v2"
EXPECTED_RUN_CONFIG_SCHEMA = "spai_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = "spai_balanced_runtime_summary_v2"
EXPECTED_CPU_PREFLIGHT_SCHEMA = "spai_balanced_cpu_preflight_v1"

DEFAULT_RESULTS_DIR = Path("results/opensource/spai")
DEFAULT_RUN_ID = "spai_any_resolution_spectral_balanced250_v1_full1775_20260726"
DEFAULT_SOURCE_ROOT = Path("/root/.cache/claimforge/third_party/spai-8ff7b3b6")
DEFAULT_CHECKPOINT = Path("/root/.cache/claimforge/third_party/spai.pth")
DEFAULT_GOLDEN_ROOT = Path(
    "/root/.cache/claimforge/third_party/spai-official-originals"
)

FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
EXPECTED_RUNTIME_SEED = 0
PATCH_SIZE = 224
PATCH_STRIDE = 224
MINIMUM_PATCHES = 4
FEATURE_EXTRACTION_BATCH = 400
FEATURE_DIMENSION = 1096
ATTENTION_HEADS = 12
FEATURE_DTYPE = np.dtype(np.float32)
PREPROCESS_PROFILE = "official_pillow_rgb_native_float32_0_1"
PATCH_FEATURE_SEMANTICS = (
    "per_patch_frequency_restoration_features_before_spectral_context_attention"
)
FEATURE_SEMANTICS = (
    "spectral_context_attention_layernorm_output_before_complete_mlp_head"
)
ATTENTION_SEMANTICS = (
    "spectral_context_attention_softmax_weights_classifier_diagnostic_"
    "not_localization"
)
PATCH_FEATURE_ABS_TOLERANCE = 1e-5
FEATURE_ABS_TOLERANCE = 1e-5
ATTENTION_ABS_TOLERANCE = 1e-6
RAW_LOGIT_ABS_TOLERANCE = 1e-5
PROBABILITY_ABS_TOLERANCE = 1e-7

EXPECTED_FROZEN_SOURCE_COMMIT = "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
EXPECTED_CHECKPOINT_SHA256 = (
    "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55"
)
EXPECTED_CHECKPOINT_BYTES = 934_865_338
EXPECTED_CHECKPOINT_SCHEMA_SHA256 = (
    "ffe751246ec65936d5583a1db62bf617697484e6185f1bfad7c678f1dad36ef8"
)
EXPECTED_FORMAL_SELECTED_ROWS_SHA256 = (
    "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
)
EXPECTED_FORMAL_SELECTED_IDS_SHA256 = (
    "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
)
EXPECTED_SMOKE5X7_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)
EXPECTED_ADAPTER_SOURCE_PATHS = (
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
        "execution_device_golden",
        "artifact_contract",
        "local_artifact_policy",
        "outputs",
    }
)
EXPECTED_ANALYSIS_PACKAGE_VERSIONS = {
    "scipy": "1.16.0",
    "scikit-learn": "1.5.2",
}
EXPECTED_FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/spai/bin/python"
)
EXPECTED_FROZEN_VENV_PREFIX = Path("/root/.cache/claimforge/venvs/spai")
EXPECTED_FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/spai-balanced-v2-empty"
)
EXPECTED_FROZEN_PYVENV_CONFIG_SHA256 = (
    "506c01d6bc866a7500bde63a24b3f0c1fb3013df41051ad9a8bf7c42c85eb091"
)
EXPECTED_FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "timm": "0.4.12",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "yacs": "0.1.8",
    "einops": "0.8.1",
    "opencv-python-headless": "4.10.0.84",
    "albumentations": "1.4.14",
    "setuptools": "79.0.1",
}
EXPECTED_FROZEN_RUNTIME_MODULE_FILES = {
    "torch": "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d",
    "torchvision": "ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73",
    "timm": "f664ef352d89e92a0c681ad812fe9772673d106332b6e1709098146025d202b8",
    "numpy": "22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e",
    "PIL": "7c95303c6848f3f99c07c8cd583fa1530ecc88c2725a0a955ff9c5b73223d59b",
    "yacs": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "einops": "acacaf13ae1b60c38c5c01b811f30c4951b1805674c79d5db7cda946cc389471",
    "cv2": "936bd94c5a5debf0212fc751af79d3a163652f3e850259df2159db6aa3ed8ad8",
    "albumentations": "881857517838ba0e9a8fc90f4ccf224863c5e019685bfac8914fa6608df9a8cf",
}
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_MINIMUM_CUDA_FREE_BYTES = 12 * 1024**3
EXPECTED_CPU_OFFICIAL_GOLDEN_CASES = (
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
        "relative_path": ("stable-diffusion-3/cfg_60/euler/steps_28/000001046_4.webp"),
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

_FALSE_SCOPE_KEYS = frozenset({"valid_for_t2", "native_dense_output", "t2_applicable"})
_NULL_SCOPE_KEYS = frozenset(
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
_FORBIDDEN_KEYS = frozenset(
    {
        "pair_rank",
        "localization",
        "localisation",
        "heatmap",
        "mask",
        "attention_map",
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
    "attention_map",
    "attention_mask",
)
_ALLOWED_DIAGNOSTIC_KEYS = frozenset(
    {
        "attention_is_diagnostic_not_t2",
        "patch_visibility",
        "edit_visibility_evidence",
        "gt_mask_kind",
        "mask_positive_pixels",
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
_ARTIFACT_VOLATILE_FIELDS = frozenset({"relative_path", "sha256", "array_sha256"})


@dataclass(frozen=True)
class ArrayArtifact:
    """One validated canonical local NumPy artifact."""

    kind: str
    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: str
    array: np.ndarray


@dataclass(frozen=True)
class SampleArtifacts:
    """All three arrays required for one SPAI score."""

    patch_features: ArrayArtifact
    feature: ArrayArtifact
    attention: ArrayArtifact


@dataclass(frozen=True)
class RunBundle:
    """A fully validated formal or smoke run."""

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
    artifact_root: Path
    artifacts: Mapping[str, SampleArtifacts]
    evidence_snapshot: Mapping[str, str]


def _runner() -> Any:
    return importlib.import_module("eval.opensource.run_spai_balanced")


def _score_spec() -> ScoreSpec:
    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
        threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
    )


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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


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


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _require_exact_json(value: Any, expected: Any, label: str) -> None:
    """Recursively compare JSON without Python's bool/int aliasing."""

    if isinstance(expected, Mapping):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{label} object key set changed")
        for key, expected_child in expected.items():
            _require_exact_json(
                value[key],
                expected_child,
                f"{label}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{label} array changed")
        for index, (child, expected_child) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_json(
                child,
                expected_child,
                f"{label}[{index}]",
            )
        return
    if type(value) is not type(expected):
        raise ValueError(f"{label} JSON scalar type changed")
    if isinstance(expected, float) and (
        not math.isfinite(value) or not math.isfinite(expected)
    ):
        raise ValueError(f"{label} is non-finite")
    if (
        isinstance(expected, float)
        and stable_json(value) != stable_json(expected)
    ):
        raise ValueError(f"{label} changed")
    if not isinstance(expected, float) and value != expected:
        raise ValueError(f"{label} changed")


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"missing regular {label}: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        return _require_mapping(
            _json_loads(path.read_text(encoding="utf-8"), label),
            label,
        )
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error


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
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _expected_npy_file_bytes(shape: tuple[int, ...]) -> int:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(FEATURE_DTYPE),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return len(header.getvalue()) + math.prod(shape) * FEATURE_DTYPE.itemsize


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
    raw_candidate = root / run_id
    if raw_candidate.is_symlink():
        raise ValueError("SPAI run directory is a symlink")
    candidate = raw_candidate.resolve()
    if candidate.parent != root:
        raise ValueError("resolved run directory escapes results root")
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
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite")


def _reject_unsupported_claims(value: Any, label: str) -> None:
    """Reject T2/localization while permitting diagnostic attention arrays."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            child = f"{label}.{raw_key}"
            if key in _ALLOWED_DIAGNOSTIC_KEYS:
                if key == "attention_is_diagnostic_not_t2" and nested is not True:
                    raise ValueError(f"{child} misrepresents attention")
                _reject_unsupported_claims(nested, child)
            elif key in _FALSE_SCOPE_KEYS:
                if nested is not False:
                    raise ValueError(f"{child} is an unsupported SPAI claim")
            elif key in _NULL_SCOPE_KEYS:
                if nested is not None:
                    raise ValueError(f"{child} is an unsupported SPAI claim")
            elif key in _FORBIDDEN_KEYS or key.startswith(_FORBIDDEN_PREFIXES):
                raise ValueError(f"{child} is an unsupported SPAI claim")
            else:
                _reject_unsupported_claims(nested, child)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_unsupported_claims(nested, f"{label}[{index}]")


def _independent_visibility_diagnostic(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Rebuild patch coverage directly from exact GT and frozen geometry."""

    width, height = int(row["width"]), int(row["height"])
    geometry = legacy_audit.compute_patch_geometry(width, height)
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        evidence = legacy_audit._visibility_from_exact_gt(
            row,
            repo_root=repo_root,
        )
        return {
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "edit_visibility_evidence": evidence,
        }
    expected = "all_zero" if row.get("condition") == "real" else "not_applicable"
    if gt_kind != expected:
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


def _independent_visibility_census(
    selected: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    all_counts: Counter[str] = Counter()
    all_modes: Counter[str] = Counter()
    all_fractions: list[float] = []
    for condition in ("local_mouse", "local_cat", "local_trash_can"):
        counts: Counter[str] = Counter()
        modes: Counter[str] = Counter()
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
            mode = str(diagnostic["edit_visibility_evidence"]["geometry"]["patch_mode"])
            if (
                category not in ("full", "partial", "none")
                or mode not in ("grid", "five_crop")
                or not math.isfinite(fraction)
                or not 0.0 <= fraction <= 1.0
            ):
                raise ValueError("SPAI visibility diagnostic is invalid")
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
            "total": len(fractions),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
            "patch_modes": {
                "grid": modes["grid"],
                "five_crop": modes["five_crop"],
            },
        }
    return {
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
        "not_applicable_images": sum(
            row.get("gt_mask_kind") != "exact_diff" for row in selected
        ),
        "basis": (
            "exact_diff_positive_pixels_in_union_of_official_native_"
            "resolution_patch_receptive_fields"
        ),
        "role": "input_condition_stratum_not_model_localization",
    }


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    for key in (
        "raw_logit",
        "probability",
        "ai_score",
        "score",
        "classification_threshold",
    ):
        if type(row.get(key)) is not float:
            raise ValueError(f"{sample_id} {key} JSON type changed")
    raw = _require_finite(row.get("raw_logit"), f"{sample_id} raw_logit")
    probability = _require_finite(
        row.get("probability"),
        f"{sample_id} probability",
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{sample_id} probability is outside [0,1]")
    ai_score = _require_finite(
        row.get("ai_score"),
        f"{sample_id} ai_score",
    )
    score = _require_finite(row.get("score"), f"{sample_id} score")
    if ai_score != probability or score != probability:
        raise ValueError(f"{sample_id} score aliases differ")
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
    manual = _require_mapping(
        row.get("manual_replay"),
        f"{sample_id} manual_replay",
    )
    stored_classification = _require_mapping(
        row.get("classification"),
        f"{sample_id} classification",
    )
    stored_t1 = _require_mapping(row.get("t1"), f"{sample_id} t1")
    if set(stored_classification) != set(classification) or set(stored_t1) != set(t1):
        raise ValueError(f"{sample_id} SPAI score/manual replay changed")
    for key in ("raw_logit", "probability", "ai_score", "score", "threshold"):
        _require_finite(
            stored_classification.get(key),
            f"{sample_id} classification.{key}",
        )
        _require_finite(
            stored_t1.get(key),
            f"{sample_id} t1.{key}",
        )
    if (
        stored_classification.get("decision") is not decision
        or stored_t1.get("decision") is not decision
    ):
        raise ValueError(f"{sample_id} SPAI score/manual replay changed")
    scalar_manual = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "model_forward_calls": 1,
        "to_kv_hook_calls": 1,
        "attention_hook_calls": 1,
        "norm_hook_calls": 1,
    }
    exact_flags = (
        "official_attention_exact_match",
        "official_aggregated_exact_match",
        "official_feature_exact_match",
        "official_logit_exact_match",
        "official_probability_exact_match",
        "sca_replay",
        "norm_replay",
        "complete_mlp_replay",
    )
    for key in ("raw_logit", "probability", "ai_score"):
        _require_finite(
            manual.get(key),
            f"{sample_id} manual_replay.{key}",
        )
    if manual.get("classification_decision") is not decision:
        raise ValueError(f"{sample_id} SPAI score/manual replay changed")
    for key in (
        "model_forward_calls",
        "to_kv_hook_calls",
        "attention_hook_calls",
        "norm_hook_calls",
    ):
        if type(manual.get(key)) is not int or manual[key] != 1:
            raise ValueError(f"{sample_id} SPAI score/manual replay changed")
    _require_exact_json(
        stored_classification,
        classification,
        f"{sample_id} classification",
    )
    _require_exact_json(stored_t1, t1, f"{sample_id} t1")
    _require_exact_json(
        manual,
        {**scalar_manual, **{key: True for key in exact_flags}},
        f"{sample_id} manual replay",
    )
    if (
        set(manual) != {*scalar_manual, *exact_flags}
        or row.get("score_semantics") != legacy.SCORE_SEMANTICS
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold") != legacy.CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or stored_classification != classification
        or stored_t1 != t1
        or any(manual.get(key) != value for key, value in scalar_manual.items())
        or any(manual.get(key) is not True for key in exact_flags)
    ):
        raise ValueError(f"{sample_id} SPAI score/manual replay changed")


def _artifact_specifications(
    patch_count: int,
) -> tuple[tuple[str, str, tuple[int, ...], str, str], ...]:
    if isinstance(patch_count, bool) or patch_count <= 0:
        raise ValueError("SPAI effective patch count is invalid")
    return (
        (
            "spai_patch_features",
            "patch_features",
            (patch_count, FEATURE_DIMENSION),
            PATCH_FEATURE_SEMANTICS,
            "spai_patch_features_npy",
        ),
        (
            "spai_feature",
            "feature",
            (FEATURE_DIMENSION,),
            FEATURE_SEMANTICS,
            "spai_feature_npy",
        ),
        (
            "spai_attention",
            "attention",
            (ATTENTION_HEADS, patch_count),
            ATTENTION_SEMANTICS,
            "spai_attention_npy",
        ),
    )


def _artifact_record(
    row: Mapping[str, Any],
    *,
    prefix: str,
    directory_name: str,
    expected_shape: tuple[int, ...],
    semantics: str,
    sample_id: str,
    run_id: str,
    repo_root: Path,
    artifact_root: Path,
) -> ArrayArtifact:
    record = _require_mapping(row.get(prefix), f"{sample_id} {prefix}")
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
    if set(record) != expected_keys:
        raise ValueError(f"{sample_id} {prefix} record key set changed")
    relative = _require_string(
        record.get("relative_path"),
        f"{sample_id} {prefix} relative path",
    )
    path = _safe_repo_path(
        relative,
        repo_root=repo_root,
        label=f"{sample_id} {prefix}",
    )
    expected_path = (artifact_root / directory_name / f"{sample_id}.npy").resolve()
    canonical_root = (repo_root / "outputs" / "opensource" / "spai" / run_id).resolve()
    if artifact_root.resolve() != canonical_root or path != expected_path:
        raise ValueError(f"{sample_id} {prefix} path is not canonical")
    file_sha = _require_sha256(
        record.get("sha256"),
        f"{sample_id} {prefix} file SHA-256",
    )
    expected_nbytes = math.prod(expected_shape) * FEATURE_DTYPE.itemsize
    expected_file_bytes = _expected_npy_file_bytes(expected_shape)
    if (
        type(record.get("file_bytes")) is not int
        or record.get("file_bytes") != expected_file_bytes
        or type(record.get("nbytes")) is not int
        or record.get("nbytes") != expected_nbytes
        or path.stat().st_size != expected_file_bytes
    ):
        raise ValueError(f"{sample_id} {prefix} file size changed")
    if sha256_file(path) != file_sha:
        raise ValueError(f"{sample_id} {prefix} file hash changed")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} {prefix} is not a safe NPY") from error
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.shape != expected_shape
        or loaded.dtype != FEATURE_DTYPE
        or not loaded.flags.c_contiguous
        or not np.isfinite(loaded).all()
        or loaded.nbytes != expected_nbytes
    ):
        raise ValueError(f"{sample_id} {prefix} array contract changed")
    array = np.ascontiguousarray(loaded)
    canonical_payload = _npy_bytes(array)
    if path.read_bytes() != canonical_payload:
        raise ValueError(f"{sample_id} {prefix} NPY bytes are non-canonical")
    array_sha = _array_sha256(array)
    if (
        record.get("file_bytes") != len(canonical_payload)
        or record.get("array_sha256") != array_sha
        or record.get("dtype") != "float32"
        or record.get("shape") != list(expected_shape)
        or record.get("nbytes") != expected_nbytes
        or record.get("finite") is not True
        or record.get("semantics") != semantics
        or record.get("allow_pickle") is not False
    ):
        raise ValueError(f"{sample_id} {prefix} metadata changed")
    aliases = {
        f"{prefix}_path": relative,
        f"{prefix}_sha256": file_sha,
        f"{prefix}_array_sha256": array_sha,
        f"{prefix}_shape": list(expected_shape),
        f"{prefix}_dtype": "float32",
        f"{prefix}_nbytes": expected_nbytes,
        f"{prefix}_semantics": semantics,
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} artifact alias {key} changed")
    return ArrayArtifact(
        kind=prefix,
        sample_id=sample_id,
        path=path,
        file_sha256=file_sha,
        file_bytes=len(canonical_payload),
        array_sha256=array_sha,
        array=array,
    )


def _sample_artifacts(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    run_id: str,
    repo_root: Path,
    artifact_root: Path,
) -> SampleArtifacts:
    patch_count = row.get("spai_effective_patch_count")
    if (
        isinstance(patch_count, bool)
        or not isinstance(patch_count, int)
        or patch_count <= 0
    ):
        preprocess = _require_mapping(
            row.get("preprocess"),
            f"{sample_id} preprocess",
        )
        geometry = _require_mapping(
            preprocess.get("geometry"),
            f"{sample_id} preprocess geometry",
        )
        patch_count = _require_nonnegative_int(
            geometry.get("effective_patch_count"),
            f"{sample_id} effective patch count",
        )
    values: dict[str, ArrayArtifact] = {}
    artifact_paths: dict[str, str] = {}
    for prefix, directory, shape, semantics, path_alias in _artifact_specifications(
        patch_count
    ):
        artifact = _artifact_record(
            row,
            prefix=prefix,
            directory_name=directory,
            expected_shape=shape,
            semantics=semantics,
            sample_id=sample_id,
            run_id=run_id,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
        values[prefix] = artifact
        artifact_paths[path_alias] = repo_relative(artifact.path, repo_root)
    if row.get("artifact_paths") != artifact_paths:
        raise ValueError(f"{sample_id} artifact_paths aliases changed")
    if row.get("feature_array_sha256") != values["spai_feature"].array_sha256:
        raise ValueError(f"{sample_id} feature array digest alias changed")
    return SampleArtifacts(
        patch_features=values["spai_patch_features"],
        feature=values["spai_feature"],
        attention=values["spai_attention"],
    )


def validate_artifact_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
) -> dict[str, SampleArtifacts]:
    root = _safe_absolute_dir(
        artifact_root,
        root=repo_root,
        label="SPAI artifact root",
    )
    directories = {
        "patch_features",
        "feature",
        "attention",
    }
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != directories or any(
        entry.is_symlink() or not entry.is_dir() for entry in entries
    ):
        raise ValueError("SPAI artifact-root directory inventory changed")
    ids = [
        _require_string(row.get("sample_id"), "latest result sample_id")
        for row in latest_results
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("latest results contain duplicate sample_id")
    expected_files = {f"{sample_id}.npy" for sample_id in ids}
    for directory in directories:
        children = list((root / directory).iterdir())
        if (
            any(child.is_symlink() or not child.is_file() for child in children)
            or {child.name for child in children} != expected_files
        ):
            raise ValueError(f"SPAI {directory} inventory changed")
    result: dict[str, SampleArtifacts] = {}
    for row in latest_results:
        sample_id = str(row["sample_id"])
        _validate_score_payload(row, sample_id=sample_id)
        result[sample_id] = _sample_artifacts(
            row,
            sample_id=sample_id,
            run_id=run_id,
            repo_root=repo_root,
            artifact_root=root,
        )
    return result


def _maximum_absolute_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.shape != second.shape:
        raise ValueError("array shapes differ")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("array comparison contains a non-finite value")
    if first.size == 0:
        return 0.0
    difference = float(
        np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))
    )
    if not math.isfinite(difference):
        raise ValueError("array difference is not finite")
    return difference


def _require_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    tolerance: float,
) -> float:
    if (
        not isinstance(actual, np.ndarray)
        or not isinstance(expected, np.ndarray)
        or actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError(f"{label} shape/dtype changed")
    difference = _maximum_absolute_difference(actual, expected)
    if difference > tolerance:
        raise ValueError(f"{label} differs by {difference}, tolerance {tolerance}")
    return difference


def _require_float32_tensor(
    value: Any,
    *,
    torch_module: Any,
    device: Any,
    label: str,
) -> Any:
    if (
        not isinstance(value, torch_module.Tensor)
        or value.dtype != torch_module.float32
        or value.device != device
        or not bool(torch_module.isfinite(value).all())
    ):
        raise ValueError(f"{label} is not finite float32 on the recorded device")
    return value


def _tensor_float32_numpy(
    value: Any,
    *,
    torch_module: Any,
    device: Any,
    label: str,
) -> np.ndarray:
    tensor = _require_float32_tensor(
        value,
        torch_module=torch_module,
        device=device,
        label=label,
    )
    array = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} became non-finite after transfer")
    return array


def _tensor_float32_scalar(
    value: Any,
    *,
    torch_module: Any,
    device: Any,
    label: str,
) -> float:
    tensor = _require_float32_tensor(
        value,
        torch_module=torch_module,
        device=device,
        label=label,
    )
    if tensor.numel() != 1:
        raise ValueError(f"{label} is not scalar")
    return _require_finite(
        tensor.reshape(()).detach().cpu().item(),
        label,
    )


def _validate_runtime_contract(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    _reject_nonfinite_numbers(runtime, label)
    runner = _runner()
    validated = runner.validate_runtime_contract(runtime, label=label)
    if validated is not None and dict(validated) != runtime:
        raise ValueError(f"{label} runner validator changed evidence")
    device = runtime.get("device")
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
    if isinstance(device, str) and device.startswith("cuda:"):
        expected_keys.add("cuda")
    expected_prefix = Path(os.path.abspath(EXPECTED_FROZEN_VENV_PREFIX))
    expected_pycache = Path(os.path.abspath(EXPECTED_FROZEN_PYTHONPYCACHEPREFIX))
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} independent key set changed")
    if runtime.get("python") != {
        "implementation": "CPython",
        "version": EXPECTED_FROZEN_RUNTIME_VERSIONS["python"],
        "executable": str(Path(os.path.abspath(EXPECTED_FROZEN_PYTHON_EXECUTABLE))),
    }:
        raise ValueError(f"{label} independent Python pin changed")
    if runtime.get("venv") != {
        "prefix": str(expected_prefix),
        "base_prefix": "/usr",
        "pyvenv_cfg_path": str(expected_prefix / "pyvenv.cfg"),
        "pyvenv_cfg_sha256": EXPECTED_FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }:
        raise ValueError(f"{label} independent venv pin changed")
    if runtime.get("versions") != EXPECTED_FROZEN_RUNTIME_VERSIONS:
        raise ValueError(f"{label} independent package pins changed")
    modules = runtime.get("module_files")
    if not isinstance(modules, Mapping) or set(modules) != set(
        EXPECTED_FROZEN_RUNTIME_MODULE_FILES
    ):
        raise ValueError(f"{label} independent module inventory changed")
    for name, digest in EXPECTED_FROZEN_RUNTIME_MODULE_FILES.items():
        record = _require_mapping(modules[name], f"{label}.{name} module")
        module_path = Path(_require_string(record.get("path"), f"{label}.{name} path"))
        if (
            set(record) != {"path", "sha256"}
            or record.get("sha256") != digest
            or not module_path.is_absolute()
            or module_path.is_symlink()
            or not module_path.is_file()
            or sha256_file(module_path) != digest
        ):
            raise ValueError(f"{label} independent module {name} changed")
    expected_environment = {
        "PYTHONHASHSEED": str(EXPECTED_RUNTIME_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(expected_pycache),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(expected_pycache),
        "pycache_prefix_initially_empty": True,
    }
    _require_exact_json(
        runtime.get("python"),
        {
            "implementation": "CPython",
            "version": EXPECTED_FROZEN_RUNTIME_VERSIONS["python"],
            "executable": str(
                Path(
                    os.path.abspath(
                        EXPECTED_FROZEN_PYTHON_EXECUTABLE
                    )
                )
            ),
        },
        f"{label}.python",
    )
    _require_exact_json(
        runtime.get("venv"),
        {
            "prefix": str(expected_prefix),
            "base_prefix": "/usr",
            "pyvenv_cfg_path": str(expected_prefix / "pyvenv.cfg"),
            "pyvenv_cfg_sha256": EXPECTED_FROZEN_PYVENV_CONFIG_SHA256,
            "include_system_site_packages": True,
        },
        f"{label}.venv",
    )
    _require_exact_json(
        runtime.get("versions"),
        EXPECTED_FROZEN_RUNTIME_VERSIONS,
        f"{label}.versions",
    )
    _require_exact_json(
        runtime.get("cudnn"),
        {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        f"{label}.cudnn",
    )
    _require_exact_json(
        runtime.get("process_environment"),
        expected_environment,
        f"{label}.process_environment",
    )
    fixed_runtime = {
        "seed": EXPECTED_RUNTIME_SEED,
        "preprocess_profile": PREPROCESS_PROFILE,
        "inference_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "grad_enabled": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "minimum_cuda_free_bytes": EXPECTED_MINIMUM_CUDA_FREE_BYTES,
        "bytecode_writes_disabled": True,
        "network_allowed": False,
    }
    _require_exact_json(
        {key: runtime.get(key) for key in fixed_runtime},
        fixed_runtime,
        label,
    )
    if (
        not isinstance(runtime.get("platform"), str)
        or not runtime["platform"]
        or runtime.get("seed") != EXPECTED_RUNTIME_SEED
        or runtime.get("preprocess_profile") != PREPROCESS_PROFILE
        or runtime.get("inference_dtype") != "float32"
        or runtime.get("batch_size") != 1
        or runtime.get("autocast") is not False
        or runtime.get("grad_enabled") is not False
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cublas_workspace_config") != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("cudnn")
        != {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        }
        or runtime.get("matmul_allow_tf32") is not False
        or runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("minimum_cuda_free_bytes") != EXPECTED_MINIMUM_CUDA_FREE_BYTES
        or runtime.get("network_allowed") is not False
        or runtime.get("bytecode_writes_disabled") is not True
        or runtime.get("process_environment") != expected_environment
    ):
        raise ValueError(f"{label} independent numerical contract changed")
    if isinstance(device, str) and device.startswith("cuda:"):
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
            or type(cuda.get("device_index")) is not int
            or cuda.get("device_index") != int(device.split(":", 1)[1])
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
            raise ValueError(f"{label} independent CUDA contract changed")
    return runtime


def _configure_exact_recorded_runtime(
    *,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
    label: str,
) -> tuple[Any, dict[str, Any]]:
    recorded = _validate_runtime_contract(recorded_runtime, label=label)
    if recorded.get("device") != device_text:
        raise ValueError(f"{label} device differs from recorded runtime")
    device, current = _configure_frozen_runtime(device_text)
    if current != recorded:
        raise ValueError(f"{label} current runtime differs from recorded")
    return device, current


def _load_independent_model(
    *,
    source_root: Path,
    checkpoint_path: Path,
    device: Any,
    runtime: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
        or not checkpoint_path.is_absolute()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_BYTES
        or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("SPAI independent source/checkpoint path changed")
    pins = legacy_audit._load_runner_pins()
    source = legacy_audit._verify_source_tree(source_root, pins=pins)
    if source.get("commit") != EXPECTED_FROZEN_SOURCE_COMMIT:
        raise ValueError("SPAI independent source commit changed")
    model, schema, safety = legacy_audit._build_and_load_model(
        source_root,
        checkpoint_path,
        torch_module=torch,
    )
    if (
        schema.get("items_sha256") != EXPECTED_CHECKPOINT_SCHEMA_SHA256
        or schema.get("tensor_count") != legacy.CHECKPOINT["tensor_count"]
        or schema.get("state_elements") != legacy.CHECKPOINT["state_elements"]
    ):
        raise ValueError("SPAI independent checkpoint schema changed")
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (
        model,
        source,
        schema,
        {
            **safety,
            "runtime": dict(runtime),
        },
    )


def replay_persisted_artifacts(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, SampleArtifacts],
    source_root: Path,
    checkpoint_path: Path,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay patch->SCA/norm/MLP and normalized-feature->MLP on-device."""

    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="persisted SPAI artifact replay",
    )
    import torch

    model = None
    maxima = {
        "sca_feature": 0.0,
        "sca_attention": 0.0,
        "sca_raw_logit": 0.0,
        "feature_mlp_raw_logit": 0.0,
        "sca_probability": 0.0,
        "feature_mlp_probability": 0.0,
    }
    replayed = 0
    try:
        model, source, schema, safety = _load_independent_model(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device=device,
            runtime=runtime,
        )
        with torch.inference_mode():
            for row in latest_results:
                sample_id = str(row["sample_id"])
                _validate_score_payload(row, sample_id=sample_id)
                sample = artifacts.get(sample_id)
                if sample is None:
                    raise ValueError(f"missing persisted artifacts for {sample_id}")
                patch = torch.from_numpy(sample.patch_features.array).to(device)
                feature = torch.from_numpy(sample.feature.array).to(device)
                replay = legacy_audit._replay_patch_artifact(model, patch)
                feature_logit = model.cls_head(feature.unsqueeze(0)).reshape(())
                replay_feature = _tensor_float32_numpy(
                    replay.normalized_feature,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted SCA feature",
                )
                replay_attention = _tensor_float32_numpy(
                    replay.attention,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted SCA attention",
                )
                maxima["sca_feature"] = max(
                    maxima["sca_feature"],
                    _require_array_close(
                        sample.feature.array,
                        replay_feature,
                        label=f"{sample_id} persisted SCA feature",
                        tolerance=FEATURE_ABS_TOLERANCE,
                    ),
                )
                maxima["sca_attention"] = max(
                    maxima["sca_attention"],
                    _require_array_close(
                        sample.attention.array,
                        replay_attention,
                        label=f"{sample_id} persisted SCA attention",
                        tolerance=ATTENTION_ABS_TOLERANCE,
                    ),
                )
                sca_logit_tensor = _require_float32_tensor(
                    replay.raw_logit,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted SCA raw logit",
                )
                feature_logit_tensor = _require_float32_tensor(
                    feature_logit,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted feature-head raw logit",
                )
                sca_probability_tensor = torch.sigmoid(sca_logit_tensor.reshape(()))
                feature_probability_tensor = torch.sigmoid(
                    feature_logit_tensor.reshape(())
                )
                sca_raw = _tensor_float32_scalar(
                    sca_logit_tensor,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted SCA raw logit",
                )
                feature_raw = _tensor_float32_scalar(
                    feature_logit_tensor,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted feature-head raw logit",
                )
                sca_probability = _tensor_float32_scalar(
                    sca_probability_tensor,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted SCA probability",
                )
                feature_probability = _tensor_float32_scalar(
                    feature_probability_tensor,
                    torch_module=torch,
                    device=device,
                    label=f"{sample_id} persisted feature-head probability",
                )
                recorded_raw = _require_finite(
                    row.get("raw_logit"),
                    f"{sample_id} recorded raw logit",
                )
                recorded_probability = _require_finite(
                    row.get("probability"),
                    f"{sample_id} recorded probability",
                )
                maxima["sca_raw_logit"] = max(
                    maxima["sca_raw_logit"],
                    abs(sca_raw - recorded_raw),
                )
                maxima["feature_mlp_raw_logit"] = max(
                    maxima["feature_mlp_raw_logit"],
                    abs(feature_raw - recorded_raw),
                )
                maxima["sca_probability"] = max(
                    maxima["sca_probability"],
                    abs(sca_probability - recorded_probability),
                )
                maxima["feature_mlp_probability"] = max(
                    maxima["feature_mlp_probability"],
                    abs(feature_probability - recorded_probability),
                )
                if (
                    maxima["sca_raw_logit"] > RAW_LOGIT_ABS_TOLERANCE
                    or maxima["feature_mlp_raw_logit"] > RAW_LOGIT_ABS_TOLERANCE
                    or maxima["sca_probability"] > PROBABILITY_ABS_TOLERANCE
                    or maxima["feature_mlp_probability"] > PROBABILITY_ABS_TOLERANCE
                ):
                    raise ValueError(f"{sample_id} persisted SPAI replay differs")
                sca_decision = sca_probability > legacy.CLASSIFICATION_THRESHOLD
                feature_decision = feature_probability > legacy.CLASSIFICATION_THRESHOLD
                if (
                    row.get("classification_decision") is not sca_decision
                    or row.get("classification_decision") is not feature_decision
                ):
                    raise ValueError(f"{sample_id} persisted replay decision changed")
                replayed += 1
                del (
                    patch,
                    feature,
                    replay,
                    feature_logit,
                    sca_probability_tensor,
                    feature_probability_tensor,
                    replay_feature,
                    replay_attention,
                )
    finally:
        if model is not None:
            del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    if replayed != len(latest_results):
        raise ValueError("persisted SPAI artifact replay coverage incomplete")
    return {
        "status": "persisted_spai_artifact_replay_passed",
        "images_replayed": replayed,
        "patch_sca_norm_complete_mlp_replays": replayed,
        "normalized_feature_complete_mlp_replays": replayed,
        "attention_role": "classifier_diagnostic_not_T2_or_localization",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_schema_sha256": schema["items_sha256"],
        "source_commit": source["commit"],
        "checkpoint_safety": safety,
        "runtime": runtime,
        "maximum_absolute_differences": maxima,
        "tolerances": {
            "feature": FEATURE_ABS_TOLERANCE,
            "attention": ATTENTION_ABS_TOLERANCE,
            "raw_logit": RAW_LOGIT_ABS_TOLERANCE,
            "probability": PROBABILITY_ABS_TOLERANCE,
        },
    }


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    checkpoint_path: Path,
    device_text: str,
) -> dict[str, Any]:
    """Freshly execute FFT, ViT, SRS, SCA, norm, and MLP for all inputs."""

    if len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh SPAI replay requires all 1,775 images")
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"),
        "immutable.runtime",
    )
    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="fresh full SPAI replay",
    )
    source_record = _require_mapping(
        bundle.immutable.get("source"),
        "immutable.source",
    )
    assets = _require_mapping(
        bundle.immutable.get("assets"),
        "immutable.assets",
    )
    checkpoint_record = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    if source_root.resolve() != Path(str(source_record["root"])).resolve():
        raise ValueError("fresh SPAI source root differs from manifest")
    if checkpoint_path.resolve() != Path(str(checkpoint_record["path"])).resolve():
        raise ValueError("fresh SPAI checkpoint differs from manifest")
    import torch

    model = None
    replayed = 0
    maxima = {
        "patch_features": 0.0,
        "feature": 0.0,
        "attention": 0.0,
        "raw_logit": 0.0,
        "probability": 0.0,
    }
    try:
        model, replay_source, schema, safety = _load_independent_model(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device=device,
            runtime=runtime,
        )
        if replay_source.get("commit") != source_record.get("commit"):
            raise ValueError("fresh SPAI source audit differs from manifest")
        for canonical, row in zip(
            bundle.selected,
            bundle.latest_results,
            strict=True,
        ):
            sample_id = str(canonical["sample_id"])
            _validate_score_payload(row, sample_id=sample_id)
            input_path = _safe_repo_path(
                canonical.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                label=f"{sample_id} canonical input",
            )
            if sha256_file(input_path) != canonical.get("canonical_sha256"):
                raise ValueError(f"{sample_id} canonical input hash changed")
            prepared = legacy_audit.preprocess_image(
                input_path,
                torch_module=torch,
            )
            _require_exact_json(
                row.get("preprocess"),
                prepared.audit,
                f"{sample_id} preprocessing evidence",
            )
            expected_visibility = _independent_visibility_diagnostic(
                canonical,
                repo_root=bundle.release.repo_root,
            )
            _require_exact_json(
                {
                    key: row.get(key)
                    for key in expected_visibility
                },
                expected_visibility,
                f"{sample_id} patch visibility evidence",
            )
            image = prepared.tensor.to(device=device, dtype=torch.float32)
            _require_float32_tensor(
                image,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh preprocessed input",
            )
            with torch.inference_mode():
                fresh = legacy_audit._forward_with_evidence(
                    model,
                    image,
                    torch_module=torch,
                    feature_extraction_batch=FEATURE_EXTRACTION_BATCH,
                )
            patch = _tensor_float32_numpy(
                fresh.patch_features,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh patch features",
            )
            feature = _tensor_float32_numpy(
                fresh.normalized_feature,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh normalized feature",
            )
            attention = _tensor_float32_numpy(
                fresh.attention,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh diagnostic attention",
            )
            persisted = bundle.artifacts[sample_id]
            maxima["patch_features"] = max(
                maxima["patch_features"],
                _require_array_close(
                    persisted.patch_features.array,
                    patch,
                    label=f"{sample_id} fresh patch features",
                    tolerance=PATCH_FEATURE_ABS_TOLERANCE,
                ),
            )
            maxima["feature"] = max(
                maxima["feature"],
                _require_array_close(
                    persisted.feature.array,
                    feature,
                    label=f"{sample_id} fresh normalized feature",
                    tolerance=FEATURE_ABS_TOLERANCE,
                ),
            )
            maxima["attention"] = max(
                maxima["attention"],
                _require_array_close(
                    persisted.attention.array,
                    attention,
                    label=f"{sample_id} fresh diagnostic attention",
                    tolerance=ATTENTION_ABS_TOLERANCE,
                ),
            )
            raw_tensor = _require_float32_tensor(
                fresh.raw_logit,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh raw logit",
            )
            probability_tensor = torch.sigmoid(raw_tensor.reshape(()))
            raw = _tensor_float32_scalar(
                raw_tensor,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh raw logit",
            )
            probability = _tensor_float32_scalar(
                probability_tensor,
                torch_module=torch,
                device=device,
                label=f"{sample_id} fresh probability",
            )
            recorded_raw = _require_finite(
                row.get("raw_logit"),
                f"{sample_id} recorded raw logit",
            )
            recorded_probability = _require_finite(
                row.get("probability"),
                f"{sample_id} recorded probability",
            )
            raw_difference = abs(raw - recorded_raw)
            probability_difference = abs(probability - recorded_probability)
            maxima["raw_logit"] = max(
                maxima["raw_logit"],
                raw_difference,
            )
            maxima["probability"] = max(
                maxima["probability"],
                probability_difference,
            )
            if (
                raw_difference > RAW_LOGIT_ABS_TOLERANCE
                or probability_difference > PROBABILITY_ABS_TOLERANCE
            ):
                raise ValueError(f"{sample_id} fresh SPAI score differs")
            if (probability > legacy.CLASSIFICATION_THRESHOLD) is not row.get(
                "classification_decision"
            ):
                raise ValueError(f"{sample_id} fresh SPAI decision differs")
            replayed += 1
            del (
                image,
                fresh,
                patch,
                feature,
                attention,
                raw_tensor,
                probability_tensor,
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    if replayed != FORMAL_IMAGES:
        raise ValueError("fresh full SPAI replay coverage is incomplete")
    return {
        "status": "fresh_full_spai_model_replay_passed",
        "images_replayed": replayed,
        "full_fft_vit_srs_sca_norm_mlp_forward_per_input": True,
        "persisted_artifact_only_replay": False,
        "attention_role": "classifier_diagnostic_not_T2_or_localization",
        "source_commit": replay_source["commit"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_schema_sha256": schema["items_sha256"],
        "checkpoint_safety": safety,
        "preprocess_profile": PREPROCESS_PROFILE,
        "runtime": runtime,
        "maximum_absolute_differences": maxima,
        "tolerances": {
            "patch_features": PATCH_FEATURE_ABS_TOLERANCE,
            "feature": FEATURE_ABS_TOLERANCE,
            "attention": ATTENTION_ABS_TOLERANCE,
            "raw_logit": RAW_LOGIT_ABS_TOLERANCE,
            "probability": PROBABILITY_ABS_TOLERANCE,
        },
    }


def _assert_runner_contract_exports() -> Any:
    runner = _runner()
    expected = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_SEED": EXPECTED_RUNTIME_SEED,
        "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "MINIMUM_CUDA_FREE_BYTES": EXPECTED_MINIMUM_CUDA_FREE_BYTES,
        "FROZEN_PYTHON_EXECUTABLE": EXPECTED_FROZEN_PYTHON_EXECUTABLE,
        "FROZEN_VENV_PREFIX": EXPECTED_FROZEN_VENV_PREFIX,
        "FROZEN_PYTHONPYCACHEPREFIX": (EXPECTED_FROZEN_PYTHONPYCACHEPREFIX),
        "FROZEN_PYVENV_CONFIG_SHA256": (EXPECTED_FROZEN_PYVENV_CONFIG_SHA256),
        "FROZEN_RUNTIME_VERSIONS": EXPECTED_FROZEN_RUNTIME_VERSIONS,
        "FROZEN_RUNTIME_MODULE_FILES": (EXPECTED_FROZEN_RUNTIME_MODULE_FILES),
        "CPU_OFFICIAL_GOLDEN_CASES": (EXPECTED_CPU_OFFICIAL_GOLDEN_CASES),
        "FORMAL_SELECTED_ROWS_SHA256": EXPECTED_FORMAL_SELECTED_ROWS_SHA256,
        "FORMAL_SELECTED_IDS_SHA256": EXPECTED_FORMAL_SELECTED_IDS_SHA256,
        "SMOKE5X7_SELECTED_IDS_SHA256": EXPECTED_SMOKE5X7_SELECTED_IDS_SHA256,
        "SCORE_SPEC": _score_spec(),
    }
    for name, value in expected.items():
        if getattr(runner, name, None) != value:
            raise ValueError(f"SPAI runner export {name} changed")
    legacy_expected = {
        "PATCH_SIZE": PATCH_SIZE,
        "PATCH_STRIDE": PATCH_STRIDE,
        "MINIMUM_PATCHES": MINIMUM_PATCHES,
        "FEATURE_EXTRACTION_BATCH": FEATURE_EXTRACTION_BATCH,
        "FEATURE_DIMENSION": FEATURE_DIMENSION,
        "ATTENTION_HEADS": ATTENTION_HEADS,
        "PREPROCESS_PROFILE": PREPROCESS_PROFILE,
        "PATCH_FEATURE_SEMANTICS": PATCH_FEATURE_SEMANTICS,
        "FEATURE_SEMANTICS": FEATURE_SEMANTICS,
        "ATTENTION_SEMANTICS": ATTENTION_SEMANTICS,
    }
    for name, value in legacy_expected.items():
        if getattr(legacy, name, None) != value:
            raise ValueError(f"legacy SPAI pin {name} changed")
    required = (
        "select_mode_inputs",
        "selection_visibility_census",
        "configure_runtime",
        "validate_runtime_contract",
        "_valid_run_id",
        "_validate_runner_attempt",
        "_validate_physical_attempt_history",
        "_validate_preflight_report",
        "_local_artifact_policy",
    )
    for name in required:
        if not callable(getattr(runner, name, None)):
            raise ValueError(f"SPAI runner lacks required callable {name}")
    return runner


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    runner = _assert_runner_contract_exports()
    expected_paths = EXPECTED_ADAPTER_SOURCE_PATHS
    if tuple(runner.ADAPTER_SOURCE_PATHS) != expected_paths:
        raise ValueError("SPAI runner adapter source inventory changed")
    records = _require_mapping(value, "immutable.adapter_sources")
    if set(records) != set(expected_paths):
        raise ValueError("SPAI adapter source inventory changed")
    for relative in expected_paths:
        record = _require_mapping(records[relative], f"adapter {relative}")
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter {relative}",
        )
        if record != {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }:
            raise ValueError(f"SPAI adapter source {relative} changed")


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    runner = _assert_runner_contract_exports()
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
        label="Balanced250 manifest",
    )
    release = load_canonical_release(
        repo_root,
        manifest_path,
        verify_files=True,
    )
    if expected_mode == "formal":
        limit = None
    elif expected_mode == "smoke":
        selection = _require_mapping(
            raw_contract.get("selection"),
            "dataset contract selection",
        )
        spec_record = _require_mapping(
            selection.get("spec"),
            "dataset selection spec",
        )
        limit = spec_record.get("per_condition_limit")
        if type(limit) is not int or limit != SMOKE_PER_CONDITION:
            raise ValueError("SPAI smoke requires five images per condition")
    else:
        raise ValueError(f"unsupported SPAI analyzer mode {expected_mode}")
    spec, selected_value = runner.select_mode_inputs(
        release,
        mode=expected_mode,
        per_condition_limit=limit,
        sample_id=None,
    )
    selected = tuple(selected_value)
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=_score_spec(),
    )
    _require_exact_json(
        raw_contract,
        rebuilt.as_dict(),
        "SPAI rebuilt dataset contract",
    )
    expected_count = FORMAL_IMAGES if expected_mode == "formal" else SMOKE_IMAGES
    if len(selected) != expected_count:
        raise ValueError("SPAI selection size changed")
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
        raise ValueError("SPAI selection condition counts changed")
    if immutable.get("mode") != expected_mode:
        raise ValueError("SPAI immutable mode changed")
    _require_exact_json(
        immutable.get("score_spec"),
        _score_spec().as_dict(),
        "SPAI immutable score spec",
    )
    rows_digest = _rows_sha256(selected)
    ids_digest = selected_ids_sha256(
        str(row["sample_id"]) for row in selected
    )
    if immutable.get("selected_rows_sha256") != rows_digest:
        raise ValueError("SPAI selected-row digest changed")
    if immutable.get("selected_ids_sha256") != ids_digest:
        raise ValueError("SPAI selected-ID digest changed")
    if expected_mode == "formal" and (
        rows_digest != EXPECTED_FORMAL_SELECTED_ROWS_SHA256
        or ids_digest != EXPECTED_FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("SPAI independent formal selection pin changed")
    if (
        expected_mode == "smoke"
        and ids_digest != EXPECTED_SMOKE5X7_SELECTED_IDS_SHA256
    ):
        raise ValueError("SPAI independent smoke selection pin changed")
    visibility = _independent_visibility_census(
        selected,
        repo_root=repo_root,
    )
    _require_exact_json(
        immutable.get("selection_visibility_census"),
        visibility,
        "SPAI immutable visibility census",
    )
    _require_exact_json(
        runner.selection_visibility_census(
            selected,
            repo_root=repo_root,
        ),
        visibility,
        "SPAI runner/independent visibility censuses",
    )
    return release, selected, rebuilt


def _validate_source_contract(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "immutable.source")
    root = Path(_require_string(source.get("root"), "source root"))
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != Path(os.path.abspath(root))
    ):
        raise ValueError("SPAI source root is unsafe")
    actual = legacy._verify_source_contract(root)
    if source != actual:
        raise ValueError("SPAI source audit differs from frozen source")
    pins = legacy_audit._load_runner_pins()
    independent = legacy_audit._verify_source_tree(root, pins=pins)
    if independent.get("commit") != source.get("commit") or independent.get(
        "config_sha256"
    ) != source.get("source_files", {}).get("configs/spai.yaml", {}).get("sha256"):
        raise ValueError("SPAI independent source verification changed")
    return source


def _validate_assets_contract(value: Any) -> dict[str, Any]:
    assets = _require_mapping(value, "immutable.assets")
    if set(assets) != {"checkpoint", "golden_assets"}:
        raise ValueError("SPAI asset key set changed")
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    expected_checkpoint_keys = {
        *legacy.CHECKPOINT,
        "path",
        "actual_bytes",
        "actual_sha256",
        "serialization_safety",
        "schema",
    }
    path = Path(_require_string(checkpoint.get("path"), "checkpoint path"))
    _require_exact_json(
        {key: checkpoint.get(key) for key in legacy.CHECKPOINT},
        legacy.CHECKPOINT,
        "SPAI checkpoint frozen metadata",
    )
    _require_exact_json(
        checkpoint.get("serialization_safety"),
        {
            "weights_only": True,
            "pickle_executed": False,
            "safe_global_allowlist": ["yacs.config.CfgNode"],
            "loader": "torch.load(map_location=cpu, weights_only=True)",
        },
        "SPAI checkpoint serialization safety",
    )
    _require_nonnegative_int(
        checkpoint.get("actual_bytes"),
        "SPAI checkpoint actual bytes",
    )
    if (
        set(checkpoint) != expected_checkpoint_keys
        or any(
            checkpoint.get(key) != expected
            for key, expected in legacy.CHECKPOINT.items()
        )
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != EXPECTED_CHECKPOINT_BYTES
        or sha256_file(path) != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("actual_sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("actual_bytes") != EXPECTED_CHECKPOINT_BYTES
    ):
        raise ValueError("SPAI checkpoint asset changed")
    schema = _require_mapping(
        checkpoint.get("schema"),
        "checkpoint schema",
    )
    expected_schema_keys = {
        "tensor_count",
        "state_elements",
        "dtype_counts",
        "items_sha256",
        "items",
        "embedded_config_minimum_patches",
        "released_inference_config_minimum_patches",
        "embedded_config_is_historical_not_restored",
    }
    for key in (
        "tensor_count",
        "state_elements",
        "embedded_config_minimum_patches",
        "released_inference_config_minimum_patches",
    ):
        _require_nonnegative_int(
            schema.get(key),
            f"checkpoint schema {key}",
        )
    if schema.get("embedded_config_is_historical_not_restored") is not True:
        raise ValueError("SPAI checkpoint schema boolean changed")
    items = _require_list(schema.get("items"), "checkpoint schema items")
    item_keys = {"key", "shape", "dtype", "numel", "sha256"}
    seen_keys: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    state_elements = 0
    for index, item in enumerate(items):
        record = _require_mapping(item, f"checkpoint schema item {index}")
        key = _require_string(
            record.get("key"),
            f"checkpoint schema item {index} key",
        )
        shape = _require_list(
            record.get("shape"),
            f"checkpoint schema item {index} shape",
        )
        dtype = _require_string(
            record.get("dtype"),
            f"checkpoint schema item {index} dtype",
        )
        numel = _require_nonnegative_int(
            record.get("numel"),
            f"checkpoint schema item {index} numel",
        )
        if (
            set(record) != item_keys
            or key in seen_keys
            or dtype not in {"torch.float32", "torch.int64"}
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            )
            or math.prod(shape) != numel
        ):
            raise ValueError("SPAI checkpoint tensor item changed")
        _require_sha256(
            record.get("sha256"),
            f"checkpoint schema item {index} SHA-256",
        )
        seen_keys.add(key)
        dtype_counts[dtype] += 1
        state_elements += numel
    if (
        set(schema) != expected_schema_keys
        or schema.get("items_sha256") != EXPECTED_CHECKPOINT_SCHEMA_SHA256
        or schema.get("tensor_count") != legacy.CHECKPOINT["tensor_count"]
        or schema.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or len(items) != legacy.CHECKPOINT["tensor_count"]
        or len(seen_keys) != len(items)
        or state_elements != legacy.CHECKPOINT["state_elements"]
        or schema.get("dtype_counts") != dict(dtype_counts)
        or dtype_counts != Counter({"torch.float32": 323, "torch.int64": 1})
        or hashlib.sha256(stable_json(items).encode()).hexdigest()
        != EXPECTED_CHECKPOINT_SCHEMA_SHA256
        or schema.get("embedded_config_minimum_patches")
        != legacy.CHECKPOINT["embedded_minimum_patches"]
        or schema.get("released_inference_config_minimum_patches") != MINIMUM_PATCHES
        or schema.get("embedded_config_is_historical_not_restored") is not True
        or checkpoint.get("serialization_safety")
        != {
            "weights_only": True,
            "pickle_executed": False,
            "safe_global_allowlist": ["yacs.config.CfgNode"],
            "loader": "torch.load(map_location=cpu, weights_only=True)",
        }
    ):
        raise ValueError("SPAI checkpoint schema/safety changed")
    _require_exact_json(
        schema.get("dtype_counts"),
        {"torch.float32": 323, "torch.int64": 1},
        "SPAI checkpoint dtype counts",
    )
    golden_assets = _require_list(
        assets.get("golden_assets"),
        "immutable.assets.golden_assets",
    )
    if len(golden_assets) != len(legacy.GOLDEN_CASES):
        raise ValueError("SPAI golden asset count changed")
    for record, frozen in zip(
        golden_assets,
        legacy.GOLDEN_CASES,
        strict=True,
    ):
        mapping = _require_mapping(record, "SPAI golden asset")
        golden_path = Path(_require_string(mapping.get("path"), "golden asset path"))
        _require_exact_json(
            {key: mapping.get(key) for key in frozen},
            frozen,
            "SPAI golden frozen metadata",
        )
        _require_nonnegative_int(
            mapping.get("bytes"),
            "SPAI golden asset bytes",
        )
        if (
            set(mapping) != {*frozen, "path", "bytes"}
            or golden_path.is_symlink()
            or not golden_path.is_file()
            or mapping.get("bytes") != golden_path.stat().st_size
            or sha256_file(golden_path) != frozen["sha256"]
            or any(mapping.get(key) != expected for key, expected in frozen.items())
        ):
            raise ValueError("SPAI official golden asset changed")
    return assets


def _golden_assets_by_relative_path(
    assets: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _require_list(
        assets.get("golden_assets"),
        "immutable.assets.golden_assets",
    ):
        mapping = _require_mapping(record, "SPAI golden asset")
        relative = _require_string(
            mapping.get("relative_path"),
            "SPAI golden relative path",
        )
        if relative in result:
            raise ValueError("SPAI golden assets contain a duplicate")
        result[relative] = mapping
    return result


def _validate_cpu_official_golden(
    value: Any,
    *,
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    report = _require_mapping(value, "SPAI CPU official golden")
    expected_keys = {
        "status",
        "device",
        "runs_per_case",
        "cpu_repeat_tolerance",
        "cross_device_bit_equality_required",
        "cuda_reference_is_not_cpu_acceptance_gate",
        "cases",
    }
    cases = _require_list(report.get("cases"), "SPAI CPU golden cases")
    _require_exact_json(
        {key: report.get(key) for key in expected_keys if key != "cases"},
        {
            "status": "passed",
            "device": "cpu",
            "runs_per_case": 2,
            "cpu_repeat_tolerance": 0.0,
            "cross_device_bit_equality_required": False,
            "cuda_reference_is_not_cpu_acceptance_gate": True,
        },
        "SPAI CPU official golden identity",
    )
    if (
        set(report) != expected_keys
        or report.get("status") != "passed"
        or report.get("device") != "cpu"
        or report.get("runs_per_case") != 2
        or report.get("cpu_repeat_tolerance") != 0.0
        or report.get("cross_device_bit_equality_required") is not False
        or report.get("cuda_reference_is_not_cpu_acceptance_gate") is not True
        or len(cases) != len(EXPECTED_CPU_OFFICIAL_GOLDEN_CASES)
    ):
        raise ValueError("SPAI CPU official golden contract changed")
    assets_by_relative = _golden_assets_by_relative_path(assets)
    import torch

    for case, expected_cpu, frozen in zip(
        cases,
        EXPECTED_CPU_OFFICIAL_GOLDEN_CASES,
        legacy.GOLDEN_CASES,
        strict=True,
    ):
        mapping = _require_mapping(case, "SPAI CPU golden case")
        relative = str(frozen["relative_path"])
        asset = assets_by_relative.get(relative)
        if asset is None:
            raise ValueError("SPAI CPU golden asset is missing")
        path = Path(str(asset["path"]))
        prepared = legacy_audit.preprocess_image(
            path,
            torch_module=torch,
        )
        expected_observed = {
            key: expected_cpu[key]
            for key in (
                "raw_logit",
                "probability",
                "patch_features_array_sha256",
                "feature_array_sha256",
                "attention_array_sha256",
            )
        }
        expected_observed["peak_cuda_memory_bytes"] = 0
        expected_cuda_reference = {
            "raw_logit": frozen["raw_logit"],
            "probability": frozen["probability"],
            "logit_absolute_difference_from_cpu": abs(
                float(frozen["raw_logit"]) - float(expected_cpu["raw_logit"])
            ),
            "probability_absolute_difference_from_cpu": abs(
                float(frozen["probability"]) - float(expected_cpu["probability"])
            ),
            "role": (
                "released CUDA implementation regression; not a CPU "
                "bit-equality acceptance value"
            ),
        }
        expected_case = {
            "relative_path": relative,
            "path": str(path.resolve()),
            "sha256": frozen["sha256"],
            "preprocess": prepared.audit,
            "cpu_observed_runs": [
                expected_observed,
                expected_observed,
            ],
            "cpu_bit_identical_repeats": True,
            "cuda_reference": expected_cuda_reference,
            "passed": True,
        }
        try:
            _require_exact_json(
                mapping,
                expected_case,
                f"SPAI CPU official golden {relative}",
            )
        except ValueError as error:
            raise ValueError(f"SPAI CPU official golden {relative} changed") from error
    return report


def _validate_cuda_official_golden(
    value: Any,
    *,
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    report = _require_mapping(value, "SPAI CUDA official golden")
    expected_keys = {
        "status",
        "source",
        "runs_per_case",
        "logit_absolute_tolerance",
        "probability_absolute_tolerance",
        "official_vs_adapter_full_forward",
        "website_display_reference",
        "cases",
    }
    cases = _require_list(report.get("cases"), "SPAI CUDA golden cases")
    _require_exact_json(
        {key: report.get(key) for key in expected_keys if key != "cases"},
        {
            "status": "passed",
            "source": (
                "official evaluation-bundle originals, current released "
                "source/checkpoint, pinned highest/no-TF32 float32 runtime"
            ),
            "runs_per_case": 2,
            "logit_absolute_tolerance": (legacy.GOLDEN_LOGIT_ABS_TOLERANCE),
            "probability_absolute_tolerance": (legacy.GOLDEN_PROBABILITY_ABS_TOLERANCE),
            "official_vs_adapter_full_forward": True,
            "website_display_reference": (
                "official website uses compressed derivatives and displays "
                "0.748/0.87; those values do not match the released "
                "executable regression and are disclosed rather than used "
                "as a gate"
            ),
        },
        "SPAI CUDA official golden identity",
    )
    if (
        set(report) != expected_keys
        or report.get("status") != "passed"
        or report.get("source")
        != (
            "official evaluation-bundle originals, current released source/"
            "checkpoint, pinned highest/no-TF32 float32 runtime"
        )
        or report.get("runs_per_case") != 2
        or report.get("logit_absolute_tolerance") != legacy.GOLDEN_LOGIT_ABS_TOLERANCE
        or report.get("probability_absolute_tolerance")
        != legacy.GOLDEN_PROBABILITY_ABS_TOLERANCE
        or report.get("official_vs_adapter_full_forward") is not True
        or report.get("website_display_reference")
        != (
            "official website uses compressed derivatives and displays "
            "0.748/0.87; those values do not match the released executable "
            "regression and are disclosed rather than used as a gate"
        )
        or len(cases) != len(legacy.GOLDEN_CASES)
    ):
        raise ValueError("SPAI CUDA official golden contract changed")
    assets_by_relative = _golden_assets_by_relative_path(assets)
    import torch

    dynamic_keys = {
        "path",
        "preprocess",
        "observed_runs",
        "artifact_hashes",
        "bit_identical_repeats",
        "logit_absolute_difference",
        "probability_absolute_difference",
        "passed",
        "website_derivative_display_matches_released_regression",
    }
    for case, frozen in zip(cases, legacy.GOLDEN_CASES, strict=True):
        mapping = _require_mapping(case, "SPAI CUDA golden case")
        relative = str(frozen["relative_path"])
        asset = assets_by_relative.get(relative)
        if asset is None:
            raise ValueError("SPAI CUDA golden asset is missing")
        path = Path(str(asset["path"]))
        prepared = legacy_audit.preprocess_image(
            path,
            torch_module=torch,
        )
        _require_exact_json(
            {key: mapping.get(key) for key in frozen},
            frozen,
            f"SPAI CUDA frozen golden {relative}",
        )
        _require_exact_json(
            mapping.get("preprocess"),
            prepared.audit,
            f"SPAI CUDA preprocess {relative}",
        )
        observed = _require_list(
            mapping.get("observed_runs"),
            f"{relative} observed CUDA runs",
        )
        artifact_hashes = _require_list(
            mapping.get("artifact_hashes"),
            f"{relative} CUDA artifact hashes",
        )
        if (
            set(mapping) != {*frozen, *dynamic_keys}
            or any(mapping.get(key) != value for key, value in frozen.items())
            or mapping.get("path") != str(path.resolve())
            or mapping.get("preprocess") != prepared.audit
            or len(observed) != 2
            or observed[0] != observed[1]
            or len(artifact_hashes) != 2
            or artifact_hashes[0] != artifact_hashes[1]
            or mapping.get("bit_identical_repeats") is not True
            or mapping.get("passed") is not True
            or mapping.get("website_derivative_display_matches_released_regression")
            is not False
        ):
            raise ValueError(f"SPAI CUDA official golden {relative} changed")
        for observed_run in observed:
            observed_mapping = _require_mapping(
                observed_run,
                f"{relative} CUDA observed run",
            )
            if set(observed_mapping) != {"raw_logit", "probability"}:
                raise ValueError("SPAI CUDA observed score keys changed")
            _require_finite(
                observed_mapping["raw_logit"],
                f"{relative} CUDA raw logit",
            )
            probability = _require_finite(
                observed_mapping["probability"],
                f"{relative} CUDA probability",
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("SPAI CUDA probability changed")
        for hashes in artifact_hashes:
            hash_mapping = _require_mapping(
                hashes,
                f"{relative} CUDA artifact hashes",
            )
            if set(hash_mapping) != {
                "patch_features",
                "feature",
                "attention",
            }:
                raise ValueError("SPAI CUDA artifact hash keys changed")
            for name, digest in hash_mapping.items():
                _require_sha256(digest, f"{relative} CUDA {name} hash")
        logit_difference = abs(
            float(observed[0]["raw_logit"]) - float(frozen["raw_logit"])
        )
        probability_difference = abs(
            float(observed[0]["probability"]) - float(frozen["probability"])
        )
        recorded_logit_difference = _require_finite(
            mapping.get("logit_absolute_difference"),
            f"{relative} CUDA logit difference",
        )
        recorded_probability_difference = _require_finite(
            mapping.get("probability_absolute_difference"),
            f"{relative} CUDA probability difference",
        )
        if (
            recorded_logit_difference != logit_difference
            or recorded_probability_difference != probability_difference
            or logit_difference > legacy.GOLDEN_LOGIT_ABS_TOLERANCE
            or probability_difference > legacy.GOLDEN_PROBABILITY_ABS_TOLERANCE
        ):
            raise ValueError(f"SPAI CUDA golden {relative} score changed")
    return report


def _validate_cpu_model_load(value: Any) -> None:
    model_load = _require_mapping(value, "SPAI CPU model load")
    expected_config = {
        "sid_approach": "freq_restoration",
        "resolution_mode": "arbitrary",
        "required_normalization": "positive_0_1",
        "image_patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "minimum_patches": MINIMUM_PATCHES,
        "feature_extraction_batch": FEATURE_EXTRACTION_BATCH,
        "num_classes": 2,
        "attention_heads": ATTENTION_HEADS,
        "attention_embed_dimension": legacy.ATTENTION_EMBED_DIMENSION,
        "original_resolution": True,
    }
    expected_load = {
        "strict": True,
        "full_state_coverage": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "loaded_tensor_exact_match": True,
    }
    expected_model = {
        "class": "spai.models.sid.PatchBasedMFViT",
        "state_tensors": legacy.CHECKPOINT["tensor_count"],
        "state_elements": legacy.CHECKPOINT["state_elements"],
        "feature_dimension": FEATURE_DIMENSION,
        "attention_heads": ATTENTION_HEADS,
        "eval": True,
    }
    expected_attempts = {
        "urllib_urlopen": 0,
        "socket_create_connection": 0,
        "socket_connect": 0,
        "torch_hub_load": 0,
        "torch_hub_load_state_dict_from_url": 0,
    }
    expected = {
        "config": expected_config,
        "load": expected_load,
        "model": expected_model,
        "network": {
            "allowed": False,
            "attempts": expected_attempts,
        },
    }
    try:
        _require_exact_json(model_load, expected, "SPAI CPU model load")
    except ValueError as error:
        raise ValueError("SPAI CPU strict model-load contract changed") from error


def _validate_cpu_preflight(
    value: Any,
    *,
    repo_root: Path,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    wrapper = _require_mapping(value, "immutable.cpu_preflight")
    if (
        set(wrapper)
        != {
            "performed_before_dataset_and_accelerator_configuration",
            "report",
        }
        or wrapper.get("performed_before_dataset_and_accelerator_configuration")
        is not True
    ):
        raise ValueError("SPAI CPU preflight ordering evidence changed")
    report = _require_mapping(
        wrapper.get("report"),
        "immutable.cpu_preflight.report",
    )
    expected_report_keys = {
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
    runner._validate_preflight_report(
        report,
        source=source,
        assets=assets,
    )
    _require_exact_json(
        report.get("source"),
        source,
        "SPAI CPU preflight source",
    )
    _require_exact_json(
        report.get("assets"),
        assets,
        "SPAI CPU preflight assets",
    )
    if (
        set(report) != expected_report_keys
        or report.get("schema_version") != EXPECTED_CPU_PREFLIGHT_SCHEMA
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
    runtime = _validate_runtime_contract(
        report.get("runtime"),
        label="SPAI CPU preflight runtime",
    )
    if runtime.get("device") != "cpu":
        raise ValueError("SPAI CPU preflight runtime is not CPU")
    _validate_cpu_model_load(report.get("model_load"))
    preprocess_equivalence = _require_mapping(
        report.get("official_preprocess_equivalence"),
        "SPAI official preprocess equivalence",
    )
    if set(preprocess_equivalence) != {"status", "official_transform", "cases"}:
        raise ValueError("SPAI CPU official gates changed")
    _require_exact_json(
        preprocess_equivalence,
        legacy.validate_official_preprocess_equivalence(),
        "SPAI official preprocess equivalence",
    )
    _validate_cpu_official_golden(
        report.get("official_golden"),
        assets=assets,
    )
    balanced = _require_mapping(
        report.get("balanced_golden"),
        "SPAI Balanced CPU golden",
    )
    image_path = _safe_repo_path(
        runner.CPU_GOLDEN_INPUT_PATH,
        repo_root=repo_root,
        label="SPAI CPU golden input",
    )
    if sha256_file(image_path) != runner.CPU_GOLDEN_IMAGE_SHA256:
        raise ValueError("SPAI CPU golden input changed")
    import torch

    independent = legacy_audit.preprocess_image(
        image_path,
        torch_module=torch,
    )
    _require_exact_json(
        balanced.get("preprocess"),
        independent.audit,
        "SPAI CPU golden independent preprocess",
    )
    expected_artifacts = {
        "patch_features": {
            "array_sha256": runner.CPU_GOLDEN_PATCH_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_PATCH_FILE_SHA256,
            "shape": [4, FEATURE_DIMENSION],
            "dtype": "float32",
            "nbytes": 4 * FEATURE_DIMENSION * 4,
            "file_bytes": 17664,
        },
        "feature": {
            "array_sha256": runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_FEATURE_FILE_SHA256,
            "shape": [FEATURE_DIMENSION],
            "dtype": "float32",
            "nbytes": FEATURE_DIMENSION * 4,
            "file_bytes": 4512,
        },
        "attention": {
            "array_sha256": runner.CPU_GOLDEN_ATTENTION_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_ATTENTION_FILE_SHA256,
            "shape": [ATTENTION_HEADS, 4],
            "dtype": "float32",
            "nbytes": ATTENTION_HEADS * 4 * 4,
            "file_bytes": 320,
        },
    }
    expected_manual = {
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "model_forward_calls": 1,
        "to_kv_hook_calls": 1,
        "attention_hook_calls": 1,
        "norm_hook_calls": 1,
        "official_attention_exact_match": True,
        "official_aggregated_exact_match": True,
        "official_feature_exact_match": True,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "sca_replay": True,
        "norm_replay": True,
        "complete_mlp_replay": True,
    }
    expected_balanced_keys = {
        "sample_id",
        "input_path",
        "image_sha256",
        "input_width",
        "input_height",
        "preprocess",
        "patch_features",
        "feature",
        "attention",
        "raw_logit",
        "probability",
        "ai_score",
        "classification_decision",
        "manual_replay",
        "peak_cuda_memory_bytes",
        "repeat_patch_features_file_sha256",
        "repeat_feature_file_sha256",
        "repeat_attention_file_sha256",
        "repeat_raw_logit",
        "repeat_probability",
        "repeat_byte_exact",
    }
    expected_balanced = {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 640,
        "input_height": 640,
        "preprocess": independent.audit,
        **expected_artifacts,
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "manual_replay": expected_manual,
        "peak_cuda_memory_bytes": 0,
        "repeat_patch_features_file_sha256": (runner.CPU_GOLDEN_PATCH_FILE_SHA256),
        "repeat_feature_file_sha256": (runner.CPU_GOLDEN_FEATURE_FILE_SHA256),
        "repeat_attention_file_sha256": (runner.CPU_GOLDEN_ATTENTION_FILE_SHA256),
        "repeat_raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": runner.CPU_GOLDEN_PROBABILITY,
        "repeat_byte_exact": True,
    }
    _require_exact_json(
        balanced,
        expected_balanced,
        "SPAI CPU Balanced golden",
    )
    if (
        set(balanced) != expected_balanced_keys
        or balanced.get("sample_id") != runner.CPU_GOLDEN_SAMPLE_ID
        or balanced.get("input_path") != runner.CPU_GOLDEN_INPUT_PATH
        or balanced.get("image_sha256") != runner.CPU_GOLDEN_IMAGE_SHA256
        or balanced.get("input_width") != 640
        or balanced.get("input_height") != 640
        or balanced.get("raw_logit") != runner.CPU_GOLDEN_RAW_LOGIT
        or balanced.get("probability") != runner.CPU_GOLDEN_PROBABILITY
        or balanced.get("ai_score") != runner.CPU_GOLDEN_PROBABILITY
        or balanced.get("classification_decision") is not False
        or balanced.get("manual_replay") != expected_manual
        or balanced.get("peak_cuda_memory_bytes") != 0
        or balanced.get("repeat_patch_features_file_sha256")
        != runner.CPU_GOLDEN_PATCH_FILE_SHA256
        or balanced.get("repeat_feature_file_sha256")
        != runner.CPU_GOLDEN_FEATURE_FILE_SHA256
        or balanced.get("repeat_attention_file_sha256")
        != runner.CPU_GOLDEN_ATTENTION_FILE_SHA256
        or balanced.get("repeat_raw_logit") != runner.CPU_GOLDEN_RAW_LOGIT
        or balanced.get("repeat_probability") != runner.CPU_GOLDEN_PROBABILITY
        or balanced.get("repeat_byte_exact") is not True
        or any(
            balanced.get(name) != expected
            for name, expected in expected_artifacts.items()
        )
    ):
        raise ValueError("SPAI CPU Balanced golden changed")
    return report


def _validate_execution_device_golden(
    value: Any,
    *,
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    wrapper = _require_mapping(
        value,
        "immutable.execution_device_golden",
    )
    if (
        set(wrapper)
        != {
            "performed_after_explicit_device_configuration_before_scoring",
            "cross_device_bit_equality_required",
            "report",
        }
        or wrapper.get("performed_after_explicit_device_configuration_before_scoring")
        is not True
        or wrapper.get("cross_device_bit_equality_required") is not False
    ):
        raise ValueError("SPAI execution-device golden wrapper changed")
    report = _require_mapping(
        wrapper.get("report"),
        "immutable.execution_device_golden.report",
    )
    device = _require_string(runtime.get("device"), "immutable runtime device")
    expected_outer = {
        "status": "passed",
        "device": device,
        "reference_device": "cpu" if device == "cpu" else "cuda",
        "gate": (
            "frozen_CPU_bit_repeat_regression"
            if device == "cpu"
            else "released_CUDA_highest_no_TF32_implementation_regression"
        ),
        "cross_device_bit_equality_required": False,
    }
    _require_exact_json(
        {key: report.get(key) for key in expected_outer},
        expected_outer,
        "SPAI execution-device golden identity",
    )
    if set(report) != {*expected_outer, "report"} or any(
        report.get(key) != expected for key, expected in expected_outer.items()
    ):
        raise ValueError("SPAI execution-device golden identity changed")
    if device == "cpu":
        nested = _validate_cpu_official_golden(
            report.get("report"),
            assets=assets,
        )
        if nested != cpu_preflight.get("official_golden"):
            raise ValueError("SPAI execution CPU golden differs from CPU preflight")
    elif re.fullmatch(r"cuda:[0-9]+", device):
        _validate_cuda_official_golden(
            report.get("report"),
            assets=assets,
        )
    else:
        raise ValueError("SPAI execution-device golden device changed")


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, dict[str, Any]]:
    runner = _assert_runner_contract_exports()
    run_id = runner._valid_run_id(run_id)
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
        raise ValueError("SPAI manifest key set changed")
    if (
        manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("SPAI analyzer requires the exact complete run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    immutable = _require_mapping(
        manifest.get("immutable"),
        "manifest.immutable",
    )
    if (
        frozenset(runner.IMMUTABLE_CONFIG_KEYS) != EXPECTED_IMMUTABLE_CONFIG_KEYS
        or set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS
    ):
        raise ValueError("SPAI immutable config key set changed")
    if (
        immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
    ):
        raise ValueError("SPAI immutable identity changed")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest fingerprint",
    )
    if fingerprint != hashlib.sha256(stable_json(immutable).encode()).hexdigest():
        raise ValueError("SPAI manifest fingerprint does not bind immutable")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    _require_exact_json(
        {
            "model": immutable.get("model"),
            "preprocess": immutable.get("preprocess"),
            "artifact_contract": immutable.get("artifact_contract"),
            "task_scope": immutable.get("task_scope"),
            "score_spec": immutable.get("score_spec"),
        },
        {
            "model": runner.MODEL_CONTRACT,
            "preprocess": runner.PREPROCESS_CONTRACT,
            "artifact_contract": runner.ARTIFACT_CONTRACT,
            "task_scope": runner.TASK_SCOPE,
            "score_spec": _score_spec().as_dict(),
        },
        "SPAI immutable method contract",
    )
    _require_exact_json(
        immutable.get("formal_local_visibility_census"),
        runner.LOCAL_VISIBILITY_CENSUS,
        "SPAI formal visibility pin",
    )
    _require_exact_json(
        immutable.get("local_artifact_policy"),
        runner._local_artifact_policy(repo_root),
        "SPAI local artifact policy",
    )
    source = _validate_source_contract(immutable.get("source"))
    assets = _validate_assets_contract(immutable.get("assets"))
    runtime = _validate_runtime_contract(
        immutable.get("runtime"),
        label="immutable.runtime",
    )
    cpu_preflight = _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        repo_root=repo_root,
        source=source,
        assets=assets,
    )
    _validate_execution_device_golden(
        immutable.get("execution_device_golden"),
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        assets=assets,
    )
    outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable.outputs",
    )
    for field, value in outputs.items():
        if field.endswith(("_path", "_dir", "_root")) or field == "run_dir":
            _safe_repo_path(
                value,
                repo_root=repo_root,
                label=f"immutable.outputs.{field}",
                require_file=False,
            )
    _reject_nonfinite_numbers(manifest, "manifest")
    _reject_unsupported_claims(manifest, "manifest")
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
    _require_exact_json(
        expected_rows,
        list(selected),
        "SPAI expected-input snapshot",
    )
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    expected = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    _require_exact_json(
        dataset,
        expected,
        "SPAI manifest dataset evidence",
    )


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
    for index, attempt in enumerate(physical):
        sample_id = _require_string(
            attempt.get("sample_id"),
            f"physical result {index} sample_id",
        )
        canonical = inputs.get(sample_id)
        if canonical is None:
            raise ValueError("SPAI physical result is outside selection")
        runner._validate_runner_attempt(
            attempt,
            input_row=canonical,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        expected_visibility = _independent_visibility_diagnostic(
            canonical,
            repo_root=repo_root,
        )
        _require_exact_json(
            {
                key: attempt.get(key)
                for key in expected_visibility
            },
            expected_visibility,
            f"{sample_id} visibility evidence",
        )
        if attempt.get("status") == "ok":
            expected_task_scope = {
                "valid_for_t1": True,
                "valid_for_t2": False,
                "native_dense_output": False,
            }
            if (
                attempt.get("preprocess_profile") != PREPROCESS_PROFILE
                or attempt.get("valid_for_metrics") is not True
                or attempt.get("attention_is_diagnostic_not_t2") is not True
            ):
                raise ValueError(f"{sample_id} SPAI T1 scope changed")
            _require_exact_json(
                attempt.get("task_scope"),
                expected_task_scope,
                f"{sample_id} SPAI T1 scope",
            )
            _validate_score_payload(attempt, sample_id=sample_id)
            input_path = _safe_repo_path(
                canonical.get("canonical_path"),
                repo_root=repo_root,
                label=f"{sample_id} canonical input",
            )
            import torch

            prepared = legacy_audit.preprocess_image(
                input_path,
                torch_module=torch,
            )
            _require_exact_json(
                attempt.get("preprocess"),
                prepared.audit,
                f"{sample_id} preprocessing",
            )
        _reject_nonfinite_numbers(attempt, f"physical result {index}")
        _reject_unsupported_claims(attempt, f"physical result {index}")


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
        dict(latest.latest_by_sample_id[str(row["sample_id"])]) for row in selected
    )
    return ordered, coverage.as_dict()


def _artifact_inventory_sha256(
    artifacts: Mapping[str, SampleArtifacts],
) -> str:
    records: list[dict[str, Any]] = []
    for sample_id, sample in sorted(artifacts.items()):
        for artifact in (
            sample.patch_features,
            sample.feature,
            sample.attention,
        ):
            records.append(
                {
                    "sample_id": sample_id,
                    "kind": artifact.kind,
                    "path": artifact.path.resolve().as_posix(),
                    "file_sha256": artifact.file_sha256,
                    "file_bytes": artifact.file_bytes,
                    "array_sha256": artifact.array_sha256,
                }
            )
    return hashlib.sha256(stable_json(records).encode()).hexdigest()


def _capture_evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    artifacts: Mapping[str, SampleArtifacts],
    primary_snapshot: Mapping[str, str] | None = None,
) -> dict[str, str]:
    current = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if primary_snapshot is not None and current != dict(primary_snapshot):
        raise ValueError("SPAI evidence changed during validation")
    for sample in artifacts.values():
        for artifact in (
            sample.patch_features,
            sample.feature,
            sample.attention,
        ):
            if (
                artifact.path.is_symlink()
                or not artifact.path.is_file()
                or artifact.path.stat().st_size != artifact.file_bytes
                or sha256_file(artifact.path) != artifact.file_sha256
            ):
                raise ValueError("SPAI artifact changed during validation")
    return {
        **current,
        "artifact_inventory_sha256": _artifact_inventory_sha256(artifacts),
    }


def _validate_execution(
    *,
    manifest: Mapping[str, Any],
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    expected_keys = {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
        "same_device_artifact_replays_before_execution",
        "same_device_artifact_replays_final",
    }
    if set(execution) != expected_keys:
        raise ValueError("SPAI execution key set changed")
    for key, value in execution.items():
        _require_nonnegative_int(value, f"execution.{key}")
    expected = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
        "new_errors": 0,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise ValueError(f"SPAI execution {key} changed")
    if execution["new_successes"] + execution["resume_skips"] != selected_images:
        raise ValueError("SPAI successful work accounting changed")
    if (
        execution["same_device_artifact_replays_before_execution"]
        != execution["resume_skips"]
        or execution["same_device_artifact_replays_final"] != selected_images
    ):
        raise ValueError("SPAI same-device artifact replay coverage changed")


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    bundle_mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    selection_visibility: Mapping[str, Any],
    coverage: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_spai_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": bundle_mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "score_spec": _score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
        "selection_visibility_census": dict(selection_visibility),
        "same_device_artifact_replays_before_execution": execution[
            "same_device_artifact_replays_before_execution"
        ],
        "same_device_artifact_replays_final": execution[
            "same_device_artifact_replays_final"
        ],
        "coverage": dict(coverage),
    }
    if set(summary) != {*required, "generated_at"}:
        raise ValueError("stored SPAI runtime summary key set changed")
    _require_exact_json(
        {key: summary[key] for key in required},
        required,
        "stored SPAI runtime summary",
    )
    _require_string(summary.get("generated_at"), "summary.generated_at")
    _reject_nonfinite_numbers(summary, "summary")
    _reject_unsupported_claims(summary, "summary")


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
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FileNotFoundError(f"missing safe SPAI run directory: {run_dir}")
    runner._validate_run_dir_inventory(
        run_dir,
        allow_missing_results=False,
    )
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
    manifest = _load_json(manifest_path, f"{mode} SPAI manifest")
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
    physical = tuple(_read_jsonl_strict(results_path, f"{mode} SPAI physical results"))
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
        raise ValueError("SPAI smoke requires exactly one attempt per selected image")
    for row in latest:
        if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
            raise ValueError("SPAI latest result coverage is not successful")

    outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    immutable_outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable.outputs",
    )
    expected_immutable_outputs = {
        "run_dir": repo_relative(run_dir, root),
        "results_path": repo_relative(results_path, root),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "artifact_root": (f"outputs/opensource/spai/{run_id}"),
        "patch_features_dir": (f"outputs/opensource/spai/{run_id}/patch_features"),
        "features_dir": f"outputs/opensource/spai/{run_id}/feature",
        "attention_dir": f"outputs/opensource/spai/{run_id}/attention",
    }
    if immutable_outputs != expected_immutable_outputs:
        raise ValueError("SPAI immutable output contract changed")
    artifact_root = _safe_repo_path(
        immutable_outputs["artifact_root"],
        repo_root=root,
        label="SPAI artifact root",
        require_file=False,
    )
    if artifact_root != (root / "outputs" / "opensource" / "spai" / run_id).resolve():
        raise ValueError("SPAI artifact root is not canonical")
    expected_outputs = {
        **expected_immutable_outputs,
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "artifact_files": len(selected) * 3,
    }
    if outputs != expected_outputs:
        raise ValueError("SPAI finalized manifest outputs changed")

    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    summary = _load_json(summary_path, f"{mode} SPAI runtime summary")
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
        execution=execution,
    )
    artifacts = validate_artifact_inventory(
        latest_results=latest,
        repo_root=root,
        artifact_root=artifact_root,
        run_id=run_id,
    )
    evidence_snapshot = _capture_evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        artifacts=artifacts,
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
        artifact_root=artifact_root,
        artifacts=artifacts,
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
    """Post-operation TOCTOU revalidate primary files and all three arrays."""

    expected = dict(bundle.evidence_snapshot)
    _assert_runner_contract_exports()._validate_run_dir_inventory(
        bundle.run_dir,
        allow_missing_results=False,
    )
    if set(expected) != {
        "manifest_sha256",
        "results_sha256",
        "expected_inputs_sha256",
        "summary_sha256",
        "artifact_inventory_sha256",
    }:
        raise ValueError("SPAI evidence snapshot key set changed")
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
            raise ValueError(f"SPAI evidence changed after validation: {key}")
    current_manifest = _load_json(
        bundle.manifest_path,
        f"{bundle.mode} SPAI manifest recheck",
    )
    fingerprint, immutable = _validate_manifest(
        manifest=current_manifest,
        repo_root=repo_root,
        run_id=bundle.run_id,
        expected_mode=bundle.mode,
    )
    if fingerprint != bundle.fingerprint:
        raise ValueError("SPAI manifest changed after validation")
    _require_exact_json(
        immutable,
        bundle.immutable,
        "SPAI immutable manifest recheck",
    )
    _require_exact_json(
        current_manifest,
        bundle.manifest,
        "SPAI manifest recheck",
    )
    _release, selected, contract = _rebuild_contract(
        repo_root=repo_root,
        immutable=bundle.immutable,
        expected_mode=bundle.mode,
    )
    _require_exact_json(
        list(selected),
        list(bundle.selected),
        "SPAI canonical selection recheck",
    )
    _require_exact_json(
        contract.as_dict(),
        bundle.contract.as_dict(),
        "SPAI canonical contract recheck",
    )
    artifacts = validate_artifact_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        artifact_root=bundle.artifact_root,
        run_id=bundle.run_id,
    )
    if _artifact_inventory_sha256(artifacts) != expected["artifact_inventory_sha256"]:
        raise ValueError("SPAI artifact evidence changed after validation")


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if (
        len(bundle.selected) != FORMAL_IMAGES
        or iterations != BOOTSTRAP_ITERATIONS
        or seed != BOOTSTRAP_SEED
    ):
        raise ValueError(
            "SPAI formal Balanced250 metrics require 1,775 images, "
            "iterations=1000, and seed=20260726"
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
        raise ValueError("formal SPAI Balanced250 metrics are incomplete")
    _reject_unsupported_claims(metrics, "shared Balanced250 T1 metrics")
    _reject_nonfinite_numbers(metrics, "shared Balanced250 T1 metrics")
    return metrics


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing = _SMOKE_IGNORED_FIELDS - set(row)
    if missing:
        raise ValueError(
            f"{sample_id} lacks ignored runtime field " f"{sorted(missing)[0]}"
        )
    result = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in _SMOKE_IGNORED_FIELDS
    }
    for prefix in (
        "spai_patch_features",
        "spai_feature",
        "spai_attention",
    ):
        nested = _require_mapping(result.get(prefix), f"{sample_id} {prefix}")
        result[prefix] = {
            key: value
            for key, value in nested.items()
            if key not in _ARTIFACT_VOLATILE_FIELDS
        }
        for suffix in ("path", "sha256", "array_sha256"):
            result.pop(f"{prefix}_{suffix}", None)
    result.pop("feature_array_sha256", None)
    result.pop("artifact_paths", None)
    return result


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_artifacts: Mapping[str, SampleArtifacts],
    replay_artifacts: Mapping[str, SampleArtifacts],
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
        raise ValueError("SPAI smoke result coverage differs")
    if set(reference_artifacts) != set(reference):
        raise ValueError("SPAI reference artifact coverage differs")
    if set(replay_artifacts) != set(replay):
        raise ValueError("SPAI replay artifact coverage differs")
    maximum = {
        "raw_logit": 0.0,
        "probability": 0.0,
        "patch_features": 0.0,
        "feature": 0.0,
        "attention": 0.0,
    }
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
                f"SPAI smoke result {sample_id} differs at " f"{differing[:3]}"
            )
        maximum["raw_logit"] = max(
            maximum["raw_logit"],
            abs(float(left["raw_logit"]) - float(right["raw_logit"])),
        )
        maximum["probability"] = max(
            maximum["probability"],
            abs(float(left["probability"]) - float(right["probability"])),
        )
        left_sample = reference_artifacts[sample_id]
        right_sample = replay_artifacts[sample_id]
        for kind, left_artifact, right_artifact in (
            (
                "patch_features",
                left_sample.patch_features,
                right_sample.patch_features,
            ),
            ("feature", left_sample.feature, right_sample.feature),
            ("attention", left_sample.attention, right_sample.attention),
        ):
            if (
                left_artifact.path.resolve() == right_artifact.path.resolve()
                or left_artifact.path.read_bytes() != right_artifact.path.read_bytes()
                or left_artifact.file_sha256 != right_artifact.file_sha256
                or left_artifact.file_bytes != right_artifact.file_bytes
                or left_artifact.array_sha256 != right_artifact.array_sha256
                or not np.array_equal(
                    left_artifact.array,
                    right_artifact.array,
                )
            ):
                raise ValueError(
                    f"SPAI smoke {kind} {sample_id} is not independently " "byte-exact"
                )
            maximum[kind] = max(
                maximum[kind],
                _maximum_absolute_difference(
                    left_artifact.array,
                    right_artifact.array,
                ),
            )
    if any(value != 0.0 for value in maximum.values()):
        raise ValueError("SPAI smoke comparison is not bit-exact")
    return {
        "images_compared": len(reference),
        "ignored_row_fields": sorted(_SMOKE_IGNORED_FIELDS),
        "ignored_artifact_metadata_fields": sorted(_ARTIFACT_VOLATILE_FIELDS),
        "exact_computational_projection": True,
        "independent_artifact_paths": True,
        "patch_feature_file_bytes_and_arrays_exact": True,
        "feature_file_bytes_and_arrays_exact": True,
        "attention_file_bytes_and_arrays_exact": True,
        "attention_role": "classifier_diagnostic_not_T2_or_localization",
        "maximum_absolute_differences": maximum,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    if (
        set(immutable) != EXPECTED_IMMUTABLE_CONFIG_KEYS
        or frozenset(runner.IMMUTABLE_CONFIG_KEYS) != EXPECTED_IMMUTABLE_CONFIG_KEYS
    ):
        raise ValueError("SPAI smoke immutable key set changed")
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def _configure_frozen_runtime(
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    if not isinstance(device_text, str) or (
        device_text != "cpu" and re.fullmatch(r"cuda:[0-9]+", device_text) is None
    ):
        raise ValueError("SPAI runtime device must be cpu or an explicit cuda:N")
    configured = _assert_runner_contract_exports().configure_runtime(
        device_text,
        seed=EXPECTED_RUNTIME_SEED,
    )
    if (
        not isinstance(configured, tuple)
        or len(configured) != 2
        or not isinstance(configured[1], Mapping)
    ):
        raise ValueError("SPAI configure_runtime return contract changed")
    device, runtime = configured[0], dict(configured[1])
    _validate_runtime_contract(runtime, label="current SPAI runtime")
    if runtime.get("device") != device_text:
        raise ValueError("SPAI configured runtime changed the device")
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_text == "cpu":
        if device_type != "cpu" or device_index is not None:
            raise ValueError("SPAI configured device differs from CPU")
    elif device_type != "cuda" or device_index != int(device_text.split(":", 1)[1]):
        raise ValueError("SPAI configured device differs from requested CUDA")
    return device, runtime


def _actual_runtime_contract(device_text: str) -> dict[str, Any]:
    _device, runtime = _configure_frozen_runtime(device_text)
    return runtime


def _analysis_runtime_contract() -> dict[str, Any]:
    inference_runtime = _actual_runtime_contract("cpu")
    actual = {
        name: importlib.metadata.version(name)
        for name in EXPECTED_ANALYSIS_PACKAGE_VERSIONS
    }
    if actual != EXPECTED_ANALYSIS_PACKAGE_VERSIONS:
        raise ValueError(
            "SPAI analysis package versions changed: "
            f"{actual} != {EXPECTED_ANALYSIS_PACKAGE_VERSIONS}"
        )
    return {
        "schema_version": "spai_balanced_analysis_runtime_v1",
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
        raise ValueError("SPAI smoke comparison requires distinct run IDs")
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
    reference_source = Path(str(reference.immutable["source"]["root"]))
    replay_source = Path(str(replay.immutable["source"]["root"]))
    reference_checkpoint = Path(
        str(reference.immutable["assets"]["checkpoint"]["path"])
    )
    replay_checkpoint = Path(str(replay.immutable["assets"]["checkpoint"]["path"]))
    resolved_results_root = _resolve_results_root(results_dir, repo_root)
    default_comparison_output = _resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=repo_root,
        results_dir=resolved_results_root,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    _validate_results_output_scope(
        {"comparison": output_path},
        results_root=resolved_results_root,
        canonical_outputs={"comparison": default_comparison_output},
    )
    _validate_output_targets(
        {"comparison": output_path},
        protected_files=(
            *_bundle_protected_files(reference, repo_root=repo_root),
            *_bundle_protected_files(replay, repo_root=repo_root),
        ),
        protected_dirs=(
            reference.run_dir,
            replay.run_dir,
            reference.artifact_root,
            replay.artifact_root,
            reference_source,
            replay_source,
            reference_checkpoint.parent,
            replay_checkpoint.parent,
            reference.release.manifest_path.parent,
            replay.release.manifest_path.parent,
        ),
    )
    if _smoke_immutable_projection(reference.immutable) != _smoke_immutable_projection(
        replay.immutable
    ):
        raise ValueError("SPAI smoke immutable computational/runtime configs differ")
    reference_runtime = _validate_runtime_contract(
        reference.immutable.get("runtime"),
        label="reference immutable.runtime",
    )
    replay_runtime = _validate_runtime_contract(
        replay.immutable.get("runtime"),
        label="replay immutable.runtime",
    )
    if reference_runtime != replay_runtime:
        raise ValueError("SPAI smoke runs have different exact runtimes")
    if reference.selected != replay.selected or len(reference.selected) != SMOKE_IMAGES:
        raise ValueError("SPAI smoke runs differ from the exact 35 inputs")
    device_text = str(reference_runtime["device"])
    reference_artifact_replay = replay_persisted_artifacts(
        latest_results=reference.latest_results,
        artifacts=reference.artifacts,
        source_root=reference_source,
        checkpoint_path=reference_checkpoint,
        device_text=device_text,
        recorded_runtime=reference_runtime,
    )
    replay_artifact_replay = replay_persisted_artifacts(
        latest_results=replay.latest_results,
        artifacts=replay.artifacts,
        source_root=replay_source,
        checkpoint_path=replay_checkpoint,
        device_text=device_text,
        recorded_runtime=replay_runtime,
    )
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
    )
    _verify_bundle_unchanged(reference, repo_root=repo_root)
    _verify_bundle_unchanged(replay, repo_root=repo_root)
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_spai_smoke_comparison_passed",
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
        "reference_persisted_patch_feature_replay": (reference_artifact_replay),
        "replay_persisted_patch_feature_replay": replay_artifact_replay,
        "comparison": comparison,
        "immutable_computational_runtime_config_exact": True,
        "evidence_reverified_after_comparison": True,
        "method_boundary": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "attention_is_diagnostic_not_t2": True,
        },
    }
    if output_path is not None:
        _write_json_verified(
            output_path,
            report,
            label="SPAI smoke comparison output",
        )
    return report


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
    resolved_outputs = {
        name: _absolute_output_path(
            Path(path),
            label=f"SPAI {name} output",
        ).resolve()
        for name, path in outputs.items()
        if path is not None
    }
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise ValueError("SPAI analysis output paths collide")
    files = {Path(path).resolve() for path in protected_files}
    directories = tuple(Path(path).resolve() for path in protected_dirs)
    for name, output in resolved_outputs.items():
        if output in files or any(
            output == directory or directory in output.parents
            for directory in directories
        ):
            raise ValueError(f"SPAI analysis output {name} would overwrite evidence")


def _absolute_output_path(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    return absolute


def _validate_results_output_scope(
    outputs: Mapping[str, Path | None],
    *,
    results_root: Path,
    canonical_outputs: Mapping[str, Path],
) -> None:
    root = results_root.resolve()
    reports_root = root / "_reports"
    canonical = {name: Path(path).resolve() for name, path in canonical_outputs.items()}
    for name, value in outputs.items():
        if value is None:
            continue
        output = _absolute_output_path(
            Path(value),
            label=f"SPAI {name} output",
        ).resolve()
        try:
            output.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"SPAI {name} output must stay under the results root"
            ) from error
        if output == canonical.get(name):
            continue
        if reports_root in output.parents:
            continue
        raise ValueError(f"SPAI {name} output may not target another run's evidence")


def _validate_formal_output_locations(
    *,
    bundle: RunBundle,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
) -> None:
    _validate_results_output_scope(
        {
            "metrics": metrics_output_path,
            "audit": audit_output_path,
        },
        results_root=bundle.run_dir.parent,
        canonical_outputs={
            "metrics": (bundle.run_dir / "balanced250_metrics.json").resolve(),
            "audit": (bundle.run_dir / "independent_audit.json").resolve(),
        },
    )


def _bundle_protected_files(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> tuple[Path, ...]:
    files = [
        bundle.manifest_path,
        bundle.results_path,
        bundle.expected_path,
        bundle.summary_path,
        *(repo_root / relative for relative in EXPECTED_ADAPTER_SOURCE_PATHS),
    ]
    source = _require_mapping(
        bundle.immutable.get("source"),
        "immutable.source",
    )
    source_files = _require_mapping(
        source.get("source_files"),
        "immutable.source.source_files",
    )
    for record in source_files.values():
        mapping = _require_mapping(record, "SPAI source file")
        files.append(Path(_require_string(mapping.get("path"), "source file path")))
    assets = _require_mapping(
        bundle.immutable.get("assets"),
        "immutable.assets",
    )
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    files.append(Path(_require_string(checkpoint.get("path"), "checkpoint path")))
    for record in _require_list(
        assets.get("golden_assets"),
        "immutable.assets.golden_assets",
    ):
        mapping = _require_mapping(record, "SPAI golden asset")
        files.append(Path(_require_string(mapping.get("path"), "golden path")))
    return tuple(files)


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    checkpoint_path: Path,
    golden_root: Path,
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
    source = _require_mapping(
        bundle.immutable.get("source"),
        "immutable.source",
    )
    assets = _require_mapping(
        bundle.immutable.get("assets"),
        "immutable.assets",
    )
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    if source_root.resolve() != Path(str(source["root"])).resolve():
        raise ValueError("SPAI analysis source root differs from manifest")
    if checkpoint_path.resolve() != Path(str(checkpoint["path"])).resolve():
        raise ValueError("SPAI analysis checkpoint differs from manifest")
    if golden_root.resolve() != DEFAULT_GOLDEN_ROOT.resolve():
        raise ValueError("SPAI analysis golden root differs from frozen root")
    for record, frozen in zip(
        _require_list(
            assets.get("golden_assets"),
            "immutable.assets.golden_assets",
        ),
        legacy.GOLDEN_CASES,
        strict=True,
    ):
        expected = (golden_root / str(frozen["relative_path"])).resolve()
        if Path(str(record["path"])).resolve() != expected:
            raise ValueError("SPAI analysis golden asset root changed")
    recorded_runtime = _validate_runtime_contract(
        bundle.immutable.get("runtime"),
        label="immutable.runtime",
    )
    if device_text != recorded_runtime.get("device"):
        raise ValueError("SPAI analysis device differs from manifest")
    _validate_formal_output_locations(
        bundle=bundle,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    _validate_output_targets(
        {
            "metrics": metrics_output_path,
            "audit": audit_output_path,
        },
        protected_files=_bundle_protected_files(
            bundle,
            repo_root=repo_root,
        ),
        protected_dirs=(
            bundle.artifact_root,
            source_root,
            checkpoint_path.parent,
            golden_root,
            bundle.release.manifest_path.parent,
        ),
    )
    analysis_runtime = _analysis_runtime_contract()
    persisted_replay = replay_persisted_artifacts(
        latest_results=bundle.latest_results,
        artifacts=bundle.artifacts,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        device_text=device_text,
        recorded_runtime=recorded_runtime,
    )
    metrics = recompute_metrics(
        bundle,
        iterations=iterations,
        seed=seed,
    )
    metrics_sha256 = _json_artifact_sha256(metrics)
    replay_report = (
        replay_model(
            bundle,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device_text=device_text,
        )
        if replay
        else None
    )
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "fresh_full_replay_audit_passed"
            if replay
            else "persisted_artifact_audit_passed"
        ),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": len(bundle.selected),
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": bundle.coverage,
        "artifact_files": len(bundle.artifacts) * 3,
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "analysis_runtime": analysis_runtime,
        "persisted_patch_sca_norm_and_feature_mlp_replay": (persisted_replay),
        "fresh_model_replay": replay_report,
        "method_boundary": {
            "method": legacy.MODEL_NAME,
            "architecture": legacy.MODEL_ARCH,
            "preprocess_profile": PREPROCESS_PROFILE,
            "released_checkpoint_evaluated": True,
            "released_executable": (
                "FFT_frequency_restoration_then_ViT_B16_SRS_SCA_"
                "LayerNorm_complete_MLP"
            ),
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "native_dense_output": False,
            "attention_is_diagnostic_not_t2": True,
            "attention_semantics": ATTENTION_SEMANTICS,
            "license": _runner().MODEL_CONTRACT["license"],
            "commercial_clearance": "unresolved_high_risk",
        },
        "contract_checks": {
            "exact_formal_whole_image_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "three_artifacts_per_image_validated": True,
            "canonical_npy_bytes_shape_dtype_hashes_finite": True,
            "patch_sca_norm_complete_mlp_replay_validated": True,
            "feature_complete_mlp_replay_validated": True,
            "attention_treated_as_classifier_diagnostic_only": True,
            "pair_rank_rejected": True,
            "unsupported_localization_claims_rejected": True,
            "source_checkpoint_runtime_adapter_hashes_validated": True,
            "cpu_gate_preceded_dataset_and_accelerator": True,
            "shared_balanced250_t1_metrics_only": True,
            "run_and_canonical_evidence_reverified_after_replay": True,
        },
        "artifacts": {
            **dict(bundle.evidence_snapshot),
            "metrics_sha256": metrics_sha256,
        },
    }
    _reject_nonfinite_numbers(audit, "SPAI independent audit")
    _reject_unsupported_claims(audit, "SPAI independent audit")
    if metrics_output_path is not None:
        _write_json_verified(
            metrics_output_path,
            metrics,
            label="SPAI Balanced250 metrics output",
        )
    if audit_output_path is not None:
        _write_json_verified(
            audit_output_path,
            audit,
            label="SPAI independent audit output",
        )
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    if metrics_output_path is not None:
        _verify_json_artifact(
            metrics_output_path,
            expected_sha256=metrics_sha256,
            label="SPAI Balanced250 metrics output",
        )
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    raw = path if path.is_absolute() else repo_root / path
    return Path(os.path.abspath(raw))


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
    fingerprint = hashlib.sha256(
        stable_json([reference_run_id, replay_run_id]).encode()
    ).hexdigest()
    return (
        results_dir
        / "_reports"
        / f"{SMOKE_COMPARISON_SCHEMA_VERSION}_{fingerprint}.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    runner = _assert_runner_contract_exports()
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
        "--checkpoint",
        type=Path,
        default=getattr(runner, "DEFAULT_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=getattr(runner, "DEFAULT_GOLDEN_ROOT", DEFAULT_GOLDEN_ROOT),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--compare-smoke-run-id")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )
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
            raise ValueError("SPAI smoke comparison cannot combine with formal options")
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
        checkpoint_path=_anchored(args.checkpoint, repo_root),
        golden_root=_anchored(args.golden_root, repo_root),
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
