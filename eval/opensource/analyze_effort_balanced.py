#!/usr/bin/env python3
"""Fail-closed audit and independent replay for Effort on Balanced250.

The runner manifest, append-only JSONL, and local NPZ files are treated as
untrusted evidence.  This analyzer rebuilds the exact canonical selection,
validates every physical attempt and every artifact, independently reconstructs
the pinned Effort graph, replays the two-class head and float32 class-1
softmax, recomputes the shared Balanced250 T1 metrics, and by default freshly
replays all 1,775 canonical JPEGs through the complete model.

Effort is T1-only.  Whole-canvas edit visibility is an input diagnostic, not a
predicted mask, and no T2 or joint metric is emitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import analyze_effort_run as legacy_audit
from eval.opensource import run_effort as legacy
from eval.opensource import run_effort_balanced as runner
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RunDatasetContract,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    require_complete_coverage,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
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


RUN_MANIFEST_SCHEMA = "effort_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "effort_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "effort_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "effort_balanced_cpu_preflight_v1"
AUDIT_SCHEMA_VERSION = "effort_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "effort_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"

DEFAULT_RESULTS_DIR = Path("results/opensource/effort")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/effort")
DEFAULT_RUN_ID = "effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727"
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_HF_CONFIG = legacy.DEFAULT_HF_CONFIG

FORMAL_IMAGES = 1_775
SMOKE_IMAGES = 35
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20260726
FEATURE_SHAPE = (1_024,)
LOGIT_SHAPE = (2,)
NPZ_FILE_BYTES = 4_640

# Fresh replay is required on the recorded device/runtime.  The frozen Mouse
# run established bit-exact repeatability there, so drift is not hidden behind
# a cross-device tolerance.
FEATURE_ABS_TOLERANCE = 0.0
LOGIT_ABS_TOLERANCE = 0.0
HEAD_ABS_TOLERANCE = 0.0
PROBABILITY_ABS_TOLERANCE = 0.0
MARGIN_ABS_TOLERANCE = 0.0
# This is only a pre-device sanity check. PyTorch's float32 CPU and CUDA
# softmax kernels can differ by a couple of ulps for the same persisted
# logits. The recorded-device head replay below remains bit-exact
# (tolerance 0).
STATIC_CPU_SOFTMAX_ABS_TOLERANCE = float(2 * np.finfo(np.float32).eps)

EXPECTED_SOURCE_COMMIT = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
EXPECTED_CHECKPOINT_SHA256 = (
    "7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813"
)
EXPECTED_CHECKPOINT_SCHEMA_SHA256 = (
    "bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489"
)
EXPECTED_HF_CONFIG_SHA256 = (
    "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"
)

_OK_EXTENSION_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
        "artifact_keys",
        "artifact_paths",
        "feature_shape",
        "feature_dtype",
        "feature_semantics",
        "feature_array_sha256",
        "class_logits_shape",
        "class_logits_dtype",
        "class_logits_array_sha256",
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
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
_ERROR_EXTENSION_FIELDS = frozenset({"error_type", "error", "traceback"})
_SMOKE_IGNORED_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "preprocess_latency_ms",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "artifact_path",
        "artifact_sha256",
        "artifact_paths",
    }
)

_FALSE_T2_DECLARATIONS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULL_T2_DECLARATIONS = frozenset({"localization_output", "localisation_output"})
_FORBIDDEN_T2_KEYS = frozenset(
    {
        "pair_rank",
        "t2",
        "joint",
        "joint_score",
        "joint_metrics",
        "localization",
        "localisation",
        "dense_output",
        "attention_map",
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
        "s_joint",
    }
)
_FORBIDDEN_T2_PREFIXES = (
    "t2_",
    "joint_",
    "pixel_",
    "localization_",
    "localisation_",
    "dense_",
    "attention_map",
    "heatmap_",
    "mask_",
    "score_map",
    "predicted_mask",
)
_ALLOWED_VISIBILITY_KEYS = frozenset(
    {
        "gt_mask_kind",
        "edit_visibility",
        "edit_visible_gt_fraction",
    }
)


@dataclass(frozen=True)
class EffortArtifact:
    """One fully validated local Effort NPZ artifact."""

    sample_id: str
    path: Path
    relative_path: str
    file_sha256: str
    file_bytes: int
    feature_array_sha256: str
    logits_array_sha256: str
    feature: np.ndarray
    logits: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    """All independently validated evidence for one formal or smoke run."""

    run_id: str
    mode: str
    fingerprint: str
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
    artifacts: Mapping[str, EffortArtifact]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
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


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return int(value)


def _same_json(left: Any, right: Any) -> bool:
    return type(left) is type(right) and stable_json(left) == stable_json(right)


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return _require_mapping(value, label)


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"{label}:{line_number} lacks final newline")
            if not line.strip():
                raise ValueError(f"{label}:{line_number} is blank")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as error:
                raise ValueError(f"{label}:{line_number} is invalid JSON") from error
            row = _require_mapping(value, f"{label}:{line_number}")
            if line != f"{stable_json(row)}\n":
                raise ValueError(f"{label}:{line_number} is not canonical JSONL")
            rows.append(row)
    return rows


def _reject_nonfinite(value: Any, label: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite")


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    """Reject pair, T2, dense, and joint claims at every nesting depth."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_VISIBILITY_KEYS:
                _reject_unsupported_claims(child, child_label)
                continue
            if normalized in _FALSE_T2_DECLARATIONS:
                if child is not False:
                    raise ValueError(f"{child_label} is an unsupported Effort claim")
                continue
            if normalized in _NULL_T2_DECLARATIONS:
                if child is not None:
                    raise ValueError(f"{child_label} is an unsupported Effort claim")
                continue
            if normalized in _FORBIDDEN_T2_KEYS or normalized.startswith(
                _FORBIDDEN_T2_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported Effort claim")
            _reject_unsupported_claims(child, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _reject_unsupported_claims(child, f"{label}[{index}]")


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
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    raw = results_dir if results_dir.is_absolute() else repo_root / results_dir
    return Path(os.path.abspath(raw))


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    valid = runner._valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / valid).resolve()
    if candidate.parent != root:
        raise ValueError("resolved Effort run directory escapes results root")
    return candidate


def _expected_artifact_dir(repo_root: Path, run_id: str) -> Path:
    return (
        repo_root.resolve()
        / DEFAULT_ARTIFACTS_DIR
        / runner._valid_run_id(run_id)
        / "artifacts"
    ).resolve()


def _visibility(row: Mapping[str, Any]) -> dict[str, Any]:
    """Independently derive whole-canvas geometric edit visibility."""

    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        return {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "basis": "full_canvas_direct_resize_without_crop",
                "preprocess_profile": legacy.PREPROCESS_PROFILE,
            },
        }
    expected = "all_zero" if row.get("condition") == "real" else "not_applicable"
    if gt_kind != expected:
        raise ValueError("Balanced250 non-local GT semantics changed")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": (
                "authentic_input"
                if gt_kind == "all_zero"
                else "fullframe_has_no_local_GT"
            ),
            "preprocess_profile": legacy.PREPROCESS_PROFILE,
        },
    }


def visibility_census(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    local_conditions = ("local_mouse", "local_cat", "local_trash_can")
    by_condition: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for condition in local_conditions:
        selected = [row for row in rows if row.get("condition") == condition]
        counts: Counter[str] = Counter()
        fractions: list[float] = []
        for row in selected:
            evidence = _visibility(row)
            category = str(evidence["edit_visibility"])
            fraction = float(evidence["edit_visible_gt_fraction"])
            counts[category] += 1
            totals[category] += 1
            fractions.append(fraction)
        by_condition[condition] = {
            "full": counts["full"],
            "partial": counts["partial"],
            "none": counts["none"],
            "total": len(selected),
            "mean_edit_visible_gt_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
        }
    result = {
        "basis": "full_canvas_direct_resize_without_crop",
        "role": "input_condition_stratum_not_model_localization",
        "by_condition": by_condition,
        "all_local": {
            "full": totals["full"],
            "partial": totals["partial"],
            "none": totals["none"],
            "total": sum(
                int(by_condition[value]["total"]) for value in local_conditions
            ),
            "mean_edit_visible_gt_fraction": (1.0 if totals["full"] else None),
        },
        "not_applicable_images": sum(
            row.get("gt_mask_kind") != "exact_diff" for row in rows
        ),
    }
    counts = Counter(str(row["condition"]) for row in rows)
    if dict(counts) == runner.FORMAL_COUNTS:
        if result != {
            "basis": "full_canvas_direct_resize_without_crop",
            "role": "input_condition_stratum_not_model_localization",
            "by_condition": {
                condition: {
                    "full": 250,
                    "partial": 0,
                    "none": 0,
                    "total": 250,
                    "mean_edit_visible_gt_fraction": 1.0,
                }
                for condition in local_conditions
            },
            "all_local": {
                "full": 750,
                "partial": 0,
                "none": 0,
                "total": 750,
                "mean_edit_visible_gt_fraction": 1.0,
            },
            "not_applicable_images": 1_025,
        }:
            raise ValueError("formal Effort visibility census changed")
    return result


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> None:
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected = set(runner.ADAPTER_SOURCE_PATHS)
    if set(sources) != expected:
        raise ValueError("immutable.adapter_sources key set changed")
    for relative, raw in sources.items():
        record = _require_mapping(raw, f"adapter source {relative}")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} schema changed")
        if record.get("path") != relative:
            raise ValueError(f"adapter source {relative} path changed")
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        if record.get("bytes") != path.stat().st_size or record.get(
            "sha256"
        ) != sha256_file(path):
            raise ValueError(f"adapter source {relative} content changed")


def _verify_source_assets_independently(
    *,
    source_root: Path,
    checkpoint_path: Path,
    hf_config_path: Path,
    recorded_source: Mapping[str, Any],
    recorded_assets: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    recorded_source_root = Path(
        _require_string(recorded_source.get("path"), "recorded source path")
    )
    recorded_checkpoint = _require_mapping(
        recorded_assets.get("checkpoint"),
        "recorded checkpoint",
    )
    recorded_config = _require_mapping(
        recorded_assets.get("clip_config"),
        "recorded CLIP config",
    )
    if (
        source_root.resolve() != recorded_source_root.resolve()
        or checkpoint_path.resolve()
        != Path(
            _require_string(
                recorded_checkpoint.get("path"),
                "recorded checkpoint path",
            )
        ).resolve()
        or hf_config_path.resolve()
        != Path(
            _require_string(
                recorded_config.get("path"),
                "recorded config path",
            )
        ).resolve()
    ):
        raise ValueError("replay asset paths differ from the recorded run")
    source = legacy_audit._verify_source(source_root)
    state, config, assets = legacy_audit._load_assets(
        checkpoint_path,
        hf_config_path,
    )
    if (
        source.get("commit") != EXPECTED_SOURCE_COMMIT
        or recorded_source.get("commit") != EXPECTED_SOURCE_COMMIT
        or recorded_source.get("tracked_dirty") is not False
    ):
        raise ValueError("Effort source provenance changed")
    recorded_files = _require_mapping(
        recorded_source.get("files"),
        "recorded source files",
    )
    if set(recorded_files) != set(legacy.SOURCE_FILES):
        raise ValueError("recorded source file set changed")
    for relative, digest in legacy.SOURCE_FILES.items():
        independent = _require_mapping(
            source["files"].get(relative),
            f"independent source {relative}",
        )
        recorded = _require_mapping(
            recorded_files.get(relative),
            f"recorded source {relative}",
        )
        if (
            independent.get("sha256") != digest
            or recorded.get("sha256") != digest
            or independent.get("bytes") != recorded.get("bytes")
        ):
            raise ValueError(f"Effort source file changed: {relative}")
    checkpoint = recorded_checkpoint
    clip_config = recorded_config
    if (
        checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("bytes") != legacy.CHECKPOINT["bytes"]
        or checkpoint.get("tensor_count") != legacy.CHECKPOINT["tensor_count"]
        or checkpoint.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or checkpoint.get("schema_sha256") != EXPECTED_CHECKPOINT_SCHEMA_SHA256
        or clip_config.get("sha256") != EXPECTED_HF_CONFIG_SHA256
        or clip_config.get("bytes") != legacy.HF_CONFIG["bytes"]
        or assets.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or assets.get("checkpoint_schema_sha256") != EXPECTED_CHECKPOINT_SCHEMA_SHA256
        or assets.get("config_sha256") != EXPECTED_HF_CONFIG_SHA256
    ):
        raise ValueError("Effort checkpoint/config provenance changed")
    return source, assets, state, config


def _validate_runtime(value: Any, *, label: str) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    expected_versions = {
        "torch": legacy.TORCH_VERSION,
        "torchvision": legacy.TORCHVISION_VERSION,
        "transformers": legacy.TRANSFORMERS_VERSION,
        "numpy": legacy.NUMPY_VERSION,
        "opencv": legacy.OPENCV_VERSION,
    }
    if any(runtime.get(key) != expected for key, expected in expected_versions.items()):
        raise ValueError(f"{label} package versions changed")
    if runtime.get("python") != "3.12.3":
        raise ValueError(f"{label} Python version changed")
    device = _require_string(runtime.get("device"), f"{label}.device")
    if device != "cpu" and not (
        device.startswith("cuda:")
        and device[5:].isdigit()
        and str(int(device[5:])) == device[5:]
    ):
        raise ValueError(f"{label} device is not explicit")
    expected = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "autocast": False,
        "model_dtype": "float32",
        "batch_size": 1,
        "cpu_threads": 16,
        "seed": 20260724,
    }
    if any(
        runtime.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(f"{label} numerical contract changed")
    if device == "cpu":
        if runtime.get("cuda_device_name") is not None:
            raise ValueError(f"{label} CPU runtime names a CUDA device")
    else:
        _require_string(runtime.get("cuda_device_name"), f"{label}.cuda name")
    _reject_nonfinite(runtime, label)
    return runtime


def _validate_cpu_preflight(
    value: Any,
    *,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    model_audit: Mapping[str, Any],
) -> None:
    wrapper = _require_mapping(value, "immutable.cpu_preflight")
    if set(wrapper) != {
        "performed_before_accelerator_configuration",
        "report",
    }:
        raise ValueError("CPU preflight wrapper changed")
    if wrapper.get("performed_before_accelerator_configuration") is not True:
        raise ValueError("CPU preflight ordering evidence changed")
    report = _require_mapping(wrapper.get("report"), "CPU preflight report")
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "assets",
        "model_audit",
        "runtime",
        "runtime_golden",
        "accelerator_model_forwards",
        "balanced250_model_scores_computed",
        "cuda_initialized_before",
        "cuda_initialized_after",
    }
    if set(report) != expected_keys:
        raise ValueError("CPU preflight report key set changed")
    if (
        report.get("schema_version") != CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("model_audit") != model_audit
        or report.get("accelerator_model_forwards") != 0
        or report.get("balanced250_model_scores_computed") != 0
        or report.get("cuda_initialized_before") is not False
        or report.get("cuda_initialized_after") is not False
    ):
        raise ValueError("CPU preflight evidence changed")
    runtime = _validate_runtime(
        report.get("runtime"),
        label="CPU preflight runtime",
    )
    if runtime.get("device") != "cpu":
        raise ValueError("CPU preflight used a non-CPU device")
    golden = _require_mapping(
        report.get("runtime_golden"),
        "CPU preflight runtime golden",
    )
    if (
        golden.get("status") != "passed"
        or golden.get("kind")
        != "repository_fixture_runtime_regression_not_author_published_golden"
        or golden.get("device_family") != "cpu"
        or golden.get("mouse_model_scores_computed") != 0
        or len(golden.get("cases", [])) != 2
    ):
        raise ValueError("CPU preflight golden changed")


def _validate_model_audit(value: Any) -> dict[str, Any]:
    audit = _require_mapping(value, "immutable.model_audit")
    expected_keys = {
        "constructor",
        "official_forward_formula",
        "strict_load",
        "missing_keys",
        "unexpected_keys",
        "state_entries",
        "state_elements",
        "svd_modules",
        "svd_module_names",
        "frozen_rank",
        "residual_rank",
        "feature_dimension",
        "head_weight_shape",
        "head_bias_shape",
        "nonpersistent_position_ids_materialized",
        "position_ids_shape",
        "parameter_count",
        "eval_mode",
    }
    if set(audit) != expected_keys:
        raise ValueError("immutable.model_audit key set changed")
    expected_names = [
        f"backbone.encoder.layers.{layer}.self_attn.{projection}"
        for layer in range(24)
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj")
    ]
    expected = {
        "constructor": "shape_only_exact_checkpoint_forward_graph",
        "official_forward_formula": (
            "weight_main + U_residual @ diag(S_residual) @ V_residual"
        ),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "state_entries": legacy.CHECKPOINT["tensor_count"],
        "state_elements": legacy.CHECKPOINT["state_elements"],
        "svd_modules": legacy.SVD_MODULE_COUNT,
        "svd_module_names": expected_names,
        "frozen_rank": legacy.SVD_FROZEN_RANK,
        "residual_rank": legacy.SVD_RESIDUAL_RANK,
        "feature_dimension": legacy.FEATURE_DIMENSION,
        "head_weight_shape": [legacy.CLASS_COUNT, legacy.FEATURE_DIMENSION],
        "head_bias_shape": [legacy.CLASS_COUNT],
        "nonpersistent_position_ids_materialized": True,
        "position_ids_shape": [1, 257],
        "parameter_count": legacy.CHECKPOINT["state_elements"],
        "eval_mode": True,
    }
    if audit != expected:
        raise ValueError("immutable.model_audit changed")
    return audit


def _validate_runtime_golden(
    value: Any,
    *,
    label: str,
    expected_device_family: str,
) -> dict[str, Any]:
    golden = _require_mapping(value, label)
    expected_keys = {
        "status",
        "kind",
        "device_family",
        "runtime_abs_tolerance",
        "cpu_cuda_abs_tolerance",
        "cases",
        "mouse_model_scores_computed",
    }
    if set(golden) != expected_keys:
        raise ValueError(f"{label} key set changed")
    if (
        golden.get("status") != "passed"
        or golden.get("kind")
        != "repository_fixture_runtime_regression_not_author_published_golden"
        or golden.get("device_family") != expected_device_family
        or golden.get("runtime_abs_tolerance") != legacy.GOLDEN_RUNTIME_ABS_TOLERANCE
        or golden.get("cpu_cuda_abs_tolerance") != legacy.GOLDEN_CPU_CUDA_ABS_TOLERANCE
        or golden.get("mouse_model_scores_computed") != 0
    ):
        raise ValueError(f"{label} contract changed")
    cases = golden.get("cases")
    if not isinstance(cases, list) or len(cases) != len(legacy.GOLDEN_CASES):
        raise ValueError(f"{label} case coverage changed")
    case_keys = {
        "path",
        "input_sha256",
        "preprocess",
        "logits",
        "fake_probability",
        "feature_sha256",
        "repeat_feature_max_abs_diff",
        "repeat_logit_max_abs_diff",
        "frozen_runtime_logit_max_abs_diff",
        "frozen_runtime_probability_abs_diff",
        "frozen_cpu_cuda_logit_max_abs_diff",
    }
    for index, (case_value, frozen) in enumerate(
        zip(cases, legacy.GOLDEN_CASES, strict=True)
    ):
        case = _require_mapping(case_value, f"{label}.cases[{index}]")
        if set(case) != case_keys:
            raise ValueError(f"{label} case key set changed")
        if (
            case.get("path") != frozen["path"]
            or case.get("input_sha256") != frozen["sha256"]
            or case.get("repeat_feature_max_abs_diff") != 0.0
            or case.get("repeat_logit_max_abs_diff") != 0.0
        ):
            raise ValueError(f"{label} case identity changed")
        logits = case.get("logits")
        if (
            not isinstance(logits, list)
            or len(logits) != legacy.CLASS_COUNT
            or any(
                not math.isfinite(_require_finite(item, "golden logit"))
                for item in logits
            )
        ):
            raise ValueError(f"{label} case logits changed")
        score = _require_finite(
            case.get("fake_probability"),
            f"{label} case probability",
        )
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{label} probability is outside [0,1]")
        _require_sha256(
            case.get("feature_sha256"),
            f"{label} feature SHA-256",
        )
        _require_mapping(case.get("preprocess"), f"{label} preprocess")
        for field in (
            "frozen_runtime_logit_max_abs_diff",
            "frozen_runtime_probability_abs_diff",
            "frozen_cpu_cuda_logit_max_abs_diff",
        ):
            if _require_finite(case.get(field), f"{label}.{field}") < 0.0:
                raise ValueError(f"{label}.{field} is negative")
    _reject_nonfinite(golden, label)
    return golden


def _validate_recorded_source(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "immutable.source")
    expected_keys = {
        "repository",
        "path",
        "commit",
        "tracked_dirty",
        "tracked_license_files",
        "files",
    }
    if set(source) != expected_keys:
        raise ValueError("immutable.source key set changed")
    root = Path(_require_string(source.get("path"), "immutable.source.path"))
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != Path(os.path.abspath(root))
        or source.get("repository") != legacy.MODEL_REPO_URL
        or source.get("commit") != EXPECTED_SOURCE_COMMIT
        or source.get("tracked_dirty") is not False
        or source.get("tracked_license_files") != []
    ):
        raise ValueError("immutable.source provenance changed")
    files = _require_mapping(source.get("files"), "immutable.source.files")
    if set(files) != set(legacy.SOURCE_FILES):
        raise ValueError("immutable.source file set changed")
    for relative, expected_sha in legacy.SOURCE_FILES.items():
        record = _require_mapping(files[relative], f"source file {relative}")
        if set(record) != {"bytes", "sha256"}:
            raise ValueError(f"source file {relative} schema changed")
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != expected_sha
        ):
            raise ValueError(f"source file {relative} provenance changed")
    return source


def _validate_recorded_assets(value: Any) -> dict[str, Any]:
    assets = _require_mapping(value, "immutable.assets")
    if set(assets) != {"checkpoint", "clip_config"}:
        raise ValueError("immutable.assets key set changed")
    checkpoint = _require_mapping(
        assets.get("checkpoint"),
        "immutable.assets.checkpoint",
    )
    checkpoint_keys = {
        *legacy.CHECKPOINT,
        "path",
        "serialization_safety",
        "top_level_type",
        "schema_verified",
    }
    if set(checkpoint) != checkpoint_keys:
        raise ValueError("checkpoint evidence key set changed")
    for key, expected in legacy.CHECKPOINT.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint evidence {key} changed")
    checkpoint_path = Path(_require_string(checkpoint.get("path"), "checkpoint path"))
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != legacy.CHECKPOINT["bytes"]
        or checkpoint.get("serialization_safety")
        != {"weights_only": True, "unsafe_globals": []}
        or checkpoint.get("top_level_type") != "collections.OrderedDict"
        or checkpoint.get("schema_verified") is not True
    ):
        raise ValueError("checkpoint evidence changed")
    clip_config = _require_mapping(
        assets.get("clip_config"),
        "immutable.assets.clip_config",
    )
    config_keys = {*legacy.HF_CONFIG, "path", "vision_config_contract"}
    if set(clip_config) != config_keys:
        raise ValueError("CLIP config evidence key set changed")
    for key, expected in legacy.HF_CONFIG.items():
        if clip_config.get(key) != expected:
            raise ValueError(f"CLIP config evidence {key} changed")
    config_path = Path(_require_string(clip_config.get("path"), "CLIP config path"))
    if (
        not config_path.is_absolute()
        or config_path.is_symlink()
        or not config_path.is_file()
        or config_path.stat().st_size != legacy.HF_CONFIG["bytes"]
    ):
        raise ValueError("CLIP config file evidence changed")
    vision = _require_mapping(
        clip_config.get("vision_config_contract"),
        "CLIP vision config contract",
    )
    expected_vision = {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "projection_dim": 768,
    }
    if vision != expected_vision:
        raise ValueError("CLIP vision config contract changed")
    return assets


def _validate_immutable_outputs(
    value: Any,
    *,
    repo_root: Path,
    run_id: str,
    run_dir: Path,
) -> tuple[dict[str, Any], Path]:
    outputs = _require_mapping(value, "immutable.outputs")
    expected_keys = {
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "artifact_dir",
    }
    if set(outputs) != expected_keys:
        raise ValueError("immutable.outputs key set changed")
    expected = {
        "results_path": run_dir / "results.jsonl",
        "expected_inputs_path": run_dir / "expected_inputs.jsonl",
        "summary_path": run_dir / "summary.json",
        "artifact_dir": _expected_artifact_dir(repo_root, run_id),
    }
    resolved: dict[str, Path] = {}
    for key, expected_path in expected.items():
        path = _safe_repo_path(
            outputs.get(key),
            repo_root=repo_root,
            label=f"immutable.outputs.{key}",
            require_file=False,
        )
        if path != expected_path.resolve():
            raise ValueError(f"immutable.outputs.{key} is not canonical")
        resolved[key] = path
    return outputs, resolved["artifact_dir"]


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    run_dir: Path,
    expected_mode: str,
) -> tuple[str, dict[str, Any], Path]:
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
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != runner._valid_run_id(run_id)
        or manifest.get("status") != "complete"
    ):
        raise ValueError("analyzer requires the exact complete Effort run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest.fingerprint",
    )
    immutable = _require_mapping(manifest.get("immutable"), "immutable")
    immutable_keys = {
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
        "source",
        "assets",
        "model_audit",
        "runtime_golden",
        "runtime",
        "cpu_preflight",
        "license",
        "artifact_contract",
        "outputs",
    }
    if set(immutable) != immutable_keys:
        raise ValueError("immutable run-config key set changed")
    if (
        immutable.get("schema_version") != RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
        or fingerprint != _fingerprint(immutable)
    ):
        raise ValueError("immutable run identity/fingerprint changed")
    _verify_adapter_sources(immutable.get("adapter_sources"), repo_root=repo_root)
    expected_model = {
        "name": legacy.MODEL_NAME,
        "slug": legacy.MODEL_SLUG,
        "architecture": legacy.MODEL_ARCH,
        "repository": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "checkpoint_id": legacy.CHECKPOINT["id"],
        "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
        "checkpoint_bytes": legacy.CHECKPOINT["bytes"],
    }
    if immutable.get("model") != expected_model:
        raise ValueError("immutable model contract changed")
    if immutable.get("preprocess") != {
        "profile": legacy.PREPROCESS_PROFILE,
        "contract": runner.PREPROCESS_CONTRACT,
        "batch_size": 1,
        "autocast": False,
    }:
        raise ValueError("immutable preprocess contract changed")
    if immutable.get("score_spec") != runner.SCORE_SPEC.as_dict():
        raise ValueError("immutable score spec changed")
    if immutable.get("task_scope") != {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
    }:
        raise ValueError("immutable task scope changed")
    if immutable.get("license") != legacy.LICENSE_RECORD:
        raise ValueError("immutable license evidence changed")
    if immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT:
        raise ValueError("immutable artifact contract changed")
    source = _validate_recorded_source(immutable.get("source"))
    assets = _validate_recorded_assets(immutable.get("assets"))
    model_audit = _validate_model_audit(immutable.get("model_audit"))
    runtime = _validate_runtime(immutable.get("runtime"), label="runtime")
    device_family = "cuda" if str(runtime["device"]).startswith("cuda:") else "cpu"
    _validate_runtime_golden(
        immutable.get("runtime_golden"),
        label="runtime golden",
        expected_device_family=device_family,
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        source=source,
        assets=assets,
        model_audit=model_audit,
    )
    _validate_runtime_golden(
        immutable["cpu_preflight"]["report"]["runtime_golden"],
        label="CPU preflight runtime golden",
        expected_device_family="cpu",
    )
    _, artifact_dir = _validate_immutable_outputs(
        immutable.get("outputs"),
        repo_root=repo_root,
        run_id=run_id,
        run_dir=run_dir,
    )
    _reject_nonfinite(manifest, "manifest")
    _reject_unsupported_claims(manifest, "manifest")
    return fingerprint, immutable, artifact_dir


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    raw = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_record = _require_mapping(
        raw.get("release"),
        "dataset contract release",
    )
    manifest_path = _safe_repo_path(
        release_record.get("manifest_path"),
        repo_root=repo_root,
        label="canonical release manifest",
    )
    release = load_canonical_release(repo_root, manifest_path, verify_files=True)
    if expected_mode == "formal":
        limit = None
    elif expected_mode == "smoke":
        selection = _require_mapping(
            raw.get("selection"),
            "dataset contract selection",
        )
        spec_record = _require_mapping(
            selection.get("spec"),
            "dataset contract selection spec",
        )
        limit = spec_record.get("per_condition_limit")
        if limit != runner.DEFAULT_SMOKE_LIMIT:
            raise ValueError("Effort smoke selection is not frozen at 5x7")
    else:
        raise ValueError("analyzer accepts only formal or smoke runs")
    spec, selected_list = runner.select_mode_inputs(
        release,
        mode=expected_mode,
        per_condition_limit=limit,
        sample_id=None,
    )
    selected = tuple(selected_list)
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    if rebuilt.as_dict() != raw:
        raise ValueError("dataset contract does not rebuild exactly")
    expected_images = FORMAL_IMAGES if expected_mode == "formal" else SMOKE_IMAGES
    if len(selected) != expected_images:
        raise ValueError(
            f"{expected_mode} selection has {len(selected)}, "
            f"not {expected_images} images"
        )
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("selected row SHA-256 changed")
    ids = [str(row["sample_id"]) for row in selected]
    if immutable.get("selected_ids_sha256") != selected_ids_sha256(ids):
        raise ValueError("selected ID SHA-256 changed")
    return release, selected, rebuilt


def _expected_result_identity(
    input_row: Mapping[str, Any],
    *,
    run_id: str,
    fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    path = str(input_row["canonical_path"])
    return {
        **build_result_identity(
            input_row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "dataset_id": str(input_row["dataset_id"]),
        "input_path": path,
        "input_sha256": str(input_row["canonical_sha256"]),
        "input_width": int(input_row["width"]),
        "input_height": int(input_row["height"]),
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "checkpoint_id": str(legacy.CHECKPOINT["id"]),
        "config_fingerprint": fingerprint,
        **_visibility(input_row),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
    }


def _validate_preprocess_record(
    value: Any,
    *,
    input_row: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    record = _require_mapping(value, f"{sample_id} preprocess")
    expected_keys = {
        "decode",
        "native_shape_hwc",
        "native_width",
        "native_height",
        "decoded_bgr_sha256",
        "color_conversion",
        "decoded_rgb_sha256",
        "resize",
        "resized_rgb_sha256",
        "to_tensor",
        "normalization_mean",
        "normalization_std",
        "tensor_shape",
        "tensor_dtype",
        "tensor_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError(f"{sample_id} preprocess key set changed")
    width = int(input_row["width"])
    height = int(input_row["height"])
    expected = {
        "decode": "cv2.imread_IMREAD_COLOR",
        "native_shape_hwc": [height, width, 3],
        "native_width": width,
        "native_height": height,
        "color_conversion": "cv2_COLOR_BGR2RGB",
        "resize": {
            "output_wh": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
            "interpolation": "cv2_INTER_LINEAR",
            "preserve_aspect_ratio": False,
            "crop": None,
            "face_alignment": False,
        },
        "to_tensor": "uint8_to_float32_divide_255_CHW",
        "normalization_mean": list(legacy.CLIP_MEAN),
        "normalization_std": list(legacy.CLIP_STD),
        "tensor_shape": [3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "tensor_dtype": "float32",
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise ValueError(f"{sample_id} preprocess {key} changed")
    for key in (
        "decoded_bgr_sha256",
        "decoded_rgb_sha256",
        "resized_rgb_sha256",
        "tensor_sha256",
    ):
        _require_sha256(record.get(key), f"{sample_id} preprocess {key}")
    return record


def _float32_softmax_class1(logits: Sequence[float]) -> float:
    """Recompute the official class-1 probability in float32."""

    import torch

    tensor = torch.tensor(
        [list(logits)],
        dtype=torch.float32,
        device="cpu",
    )
    return float(torch.softmax(tensor, dim=1)[0, 1].item())


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    artifact_logits: np.ndarray | None = None,
) -> None:
    logits_raw = row.get("class_logits")
    if not isinstance(logits_raw, list) or len(logits_raw) != LOGIT_SHAPE[0]:
        raise ValueError(f"{sample_id} class_logits changed")
    logits_values = [
        _require_finite(value, f"{sample_id} class_logits[{index}]")
        for index, value in enumerate(logits_raw)
    ]
    logits = np.asarray(logits_values, dtype=np.float32)
    if logits.tolist() != logits_raw:
        raise ValueError(f"{sample_id} embedded logits are not float32 values")
    if artifact_logits is not None and not np.array_equal(
        logits,
        artifact_logits,
    ):
        raise ValueError(f"{sample_id} embedded/artifact logits differ")
    margin = float(np.float32(logits[1] - logits[0]))
    if row.get("raw_logit_margin") != margin:
        raise ValueError(f"{sample_id} raw logit margin changed")
    static_cpu_probability = _float32_softmax_class1(logits_values)
    if not 0.0 <= static_cpu_probability <= 1.0:
        raise ValueError(f"{sample_id} probability is outside [0,1]")
    probability = _require_finite(
        row.get("ai_score"),
        f"{sample_id} ai_score",
    )
    if (
        not 0.0 <= probability <= 1.0
        or abs(probability - static_cpu_probability) > STATIC_CPU_SOFTMAX_ABS_TOLERANCE
    ):
        raise ValueError(f"{sample_id} static CPU softmax sanity check changed")
    for alias in ("fake_probability", "probability", "score"):
        if row.get(alias) != probability:
            raise ValueError(f"{sample_id} score alias {alias} changed")
    decision = probability > legacy.CLASSIFICATION_THRESHOLD
    expected = {
        "score_semantics": legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
        "classification": {
            "decision": decision,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        },
        "t1": {
            "valid": True,
            "score": probability,
            "decision": decision,
        },
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
        },
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(f"{sample_id} score payload {key} changed")


def _validate_attempt(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    sample_id = str(input_row["sample_id"])
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError(f"{sample_id} result status changed")
    identity = _expected_result_identity(
        input_row,
        run_id=run_id,
        fingerprint=fingerprint,
        valid_for_metrics=status == "ok",
    )
    for key, expected in identity.items():
        if key not in row or not _same_json(row[key], expected):
            raise ValueError(f"{sample_id} result identity {key} changed")
    base_keys = set(identity) | {"status", "completed_at"}
    expected_keys = base_keys | (
        set(_OK_EXTENSION_FIELDS) if status == "ok" else set(_ERROR_EXTENSION_FIELDS)
    )
    if set(row) != expected_keys:
        raise ValueError(f"{sample_id} result row key set changed")
    _require_string(row.get("completed_at"), f"{sample_id} completed_at")
    if status == "error":
        for key in _ERROR_EXTENSION_FIELDS:
            _require_string(row.get(key), f"{sample_id} {key}")
    else:
        _validate_preprocess_record(
            row.get("preprocess"),
            input_row=input_row,
            sample_id=sample_id,
        )
        for key in ("preprocess_latency_ms", "latency_ms"):
            if _require_finite(row.get(key), f"{sample_id} {key}") < 0.0:
                raise ValueError(f"{sample_id} {key} is negative")
        peak = row.get("peak_cuda_memory_bytes")
        if peak is not None:
            _require_nonnegative_int(peak, f"{sample_id} peak CUDA memory")
        _validate_score_payload(row, sample_id=sample_id)
    _reject_nonfinite(row, f"result {sample_id}")
    _reject_unsupported_claims(row, f"result {sample_id}")


def _validate_physical_attempts(
    *,
    physical: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    inputs = {str(row["sample_id"]): row for row in selected}
    history: dict[str, list[str]] = {sample_id: [] for sample_id in inputs}
    for index, row in enumerate(physical):
        sample_id = _require_string(
            row.get("sample_id"),
            f"physical result {index} sample_id",
        )
        input_row = inputs.get(sample_id)
        if input_row is None:
            raise ValueError(f"physical result {index} has unknown sample_id")
        _validate_attempt(
            row,
            input_row=input_row,
            repo_root=repo_root,
            run_id=run_id,
            fingerprint=fingerprint,
        )
        statuses = history[sample_id]
        if "ok" in statuses:
            raise ValueError(f"{sample_id} has an attempt after success")
        statuses.append(str(row["status"]))


def _artifact_from_result(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    artifact_dir: Path,
) -> EffortArtifact:
    sample_id = _require_string(row.get("sample_id"), "artifact sample_id")
    path = _safe_repo_path(
        row.get("artifact_path"),
        repo_root=repo_root,
        label=f"{sample_id} artifact path",
    )
    expected_path = (artifact_dir / f"{sample_id}.npz").resolve()
    if path != expected_path:
        raise ValueError(f"{sample_id} artifact path is not canonical")
    relative = repo_relative(path, repo_root)
    file_sha = _require_sha256(
        row.get("artifact_sha256"),
        f"{sample_id} artifact SHA-256",
    )
    if (
        path.stat().st_size != NPZ_FILE_BYTES
        or row.get("artifact_bytes") != NPZ_FILE_BYTES
        or sha256_file(path) != file_sha
        or row.get("artifact_keys") != ["pooler_output", "class_logits"]
        or row.get("artifact_paths") != {"effort_npz": relative}
    ):
        raise ValueError(f"{sample_id} artifact file metadata changed")
    feature, logits = legacy_audit._load_artifact(path)
    if (
        feature.shape != FEATURE_SHAPE
        or feature.dtype != np.float32
        or logits.shape != LOGIT_SHAPE
        or logits.dtype != np.float32
        or not feature.flags.c_contiguous
        or not logits.flags.c_contiguous
        or not np.isfinite(feature).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError(f"{sample_id} artifact arrays changed")
    feature_sha = _array_sha256(feature)
    logits_sha = _array_sha256(logits)
    expected_metadata = {
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": feature_sha,
        "class_logits_shape": list(LOGIT_SHAPE),
        "class_logits_dtype": "float32",
        "class_logits_array_sha256": logits_sha,
    }
    for key, expected in expected_metadata.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} artifact metadata {key} changed")
    _validate_score_payload(
        row,
        sample_id=sample_id,
        artifact_logits=logits,
    )
    return EffortArtifact(
        sample_id=sample_id,
        path=path,
        relative_path=relative,
        file_sha256=file_sha,
        file_bytes=NPZ_FILE_BYTES,
        feature_array_sha256=feature_sha,
        logits_array_sha256=logits_sha,
        feature=feature,
        logits=logits,
    )


def validate_artifact_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_dir: Path,
) -> dict[str, EffortArtifact]:
    root = repo_root.resolve()
    try:
        artifact_dir.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("artifact directory escapes repository root") from error
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise FileNotFoundError(f"missing or unsafe artifact dir: {artifact_dir}")
    ids = [
        _require_string(row.get("sample_id"), "latest result sample_id")
        for row in latest_results
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("latest result IDs are duplicated")
    entries = list(artifact_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("artifact inventory contains a non-regular entry")
    expected_names = {f"{sample_id}.npz" for sample_id in ids}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            "Effort artifact inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    return {
        str(row["sample_id"]): _artifact_from_result(
            row=row,
            repo_root=root,
            artifact_dir=artifact_dir,
        )
        for row in latest_results
    }


def _validate_dataset_snapshot(
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
        raise ValueError("expected-input snapshot differs from selection")
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    expected = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    if set(dataset) != set(expected):
        raise ValueError("manifest.dataset key set changed")
    for key, expected_value in expected.items():
        if dataset.get(key) != expected_value:
            raise ValueError(f"manifest.dataset.{key} changed")


def _validate_execution(
    *,
    manifest: Mapping[str, Any],
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(manifest.get("execution"), "execution")
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
    for key in expected_keys:
        _require_nonnegative_int(execution.get(key), f"execution.{key}")
    expected = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
        "new_errors": 0,
    }
    for key, expected_value in expected.items():
        if execution.get(key) != expected_value:
            raise ValueError(f"execution.{key} changed")
    if execution["new_successes"] + execution["resume_skips"] != selected_images:
        raise ValueError("execution successful-work accounting changed")


def _validate_summary(
    *,
    summary: Mapping[str, Any],
    mode: str,
    run_id: str,
    fingerprint: str,
    contract: RunDatasetContract,
    coverage: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_effort_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "score_spec": runner.SCORE_SPEC.as_dict(),
        "dataset_contract": contract.as_dict(),
        "coverage": dict(coverage),
    }
    if set(summary) != {*expected, "generated_at"}:
        raise ValueError("runtime summary key set changed")
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"runtime summary {key} changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    _reject_nonfinite(summary, "summary")
    _reject_unsupported_claims(summary, "summary")


def _evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    artifacts: Mapping[str, EffortArtifact] | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if artifacts is not None:
        snapshot["artifacts"] = {
            sample_id: {
                "file_sha256": artifact.file_sha256,
                "file_bytes": artifact.file_bytes,
                "feature_array_sha256": artifact.feature_array_sha256,
                "logits_array_sha256": artifact.logits_array_sha256,
            }
            for sample_id, artifact in sorted(artifacts.items())
        }
    return snapshot


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    mode: str,
) -> RunBundle:
    root = repo_root.resolve()
    results_root = _resolve_results_root(results_dir, root)
    run_dir = _resolve_run_dir(results_root, run_id)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for path, label in (
        (manifest_path, "manifest"),
        (results_path, "results"),
        (expected_path, "expected inputs"),
        (summary_path, "summary"),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    before = _evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
    )
    manifest = _load_json(manifest_path, f"{mode} run manifest")
    fingerprint, immutable, artifact_dir = _validate_manifest(
        manifest=manifest,
        repo_root=root,
        run_id=run_id,
        run_dir=run_dir,
        expected_mode=mode,
    )
    release, selected, contract = _rebuild_contract(
        repo_root=root,
        immutable=immutable,
        expected_mode=mode,
    )
    _validate_dataset_snapshot(
        manifest=manifest,
        repo_root=root,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
    )
    physical = tuple(_read_jsonl_strict(results_path, f"{mode} results"))
    _validate_physical_attempts(
        physical=physical,
        selected=selected,
        repo_root=root,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    latest_index = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=runner.SCORE_SPEC,
    )
    coverage_value = summarize_coverage(latest_index)
    require_complete_coverage(coverage_value)
    latest = tuple(
        dict(latest_index.latest_by_sample_id[str(row["sample_id"])])
        for row in selected
    )
    if any(
        row.get("status") != "ok" or row.get("valid_for_metrics") is not True
        for row in latest
    ):
        raise ValueError("latest Effort coverage is not entirely successful")
    if mode == "smoke" and len(physical) != SMOKE_IMAGES:
        raise ValueError("smoke requires one physical attempt per input")
    _validate_execution(
        manifest=manifest,
        selected_images=len(selected),
        physical_rows=len(physical),
        latest_rows=len(latest),
    )
    artifacts = validate_artifact_inventory(
        latest_results=latest,
        repo_root=root,
        artifact_dir=artifact_dir,
    )
    outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    immutable_outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable.outputs",
    )
    expected_outputs = {
        **dict(immutable_outputs),
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "artifact_files": len(artifacts),
    }
    if set(outputs) != set(expected_outputs):
        raise ValueError("manifest.outputs key set changed")
    for key, expected_value in expected_outputs.items():
        if outputs.get(key) != expected_value:
            raise ValueError(f"manifest.outputs.{key} changed")
    summary = _load_json(summary_path, f"{mode} run summary")
    coverage = coverage_value.as_dict()
    _validate_summary(
        summary=summary,
        mode=mode,
        run_id=run_id,
        fingerprint=fingerprint,
        contract=contract,
        coverage=coverage,
    )
    after = _evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
    )
    if before != after:
        raise ValueError("run evidence changed while being validated")
    return RunBundle(
        run_id=run_id,
        mode=mode,
        fingerprint=fingerprint,
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
            "Effort Balanced250 metrics require " "iterations=1000 seed=20260726"
        )
    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("Balanced250 metrics require the full formal run")
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
    _reject_nonfinite(metrics, "metrics")
    _reject_unsupported_claims(metrics, "metrics")
    return metrics


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("smoke comparison requires successful result rows")
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing = _SMOKE_IGNORED_FIELDS - set(row)
    if missing:
        raise ValueError(f"{sample_id} lacks ignored field {sorted(missing)[0]}")
    return {
        key: value for key, value in row.items() if key not in _SMOKE_IGNORED_FIELDS
    }


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_artifacts: Mapping[str, EffortArtifact],
    replay_artifacts: Mapping[str, EffortArtifact],
) -> dict[str, Any]:
    def unique(
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"),
                f"{label}[{index}].sample_id",
            )
            if sample_id in result:
                raise ValueError(f"{label} duplicates {sample_id}")
            result[sample_id] = row
        if not result:
            raise ValueError(f"{label} is empty")
        return result

    reference = unique(reference_rows, "reference results")
    replay = unique(replay_rows, "replay results")
    if set(reference) != set(replay):
        raise ValueError("smoke result coverage differs")
    if set(reference_artifacts) != set(reference):
        raise ValueError("reference artifact coverage differs")
    if set(replay_artifacts) != set(replay):
        raise ValueError("replay artifact coverage differs")
    max_feature = 0.0
    max_logits = 0.0
    for sample_id in sorted(reference):
        left_projection = _exact_smoke_projection(reference[sample_id])
        right_projection = _exact_smoke_projection(replay[sample_id])
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
                f"{sample_id} smoke computational row differs at " f"{differing[:3]}"
            )
        left = reference_artifacts[sample_id]
        right = replay_artifacts[sample_id]
        if (
            left.feature.shape != right.feature.shape
            or left.feature.dtype != right.feature.dtype
            or left.logits.shape != right.logits.shape
            or left.logits.dtype != right.logits.dtype
        ):
            raise ValueError(f"{sample_id} artifact shape/dtype differs")
        feature_difference = float(
            np.max(
                np.abs(
                    left.feature.astype(np.float64) - right.feature.astype(np.float64)
                )
            )
        )
        logit_difference = float(
            np.max(
                np.abs(left.logits.astype(np.float64) - right.logits.astype(np.float64))
            )
        )
        max_feature = max(max_feature, feature_difference)
        max_logits = max(max_logits, logit_difference)
        if (
            feature_difference != 0.0
            or logit_difference != 0.0
            or left.feature_array_sha256 != right.feature_array_sha256
            or left.logits_array_sha256 != right.logits_array_sha256
        ):
            raise ValueError(f"{sample_id} smoke artifact arrays differ")
    return {
        "images_compared": len(reference),
        "exact_computational_projection": True,
        "identity_and_volatile_fields_ignored": sorted(_SMOKE_IGNORED_FIELDS),
        "npz_file_sha256_ignored_due_to_zip_timestamps": True,
        "static_cpu_softmax_sanity_abs_tolerance": (STATIC_CPU_SOFTMAX_ABS_TOLERANCE),
        "max_feature_abs_difference": max_feature,
        "max_class_logit_abs_difference": max_logits,
        "feature_and_logit_array_sha256_exact": True,
        "score_and_decision_exact": True,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    expected = set(immutable)
    if "run_id" not in expected or "outputs" not in expected:
        raise ValueError("smoke immutable config lacks run-specific fields")
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def _configure_exact_recorded_runtime(
    *,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    recorded = _validate_runtime(recorded_runtime, label="recorded runtime")
    if device_text != recorded.get("device"):
        raise ValueError("replay device must exactly match the recorded run device")
    device, runtime = legacy_audit._configure_runtime(device_text)
    for key, actual in runtime.items():
        if key in recorded and recorded[key] != actual:
            raise ValueError(f"independent replay runtime {key} differs")
    return device, runtime


def _validate_head_state(state: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    if "head.weight" not in state or "head.bias" not in state:
        raise ValueError("Effort state lacks the two-class head")
    weight = state["head.weight"]
    bias = state["head.bias"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or tuple(weight.shape) != (legacy.CLASS_COUNT, legacy.FEATURE_DIMENSION)
        or tuple(bias.shape) != (legacy.CLASS_COUNT,)
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("Effort two-class head tensors changed")
    return weight, bias


def replay_linear_head(
    bundle: RunBundle,
    *,
    state: Mapping[str, Any],
    device_text: str,
) -> dict[str, Any]:
    """Replay every persisted 1,024-D feature through the exact 2-class head."""

    import torch
    from torch.nn import functional

    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"),
        "bundle recorded runtime",
    )
    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
    )
    weight_cpu, bias_cpu = _validate_head_state(state)
    weight = weight_cpu.detach().to(device=device, dtype=torch.float32)
    bias = bias_cpu.detach().to(device=device, dtype=torch.float32)
    max_logits = 0.0
    max_probability = 0.0
    max_margin = 0.0
    replayed = 0
    with torch.inference_mode():
        for row in bundle.latest_results:
            sample_id = str(row["sample_id"])
            artifact = bundle.artifacts.get(sample_id)
            if artifact is None:
                raise ValueError(f"missing artifact for head replay {sample_id}")
            feature = (
                torch.from_numpy(artifact.feature)
                .reshape(1, -1)
                .to(device=device, dtype=torch.float32)
            )
            logits = functional.linear(feature, weight, bias)
            probability = torch.softmax(logits, dim=1)[:, 1]
            margin = logits[:, 1] - logits[:, 0]
            logits_array = np.ascontiguousarray(
                logits[0].detach().cpu().numpy(),
                dtype=np.float32,
            )
            logit_difference = float(
                np.max(
                    np.abs(
                        logits_array.astype(np.float64)
                        - artifact.logits.astype(np.float64)
                    )
                )
            )
            probability_value = float(probability.item())
            margin_value = float(margin.item())
            probability_difference = abs(probability_value - float(row["ai_score"]))
            margin_difference = abs(margin_value - float(row["raw_logit_margin"]))
            max_logits = max(max_logits, logit_difference)
            max_probability = max(max_probability, probability_difference)
            max_margin = max(max_margin, margin_difference)
            if logit_difference > HEAD_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} independent head logits differ")
            if probability_difference > PROBABILITY_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} independent softmax differs")
            if margin_difference > MARGIN_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} independent margin differs")
            if (probability_value > legacy.CLASSIFICATION_THRESHOLD) is not row.get(
                "classification_decision"
            ):
                raise ValueError(f"{sample_id} independent decision differs")
            replayed += 1
    if replayed != len(bundle.latest_results):
        raise ValueError("independent head replay coverage is incomplete")
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "independent_two_class_head_replay_passed",
        "features_replayed": replayed,
        "feature_dimension": legacy.FEATURE_DIMENSION,
        "class_count": legacy.CLASS_COUNT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "device": device_text,
        "runtime": runtime,
        "recorded_runtime_exact_match": True,
        "head_logit_abs_tolerance": HEAD_ABS_TOLERANCE,
        "probability_abs_tolerance": PROBABILITY_ABS_TOLERANCE,
        "margin_abs_tolerance": MARGIN_ABS_TOLERANCE,
        "max_head_logit_abs_difference": max_logits,
        "max_probability_abs_difference": max_probability,
        "max_margin_abs_difference": max_margin,
    }


def _validate_independent_model_audit(
    independent: Mapping[str, Any],
    recorded: Mapping[str, Any],
) -> None:
    expected_names = recorded.get("svd_module_names")
    expected = {
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "svd_modules": legacy.SVD_MODULE_COUNT,
        "svd_module_names": expected_names,
        "parameter_count": legacy.CHECKPOINT["state_elements"],
        "position_ids_materialized": True,
    }
    if dict(independent) != expected:
        raise ValueError("independent Effort model reconstruction changed")


def _replay_runtime_golden_independently(
    *,
    model: Any,
    device: Any,
    source_root: Path,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    device_family = "cuda" if getattr(device, "type", None) == "cuda" else "cpu"
    golden = _validate_runtime_golden(
        recorded,
        label="recorded runtime golden replay",
        expected_device_family=device_family,
    )
    max_logits = 0.0
    max_probability = 0.0
    replayed = 0
    with torch.inference_mode():
        for index, (frozen, recorded_value) in enumerate(
            zip(legacy.GOLDEN_CASES, golden["cases"], strict=True)
        ):
            case = _require_mapping(
                recorded_value,
                f"recorded runtime golden case {index}",
            )
            path = source_root / str(frozen["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != frozen["sha256"]
            ):
                raise ValueError("Effort golden fixture provenance changed")
            image, preprocess = legacy_audit._preprocess(path)
            if case.get("preprocess") != preprocess:
                raise ValueError("recorded golden preprocessing changed")
            tensor = (
                torch.from_numpy(image)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            logits, feature = model(tensor)
            probability = torch.softmax(logits, dim=1)[:, 1]
            logits_array = np.ascontiguousarray(
                logits[0].detach().cpu().numpy(),
                dtype=np.float32,
            )
            feature_array = np.ascontiguousarray(
                feature[0].detach().cpu().numpy(),
                dtype=np.float32,
            )
            recorded_logits = np.asarray(
                case.get("logits"),
                dtype=np.float32,
            )
            logit_difference = float(
                np.max(
                    np.abs(
                        logits_array.astype(np.float64)
                        - recorded_logits.astype(np.float64)
                    )
                )
            )
            probability_value = float(probability.item())
            probability_difference = abs(
                probability_value - float(case["fake_probability"])
            )
            max_logits = max(max_logits, logit_difference)
            max_probability = max(max_probability, probability_difference)
            if logit_difference != 0.0 or probability_difference != 0.0:
                raise ValueError("independent recorded golden replay differs")
            if _array_sha256(feature_array) != case.get("feature_sha256"):
                raise ValueError("independent golden feature differs")
            frozen_logits = np.asarray(
                frozen[f"{device_family}_logits"],
                dtype=np.float64,
            )
            frozen_difference = float(
                np.max(np.abs(logits_array.astype(np.float64) - frozen_logits))
            )
            frozen_probability_difference = abs(
                probability_value - float(frozen[f"{device_family}_probability"])
            )
            if (
                frozen_difference > legacy.GOLDEN_RUNTIME_ABS_TOLERANCE
                or frozen_probability_difference > legacy.GOLDEN_RUNTIME_ABS_TOLERANCE
            ):
                raise ValueError("independent frozen runtime golden changed")
            replayed += 1
    return {
        "status": "independent_runtime_golden_replay_passed",
        "cases_replayed": replayed,
        "device_family": device_family,
        "max_recorded_logit_abs_difference": max_logits,
        "max_recorded_probability_abs_difference": max_probability,
        "recorded_values_exact": True,
    }


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    device_text: str,
) -> dict[str, Any]:
    """Freshly preprocess and forward every one of the 1,775 formal inputs."""

    import torch
    from torch.nn import functional

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh Effort replay requires all formal Balanced250 images")
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"),
        "bundle recorded runtime",
    )
    device, runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
    )
    model, independent_model_audit = legacy_audit._build_model(
        state,
        config,
        device,
    )
    _validate_independent_model_audit(
        independent_model_audit,
        _require_mapping(
            bundle.immutable.get("model_audit"),
            "recorded model audit",
        ),
    )
    runtime_golden_report = _replay_runtime_golden_independently(
        model=model,
        device=device,
        source_root=source_root,
        recorded=_require_mapping(
            bundle.immutable.get("runtime_golden"),
            "recorded runtime golden",
        ),
    )
    max_feature = 0.0
    max_logits = 0.0
    max_fresh_head = 0.0
    max_saved_head = 0.0
    max_probability = 0.0
    max_margin = 0.0
    replayed = 0
    try:
        with torch.inference_mode():
            for input_row, row in zip(
                bundle.selected,
                bundle.latest_results,
                strict=True,
            ):
                sample_id = str(input_row["sample_id"])
                path = _safe_repo_path(
                    input_row.get("canonical_path"),
                    repo_root=bundle.release.repo_root,
                    label=f"{sample_id} canonical input",
                )
                if sha256_file(path) != input_row.get("canonical_sha256"):
                    raise ValueError(f"{sample_id} canonical input changed")
                image, preprocess = legacy_audit._preprocess(path)
                if row.get("preprocess") != preprocess:
                    raise ValueError(f"{sample_id} fresh preprocessing differs")
                tensor = (
                    torch.from_numpy(image)
                    .unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                )
                fresh_logits, fresh_feature = model(tensor)
                fresh_head = functional.linear(
                    fresh_feature,
                    model.head.weight,
                    model.head.bias,
                )
                artifact = bundle.artifacts.get(sample_id)
                if artifact is None:
                    raise ValueError(f"{sample_id} persisted artifact missing")
                saved_feature = (
                    torch.from_numpy(artifact.feature)
                    .reshape(1, -1)
                    .to(device=device, dtype=torch.float32)
                )
                saved_head = functional.linear(
                    saved_feature,
                    model.head.weight,
                    model.head.bias,
                )
                probability = torch.softmax(fresh_logits, dim=1)[:, 1]
                saved_probability = torch.softmax(saved_head, dim=1)[:, 1]
                margin = fresh_logits[:, 1] - fresh_logits[:, 0]
                fresh_feature_array = np.ascontiguousarray(
                    fresh_feature[0].detach().cpu().numpy(),
                    dtype=np.float32,
                )
                fresh_logits_array = np.ascontiguousarray(
                    fresh_logits[0].detach().cpu().numpy(),
                    dtype=np.float32,
                )
                fresh_head_array = np.ascontiguousarray(
                    fresh_head[0].detach().cpu().numpy(),
                    dtype=np.float32,
                )
                saved_head_array = np.ascontiguousarray(
                    saved_head[0].detach().cpu().numpy(),
                    dtype=np.float32,
                )
                feature_difference = float(
                    np.max(
                        np.abs(
                            fresh_feature_array.astype(np.float64)
                            - artifact.feature.astype(np.float64)
                        )
                    )
                )
                logits_difference = float(
                    np.max(
                        np.abs(
                            fresh_logits_array.astype(np.float64)
                            - artifact.logits.astype(np.float64)
                        )
                    )
                )
                fresh_head_difference = float(
                    np.max(
                        np.abs(
                            fresh_head_array.astype(np.float64)
                            - fresh_logits_array.astype(np.float64)
                        )
                    )
                )
                saved_head_difference = float(
                    np.max(
                        np.abs(
                            saved_head_array.astype(np.float64)
                            - artifact.logits.astype(np.float64)
                        )
                    )
                )
                probability_value = float(probability.item())
                probability_difference = abs(probability_value - float(row["ai_score"]))
                saved_probability_difference = abs(
                    float(saved_probability.item()) - float(row["ai_score"])
                )
                margin_value = float(margin.item())
                margin_difference = abs(margin_value - float(row["raw_logit_margin"]))
                max_feature = max(max_feature, feature_difference)
                max_logits = max(max_logits, logits_difference)
                max_fresh_head = max(
                    max_fresh_head,
                    fresh_head_difference,
                )
                max_saved_head = max(
                    max_saved_head,
                    saved_head_difference,
                )
                max_probability = max(
                    max_probability,
                    probability_difference,
                    saved_probability_difference,
                )
                max_margin = max(max_margin, margin_difference)
                if feature_difference > FEATURE_ABS_TOLERANCE:
                    raise ValueError(f"{sample_id} fresh feature differs")
                if logits_difference > LOGIT_ABS_TOLERANCE:
                    raise ValueError(f"{sample_id} fresh logits differ")
                if (
                    fresh_head_difference > HEAD_ABS_TOLERANCE
                    or saved_head_difference > HEAD_ABS_TOLERANCE
                ):
                    raise ValueError(f"{sample_id} fresh head replay differs")
                if (
                    probability_difference > PROBABILITY_ABS_TOLERANCE
                    or saved_probability_difference > PROBABILITY_ABS_TOLERANCE
                ):
                    raise ValueError(f"{sample_id} fresh softmax differs")
                if margin_difference > MARGIN_ABS_TOLERANCE:
                    raise ValueError(f"{sample_id} fresh margin differs")
                if (probability_value > legacy.CLASSIFICATION_THRESHOLD) is not row.get(
                    "classification_decision"
                ):
                    raise ValueError(f"{sample_id} fresh decision differs")
                replayed += 1
                if replayed % 25 == 0 or replayed == FORMAL_IMAGES:
                    print(
                        f"[Effort audit {replayed}/{FORMAL_IMAGES}] " f"{sample_id}",
                        flush=True,
                    )
    finally:
        del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    if replayed != FORMAL_IMAGES:
        raise ValueError("fresh full-model replay coverage is incomplete")
    return {
        "status": "independent_full_model_replay_passed",
        "images_replayed": replayed,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_HF_CONFIG_SHA256,
        "runtime": runtime,
        "runtime_golden": runtime_golden_report,
        "feature_abs_tolerance": FEATURE_ABS_TOLERANCE,
        "class_logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "head_logit_abs_tolerance": HEAD_ABS_TOLERANCE,
        "probability_abs_tolerance": PROBABILITY_ABS_TOLERANCE,
        "margin_abs_tolerance": MARGIN_ABS_TOLERANCE,
        "max_feature_abs_difference": max_feature,
        "max_class_logit_abs_difference": max_logits,
        "max_fresh_head_logit_abs_difference": max_fresh_head,
        "max_saved_feature_head_logit_abs_difference": max_saved_head,
        "max_probability_abs_difference": max_probability,
        "max_margin_abs_difference": max_margin,
        "all_preprocess_records_exact": True,
        "all_decisions_exact": True,
        "fresh_model_forwards": replayed,
    }


def _bundle_snapshot(bundle: RunBundle) -> dict[str, Any]:
    return _evidence_snapshot(
        manifest_path=bundle.manifest_path,
        results_path=bundle.results_path,
        expected_path=bundle.expected_path,
        summary_path=bundle.summary_path,
        artifacts=bundle.artifacts,
    )


def _verify_bundle_unchanged(
    bundle: RunBundle,
    expected: Mapping[str, Any],
) -> None:
    for artifact in bundle.artifacts.values():
        if (
            artifact.path.is_symlink()
            or not artifact.path.is_file()
            or artifact.path.stat().st_size != artifact.file_bytes
            or sha256_file(artifact.path) != artifact.file_sha256
        ):
            raise ValueError("Effort artifact changed during analysis")
    if _bundle_snapshot(bundle) != dict(expected):
        raise ValueError("Effort run evidence changed during analysis")


def _json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _json_file_sha256(value: Any) -> str:
    return hashlib.sha256(_json_file_bytes(value)).hexdigest()


def _write_json_verified(path: Path, value: Any, *, label: str) -> None:
    expected_sha = _json_file_sha256(value)
    atomic_write_json(path, value)
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_sha
        or _load_json(path, label) != value
    ):
        raise ValueError(f"written {label} failed verification")


def _formal_output_paths(
    *,
    bundle: RunBundle,
    repo_root: Path,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
) -> tuple[Path, Path]:
    metrics = (
        bundle.run_dir / "balanced250_metrics.json"
        if metrics_output_path is None
        else (
            metrics_output_path.resolve()
            if metrics_output_path.is_absolute()
            else (repo_root / metrics_output_path).resolve()
        )
    )
    audit = (
        bundle.run_dir / "independent_audit.json"
        if audit_output_path is None
        else (
            audit_output_path.resolve()
            if audit_output_path.is_absolute()
            else (repo_root / audit_output_path).resolve()
        )
    )
    expected = (
        (bundle.run_dir / "balanced250_metrics.json").resolve(),
        (bundle.run_dir / "independent_audit.json").resolve(),
    )
    if (metrics.resolve(), audit.resolve()) != expected:
        raise ValueError(
            "formal outputs must be the canonical files inside the run dir"
        )
    for path, label in ((metrics, "metrics output"), (audit, "audit output")):
        relative = repo_relative(path, repo_root)
        if Path(relative).is_absolute():
            raise ValueError(f"{label} escapes repository root")
        _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=label,
            require_file=False,
        )
        if path in {
            bundle.manifest_path,
            bundle.results_path,
            bundle.expected_path,
            bundle.summary_path,
        }:
            raise ValueError(f"{label} overlaps protected run evidence")
    return metrics.resolve(), audit.resolve()


def _comparison_output_path(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    output_path: Path | None,
) -> Path:
    results_root = _resolve_results_root(results_dir, repo_root)
    expected = (
        results_root
        / "_reports"
        / f"{reference_run_id}__vs__{replay_run_id}_comparison.json"
    ).resolve()
    actual = (
        expected
        if output_path is None
        else (
            output_path.resolve()
            if output_path.is_absolute()
            else (repo_root / output_path).resolve()
        )
    )
    if actual != expected:
        raise ValueError("smoke comparison output must use the canonical _reports path")
    relative = repo_relative(actual, repo_root)
    if Path(relative).is_absolute():
        raise ValueError("smoke comparison output escapes repository root")
    _safe_repo_path(
        relative,
        repo_root=repo_root,
        label="smoke comparison output",
        require_file=False,
    )
    return actual


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    source_root: Path,
    checkpoint_path: Path,
    hf_config_path: Path,
    device_text: str | None,
    output_path: Path | None,
) -> dict[str, Any]:
    reference_run_id = runner._valid_run_id(reference_run_id)
    replay_run_id = runner._valid_run_id(replay_run_id)
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
    reference_snapshot = _bundle_snapshot(reference)
    replay_snapshot = _bundle_snapshot(replay)
    if reference.selected != replay.selected:
        raise ValueError("smoke canonical selections differ")
    if (
        reference.contract.as_dict() != replay.contract.as_dict()
        or _smoke_immutable_projection(reference.immutable)
        != _smoke_immutable_projection(replay.immutable)
    ):
        raise ValueError("smoke immutable computational contracts differ")
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
    )
    recorded_runtime = _require_mapping(
        reference.immutable.get("runtime"),
        "reference recorded runtime",
    )
    replay_device = (
        str(recorded_runtime["device"]) if device_text is None else device_text
    )
    source, assets, state, _config = _verify_source_assets_independently(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_config_path=hf_config_path,
        recorded_source=_require_mapping(
            reference.immutable.get("source"),
            "reference source",
        ),
        recorded_assets=_require_mapping(
            reference.immutable.get("assets"),
            "reference assets",
        ),
    )
    head_reference = replay_linear_head(
        reference,
        state=state,
        device_text=replay_device,
    )
    head_replay = replay_linear_head(
        replay,
        state=state,
        device_text=replay_device,
    )
    _verify_bundle_unchanged(reference, reference_snapshot)
    _verify_bundle_unchanged(replay, replay_snapshot)
    target = _comparison_output_path(
        repo_root=repo_root,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
        output_path=output_path,
    )
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_smoke_comparison_passed",
        "compared_at": utc_now(),
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            **reference_snapshot,
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            **replay_snapshot,
        },
        "selection": reference.contract.selection.as_dict(),
        "source": source,
        "assets": assets,
        "comparison": comparison,
        "reference_head_replay": head_reference,
        "replay_head_replay": head_replay,
    }
    _reject_nonfinite(report, "smoke comparison")
    _reject_unsupported_claims(report, "smoke comparison")
    _write_json_verified(target, report, label="smoke comparison")
    return report


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    checkpoint_path: Path,
    hf_config_path: Path,
    device_text: str | None,
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
    snapshot = _bundle_snapshot(bundle)
    metrics_path, audit_path = _formal_output_paths(
        bundle=bundle,
        repo_root=repo_root,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    metrics = recompute_metrics(bundle, iterations=iterations, seed=seed)
    source, assets, state, config = _verify_source_assets_independently(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_config_path=hf_config_path,
        recorded_source=_require_mapping(
            bundle.immutable.get("source"),
            "recorded source",
        ),
        recorded_assets=_require_mapping(
            bundle.immutable.get("assets"),
            "recorded assets",
        ),
    )
    recorded_runtime = _require_mapping(
        bundle.immutable.get("runtime"),
        "recorded runtime",
    )
    replay_device = (
        str(recorded_runtime["device"]) if device_text is None else device_text
    )
    head_report = replay_linear_head(
        bundle,
        state=state,
        device_text=replay_device,
    )
    model_report = (
        replay_model(
            bundle,
            source_root=source_root,
            state=state,
            config=config,
            device_text=replay_device,
        )
        if replay
        else None
    )
    _verify_bundle_unchanged(bundle, snapshot)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "full_fresh_replay_audit_passed"
            if replay
            else "artifact_and_head_audit_passed"
        ),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": len(bundle.selected),
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": bundle.coverage,
        "artifact_files": len(bundle.artifacts),
        "visibility_census": visibility_census(bundle.selected),
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "source": source,
        "assets": assets,
        "independent_head_replay": head_report,
        "fresh_full_model_replay": model_report,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        "contract_checks": {
            "exact_formal_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "artifact_path_sha256_shape_dtype_finiteness_validated": True,
            "embedded_logits_and_float32_softmax_recomputed": True,
            "static_cpu_softmax_sanity_abs_tolerance": (
                STATIC_CPU_SOFTMAX_ABS_TOLERANCE
            ),
            "balanced250_primary_and_secondary_t1_metrics_recomputed": True,
            "predicted_dense_output": False,
        },
        "evidence": snapshot,
        "outputs": {
            "metrics_path": repo_relative(metrics_path, repo_root),
            "metrics_sha256": _json_file_sha256(metrics),
            "audit_path": repo_relative(audit_path, repo_root),
        },
    }
    _reject_nonfinite(audit, "audit")
    _reject_unsupported_claims(audit, "audit")
    # Writes happen only after every source, run, artifact, metric, and replay
    # check above has passed.
    _write_json_verified(metrics_path, metrics, label="Balanced250 metrics")
    _write_json_verified(audit_path, audit, label="independent audit")
    if sha256_file(metrics_path) != audit["outputs"]["metrics_sha256"]:
        raise ValueError("written metrics digest changed")
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-config", type=Path, default=DEFAULT_HF_CONFIG)
    parser.add_argument(
        "--device",
        help=(
            "must equal the run's recorded explicit device; omitted means "
            "use that recorded device"
        ),
    )
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument(
        "--compare-smoke-run-id",
        help=(
            "validate this run and a second frozen 5x7 smoke run, replay "
            "both heads, and write their exact comparison"
        ),
    )
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
    repo_root = args.repo_root.resolve()
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    source_root = _anchored(args.source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    hf_config_path = _anchored(args.hf_config, repo_root)
    run_id = runner._valid_run_id(args.run_id)
    if args.compare_smoke_run_id is not None:
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
            or args.bootstrap_iterations != BOOTSTRAP_ITERATIONS
            or args.bootstrap_seed != BOOTSTRAP_SEED
        ):
            raise ValueError("smoke comparison cannot use formal-analysis options")
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=runner._valid_run_id(args.compare_smoke_run_id),
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            hf_config_path=hf_config_path,
            device_text=args.device,
            output_path=args.comparison_output,
        )
    else:
        if args.comparison_output is not None:
            raise ValueError("--comparison-output requires --compare-smoke-run-id")
        report = analyze(
            repo_root=repo_root,
            results_dir=results_dir,
            run_id=run_id,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            hf_config_path=hf_config_path,
            device_text=args.device,
            metrics_output_path=args.metrics_output,
            audit_output_path=args.audit_output,
            replay=not args.skip_model_replay,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
