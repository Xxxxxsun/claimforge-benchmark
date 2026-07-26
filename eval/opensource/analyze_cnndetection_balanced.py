#!/usr/bin/env python3
"""Audit and independently replay a formal CNNDetection Balanced250 run.

The runner is allowed to keep append-only attempt history.  This analyzer
validates every physical attempt with the shared v2 contract, requires a
successful latest attempt for all 1,775 formal inputs, verifies the exact
feature inventory, recomputes the frozen Balanced250 statistics, and replays
every image through a freshly loaded official model.

The optional run-to-run comparator is used for deterministic smoke and fresh
replay evidence.  It deliberately ignores only run identity and timing while
requiring identical input identity, scores, decisions, and feature bytes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import analyze_cnndetection_run as legacy_analyzer
from eval.opensource import run_cnndetection as legacy_runner
from eval.opensource import run_cnndetection_balanced as balanced_runner
from eval.opensource.balanced250_metrics import (
    summarize_balanced250_t1,
)
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RunDatasetContract,
    ScoreSpec,
    build_run_dataset_contract,
    index_latest_attempts,
    require_complete_coverage,
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


MANIFEST_SCHEMA_VERSION = balanced_runner.RUN_MANIFEST_SCHEMA
AUDIT_SCHEMA_VERSION = "cnndetection_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "cnndetection_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
DEFAULT_RESULTS_DIR = balanced_runner.DEFAULT_RESULTS_DIR
DEFAULT_RUN_ID = balanced_runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SOURCE_ROOT = legacy_runner.DEFAULT_SOURCE_ROOT
DEFAULT_CHECKPOINT = legacy_runner.DEFAULT_CHECKPOINT
FORMAL_IMAGES = 1775
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
RAW_LOGIT_ABS_TOLERANCE = 1e-4
SCORE_ABS_TOLERANCE = 1e-7
FEATURE_ABS_TOLERANCE = 1e-4

_RUN_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
    }
)
_SMOKE_ROW_IGNORED_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "preprocess_latency_ms",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "cnndetection_feature_path",
        "cnndetection_feature_sha256",
        "cnndetection_feature_bytes",
    }
)
_RESULT_PROJECTION_FIELDS = (
    "schema_version",
    "dataset_id",
    "id",
    "sample_id",
    "rank",
    "condition",
    "condition_family",
    "manipulation_scope",
    "normalized_task_id",
    "task_id",
    "kind",
    "label",
    "domain",
    "gt_mask_kind",
    "input_path",
    "input_sha256",
    "input_width",
    "input_height",
    "status",
    "valid_for_metrics",
    "model",
    "model_slug",
    "preprocess_profile",
    "checkpoint_id",
    "task_scope",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
    "preprocess",
    "cnndetection_feature_shape",
    "cnndetection_feature_dtype",
    "cnndetection_feature_semantics",
    "raw_logit",
    "probability",
    "ai_score",
    "score",
    "score_semantics",
    "calibrated_probability",
    "classification_decision",
    "classification_threshold",
    "classification_threshold_operator",
    "classification",
    "t1",
    "manual_replay",
)


@dataclass(frozen=True)
class FeatureArtifact:
    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: str
    array: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    fingerprint: str
    run_dir: Path
    manifest_path: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    release: CanonicalRelease
    selected: tuple[dict[str, Any], ...]
    contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]
    features: Mapping[str, FeatureArtifact]


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


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} is not boolean")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return _require_mapping(value, label)


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"{label}:{line_number} lacks final newline")
            if not line.strip():
                raise ValueError(f"{label}:{line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label}:{line_number} is invalid JSON"
                ) from error
            row = _require_mapping(value, f"{label}:{line_number}")
            if line != f"{stable_json(row)}\n":
                raise ValueError(
                    f"{label}:{line_number} is not canonical JSONL"
                )
            rows.append(row)
    return rows


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _resolve_results_root(results_dir: Path, repo_root: Path) -> Path:
    return (
        results_dir.resolve()
        if results_dir.is_absolute()
        else (repo_root.resolve() / results_dir).resolve()
    )


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    valid_run_id = balanced_runner._valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / valid_run_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("resolved run directory escapes results root") from error
    if candidate == root:
        raise ValueError("run directory must be below results root")
    return candidate


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> None:
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected = set(balanced_runner.ADAPTER_SOURCE_PATHS)
    actual = set(sources)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "immutable.adapter_sources key set mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative, raw in sources.items():
        record = _require_mapping(raw, f"adapter source {relative}")
        if record.get("path", relative) != relative:
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


def _reject_t2_claims(value: Any, label: str) -> None:
    forbidden = balanced_runner._forbidden_t2_claims(value)
    if forbidden:
        raise ValueError(
            f"{label} contains unsupported T2/localization claim "
            f"{sorted(forbidden)[0]!r}"
        )


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


def _rebuild_formal_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    raw_contract = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_value = _require_mapping(
        raw_contract.get("release"),
        "dataset contract release",
    )
    manifest_path = _safe_repo_path(
        release_value.get("manifest_path"),
        repo_root=repo_root,
        label="dataset manifest",
    )
    release = load_canonical_release(repo_root, manifest_path, verify_files=True)
    spec, selected_value = balanced_runner.select_mode_inputs(
        release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    selected = tuple(selected_value)
    score_spec = ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=legacy_runner.CLASSIFICATION_THRESHOLD,
        threshold_operator=legacy_runner.CLASSIFICATION_THRESHOLD_OPERATOR,
    )
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=score_spec,
    )
    if rebuilt.as_dict() != raw_contract:
        raise ValueError("immutable dataset contract does not rebuild exactly")
    if len(selected) != FORMAL_IMAGES:
        raise ValueError(
            f"formal selection has {len(selected)} images, not {FORMAL_IMAGES}"
        )
    if immutable.get("mode") != "formal":
        raise ValueError("formal analyzer accepts only immutable.mode=formal")
    if immutable.get("score_spec") != score_spec.as_dict():
        raise ValueError("immutable score spec changed")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("immutable selected-row SHA-256 changed")
    return release, selected, rebuilt


def _rebuild_smoke_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    raw_contract = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_value = _require_mapping(
        raw_contract.get("release"),
        "dataset contract release",
    )
    manifest_path = _safe_repo_path(
        release_value.get("manifest_path"),
        repo_root=repo_root,
        label="dataset manifest",
    )
    release = load_canonical_release(repo_root, manifest_path, verify_files=True)
    selection = _require_mapping(
        raw_contract.get("selection"),
        "dataset contract selection",
    )
    spec_value = _require_mapping(
        selection.get("spec"),
        "dataset contract selection spec",
    )
    limit = spec_value.get("per_condition_limit")
    spec, selected_value = balanced_runner.select_mode_inputs(
        release,
        mode="smoke",
        per_condition_limit=limit,
        sample_id=None,
    )
    selected = tuple(selected_value)
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=balanced_runner.SCORE_SPEC,
    )
    if rebuilt.as_dict() != raw_contract:
        raise ValueError("smoke dataset contract does not rebuild exactly")
    if immutable.get("mode") != "smoke":
        raise ValueError("smoke comparator accepts only immutable.mode=smoke")
    if immutable.get("score_spec") != balanced_runner.SCORE_SPEC.as_dict():
        raise ValueError("immutable score spec changed")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("immutable selected-row SHA-256 changed")
    if len(selected) != 7 * int(limit):
        raise ValueError("smoke selection size changed")
    return release, selected, rebuilt


def _validate_source_evidence(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "immutable.source")
    expected_keys = {
        "repo_url",
        "root",
        "commit",
        "paper_era_stable_commit",
        "tracked_dirty",
        "source_files",
        "core_inference_byte_identical_to_paper_era_commit",
    }
    if set(source) != expected_keys:
        raise ValueError("immutable.source key set changed")
    expected = {
        "repo_url": legacy_runner.MODEL_REPO_URL,
        "commit": legacy_runner.MODEL_SOURCE_COMMIT,
        "paper_era_stable_commit": legacy_runner.PAPER_ERA_STABLE_COMMIT,
        "tracked_dirty": False,
        "source_files": legacy_runner.SOURCE_FILES,
        "core_inference_byte_identical_to_paper_era_commit": True,
    }
    for field, expected_value in expected.items():
        if source.get(field) != expected_value:
            raise ValueError(f"immutable.source.{field} changed")
    root = Path(_require_string(source.get("root"), "immutable.source.root"))
    if not root.is_absolute():
        raise ValueError("immutable.source.root is not absolute")
    return source


def _validate_asset_evidence(value: Any) -> dict[str, Any]:
    asset = _require_mapping(value, "immutable.asset")
    expected_keys = set(legacy_runner.CHECKPOINT) | {
        "path",
        "serialization_safety",
        "schema",
    }
    if set(asset) != expected_keys:
        raise ValueError("immutable.asset key set changed")
    for field, expected_value in legacy_runner.CHECKPOINT.items():
        if asset.get(field) != expected_value:
            raise ValueError(f"immutable.asset.{field} changed")
    path = Path(_require_string(asset.get("path"), "immutable.asset.path"))
    if not path.is_absolute():
        raise ValueError("immutable.asset.path is not absolute")
    safety = _require_mapping(
        asset.get("serialization_safety"),
        "immutable.asset.serialization_safety",
    )
    if set(safety) != {
        "weights_only",
        "weights_only_load_succeeded",
        "static_unsafe_global_scan",
        "unrestricted_pickle_used",
    }:
        raise ValueError("immutable.asset serialization safety keys changed")
    if (
        safety.get("weights_only") is not True
        or safety.get("weights_only_load_succeeded") is not True
        or safety.get("unrestricted_pickle_used") is not False
    ):
        raise ValueError("immutable.asset serialization safety changed")
    static_scan = _require_mapping(
        safety.get("static_unsafe_global_scan"),
        "immutable.asset static unsafe-global scan",
    )
    if set(static_scan) not in (
        {"supported", "unsafe_globals"},
        {"supported", "unsafe_globals", "reason"},
    ):
        raise ValueError("immutable.asset static scan keys changed")
    _require_bool(static_scan.get("supported"), "static scan supported")
    if static_scan.get("supported") is True:
        if static_scan.get("unsafe_globals") != []:
            raise ValueError("immutable.asset has unsafe checkpoint globals")
        if "reason" in static_scan:
            raise ValueError("supported static scan must not have a reason")
    else:
        if static_scan.get("unsafe_globals") is not None:
            raise ValueError("unsupported static scan has unsafe globals")
        _require_string(static_scan.get("reason"), "static scan reason")

    schema = _require_mapping(asset.get("schema"), "immutable.asset.schema")
    expected_schema = {
        "outer_type": "dict",
        "outer_keys": list(legacy_runner.CHECKPOINT["outer_keys"]),
        "model_type": "OrderedDict",
        "state_entries": legacy_runner.CHECKPOINT["state_entries"],
        "state_elements": legacy_runner.CHECKPOINT["state_elements"],
        "state_payload_sha256": legacy_runner.CHECKPOINT[
            "state_payload_sha256"
        ],
        "conv1_weight_shape": [64, 3, 7, 7],
        "fc_weight_shape": [1, legacy_runner.FEATURE_DIMENSION],
        "fc_bias_shape": [1],
        "optimizer_state_entries": legacy_runner.CHECKPOINT[
            "optimizer_state_entries"
        ],
        "optimizer_param_groups": legacy_runner.CHECKPOINT[
            "optimizer_param_groups"
        ],
        "total_steps": legacy_runner.CHECKPOINT["total_steps"],
        "strict_model_load": True,
    }
    if schema != expected_schema:
        raise ValueError("immutable.asset checkpoint schema changed")
    return asset


def _validate_runtime_evidence(
    value: Any,
    *,
    label: str,
    expected_device: str | None = None,
) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    base_keys = {
        "device",
        "python",
        "platform",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "seed",
        "dtype",
        "batch_size",
        "autocast",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
    }
    device = _require_string(runtime.get("device"), f"{label}.device")
    expected_keys = base_keys | ({"cuda"} if device.startswith("cuda:") else set())
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} key set changed")
    if expected_device is not None and device != expected_device:
        raise ValueError(f"{label}.device changed")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError(f"{label}.device is unsupported")
    for field in (
        "python",
        "platform",
        "torch",
        "pillow",
        "numpy",
        "cublas_workspace_config",
    ):
        _require_string(runtime.get(field), f"{label}.{field}")
    torchvision = runtime.get("torchvision")
    if torchvision is not None:
        _require_string(torchvision, f"{label}.torchvision")
    expected_values = {
        "seed": legacy_runner.MODEL_SEED,
        "dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "matmul_allow_tf32": False,
    }
    for field, expected_value in expected_values.items():
        if runtime.get(field) != expected_value:
            raise ValueError(f"{label}.{field} changed")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    if set(cudnn) != {
        "enabled",
        "benchmark",
        "deterministic",
        "allow_tf32",
    }:
        raise ValueError(f"{label}.cudnn key set changed")
    for field in cudnn:
        _require_bool(cudnn[field], f"{label}.cudnn.{field}")
    if (
        cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if "cuda" in runtime:
        cuda = _require_mapping(runtime["cuda"], f"{label}.cuda")
        if set(cuda) != {
            "runtime",
            "device_index",
            "device_name",
            "total_memory_bytes",
            "capability",
        }:
            raise ValueError(f"{label}.cuda key set changed")
        _require_string(cuda.get("runtime"), f"{label}.cuda.runtime")
        _require_string(cuda.get("device_name"), f"{label}.cuda.device_name")
        _require_nonnegative_int(
            cuda.get("device_index"),
            f"{label}.cuda.device_index",
        )
        if _require_nonnegative_int(
            cuda.get("total_memory_bytes"),
            f"{label}.cuda.total_memory_bytes",
        ) == 0:
            raise ValueError(f"{label}.cuda.total_memory_bytes is zero")
        capability = cuda.get("capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{label}.cuda.capability changed")
    return runtime


def _validate_cpu_preflight(
    value: Any,
    *,
    source: Mapping[str, Any],
    asset: Mapping[str, Any],
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
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "asset",
        "runtime",
        "golden_reference_runtime",
        "golden_cases",
        "cuda_used",
        "mouse_inference_run",
    }
    if set(report) != expected_keys:
        raise ValueError("CPU preflight report key set changed")
    if (
        report.get("schema_version") != "cnndetection_preflight_v1"
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("asset") != asset
        or report.get("cuda_used") is not False
        or report.get("mouse_inference_run") is not False
    ):
        raise ValueError("CPU preflight report changed")
    _validate_runtime_evidence(
        report.get("runtime"),
        label="CPU preflight runtime",
        expected_device="cpu",
    )
    _require_mapping(
        report.get("golden_reference_runtime"),
        "CPU preflight golden reference runtime",
    )
    cases = report.get("golden_cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("CPU preflight must contain four golden cases")
    expected_cases = {
        (profile, filename)
        for profile in (
            legacy_runner.PRIMARY_PROFILE,
            legacy_runner.PAPER_CROP_PROFILE,
        )
        for filename in ("real.png", "fake.png")
    }
    actual_cases: set[tuple[str, str]] = set()
    case_keys = {
        "profile",
        "filename",
        "input_sha256",
        "tensor_sha256",
        "raw_logit",
        "fake_score",
        "classification_decision",
        "feature_sha256",
    }
    for index, case_value in enumerate(cases):
        case = _require_mapping(
            case_value,
            f"CPU preflight golden case {index}",
        )
        if set(case) != case_keys:
            raise ValueError("CPU preflight golden case key set changed")
        actual_cases.add(
            (
                _require_string(case.get("profile"), "golden profile"),
                _require_string(case.get("filename"), "golden filename"),
            )
        )
        for field in ("input_sha256", "tensor_sha256", "feature_sha256"):
            _require_sha256(case.get(field), f"golden case {field}")
        _require_finite(case.get("raw_logit"), "golden raw_logit")
        score = _require_finite(case.get("fake_score"), "golden fake_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("CPU preflight golden score is not a probability")
        _require_bool(
            case.get("classification_decision"),
            "golden classification_decision",
        )
    if actual_cases != expected_cases:
        raise ValueError("CPU preflight golden case coverage changed")


def _validate_immutable_outputs(value: Any, *, repo_root: Path) -> dict[str, Any]:
    outputs = _require_mapping(value, "immutable.outputs")
    expected_keys = {
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "feature_dir",
    }
    if set(outputs) != expected_keys:
        raise ValueError("immutable.outputs key set changed")
    for field in expected_keys:
        _safe_repo_path(
            outputs.get(field),
            repo_root=repo_root,
            label=f"immutable.outputs.{field}",
            require_file=False,
        )
    return outputs


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str | None = None,
) -> tuple[str, Mapping[str, Any]]:
    run_id = balanced_runner._valid_run_id(run_id)
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
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported CNNDetection Balanced run manifest")
    if manifest.get("run_id") != run_id:
        raise ValueError("manifest run_id mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("formal analyzer requires manifest status complete")
    _require_string(manifest.get("started_at"), "manifest started_at")
    _require_string(manifest.get("completed_at"), "manifest completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest immutable")
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
        "asset",
        "runtime",
        "cpu_preflight",
        "artifact_contract",
        "outputs",
    }
    if set(immutable) != immutable_keys:
        raise ValueError("manifest immutable key set changed")
    if immutable.get("schema_version") != balanced_runner.RUN_CONFIG_SCHEMA:
        raise ValueError("immutable schema_version changed")
    if immutable.get("run_id") != run_id:
        raise ValueError("immutable run_id mismatch")
    mode = immutable.get("mode")
    if mode not in ("formal", "smoke"):
        raise ValueError("immutable mode is unsupported by this analyzer")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"analyzer requires immutable.mode={expected_mode}")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint")
    expected = hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()
    if fingerprint != expected:
        raise ValueError("manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    if immutable.get("score_spec") != balanced_runner.SCORE_SPEC.as_dict():
        raise ValueError("immutable score_spec changed")
    task_scope = {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
    }
    if immutable.get("task_scope") != task_scope:
        raise ValueError("immutable task_scope changed")
    _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    _require_sha256(
        immutable.get("selected_rows_sha256"),
        "immutable.selected_rows_sha256",
    )
    _require_sha256(
        immutable.get("selected_ids_sha256"),
        "immutable.selected_ids_sha256",
    )
    model = _require_mapping(immutable.get("model"), "immutable.model")
    expected_model_keys = {
        "name",
        "slug",
        "architecture",
        "repository",
        "source_commit",
        "checkpoint_id",
        "checkpoint_sha256",
        "checkpoint_bytes",
    }
    if set(model) != expected_model_keys:
        raise ValueError("immutable.model key set changed")
    for key, expected_value in (
        ("name", legacy_runner.MODEL_NAME),
        ("slug", legacy_runner.MODEL_SLUG),
        ("architecture", legacy_runner.MODEL_ARCH),
        ("repository", legacy_runner.MODEL_REPO_URL),
        ("source_commit", legacy_runner.MODEL_SOURCE_COMMIT),
        ("checkpoint_id", legacy_runner.CHECKPOINT["id"]),
        ("checkpoint_sha256", legacy_runner.CHECKPOINT["sha256"]),
        ("checkpoint_bytes", legacy_runner.CHECKPOINT["bytes"]),
    ):
        if model.get(key) != expected_value:
            raise ValueError(f"immutable.model.{key} changed")
    preprocess = _require_mapping(
        immutable.get("preprocess"),
        "immutable.preprocess",
    )
    if set(preprocess) != {
        "profile",
        "contract",
        "batch_size",
        "test_time_blur_or_jpeg",
    }:
        raise ValueError("immutable.preprocess key set changed")
    profile = preprocess.get("profile")
    if profile != legacy_runner.PRIMARY_PROFILE:
        raise ValueError("formal analyzer requires native primary preprocessing")
    if preprocess.get("contract") != legacy_runner.PREPROCESS_PROFILES[profile]:
        raise ValueError("immutable preprocessing contract changed")
    if (
        preprocess.get("batch_size") != 1
        or preprocess.get("test_time_blur_or_jpeg") is not False
    ):
        raise ValueError("immutable preprocessing execution contract changed")
    source = _validate_source_evidence(immutable.get("source"))
    asset = _validate_asset_evidence(immutable.get("asset"))
    _validate_runtime_evidence(
        immutable.get("runtime"),
        label="immutable.runtime",
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        source=source,
        asset=asset,
    )
    expected_artifact_contract = {
        "feature": {
            "format": "NumPy .npy, allow_pickle=False",
            "shape": [legacy_runner.FEATURE_DIMENSION],
            "dtype": "float32",
            "semantics": (
                "official_fc_input_after_adaptive_global_average_pool"
            ),
            "exact_fc_and_sigmoid_replay": True,
        },
    }
    if immutable.get("artifact_contract") != expected_artifact_contract:
        raise ValueError("immutable artifact_contract changed")
    immutable_outputs = _validate_immutable_outputs(
        immutable.get("outputs"),
        repo_root=repo_root,
    )
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    if set(dataset) != {
        "contract",
        "manifest_path",
        "manifest_sha256",
        "expected_inputs_path",
        "expected_inputs_sha256",
        "selected_images",
    }:
        raise ValueError("manifest dataset key set changed")
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    if set(outputs) != {
        *immutable_outputs,
        "results_sha256",
        "summary_sha256",
        "feature_files",
    }:
        raise ValueError("manifest outputs key set changed")
    for field, value in immutable_outputs.items():
        if outputs.get(field) != value:
            raise ValueError(f"manifest outputs {field} changed")
    _require_sha256(outputs.get("results_sha256"), "outputs results SHA-256")
    _require_sha256(outputs.get("summary_sha256"), "outputs summary SHA-256")
    _require_nonnegative_int(outputs.get("feature_files"), "outputs feature_files")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest execution",
    )
    if set(execution) != {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
    }:
        raise ValueError("manifest execution key set changed")
    for field, value in execution.items():
        _require_nonnegative_int(value, f"manifest execution {field}")
    _reject_t2_claims(manifest, "manifest")
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
) -> list[dict[str, Any]]:
    expected = _read_jsonl_strict(expected_path, "expected inputs")
    if expected != list(selected):
        raise ValueError("expected-input snapshot is not the exact formal selection")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    checks = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    for key, expected_value in checks.items():
        if dataset.get(key) != expected_value:
            raise ValueError(f"manifest dataset {key} mismatch")
    return expected


def _expected_output_paths(
    *,
    repo_root: Path,
    run_dir: Path,
) -> dict[str, str]:
    return {
        "results_path": repo_relative(run_dir / "results.jsonl", repo_root),
        "expected_inputs_path": repo_relative(
            run_dir / "expected_inputs.jsonl",
            repo_root,
        ),
        "summary_path": repo_relative(run_dir / "summary.json", repo_root),
        "feature_dir": repo_relative(run_dir / "features", repo_root),
    }


def _validate_execution_accounting(
    *,
    manifest: Mapping[str, Any],
    expected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest execution",
    )
    required = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
    }
    for field, expected in required.items():
        if execution.get(field) != expected:
            raise ValueError(f"manifest execution {field} mismatch")
    if (
        execution.get("new_successes", 0)
        + execution.get("resume_skips", 0)
        != expected_images
    ):
        raise ValueError("manifest execution successful work count mismatch")
    if execution.get("new_errors") != 0:
        raise ValueError("complete manifest execution contains new errors")


def _validate_runtime_summary(
    *,
    summary: Mapping[str, Any],
    run_id: str,
    fingerprint: str,
    mode: str,
    contract: RunDatasetContract,
    coverage: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": balanced_runner.RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": (
            "analyze_cnndetection_balanced.py"
        ),
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": mode,
        "model": legacy_runner.MODEL_NAME,
        "model_slug": legacy_runner.MODEL_SLUG,
        "score_spec": balanced_runner.SCORE_SPEC.as_dict(),
        "dataset_contract": contract.as_dict(),
        "coverage": dict(coverage),
    }
    for field, expected in required.items():
        if summary.get(field) != expected:
            raise ValueError(f"stored run summary {field} mismatch")
    _require_string(summary.get("generated_at"), "run summary generated_at")
    _reject_t2_claims(summary, "summary")
    _reject_nonfinite_numbers(summary, "summary")


def _validate_physical_extensions(
    *,
    physical: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    inputs = {str(row["sample_id"]): row for row in selected}
    for index, row in enumerate(physical):
        sample_id = _require_string(
            row.get("sample_id"),
            f"physical result row {index} sample_id",
        )
        expected = inputs.get(sample_id)
        if expected is None:
            raise ValueError(
                f"physical result row {index} has unexpected sample_id"
            )
        balanced_runner._validate_runner_attempt(
            row,
            input_row=expected,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        _reject_t2_claims(row, f"physical result row {index}")
        _reject_nonfinite_numbers(row, f"physical result row {index}")


def _validate_score_payload(row: Mapping[str, Any], *, sample_id: str) -> None:
    logit = _require_finite(row.get("raw_logit"), f"{sample_id} raw_logit")
    score = _require_finite(row.get("ai_score"), f"{sample_id} ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{sample_id} ai_score falls outside [0, 1]")
    if row.get("probability") != score or row.get("score") != score:
        raise ValueError(f"{sample_id} score aliases differ")
    if row.get("classification_decision") is not (score > 0.5):
        raise ValueError(f"{sample_id} decision disagrees with strict threshold")
    if row.get("classification_threshold") != 0.5:
        raise ValueError(f"{sample_id} classification threshold changed")
    if row.get("classification_threshold_operator") != ">":
        raise ValueError(f"{sample_id} threshold operator changed")
    if row.get("score_semantics") != (
        "official_float32_sigmoid_uncalibrated_fake_score"
    ):
        raise ValueError(f"{sample_id} score semantics changed")
    if row.get("calibrated_probability") is not False:
        raise ValueError(f"{sample_id} score is incorrectly marked calibrated")
    sigmoid = (
        1.0 / (1.0 + math.exp(-logit))
        if logit >= 0.0
        else math.exp(logit) / (1.0 + math.exp(logit))
    )
    if not math.isclose(
        sigmoid,
        score,
        rel_tol=0.0,
        abs_tol=SCORE_ABS_TOLERANCE,
    ):
        raise ValueError(f"{sample_id} sigmoid/logit relationship changed")
    legacy_analyzer._compare_score_fields(
        row,
        replay_logit=logit,
        replay_score=score,
        replay_decision=score > 0.5,
    )


def _validate_runner_extension(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    fingerprint: str,
) -> None:
    sample_id = str(expected["sample_id"])
    gt_kind = expected.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        visibility = {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "basis": "native_rgb_no_resize_no_crop",
                "preprocess_profile": legacy_runner.PRIMARY_PROFILE,
            },
        }
    elif gt_kind in ("all_zero", "not_applicable"):
        visibility = {
            "edit_visibility": "not_applicable",
            "edit_visible_gt_fraction": None,
            "edit_visibility_evidence": {
                "basis": (
                    "authentic_input"
                    if gt_kind == "all_zero"
                    else "fullframe_has_no_local_GT"
                ),
                "preprocess_profile": legacy_runner.PRIMARY_PROFILE,
            },
        }
    else:
        raise ValueError(f"{sample_id} has unsupported GT kind")
    required = {
        "model": legacy_runner.MODEL_NAME,
        "model_slug": legacy_runner.MODEL_SLUG,
        "preprocess_profile": legacy_runner.PRIMARY_PROFILE,
        "checkpoint_id": legacy_runner.CHECKPOINT["id"],
        "config_fingerprint": fingerprint,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        **visibility,
    }
    for field, expected_value in required.items():
        if row.get(field) != expected_value:
            raise ValueError(f"result {sample_id} {field} changed")
    _reject_t2_claims(row, f"result {sample_id}")


def _feature_artifact(
    *,
    row: Mapping[str, Any],
    sample_id: str,
    repo_root: Path,
    feature_dir: Path,
) -> FeatureArtifact:
    feature_path = _safe_repo_path(
        row.get("cnndetection_feature_path"),
        repo_root=repo_root,
        label=f"{sample_id} feature path",
    )
    expected_path = (feature_dir / f"{sample_id}.npy").resolve()
    if feature_path != expected_path:
        raise ValueError(f"{sample_id} feature path is not canonical")
    file_sha = _require_sha256(
        row.get("cnndetection_feature_sha256"),
        f"{sample_id} feature SHA-256",
    )
    if sha256_file(feature_path) != file_sha:
        raise ValueError(f"{sample_id} feature artifact hash mismatch")
    file_bytes = feature_path.stat().st_size
    persisted_bytes = row.get("cnndetection_feature_bytes")
    if (
        isinstance(persisted_bytes, bool)
        or not isinstance(persisted_bytes, int)
        or persisted_bytes != file_bytes
    ):
        raise ValueError(f"{sample_id} feature byte-size metadata mismatch")
    feature = np.load(feature_path, allow_pickle=False)
    if feature.shape != (legacy_runner.FEATURE_DIMENSION,):
        raise ValueError(f"{sample_id} feature shape changed")
    if (
        feature.dtype != np.float32
        or not feature.flags.c_contiguous
        or not np.isfinite(feature).all()
    ):
        raise ValueError(f"{sample_id} feature dtype/content is invalid")
    if row.get("cnndetection_feature_shape") != [
        legacy_runner.FEATURE_DIMENSION
    ]:
        raise ValueError(f"{sample_id} feature shape metadata changed")
    if row.get("cnndetection_feature_dtype") != "float32":
        raise ValueError(f"{sample_id} feature dtype metadata changed")
    if row.get("cnndetection_feature_semantics") != (
        "official_fc_input_after_adaptive_global_average_pool"
    ):
        raise ValueError(f"{sample_id} feature semantics changed")
    array = np.ascontiguousarray(feature)
    return FeatureArtifact(
        sample_id=sample_id,
        path=feature_path,
        file_sha256=file_sha,
        file_bytes=file_bytes,
        array_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        array=array,
    )


def validate_feature_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    feature_dir: Path,
) -> dict[str, FeatureArtifact]:
    expected_ids = [
        _require_string(row.get("sample_id"), "result sample_id")
        for row in latest_results
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("latest results contain duplicate sample_id")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"missing feature directory: {feature_dir}")
    inventory = {
        path.name
        for path in feature_dir.iterdir()
        if path.is_file()
    }
    expected_names = {f"{sample_id}.npy" for sample_id in expected_ids}
    if inventory != expected_names:
        missing = sorted(expected_names - inventory)
        extra = sorted(inventory - expected_names)
        raise ValueError(
            "feature inventory mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    artifacts: dict[str, FeatureArtifact] = {}
    for row in latest_results:
        sample_id = str(row["sample_id"])
        _validate_score_payload(row, sample_id=sample_id)
        artifacts[sample_id] = _feature_artifact(
            row=row,
            sample_id=sample_id,
            repo_root=repo_root,
            feature_dir=feature_dir,
        )
    return artifacts


def _latest_in_selection_order(
    *,
    selected: Sequence[Mapping[str, Any]],
    physical_results: Sequence[Mapping[str, Any]],
    run_id: str,
    fingerprint: str,
    score_spec: ScoreSpec,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=score_spec,
    )
    coverage = summarize_coverage(latest)
    require_complete_coverage(coverage)
    rows = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])])
        for row in selected
    )
    return rows, coverage.as_dict()


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
) -> RunBundle:
    root = repo_root.resolve()
    run_id = balanced_runner._valid_run_id(run_id)
    results_root = _resolve_results_root(results_dir, root)
    run_dir = _resolve_run_dir(results_root, run_id)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "run manifest")
    fingerprint, immutable = _validate_manifest(
        manifest=manifest,
        repo_root=root,
        run_id=run_id,
        expected_mode="formal",
    )
    if immutable.get("outputs") != _expected_output_paths(
        repo_root=root,
        run_dir=run_dir,
    ):
        raise ValueError("immutable output paths do not match formal run dir")
    release, selected, contract = _rebuild_formal_contract(
        repo_root=root,
        immutable=immutable,
    )
    _validate_dataset_artifacts(
        manifest=manifest,
        repo_root=root,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
    )
    physical = tuple(_read_jsonl_strict(results_path, "physical results"))
    score_spec = contract.score_spec
    if score_spec is None:
        raise ValueError("formal run dataset contract has no score spec")
    _validate_physical_extensions(
        physical=physical,
        selected=selected,
        repo_root=root,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    latest, coverage = _latest_in_selection_order(
        selected=selected,
        physical_results=physical,
        run_id=run_id,
        fingerprint=fingerprint,
        score_spec=score_spec,
    )
    _validate_execution_accounting(
        manifest=manifest,
        expected_images=len(selected),
        physical_rows=len(physical),
        latest_rows=len(latest),
    )
    for expected, row in zip(selected, latest, strict=True):
        _validate_runner_extension(
            row,
            expected=expected,
            fingerprint=fingerprint,
        )
    summary = _load_json(summary_path, "run summary")
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    expected_outputs = {
        "results_path": repo_relative(results_path, root),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "summary_sha256": sha256_file(summary_path),
        "feature_dir": repo_relative(run_dir / "features", root),
        "feature_files": len(selected),
    }
    for key, expected_value in expected_outputs.items():
        if outputs.get(key) != expected_value:
            raise ValueError(f"manifest outputs {key} mismatch")
    _validate_runtime_summary(
        summary=summary,
        run_id=run_id,
        fingerprint=fingerprint,
        mode="formal",
        contract=contract,
        coverage=coverage,
    )
    features = validate_feature_inventory(
        latest_results=latest,
        repo_root=root,
        feature_dir=run_dir / "features",
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest=manifest,
        release=release,
        selected=selected,
        contract=contract,
        physical_results=physical,
        latest_results=latest,
        coverage=coverage,
        features=features,
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
) -> RunBundle:
    root = repo_root.resolve()
    run_id = balanced_runner._valid_run_id(run_id)
    results_root = _resolve_results_root(results_dir, root)
    run_dir = _resolve_run_dir(results_root, run_id)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "smoke run manifest")
    fingerprint, immutable = _validate_manifest(
        manifest=manifest,
        repo_root=root,
        run_id=run_id,
        expected_mode="smoke",
    )
    if immutable.get("outputs") != _expected_output_paths(
        repo_root=root,
        run_dir=run_dir,
    ):
        raise ValueError("immutable output paths do not match smoke run dir")
    release, selected, contract = _rebuild_smoke_contract(
        repo_root=root,
        immutable=immutable,
    )
    _validate_dataset_artifacts(
        manifest=manifest,
        repo_root=root,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
    )
    physical = tuple(_read_jsonl_strict(results_path, "smoke physical results"))
    score_spec = contract.score_spec
    if score_spec is None:
        raise ValueError("smoke run dataset contract has no score spec")
    _validate_physical_extensions(
        physical=physical,
        selected=selected,
        repo_root=root,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    latest, coverage = _latest_in_selection_order(
        selected=selected,
        physical_results=physical,
        run_id=run_id,
        fingerprint=fingerprint,
        score_spec=score_spec,
    )
    _validate_execution_accounting(
        manifest=manifest,
        expected_images=len(selected),
        physical_rows=len(physical),
        latest_rows=len(latest),
    )
    if len(physical) != len(selected):
        raise ValueError(
            "smoke comparison requires exactly one physical attempt per input"
        )
    for expected, row in zip(selected, latest, strict=True):
        _validate_runner_extension(
            row,
            expected=expected,
            fingerprint=fingerprint,
        )
    summary = _load_json(summary_path, "smoke run summary")
    outputs = _require_mapping(manifest.get("outputs"), "smoke manifest outputs")
    expected_outputs = {
        "results_path": repo_relative(results_path, root),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_path": repo_relative(expected_path, root),
        "summary_path": repo_relative(summary_path, root),
        "summary_sha256": sha256_file(summary_path),
        "feature_dir": repo_relative(run_dir / "features", root),
        "feature_files": len(selected),
    }
    for key, expected_value in expected_outputs.items():
        if outputs.get(key) != expected_value:
            raise ValueError(f"smoke manifest outputs {key} mismatch")
    _validate_runtime_summary(
        summary=summary,
        run_id=run_id,
        fingerprint=fingerprint,
        mode="smoke",
        contract=contract,
        coverage=coverage,
    )
    features = validate_feature_inventory(
        latest_results=latest,
        repo_root=root,
        feature_dir=run_dir / "features",
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest=manifest,
        release=release,
        selected=selected,
        contract=contract,
        physical_results=physical,
        latest_results=latest,
        coverage=coverage,
        features=features,
    )


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
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
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("shared Balanced250 metrics schema changed")
    if metrics.get("coverage", {}).get("is_complete") is not True:
        raise ValueError("formal Balanced250 metrics are incomplete")
    return metrics


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    output_path: Path | None,
) -> dict[str, Any]:
    reference_run_id = balanced_runner._valid_run_id(reference_run_id)
    replay_run_id = balanced_runner._valid_run_id(replay_run_id)
    if reference_run_id == replay_run_id:
        raise ValueError("smoke comparison requires two distinct run IDs")
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
    if reference.selected != replay.selected:
        raise ValueError("smoke runs do not use the same canonical selection")
    if (
        reference.contract.selection.as_dict()
        != replay.contract.selection.as_dict()
    ):
        raise ValueError("smoke run selection contracts differ")
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_features=reference.features,
        replay_features=replay.features,
        exact=True,
    )
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_smoke_comparison_passed",
        "compared_at": utc_now(),
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            "manifest_sha256": sha256_file(reference.manifest_path),
            "results_sha256": sha256_file(reference.results_path),
            "expected_inputs_sha256": sha256_file(reference.expected_path),
            "summary_sha256": sha256_file(reference.summary_path),
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            "manifest_sha256": sha256_file(replay.manifest_path),
            "results_sha256": sha256_file(replay.results_path),
            "expected_inputs_sha256": sha256_file(replay.expected_path),
            "summary_sha256": sha256_file(replay.summary_path),
        },
        "selection": reference.contract.selection.as_dict(),
        "comparison": comparison,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def _projection(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("computational projection requires one successful row")
    sample_id = _require_string(row.get("sample_id"), "result sample_id")
    projected = {}
    for field in _RESULT_PROJECTION_FIELDS:
        if field not in row:
            raise ValueError(f"result {sample_id} lacks projection field {field}")
        projected[field] = row[field]
    if projected["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"result {sample_id} schema changed")
    _validate_score_payload(row, sample_id=sample_id)
    return projected


def _exact_smoke_row_projection(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("smoke comparison requires one successful row")
    sample_id = _require_string(row.get("sample_id"), "result sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    missing_ignored = _SMOKE_ROW_IGNORED_FIELDS - set(row)
    if missing_ignored:
        raise ValueError(
            f"result {sample_id} lacks ignored runtime field "
            f"{sorted(missing_ignored)[0]}"
        )
    return {
        key: value
        for key, value in row.items()
        if key not in _SMOKE_ROW_IGNORED_FIELDS
    }


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_features: Mapping[str, FeatureArtifact],
    replay_features: Mapping[str, FeatureArtifact],
    exact: bool,
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
                raise ValueError(f"{label} contains duplicate sample_id {sample_id}")
            result[sample_id] = row
        if not result:
            raise ValueError(f"{label} is empty")
        return result

    reference = unique(reference_rows, "reference results")
    replay = unique(replay_rows, "replay results")
    if set(reference) != set(replay):
        raise ValueError("reference/replay result coverage differs")
    if set(reference_features) != set(reference):
        raise ValueError("reference feature coverage differs")
    if set(replay_features) != set(replay):
        raise ValueError("replay feature coverage differs")

    max_logit = 0.0
    max_score = 0.0
    max_feature = 0.0
    for sample_id in sorted(reference):
        left = reference[sample_id]
        right = replay[sample_id]
        if exact:
            left_projection = _exact_smoke_row_projection(left)
            right_projection = _exact_smoke_row_projection(right)
            if left_projection != right_projection:
                differing = sorted(
                    {
                        *(
                            set(left_projection)
                            ^ set(right_projection)
                        ),
                        *(
                            key
                            for key in set(left_projection)
                            & set(right_projection)
                            if left_projection[key] != right_projection[key]
                        ),
                    }
                )
                raise ValueError(
                    f"result {sample_id} full computational row differs at "
                    f"{differing[:3]}"
                )
        else:
            left_projection = _projection(left)
            right_projection = _projection(right)
            for field in _RESULT_PROJECTION_FIELDS:
                if field in ("raw_logit", "probability", "ai_score", "score"):
                    continue
                if left_projection[field] != right_projection[field]:
                    raise ValueError(f"result {sample_id} {field} differs")
            logit_difference = abs(
                float(left["raw_logit"]) - float(right["raw_logit"])
            )
            score_difference = abs(
                float(left["ai_score"]) - float(right["ai_score"])
            )
            max_logit = max(max_logit, logit_difference)
            max_score = max(max_score, score_difference)
            if logit_difference > RAW_LOGIT_ABS_TOLERANCE:
                raise ValueError(f"result {sample_id} raw_logit replay mismatch")
            if score_difference > SCORE_ABS_TOLERANCE:
                raise ValueError(f"result {sample_id} ai_score replay mismatch")

        left_feature = reference_features[sample_id]
        right_feature = replay_features[sample_id]
        if (
            left_feature.array.shape != right_feature.array.shape
            or left_feature.array.dtype != right_feature.array.dtype
        ):
            raise ValueError(f"result {sample_id} feature metadata differs")
        difference = float(
            np.max(
                np.abs(
                    left_feature.array.astype(np.float64)
                    - right_feature.array.astype(np.float64)
                )
            )
        )
        max_feature = max(max_feature, difference)
        if exact:
            if (
                left_feature.file_bytes != right_feature.file_bytes
                or left_feature.file_sha256 != right_feature.file_sha256
                or left_feature.path.read_bytes()
                != right_feature.path.read_bytes()
            ):
                raise ValueError(f"result {sample_id} feature bytes differ")
        elif difference > FEATURE_ABS_TOLERANCE:
            raise ValueError(f"result {sample_id} feature replay mismatch")
    return {
        "images_compared": len(reference),
        "identity_fields_ignored": sorted(
            _RUN_IDENTITY_FIELDS
            & _SMOKE_ROW_IGNORED_FIELDS
        ),
        "volatile_fields_ignored": sorted(
            _SMOKE_ROW_IGNORED_FIELDS - _RUN_IDENTITY_FIELDS
        ),
        "exact_computational_projection": exact,
        "max_raw_logit_abs_difference": max_logit,
        "max_ai_score_abs_difference": max_score,
        "max_feature_abs_difference": max_feature,
        "feature_shape_dtype_verified": True,
        "feature_file_sha256_and_bytes_verified": exact,
    }


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    checkpoint_path: Path,
    device_text: str,
) -> dict[str, Any]:
    source, asset, state, module = legacy_runner.verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
    immutable = _require_mapping(
        bundle.manifest.get("immutable"),
        "manifest immutable",
    )
    if immutable.get("source") != source or immutable.get("asset") != asset:
        raise ValueError("source/checkpoint provenance differs from manifest")
    device, runtime = legacy_runner.configure_runtime(device_text)
    model = legacy_runner.load_model(module=module, state=state, device=device)
    replayed = 0
    max_logit = 0.0
    max_score = 0.0
    max_feature = 0.0
    try:
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
            tensor, preprocess = legacy_analyzer.independent_preprocess_image(
                input_path,
                legacy_runner.PRIMARY_PROFILE,
            )
            if row.get("preprocess") != preprocess:
                raise ValueError(f"{sample_id} persisted preprocessing differs")
            logit, score, decision, feature = legacy_analyzer._independent_infer(
                model,
                tensor,
                device,
            )
            stored_logit = _require_finite(
                row.get("raw_logit"),
                f"{sample_id} raw_logit",
            )
            stored_score = _require_finite(
                row.get("ai_score"),
                f"{sample_id} ai_score",
            )
            logit_difference = abs(stored_logit - logit)
            score_difference = abs(stored_score - score)
            feature_difference = float(
                np.max(
                    np.abs(
                        bundle.features[sample_id].array.astype(np.float64)
                        - feature.astype(np.float64)
                    )
                )
            )
            max_logit = max(max_logit, logit_difference)
            max_score = max(max_score, score_difference)
            max_feature = max(max_feature, feature_difference)
            if logit_difference > RAW_LOGIT_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} raw logit fresh replay mismatch")
            if score_difference > SCORE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} score fresh replay mismatch")
            if row.get("classification_decision") is not decision:
                raise ValueError(f"{sample_id} decision fresh replay mismatch")
            if feature_difference > FEATURE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} feature fresh replay mismatch")
            import torch

            with torch.inference_mode():
                persisted_feature = torch.from_numpy(
                    bundle.features[sample_id].array
                ).to(device=device, dtype=torch.float32)
                persisted_logit = float(
                    torch.nn.functional.linear(
                        persisted_feature.unsqueeze(0),
                        model.fc.weight,
                        model.fc.bias,
                    ).reshape(()).item()
                )
                persisted_score = float(
                    torch.sigmoid(
                        torch.tensor(
                            persisted_logit,
                            device=device,
                            dtype=torch.float32,
                        )
                    ).item()
                )
            if not math.isclose(
                persisted_logit,
                stored_logit,
                rel_tol=0.0,
                abs_tol=RAW_LOGIT_ABS_TOLERANCE,
            ):
                raise ValueError(
                    f"{sample_id} saved feature/fresh FC logit mismatch"
                )
            if not math.isclose(
                persisted_score,
                stored_score,
                rel_tol=0.0,
                abs_tol=SCORE_ABS_TOLERANCE,
            ):
                raise ValueError(
                    f"{sample_id} saved feature/fresh sigmoid mismatch"
                )
            replayed += 1
    finally:
        del model
        del state
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()
    if replayed != len(bundle.selected):
        raise ValueError("fresh replay did not cover every formal input")
    return {
        "images_replayed": replayed,
        "source_commit": source["commit"],
        "checkpoint_sha256": asset["sha256"],
        "runtime": runtime,
        "raw_logit_abs_tolerance": RAW_LOGIT_ABS_TOLERANCE,
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "feature_abs_tolerance": FEATURE_ABS_TOLERANCE,
        "max_raw_logit_abs_difference": max_logit,
        "max_ai_score_abs_difference": max_score,
        "max_feature_abs_difference": max_feature,
        "saved_feature_fresh_fc_logit_verified": True,
        "saved_feature_fresh_sigmoid_verified": True,
    }


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    checkpoint_path: Path,
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
    metrics = recompute_metrics(bundle, iterations=iterations, seed=seed)
    if metrics_output_path is not None:
        atomic_write_json(metrics_output_path, metrics)
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
        "fresh_model_replay": replay_report,
        "contract_checks": {
            "exact_formal_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "result_identity_run_id_fingerprint_status_validated": True,
            "feature_inventory_sha256_shape_dtype_finiteness_validated": True,
            "balanced250_primary_and_secondary_metrics_recomputed": True,
            "localization_claims_made": False,
        },
        "artifacts": {
            "manifest_sha256": sha256_file(bundle.manifest_path),
            "results_sha256": sha256_file(bundle.results_path),
            "expected_inputs_sha256": sha256_file(bundle.expected_path),
            "summary_sha256": sha256_file(bundle.summary_path),
            "metrics_sha256": (
                sha256_file(metrics_output_path)
                if metrics_output_path is not None
                else None
            ),
        },
    }
    if audit_output_path is not None:
        atomic_write_json(audit_output_path, audit)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument(
        "--compare-smoke-run-id",
        help=(
            "validate --run-id and this second smoke run, then write an "
            "exact computational comparison instead of formal metrics"
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
    run_id = balanced_runner._valid_run_id(args.run_id)
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    run_dir = _resolve_run_dir(results_dir, run_id)
    if args.compare_smoke_run_id is not None:
        compare_run_id = balanced_runner._valid_run_id(
            args.compare_smoke_run_id
        )
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
        ):
            raise ValueError(
                "smoke comparison cannot be combined with formal audit options"
            )
        comparison_output = (
            _anchored(args.comparison_output, repo_root)
            if args.comparison_output is not None
            else results_dir
            / (
                f"{run_id}__vs__{compare_run_id}"
                "_comparison.json"
            )
        )
        if args.comparison_output is None:
            resolved_comparison = comparison_output.resolve()
            try:
                resolved_comparison.relative_to(results_dir)
            except ValueError as error:
                raise ValueError(
                    "default comparison output escapes results root"
                ) from error
            comparison_output = resolved_comparison
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_run_id,
            output_path=comparison_output,
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
        checkpoint_path=_anchored(args.checkpoint, repo_root),
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
