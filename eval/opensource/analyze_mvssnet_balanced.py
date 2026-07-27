#!/usr/bin/env python3
"""Fail-closed audit, metrics, smoke comparison, and replay for MVSS-Net.

The frozen Balanced250 runner persists the official 512x512 segmentation
logits, their float32 sigmoid map, the official native-resolution uint8 PNG,
and (for real/local inputs only) the strict ``probability > 0.5`` mask.
Full-frame edits retain diagnostic dense maps but are localization-N/A.

This analyzer treats every persisted envelope and artifact as untrusted.  It
rebuilds the exact 1,775/1,025 selection, independently reopens and
preprocesses every image, replays every persisted raster transformation and
localization statistic, computes only the shared Balanced250 T1/T2 summaries,
requires exact frozen A/B smoke reproduction, and by default reloads one fresh
model for an ordered 1,775-forward replay.  A single model is deliberately
used for the whole sequence because the official Bayar constraint normalizes
its kernel in place on every forward.
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
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from eval.opensource import run_mvssnet as legacy
from eval.opensource import run_mvssnet_balanced as runner
from eval.opensource.balanced250_localization_metrics import (
    summarize_balanced250_t2,
)
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
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
    load_ground_truth,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "mvssnet_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "mvssnet_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "mvssnet_balanced250_summary_v2"
T1_METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
T2_METRICS_SCHEMA_VERSION = "balanced250_t2_summary_v1"

DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = runner.DEFAULT_ARTIFACTS_DIR
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
DEFAULT_MVSSNET_ROOT = legacy.DEFAULT_MVSSNET_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT

FORMAL_IMAGES = 1_775
FORMAL_T2_IMAGES = 1_025
SMOKE_IMAGES = 35
SMOKE_T2_IMAGES = 20
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_726
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

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
SMOKE_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

EXPECTED_ADAPTER_SOURCE_PATHS = runner.ADAPTER_SOURCE_PATHS
EXPECTED_IMMUTABLE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "adapter_sources_sha256",
        "model",
        "preprocess",
        "inference",
        "score_spec",
        "t2_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "source",
        "checkpoint",
        "environment",
        "checkpoint_audit",
        "model_audit",
        "mouse_reference",
        "license",
        "runtime",
        "cpu_preflight",
        "artifact_contract",
        "outputs",
    }
)
EXPECTED_MANIFEST_KEYS = frozenset(
    {
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
        "execution",
    }
)
EXPECTED_EXECUTION_KEYS = frozenset(
    {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
        "stateful_prefix_replayed",
    }
)
EXPECTED_ARTIFACT_INVENTORY = runner.ARTIFACT_DIRECTORIES
_T2_GT_KINDS = frozenset({"all_zero", "exact_diff"})
_FULLFRAME_GT_KIND = "not_applicable"
_RUN_SPECIFIC_RESULT_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "raw_logits_model_path",
        "score_map_model_path",
        "score_map_native_path",
        "mask_path",
        "artifact_paths",
    }
)


@dataclass(frozen=True)
class DenseArtifacts:
    """Independently validated artifacts for one terminal successful result."""

    sample_id: str
    raw_logits_path: Path
    raw_logits_file_sha256: str
    raw_logits_array_sha256: str
    model_score_path: Path
    model_score_file_sha256: str
    model_score_array_sha256: str
    native_score_path: Path
    native_score_file_sha256: str
    native_score_array_sha256: str
    mask_path: Path | None
    mask_file_sha256: str | None
    mask_array_sha256: str | None
    t2_applicable: bool
    width: int
    height: int


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    fingerprint: str
    mode: str
    run_dir: Path
    artifact_root: Path
    manifest_path: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest: Mapping[str, Any]
    immutable: Mapping[str, Any]
    release: CanonicalRelease
    selected: tuple[dict[str, Any], ...]
    contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    coverage: Mapping[str, Any]
    stateful_history: Mapping[str, Any]
    artifacts: Mapping[str, DenseArtifacts]
    evidence_snapshot: Mapping[str, str]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return digest


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return _require_mapping(value, label)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"{label}:{line_number} lacks final newline")
                if not line.strip():
                    raise ValueError(f"{label}:{line_number} is blank")
                row = _require_mapping(
                    json.loads(
                        line,
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_json_constant,
                    ),
                    f"{label}:{line_number}",
                )
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{label}:{line_number} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return rows


def _valid_run_id(value: Any) -> str:
    return runner._valid_run_id(value)


def _lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    candidate = path if path.is_absolute() else (base or Path.cwd()) / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")


def _safe_standard_root(
    path: Path,
    *,
    repo_root: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    root = repo_root.resolve()
    candidate = _lexical_absolute(path, base=root)
    expected = _lexical_absolute(expected_relative, base=root)
    _reject_symlink_components(candidate, label)
    _reject_symlink_components(expected, f"expected {label}")
    if candidate.resolve() != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected_relative}")
    return expected.resolve()


def _resolve_run_dir(root: Path, run_id: Any, label: str) -> Path:
    safe_id = _valid_run_id(run_id)
    candidate = root / safe_id
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve()
    if resolved.parent != root.resolve():
        raise ValueError(f"{label} escapes its standard root")
    return resolved


def _safe_repo_file(
    value: Any,
    *,
    repo_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    relative = _require_string(value, label)
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} is non-canonical or traversing")
    candidate = _lexical_absolute(Path(relative), base=repo_root)
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve()
    if resolved != expected_path.resolve():
        raise ValueError(f"{label} is not the canonical path")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {resolved}")
    return resolved


def _score_spec() -> ScoreSpec:
    value = runner.SCORE_SPEC
    if not isinstance(value, ScoreSpec) or value.as_dict() != {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": ">",
    }:
        raise ValueError("MVSS-Net runner SCORE_SPEC changed")
    return value


def _selection_for_mode(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
) -> tuple[SelectionSpec, tuple[dict[str, Any], ...]]:
    spec, selected = runner.select_mode_inputs(
        release,
        mode=mode,
        per_condition_limit=per_condition_limit,
        sample_id=None,
    )
    if (
        not isinstance(spec, SelectionSpec)
        or spec.capability is not Capability.LOCAL_T1_T2
    ):
        raise ValueError("MVSS-Net selection capability changed")
    rows = tuple(dict(row) for row in selected)
    counts = Counter(str(row["condition"]) for row in rows)
    t2_count = sum(row.get("gt_mask_kind") in _T2_GT_KINDS for row in rows)
    if mode == "formal":
        if (
            len(rows) != FORMAL_IMAGES
            or dict(counts) != FORMAL_COUNTS
            or t2_count != FORMAL_T2_IMAGES
            or _rows_sha256(rows) != FORMAL_SELECTED_ROWS_SHA256
            or selected_ids_sha256(str(row["sample_id"]) for row in rows)
            != FORMAL_SELECTED_IDS_SHA256
        ):
            raise ValueError("formal MVSS-Net selection drifted")
    elif mode == "smoke":
        if (
            len(rows) != SMOKE_IMAGES
            or counts
            != Counter(
                {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
            )
            or t2_count != SMOKE_T2_IMAGES
            or selected_ids_sha256(str(row["sample_id"]) for row in rows)
            != SMOKE_SELECTED_IDS_SHA256
        ):
            raise ValueError("smoke MVSS-Net selection drifted")
    else:
        raise ValueError("analyzer only accepts formal or smoke runs")
    return spec, rows


def _verify_adapter_sources(
    value: Any,
    *,
    aggregate: Any,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    sources = _require_mapping(value, "immutable.adapter_sources")
    if tuple(sources) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("MVSS-Net adapter source order/set changed")
    independently_verified: dict[str, dict[str, Any]] = {}
    for relative in EXPECTED_ADAPTER_SOURCE_PATHS:
        record = _require_mapping(sources.get(relative), f"adapter source {relative}")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} metadata changed")
        path = _safe_repo_file(
            record.get("path"),
            repo_root=repo_root,
            expected_path=repo_root / relative,
            label=f"adapter source {relative}",
        )
        expected = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if record != expected:
            raise ValueError(f"adapter source {relative} content changed")
        independently_verified[relative] = expected
    if aggregate != _fingerprint(independently_verified):
        raise ValueError("MVSS-Net adapter aggregate SHA-256 changed")
    return independently_verified


def _validate_runtime(value: Any, *, label: str) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    keys = {
        "device",
        "device_type",
        "gpu_name",
        "gpu_compute_capability",
        "gpu_total_memory_bytes",
        "torch_version",
        "torch_cuda_version",
        "precision",
        "batch_size",
        "autocast",
        "apex",
        "seed",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_tf32",
        "cudnn_tf32",
        "cublas_workspace_config",
        "contract_sha256",
    }
    if set(runtime) != keys:
        raise ValueError(f"{label} key set changed")
    device = _require_string(runtime.get("device"), f"{label}.device")
    suffix = device.removeprefix("cuda:")
    if (
        not device.startswith("cuda:")
        or not suffix.isdigit()
        or str(int(suffix)) != suffix
        or runtime.get("device_type") != "cuda"
    ):
        raise ValueError(f"{label}.device is not an explicit canonical CUDA device")
    if (
        not _require_string(runtime.get("gpu_name"), f"{label}.gpu_name")
        or _require_nonnegative_int(
            runtime.get("gpu_total_memory_bytes"),
            f"{label}.gpu_total_memory_bytes",
        )
        == 0
    ):
        raise ValueError(f"{label} GPU metadata changed")
    capability = runtime.get("gpu_compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in capability
        )
    ):
        raise ValueError(f"{label}.gpu_compute_capability changed")
    expected = {
        "torch_version": runner.EXPECTED_PACKAGES["torch"],
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "apex": False,
        "seed": runner.MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cublas_workspace_config": runner.CUBLAS_WORKSPACE_CONFIG,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(f"{label}.{key} changed")
    _require_string(runtime.get("torch_cuda_version"), f"{label}.torch_cuda_version")
    contract_sha = _require_sha256(
        runtime.get("contract_sha256"),
        f"{label}.contract_sha256",
    )
    unsigned = {
        key: value for key, value in runtime.items() if key != "contract_sha256"
    }
    if _fingerprint(unsigned) != contract_sha:
        raise ValueError(f"{label} contract SHA-256 changed")
    _reject_nonfinite(runtime, label)
    return runtime


def _independent_source_record(mvssnet_root: Path) -> dict[str, Any]:
    root = mvssnet_root.resolve()
    if root.name != "MVSS-Net" or not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe MVSS-Net source root: {root}")
    commit = legacy._git_value(root, "rev-parse", "HEAD")
    status = legacy._git_value(root, "status", "--short", "--untracked-files=all")
    if commit != legacy.MODEL_SOURCE_COMMIT or status is None or status:
        raise ValueError("live MVSS-Net source commit/worktree changed")
    bindings: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha) in runner.MVSSNET_SOURCE_FILES.items():
        path = root / relative
        _reject_symlink_components(path, f"MVSS-Net source {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
            or legacy._git_value(root, "ls-files", "--error-unmatch", relative)
            != relative
        ):
            raise ValueError(f"MVSS-Net source-bound file changed: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
            "git_tracked": True,
        }
    value = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bindings,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_checkpoint_record(checkpoint_path: Path) -> dict[str, Any]:
    path = checkpoint_path.resolve()
    _reject_symlink_components(path, "MVSS-Net checkpoint")
    if (
        path.name != "mvssnet_casia.pt"
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != runner.CHECKPOINT_BYTES
        or sha256_file(path) != legacy.CHECKPOINT_SHA256
    ):
        raise ValueError("live MVSS-Net checkpoint changed")
    value = {
        "path": str(path),
        "filename": path.name,
        "bytes": runner.CHECKPOINT_BYTES,
        "sha256": legacy.CHECKPOINT_SHA256,
        "provider": "official_author_google_drive",
        "drive_file_id": legacy.CHECKPOINT_DRIVE_ID,
        "format": "raw_collections_OrderedDict_state_dict",
        "weights_only": True,
        "strict_model_load": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_environment_record() -> dict[str, Any]:
    import cv2

    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    pyvenv = prefix / "pyvenv.cfg"
    versions: dict[str, str | None] = {}
    for name in runner.EXPECTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if (
        executable != runner.EXPECTED_PYTHON_EXECUTABLE
        or prefix != runner.EXPECTED_VENV_ROOT
        or platform.python_version() != "3.12.3"
        or not pyvenv.is_file()
        or pyvenv.is_symlink()
        or pyvenv.stat().st_size != runner.EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != runner.EXPECTED_PYVENV_SHA256
        or versions != runner.EXPECTED_PACKAGES
        or cv2.__version__ != runner.EXPECTED_CV2_VERSION
    ):
        raise ValueError("live MVSS-Net analysis environment changed")
    value = {
        "python_executable": str(executable),
        "python_prefix": str(prefix),
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "bytes": runner.EXPECTED_PYVENV_BYTES,
            "sha256": runner.EXPECTED_PYVENV_SHA256,
            "include_system_site_packages": True,
        },
        "packages": versions,
        "cv2_import": {
            "version": cv2.__version__,
            "path": str(Path(cv2.__file__).resolve()),
        },
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_mouse_reference(repo_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative, (
        expected_bytes,
        expected_sha,
    ) in runner.MOUSE_REFERENCE_FILES.items():
        path = repo_root / relative
        _reject_symlink_components(path, f"MVSS-Net Mouse reference {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"MVSS-Net Mouse reference changed: {relative}")
        files[relative] = {"bytes": expected_bytes, "sha256": expected_sha}
    value = {
        "run_id": "mvssnet_casia_mouse_canonical_v1_full275_20260723",
        "expected_tasks": 275,
        "expected_images": 550,
        "role": "protocol_and_regression_anchor_only_not_score_based_selection",
        "files": files,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _expected_checkpoint_audit() -> dict[str, Any]:
    value = {
        "outer_type": "collections.OrderedDict",
        "state_dict_tensors": runner.CHECKPOINT_STATE_KEYS,
        "state_dict_elements": runner.CHECKPOINT_STATE_ELEMENTS,
        "dtype_counts": {
            "torch.float32": runner.CHECKPOINT_FLOAT32_TENSORS,
            "torch.int64": runner.CHECKPOINT_INT64_TENSORS,
        },
        "ordered_keys_sha256": runner.CHECKPOINT_ORDERED_KEYS_SHA256,
        "tensor_schema_sha256": runner.CHECKPOINT_TENSOR_SCHEMA_SHA256,
        "all_floating_tensors_finite": True,
        "static_unsafe_globals": list(runner.CHECKPOINT_UNSAFE_GLOBALS),
        "weights_only": True,
        "map_location": "cpu",
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _expected_model_audit() -> dict[str, Any]:
    value = {
        "construction_device": "cpu",
        "constructor": {
            "backbone": "resnet50",
            "pretrained_base": True,
            "nclass": 1,
            "sobel": True,
            "constrain": True,
            "n_input": 3,
            "external_ImageNet_downloads_suppressed": True,
            "complete_checkpoint_replaces_constructor_state": True,
        },
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "state_key_set_matches_checkpoint": True,
        "model_ordered_keys_sha256": runner.MODEL_ORDERED_KEYS_SHA256,
        "checkpoint_and_model_key_order_differ": True,
        "eval_mode": True,
        "parameter_count": runner.EXPECTED_MODEL_PARAMETERS,
        "trainable_parameter_count": runner.EXPECTED_TRAINABLE_PARAMETERS,
        "buffer_elements": runner.EXPECTED_MODEL_BUFFERS,
        "module_count": runner.EXPECTED_MODEL_MODULES,
        "forward_performed": False,
        "bayar_constraint_mutates_kernel_per_forward": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _validate_provenance(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    adapters = _verify_adapter_sources(
        immutable.get("adapter_sources"),
        aggregate=immutable.get("adapter_sources_sha256"),
        repo_root=repo_root,
    )
    source = _independent_source_record(mvssnet_root)
    checkpoint = _independent_checkpoint_record(checkpoint_path)
    environment = _independent_environment_record()
    mouse_reference = _independent_mouse_reference(repo_root)
    checkpoint_audit = _expected_checkpoint_audit()
    model_audit = _expected_model_audit()
    expected_records = {
        "source": source,
        "checkpoint": checkpoint,
        "environment": environment,
        "mouse_reference": mouse_reference,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
    }
    for key, expected in expected_records.items():
        if immutable.get(key) != expected:
            raise ValueError(f"recorded/live MVSS-Net {key} evidence differs")
    if immutable.get("license") != runner.LICENSE_RECORD:
        raise ValueError("MVSS-Net license declaration changed")
    wrapper = _require_mapping(
        immutable.get("cpu_preflight"),
        "immutable.cpu_preflight",
    )
    if (
        set(wrapper) != {"performed_before_accelerator_configuration", "report"}
        or wrapper.get("performed_before_accelerator_configuration") is not True
    ):
        raise ValueError("MVSS-Net CPU preflight ordering changed")
    report = _require_mapping(wrapper.get("report"), "immutable.cpu_preflight.report")
    expected_report_unsigned = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "environment": environment,
        "source": source,
        "checkpoint": checkpoint,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "adapter_sources": adapters,
        "adapter_sources_sha256": _fingerprint(adapters),
        "mouse_reference": mouse_reference,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "balanced_scores_computed": False,
        "model_forward_performed": False,
    }
    expected_report = {
        **expected_report_unsigned,
        "contract_sha256": _fingerprint(expected_report_unsigned),
    }
    if report != expected_report:
        raise ValueError("MVSS-Net recorded CPU preflight changed")
    return {
        "adapter_sources": adapters,
        **expected_records,
        "license": runner.LICENSE_RECORD,
        "cpu_preflight_ordering": "before_accelerator_configuration",
    }


def independent_structural_golden(
    *,
    checkpoint_path: Path,
    mvssnet_root: Path,
    recorded_checkpoint_audit: Mapping[str, Any],
    recorded_model_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat the strict CPU load without a model forward or CUDA call."""

    import torch

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise RuntimeError("MVSS-Net structural golden started after CUDA init")
    checkpoint, model = runner._build_cpu_model_audit(
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after:
        raise RuntimeError("MVSS-Net structural golden initialized CUDA")
    if checkpoint != dict(recorded_checkpoint_audit) or model != dict(
        recorded_model_audit
    ):
        raise ValueError("MVSS-Net independent CPU strict-load replay changed")
    return {
        "status": "independent_cpu_structural_golden_passed",
        "kind": "checkpoint_schema_and_strict_model_load_no_forward",
        "author_published_numerical_golden": None,
        "author_published_numerical_golden_available": False,
        "reason": (
            "official MVSS-Net release provides checkpoint/inference code but "
            "no frozen numerical output fixture for this checkpoint"
        ),
        "construction_device": "cpu",
        "model_forwards": 0,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "checkpoint_audit_sha256": _fingerprint(checkpoint),
        "model_audit_sha256": _fingerprint(model),
        "executable_numeric_gates": [
            "frozen_smoke_A_B_exact_reproduction",
            "formal_full_stateful_sequence_exact_fresh_replay",
        ],
    }


def _rebuild_contract(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_mode: str,
) -> tuple[
    CanonicalRelease,
    tuple[dict[str, Any], ...],
    RunDatasetContract,
]:
    raw = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_binding = _require_mapping(
        raw.get("release"),
        "dataset contract release",
    )
    manifest_path = _safe_repo_file(
        release_binding.get("manifest_path"),
        repo_root=repo_root,
        expected_path=repo_root / runner.DEFAULT_DATASET_MANIFEST,
        label="Balanced250 dataset manifest",
    )
    release = load_canonical_release(
        repo_root,
        manifest_path,
        verify_files=True,
    )
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
    ):
        raise ValueError("MVSS-Net canonical Balanced250 release changed")
    selection = _require_mapping(
        raw.get("selection"),
        "dataset contract selection",
    )
    spec_record = _require_mapping(
        selection.get("spec"),
        "dataset contract selection spec",
    )
    limit = spec_record.get("per_condition_limit")
    if expected_mode == "formal" and limit is not None:
        raise ValueError("formal dataset contract has a smoke limit")
    if expected_mode == "smoke" and limit != SMOKE_PER_CONDITION:
        raise ValueError("smoke dataset contract limit changed")
    spec, selected = _selection_for_mode(
        release,
        mode=expected_mode,
        per_condition_limit=limit,
    )
    contract = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=_score_spec(),
    )
    if contract.as_dict() != raw:
        raise ValueError("MVSS-Net dataset contract does not rebuild exactly")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("MVSS-Net selected-row SHA-256 changed")
    if immutable.get("selected_ids_sha256") != contract.selection.selected_ids_sha256:
        raise ValueError("MVSS-Net selected-ID SHA-256 changed")
    return release, selected, contract


def _validate_immutable_static(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> None:
    if set(immutable) != EXPECTED_IMMUTABLE_KEYS:
        raise ValueError("MVSS-Net immutable key set changed")
    if (
        immutable.get("schema_version") != runner.RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
        or immutable.get("score_spec") != _score_spec().as_dict()
        or immutable.get("t2_spec") != runner.T2_SPEC
        or immutable.get("task_scope") != runner.TASK_SCOPE
        or immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT
        or immutable.get("license") != runner.LICENSE_RECORD
    ):
        raise ValueError("MVSS-Net immutable scientific contract changed")
    expected_model = {
        "name": runner.MODEL_NAME,
        "slug": runner.MODEL_SLUG,
        "architecture": runner.MODEL_ARCHITECTURE,
        "repository": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "checkpoint_id": runner.CHECKPOINT_ID,
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "checkpoint_bytes": runner.CHECKPOINT_BYTES,
        "training_dataset": "CASIAv2",
        "variant": "original_MVSS-Net_not_MVSS-Net++",
    }
    expected_preprocess = {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "opencv_imread_color",
        "channel_order": "BGR",
        "resize": "opencv_INTER_LINEAR_stretch_512x512",
        "scale": "uint8_divide_255",
        "normalization_mean_in_BGR_order": legacy.NORMALIZE_MEAN.tolist(),
        "normalization_std_in_BGR_order": legacy.NORMALIZE_STD.tolist(),
        "tensor_layout": "CHW",
        "tensor_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "apex": False,
    }
    expected_inference = {
        "raw_output": "one_channel_segmentation_logits_512",
        "probability_map": "sigmoid_segmentation_logits",
        "auxiliary_edge_output": "discarded_as_official_inference",
        "primary_T1": "continuous_global_max_of_model_512_probability",
        "secondary_T1": "official_native_saved_uint8_PNG_global_max_divide_255",
        "native_restore": (
            "probability_times_255_uint8_truncate_before_"
            "opencv_INTER_LINEAR_native_resize"
        ),
        "bayar_state": "constraint_kernel_normalized_in_place_each_forward",
        "resume": "fresh_checkpoint_and_replay_of_every_successful_prefix_input",
    }
    if immutable.get("model") != expected_model:
        raise ValueError("MVSS-Net immutable model contract changed")
    if immutable.get("preprocess") != expected_preprocess:
        raise ValueError("MVSS-Net immutable preprocessing changed")
    if immutable.get("inference") != expected_inference:
        raise ValueError("MVSS-Net immutable inference contract changed")
    _validate_runtime(immutable.get("runtime"), label="immutable.runtime")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        aggregate=immutable.get("adapter_sources_sha256"),
        repo_root=repo_root,
    )


def _validate_manifest_envelope(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> tuple[str, Mapping[str, Any]]:
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise ValueError("MVSS-Net finalized manifest key set changed")
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("MVSS-Net manifest is not a complete v2 run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest.fingerprint",
    )
    immutable = _require_mapping(manifest.get("immutable"), "manifest.immutable")
    if _fingerprint(immutable) != fingerprint:
        raise ValueError("MVSS-Net manifest fingerprint does not bind immutable")
    _validate_immutable_static(
        immutable,
        repo_root=repo_root,
        run_id=run_id,
        expected_mode=expected_mode,
    )
    disk = _require_mapping(
        manifest.get("disk_preflight"),
        "manifest.disk_preflight",
    )
    if (
        set(disk)
        != {
            "free_bytes_before_inference",
            "conservative_pending_bytes_plus_reserve",
            "fixed_reserve_bytes",
        }
        or any(
            _require_nonnegative_int(value, f"disk_preflight.{key}") < 0
            for key, value in disk.items()
        )
        or disk.get("fixed_reserve_bytes") != runner.MIN_DISK_RESERVE_BYTES
    ):
        raise ValueError("MVSS-Net disk preflight evidence changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    if set(execution) != EXPECTED_EXECUTION_KEYS:
        raise ValueError("MVSS-Net execution accounting key set changed")
    for key in EXPECTED_EXECUTION_KEYS:
        _require_nonnegative_int(execution.get(key), f"manifest.execution.{key}")
    if execution["physical_result_rows"] < execution["latest_result_rows"]:
        raise ValueError("MVSS-Net execution row accounting is impossible")
    _reject_nonfinite(manifest, "manifest")
    return fingerprint, immutable


def _independent_preprocess_tensor(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    import cv2

    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if (
        decoded is None
        or decoded.dtype != np.uint8
        or decoded.ndim != 3
        or decoded.shape[2] != 3
    ):
        raise ValueError("MVSS-Net independent OpenCV BGR decode changed")
    height, width = decoded.shape[:2]
    resized = cv2.resize(
        decoded,
        (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = resized.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - legacy.NORMALIZE_MEAN) / legacy.NORMALIZE_STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    if (
        tensor.shape != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("MVSS-Net independent normalized tensor changed")
    audit = {
        "native_size": [width, height],
        "model_size": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "decoder": "opencv_imread_color",
        "channel_order": "BGR",
        "resize": "opencv_inter_linear_stretch",
        "normalization": {
            "scale": "uint8_divide_255",
            "mean_in_bgr_order": legacy.NORMALIZE_MEAN.tolist(),
            "std_in_bgr_order": legacy.NORMALIZE_STD.tolist(),
        },
        "decoded_bgr_dtype": str(decoded.dtype),
        "decoded_bgr_shape": list(decoded.shape),
        "decoded_bgr_sha256": _array_sha256(decoded),
        "resized_bgr_dtype": str(resized.dtype),
        "resized_bgr_shape": list(resized.shape),
        "resized_bgr_sha256": _array_sha256(resized),
        "normalized_chw_dtype": str(tensor.dtype),
        "normalized_chw_shape": list(tensor.shape),
        "normalized_chw_sha256": _array_sha256(tensor),
    }
    return tensor, (width, height), audit


def _stable_sigmoid_array(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return np.ascontiguousarray(result.astype(np.float32))


def _independent_native_postprocess(
    model_score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    scores = np.asarray(model_score_map, dtype=np.float32)
    if (
        scores.shape != (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
        or width <= 0
        or height <= 0
    ):
        raise ValueError("MVSS-Net model score map cannot be postprocessed")
    quantized = (scores * np.float32(255.0)).astype(np.uint8)
    native = cv2.resize(
        quantized,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(native, dtype=np.uint8)


def _score_payload(
    raw_logits: np.ndarray,
    model_score_map: np.ndarray,
    native_uint8: np.ndarray,
) -> dict[str, Any]:
    logits = np.ascontiguousarray(raw_logits, dtype=np.float32)
    scores = np.ascontiguousarray(model_score_map, dtype=np.float32)
    native = np.ascontiguousarray(native_uint8, dtype=np.uint8)
    expected_shape = (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
    if (
        logits.shape != expected_shape
        or scores.shape != expected_shape
        or native.ndim != 2
        or native.size == 0
        or not np.isfinite(logits).all()
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
    ):
        raise ValueError("MVSS-Net raw score outputs are invalid")
    cpu_sigmoid = _stable_sigmoid_array(logits)
    max_abs = float(np.max(np.abs(cpu_sigmoid - scores)))
    if max_abs > runner.STATIC_CPU_SIGMOID_ABS_TOLERANCE:
        raise ValueError("MVSS-Net logit/probability static sanity check failed")
    score = float(np.max(scores))
    official = float(np.max(native)) / 255.0
    return {
        "raw_outputs": {
            "segmentation_logits_shape": list(logits.shape),
            "auxiliary_edge_output": "discarded_as_in_official_inference",
            "static_cpu_sigmoid_max_abs_diff": max_abs,
            "static_cpu_sigmoid_abs_tolerance": (
                runner.STATIC_CPU_SIGMOID_ABS_TOLERANCE
            ),
        },
        "ai_score": score,
        "probability": score,
        "score": score,
        "score_margin": None,
        "score_semantics": ("continuous_global_max_of_model_512_sigmoid_probability"),
        "calibrated_probability": False,
        "classification_decision": score > CLASSIFICATION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": ">",
        "official_png_score": official,
        "official_png_score_semantics": (
            "maximum_of_saved_native_uint8_map_divided_by_255"
        ),
        "official_png_decision": official > CLASSIFICATION_THRESHOLD,
        "official_png_threshold": CLASSIFICATION_THRESHOLD,
        "official_png_threshold_operator": ">",
    }


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _independent_pixel_metrics(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    include_ap: bool,
) -> dict[str, Any]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if (
        scores.shape != truth.shape
        or scores.size == 0
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
    ):
        raise ValueError("MVSS-Net score/target pixel contract changed")
    prediction = scores > MASK_THRESHOLD
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    pixel_ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )
    return {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": ">",
        "pixels": int(scores.size),
        "target_positive_pixels": int(np.count_nonzero(truth)),
        "predicted_positive_pixels": int(np.count_nonzero(prediction)),
        "predicted_positive_fraction": float(np.mean(prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "mcc": ((tp * tn - fp * fn) / denominator if denominator else None),
        "pixel_ap": pixel_ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


def _assert_nested_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float = 0.0,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} key set changed")
        for key, nested in expected.items():
            _assert_nested_close(
                actual[key],
                nested,
                label=f"{label}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list changed")
        for index, nested in enumerate(expected):
            _assert_nested_close(
                actual[index],
                nested,
                label=f"{label}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, float):
        if actual is None or not math.isclose(
            _require_finite(actual, label),
            expected,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"{label} numeric value changed")
        return
    if actual != expected:
        raise ValueError(f"{label} changed")


def _independent_resize_target(
    target: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        np.asarray(target, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _validate_score_payload(row: Mapping[str, Any], *, sample_id: str) -> None:
    score = _require_finite(row.get("ai_score"), f"{sample_id}.ai_score")
    official = _require_finite(
        row.get("official_png_score"),
        f"{sample_id}.official_png_score",
    )
    if (
        not 0.0 <= score <= 1.0
        or not 0.0 <= official <= 1.0
        or row.get("probability") != score
        or row.get("score") != score
        or row.get("score_margin") is not None
        or row.get("calibrated_probability") is not False
        or row.get("classification_decision") is not (score > 0.5)
        or row.get("classification_threshold") != 0.5
        or row.get("classification_threshold_operator") != ">"
        or row.get("official_png_decision") is not (official > 0.5)
        or row.get("official_png_threshold") != 0.5
        or row.get("official_png_threshold_operator") != ">"
    ):
        raise ValueError(f"{sample_id} score payload changed")


def _validate_attempt(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    sample_id = str(expected["sample_id"])
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError(f"{sample_id} result status changed")
    identity = runner.result_identity(
        expected,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        valid_for_metrics=status == "ok",
    )
    expected_keys = (
        runner._OK_RESULT_KEYS if status == "ok" else runner._ERROR_RESULT_KEYS
    )
    if set(row) != expected_keys:
        raise ValueError(f"{sample_id} result key set changed")
    for key, value in identity.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id} identity field {key} changed")
    _require_string(row.get("completed_at"), f"{sample_id}.completed_at")
    if status == "error":
        for field in ("error_type", "error", "traceback"):
            _require_string(row.get(field), f"{sample_id}.{field}")
        return
    _validate_score_payload(row, sample_id=sample_id)
    input_path = _safe_repo_file(
        expected.get("canonical_path"),
        repo_root=repo_root,
        expected_path=repo_root / str(expected["canonical_path"]),
        label=f"{sample_id} canonical input",
    )
    _, native_size, preprocess = _independent_preprocess_tensor(input_path)
    if native_size != (int(expected["width"]), int(expected["height"])):
        raise ValueError(f"{sample_id} native dimensions changed")
    if row.get("preprocess") != preprocess:
        raise ValueError(f"{sample_id} independent preprocessing changed")
    latency = _require_finite(row.get("latency_ms"), f"{sample_id}.latency_ms")
    if latency < 0.0:
        raise ValueError(f"{sample_id} latency is negative")
    _require_nonnegative_int(
        row.get("peak_cuda_memory_bytes"),
        f"{sample_id}.peak_cuda_memory_bytes",
    )
    if (
        row.get("mask_threshold") != MASK_THRESHOLD
        or row.get("mask_threshold_operator") != ">"
    ):
        raise ValueError(f"{sample_id} mask threshold changed")
    _reject_nonfinite(row, f"result.{sample_id}")


def _load_model_array(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_path: Path,
    expected_semantics: str,
    repo_root: Path,
    sample_id: str,
    bounded: bool,
) -> tuple[Path, str, str, np.ndarray]:
    path = _safe_repo_file(
        row.get(f"{prefix}_path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=f"{sample_id} {prefix}",
    )
    file_sha = _require_sha256(
        row.get(f"{prefix}_sha256"),
        f"{sample_id}.{prefix}_sha256",
    )
    if (
        sha256_file(path) != file_sha
        or path.stat().st_size != runner.MODEL_ARRAY_FILE_BYTES
        or row.get(f"{prefix}_bytes") != runner.MODEL_ARRAY_FILE_BYTES
        or row.get(f"{prefix}_shape")
        != [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE]
        or row.get(f"{prefix}_dtype") != "float32"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"{sample_id} {prefix} file metadata changed")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} cannot load {prefix}") from error
    if (
        type(array) not in (np.ndarray, np.memmap)
        or array.shape != (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or (bounded and (float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0))
    ):
        raise ValueError(f"{sample_id} {prefix} array contract changed")
    array_sha = _array_sha256(array)
    if row.get(f"{prefix}_array_sha256") != array_sha:
        raise ValueError(f"{sample_id} {prefix} array SHA-256 changed")
    return path, file_sha, array_sha, array


def _load_png(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_path: Path,
    expected_shape: tuple[int, int],
    expected_semantics: str,
    repo_root: Path,
    sample_id: str,
    binary: bool,
) -> tuple[Path, str, str, np.ndarray]:
    path = _safe_repo_file(
        row.get(f"{prefix}_path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=f"{sample_id} {prefix}",
    )
    file_sha = _require_sha256(
        row.get(f"{prefix}_sha256"),
        f"{sample_id}.{prefix}_sha256",
    )
    if (
        sha256_file(path) != file_sha
        or row.get(f"{prefix}_bytes") != path.stat().st_size
        or row.get(f"{prefix}_shape") != list(expected_shape)
        or row.get(f"{prefix}_dtype") != "uint8"
        or row.get(f"{prefix}_mode") != "L"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"{sample_id} {prefix} PNG metadata changed")
    try:
        with Image.open(path) as opened:
            opened.load()
            if (
                opened.format != "PNG"
                or opened.mode != "L"
                or opened.size != (expected_shape[1], expected_shape[0])
            ):
                raise ValueError(f"{sample_id} {prefix} PNG encoding changed")
            pixels = np.asarray(opened, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"{sample_id} {prefix} cannot be decoded") from error
    if (
        pixels.shape != expected_shape
        or pixels.dtype != np.uint8
        or (binary and not np.isin(pixels, (0, 255)).all())
    ):
        raise ValueError(f"{sample_id} {prefix} PNG pixels changed")
    array_sha = _array_sha256(pixels)
    if row.get(f"{prefix}_array_sha256") != array_sha:
        raise ValueError(f"{sample_id} {prefix} array SHA-256 changed")
    return path, file_sha, array_sha, pixels


def _validate_artifact_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> DenseArtifacts:
    sample_id = str(expected["sample_id"])
    width = int(expected["width"])
    height = int(expected["height"])
    native_shape = (height, width)
    paths = runner.artifact_paths(artifact_root, sample_id)
    (
        logits_path,
        logits_file_sha,
        logits_array_sha,
        logits,
    ) = _load_model_array(
        row,
        prefix="raw_logits_model",
        expected_path=paths["raw_logits"],
        expected_semantics="official_one_channel_segmentation_logits",
        repo_root=repo_root,
        sample_id=sample_id,
        bounded=False,
    )
    (
        model_path,
        model_file_sha,
        model_array_sha,
        model_score,
    ) = _load_model_array(
        row,
        prefix="score_map_model",
        expected_path=paths["model_score"],
        expected_semantics="official_sigmoid_segmentation_probability",
        repo_root=repo_root,
        sample_id=sample_id,
        bounded=True,
    )
    (
        native_path,
        native_file_sha,
        native_array_sha,
        native,
    ) = _load_png(
        row,
        prefix="score_map_native",
        expected_path=paths["native_score"],
        expected_shape=native_shape,
        expected_semantics=(
            "official_probability_times_255_uint8_truncate_then_"
            "opencv_INTER_LINEAR_native_resize"
        ),
        repo_root=repo_root,
        sample_id=sample_id,
        binary=False,
    )
    expected_native = _independent_native_postprocess(
        model_score,
        width=width,
        height=height,
    )
    if not np.array_equal(native, expected_native):
        raise ValueError(f"{sample_id} official native postprocess changed")
    expected_score = _score_payload(logits, model_score, native)
    for key, value in expected_score.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id} persisted score replay changed: {key}")

    applicable = expected.get("gt_mask_kind") in _T2_GT_KINDS
    expected_paths = {
        "raw_logits_model_512": repo_relative(logits_path, repo_root),
        "score_map_model_512": repo_relative(model_path, repo_root),
        "score_map_native_official": repo_relative(native_path, repo_root),
        "mask_native": (
            repo_relative(paths["mask"], repo_root) if applicable else None
        ),
    }
    if row.get("artifact_paths") != expected_paths:
        raise ValueError(f"{sample_id} artifact path mapping changed")
    if not applicable:
        if expected.get("gt_mask_kind") != _FULLFRAME_GT_KIND:
            raise ValueError(f"{sample_id} has unsupported T2 semantics")
        for field in (
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_array_sha256",
            "mask_shape",
            "mask_dtype",
            "mask_mode",
            "mask_semantics",
            "localization",
        ):
            if row.get(field) is not None:
                raise ValueError(
                    f"{sample_id} fullframe result fabricates T2 field {field}"
                )
        if paths["mask"].exists():
            raise ValueError(f"{sample_id} fullframe mask file exists")
        return DenseArtifacts(
            sample_id=sample_id,
            raw_logits_path=logits_path,
            raw_logits_file_sha256=logits_file_sha,
            raw_logits_array_sha256=logits_array_sha,
            model_score_path=model_path,
            model_score_file_sha256=model_file_sha,
            model_score_array_sha256=model_array_sha,
            native_score_path=native_path,
            native_score_file_sha256=native_file_sha,
            native_score_array_sha256=native_array_sha,
            mask_path=None,
            mask_file_sha256=None,
            mask_array_sha256=None,
            t2_applicable=False,
            width=width,
            height=height,
        )

    mask_path, mask_file_sha, mask_array_sha, mask = _load_png(
        row,
        prefix="mask",
        expected_path=paths["mask"],
        expected_shape=native_shape,
        expected_semantics="official_native_uint8_divide_255_strict_gt_0_5",
        repo_root=repo_root,
        sample_id=sample_id,
        binary=True,
    )
    native_score = native.astype(np.float32) / np.float32(255.0)
    if not np.array_equal(mask == 255, native_score > MASK_THRESHOLD):
        raise ValueError(f"{sample_id} mask is not native score map > 0.5")
    target_native = load_ground_truth(expected, repo_root)
    if target_native is None or target_native.shape != native_shape:
        raise ValueError(f"{sample_id} T2 ground truth changed")
    target_model = _independent_resize_target(
        target_native,
        width=legacy.MODEL_INPUT_SIZE,
        height=legacy.MODEL_INPUT_SIZE,
    )
    include_ap = expected.get("gt_mask_kind") == "exact_diff"
    expected_localization = {
        "model_512": _independent_pixel_metrics(
            model_score,
            target_model,
            include_ap=include_ap,
        ),
        "native": _independent_pixel_metrics(
            native_score,
            target_native,
            include_ap=include_ap,
        ),
    }
    _assert_nested_close(
        row.get("localization"),
        expected_localization,
        label=f"{sample_id}.localization",
    )
    return DenseArtifacts(
        sample_id=sample_id,
        raw_logits_path=logits_path,
        raw_logits_file_sha256=logits_file_sha,
        raw_logits_array_sha256=logits_array_sha,
        model_score_path=model_path,
        model_score_file_sha256=model_file_sha,
        model_score_array_sha256=model_array_sha,
        native_score_path=native_path,
        native_score_file_sha256=native_file_sha,
        native_score_array_sha256=native_array_sha,
        mask_path=mask_path,
        mask_file_sha256=mask_file_sha,
        mask_array_sha256=mask_array_sha,
        t2_applicable=True,
        width=width,
        height=height,
    )


def _exact_directory_inventory(
    directory: Path,
    *,
    expected_names: set[str],
    label: str,
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {directory}")
    entries = list(directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} contains an unsafe/non-file entry")
    actual = {entry.name for entry in entries}
    if actual != expected_names:
        raise ValueError(
            f"{label} inventory mismatch: "
            f"missing={sorted(expected_names - actual)[:1]}, "
            f"extra={sorted(actual - expected_names)[:1]}"
        )


def validate_artifact_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, DenseArtifacts]:
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe MVSS-Net artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != set(EXPECTED_ARTIFACT_INVENTORY) or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("MVSS-Net artifact-root inventory changed")
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for row in latest_results:
        sample_id = _require_string(row.get("sample_id"), "artifact result sample_id")
        if sample_id in result_by_id:
            raise ValueError(f"duplicate latest result {sample_id}")
        if row.get("status") != "ok":
            raise ValueError(f"latest result {sample_id} is not successful")
        result_by_id[sample_id] = row
    if set(result_by_id) != set(expected_by_id):
        raise ValueError("MVSS-Net artifact/result coverage changed")
    all_npy = {f"{sample_id}.npy" for sample_id in expected_by_id}
    all_png = {f"{sample_id}.png" for sample_id in expected_by_id}
    applicable = {
        sample_id
        for sample_id, row in expected_by_id.items()
        if row.get("gt_mask_kind") in _T2_GT_KINDS
    }
    expected_names = {
        "raw_logits_model_512": all_npy,
        "score_maps_model_512": all_npy,
        "score_maps_native_official": all_png,
        "masks_native": {f"{sample_id}.png" for sample_id in applicable},
    }
    for directory_name, names in expected_names.items():
        _exact_directory_inventory(
            artifact_root / directory_name,
            expected_names=names,
            label=f"MVSS-Net {directory_name} directory",
        )
    artifacts: dict[str, DenseArtifacts] = {}
    for expected in selected:
        sample_id = str(expected["sample_id"])
        artifacts[sample_id] = _validate_artifact_row(
            result_by_id[sample_id],
            expected=expected,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
    if len(artifacts) != len(selected) or sum(
        item.t2_applicable for item in artifacts.values()
    ) != len(applicable):
        raise ValueError("MVSS-Net validated artifact coverage changed")
    return artifacts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, DenseArtifacts],
) -> str:
    records = []
    for sample_id, artifact in sorted(artifacts.items()):
        records.append(
            {
                "sample_id": sample_id,
                "raw_logits_path": artifact.raw_logits_path.as_posix(),
                "raw_logits_file_sha256": artifact.raw_logits_file_sha256,
                "raw_logits_array_sha256": artifact.raw_logits_array_sha256,
                "model_score_path": artifact.model_score_path.as_posix(),
                "model_score_file_sha256": artifact.model_score_file_sha256,
                "model_score_array_sha256": artifact.model_score_array_sha256,
                "native_score_path": artifact.native_score_path.as_posix(),
                "native_score_file_sha256": artifact.native_score_file_sha256,
                "native_score_array_sha256": artifact.native_score_array_sha256,
                "mask_path": (
                    artifact.mask_path.as_posix()
                    if artifact.mask_path is not None
                    else None
                ),
                "mask_file_sha256": artifact.mask_file_sha256,
                "mask_array_sha256": artifact.mask_array_sha256,
                "t2_applicable": artifact.t2_applicable,
            }
        )
    return _fingerprint(records)


def _validate_stateful_history(
    selected: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently require errors/OKs to form one ordered selected prefix."""

    expected_ids = [str(row["sample_id"]) for row in selected]
    selected_index = 0
    errors = 0
    recovered = 0
    pending_had_error = False
    for line_number, attempt in enumerate(attempts, start=1):
        if selected_index >= len(expected_ids):
            raise ValueError("MVSS-Net history appends after full success")
        sample_id = attempt.get("sample_id")
        expected = expected_ids[selected_index]
        if sample_id != expected:
            raise ValueError(
                "MVSS-Net stateful result history is out of selected order "
                f"at row {line_number}: {sample_id!r} != {expected!r}"
            )
        status = attempt.get("status")
        if status == "error":
            errors += 1
            pending_had_error = True
        elif status == "ok":
            if pending_had_error:
                recovered += 1
            pending_had_error = False
            selected_index += 1
        else:
            raise ValueError(f"MVSS-Net history row {line_number} has invalid status")
    return {
        "policy": "exact_selected_prefix_with_zero_or_more_errors_before_each_ok",
        "physical_attempts": len(attempts),
        "successful_prefix": selected_index,
        "errors": errors,
        "recovered_error_to_ok": recovered,
    }


def _validate_dataset_manifest_outputs(
    *,
    manifest: Mapping[str, Any],
    immutable: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    release: CanonicalRelease,
    selected: Sequence[Mapping[str, Any]],
    contract: RunDatasetContract,
    expected_path: Path,
    results_path: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    physical_results: Sequence[Mapping[str, Any]],
    latest_results: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    stateful_history: Mapping[str, Any],
) -> None:
    if _read_jsonl(expected_path, "MVSS-Net expected inputs") != list(selected):
        raise ValueError("MVSS-Net expected-input snapshot changed")
    expected_dataset = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t2_applicable_images": sum(
            row.get("gt_mask_kind") in _T2_GT_KINDS for row in selected
        ),
    }
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("MVSS-Net manifest dataset binding changed")
    expected_base_outputs = {
        "results_path": repo_relative(results_path, repo_root),
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "summary_path": repo_relative(summary_path, repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        **{
            f"{name}_dir": repo_relative(artifact_root / name, repo_root)
            for name in runner.ARTIFACT_DIRECTORIES
        },
    }
    if immutable.get("outputs") != expected_base_outputs:
        raise ValueError("MVSS-Net immutable output bindings changed")
    inventory = {
        "raw_logits_model_512": len(selected),
        "score_maps_model_512": len(selected),
        "score_maps_native_official": len(selected),
        "masks_native": sum(
            row.get("gt_mask_kind") in _T2_GT_KINDS for row in selected
        ),
    }
    expected_outputs = {
        **expected_base_outputs,
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "artifact_inventory": inventory,
    }
    if manifest.get("outputs") != expected_outputs:
        raise ValueError("MVSS-Net finalized output bindings changed")
    expected_summary = {
        "schema_version": runner.RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_mvssnet_balanced.py",
        "run_id": str(manifest["run_id"]),
        "run_manifest_fingerprint": str(manifest["fingerprint"]),
        "status": "complete",
        "mode": str(immutable["mode"]),
        "model": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "score_spec": runner.SCORE_SPEC.as_dict(),
        "t2_spec": runner.T2_SPEC,
        "dataset_contract": contract.as_dict(),
        "coverage": dict(coverage),
        "stateful_history": dict(stateful_history),
        "artifact_inventory": inventory,
    }
    if set(summary) != set(expected_summary) | {"generated_at"}:
        raise ValueError("MVSS-Net runtime summary key set changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"MVSS-Net runtime summary {key} changed")

    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    indexed = index_latest_attempts(
        selected,
        physical_results,
        run_id=str(manifest["run_id"]),
        run_manifest_fingerprint=str(manifest["fingerprint"]),
        score_spec=_score_spec(),
    )
    new_successes = int(execution["new_successes"])
    resume_skips = int(execution["resume_skips"])
    replayed = int(execution["stateful_prefix_replayed"])
    if (
        execution.get("physical_result_rows") != len(physical_results)
        or execution.get("latest_result_rows") != len(latest_results)
        or execution.get("superseded_attempts") != indexed.superseded_attempts
        or new_successes + resume_skips != len(selected)
        or new_successes > len(selected)
        or resume_skips > len(selected)
        or execution.get("new_errors")
        > sum(row.get("status") == "error" for row in physical_results)
        or (new_successes > 0 and replayed != resume_skips)
        or (new_successes == 0 and replayed != 0)
    ):
        raise ValueError("MVSS-Net execution/stateful replay accounting changed")


def _validate_run_directory(run_dir: Path) -> None:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError(f"missing/unsafe MVSS-Net run dir: {run_dir}")
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
        "balanced250_metrics.json",
        "independent_audit.json",
    }
    for entry in run_dir.iterdir():
        if entry.name not in allowed or entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"MVSS-Net run directory contains unsafe entry: {entry.name}"
            )


def _capture_snapshot(
    *,
    manifest_path: Path,
    expected_path: Path,
    results_path: Path,
    summary_path: Path,
    release: CanonicalRelease,
    artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, str]:
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "results_sha256": sha256_file(results_path),
        "runtime_summary_sha256": sha256_file(summary_path),
        "dataset_manifest_sha256": release.manifest_sha256,
        "artifact_inventory_sha256": _artifact_inventory_sha256(artifacts),
    }


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    expected_mode: str,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> RunBundle:
    run_dir = _resolve_run_dir(results_dir, run_id, "MVSS-Net run directory")
    artifact_root = _resolve_run_dir(
        artifacts_dir,
        run_id,
        "MVSS-Net artifact directory",
    )
    _validate_run_directory(run_dir)
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe MVSS-Net artifact directory: {artifact_root}"
        )
    manifest_path = run_dir / "manifest.json"
    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "MVSS-Net manifest")
    summary = _load_json(summary_path, "MVSS-Net runtime summary")
    fingerprint, immutable = _validate_manifest_envelope(
        manifest,
        repo_root=repo_root,
        run_id=run_id,
        expected_mode=expected_mode,
    )
    release, selected, contract = _rebuild_contract(
        immutable,
        repo_root=repo_root,
        expected_mode=expected_mode,
    )
    _validate_provenance(
        immutable,
        repo_root=repo_root,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    physical = tuple(_read_jsonl(results_path, "MVSS-Net physical results"))
    stateful_history = _validate_stateful_history(selected, physical)
    if stateful_history != runner._validate_physical_attempt_history(
        selected,
        physical,
    ):
        raise ValueError("MVSS-Net independent/runner history audits disagree")
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    for row in physical:
        sample_id = _require_string(row.get("sample_id"), "result.sample_id")
        expected = expected_by_id.get(sample_id)
        if expected is None:
            raise ValueError(f"unexpected MVSS-Net result {sample_id}")
        _validate_attempt(
            row,
            expected=expected,
            repo_root=repo_root,
            run_id=run_id,
            fingerprint=fingerprint,
        )
    indexed = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=_score_spec(),
    )
    coverage_object = summarize_coverage(indexed)
    require_complete_coverage(coverage_object)
    latest = tuple(
        indexed.latest_by_sample_id[str(row["sample_id"])] for row in selected
    )
    if stateful_history["successful_prefix"] != len(selected) or any(
        row.get("status") != "ok" for row in latest
    ):
        raise ValueError("MVSS-Net terminal coverage is not a full successful prefix")
    coverage = coverage_object.as_dict()
    artifacts = validate_artifact_inventory(
        latest_results=latest,
        selected=selected,
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    _validate_dataset_manifest_outputs(
        manifest=manifest,
        immutable=immutable,
        repo_root=repo_root,
        artifact_root=artifact_root,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        summary=summary,
        physical_results=physical,
        latest_results=latest,
        coverage=coverage,
        stateful_history=stateful_history,
    )
    snapshot = _capture_snapshot(
        manifest_path=manifest_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        release=release,
        artifacts=artifacts,
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        mode=expected_mode,
        run_dir=run_dir,
        artifact_root=artifact_root,
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
        stateful_history=stateful_history,
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="formal",
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    mvssnet_root: Path,
    checkpoint_path: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="smoke",
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )


def _native_map_loader(bundle: RunBundle):
    selected_by_id = {str(row["sample_id"]): row for row in bundle.selected}

    def load(
        input_row: Mapping[str, Any],
        result_row: Mapping[str, Any],
    ) -> np.ndarray:
        sample_id = _require_string(
            input_row.get("sample_id"),
            "T2 callback input sample_id",
        )
        if result_row.get("sample_id") != sample_id:
            raise ValueError("T2 callback result identity changed")
        expected = selected_by_id.get(sample_id)
        artifact = bundle.artifacts.get(sample_id)
        if (
            expected is None
            or artifact is None
            or not artifact.t2_applicable
            or expected.get("gt_mask_kind") not in _T2_GT_KINDS
        ):
            raise ValueError(f"T2 callback requested non-applicable map {sample_id}")
        try:
            with Image.open(artifact.native_score_path) as opened:
                opened.load()
                pixels = np.asarray(opened, dtype=np.uint8)
        except OSError as error:
            raise ValueError(f"T2 callback cannot load {sample_id}") from error
        if (
            pixels.shape != (artifact.height, artifact.width)
            or _array_sha256(pixels) != artifact.native_score_array_sha256
        ):
            raise ValueError(f"T2 callback native map changed for {sample_id}")
        return np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255.0))

    return load


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "MVSS-Net formal metrics require iterations=1000 and seed=20260726"
        )
    if (
        bundle.mode != "formal"
        or len(bundle.selected) != FORMAL_IMAGES
        or sum(row.get("gt_mask_kind") in _T2_GT_KINDS for row in bundle.selected)
        != FORMAL_T2_IMAGES
    ):
        raise ValueError("MVSS-Net metrics require formal 1775/1025 coverage")
    selected_results = (
        tuple(bundle.latest_results) if results is None else tuple(results)
    )
    t1 = summarize_balanced250_t1(
        bundle.release.inputs,
        bundle.release.panel,
        bundle.release.source_pairs,
        selected_results,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        iterations=iterations,
        seed=seed,
    )
    if (
        t1.get("schema_version") != T1_METRICS_SCHEMA_VERSION
        or t1.get("coverage", {}).get("is_complete") is not True
    ):
        raise ValueError("MVSS-Net shared T1 metrics are incomplete")
    # The shared T2 reducer freezes >=0.5.  This runner's native score is
    # uint8/255, for which exactly 0.5 is unrepresentable, so >= and > yield
    # identical native masks.  The official strict comparator is audited
    # separately above and remains the scientific runner contract.
    t2 = summarize_balanced250_t2(
        bundle.release.inputs,
        selected_results,
        repo_root=bundle.release.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name="mvssnet_official_native_uint8_probability_map",
        threshold=MASK_THRESHOLD,
        threshold_operator=">=",
        iterations=iterations,
        seed=seed,
    )
    if (
        t2.get("schema_version") != T2_METRICS_SCHEMA_VERSION
        or t2.get("coverage", {}).get("is_complete") is not True
        or t2.get("coverage", {}).get("native_maps_evaluated") != FORMAL_T2_IMAGES
    ):
        raise ValueError("MVSS-Net shared T2 metrics are incomplete")
    excluded = t2.get("excluded_not_applicable", {}).get("counts_by_condition")
    if not isinstance(excluded, Mapping) or {
        condition: excluded.get(condition)
        for condition in (
            "fullframe_mouse",
            "fullframe_cat",
            "fullframe_trash_can",
        )
    } != {
        "fullframe_mouse": 250,
        "fullframe_cat": 250,
        "fullframe_trash_can": 250,
    }:
        raise ValueError("MVSS-Net T2 fullframe exclusion evidence changed")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "formal_images_t1": FORMAL_IMAGES,
        "formal_images_t2": FORMAL_T2_IMAGES,
        "primary_t1_threshold": CLASSIFICATION_THRESHOLD,
        "primary_t1_threshold_operator": ">",
        "official_t2_threshold": MASK_THRESHOLD,
        "official_t2_threshold_operator": ">",
        "shared_t2_threshold_operator": ">=",
        "shared_t2_operator_equivalent_on_uint8_divide_255": True,
        "t1": t1,
        "t2": t2,
    }


def _verify_bundle_unchanged(bundle: RunBundle) -> None:
    expected = bundle.evidence_snapshot
    paths = {
        "manifest_sha256": bundle.manifest_path,
        "expected_inputs_sha256": bundle.expected_path,
        "results_sha256": bundle.results_path,
        "runtime_summary_sha256": bundle.summary_path,
    }
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise ValueError(f"MVSS-Net evidence changed during audit: {key}")
    release = load_canonical_release(
        bundle.release.repo_root,
        bundle.release.manifest_path,
        verify_files=True,
    )
    if release.manifest_sha256 != expected["dataset_manifest_sha256"]:
        raise ValueError("Balanced250 release changed during MVSS-Net audit")
    for sample_id, artifact in bundle.artifacts.items():
        paths_and_hashes = (
            (artifact.raw_logits_path, artifact.raw_logits_file_sha256),
            (artifact.model_score_path, artifact.model_score_file_sha256),
            (artifact.native_score_path, artifact.native_score_file_sha256),
            (artifact.mask_path, artifact.mask_file_sha256),
        )
        for path, digest in paths_and_hashes:
            if path is not None and sha256_file(path) != digest:
                raise ValueError(f"MVSS-Net artifact changed during audit: {sample_id}")
    if (
        _artifact_inventory_sha256(bundle.artifacts)
        != expected["artifact_inventory_sha256"]
    ):
        raise ValueError("MVSS-Net artifact inventory changed during audit")


def _smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = _RUN_SPECIFIC_RESULT_FIELDS - set(row)
    if missing:
        raise ValueError(f"smoke result lacks field {sorted(missing)[0]}")
    return {
        key: value
        for key, value in row.items()
        if key not in _RUN_SPECIFIC_RESULT_FIELDS
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(stable_json(immutable))
    value["run_id"] = "<run-id>"
    outputs = _require_mapping(value.get("outputs"), "smoke outputs")
    for key in outputs:
        if key == "artifact_root" or key.endswith("_dir"):
            outputs[key] = f"<run-specific:{key}>"
        elif key.endswith("_path"):
            outputs[key] = f"<run-specific:{key}>"
    return value


def compare_computational_results(
    reference: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    reference_artifacts: Mapping[str, DenseArtifacts],
    replay_artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, Any]:
    if len(reference) != SMOKE_IMAGES or len(replay) != SMOKE_IMAGES:
        raise ValueError("MVSS-Net smoke comparison requires 35+35 rows")
    if set(reference_artifacts) != {str(row["sample_id"]) for row in reference} or set(
        replay_artifacts
    ) != {str(row["sample_id"]) for row in replay}:
        raise ValueError("MVSS-Net smoke artifact coverage changed")
    applicable = 0
    for left, right in zip(reference, replay, strict=True):
        sample_id = str(left["sample_id"])
        if right.get("sample_id") != sample_id:
            raise ValueError("MVSS-Net smoke result order changed")
        if _smoke_projection(left) != _smoke_projection(right):
            raise ValueError(f"MVSS-Net smoke computational row changed: {sample_id}")
        left_artifact = reference_artifacts[sample_id]
        right_artifact = replay_artifacts[sample_id]
        if left_artifact.t2_applicable != right_artifact.t2_applicable:
            raise ValueError(f"MVSS-Net smoke T2 applicability changed: {sample_id}")
        for label, left_path, right_path in (
            (
                "raw logits",
                left_artifact.raw_logits_path,
                right_artifact.raw_logits_path,
            ),
            (
                "model score map",
                left_artifact.model_score_path,
                right_artifact.model_score_path,
            ),
            (
                "native score map",
                left_artifact.native_score_path,
                right_artifact.native_score_path,
            ),
            ("mask", left_artifact.mask_path, right_artifact.mask_path),
        ):
            if (left_path is None) is not (right_path is None):
                raise ValueError(f"{sample_id} smoke {label} applicability changed")
            if (
                left_path is not None
                and right_path is not None
                and left_path.read_bytes() != right_path.read_bytes()
            ):
                raise ValueError(f"{sample_id} smoke {label} bytes changed")
        applicable += int(left_artifact.t2_applicable)
    if applicable != SMOKE_T2_IMAGES:
        raise ValueError("MVSS-Net smoke T2 comparison coverage changed")
    return {
        "images_compared": SMOKE_IMAGES,
        "t2_applicable_images_compared": SMOKE_T2_IMAGES,
        "t2_not_applicable_images_compared": SMOKE_IMAGES - SMOKE_T2_IMAGES,
        "computational_result_projection_exact": True,
        "raw_logits_file_bytes_exact": True,
        "model_score_map_file_bytes_exact": True,
        "native_score_map_png_file_bytes_exact": True,
        "applicable_threshold_png_file_bytes_exact": True,
        "fullframe_masks_absent_in_both_runs": True,
        "t1_scores_exact": True,
        "strict_localization_summaries_exact": True,
        "stateful_selected_order_exact": True,
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_output_path(
    path: Path,
    *,
    expected_path: Path,
    protected: Sequence[Path],
    label: str,
) -> Path:
    lexical = _lexical_absolute(path)
    expected = _lexical_absolute(expected_path)
    _reject_symlink_components(lexical, label)
    if lexical != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    if lexical in {item.resolve() for item in protected}:
        raise ValueError(f"{label} overlaps protected evidence")
    if lexical.exists() and (lexical.is_symlink() or not lexical.is_file()):
        raise ValueError(f"{label} is not a regular output file")
    parent = lexical.parent
    _reject_symlink_components(parent, f"{label} parent")
    if parent.exists() and (not parent.is_dir() or parent.is_symlink()):
        raise ValueError(f"{label} parent is unsafe")
    return lexical


def _write_json_verified(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected_sha = _json_sha256(value)
    atomic_write_json(path, dict(value))
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} changed during verified write")


def _comparison_output_path(
    *,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    return (
        results_dir
        / "_reports"
        / f"{reference_run_id}__vs__{replay_run_id}_comparison.json"
    )


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    mvssnet_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        reference_run_id != DEFAULT_SMOKE_RUN_ID_A
        or replay_run_id != DEFAULT_SMOKE_RUN_ID_B
        or reference_run_id == replay_run_id
    ):
        raise ValueError("MVSS-Net smoke comparison requires frozen A then B")
    reference = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=reference_run_id,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    replay = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=replay_run_id,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    expected_history = {
        "policy": "exact_selected_prefix_with_zero_or_more_errors_before_each_ok",
        "physical_attempts": SMOKE_IMAGES,
        "successful_prefix": SMOKE_IMAGES,
        "errors": 0,
        "recovered_error_to_ok": 0,
    }
    if (
        [row["sample_id"] for row in reference.selected]
        != [row["sample_id"] for row in replay.selected]
        or reference.contract.as_dict() != replay.contract.as_dict()
        or _smoke_immutable_projection(reference.immutable)
        != _smoke_immutable_projection(replay.immutable)
        or reference.immutable.get("runtime") != replay.immutable.get("runtime")
        or reference.stateful_history != expected_history
        or replay.stateful_history != expected_history
    ):
        raise ValueError(
            "MVSS-Net A/B smoke immutable/runtime/stateful history changed"
        )
    comparison = compare_computational_results(
        reference.latest_results,
        replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
    )
    _verify_bundle_unchanged(reference)
    _verify_bundle_unchanged(replay)
    analyzer_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "exact_reproduction_passed",
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "compared_at": utc_now(),
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            "stateful_history": dict(reference.stateful_history),
            "evidence_snapshot": dict(reference.evidence_snapshot),
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            "stateful_history": dict(replay.stateful_history),
            "evidence_snapshot": dict(replay.evidence_snapshot),
        },
        "selection": {
            "selected_images": SMOKE_IMAGES,
            "per_condition": SMOKE_PER_CONDITION,
            "selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
            "t2_applicable_images": SMOKE_T2_IMAGES,
            "t2_not_applicable_images": SMOKE_IMAGES - SMOKE_T2_IMAGES,
        },
        "recorded_runtime_exact": True,
        "comparison": comparison,
        "numerical_golden_boundary": {
            "author_published_numerical_golden": None,
            "smoke_A_B_is_executable_reproduction_gate": True,
            "no_author_golden_claim_fabricated": True,
        },
        "analyzer_source": {
            "path": "eval/opensource/analyze_mvssnet_balanced.py",
            "sha256": sha256_file(analyzer_path),
            "bytes": analyzer_path.stat().st_size,
        },
    }
    expected_output = _comparison_output_path(
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    output = _validate_output_path(
        output_path,
        expected_path=expected_output,
        protected=[
            reference.manifest_path,
            reference.expected_path,
            reference.results_path,
            reference.summary_path,
            replay.manifest_path,
            replay.expected_path,
            replay.results_path,
            replay.summary_path,
        ],
        label="MVSS-Net smoke comparison output",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_verified(output, report, label="MVSS-Net smoke comparison")
    return report


def _configure_recorded_runtime(
    *,
    recorded: Mapping[str, Any],
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    expected = _validate_runtime(recorded, label="recorded runtime")
    if device_text != expected.get("device"):
        raise ValueError(
            "MVSS-Net replay device must equal the recorded runtime device"
        )
    device, current = runner.configure_runtime(device_text)
    validated = _validate_runtime(current, label="current replay runtime")
    if validated != expected:
        raise ValueError("MVSS-Net current runtime differs from recorded")
    return device, validated


def replay_model(
    bundle: RunBundle,
    *,
    mvssnet_root: Path,
    checkpoint_path: Path,
    device: Any,
) -> dict[str, Any]:
    """Replay the complete ordered sequence with one fresh stateful model."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("MVSS-Net fresh replay requires formal 1775")
    if bundle.stateful_history.get("successful_prefix") != FORMAL_IMAGES:
        raise ValueError("MVSS-Net fresh replay requires complete persisted prefix")
    model = None
    logits_compared = 0
    model_maps_compared = 0
    native_maps_compared = 0
    masks_derived = 0
    max_sigmoid_abs_diff = 0.0
    try:
        model, loaded_device = legacy.load_model(
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint_path,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("MVSS-Net fresh model loaded on wrong device")
        for index, (input_row, persisted) in enumerate(
            zip(bundle.selected, bundle.latest_results, strict=True),
            start=1,
        ):
            sample_id = str(input_row["sample_id"])
            if persisted.get("sample_id") != sample_id:
                raise ValueError("MVSS-Net fresh replay order changed")
            input_path = _safe_repo_file(
                input_row.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                expected_path=(
                    bundle.release.repo_root / str(input_row["canonical_path"])
                ),
                label=f"{sample_id} replay input",
            )
            tensor, native_size, preprocess = _independent_preprocess_tensor(input_path)
            if native_size != (
                int(input_row["width"]),
                int(input_row["height"]),
            ) or preprocess != persisted.get("preprocess"):
                raise ValueError(f"{sample_id} fresh preprocessing changed")
            raw_logits, model_score, _peak, _latency = legacy.infer_one(
                model,
                device,
                tensor,
            )
            raw_logits = np.ascontiguousarray(raw_logits, dtype=np.float32)
            model_score = np.ascontiguousarray(model_score, dtype=np.float32)
            artifact = bundle.artifacts[sample_id]
            stored_logits = np.load(
                artifact.raw_logits_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            stored_model = np.load(
                artifact.model_score_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if not np.array_equal(raw_logits, stored_logits):
                raise ValueError(f"{sample_id} fresh raw logits changed")
            if not np.array_equal(model_score, stored_model):
                raise ValueError(f"{sample_id} fresh model score map changed")
            native = _independent_native_postprocess(
                model_score,
                width=artifact.width,
                height=artifact.height,
            )
            with Image.open(artifact.native_score_path) as opened:
                opened.load()
                stored_native = np.asarray(opened, dtype=np.uint8)
            if not np.array_equal(native, stored_native):
                raise ValueError(f"{sample_id} fresh native score map changed")
            payload = _score_payload(raw_logits, model_score, native)
            for key, value in payload.items():
                if persisted.get(key) != value:
                    raise ValueError(f"{sample_id} fresh T1 payload changed: {key}")
            max_sigmoid_abs_diff = max(
                max_sigmoid_abs_diff,
                float(payload["raw_outputs"]["static_cpu_sigmoid_max_abs_diff"]),
            )
            logits_compared += 1
            model_maps_compared += 1
            native_maps_compared += 1
            if artifact.t2_applicable:
                if artifact.mask_path is None:
                    raise ValueError(f"{sample_id} applicable replay mask missing")
                with Image.open(artifact.mask_path) as opened:
                    opened.load()
                    stored_mask = np.asarray(opened, dtype=np.uint8)
                if not np.array_equal(
                    stored_mask == 255,
                    native.astype(np.float32) / np.float32(255.0) > MASK_THRESHOLD,
                ):
                    raise ValueError(f"{sample_id} fresh derived mask changed")
                masks_derived += 1
            elif artifact.mask_path is not None:
                raise ValueError(f"{sample_id} fullframe replay fabricates a mask")
            del (
                tensor,
                raw_logits,
                model_score,
                native,
                stored_logits,
                stored_model,
                stored_native,
            )
            if index % 25 == 0 or index == FORMAL_IMAGES:
                print(
                    f"[fresh replay {index}/{FORMAL_IMAGES}] exact {sample_id}",
                    flush=True,
                )
            gc.collect()
            if getattr(device, "type", None) == "cuda":
                import torch

                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            import torch

            torch.cuda.empty_cache()
    if (
        logits_compared != FORMAL_IMAGES
        or model_maps_compared != FORMAL_IMAGES
        or native_maps_compared != FORMAL_IMAGES
        or masks_derived != FORMAL_T2_IMAGES
    ):
        raise ValueError("MVSS-Net fresh replay coverage changed")
    return {
        "status": "fresh_full_stateful_model_replay_exact",
        "fresh_model_instances": 1,
        "selected_images_freshly_reopened": FORMAL_IMAGES,
        "selected_images_freshly_preprocessed": FORMAL_IMAGES,
        "model_forwards_in_selected_order": FORMAL_IMAGES,
        "bayar_prefix_replayed_from_checkpoint": FORMAL_IMAGES,
        "raw_logits_compared_exact": logits_compared,
        "model_score_maps_compared_exact": model_maps_compared,
        "native_score_maps_rederived_exact": native_maps_compared,
        "t1_score_payloads_compared_exact": FORMAL_IMAGES,
        "applicable_masks_rederived_exact": masks_derived,
        "fullframe_masks_not_created": FORMAL_IMAGES - FORMAL_T2_IMAGES,
        "maximum_raw_logits_abs_diff": 0.0,
        "maximum_model_score_map_abs_diff": 0.0,
        "maximum_native_score_map_abs_diff": 0.0,
        "maximum_static_cpu_sigmoid_abs_diff": max_sigmoid_abs_diff,
    }


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    mvssnet_root: Path,
    checkpoint_path: Path,
    device_text: str,
    metrics_output_path: Path,
    audit_output_path: Path,
    comparison_output_path: Path,
    replay: bool = True,
    compare_smoke: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError(f"MVSS-Net formal audit requires {DEFAULT_FORMAL_RUN_ID}")
    smoke_report: dict[str, Any] | None = None
    smoke_comparison_sha256: str | None = None
    if compare_smoke:
        smoke_report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            reference_run_id=DEFAULT_SMOKE_RUN_ID_A,
            replay_run_id=DEFAULT_SMOKE_RUN_ID_B,
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint_path,
            output_path=comparison_output_path,
        )
        smoke_comparison_sha256 = sha256_file(comparison_output_path)
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    protected = [
        bundle.manifest_path,
        bundle.expected_path,
        bundle.results_path,
        bundle.summary_path,
        *[
            path
            for artifact in bundle.artifacts.values()
            for path in (
                artifact.raw_logits_path,
                artifact.model_score_path,
                artifact.native_score_path,
                artifact.mask_path,
            )
            if path is not None
        ],
    ]
    metrics_output = _validate_output_path(
        metrics_output_path,
        expected_path=bundle.run_dir / "balanced250_metrics.json",
        protected=protected,
        label="MVSS-Net Balanced250 metrics output",
    )
    audit_output = _validate_output_path(
        audit_output_path,
        expected_path=bundle.run_dir / "independent_audit.json",
        protected=[*protected, metrics_output],
        label="MVSS-Net independent audit output",
    )
    structural_golden = independent_structural_golden(
        checkpoint_path=checkpoint_path,
        mvssnet_root=mvssnet_root,
        recorded_checkpoint_audit=_require_mapping(
            bundle.immutable.get("checkpoint_audit"),
            "recorded checkpoint audit",
        ),
        recorded_model_audit=_require_mapping(
            bundle.immutable.get("model_audit"),
            "recorded model audit",
        ),
    )
    metrics = recompute_metrics(
        bundle,
        iterations=iterations,
        seed=seed,
    )
    recorded_runtime = _validate_runtime(
        bundle.immutable.get("runtime"),
        label="formal recorded runtime",
    )
    current_runtime: dict[str, Any] | None = None
    fresh_replay: dict[str, Any] | None = None
    if replay:
        device, current_runtime = _configure_recorded_runtime(
            recorded=recorded_runtime,
            device_text=device_text,
        )
        fresh_replay = replay_model(
            bundle,
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    provenance_after = _validate_provenance(
        bundle.immutable,
        repo_root=repo_root,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
    )
    if (
        provenance_after["source"] != bundle.immutable.get("source")
        or provenance_after["checkpoint"] != bundle.immutable.get("checkpoint")
        or provenance_after["environment"] != bundle.immutable.get("environment")
        or provenance_after["mouse_reference"]
        != bundle.immutable.get("mouse_reference")
        or provenance_after["license"] != bundle.immutable.get("license")
    ):
        raise ValueError("MVSS-Net provenance changed during analysis")
    _verify_bundle_unchanged(bundle)

    metrics_sha = _json_sha256(metrics)
    analyzer_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "replay_and_smoke_audit_passed"
            if replay and compare_smoke
            else "artifact_audit_passed_with_explicit_gates_skipped"
        ),
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "selection": {
            "formal_images_t1": FORMAL_IMAGES,
            "formal_images_t2": FORMAL_T2_IMAGES,
            "condition_counts": FORMAL_COUNTS,
            "selected_rows_sha256": FORMAL_SELECTED_ROWS_SHA256,
            "selected_ids_sha256": FORMAL_SELECTED_IDS_SHA256,
            "fullframe_t2_not_applicable_images": FORMAL_IMAGES - FORMAL_T2_IMAGES,
        },
        "coverage": dict(bundle.coverage),
        "stateful_history": dict(bundle.stateful_history),
        "evidence_snapshot": dict(bundle.evidence_snapshot),
        "independent_provenance": {
            "source_exact_and_clean": True,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_bound_files": len(runner.MVSSNET_SOURCE_FILES),
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": runner.CHECKPOINT_BYTES,
            "mouse_reference_files": len(runner.MOUSE_REFERENCE_FILES),
            "analysis_environment_exact": True,
            "source_and_checkpoint_unchanged_after_replay": True,
        },
        "independent_structural_golden": structural_golden,
        "smoke_A_B": (
            {
                "status": smoke_report["status"],
                "schema_version": smoke_report["schema_version"],
                "comparison": smoke_report["comparison"],
                "output_path": repo_relative(comparison_output_path, repo_root),
                "output_sha256": smoke_comparison_sha256,
            }
            if smoke_report is not None
            else None
        ),
        "recorded_runtime": recorded_runtime,
        "recorded_runtime_reproduced": current_runtime,
        "artifact_audit": {
            "raw_logits_float32_verified": FORMAL_IMAGES,
            "model_probability_maps_float32_verified": FORMAL_IMAGES,
            "official_native_uint8_maps_verified": FORMAL_IMAGES,
            "native_threshold_png_masks_verified": FORMAL_T2_IMAGES,
            "fullframe_masks_absent": FORMAL_IMAGES - FORMAL_T2_IMAGES,
            "all_paths_canonical": True,
            "all_file_hashes_exact": True,
            "all_array_hashes_exact": True,
            "all_shapes_and_dtypes_exact": True,
            "all_values_finite": True,
            "model_maps_in_unit_interval": True,
            "logit_to_sigmoid_static_sanity_exact_within_recorded_tolerance": True,
            "official_quantize_before_resize_postprocess_exact": True,
            "applicable_png_exact_native_score_strict_gt_0_5": True,
            "model_and_native_strict_localization_recomputed": FORMAL_T2_IMAGES,
            "fullframe_localization_metrics_absent": True,
        },
        "metrics": {
            "schema_version": metrics["schema_version"],
            "t1_schema_version": metrics["t1"]["schema_version"],
            "t2_schema_version": metrics["t2"]["schema_version"],
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "shared_balanced250_metrics_only": True,
            "metrics_sha256": metrics_sha,
            "shared_t2_ge_equivalent_to_official_gt_on_uint8_divide_255": True,
        },
        "fresh_model_replay": fresh_replay,
        "fresh_model_metrics_exact": True if replay else None,
        "fresh_model_metrics_equivalence_proof": (
            {
                "t1_inputs": "all_1775_score_payloads_exact",
                "t2_inputs": "all_1025_applicable_native_score_maps_exact",
                "stateful_model_instances": 1,
                "ordered_forwards": FORMAL_IMAGES,
                "metric_reducer_rerun_on_identical_inputs": False,
            }
            if replay
            else None
        ),
        "scientific_boundaries": {
            "t1_uses_all_1775_images": True,
            "t2_uses_only_real_and_local_1025_images": True,
            "fullframe_t2_is_not_applicable": True,
            "fullframe_diagnostic_maps_are_not_t2_results": True,
            "primary_t1_is_model_512_global_max_not_native_png_max": True,
            "official_bgr_normalization_order_preserved": True,
            "bayar_kernel_state_requires_ordered_prefix_replay": True,
            "apex_O1_not_used": True,
            "float32_deterministic_runtime_required": True,
            "author_published_numeric_golden_available": False,
            "no_numeric_golden_was_fabricated": True,
            "fresh_full_stateful_model_replay_is_default": True,
            "frozen_A_B_smoke_comparison_is_default": True,
        },
        "license": {
            "classification": runner.LICENSE_RECORD["classification"],
            "commercial_use_cleared": False,
            "redistribution_cleared": False,
            "no_license_file_or_separate_checkpoint_terms_found": True,
            "benchmark_execution_does_not_establish_permission": True,
        },
        "release_eligibility": {
            "scientific_artifact_audit_passed": True,
            "full_reproduction_gates_passed": replay and compare_smoke,
            "commercial_release_cleared": False,
            "blocking_reason": (
                "no explicit source/checkpoint commercial or redistribution "
                "grant was found"
            ),
        },
        "analyzer_source": {
            "path": "eval/opensource/analyze_mvssnet_balanced.py",
            "bytes": analyzer_path.stat().st_size,
            "sha256": sha256_file(analyzer_path),
        },
        "artifacts": {
            "metrics_path": repo_relative(metrics_output, repo_root),
            "metrics_sha256": metrics_sha,
            "audit_path": repo_relative(audit_output, repo_root),
            "smoke_comparison_path": (
                repo_relative(comparison_output_path, repo_root)
                if compare_smoke
                else None
            ),
            "smoke_comparison_sha256": smoke_comparison_sha256,
        },
    }
    _write_json_verified(
        metrics_output,
        metrics,
        label="MVSS-Net Balanced250 metrics",
    )
    _write_json_verified(
        audit_output,
        report,
        label="MVSS-Net independent audit",
    )
    if sha256_file(metrics_output) != metrics_sha:
        raise ValueError("MVSS-Net metrics changed after write")
    return report


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument("--mvssnet-root", type=Path, default=DEFAULT_MVSSNET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="must exactly reproduce the recorded formal runtime device",
    )
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--skip-smoke-comparison", action="store_true")
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
    repo_root = args.repo_root.resolve()
    results_dir = _safe_standard_root(
        args.results_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="MVSS-Net results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="MVSS-Net artifacts root",
    )
    run_id = _valid_run_id(args.run_id)
    mvssnet_root = _anchored(args.mvssnet_root, repo_root)
    checkpoint = _anchored(args.checkpoint, repo_root)
    expected_comparison_output = _comparison_output_path(
        results_dir=results_dir,
        reference_run_id=DEFAULT_SMOKE_RUN_ID_A,
        replay_run_id=DEFAULT_SMOKE_RUN_ID_B,
    )
    comparison_output = (
        _anchored(args.comparison_output, repo_root)
        if args.comparison_output is not None
        else expected_comparison_output
    )
    if args.compare_smoke_run_id is not None:
        compare_id = _valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
            or args.skip_smoke_comparison
        ):
            raise ValueError("MVSS-Net smoke-only comparison accepts no formal options")
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint,
            output_path=comparison_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError("MVSS-Net formal run ID is frozen")
    run_dir = _resolve_run_dir(
        results_dir,
        run_id,
        "MVSS-Net formal run directory",
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
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint,
        device_text=str(args.device),
        metrics_output_path=metrics_output,
        audit_output_path=audit_output,
        comparison_output_path=comparison_output,
        replay=not args.skip_model_replay,
        compare_smoke=not args.skip_smoke_comparison,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
