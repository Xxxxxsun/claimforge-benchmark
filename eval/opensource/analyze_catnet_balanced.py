#!/usr/bin/env python3
"""Audit, analyze, and freshly replay CAT-Net v2 Balanced250 runs.

The analyzer has three independent responsibilities:

* replay every persisted JPEG/DCT and dense artifact contract;
* recompute the frozen Balanced250 T2 statistics with the shared reducer; and
* perform a complete fresh-model replay on the recorded device.

It also enforces the deterministic smoke A/B gate.  CAT-Net is native T2-only:
the analyzer rejects image scores, T1 decisions, full-frame result rows, and
any attempt to promote a localization-map statistic to T1.
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
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from eval.opensource import run_catnet as legacy
from eval.opensource import run_catnet_balanced as runner
from eval.opensource.balanced250_localization_metrics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    summarize_balanced250_t2,
)
from eval.opensource.balanced_run_contract import (
    RunDatasetContract,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    LOCALIZATION_CONDITIONS,
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
    load_ground_truth,
    select_inputs,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.maskclip_metrics import binary_pixel_metrics


ANALYSIS_SCHEMA = "catnet_balanced_analysis_v2"
ARTIFACT_AUDIT_SCHEMA = "catnet_balanced_artifact_audit_v2"
SMOKE_COMPARISON_SCHEMA = "catnet_balanced_smoke_comparison_v2"
FRESH_REPLAY_SCHEMA = "catnet_balanced_fresh_replay_v2"
STATIC_LOGIT_RESTORE_ABS_TOLERANCE = float(
    2 * np.finfo(np.float32).eps
)

DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = runner.DEFAULT_ARTIFACTS_DIR
DEFAULT_DATASET_MANIFEST = runner.DEFAULT_DATASET_MANIFEST
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
FORMAL_SELECTED_IDS_SHA256 = runner.FORMAL_SELECTED_IDS_SHA256
FORMAL_SELECTED_ROWS_SHA256 = runner.FORMAL_SELECTED_ROWS_SHA256
SMOKE_SELECTED_IDS_SHA256 = runner.SMOKE_SELECTED_IDS_SHA256
SMOKE_SELECTED_ROWS_SHA256 = runner.SMOKE_SELECTED_ROWS_SHA256

ANALYZER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/analyze_catnet_balanced.py",
    "eval/opensource/run_catnet_balanced.py",
    "eval/opensource/run_catnet.py",
    "eval/opensource/catnet_metrics.py",
    "eval/opensource/maskclip_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_localization_metrics.py",
    "eval/opensource/common.py",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_T1_TOP_LEVEL = runner._FORBIDDEN_T1_TOP_LEVEL
_STATIC_RESULT_FIELDS = (
    "model",
    "model_slug",
    "checkpoint_sha256",
    "valid_for_metrics",
    "valid_for_t1",
    "valid_for_t2",
    "t2_applicable",
    "task_scope",
)
_COMPUTATIONAL_RESULT_FIELDS = (
    "sample_id",
    "condition",
    "preprocess",
    "qtable_sha256",
    "dct_y_sha256",
    "raw_logits_sha256",
    "raw_logits_array_sha256",
    "raw_logits_shape",
    "raw_logits_dtype",
    "score_map_sha256",
    "score_map_array_sha256",
    "score_map_shape",
    "score_map_dtype",
    "mask_sha256",
    "mask_array_sha256",
    "mask_shape",
    "mask_dtype",
    "mask_threshold",
    "mask_threshold_operator",
    "localization",
)


@dataclass(frozen=True)
class RunBundle:
    repo_root: Path
    run_id: str
    mode: str
    run_dir: Path
    artifact_root: Path
    manifest_path: Path
    expected_path: Path
    results_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    release: CanonicalRelease
    selected: tuple[dict[str, Any], ...]
    dataset_contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    fingerprint: str
    snapshot: Mapping[str, tuple[int, str]]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a nonnegative integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        raise ValueError(f"{label} is non-finite")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_json_constant,
    )
    result = _require_mapping(value, label)
    _reject_nonfinite(result, label)
    return result


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ValueError(f"{label} line {line_number} has no LF")
            try:
                text = raw[:-1].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{label} line {line_number} is not UTF-8"
                ) from error
            if not text or text.strip() != text:
                raise ValueError(
                    f"{label} line {line_number} has outer whitespace"
                )
            value = json.loads(
                text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
            row = _require_mapping(value, f"{label} line {line_number}")
            if stable_json(row) != text:
                raise ValueError(
                    f"{label} line {line_number} is not canonical JSON"
                )
            _reject_nonfinite(row, f"{label} line {line_number}")
            rows.append(row)
    return rows


def _lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    if path.is_absolute():
        return path
    if base is None:
        raise ValueError("relative path has no base")
    return base / path


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_standard_root(
    requested: Path,
    *,
    expected: Path,
    label: str,
) -> Path:
    _reject_symlink_components(requested, label)
    if requested.resolve() != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected}")
    return requested.resolve()


def _resolve_run_dir(root: Path, run_id: Any, label: str) -> Path:
    value = runner._valid_run_id(run_id)
    path = root / value
    _reject_symlink_components(path, label)
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _safe_repo_file(
    repo_root: Path,
    value: Any,
    *,
    expected: Path | None,
    label: str,
) -> Path:
    text = _require_string(value, f"{label} path")
    path = _lexical_absolute(Path(text), base=repo_root)
    _reject_symlink_components(path, label)
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    if expected is not None and resolved != expected.resolve():
        raise ValueError(f"{label} path changed")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyzer_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ANALYZER_SOURCE_PATHS:
        path = root / relative
        _reject_symlink_components(path, f"analyzer source {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _independent_environment_record() -> dict[str, Any]:
    executable = Path(sys.executable)
    if executable != runner.EXPECTED_PYTHON_EXECUTABLE:
        raise ValueError("analyzer is not running in the frozen CAT-Net venv")
    if Path(sys.prefix).resolve() != runner.EXPECTED_VENV_ROOT.resolve():
        raise ValueError("analyzer sys.prefix is not the CAT-Net venv")
    pyvenv = runner.EXPECTED_VENV_ROOT / "pyvenv.cfg"
    if (
        not pyvenv.is_file()
        or pyvenv.stat().st_size != runner.EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != runner.EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("analyzer pyvenv.cfg binding changed")
    packages = {
        name: _package_version(name) for name in runner.EXPECTED_PACKAGES
    }
    if packages != runner.EXPECTED_PACKAGES:
        raise ValueError("analyzer package lock changed")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or not sys.dont_write_bytecode
        or os.environ.get("PYTHONPYCACHEPREFIX")
        != str(runner.FROZEN_PYTHONPYCACHEPREFIX)
        or sys.pycache_prefix is None
        or Path(sys.pycache_prefix) != runner.FROZEN_PYTHONPYCACHEPREFIX
    ):
        raise ValueError("analyzer bytecode-isolation environment changed")
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before analyzer environment audit")
    if (
        torch.__version__ != runner.EXPECTED_PACKAGES["torch"]
        or torch.version.cuda != runner.EXPECTED_TORCH_CUDA_VERSION
    ):
        raise ValueError("analyzer torch build identity changed")
    value = {
        "python_executable": str(executable),
        "venv_root": str(runner.EXPECTED_VENV_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "bytes": pyvenv.stat().st_size,
            "sha256": sha256_file(pyvenv),
        },
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "python_dont_write_bytecode": True,
        "python_pycache_prefix": str(runner.FROZEN_PYTHONPYCACHEPREFIX),
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_source_record(catnet_root: Path) -> dict[str, Any]:
    if catnet_root.resolve() != legacy.DEFAULT_CATNET_ROOT.resolve():
        raise ValueError("analyzer CAT-Net source root changed")
    _reject_symlink_components(catnet_root, "CAT-Net source root")
    commit = _git_value(catnet_root, "rev-parse", "HEAD")
    tree = _git_value(catnet_root, "rev-parse", "HEAD^{tree}")
    origin = _git_value(catnet_root, "remote", "get-url", "origin")
    status = _git_value(
        catnet_root, "status", "--short", "--untracked-files=all"
    )
    if (
        commit != legacy.MODEL_SOURCE_COMMIT
        or tree != runner.MODEL_TREE
        or origin != runner.MODEL_GIT_ORIGIN
        or status
    ):
        raise ValueError("analyzer CAT-Net source checkout changed")
    bound: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha) in (
        runner.SOURCE_BOUND_FILES.items()
    ):
        path = catnet_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"analyzer source binding changed: {relative}")
        bound[relative] = {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }
    value = {
        "repo_url": legacy.MODEL_REPO_URL,
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bound,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_asset_record(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.resolve() != legacy.DEFAULT_CHECKPOINT.resolve():
        raise ValueError("analyzer checkpoint path changed")
    _reject_symlink_components(checkpoint_path, "CAT-Net checkpoint")
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or checkpoint_path.stat().st_size != legacy.CHECKPOINT_BYTES
        or sha256_file(checkpoint_path) != legacy.CHECKPOINT_SHA256
    ):
        raise ValueError("analyzer CAT_full_v2 checkpoint binding changed")
    value = {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "filename": legacy.CHECKPOINT_FILENAME,
            "provider": "official_author_google_drive",
            "drive_file_id": legacy.CHECKPOINT_DRIVE_ID,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "epoch": legacy.CHECKPOINT_EPOCH,
            "state_keys": legacy.CHECKPOINT_STATE_KEYS,
            "safe_weights_only_load": True,
            "strict_model_load": True,
        }
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    recorded = _require_mapping(value, "runner adapter sources")
    if tuple(recorded) != runner.ADAPTER_SOURCE_PATHS:
        raise ValueError("runner adapter source inventory changed")
    actual = runner.adapter_source_contract(repo_root)
    if stable_json(recorded) != stable_json(actual):
        raise ValueError("runner adapter source hash binding changed")
    return actual


def _validate_runtime(value: Any, *, label: str) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    base_keys = {
        "device",
        "seed",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
        "precision",
        "batch_size",
        "autocast",
        "torch_version",
        "torch_cuda_version",
        "contract_sha256",
    }
    device = _require_string(runtime.get("device"), f"{label}.device")
    expected_keys = (
        base_keys if device == "cpu" else base_keys | {"cuda"}
    )
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} does not have the exact frozen schema")
    stable = {key: child for key, child in runtime.items() if key != "contract_sha256"}
    if runtime.get("contract_sha256") != _fingerprint(stable):
        raise ValueError(f"{label} fingerprint changed")
    expected = {
        "seed": runner.MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": runner.CUBLAS_WORKSPACE_CONFIG,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "torch_version": runner.EXPECTED_PACKAGES["torch"],
        "torch_cuda_version": runner.EXPECTED_TORCH_CUDA_VERSION,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(f"{label}.{key} changed")
    if device == "cpu":
        pass
    elif re.fullmatch(r"cuda:[0-9]+", device):
        cuda = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        logical_index = _require_nonnegative_int(
            cuda.get("logical_device_index"), f"{label}.cuda index"
        )
        if logical_index != int(device[5:]):
            raise ValueError(f"{label}.cuda index differs from device")
        _require_string(cuda.get("device_name"), f"{label}.cuda name")
        if (
            _require_nonnegative_int(
                cuda.get("total_memory_bytes"), f"{label}.cuda memory"
            )
            <= 0
        ):
            raise ValueError(f"{label}.cuda memory is zero")
        capability = cuda.get("compute_capability")
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
            raise ValueError(f"{label}.cuda capability is invalid")
    else:
        raise ValueError(f"{label}.device is invalid")
    return runtime


def _selection_for_mode(
    release: CanonicalRelease,
    mode: str,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if mode == "formal":
        spec = SelectionSpec(capability=Capability.LOCAL_T2_ONLY)
        selected = select_inputs(release, spec)
        if (
            len(selected) != runner.FORMAL_IMAGES
            or Counter(str(row["condition"]) for row in selected)
            != runner.FORMAL_COUNTS
            or _fingerprint([str(row["sample_id"]) for row in selected])
            != FORMAL_SELECTED_IDS_SHA256
            or _fingerprint(selected) != FORMAL_SELECTED_ROWS_SHA256
        ):
            raise ValueError("analyzer formal selection drifted")
        return spec, selected
    if mode == "smoke":
        spec = SelectionSpec(
            capability=Capability.LOCAL_T2_ONLY,
            per_condition_limit=runner.DEFAULT_SMOKE_LIMIT,
        )
        selected = select_inputs(release, spec)
        if (
            len(selected) != runner.SMOKE_IMAGES
            or Counter(str(row["condition"]) for row in selected)
            != runner.SMOKE_COUNTS
            or _fingerprint([str(row["sample_id"]) for row in selected])
            != SMOKE_SELECTED_IDS_SHA256
            or _fingerprint(selected) != SMOKE_SELECTED_ROWS_SHA256
            or any(row.get("panel") is not True for row in selected)
        ):
            raise ValueError("analyzer smoke selection drifted")
        return spec, selected
    raise ValueError("analyzer accepts only formal or smoke runs")


def _validate_cpu_preflight(
    value: Any,
    *,
    repo_root: Path,
    catnet_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    preflight = _require_mapping(value, "CPU preflight")
    if preflight.get("schema_version") != runner.CPU_PREFLIGHT_SCHEMA:
        raise ValueError("CPU preflight schema changed")
    stable = {
        key: child
        for key, child in preflight.items()
        if key != "contract_sha256"
    }
    if preflight.get("contract_sha256") != _fingerprint(stable):
        raise ValueError("CPU preflight fingerprint changed")
    for key, expected in (
        ("cuda_initialized_before", False),
        ("cuda_initialized_after", False),
        ("balanced250_forward_performed", False),
        ("balanced250_score_computed", False),
        ("t1_output_computed", False),
    ):
        if preflight.get(key) is not expected:
            raise ValueError(f"CPU preflight {key} changed")
    if stable_json(preflight.get("environment")) != stable_json(
        _independent_environment_record()
    ):
        raise ValueError("CPU preflight environment record changed")
    if stable_json(preflight.get("source")) != stable_json(
        _independent_source_record(catnet_root)
    ):
        raise ValueError("CPU preflight source record changed")
    if stable_json(preflight.get("assets")) != stable_json(
        _independent_asset_record(checkpoint_path)
    ):
        raise ValueError("CPU preflight asset record changed")
    _verify_adapter_sources(preflight.get("adapter_sources"), repo_root=repo_root)
    checkpoint = _require_mapping(
        preflight.get("checkpoint_audit"), "checkpoint audit"
    )
    if (
        checkpoint.get("state_keys") != legacy.CHECKPOINT_STATE_KEYS
        or checkpoint.get("epoch") != legacy.CHECKPOINT_EPOCH
        or checkpoint.get("load")
        != "torch.load_weights_only_true_with_minimal_numpy_safe_globals"
    ):
        raise ValueError("checkpoint audit changed")
    model = _require_mapping(preflight.get("model_audit"), "model audit")
    for key, expected in (
        ("strict_load", True),
        ("missing_keys", 0),
        ("unexpected_keys", 0),
        ("parameter_count", runner.EXPECTED_MODEL_PARAMETERS),
        ("state_keys", legacy.CHECKPOINT_STATE_KEYS),
        ("device", "cpu"),
        ("eval_mode", True),
        ("forward_performed", False),
    ):
        if model.get(key) != expected:
            raise ValueError(f"CPU model audit {key} changed")
    _require_nonnegative_int(model.get("buffer_elements"), "model buffer count")
    if _require_nonnegative_int(model.get("module_count"), "model module count") <= 0:
        raise ValueError("CPU model audit module count is zero")
    if stable_json(preflight.get("artifact_ignore")) != stable_json(
        runner.verify_artifact_ignore(repo_root)
    ):
        raise ValueError("CPU preflight artifact-ignore evidence changed")
    return preflight


def _validate_immutable_static(
    immutable: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    if immutable.get("schema_version") != runner.RUN_CONFIG_SCHEMA:
        raise ValueError("immutable config schema changed")
    if immutable.get("mode") != mode:
        raise ValueError("immutable mode changed")
    model = _require_mapping(immutable.get("model"), "immutable model")
    expected_model = {
        "name": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "architecture": runner.MODEL_ARCHITECTURE,
        "repo_url": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "source_tree": runner.MODEL_TREE,
        "checkpoint_filename": legacy.CHECKPOINT_FILENAME,
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "checkpoint_bytes": legacy.CHECKPOINT_BYTES,
        "checkpoint_epoch": legacy.CHECKPOINT_EPOCH,
        "checkpoint_state_keys": legacy.CHECKPOINT_STATE_KEYS,
        "checkpoint_strict_load": True,
        "checkpoint_safe_weights_only_load": True,
        "license": runner.LICENSE_RECORD,
    }
    if stable_json(model) != stable_json(expected_model):
        raise ValueError("immutable CAT-Net model contract changed")
    if immutable.get("task_scope") != runner.TASK_SCOPE:
        raise ValueError("immutable CAT-Net task scope changed")
    if immutable.get("t2_spec") != runner.T2_SPEC:
        raise ValueError("immutable CAT-Net T2 spec changed")
    if immutable.get("score_spec") is not None:
        raise ValueError("CAT-Net T2-only run exposes a score spec")
    if immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT:
        raise ValueError("immutable CAT-Net artifact contract changed")
    if immutable.get("resource_expectation") != runner.RESOURCE_EXPECTATION:
        raise ValueError("immutable CAT-Net resource expectation changed")
    inference = _require_mapping(
        immutable.get("inference"), "immutable inference"
    )
    if (
        inference.get("t1_policy") != "unsupported_no_derived_image_score"
        or inference.get("mask_threshold") != runner.MASK_THRESHOLD
        or inference.get("mask_threshold_operator")
        != runner.MASK_THRESHOLD_OPERATOR
        or inference.get("map_restore")
        != (
            "bilinear_logits_to_padded_native_align_corners_false_then_"
            "softmax_channel_1_then_native_crop"
        )
    ):
        raise ValueError("immutable CAT-Net inference contract changed")
    preprocess = _require_mapping(
        immutable.get("preprocess"), "immutable preprocess"
    )
    if (
        preprocess.get("profile") != runner.PREPROCESS_PROFILE
        or preprocess.get("input_resize") != "none"
        or preprocess.get("input_reencode") is not False
        or preprocess.get("jpeg_reader") != "jpegio"
    ):
        raise ValueError("immutable CAT-Net JPEG/DCT protocol changed")


def _expected_result_extras(
    input_row: Mapping[str, Any],
    *,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    return {
        "model": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "valid_for_metrics": valid_for_metrics,
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "task_scope": {
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t2_applicable": True,
            "t2_target_semantics": (
                "all_zero_real_false_positive_area"
                if input_row["condition"] == "real"
                else "exact_diff_local_insertion"
            ),
            "map_statistic_promoted_to_t1": False,
        },
    }


def _validate_attempt(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    run_id: str,
    fingerprint: str,
) -> None:
    forbidden = sorted(_FORBIDDEN_T1_TOP_LEVEL.intersection(row))
    if forbidden:
        raise ValueError(f"result contains forbidden T1 fields: {forbidden}")
    if str(row.get("condition")).startswith("fullframe_"):
        raise ValueError("CAT-Net result contains a full-frame condition")
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result status is invalid")
    identity = build_result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    for key, expected in identity.items():
        if row.get(key) != expected:
            raise ValueError(f"result identity changed at {key}")
    extras = _expected_result_extras(
        input_row, valid_for_metrics=status == "ok"
    )
    for key, expected in extras.items():
        if row.get(key) != expected:
            raise ValueError(f"result CAT-Net field changed at {key}")
    if status == "ok":
        _require_finite(row.get("latency_ms"), "result latency")
        _require_nonnegative_int(
            row.get("peak_cuda_memory_bytes"), "result CUDA peak"
        )
        if (
            row.get("mask_threshold") != runner.MASK_THRESHOLD
            or row.get("mask_threshold_operator")
            != runner.MASK_THRESHOLD_OPERATOR
        ):
            raise ValueError("result mask threshold changed")
    else:
        for key in (
            "raw_logits_path",
            "score_map_path",
            "mask_path",
            "localization",
            "latency_ms",
            "peak_cuda_memory_bytes",
        ):
            if key in row:
                raise ValueError(f"error result contains {key}")
        _require_string(row.get("error_type"), "error type")
        _require_string(row.get("error"), "error text")


def _validate_history(
    selected: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {str(row["sample_id"]) for row in selected}
    histories: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected:
            raise ValueError(f"result row {index} has unexpected sample_id")
        histories.setdefault(sample_id, []).append(str(row.get("status")))
    recovered = 0
    for sample_id, statuses in histories.items():
        if any(status not in ("ok", "error") for status in statuses):
            raise ValueError(f"invalid history status for {sample_id}")
        if "ok" in statuses[:-1] or statuses.count("ok") > 1:
            raise ValueError(f"attempt exists after success for {sample_id}")
        recovered += int(statuses[-1] == "ok" and "error" in statuses[:-1])
    return {
        "physical_attempts": len(rows),
        "unique_sample_ids": len(histories),
        "superseded_attempts": len(rows) - len(histories),
        "recovered_error_to_ok": recovered,
        "success_is_terminal": True,
        "append_only": True,
    }


def _capture_snapshot(
    paths: Sequence[Path],
) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        result[str(path)] = (path.stat().st_size, sha256_file(path))
    return result


def _verify_bundle_unchanged(bundle: RunBundle) -> None:
    current = _capture_snapshot(
        [
            bundle.manifest_path,
            bundle.expected_path,
            bundle.results_path,
            bundle.summary_path,
        ]
    )
    if current != dict(bundle.snapshot):
        raise ValueError(f"run changed during analysis: {bundle.run_id}")


def _validate_run_directory(run_dir: Path) -> None:
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_audit.json",
        "metrics.json",
        "fresh_replay.json",
    }
    entries = list(run_dir.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("run directory contains a symlink")
    unexpected = {path.name for path in entries} - allowed
    if unexpected:
        raise ValueError(
            f"run directory contains unexpected files: {sorted(unexpected)}"
        )
    required = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
    }
    if not required.issubset({path.name for path in entries}):
        raise ValueError("run directory lacks required files")


def load_run(
    *,
    repo_root: Path,
    run_id: str,
    expected_mode: str,
    release: CanonicalRelease,
    results_root: Path,
    artifacts_root: Path,
    catnet_root: Path = legacy.DEFAULT_CATNET_ROOT,
    checkpoint_path: Path = legacy.DEFAULT_CHECKPOINT,
) -> RunBundle:
    run_dir = _resolve_run_dir(results_root, run_id, "CAT-Net run")
    artifact_root = _resolve_run_dir(
        artifacts_root, run_id, "CAT-Net artifacts"
    )
    _validate_run_directory(run_dir)
    manifest_path = run_dir / "manifest.json"
    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "run manifest")
    summary = _load_json(summary_path, "runtime summary")
    expected_rows = _read_jsonl(expected_path, "expected inputs")
    physical = _read_jsonl(results_path, "physical results")
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
        or set(manifest)
        != {
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
    ):
        raise ValueError("run manifest envelope changed or is incomplete")
    _require_string(manifest.get("started_at"), "manifest started_at")
    _require_string(manifest.get("completed_at"), "manifest completed_at")
    immutable = _require_mapping(
        manifest.get("immutable"), "immutable run config"
    )
    fingerprint = _require_sha256(
        manifest.get("fingerprint"), "run fingerprint"
    )
    if fingerprint != _fingerprint(immutable):
        raise ValueError("run manifest fingerprint does not recompute")
    if immutable.get("run_id") != run_id:
        raise ValueError("immutable run ID changed")
    _validate_immutable_static(immutable, mode=expected_mode)
    _validate_runtime(immutable.get("runtime"), label="recorded runtime")
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        repo_root=repo_root,
        catnet_root=catnet_root,
        checkpoint_path=checkpoint_path,
    )
    _verify_adapter_sources(
        immutable.get("adapter_sources"), repo_root=repo_root
    )

    spec, selected = _selection_for_mode(release, expected_mode)
    if stable_json(expected_rows) != stable_json(selected):
        raise ValueError("expected_inputs.jsonl does not match frozen selection")
    contract = build_run_dataset_contract(
        release, spec, selected, score_spec=None
    )
    if stable_json(immutable.get("dataset_contract")) != stable_json(
        contract.as_dict()
    ):
        raise ValueError("immutable dataset contract changed")
    selection = _require_mapping(
        immutable.get("selection"), "immutable selection"
    )
    expected_selection = {
        "selected_images": len(selected),
        "selected_ids_sha256": _fingerprint(
            [str(row["sample_id"]) for row in selected]
        ),
        "selected_rows_sha256": _fingerprint(selected),
        "counts_by_condition": dict(
            sorted(Counter(str(row["condition"]) for row in selected).items())
        ),
    }
    if stable_json(selection) != stable_json(expected_selection):
        raise ValueError("immutable selection binding changed")
    expected_immutable = runner.build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=expected_mode,
        dataset_contract=contract.as_dict(),
        selected=selected,
        cpu_preflight=_require_mapping(
            immutable.get("cpu_preflight"), "immutable CPU preflight"
        ),
        runtime=_require_mapping(
            immutable.get("runtime"), "immutable runtime"
        ),
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    if stable_json(immutable) != stable_json(expected_immutable):
        raise ValueError("immutable run config cannot be reconstructed exactly")

    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    expected_dataset = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t1_applicable_images": 0,
        "t2_applicable_images": len(selected),
        "fullframe_selected_images": 0,
    }
    if stable_json(dataset) != stable_json(expected_dataset):
        raise ValueError("manifest dataset envelope changed")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    immutable_outputs = _require_mapping(
        immutable.get("outputs"), "immutable outputs"
    )
    expected_outputs = {
        "results_path": repo_relative(results_path, repo_root),
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "summary_path": repo_relative(summary_path, repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
    }
    if stable_json(immutable_outputs) != stable_json(expected_outputs):
        raise ValueError("immutable output paths changed")
    if set(outputs) != {
        *expected_outputs,
        "results_sha256",
        "summary_sha256",
        "artifact_inventory",
    }:
        raise ValueError("manifest outputs do not have the exact schema")
    for key, expected_value in expected_outputs.items():
        if outputs.get(key) != expected_value:
            raise ValueError(f"manifest output path changed: {key}")
    if outputs.get("results_sha256") != sha256_file(results_path):
        raise ValueError("manifest results SHA-256 changed")
    if outputs.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("manifest summary SHA-256 changed")

    history = _validate_history(selected, physical)
    latest = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=None,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for row in physical:
        _validate_attempt(
            row,
            inputs_by_id[str(row["sample_id"])],
            run_id=run_id,
            fingerprint=fingerprint,
        )
    coverage = summarize_coverage(latest)
    coverage.require_complete()
    latest_rows = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])])
        for row in selected
    )
    inventory = runner.validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    if stable_json(outputs.get("artifact_inventory")) != stable_json(
        inventory
    ):
        raise ValueError("manifest artifact inventory changed")
    execution = _require_mapping(
        manifest.get("execution"), "manifest execution"
    )
    if set(execution) != {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
    }:
        raise ValueError("manifest execution does not have the exact schema")
    new_successes = _require_nonnegative_int(
        execution.get("new_successes"), "execution new_successes"
    )
    resume_skips = _require_nonnegative_int(
        execution.get("resume_skips"), "execution resume_skips"
    )
    new_errors = _require_nonnegative_int(
        execution.get("new_errors"), "execution new_errors"
    )
    physical_result_rows = _require_nonnegative_int(
        execution.get("physical_result_rows"),
        "execution physical_result_rows",
    )
    latest_result_rows = _require_nonnegative_int(
        execution.get("latest_result_rows"),
        "execution latest_result_rows",
    )
    superseded_attempts = _require_nonnegative_int(
        execution.get("superseded_attempts"),
        "execution superseded_attempts",
    )
    if (
        new_errors != 0
        or new_successes + resume_skips != len(selected)
        or physical_result_rows != len(physical)
        or latest_result_rows != len(latest.latest_by_sample_id)
        or superseded_attempts != latest.superseded_attempts
    ):
        raise ValueError("manifest execution accounting changed")
    disk = _require_mapping(
        manifest.get("disk_preflight"), "manifest disk preflight"
    )
    if (
        set(disk)
        != {
            "pending_images",
            "estimated_artifact_bytes",
            "reserve_bytes",
            "required_available_bytes",
            "available_bytes",
            "passed",
        }
        or disk.get("reserve_bytes") != runner.MIN_DISK_RESERVE_BYTES
        or disk.get("passed") is not True
    ):
        raise ValueError("manifest disk preflight changed")
    if (
        _require_nonnegative_int(
            disk.get("pending_images"), "disk pending images"
        )
        != new_successes
    ):
        raise ValueError("manifest disk pending-image count changed")
    estimated = _require_nonnegative_int(
        disk.get("estimated_artifact_bytes"),
        "disk estimated artifact bytes",
    )
    required = _require_nonnegative_int(
        disk.get("required_available_bytes"),
        "disk required available bytes",
    )
    available = _require_nonnegative_int(
        disk.get("available_bytes"), "disk available bytes"
    )
    if (
        required != estimated + runner.MIN_DISK_RESERVE_BYTES
        or available < required
    ):
        raise ValueError("manifest disk capacity accounting changed")
    expected_summary_keys = {
        "schema_version",
        "summary_kind",
        "scientific_metrics",
        "scientific_metrics_owner",
        "run_id",
        "run_manifest_fingerprint",
        "status",
        "mode",
        "model",
        "model_slug",
        "score_spec",
        "task_scope",
        "t2_spec",
        "dataset_contract",
        "coverage",
        "attempt_history",
        "artifact_inventory",
        "generated_at",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != runner.RUNTIME_SUMMARY_SCHEMA
        or summary.get("summary_kind")
        != "runtime_coverage_and_artifact_inventory_only"
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != fingerprint
        or summary.get("status") != "complete"
        or summary.get("mode") != expected_mode
        or summary.get("score_spec") is not None
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner")
        != "analyze_catnet_balanced.py"
        or summary.get("model") != runner.MODEL_NAME
        or summary.get("model_slug") != runner.MODEL_SLUG
    ):
        raise ValueError("runtime summary envelope changed")
    _require_string(summary.get("generated_at"), "summary generated_at")
    if stable_json(summary.get("dataset_contract")) != stable_json(
        contract.as_dict()
    ):
        raise ValueError("runtime summary dataset contract changed")
    if stable_json(summary.get("coverage")) != stable_json(coverage.as_dict()):
        raise ValueError("runtime summary coverage changed")
    if stable_json(summary.get("attempt_history")) != stable_json(history):
        raise ValueError("runtime summary attempt history changed")
    if stable_json(summary.get("artifact_inventory")) != stable_json(
        inventory
    ):
        raise ValueError("runtime summary artifact inventory changed")
    if summary.get("task_scope") != runner.TASK_SCOPE:
        raise ValueError("runtime summary task scope changed")
    if summary.get("t2_spec") != runner.T2_SPEC:
        raise ValueError("runtime summary T2 spec changed")
    snapshot = _capture_snapshot(
        [manifest_path, expected_path, results_path, summary_path]
    )
    return RunBundle(
        repo_root=repo_root,
        run_id=run_id,
        mode=expected_mode,
        run_dir=run_dir,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
        release=release,
        selected=tuple(selected),
        dataset_contract=contract,
        physical_results=tuple(physical),
        latest_results=latest_rows,
        fingerprint=fingerprint,
        snapshot=snapshot,
    )


def load_formal_run(**kwargs: Any) -> RunBundle:
    return load_run(expected_mode="formal", **kwargs)


def load_smoke_run(**kwargs: Any) -> RunBundle:
    return load_run(expected_mode="smoke", **kwargs)


def _independent_jpeg_evidence(
    path: Path,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    import jpegio

    with Image.open(path) as opened:
        if (
            opened.format != "JPEG"
            or opened.mode != "RGB"
            or opened.size != (width, height)
        ):
            raise ValueError("canonical CAT-Net JPEG decode contract changed")
    jpeg = jpegio.read(str(path))
    if not jpeg.coef_arrays or not jpeg.comp_info:
        raise ValueError("jpegio returned no CAT-Net JPEG components")
    sampling = [
        [int(component.h_samp_factor), int(component.v_samp_factor)]
        for component in jpeg.comp_info
    ]
    if sampling[:3] != [[1, 1], [1, 1], [1, 1]]:
        raise ValueError("canonical CAT-Net JPEG is not 4:4:4")
    coefficients = np.asarray(jpeg.coef_arrays[0], dtype=np.int32)
    padded_width = math.ceil(width / 8) * 8
    padded_height = math.ceil(height / 8) * 8
    if coefficients.shape != (padded_height, padded_width):
        raise ValueError("canonical CAT-Net Y-DCT shape changed")
    qtable_index = int(jpeg.comp_info[0].quant_tbl_no)
    if qtable_index < 0 or qtable_index >= len(jpeg.quant_tables):
        raise ValueError("canonical CAT-Net qtable index is invalid")
    qtable = np.asarray(jpeg.quant_tables[qtable_index], dtype=np.int32)
    if qtable.shape != (8, 8):
        raise ValueError("canonical CAT-Net qtable shape changed")
    return {
        "native_size": [width, height],
        "padded_size": [padded_width, padded_height],
        "padding": {
            "left": 0,
            "top": 0,
            "right": padded_width - width,
            "bottom": padded_height - height,
        },
        "jpeg_sampling_factors": sampling,
        "luminance_qtable_index": qtable_index,
        "qtable_sha256": _array_sha256(qtable),
        "dct_y_sha256": _array_sha256(coefficients),
    }


def _bilinear_align_corners_false(
    value: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Pure NumPy half-pixel bilinear resize matching PyTorch geometry."""

    source = np.asarray(value, dtype=np.float32)
    if (
        source.ndim != 2
        or source.size == 0
        or not np.isfinite(source).all()
        or width <= 0
        or height <= 0
    ):
        raise ValueError("CAT-Net independent bilinear input is invalid")
    source_height, source_width = source.shape
    if source.shape == (height, width):
        return np.ascontiguousarray(source)
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        source_width / width
    ) - np.float32(0.5)
    y = (np.arange(height, dtype=np.float32) + np.float32(0.5)) * np.float32(
        source_height / height
    ) - np.float32(0.5)
    x = np.maximum(x, np.float32(0.0))
    y = np.maximum(y, np.float32(0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]
    horizontal = (
        source[:, x0] * (np.float32(1.0) - wx)
        + source[:, x1] * wx
    )
    restored = (
        horizontal[y0, :] * (np.float32(1.0) - wy)
        + horizontal[y1, :] * wy
    )
    return np.ascontiguousarray(restored, dtype=np.float32)


def _softmax_channel1_float32(
    channel0: np.ndarray,
    channel1: np.ndarray,
) -> np.ndarray:
    first = np.asarray(channel0, dtype=np.float32)
    second = np.asarray(channel1, dtype=np.float32)
    if (
        first.shape != second.shape
        or first.ndim != 2
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("CAT-Net independent softmax input is invalid")
    maximum = np.maximum(first, second)
    first_exp = np.exp(first - maximum)
    second_exp = np.exp(second - maximum)
    probability = second_exp / (first_exp + second_exp)
    return np.ascontiguousarray(probability, dtype=np.float32)


def _independent_restore_from_logits(
    raw_logits: np.ndarray,
    *,
    native_width: int,
    native_height: int,
) -> np.ndarray:
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 2:
        raise ValueError("CAT-Net raw logits have the wrong independent shape")
    padded_width = math.ceil(native_width / 8) * 8
    padded_height = math.ceil(native_height / 8) * 8
    expected = (2, padded_height // 4, padded_width // 4)
    if raw.shape != expected:
        raise ValueError("CAT-Net raw logits do not match native ceil-8 geometry")
    restored0 = _bilinear_align_corners_false(
        raw[0], width=padded_width, height=padded_height
    )
    restored1 = _bilinear_align_corners_false(
        raw[1], width=padded_width, height=padded_height
    )
    probability = _softmax_channel1_float32(restored0, restored1)
    return np.ascontiguousarray(
        probability[:native_height, :native_width],
        dtype=np.float32,
    )


def _exact_artifact_path(
    bundle: RunBundle,
    value: Any,
    *,
    directory: str,
    filename: str,
    label: str,
) -> Path:
    expected = bundle.artifact_root / directory / filename
    return _safe_repo_file(
        bundle.repo_root, value, expected=expected, label=label
    )


def _load_npy_record(
    bundle: RunBundle,
    row: Mapping[str, Any],
    *,
    prefix: str,
    directory: str,
    filename: str,
    shape: tuple[int, ...],
    bounded: bool,
) -> tuple[Path, np.ndarray]:
    path = _exact_artifact_path(
        bundle,
        row.get(f"{prefix}_path"),
        directory=directory,
        filename=filename,
        label=prefix,
    )
    if sha256_file(path) != _require_sha256(
        row.get(f"{prefix}_sha256"), f"{prefix} SHA-256"
    ):
        raise ValueError(f"{prefix} file SHA-256 changed")
    if row.get(f"{prefix}_bytes") != path.stat().st_size:
        raise ValueError(f"{prefix} byte count changed")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != shape
        or array.dtype != np.float32
        or row.get(f"{prefix}_shape") != list(shape)
        or row.get(f"{prefix}_dtype") != "float32"
    ):
        raise ValueError(f"{prefix} shape/dtype changed")
    if not np.isfinite(array).all():
        raise ValueError(f"{prefix} contains non-finite values")
    if bounded and (
        float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0
    ):
        raise ValueError(f"{prefix} falls outside [0,1]")
    if row.get(f"{prefix}_array_sha256") != _array_sha256(array):
        raise ValueError(f"{prefix} array SHA-256 changed")
    return path, array


def _audit_artifact_row(
    bundle: RunBundle,
    input_row: Mapping[str, Any],
    result_row: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    width = int(input_row["width"])
    height = int(input_row["height"])
    canonical = _safe_repo_file(
        bundle.repo_root,
        input_row.get("canonical_path"),
        expected=None,
        label=f"canonical input {sample_id}",
    )
    if sha256_file(canonical) != input_row.get("canonical_sha256"):
        raise ValueError(f"canonical input SHA-256 changed: {sample_id}")
    jpeg = _independent_jpeg_evidence(
        canonical, width=width, height=height
    )
    if stable_json(result_row.get("preprocess")) != stable_json(jpeg):
        raise ValueError(f"preprocess evidence changed: {sample_id}")
    if (
        result_row.get("qtable_sha256") != jpeg["qtable_sha256"]
        or result_row.get("dct_y_sha256") != jpeg["dct_y_sha256"]
    ):
        raise ValueError(f"JPEG evidence hash changed: {sample_id}")

    padded_width, padded_height = jpeg["padded_size"]
    raw_path, raw = _load_npy_record(
        bundle,
        result_row,
        prefix="raw_logits",
        directory="raw_logits_quarter",
        filename=f"{sample_id}.npy",
        shape=(2, padded_height // 4, padded_width // 4),
        bounded=False,
    )
    map_path, score_map = _load_npy_record(
        bundle,
        result_row,
        prefix="score_map",
        directory="score_maps_native",
        filename=f"{sample_id}.npy",
        shape=(height, width),
        bounded=True,
    )
    independently_restored = _independent_restore_from_logits(
        raw,
        native_width=width,
        native_height=height,
    )
    restore_max_abs_difference = float(
        np.max(
            np.abs(
                independently_restored.astype(np.float64)
                - score_map.astype(np.float64)
            )
        )
    )
    if restore_max_abs_difference > STATIC_LOGIT_RESTORE_ABS_TOLERANCE:
        raise ValueError(
            f"independent logits-to-native replay changed: {sample_id}"
        )
    mask_path = _exact_artifact_path(
        bundle,
        result_row.get("mask_path"),
        directory="masks_native",
        filename=f"{sample_id}.png",
        label="native mask",
    )
    if sha256_file(mask_path) != _require_sha256(
        result_row.get("mask_sha256"), "mask SHA-256"
    ):
        raise ValueError(f"mask SHA-256 changed: {sample_id}")
    if result_row.get("mask_bytes") != mask_path.stat().st_size:
        raise ValueError(f"mask byte count changed: {sample_id}")
    with Image.open(mask_path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError(f"mask format/mode changed: {sample_id}")
        mask = np.asarray(opened, dtype=np.uint8)
    if (
        mask.shape != (height, width)
        or result_row.get("mask_shape") != [height, width]
        or result_row.get("mask_dtype") != "uint8"
        or not set(np.unique(mask).tolist()).issubset({0, 255})
        or result_row.get("mask_array_sha256") != _array_sha256(mask)
    ):
        raise ValueError(f"mask schema changed: {sample_id}")
    expected_mask = np.where(
        np.asarray(score_map) >= runner.MASK_THRESHOLD,
        np.uint8(255),
        np.uint8(0),
    )
    if not np.array_equal(mask, expected_mask):
        raise ValueError(f"mask is not score_map >= 0.5: {sample_id}")
    if (
        result_row.get("raw_logits_semantics")
        != "official_two_channel_quarter_resolution_logits"
        or result_row.get("score_map_semantics")
        != "native_probability_of_channel_1_tampered"
        or result_row.get("mask_semantics")
        != "score_map_greater_than_or_equal_to_0_5"
    ):
        raise ValueError(f"artifact semantics changed: {sample_id}")
    target = load_ground_truth(input_row, bundle.repo_root)
    if target is None:
        raise ValueError(f"selected CAT-Net row has no T2 GT: {sample_id}")
    metrics = binary_pixel_metrics(
        np.asarray(score_map),
        np.asarray(target, dtype=bool),
        runner.MASK_THRESHOLD,
        include_ap=input_row["condition"] != "real",
    )
    localization = _require_mapping(
        result_row.get("localization"), f"localization {sample_id}"
    )
    if stable_json(localization.get("native")) != stable_json(metrics):
        raise ValueError(f"localization metrics changed: {sample_id}")
    checked = 4
    if input_row["condition"] != "real":
        mask_value = _require_string(
            input_row.get("gt_mask_path"), f"GT path {sample_id}"
        )
        gt_path = _safe_repo_file(
            bundle.repo_root,
            mask_value,
            expected=None,
            label=f"GT mask {sample_id}",
        )
        if sha256_file(gt_path) != input_row.get("gt_mask_sha256"):
            raise ValueError(f"GT mask SHA-256 changed: {sample_id}")
        checked += 1
    value = {
        "sample_id": sample_id,
        "condition": input_row["condition"],
        "checked_files": checked,
        "canonical_sha256": sha256_file(canonical),
        "qtable_sha256": jpeg["qtable_sha256"],
        "dct_y_sha256": jpeg["dct_y_sha256"],
        "raw_logits_path": repo_relative(raw_path, bundle.repo_root),
        "raw_logits_sha256": sha256_file(raw_path),
        "raw_logits_array_sha256": _array_sha256(raw),
        "score_map_path": repo_relative(map_path, bundle.repo_root),
        "score_map_sha256": sha256_file(map_path),
        "score_map_array_sha256": _array_sha256(score_map),
        "mask_path": repo_relative(mask_path, bundle.repo_root),
        "mask_sha256": sha256_file(mask_path),
        "mask_array_sha256": _array_sha256(mask),
        "localization_sha256": _fingerprint(metrics),
        "logits_to_native_max_abs_difference": restore_max_abs_difference,
    }
    del raw, score_map, independently_restored
    return value


def _exact_directory_inventory(bundle: RunBundle) -> dict[str, Any]:
    if (
        not bundle.artifact_root.is_dir()
        or bundle.artifact_root.is_symlink()
    ):
        raise ValueError("artifact root is invalid")
    root_entries = list(bundle.artifact_root.iterdir())
    if (
        {path.name for path in root_entries}
        != set(runner.ARTIFACT_DIRECTORIES)
        or any(path.is_symlink() or not path.is_dir() for path in root_entries)
    ):
        raise ValueError("artifact root inventory changed")
    expected_ids = {str(row["sample_id"]) for row in bundle.selected}
    expected = {
        "raw_logits_quarter": {
            f"{sample_id}.npy" for sample_id in expected_ids
        },
        "score_maps_native": {
            f"{sample_id}.npy" for sample_id in expected_ids
        },
        "masks_native": {f"{sample_id}.png" for sample_id in expected_ids},
    }
    actual: dict[str, set[str]] = {}
    bytes_by_directory: dict[str, int] = {}
    for directory in runner.ARTIFACT_DIRECTORIES:
        root = bundle.artifact_root / directory
        entries = list(root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError(f"{directory} contains a non-regular file")
        actual[directory] = {path.name for path in entries}
        bytes_by_directory[directory] = sum(
            path.stat().st_size for path in entries
        )
        if actual[directory] != expected[directory]:
            raise ValueError(f"{directory} artifact inventory changed")
    value = {
        "successful_images": len(expected_ids),
        "files": sum(len(names) for names in actual.values()),
        "files_by_directory": {
            name: len(actual[name]) for name in runner.ARTIFACT_DIRECTORIES
        },
        "bytes_by_directory": bytes_by_directory,
        "total_bytes": sum(bytes_by_directory.values()),
        "exact_inventory": True,
    }
    return {**value, "inventory_sha256": _fingerprint(value)}


def audit_artifacts(bundle: RunBundle) -> dict[str, Any]:
    if bundle.mode not in ("formal", "smoke"):
        raise ValueError("unsupported bundle mode for artifact audit")
    input_by_id = {
        str(row["sample_id"]): row for row in bundle.selected
    }
    rows: list[dict[str, Any]] = []
    for result in bundle.latest_results:
        sample_id = str(result["sample_id"])
        rows.append(
            _audit_artifact_row(bundle, input_by_id[sample_id], result)
        )
    inventory = _exact_directory_inventory(bundle)
    if stable_json(bundle.summary.get("artifact_inventory")) != stable_json(
        inventory
    ):
        raise ValueError("runtime artifact inventory changed")
    manifest_inventory = _require_mapping(
        bundle.manifest.get("outputs"), "manifest outputs"
    ).get("artifact_inventory")
    if stable_json(manifest_inventory) != stable_json(inventory):
        raise ValueError("manifest artifact inventory changed")
    report = {
        "schema_version": ARTIFACT_AUDIT_SCHEMA,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "status": "ok",
        "selected_images": len(bundle.selected),
        "condition_counts": dict(
            sorted(
                Counter(
                    str(row["condition"]) for row in bundle.selected
                ).items()
            )
        ),
        "t1_results": 0,
        "t2_results": len(bundle.selected),
        "fullframe_results": 0,
        "checked_files": sum(row["checked_files"] for row in rows),
        "artifact_inventory": inventory,
        "artifact_rows_sha256": _fingerprint(rows),
        "independent_logits_to_native_replays": len(rows),
        "independent_logits_to_native_max_abs_difference": max(
            float(row["logits_to_native_max_abs_difference"])
            for row in rows
        ),
        "independent_logits_to_native_abs_tolerance": (
            STATIC_LOGIT_RESTORE_ABS_TOLERANCE
        ),
        "checks": [
            "canonical JPEG SHA-256, RGB/JPEG geometry, and 4:4:4 sampling",
            "independent luminance qtable and quantized Y-DCT hashes",
            "finite float32 quarter logits with exact ceil-8-derived shape",
            "finite float32 native score maps bounded by [0,1]",
            (
                "pure NumPy bilinear-logits then float32-softmax channel-1 "
                "native restoration"
            ),
            "binary PNG masks bit-exactly equal score_map >= 0.5",
            "exact-difference GT hashes and independently replayed T2 metrics",
            "all-and-only 3 artifacts for every selected applicable input",
            "no full-frame row, artifact, T1 score, or map-derived decision",
        ],
        "sources": {
            "manifest_sha256": sha256_file(bundle.manifest_path),
            "expected_inputs_sha256": sha256_file(bundle.expected_path),
            "results_sha256": sha256_file(bundle.results_path),
            "summary_sha256": sha256_file(bundle.summary_path),
            "analyzer_sources": analyzer_source_contract(bundle.repo_root),
        },
        "generated_at": utc_now(),
    }
    _verify_bundle_unchanged(bundle)
    return report


def _native_map_loader(bundle: RunBundle) -> Callable[
    [Mapping[str, Any], Mapping[str, Any]], np.ndarray
]:
    def load_map(
        input_row: Mapping[str, Any],
        result_row: Mapping[str, Any],
    ) -> np.ndarray:
        sample_id = str(input_row["sample_id"])
        path = _exact_artifact_path(
            bundle,
            result_row.get("score_map_path"),
            directory="score_maps_native",
            filename=f"{sample_id}.npy",
            label=f"metric score map {sample_id}",
        )
        if sha256_file(path) != result_row.get("score_map_sha256"):
            raise ValueError(f"metric score-map SHA-256 changed: {sample_id}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if result_row.get("score_map_array_sha256") != _array_sha256(value):
            raise ValueError(
                f"metric score-map array SHA-256 changed: {sample_id}"
            )
        return value

    return load_map


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if bundle.mode != "formal":
        raise ValueError("scientific metrics require the formal CAT-Net run")
    if (
        len(bundle.selected) != runner.FORMAL_IMAGES
        or Counter(str(row["condition"]) for row in bundle.selected)
        != runner.FORMAL_COUNTS
    ):
        raise ValueError("CAT-Net metrics require formal 1025 coverage")
    metrics = summarize_balanced250_t2(
        list(bundle.release.inputs),
        list(bundle.latest_results),
        repo_root=bundle.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.dataset_contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name="catnet_channel_1_native_probability",
        threshold=runner.MASK_THRESHOLD,
        threshold_operator=runner.MASK_THRESHOLD_OPERATOR,
        iterations=iterations,
        seed=seed,
    )
    coverage = _require_mapping(metrics.get("coverage"), "T2 coverage")
    if (
        coverage.get("selected_results") != runner.FORMAL_IMAGES
        or coverage.get("native_maps_evaluated") != runner.FORMAL_IMAGES
        or coverage.get("all_zero_real_images") != 275
        or coverage.get("exact_diff_local_images") != 750
        or coverage.get("not_applicable_selected_images") != 0
    ):
        raise ValueError("CAT-Net formal T2 metric coverage changed")
    excluded = _require_mapping(
        metrics.get("excluded_not_applicable"), "T2 excluded conditions"
    )
    if (
        excluded.get("selected_images") != 0
        or excluded.get("score_map_loader_calls") != 0
        or excluded.get("counts_by_condition")
        != {
            "fullframe_mouse": 0,
            "fullframe_cat": 0,
            "fullframe_trash_can": 0,
        }
    ):
        raise ValueError("CAT-Net full-frame N/A metric contract changed")
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "task_scope": runner.TASK_SCOPE,
        "score_spec": None,
        "metrics": metrics,
        "sources": {
            "dataset_manifest_sha256": bundle.release.manifest_sha256,
            "results_sha256": sha256_file(bundle.results_path),
            "manifest_sha256": sha256_file(bundle.manifest_path),
            "summary_sha256": sha256_file(bundle.summary_path),
            "analyzer_sources": analyzer_source_contract(bundle.repo_root),
        },
        "generated_at": utc_now(),
    }
    _verify_bundle_unchanged(bundle)
    return report


def _computational_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok":
        raise ValueError("computational projection requires status=ok")
    return {key: row.get(key) for key in _COMPUTATIONAL_RESULT_FIELDS}


def compare_computational_results(
    left: RunBundle,
    right: RunBundle,
) -> dict[str, Any]:
    left_rows = {
        str(row["sample_id"]): row for row in left.latest_results
    }
    right_rows = {
        str(row["sample_id"]): row for row in right.latest_results
    }
    if set(left_rows) != set(right_rows):
        raise ValueError("computational comparison ID sets differ")
    mismatches: list[str] = []
    for sample_id in sorted(left_rows):
        if stable_json(_computational_projection(left_rows[sample_id])) != (
            stable_json(_computational_projection(right_rows[sample_id]))
        ):
            mismatches.append(sample_id)
    if mismatches:
        raise ValueError(
            f"CAT-Net computational results differ: {mismatches[:5]}"
        )
    return {
        "images_compared": len(left_rows),
        "fields_compared_per_image": len(_COMPUTATIONAL_RESULT_FIELDS),
        "mismatches": 0,
        "bit_exact": True,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def compare_smoke_runs(
    smoke_a: RunBundle,
    smoke_b: RunBundle,
) -> dict[str, Any]:
    if (
        smoke_a.mode != "smoke"
        or smoke_b.mode != "smoke"
        or smoke_a.run_id != DEFAULT_SMOKE_RUN_ID_A
        or smoke_b.run_id != DEFAULT_SMOKE_RUN_ID_B
    ):
        raise ValueError("smoke comparison requires the frozen A/B runs")
    if len(smoke_a.selected) != runner.SMOKE_IMAGES or len(
        smoke_b.selected
    ) != runner.SMOKE_IMAGES:
        raise ValueError("smoke A/B must each contain exactly 20 images")
    if [row["sample_id"] for row in smoke_a.selected] != [
        row["sample_id"] for row in smoke_b.selected
    ]:
        raise ValueError("smoke A/B selections differ")
    left_immutable = _require_mapping(
        smoke_a.manifest.get("immutable"), "smoke A immutable"
    )
    right_immutable = _require_mapping(
        smoke_b.manifest.get("immutable"), "smoke B immutable"
    )
    if stable_json(_smoke_immutable_projection(left_immutable)) != stable_json(
        _smoke_immutable_projection(right_immutable)
    ):
        raise ValueError("smoke A/B immutable computational configs differ")
    audit_a = audit_artifacts(smoke_a)
    audit_b = audit_artifacts(smoke_b)
    for label, audit in (("A", audit_a), ("B", audit_b)):
        inventory = _require_mapping(
            audit.get("artifact_inventory"), f"smoke {label} inventory"
        )
        if (
            audit.get("status") != "ok"
            or audit.get("selected_images") != runner.SMOKE_IMAGES
            or audit.get("independent_logits_to_native_replays")
            != runner.SMOKE_IMAGES
            or inventory.get("files") != runner.SMOKE_IMAGES * 3
        ):
            raise ValueError(f"smoke {label} physical artifact audit changed")
    comparison = compare_computational_results(smoke_a, smoke_b)
    _verify_bundle_unchanged(smoke_a)
    _verify_bundle_unchanged(smoke_b)
    return {
        "schema_version": SMOKE_COMPARISON_SCHEMA,
        "smoke_a_run_id": smoke_a.run_id,
        "smoke_b_run_id": smoke_b.run_id,
        "status": "pass",
        "selection": {
            "images": runner.SMOKE_IMAGES,
            "counts_by_condition": runner.SMOKE_COUNTS,
            "fullframe_images": 0,
            "selection_sha256": _fingerprint(
                [str(row["sample_id"]) for row in smoke_a.selected]
            ),
        },
        "comparison": comparison,
        "physical_artifact_audits_passed": 2,
        "artifact_files_audited": runner.SMOKE_IMAGES * 3 * 2,
        "artifact_files_compared_byte_exact": runner.SMOKE_IMAGES * 3,
        "t1_fields_compared": 0,
        "map_statistic_promoted_to_t1": False,
        "generated_at": utc_now(),
    }


def validate_smoke_gate(
    formal: RunBundle,
    smoke_a: RunBundle,
    smoke_b: RunBundle,
) -> dict[str, Any]:
    ab = compare_smoke_runs(smoke_a, smoke_b)
    formal_by_id = {
        str(row["sample_id"]): row for row in formal.latest_results
    }
    for smoke in (smoke_a, smoke_b):
        smoke_by_id = {
            str(row["sample_id"]): row for row in smoke.latest_results
        }
        projected_formal = RunBundle(
            **{
                **formal.__dict__,
                "latest_results": tuple(
                    formal_by_id[sample_id]
                    for sample_id in smoke_by_id
                ),
            }
        )
        projected_smoke = RunBundle(
            **{
                **smoke.__dict__,
                "latest_results": tuple(
                    smoke_by_id[sample_id] for sample_id in smoke_by_id
                ),
            }
        )
        compare_computational_results(projected_smoke, projected_formal)
    return {
        "status": "pass",
        "smoke_ab": ab,
        "formal_overlap_images_per_smoke": runner.SMOKE_IMAGES,
        "formal_overlap_bit_exact": True,
    }


def _configure_recorded_runtime(bundle: RunBundle) -> tuple[Any, dict[str, Any]]:
    recorded = _validate_runtime(
        _require_mapping(
            bundle.manifest.get("immutable"), "immutable"
        ).get("runtime"),
        label="recorded runtime",
    )
    device_text = str(recorded["device"])
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, runner.CUBLAS_WORKSPACE_CONFIG):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with replay")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = runner.CUBLAS_WORKSPACE_CONFIG

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before fresh replay configuration")
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recorded CUDA device is unavailable")
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError("recorded CUDA device index is unavailable")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        cuda = _require_mapping(recorded.get("cuda"), "recorded CUDA")
        observed = {
            "logical_device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
        if observed != cuda:
            raise ValueError("fresh replay CUDA identity differs from formal run")
    random.seed(runner.MODEL_SEED)
    np.random.seed(runner.MODEL_SEED)
    torch.manual_seed(runner.MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(runner.MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device, recorded


def _fresh_cpu_strict_gate(
    *,
    repo_root: Path,
    catnet_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Independently strict-load a fresh CPU model before accelerator replay."""

    environment = _independent_environment_record()
    source = _independent_source_record(catnet_root)
    assets = _independent_asset_record(checkpoint_path)
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before fresh CPU strict gate")
    model, device = legacy.load_model(
        catnet_root=catnet_root,
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )
    parameters = sum(int(value.numel()) for value in model.parameters())
    state_keys = len(model.state_dict())
    if (
        str(device) != "cpu"
        or parameters != runner.EXPECTED_MODEL_PARAMETERS
        or state_keys != legacy.CHECKPOINT_STATE_KEYS
        or model.training
    ):
        raise ValueError("fresh CPU strict model gate changed")
    del model
    gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("fresh CPU strict gate initialized CUDA")
    return {
        "environment": environment,
        "source": source,
        "assets": assets,
        "strict_load": True,
        "parameter_count": parameters,
        "state_keys": state_keys,
        "forward_performed": False,
        "cuda_initialized_after": False,
        "adapter_sources": analyzer_source_contract(repo_root),
    }


def replay_model(
    bundle: RunBundle,
    *,
    catnet_root: Path = legacy.DEFAULT_CATNET_ROOT,
    checkpoint_path: Path = legacy.DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    if bundle.mode != "formal" or len(bundle.selected) != runner.FORMAL_IMAGES:
        raise ValueError("fresh replay requires the complete formal1025 run")
    cpu_gate = _fresh_cpu_strict_gate(
        repo_root=bundle.repo_root,
        catnet_root=catnet_root,
        checkpoint_path=checkpoint_path,
    )
    device, recorded_runtime = _configure_recorded_runtime(bundle)
    import torch

    model = None
    compared = 0
    maximum_peak = 0
    try:
        model, loaded_device = legacy.load_model(
            catnet_root=catnet_root,
            checkpoint_path=checkpoint_path,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("fresh CAT-Net model loaded on wrong device")
        result_by_id = {
            str(row["sample_id"]): row for row in bundle.latest_results
        }
        for input_row in bundle.selected:
            sample_id = str(input_row["sample_id"])
            recorded = result_by_id[sample_id]
            image_path = _safe_repo_file(
                bundle.repo_root,
                input_row.get("canonical_path"),
                expected=None,
                label=f"fresh input {sample_id}",
            )
            image, qtable, preprocess = legacy.preprocess_jpeg(image_path)
            if stable_json(preprocess) != stable_json(
                recorded.get("preprocess")
            ):
                raise ValueError(f"fresh preprocess mismatch: {sample_id}")
            raw, score_map, peak, _ = legacy.infer_one(
                model,
                loaded_device,
                image,
                qtable,
                preprocess,
            )
            raw = np.ascontiguousarray(raw, dtype=np.float32)
            score_map = np.ascontiguousarray(score_map, dtype=np.float32)
            raw_path = _exact_artifact_path(
                bundle,
                recorded.get("raw_logits_path"),
                directory="raw_logits_quarter",
                filename=f"{sample_id}.npy",
                label=f"fresh recorded logits {sample_id}",
            )
            map_path = _exact_artifact_path(
                bundle,
                recorded.get("score_map_path"),
                directory="score_maps_native",
                filename=f"{sample_id}.npy",
                label=f"fresh recorded map {sample_id}",
            )
            persisted_raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            persisted_map = np.load(map_path, mmap_mode="r", allow_pickle=False)
            if (
                not np.array_equal(raw, persisted_raw)
                or not np.array_equal(score_map, persisted_map)
                or _array_sha256(raw)
                != recorded.get("raw_logits_array_sha256")
                or _array_sha256(score_map)
                != recorded.get("score_map_array_sha256")
            ):
                raise ValueError(f"fresh dense output mismatch: {sample_id}")
            target = load_ground_truth(input_row, bundle.repo_root)
            if target is None:
                raise ValueError(f"fresh T2 target absent: {sample_id}")
            metrics = binary_pixel_metrics(
                score_map,
                np.asarray(target, dtype=bool),
                runner.MASK_THRESHOLD,
                include_ap=input_row["condition"] != "real",
            )
            if stable_json(
                _require_mapping(
                    recorded.get("localization"),
                    f"recorded localization {sample_id}",
                ).get("native")
            ) != stable_json(metrics):
                raise ValueError(f"fresh metric mismatch: {sample_id}")
            expected_mask_sha = recorded.get("mask_array_sha256")
            fresh_mask = np.where(
                score_map >= runner.MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if _array_sha256(fresh_mask) != expected_mask_sha:
                raise ValueError(f"fresh mask mismatch: {sample_id}")
            compared += 1
            maximum_peak = max(maximum_peak, int(peak))
            del raw, score_map, persisted_raw, persisted_map
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if compared != runner.FORMAL_IMAGES:
        raise ValueError("fresh CAT-Net replay coverage changed")
    _verify_bundle_unchanged(bundle)
    return {
        "schema_version": FRESH_REPLAY_SCHEMA,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "status": "pass",
        "fresh_model_instance": True,
        "cpu_strict_load_before_recorded_device": cpu_gate,
        "recorded_runtime": recorded_runtime,
        "images_replayed": compared,
        "raw_logits_bit_exact": True,
        "native_score_maps_bit_exact": True,
        "native_masks_bit_exact": True,
        "jpeg_qtable_and_dct_evidence_exact": True,
        "localization_metrics_exact": True,
        "t1_outputs_compared": 0,
        "map_statistic_promoted_to_t1": False,
        "maximum_replay_peak_cuda_memory_bytes": maximum_peak,
        "generated_at": utc_now(),
    }


def _write_json_verified(path: Path, value: Mapping[str, Any]) -> str:
    _reject_symlink_components(path, "analysis output")
    if path.exists() and path.is_symlink():
        raise ValueError("analysis output is a symlink")
    atomic_write_json(path, dict(value))
    loaded = _load_json(path, "written analysis output")
    if stable_json(loaded) != stable_json(dict(value)):
        raise ValueError("analysis output did not round-trip")
    return sha256_file(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("artifact", "smoke", "metrics", "fresh", "all"),
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR
    )
    parser.add_argument("--formal-run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument("--smoke-run-id-a", default=DEFAULT_SMOKE_RUN_ID_A)
    parser.add_argument("--smoke-run-id-b", default=DEFAULT_SMOKE_RUN_ID_B)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    if args.formal_run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError("formal run ID is frozen")
    if args.smoke_run_id_a != DEFAULT_SMOKE_RUN_ID_A:
        raise ValueError("smoke A run ID is frozen")
    if args.smoke_run_id_b != DEFAULT_SMOKE_RUN_ID_B:
        raise ValueError("smoke B run ID is frozen")
    dataset_manifest = _safe_standard_root(
        _lexical_absolute(args.dataset_manifest, base=repo_root),
        expected=repo_root / DEFAULT_DATASET_MANIFEST,
        label="dataset manifest",
    )
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    results_root = _safe_standard_root(
        _lexical_absolute(args.results_dir, base=repo_root),
        expected=repo_root / DEFAULT_RESULTS_DIR,
        label="results root",
    )
    artifacts_root = _safe_standard_root(
        _lexical_absolute(args.artifacts_dir, base=repo_root),
        expected=repo_root / DEFAULT_ARTIFACTS_DIR,
        label="artifacts root",
    )
    release = load_canonical_release(
        repo_root, dataset_manifest, verify_files=True
    )
    outputs: dict[str, Any] = {}
    formal: RunBundle | None = None
    smoke_a: RunBundle | None = None
    smoke_b: RunBundle | None = None
    if args.phase in ("artifact", "metrics", "fresh", "all"):
        formal = load_formal_run(
            repo_root=repo_root,
            run_id=DEFAULT_FORMAL_RUN_ID,
            release=release,
            results_root=results_root,
            artifacts_root=artifacts_root,
        )
    if args.phase in ("smoke", "all"):
        smoke_a = load_smoke_run(
            repo_root=repo_root,
            run_id=DEFAULT_SMOKE_RUN_ID_A,
            release=release,
            results_root=results_root,
            artifacts_root=artifacts_root,
        )
        smoke_b = load_smoke_run(
            repo_root=repo_root,
            run_id=DEFAULT_SMOKE_RUN_ID_B,
            release=release,
            results_root=results_root,
            artifacts_root=artifacts_root,
        )
    if args.phase in ("artifact", "all"):
        assert formal is not None
        report = audit_artifacts(formal)
        path = formal.run_dir / "artifact_audit.json"
        outputs["artifact_audit"] = {
            "path": repo_relative(path, repo_root),
            "sha256": _write_json_verified(path, report),
        }
    if args.phase in ("smoke", "all"):
        assert smoke_a is not None and smoke_b is not None
        report = compare_smoke_runs(smoke_a, smoke_b)
        if args.phase == "all":
            assert formal is not None
            report["formal_gate"] = validate_smoke_gate(
                formal, smoke_a, smoke_b
            )
        report_root = results_root / "_reports"
        report_root.mkdir(parents=True, exist_ok=True)
        path = report_root / "catnet_balanced_smoke_ab.json"
        outputs["smoke_comparison"] = {
            "path": repo_relative(path, repo_root),
            "sha256": _write_json_verified(path, report),
        }
    if args.phase in ("metrics", "all"):
        assert formal is not None
        report = recompute_metrics(
            formal,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        path = formal.run_dir / "metrics.json"
        outputs["metrics"] = {
            "path": repo_relative(path, repo_root),
            "sha256": _write_json_verified(path, report),
        }
    if args.phase in ("fresh", "all"):
        assert formal is not None
        report = replay_model(formal)
        path = formal.run_dir / "fresh_replay.json"
        outputs["fresh_replay"] = {
            "path": repo_relative(path, repo_root),
            "sha256": _write_json_verified(path, report),
        }
    value = {
        "status": "ok",
        "phase": args.phase,
        "formal_run_id": DEFAULT_FORMAL_RUN_ID,
        "smoke_run_ids": [
            DEFAULT_SMOKE_RUN_ID_A,
            DEFAULT_SMOKE_RUN_ID_B,
        ],
        "outputs": outputs,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    return value


def main(argv: list[str] | None = None) -> int:
    analyze(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
