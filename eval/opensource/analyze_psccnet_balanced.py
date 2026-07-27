#!/usr/bin/env python3
"""Fail-closed audit, metrics, smoke comparison, and replay for PSCC-Net.

The analyzer treats every run file and raw artifact as untrusted.  It rebuilds
the Balanced250 selection, independently validates source/assets/environment
and the strict CPU model load, reopens every canonical image, replays the
DetectionHead float32 softmax and stage-1 native interpolation, verifies all
four progressive maps plus native artifacts, recomputes the shared T1/T2
metrics, and compares the frozen A/B smoke runs byte-for-byte.

By default a formal audit also reloads the official three-component model and
performs a fresh 1,775-image replay.  Full-frame maps remain diagnostic only:
the 750 full-frame images must never acquire a T2 target, threshold mask, or
localization metric.
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
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from eval.opensource import run_psccnet as legacy
from eval.opensource import run_psccnet_balanced as runner
from eval.opensource.balanced250_localization_metrics import (
    summarize_balanced250_t2,
)
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
from eval.opensource.balanced_run_contract import (
    RunDatasetContract,
    ScoreSpec,
    build_result_identity,
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
    select_inputs,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "psccnet_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "psccnet_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "psccnet_balanced250_summary_v2"
T1_METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
T2_METRICS_SCHEMA_VERSION = "balanced250_t2_summary_v1"

DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = runner.DEFAULT_ARTIFACTS_DIR
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
DEFAULT_PSCCNET_ROOT = legacy.DEFAULT_PSCCNET_ROOT

FORMAL_IMAGES = 1_775
FORMAL_T2_IMAGES = 1_025
FORMAL_T2_NOT_APPLICABLE = 750
SMOKE_IMAGES = 35
SMOKE_T2_IMAGES = 20
SMOKE_T2_NOT_APPLICABLE = 15
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_726
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
THRESHOLD_OPERATOR = ">"
SHARED_T2_THRESHOLD_OPERATOR = ">="
STATIC_SOFTMAX_ABS_TOLERANCE = float(2 * np.finfo(np.float32).eps)
STATIC_NATIVE_REPLAY_RTOL = 1e-5
STATIC_NATIVE_REPLAY_ATOL = 2e-6

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
SMOKE_SELECTED_ROWS_SHA256 = (
    "21e556dd791960afde6cc900d9aba79da61a1a935c6da7b0e0928d3f6b26afa0"
)
SMOKE_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

EXPECTED_IMMUTABLE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
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
        "assets",
        "environment",
        "artifact_ignore",
        "checkpoint_audit",
        "model_audit",
        "license",
        "resource_expectation",
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
    }
)
EXPECTED_ARTIFACT_INVENTORY = runner.ARTIFACT_DIRECTORIES
EXPECTED_ADAPTER_SOURCE_PATHS = runner.ADAPTER_SOURCE_PATHS
PROGRESSIVE_SHAPES = runner.PROGRESSIVE_SHAPES
_T2_GT_KINDS = frozenset({"all_zero", "exact_diff"})


@dataclass(frozen=True)
class DenseArtifacts:
    sample_id: str
    progressive_paths: tuple[Path, Path, Path, Path]
    progressive_file_sha256: tuple[str, str, str, str]
    progressive_array_sha256: tuple[str, str, str, str]
    native_path: Path
    native_file_sha256: str
    native_array_sha256: str
    mask_path: Path | None
    mask_file_sha256: str | None
    mask_array_sha256: str | None
    t2_applicable: bool
    width: int
    height: int
    static_softmax_max_abs_diff: float
    static_native_max_abs_diff: float


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
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(value)


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


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    result = _require_mapping(value, label)
    _reject_nonfinite(result, label)
    return result


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"{label}:{line_number} lacks terminating newline")
                if not line.strip():
                    raise ValueError(f"{label}:{line_number} is blank")
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                row = _require_mapping(
                    value,
                    f"{label}:{line_number}",
                )
                _reject_nonfinite(row, f"{label}:{line_number}")
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
    return Path(os.path.abspath(candidate))


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains symlink component: {current}")


def _safe_standard_root(
    value: Path,
    *,
    repo_root: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    lexical = _lexical_absolute(value, base=repo_root)
    expected = _lexical_absolute(expected_relative, base=repo_root)
    _reject_symlink_components(lexical, label)
    if lexical != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    if lexical.exists() and (not lexical.is_dir() or lexical.is_symlink()):
        raise ValueError(f"{label} is unsafe")
    return lexical


def _resolve_run_dir(root: Path, run_id: Any, label: str) -> Path:
    safe_id = _valid_run_id(run_id)
    candidate = _lexical_absolute(root / safe_id)
    _reject_symlink_components(candidate, label)
    if candidate.parent != root:
        raise ValueError(f"{label} escapes its root")
    if not candidate.is_dir() or candidate.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {candidate}")
    return candidate


def _safe_repo_file(
    value: Any,
    *,
    repo_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    relative = _require_string(value, f"{label} path")
    if "\\" in relative:
        raise ValueError(f"{label} path must use POSIX separators")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} path is non-canonical")
    path = _lexical_absolute(Path(relative), base=repo_root)
    _reject_symlink_components(path, label)
    if path != expected_path.resolve():
        raise ValueError(f"{label} path changed")
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository") from error
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    return path


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
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
    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=CLASSIFICATION_THRESHOLD,
        threshold_operator=THRESHOLD_OPERATOR,
    )


def _selection_for_mode(
    release: CanonicalRelease,
    mode: str,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if mode == "formal":
        spec = SelectionSpec(capability=Capability.LOCAL_T1_T2)
        expected_images = FORMAL_IMAGES
        expected_rows_sha256 = FORMAL_SELECTED_ROWS_SHA256
        expected_ids_sha256 = FORMAL_SELECTED_IDS_SHA256
    elif mode == "smoke":
        spec = SelectionSpec(
            capability=Capability.LOCAL_T1_T2,
            per_condition_limit=SMOKE_PER_CONDITION,
        )
        expected_images = SMOKE_IMAGES
        expected_rows_sha256 = SMOKE_SELECTED_ROWS_SHA256
        expected_ids_sha256 = SMOKE_SELECTED_IDS_SHA256
    else:
        raise ValueError(f"unsupported PSCC-Net audit mode: {mode}")
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    expected_counts = (
        FORMAL_COUNTS
        if mode == "formal"
        else {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
    )
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or len(selected) != expected_images
        or dict(counts) != expected_counts
        or _rows_sha256(selected) != expected_rows_sha256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != expected_ids_sha256
    ):
        raise ValueError(f"PSCC-Net {mode} selection drifted")
    applicable = sum(row.get("gt_mask_kind") in _T2_GT_KINDS for row in selected)
    if applicable != (FORMAL_T2_IMAGES if mode == "formal" else SMOKE_T2_IMAGES):
        raise ValueError(f"PSCC-Net {mode} T2 applicability drifted")
    return spec, selected


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    recorded = _require_mapping(value, "adapter sources")
    if tuple(recorded) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("PSCC-Net adapter source inventory changed")
    checked: dict[str, dict[str, Any]] = {}
    for relative in EXPECTED_ADAPTER_SOURCE_PATHS:
        item = _require_mapping(
            recorded.get(relative),
            f"adapter source {relative}",
        )
        if set(item) != {"path", "bytes", "sha256"}:
            raise ValueError(f"PSCC-Net adapter source {relative} keys changed")
        path = _safe_repo_file(
            item.get("path"),
            repo_root=repo_root,
            expected_path=repo_root / relative,
            label=f"adapter source {relative}",
        )
        expected = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if item != expected:
            raise ValueError(f"PSCC-Net adapter source {relative} changed")
        checked[relative] = expected
    return checked


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
    device = runtime.get("device")
    expected_keys = (
        base_keys | {"cuda"}
        if isinstance(device, str) and device.startswith("cuda:")
        else base_keys
    )
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} key set changed")
    if device != "cpu":
        if (
            not isinstance(device, str)
            or not device.startswith("cuda:")
            or not device[5:].isdigit()
            or str(int(device[5:])) != device[5:]
        ):
            raise ValueError(f"{label} device is invalid")
        cuda = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        if set(cuda) != {
            "logical_device_index",
            "device_name",
            "total_memory_bytes",
            "compute_capability",
        }:
            raise ValueError(f"{label} CUDA record changed")
        if (
            cuda.get("logical_device_index") != int(device[5:])
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(cuda.get("compute_capability"), list)
            or len(cuda["compute_capability"]) != 2
            or any(
                isinstance(child, bool) or not isinstance(child, int) or child < 0
                for child in cuda["compute_capability"]
            )
        ):
            raise ValueError(f"{label} CUDA values changed")
    expected_values = {
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
        "torch_cuda_version": "12.8",
    }
    for key, expected in expected_values.items():
        if runtime.get(key) != expected:
            raise ValueError(f"{label} {key} changed")
    expected_hash = _fingerprint(
        {key: child for key, child in runtime.items() if key != "contract_sha256"}
    )
    if runtime.get("contract_sha256") != expected_hash:
        raise ValueError(f"{label} contract SHA-256 changed")
    return runtime


def _independent_source_record(psccnet_root: Path) -> dict[str, Any]:
    _reject_symlink_components(psccnet_root, "PSCC-Net source root")
    root = psccnet_root.resolve()
    if root.name != "PSCC-Net" or not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe PSCC-Net source: {root}")
    commit = _git_value(root, "rev-parse", "HEAD")
    origin = _git_value(root, "remote", "get-url", "origin")
    status = _git_value(root, "status", "--short", "--untracked-files=all")
    if (
        commit != legacy.MODEL_SOURCE_COMMIT
        or origin != runner.MODEL_GIT_ORIGIN
        or status is None
        or status
    ):
        raise ValueError("PSCC-Net official source identity/cleanliness changed")
    bindings: dict[str, dict[str, Any]] = {}
    for relative, (
        expected_bytes,
        expected_sha256,
    ) in runner.SOURCE_BOUND_FILES.items():
        path = root / relative
        _reject_symlink_components(
            path,
            f"PSCC-Net source file {relative}",
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha256
            or _git_value(root, "ls-files", "--error-unmatch", relative) != relative
        ):
            raise ValueError(f"PSCC-Net source file changed: {relative}")
        bindings[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "git_tracked": True,
        }
    record = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "origin": origin,
        "tracked_and_untracked_clean": True,
        "source_bound_files": bindings,
    }
    return {**record, "contract_sha256": _fingerprint(record)}


def _independent_assets_record(psccnet_root: Path) -> dict[str, Any]:
    root = psccnet_root.resolve()
    contracts = {
        "initialization_weight": legacy.INITIALIZATION_WEIGHT,
        **legacy.CHECKPOINTS,
    }
    assets: dict[str, dict[str, Any]] = {}
    for role, contract in contracts.items():
        relative = str(contract["path"])
        path = root / relative
        _reject_symlink_components(path, f"PSCC-Net asset {role}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(contract["bytes"])
            or sha256_file(path) != str(contract["sha256"])
            or _git_value(root, "ls-files", "--error-unmatch", relative) != relative
        ):
            raise ValueError(f"PSCC-Net official asset changed: {role}")
        assets[role] = {
            "path": str(path),
            "repository_path": relative,
            "bytes": int(contract["bytes"]),
            "sha256": str(contract["sha256"]),
            "git_tracked": True,
            "provider": "official_author_git_repository",
        }
    record = {
        "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "assets": assets,
    }
    return {**record, "contract_sha256": _fingerprint(record)}


def _independent_environment_record() -> dict[str, Any]:
    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    if (
        executable != runner.EXPECTED_PYTHON_EXECUTABLE
        or prefix != runner.EXPECTED_VENV_ROOT
        or platform.python_version() != "3.12.3"
    ):
        raise ValueError("PSCC-Net analysis environment changed")
    pyvenv = prefix / "pyvenv.cfg"
    if (
        not pyvenv.is_file()
        or pyvenv.is_symlink()
        or pyvenv.stat().st_size != runner.EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != runner.EXPECTED_PYVENV_SHA256
    ):
        raise ValueError("PSCC-Net analysis pyvenv.cfg changed")
    versions = {name: _package_version(name) for name in runner.EXPECTED_PACKAGES}
    if versions != runner.EXPECTED_PACKAGES:
        raise ValueError("PSCC-Net analysis package environment changed")
    record = {
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
    }
    return {**record, "contract_sha256": _fingerprint(record)}


def _independent_artifact_ignore(repo_root: Path) -> dict[str, Any]:
    probe = "outputs/opensource/psccnet/_contract_probe/artifact.npy"
    try:
        evidence = subprocess.check_output(
            ["git", "-C", str(repo_root), "check-ignore", "-v", "--", probe],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("PSCC-Net raw artifacts are not gitignored") from error
    if not evidence or not evidence.endswith(f"\t{probe}"):
        raise ValueError("PSCC-Net artifact ignore evidence changed")
    record = {
        "probe": probe,
        "git_check_ignore_evidence": evidence,
        "ignored": True,
    }
    return {**record, "contract_sha256": _fingerprint(record)}


def _tensor_schema(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in state.items()
    ]


def _independent_state_audit(
    path: Path,
    expected: Mapping[str, Any],
    *,
    scan_unsafe_globals: bool,
) -> tuple[dict[str, Any], tuple[str, ...], Mapping[str, Any]]:
    import torch

    unsafe: tuple[str, ...] | None
    if scan_unsafe_globals:
        unsafe = tuple(
            sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        )
        if unsafe:
            raise ValueError("PSCC-Net checkpoint unsafe globals changed")
    else:
        unsafe = None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        type(state).__name__ != "OrderedDict"
        or any(not isinstance(key, str) for key in state)
        or any(not isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError("PSCC-Net checkpoint container changed")
    dtype_counts = Counter(str(value.dtype) for value in state.values())
    elements = sum(int(value.numel()) for value in state.values())
    keys_hash = hashlib.sha256("\n".join(state).encode("utf-8")).hexdigest()
    schema_hash = _fingerprint(_tensor_schema(state))
    if (
        len(state) != int(expected["state_keys"])
        or elements != int(expected["state_elements"])
        or dict(sorted(dtype_counts.items())) != expected["dtype_counts"]
        or keys_hash != expected["ordered_keys_sha256"]
        or schema_hash != expected["tensor_schema_sha256"]
        or any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in state.values()
        )
    ):
        raise ValueError("PSCC-Net checkpoint tensor audit changed")
    value = {
        "outer_type": "collections.OrderedDict",
        "state_dict_tensors": len(state),
        "state_dict_elements": elements,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "ordered_keys_sha256": keys_hash,
        "tensor_schema_sha256": schema_hash,
        "all_floating_tensors_finite": True,
        "weights_only": True,
        "map_location": "cpu",
        "unsafe_globals": (
            list(unsafe) if unsafe is not None else "legacy_pickle_not_scanable"
        ),
    }
    return (
        {**value, "contract_sha256": _fingerprint(value)},
        tuple(state),
        state,
    )


def independent_structural_golden(
    *,
    psccnet_root: Path,
    recorded_checkpoint_audit: Mapping[str, Any],
    recorded_model_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently construct and strictly load all three modules on CPU."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "PSCC-Net structural golden must precede CUDA initialization"
        )
    root = psccnet_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not hasattr(np, "int"):
        setattr(np, "int", int)
    previous = Path.cwd()
    os.chdir(root)
    try:
        from models.NLCDetection import NLCDetection
        from models.detection_head import DetectionHead
        from models.seg_hrnet import get_seg_model
        from models.seg_hrnet_config import get_hrnet_cfg

        config = get_hrnet_cfg()
        config.PRETRAINED = ""
        modules = {
            "feature_extractor": get_seg_model(config),
            "localization_head": NLCDetection(
                {"crop_size": list(legacy.MODEL_CROP_SIZE)}
            ),
            "classification_head": DetectionHead(
                {"crop_size": list(legacy.MODEL_CROP_SIZE)}
            ),
        }
    finally:
        os.chdir(previous)

    checkpoint_components: dict[str, Any] = {}
    model_components: dict[str, Any] = {}
    try:
        for role, module in modules.items():
            path = root / str(legacy.CHECKPOINTS[role]["path"])
            audit, checkpoint_keys, state = _independent_state_audit(
                path,
                runner.CHECKPOINT_AUDIT[role],
                scan_unsafe_globals=True,
            )

            class CPUWrapper(torch.nn.Module):
                def __init__(self, child: Any) -> None:
                    super().__init__()
                    self.module = child

                def forward(self, *args: Any, **kwargs: Any) -> Any:
                    return self.module(*args, **kwargs)

            wrapped = CPUWrapper(module)
            incompatible = wrapped.load_state_dict(state, strict=True)
            wrapped.eval()
            parameters = sum(int(value.numel()) for value in module.parameters())
            trainable = sum(
                int(value.numel())
                for value in module.parameters()
                if value.requires_grad
            )
            buffers = sum(int(value.numel()) for value in module.buffers())
            module_count = sum(1 for _ in module.modules())
            if (
                tuple(wrapped.state_dict()) != checkpoint_keys
                or incompatible.missing_keys
                or incompatible.unexpected_keys
                or parameters != int(legacy.CHECKPOINTS[role]["parameters"])
                or trainable != parameters
                or buffers != int(legacy.CHECKPOINTS[role]["buffers"])
                or module_count != runner.EXPECTED_COMPONENT_MODULES[role]
                or wrapped.training
            ):
                raise ValueError(f"PSCC-Net independent {role} strict load changed")
            checkpoint_components[role] = audit
            model_components[role] = {
                "construction_device": "cpu",
                "strict_state_dict_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "state_key_order_matches_checkpoint": True,
                "eval_mode": True,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "buffer_elements": buffers,
                "module_count": module_count,
            }
            del state, wrapped

        init_path = root / str(legacy.INITIALIZATION_WEIGHT["path"])
        init_audit, _, init_state = _independent_state_audit(
            init_path,
            runner.INITIALIZATION_AUDIT,
            scan_unsafe_globals=False,
        )
        del init_state
        model_value = {
            "construction_device": "cpu",
            "constructor_initialization_weight_loaded": False,
            "constructor_initialization_weight_suppression_reason": (
                "legacy_cuda_pickle_is_fully_overwritten_by_complete_"
                "strict_task_checkpoint"
            ),
            "complete_task_checkpoint_replaces_constructor_state": True,
            "components": model_components,
            "parameter_count": sum(
                item["parameters"] for item in model_components.values()
            ),
            "trainable_parameter_count": sum(
                item["trainable_parameters"] for item in model_components.values()
            ),
            "buffer_elements": sum(
                item["buffer_elements"] for item in model_components.values()
            ),
            "module_count": sum(
                item["module_count"] for item in model_components.values()
            ),
            "forward_performed": False,
        }
        checkpoint_value = {
            "initialization_weight": init_audit,
            "task_components": checkpoint_components,
            "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        }
        computed_checkpoint = {
            **checkpoint_value,
            "contract_sha256": _fingerprint(checkpoint_value),
        }
        computed_model = {
            **model_value,
            "contract_sha256": _fingerprint(model_value),
        }
        if computed_checkpoint != dict(
            recorded_checkpoint_audit
        ) or computed_model != dict(recorded_model_audit):
            raise ValueError(
                "PSCC-Net recorded structural audit differs from "
                "independent strict load"
            )
        if torch.cuda.is_initialized():
            raise RuntimeError("PSCC-Net structural golden initialized CUDA")
        return {
            "status": "independent_cpu_strict_load_passed",
            "checkpoint_audit_exact": True,
            "model_audit_exact": True,
            "components_strictly_loaded": 3,
            "parameter_count": runner.EXPECTED_MODEL_PARAMETERS,
            "buffer_elements": runner.EXPECTED_MODEL_BUFFERS,
            "module_count": runner.EXPECTED_MODEL_MODULES,
            "forward_performed": False,
            "cuda_initialized": False,
        }
    finally:
        modules.clear()
        gc.collect()


def _validate_provenance(
    *,
    immutable: Mapping[str, Any],
    repo_root: Path,
    psccnet_root: Path,
) -> dict[str, Any]:
    source = _independent_source_record(psccnet_root)
    assets = _independent_assets_record(psccnet_root)
    environment = _independent_environment_record()
    artifact_ignore = _independent_artifact_ignore(repo_root)
    if immutable.get("source") != source:
        raise ValueError("PSCC-Net recorded source provenance changed")
    if immutable.get("assets") != assets:
        raise ValueError("PSCC-Net recorded asset provenance changed")
    if immutable.get("environment") != environment:
        raise ValueError("PSCC-Net recorded environment changed")
    if immutable.get("artifact_ignore") != artifact_ignore:
        raise ValueError("PSCC-Net recorded git-ignore evidence changed")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )

    license_record = _require_mapping(
        immutable.get("license"),
        "PSCC-Net license",
    )
    if license_record != runner.LICENSE_RECORD:
        raise ValueError("PSCC-Net license record changed")
    project_license = _require_mapping(
        license_record.get("project_license"),
        "PSCC-Net project license",
    )
    if (
        project_license.get("spdx") != "MIT"
        or project_license.get("commercial_use_permission") is not True
        or project_license.get("redistribution_permission") is not True
        or project_license.get("sha256") != legacy.SOURCE_FILES["LICENSE"]
    ):
        raise ValueError("PSCC-Net MIT permission record changed")

    checkpoint_audit = _require_mapping(
        immutable.get("checkpoint_audit"),
        "checkpoint audit",
    )
    model_audit = _require_mapping(
        immutable.get("model_audit"),
        "model audit",
    )
    preflight = _require_mapping(
        immutable.get("cpu_preflight"),
        "CPU preflight envelope",
    )
    if (
        set(preflight)
        != {
            "performed_before_accelerator_configuration",
            "report",
        }
        or preflight.get("performed_before_accelerator_configuration") is not True
    ):
        raise ValueError("PSCC-Net CPU preflight envelope changed")
    report = _require_mapping(
        preflight.get("report"),
        "CPU preflight report",
    )
    expected_report_keys = {
        "schema_version",
        "cuda_initialized_before",
        "cuda_initialized_after",
        "environment",
        "source",
        "assets",
        "adapter_sources",
        "artifact_ignore",
        "checkpoint_audit",
        "model_audit",
        "balanced250_forward_performed",
        "balanced250_score_computed",
        "contract_sha256",
    }
    if set(report) != expected_report_keys:
        raise ValueError("PSCC-Net CPU preflight key set changed")
    report_payload = {
        key: value for key, value in report.items() if key != "contract_sha256"
    }
    if (
        report.get("schema_version") != runner.CPU_PREFLIGHT_SCHEMA
        or report.get("cuda_initialized_before") is not False
        or report.get("cuda_initialized_after") is not False
        or report.get("balanced250_forward_performed") is not False
        or report.get("balanced250_score_computed") is not False
        or report.get("environment") != environment
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("adapter_sources") != immutable.get("adapter_sources")
        or report.get("artifact_ignore") != artifact_ignore
        or report.get("checkpoint_audit") != checkpoint_audit
        or report.get("model_audit") != model_audit
        or report.get("contract_sha256") != _fingerprint(report_payload)
    ):
        raise ValueError("PSCC-Net CPU preflight evidence changed")
    return {
        "source": source,
        "assets": assets,
        "environment": environment,
        "artifact_ignore": artifact_ignore,
        "checkpoint_audit": checkpoint_audit,
        "model_audit": model_audit,
        "license": license_record,
    }


def _rebuild_contract(
    *,
    release: CanonicalRelease,
    mode: str,
    recorded: Any,
) -> tuple[tuple[dict[str, Any], ...], RunDatasetContract]:
    spec, selected = _selection_for_mode(release, mode)
    contract = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=_score_spec(),
    )
    if contract.as_dict() != recorded:
        raise ValueError("PSCC-Net run dataset contract changed")
    return tuple(selected), contract


def _expected_model_record() -> dict[str, Any]:
    return {
        "name": runner.MODEL_NAME,
        "slug": runner.MODEL_SLUG,
        "architecture": runner.MODEL_ARCHITECTURE,
        "repository": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "checkpoint_id": runner.CHECKPOINT_ID,
        "checkpoint_bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "checkpoint_components": {
            role: dict(contract) for role, contract in legacy.CHECKPOINTS.items()
        },
        "initialization_weight": dict(legacy.INITIALIZATION_WEIGHT),
        "training_manipulations": [
            "authentic",
            "splicing",
            "copy_move",
            "RFR_Net_object_removal_inpainting",
        ],
        "variant": "official_synthetic_pretrained_not_retrained",
    }


def _expected_preprocess_record() -> dict[str, Any]:
    return {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "imageio.v2.imread",
        "channel_order": "RGB",
        "rgba": "official_float32_white_background_composite",
        "input_resize": None,
        "input_crop": None,
        "input_reencode": False,
        "scale": "uint8_divide_255",
        "normalization_mean_std": None,
        "tensor_layout": "CHW",
        "tensor_dtype": "float32",
        "batch_size": 1,
    }


def _expected_inference_record() -> dict[str, Any]:
    return {
        "feature_extractor": "HRNet_W18_small_v2",
        "localization_head": "NLCDetection",
        "classification_head": "DetectionHead",
        "progressive_output_shapes": [list(shape) for shape in PROGRESSIVE_SHAPES],
        "primary_localization_output": "progressive_mask1",
        "localization_outputs_are_already_sigmoid_probabilities": True,
        "second_sigmoid_applied": False,
        "native_restore": "torch_bilinear_probability_align_corners_true",
        "classification_output": ("float32_softmax_two_class_logits_positive_index_1"),
        "test_time_augmentation": False,
        "ensemble": False,
        "autocast": False,
    }


def _validate_immutable_static(
    immutable: Mapping[str, Any],
    *,
    run_id: str,
    mode: str,
    selected: Sequence[Mapping[str, Any]],
    contract: RunDatasetContract,
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    if set(immutable) != EXPECTED_IMMUTABLE_KEYS:
        raise ValueError("PSCC-Net immutable key set changed")
    expected_rows_hash = (
        FORMAL_SELECTED_ROWS_SHA256 if mode == "formal" else SMOKE_SELECTED_ROWS_SHA256
    )
    expected_ids_hash = (
        FORMAL_SELECTED_IDS_SHA256 if mode == "formal" else SMOKE_SELECTED_IDS_SHA256
    )
    expected_values = {
        "schema_version": runner.RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "model": _expected_model_record(),
        "preprocess": _expected_preprocess_record(),
        "inference": _expected_inference_record(),
        "score_spec": _score_spec().as_dict(),
        "t2_spec": runner.T2_SPEC,
        "task_scope": runner.TASK_SCOPE,
        "dataset_contract": contract.as_dict(),
        "selected_rows_sha256": expected_rows_hash,
        "selected_ids_sha256": expected_ids_hash,
        "license": runner.LICENSE_RECORD,
        "resource_expectation": runner.RESOURCE_EXPECTATION,
        "artifact_contract": runner.ARTIFACT_CONTRACT,
    }
    for key, expected in expected_values.items():
        if immutable.get(key) != expected:
            raise ValueError(f"PSCC-Net immutable {key} changed")
    if _rows_sha256(selected) != expected_rows_hash:
        raise ValueError("PSCC-Net selected row snapshot changed")
    runtime = _validate_runtime(
        immutable.get("runtime"),
        label="recorded PSCC-Net runtime",
    )
    outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable outputs",
    )
    expected_outputs = {
        "results_path": repo_relative(run_dir / "results.jsonl", repo_root),
        "expected_inputs_path": repo_relative(
            run_dir / "expected_inputs.jsonl",
            repo_root,
        ),
        "summary_path": repo_relative(run_dir / "summary.json", repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        **{
            f"{name}_dir": repo_relative(
                artifact_root / name,
                repo_root,
            )
            for name in runner.ARTIFACT_DIRECTORIES
        },
    }
    if outputs != expected_outputs:
        raise ValueError("PSCC-Net immutable output paths changed")
    return runtime


def _validate_manifest_envelope(
    *,
    manifest: Mapping[str, Any],
    immutable: Mapping[str, Any],
    run_id: str,
    fingerprint: str,
    release: CanonicalRelease,
    selected: Sequence[Mapping[str, Any]],
    contract: RunDatasetContract,
    expected_path: Path,
    results_path: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise ValueError("PSCC-Net manifest key set changed")
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
        or not isinstance(manifest.get("started_at"), str)
        or not manifest["started_at"]
        or not isinstance(manifest.get("completed_at"), str)
        or not manifest["completed_at"]
        or manifest.get("fingerprint") != fingerprint
        or _fingerprint(immutable) != fingerprint
    ):
        raise ValueError("PSCC-Net complete manifest identity changed")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    expected_dataset = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(
            release.manifest_path,
            release.repo_root,
        ),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(
            expected_path,
            release.repo_root,
        ),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t2_applicable_images": sum(
            row.get("gt_mask_kind") in _T2_GT_KINDS for row in selected
        ),
    }
    if dataset != expected_dataset:
        raise ValueError("PSCC-Net manifest dataset envelope changed")

    disk = _require_mapping(
        manifest.get("disk_preflight"),
        "manifest disk preflight",
    )
    if set(disk) != {
        "free_bytes_before_inference",
        "conservative_pending_bytes_plus_reserve",
        "fixed_reserve_bytes",
    } or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in disk.values()
    ):
        raise ValueError("PSCC-Net disk preflight record changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest execution",
    )
    if set(execution) != EXPECTED_EXECUTION_KEYS or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in execution.values()
    ):
        raise ValueError("PSCC-Net execution record changed")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    expected_output_keys = set(immutable["outputs"]) | {
        "results_sha256",
        "summary_sha256",
        "artifact_inventory",
    }
    if set(outputs) != expected_output_keys:
        raise ValueError("PSCC-Net manifest output key set changed")
    for key, expected in immutable["outputs"].items():
        if outputs.get(key) != expected:
            raise ValueError(f"PSCC-Net manifest output {key} changed")
    if outputs.get("results_sha256") != sha256_file(results_path) or outputs.get(
        "summary_sha256"
    ) != sha256_file(summary_path):
        raise ValueError("PSCC-Net finalized result/summary hash changed")
    inventory = _require_mapping(
        outputs.get("artifact_inventory"),
        "manifest artifact inventory",
    )
    if set(inventory) != set(EXPECTED_ARTIFACT_INVENTORY) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in inventory.values()
    ):
        raise ValueError("PSCC-Net recorded artifact inventory changed")

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
        or summary.get("summary_kind") != "runtime_coverage_and_artifact_inventory_only"
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner") != "analyze_psccnet_balanced.py"
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != fingerprint
        or summary.get("status") != "complete"
        or summary.get("mode") != immutable["mode"]
        or summary.get("model") != runner.MODEL_NAME
        or summary.get("model_slug") != runner.MODEL_SLUG
        or summary.get("score_spec") != _score_spec().as_dict()
        or summary.get("t2_spec") != runner.T2_SPEC
        or summary.get("dataset_contract") != contract.as_dict()
        or summary.get("artifact_inventory") != inventory
        or not isinstance(summary.get("generated_at"), str)
        or not summary["generated_at"]
    ):
        raise ValueError("PSCC-Net runtime summary changed")


EXPECTED_OK_ONLY_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "preprocess",
        "classification_logits",
        "classification_logits_dtype",
        "classification_logits_sha256",
        "classification_probabilities",
        "classification_probabilities_dtype",
        "classification_probabilities_sha256",
        "ai_score",
        "score_semantics",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "artifact_paths",
        "progressive_maps",
        "primary_model_score_map_path",
        "primary_model_score_map_sha256",
        "primary_model_score_map_bytes",
        "primary_model_score_map_shape",
        "primary_model_score_map_dtype",
        "primary_model_score_map_semantics",
        "score_map_path",
        "score_map_sha256",
        "score_map_bytes",
        "score_map_shape",
        "score_map_dtype",
        "score_map_semantics",
        "score_map_array_sha256",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
        "mask_threshold",
        "mask_threshold_operator",
        "localization",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
EXPECTED_ERROR_ONLY_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "error_type",
        "error",
        "traceback",
    }
)


def _expected_result_identity(
    row: Mapping[str, Any],
    *,
    run_id: str,
    fingerprint: str,
    valid: bool,
) -> dict[str, Any]:
    condition = str(row["condition"])
    if condition == "real":
        applicable = True
        semantics = "all_zero_real_false_positive_area"
    elif condition in {"local_mouse", "local_cat", "local_trash_can"}:
        applicable = True
        semantics = "exact_diff_local_insertion"
    elif condition in {
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    }:
        applicable = False
        semantics = "not_applicable_fullframe"
    else:
        raise ValueError(f"unsupported PSCC-Net condition: {condition}")
    task_scope = {
        "primary_task": "T1_image_detection_and_T2_localization",
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "t2_target_semantics": semantics,
        "fullframe_t2_not_applicable": not applicable,
        "native_dense_output_present": True,
        "image_score_source": "independent_detection_head",
    }
    return {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        ),
        "valid_for_metrics": valid,
        "model": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "checkpoint_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "task_scope": task_scope,
        "t2_applicable": applicable,
        "t2_target_semantics": semantics,
    }


def _independent_preprocess_tensor(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as opened:
        decoded = np.asarray(opened)
    if decoded.ndim == 2:
        decoded = np.repeat(decoded[..., None], 3, axis=2)
        alpha_policy = "grayscale_repeated_to_rgb"
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        rgba = decoded.astype(np.float32)
        alpha = rgba[..., 3:4] / np.float32(255.0)
        decoded = (
            rgba[..., :3] * alpha + np.float32(255.0) * (np.float32(1.0) - alpha)
        ).astype(np.uint8)
        alpha_policy = "official_white_background_rgba_composite"
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        alpha_policy = "not_applicable"
    else:
        raise ValueError(f"unexpected PSCC-Net decoded shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError("PSCC-Net decoded dtype changed")
    height, width = decoded.shape[:2]
    tensor = np.ascontiguousarray(
        decoded.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0),
        dtype=np.float32,
    )
    audit = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "native_size": [width, height],
        "input_resize": "none",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255",
        "alpha_policy": alpha_policy,
        "tensor_shape": list(tensor.shape),
        "tensor_sha256": _array_sha256(tensor),
    }
    return tensor, audit


def _float32_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("PSCC-Net logits are invalid")
    shifted = values - np.max(values)
    exponentials = np.exp(shifted, dtype=np.float32)
    return np.asarray(
        exponentials / np.sum(exponentials, dtype=np.float32),
        dtype=np.float32,
    )


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> float:
    logits_raw = row.get("classification_logits")
    probabilities_raw = row.get("classification_probabilities")
    if (
        not isinstance(logits_raw, list)
        or len(logits_raw) != 2
        or not isinstance(probabilities_raw, list)
        or len(probabilities_raw) != 2
    ):
        raise ValueError(f"{sample_id} classification vectors changed")
    logits = np.asarray(logits_raw, dtype=np.float32)
    recorded = np.asarray(probabilities_raw, dtype=np.float32)
    expected = _float32_softmax(logits)
    difference = float(np.max(np.abs(recorded - expected)))
    if (
        not np.isfinite(recorded).all()
        or float(recorded.min()) < 0.0
        or float(recorded.max()) > 1.0
        or difference > STATIC_SOFTMAX_ABS_TOLERANCE
        or row.get("classification_logits_dtype") != "float32"
        or row.get("classification_probabilities_dtype") != "float32"
        or row.get("classification_logits_sha256") != _array_sha256(logits)
        or row.get("classification_probabilities_sha256") != _array_sha256(recorded)
    ):
        raise ValueError(f"{sample_id} persisted DetectionHead softmax replay changed")
    score = float(recorded[1])
    if (
        row.get("ai_score") != score
        or row.get("score_semantics") != "softmax_probability_class_1_forged"
        or row.get("classification_decision")
        != ("forged" if score > CLASSIFICATION_THRESHOLD else "authentic")
        or row.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator") != THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} persisted T1 score semantics changed")
    return difference


def _validate_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    run_id: str,
    fingerprint: str,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("PSCC-Net result status changed")
    expected = _expected_result_identity(
        input_row,
        run_id=run_id,
        fingerprint=fingerprint,
        valid=status == "ok",
    )
    expected_keys = set(expected) | (
        set(EXPECTED_OK_ONLY_KEYS) if status == "ok" else set(EXPECTED_ERROR_ONLY_KEYS)
    )
    if set(attempt) != expected_keys:
        raise ValueError("PSCC-Net result key set changed")
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"PSCC-Net result identity {key} changed")
    if not isinstance(attempt.get("completed_at"), str) or not attempt["completed_at"]:
        raise ValueError("PSCC-Net result completed_at changed")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
        ):
            raise ValueError("PSCC-Net error attempt changed")
        return
    _validate_score_payload(
        attempt,
        sample_id=str(input_row["sample_id"]),
    )
    if (
        attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != THRESHOLD_OPERATOR
        or _require_finite(
            attempt.get("latency_ms"),
            "PSCC-Net latency",
        )
        < 0.0
        or isinstance(attempt.get("peak_cuda_memory_bytes"), bool)
        or not isinstance(attempt.get("peak_cuda_memory_bytes"), int)
        or attempt["peak_cuda_memory_bytes"] < 0
    ):
        raise ValueError("PSCC-Net runtime result metadata changed")


def _resolve_artifact(
    value: Any,
    *,
    repo_root: Path,
    artifact_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    path = _safe_repo_file(
        value,
        repo_root=repo_root,
        expected_path=expected_path,
        label=label,
    )
    try:
        path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    return path


def _load_probability_map(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_file_sha256: Any,
    expected_file_bytes: Any,
    expected_array_sha256: Any,
    label: str,
) -> np.ndarray:
    file_hash = _require_sha256(
        expected_file_sha256,
        f"{label} file SHA-256",
    )
    array_hash = _require_sha256(
        expected_array_sha256,
        f"{label} array SHA-256",
    )
    expected_bytes = (
        int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        + runner.NPY_HEADER_BYTES
    )
    if (
        sha256_file(path) != file_hash
        or expected_file_bytes != expected_bytes
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError(f"{label} persisted file changed")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != expected_shape
        or array.dtype != np.float32
        or not np.isfinite(array).all()
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
        or _array_sha256(array) != array_hash
    ):
        raise ValueError(f"{label} persisted array changed")
    return np.asarray(array)


def _bilinear_align_corners_true(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(score_map, dtype=np.float32)
    if source.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("PSCC-Net native interpolation input changed")
    source_height, source_width = source.shape
    if source.shape == (height, width):
        return np.ascontiguousarray(source)
    y = (
        np.zeros(height, dtype=np.float32)
        if height == 1
        else np.arange(height, dtype=np.float32)
        * (np.float32(source_height - 1) / np.float32(height - 1))
    )
    x = (
        np.zeros(width, dtype=np.float32)
        if width == 1
        else np.arange(width, dtype=np.float32)
        * (np.float32(source_width - 1) / np.float32(width - 1))
    )
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).astype(np.float32).reshape(-1, 1)
    wx = (x - x0).astype(np.float32).reshape(1, -1)
    top = (
        source[y0[:, None], x0[None, :]] * (np.float32(1.0) - wx)
        + source[y0[:, None], x1[None, :]] * wx
    )
    bottom = (
        source[y1[:, None], x0[None, :]] * (np.float32(1.0) - wx)
        + source[y1[:, None], x1[None, :]] * wx
    )
    restored = top * (np.float32(1.0) - wy) + bottom * wy
    return np.ascontiguousarray(restored, dtype=np.float32)


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _independent_pixel_metrics(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    include_ap: bool,
) -> dict[str, Any]:
    scores = np.asarray(score_map, dtype=np.float64)
    truth = np.asarray(target, dtype=bool)
    if (
        scores.shape != truth.shape
        or scores.ndim != 2
        or not scores.size
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
    ):
        raise ValueError("PSCC-Net independent pixel metric input changed")
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
        "threshold_operator": THRESHOLD_OPERATOR,
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
    recorded: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(recorded, Mapping) or set(recorded) != set(expected):
            raise ValueError(f"{label} mapping changed")
        for key, child in expected.items():
            _assert_nested_close(
                recorded[key],
                child,
                label=f"{label}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(recorded, list) or len(recorded) != len(expected):
            raise ValueError(f"{label} list changed")
        for index, child in enumerate(expected):
            _assert_nested_close(
                recorded[index],
                child,
                label=f"{label}[{index}]",
            )
        return
    if isinstance(expected, float):
        if (
            not isinstance(recorded, (int, float))
            or isinstance(recorded, bool)
            or not math.isclose(
                float(recorded),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{label} changed")
        return
    if recorded != expected:
        raise ValueError(f"{label} changed")


def _resize_target_model_256(target: np.ndarray) -> np.ndarray:
    image = Image.fromarray(
        np.where(np.asarray(target, dtype=bool), 255, 0).astype(np.uint8),
        mode="L",
    )
    resized = image.resize(
        legacy.MODEL_CROP_SIZE,
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def _validate_artifact_row(
    *,
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> DenseArtifacts:
    sample_id = str(input_row["sample_id"])
    width = int(input_row["width"])
    height = int(input_row["height"])
    input_path = _safe_repo_file(
        input_row.get("canonical_path"),
        repo_root=repo_root,
        expected_path=repo_root / str(input_row["canonical_path"]),
        label=f"{sample_id} canonical input",
    )
    tensor, preprocess = _independent_preprocess_tensor(input_path)
    if tensor.shape != (3, height, width) or row.get("preprocess") != preprocess:
        raise ValueError(f"{sample_id} independent preprocessing changed")
    softmax_difference = _validate_score_payload(row, sample_id=sample_id)

    artifact_paths = _require_mapping(
        row.get("artifact_paths"),
        f"{sample_id} artifact paths",
    )
    expected_artifact_keys = {
        "progressive_mask1",
        "progressive_mask2",
        "progressive_mask3",
        "progressive_mask4",
        "native_probability",
        "native_mask",
    }
    if set(artifact_paths) != expected_artifact_keys:
        raise ValueError(f"{sample_id} artifact path keys changed")
    progressive_raw = row.get("progressive_maps")
    if not isinstance(progressive_raw, list) or len(progressive_raw) != 4:
        raise ValueError(f"{sample_id} progressive artifact list changed")

    progressive_paths: list[Path] = []
    progressive_file_hashes: list[str] = []
    progressive_array_hashes: list[str] = []
    progressive_arrays: list[np.ndarray] = []
    for stage, (raw, shape) in enumerate(
        zip(progressive_raw, PROGRESSIVE_SHAPES, strict=True),
        start=1,
    ):
        item = _require_mapping(
            raw,
            f"{sample_id} progressive stage {stage}",
        )
        if set(item) != {
            "stage",
            "path",
            "sha256",
            "bytes",
            "shape",
            "dtype",
            "semantics",
            "primary",
            "array_sha256",
        }:
            raise ValueError(f"{sample_id} progressive stage {stage} keys changed")
        key = f"progressive_mask{stage}"
        semantics = (
            "official_primary_NLC_sigmoid_probability"
            if stage == 1
            else "official_auxiliary_NLC_sigmoid_probability"
        )
        if (
            item.get("stage") != stage
            or item.get("shape") != list(shape)
            or item.get("dtype") != "float32"
            or item.get("semantics") != semantics
            or item.get("primary") is not (stage == 1)
            or artifact_paths.get(key) != item.get("path")
        ):
            raise ValueError(f"{sample_id} progressive stage {stage} metadata changed")
        expected_path = artifact_root / key / f"{sample_id}.npy"
        path = _resolve_artifact(
            item.get("path"),
            repo_root=repo_root,
            artifact_root=artifact_root,
            expected_path=expected_path,
            label=f"{sample_id} progressive stage {stage}",
        )
        array = _load_probability_map(
            path,
            expected_shape=shape,
            expected_file_sha256=item.get("sha256"),
            expected_file_bytes=item.get("bytes"),
            expected_array_sha256=item.get("array_sha256"),
            label=f"{sample_id} progressive stage {stage}",
        )
        progressive_paths.append(path)
        progressive_file_hashes.append(str(item["sha256"]))
        progressive_array_hashes.append(str(item["array_sha256"]))
        progressive_arrays.append(array)

    primary = progressive_raw[0]
    aliases = {
        "primary_model_score_map_path": primary["path"],
        "primary_model_score_map_sha256": primary["sha256"],
        "primary_model_score_map_bytes": primary["bytes"],
        "primary_model_score_map_shape": primary["shape"],
        "primary_model_score_map_dtype": "float32",
        "primary_model_score_map_semantics": (
            "official_primary_NLC_sigmoid_probability"
        ),
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} primary map alias {key} changed")

    native_expected_path = artifact_root / "score_maps_native" / f"{sample_id}.npy"
    native_path = _resolve_artifact(
        row.get("score_map_path"),
        repo_root=repo_root,
        artifact_root=artifact_root,
        expected_path=native_expected_path,
        label=f"{sample_id} native probability map",
    )
    if (
        artifact_paths.get("native_probability") != row.get("score_map_path")
        or row.get("score_map_shape") != [height, width]
        or row.get("score_map_dtype") != "float32"
        or row.get("score_map_semantics")
        != "primary_probability_bilinear_align_corners_true_native_restore"
    ):
        raise ValueError(f"{sample_id} native map metadata changed")
    native_map = _load_probability_map(
        native_path,
        expected_shape=(height, width),
        expected_file_sha256=row.get("score_map_sha256"),
        expected_file_bytes=row.get("score_map_bytes"),
        expected_array_sha256=row.get("score_map_array_sha256"),
        label=f"{sample_id} native probability map",
    )
    expected_native = _bilinear_align_corners_true(
        progressive_arrays[0],
        width=width,
        height=height,
    )
    native_difference = float(np.max(np.abs(native_map - expected_native)))
    if not np.allclose(
        native_map,
        expected_native,
        rtol=STATIC_NATIVE_REPLAY_RTOL,
        atol=STATIC_NATIVE_REPLAY_ATOL,
    ):
        raise ValueError(f"{sample_id} persisted native interpolation replay changed")

    applicable = input_row.get("gt_mask_kind") in _T2_GT_KINDS
    mask_path: Path | None = None
    mask_file_hash: str | None = None
    mask_array_hash: str | None = None
    target = load_ground_truth(input_row, repo_root)
    if applicable:
        if target is None:
            raise ValueError(f"{sample_id} applicable T2 target disappeared")
        expected_mask_path = artifact_root / "masks_native" / f"{sample_id}.png"
        mask_path = _resolve_artifact(
            row.get("mask_path"),
            repo_root=repo_root,
            artifact_root=artifact_root,
            expected_path=expected_mask_path,
            label=f"{sample_id} native threshold mask",
        )
        mask_file_hash = _require_sha256(
            row.get("mask_sha256"),
            f"{sample_id} mask SHA-256",
        )
        if (
            artifact_paths.get("native_mask") != row.get("mask_path")
            or sha256_file(mask_path) != mask_file_hash
            or isinstance(row.get("mask_bytes"), bool)
            or not isinstance(row.get("mask_bytes"), int)
            or row["mask_bytes"] <= 0
            or mask_path.stat().st_size != row["mask_bytes"]
            or row.get("mask_shape") != [height, width]
            or row.get("mask_dtype") != "uint8"
            or row.get("mask_semantics") != "strict_probability_greater_than_0_5"
        ):
            raise ValueError(f"{sample_id} mask metadata changed")
        with Image.open(mask_path) as opened:
            if opened.mode != "L" or opened.size != (width, height):
                raise ValueError(f"{sample_id} mask image contract changed")
            mask = np.asarray(opened, dtype=np.uint8)
        if not np.isin(mask, (0, 255)).all() or not np.array_equal(
            mask,
            np.where(
                native_map > MASK_THRESHOLD,
                255,
                0,
            ).astype(np.uint8),
        ):
            raise ValueError(f"{sample_id} strict threshold mask changed")
        mask_array_hash = _array_sha256(mask)
        include_ap = str(input_row["condition"]) != "real"
        expected_localization = {
            "model_256": _independent_pixel_metrics(
                progressive_arrays[0],
                _resize_target_model_256(target),
                include_ap=include_ap,
            ),
            "native": _independent_pixel_metrics(
                native_map,
                target,
                include_ap=include_ap,
            ),
        }
        _assert_nested_close(
            row.get("localization"),
            expected_localization,
            label=f"{sample_id} localization",
        )
    else:
        if target is not None:
            raise ValueError(f"{sample_id} full-frame target was fabricated")
        for key in (
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_shape",
            "mask_dtype",
            "mask_semantics",
            "localization",
        ):
            if row.get(key) is not None:
                raise ValueError(f"{sample_id} full-frame row claims T2 field {key}")
        if artifact_paths.get("native_mask") is not None:
            raise ValueError(f"{sample_id} full-frame mask path was fabricated")

    return DenseArtifacts(
        sample_id=sample_id,
        progressive_paths=tuple(progressive_paths),  # type: ignore[arg-type]
        progressive_file_sha256=tuple(  # type: ignore[arg-type]
            progressive_file_hashes
        ),
        progressive_array_sha256=tuple(  # type: ignore[arg-type]
            progressive_array_hashes
        ),
        native_path=native_path,
        native_file_sha256=str(row["score_map_sha256"]),
        native_array_sha256=str(row["score_map_array_sha256"]),
        mask_path=mask_path,
        mask_file_sha256=mask_file_hash,
        mask_array_sha256=mask_array_hash,
        t2_applicable=applicable,
        width=width,
        height=height,
        static_softmax_max_abs_diff=softmax_difference,
        static_native_max_abs_diff=native_difference,
    )


def _exact_directory_inventory(
    directory: Path,
    expected_names: set[str],
    *,
    label: str,
) -> int:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {directory}")
    children = list(directory.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError(f"{label} contains unsafe/non-file entries")
    actual = {child.name for child in children}
    if actual != expected_names:
        raise ValueError(
            f"{label} inventory changed: "
            f"missing={sorted(expected_names - actual)[:1]}, "
            f"extra={sorted(actual - expected_names)[:1]}"
        )
    return len(actual)


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    selected: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, int]:
    _reject_symlink_components(artifact_root, "PSCC-Net artifact root")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe PSCC-Net artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != set(EXPECTED_ARTIFACT_INVENTORY) or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("PSCC-Net artifact root inventory changed")
    selected_ids = {str(row["sample_id"]) for row in selected}
    if set(artifacts) != selected_ids:
        raise ValueError("PSCC-Net validated artifact coverage changed")
    npy_names = {f"{sample_id}.npy" for sample_id in selected_ids}
    counts = {
        name: _exact_directory_inventory(
            artifact_root / name,
            npy_names,
            label=f"PSCC-Net {name}",
        )
        for name in EXPECTED_ARTIFACT_INVENTORY[:-1]
    }
    mask_names = {
        f"{sample_id}.png"
        for sample_id, artifact in artifacts.items()
        if artifact.t2_applicable
    }
    counts["masks_native"] = _exact_directory_inventory(
        artifact_root / "masks_native",
        mask_names,
        label="PSCC-Net masks_native",
    )
    return counts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, DenseArtifacts],
) -> str:
    records: list[dict[str, Any]] = []
    for sample_id, artifact in sorted(artifacts.items()):
        records.append(
            {
                "sample_id": sample_id,
                "progressive_paths": [
                    path.as_posix() for path in artifact.progressive_paths
                ],
                "progressive_file_sha256": list(artifact.progressive_file_sha256),
                "progressive_array_sha256": list(artifact.progressive_array_sha256),
                "native_path": artifact.native_path.as_posix(),
                "native_file_sha256": artifact.native_file_sha256,
                "native_array_sha256": artifact.native_array_sha256,
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


def _validate_attempt_history(
    selected: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = {str(row["sample_id"]) for row in selected}
    histories: dict[str, list[str]] = {}
    for line_number, attempt in enumerate(attempts, start=1):
        sample_id = attempt.get("sample_id")
        status = attempt.get("status")
        if sample_id not in expected_ids or status not in ("ok", "error"):
            raise ValueError(f"PSCC-Net physical attempt {line_number} changed")
        history = histories.setdefault(str(sample_id), [])
        if "ok" in history:
            raise ValueError("PSCC-Net has an attempt after terminal success")
        history.append(str(status))
    if any(not statuses or statuses[-1] != "ok" for statuses in histories.values()):
        raise ValueError("PSCC-Net latest attempt is not successful")
    return {
        "policy": "zero_or_more_errors_then_at_most_one_terminal_success_per_id",
        "physical_attempts": len(attempts),
        "ids_with_attempts": len(histories),
        "errors": sum(
            status == "error" for statuses in histories.values() for status in statuses
        ),
        "recovered_error_to_ok": sum(
            "error" in statuses[:-1] for statuses in histories.values()
        ),
    }


def _validate_run_directory(run_dir: Path) -> None:
    allowed = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
        "balanced250_metrics.json",
        "independent_audit.json",
    }
    children = list(run_dir.iterdir())
    if any(
        child.name not in allowed or child.is_symlink() or not child.is_file()
        for child in children
    ):
        raise ValueError("PSCC-Net run directory inventory changed")
    required = {
        "manifest.json",
        "expected_inputs.jsonl",
        "results.jsonl",
        "summary.json",
    }
    if not required.issubset({child.name for child in children}):
        raise FileNotFoundError("PSCC-Net run evidence is incomplete")


def _capture_snapshot(
    *,
    release: CanonicalRelease,
    manifest_path: Path,
    expected_path: Path,
    results_path: Path,
    summary_path: Path,
    artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, str]:
    return {
        "dataset_manifest_sha256": release.manifest_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "results_sha256": sha256_file(results_path),
        "runtime_summary_sha256": sha256_file(summary_path),
        "artifact_inventory_sha256": _artifact_inventory_sha256(artifacts),
    }


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    mode: str,
    psccnet_root: Path,
) -> RunBundle:
    expected_run_id = {
        "formal": DEFAULT_FORMAL_RUN_ID,
        "smoke_a": DEFAULT_SMOKE_RUN_ID_A,
        "smoke_b": DEFAULT_SMOKE_RUN_ID_B,
    }.get(mode)
    if expected_run_id is None or run_id != expected_run_id:
        raise ValueError(f"PSCC-Net {mode} requires frozen run ID")
    selection_mode = "formal" if mode == "formal" else "smoke"
    run_dir = _resolve_run_dir(
        results_dir,
        run_id,
        f"PSCC-Net {mode} run directory",
    )
    artifact_root = _resolve_run_dir(
        artifacts_dir,
        run_id,
        f"PSCC-Net {mode} artifact root",
    )
    _validate_run_directory(run_dir)
    manifest_path = run_dir / "manifest.json"
    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "PSCC-Net manifest")
    immutable = _require_mapping(
        manifest.get("immutable"),
        "PSCC-Net immutable config",
    )
    summary = _load_json(summary_path, "PSCC-Net runtime summary")
    expected_snapshot = _read_jsonl(
        expected_path,
        "PSCC-Net expected inputs",
    )
    physical = _read_jsonl(results_path, "PSCC-Net results")

    release = load_canonical_release(
        repo_root,
        runner.DEFAULT_DATASET_MANIFEST,
        verify_files=True,
    )
    selected, contract = _rebuild_contract(
        release=release,
        mode=selection_mode,
        recorded=immutable.get("dataset_contract"),
    )
    if expected_snapshot != list(selected):
        raise ValueError("PSCC-Net expected input snapshot changed")
    _validate_provenance(
        immutable=immutable,
        repo_root=repo_root,
        psccnet_root=psccnet_root,
    )
    _validate_immutable_static(
        immutable,
        run_id=run_id,
        mode=selection_mode,
        selected=selected,
        contract=contract,
        repo_root=repo_root,
        run_dir=run_dir,
        artifact_root=artifact_root,
    )
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "PSCC-Net manifest fingerprint",
    )
    _validate_manifest_envelope(
        manifest=manifest,
        immutable=immutable,
        run_id=run_id,
        fingerprint=fingerprint,
        release=release,
        selected=selected,
        contract=contract,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        summary=summary,
    )

    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical:
        sample_id = attempt.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in inputs_by_id:
            raise ValueError("PSCC-Net physical result has unknown sample")
        _validate_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            run_id=run_id,
            fingerprint=fingerprint,
        )
    history = _validate_attempt_history(selected, physical)
    latest = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=_score_spec(),
    )
    coverage = summarize_coverage(latest)
    require_complete_coverage(coverage)
    latest_results = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])]) for row in selected
    )
    if (
        summary.get("coverage") != coverage.as_dict()
        or summary.get("attempt_history") != history
    ):
        raise ValueError("PSCC-Net summary coverage/history changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "PSCC-Net execution",
    )
    if (
        execution.get("physical_result_rows") != len(physical)
        or execution.get("latest_result_rows") != len(latest.latest_by_sample_id)
        or execution.get("superseded_attempts") != latest.superseded_attempts
    ):
        raise ValueError("PSCC-Net execution/result counts changed")

    artifacts: dict[str, DenseArtifacts] = {}
    for index, (input_row, result) in enumerate(
        zip(selected, latest_results, strict=True),
        start=1,
    ):
        sample_id = str(input_row["sample_id"])
        if result.get("sample_id") != sample_id:
            raise ValueError("PSCC-Net terminal result order changed")
        artifacts[sample_id] = _validate_artifact_row(
            row=result,
            input_row=input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
        if len(selected) > SMOKE_IMAGES and (
            index % 100 == 0 or index == len(selected)
        ):
            print(
                f"[PSCC-Net artifact audit {index}/{len(selected)}] "
                f"exact {sample_id}",
                flush=True,
            )
    inventory = validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        artifacts=artifacts,
    )
    recorded_inventory = _require_mapping(
        manifest["outputs"].get("artifact_inventory"),
        "recorded artifact inventory",
    )
    if (
        inventory != recorded_inventory
        or summary.get("artifact_inventory") != inventory
    ):
        raise ValueError("PSCC-Net artifact inventory counts changed")
    expected_count = len(selected)
    expected_masks = FORMAL_T2_IMAGES if selection_mode == "formal" else SMOKE_T2_IMAGES
    if inventory != {
        "progressive_mask1": expected_count,
        "progressive_mask2": expected_count,
        "progressive_mask3": expected_count,
        "progressive_mask4": expected_count,
        "score_maps_native": expected_count,
        "masks_native": expected_masks,
    }:
        raise ValueError("PSCC-Net artifact coverage changed")

    snapshot = _capture_snapshot(
        release=release,
        manifest_path=manifest_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        artifacts=artifacts,
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        mode=selection_mode,
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
        physical_results=tuple(physical),
        latest_results=latest_results,
        coverage=coverage.as_dict(),
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    psccnet_root: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mode="formal",
        psccnet_root=psccnet_root,
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    psccnet_root: Path,
) -> RunBundle:
    mode = (
        "smoke_a"
        if run_id == DEFAULT_SMOKE_RUN_ID_A
        else "smoke_b" if run_id == DEFAULT_SMOKE_RUN_ID_B else "invalid"
    )
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mode=mode,
        psccnet_root=psccnet_root,
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
            raise ValueError("PSCC-Net T2 callback result identity changed")
        expected = selected_by_id.get(sample_id)
        artifact = bundle.artifacts.get(sample_id)
        if (
            expected is None
            or artifact is None
            or not artifact.t2_applicable
            or expected.get("gt_mask_kind") not in _T2_GT_KINDS
        ):
            raise ValueError(
                f"PSCC-Net shared T2 requested inapplicable map {sample_id}"
            )
        array = np.load(
            artifact.native_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            array.shape != (artifact.height, artifact.width)
            or array.dtype != np.float32
            or _array_sha256(array) != artifact.native_array_sha256
        ):
            raise ValueError(f"PSCC-Net T2 map changed during metrics: {sample_id}")
        return array

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
            "PSCC-Net formal metrics require iterations=1000 and " "seed=20260726"
        )
    if (
        bundle.mode != "formal"
        or len(bundle.selected) != FORMAL_IMAGES
        or sum(row.get("gt_mask_kind") in _T2_GT_KINDS for row in bundle.selected)
        != FORMAL_T2_IMAGES
    ):
        raise ValueError("PSCC-Net metrics require formal 1775/1025")
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
        raise ValueError("PSCC-Net shared T1 metrics are incomplete")
    # PSCC-Net's official mask, artifact audit, and fresh replay all use the
    # strict ``score > 0.5`` comparator above.  The shared Balanced250 T2
    # reducer has a separately frozen ``score >= 0.5`` contract, so adapt only
    # this reducer call.  Unlike MVSS-Net's uint8/255 maps, PSCC-Net persists
    # float32 probabilities that can equal 0.5 exactly; the two operators are
    # therefore not generally equivalent and must remain explicit.
    t2 = summarize_balanced250_t2(
        bundle.release.inputs,
        selected_results,
        repo_root=bundle.release.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name="psccnet_native_manipulation_probability",
        threshold=MASK_THRESHOLD,
        threshold_operator=SHARED_T2_THRESHOLD_OPERATOR,
        iterations=iterations,
        seed=seed,
    )
    if (
        t2.get("schema_version") != T2_METRICS_SCHEMA_VERSION
        or t2.get("coverage", {}).get("is_complete") is not True
        or t2.get("coverage", {}).get("native_maps_evaluated") != FORMAL_T2_IMAGES
    ):
        raise ValueError("PSCC-Net shared T2 metrics are incomplete")
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
        raise ValueError("PSCC-Net T2 full-frame exclusion changed")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "formal_images_t1": FORMAL_IMAGES,
        "formal_images_t2": FORMAL_T2_IMAGES,
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_operator": THRESHOLD_OPERATOR,
        "official_t2_threshold": MASK_THRESHOLD,
        "official_t2_threshold_operator": THRESHOLD_OPERATOR,
        "shared_t2_threshold": MASK_THRESHOLD,
        "shared_t2_threshold_operator": SHARED_T2_THRESHOLD_OPERATOR,
        "shared_t2_operator_equivalent_to_official": False,
        "t1": t1,
        "t2": t2,
    }


def _verify_bundle_unchanged(bundle: RunBundle) -> None:
    paths = {
        "manifest_sha256": bundle.manifest_path,
        "expected_inputs_sha256": bundle.expected_path,
        "results_sha256": bundle.results_path,
        "runtime_summary_sha256": bundle.summary_path,
    }
    for key, path in paths.items():
        if sha256_file(path) != bundle.evidence_snapshot[key]:
            raise ValueError(f"PSCC-Net evidence changed during audit: {key}")
    release = load_canonical_release(
        bundle.release.repo_root,
        bundle.release.manifest_path,
        verify_files=True,
    )
    if release.manifest_sha256 != bundle.evidence_snapshot["dataset_manifest_sha256"]:
        raise ValueError("Balanced250 release changed during PSCC-Net audit")
    for sample_id, artifact in bundle.artifacts.items():
        if any(
            sha256_file(path) != digest
            for path, digest in zip(
                artifact.progressive_paths,
                artifact.progressive_file_sha256,
                strict=True,
            )
        ):
            raise ValueError(f"PSCC-Net progressive artifact changed: {sample_id}")
        if sha256_file(artifact.native_path) != artifact.native_file_sha256 or (
            artifact.mask_path is not None
            and sha256_file(artifact.mask_path) != artifact.mask_file_sha256
        ):
            raise ValueError(f"PSCC-Net artifact changed: {sample_id}")
    if (
        _artifact_inventory_sha256(bundle.artifacts)
        != bundle.evidence_snapshot["artifact_inventory_sha256"]
    ):
        raise ValueError("PSCC-Net artifact inventory changed during audit")


def _smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(stable_json(row))
    for key in (
        "run_id",
        "run_manifest_fingerprint",
        "completed_at",
        "latency_ms",
        "peak_cuda_memory_bytes",
    ):
        value[key] = f"<run-specific:{key}>"
    for key in (
        "primary_model_score_map_path",
        "score_map_path",
        "mask_path",
    ):
        if value.get(key) is not None:
            value[key] = f"<run-specific:{key}>"
    artifact_paths = _require_mapping(
        value.get("artifact_paths"),
        "smoke artifact paths",
    )
    for key, child in artifact_paths.items():
        if child is not None:
            artifact_paths[key] = f"<run-specific:{key}>"
    value["artifact_paths"] = artifact_paths
    progressive = value.get("progressive_maps")
    if not isinstance(progressive, list) or len(progressive) != 4:
        raise ValueError("PSCC-Net smoke progressive maps changed")
    for index, stage in enumerate(progressive):
        item = _require_mapping(stage, "smoke progressive map")
        item["path"] = f"<run-specific:stage{item.get('stage')}>"
        progressive[index] = item
    return value


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(stable_json(immutable))
    value["run_id"] = "<run-id>"
    outputs = _require_mapping(value.get("outputs"), "smoke outputs")
    for key in outputs:
        outputs[key] = f"<run-specific:{key}>"
    value["outputs"] = outputs
    return value


def compare_computational_results(
    reference: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    reference_artifacts: Mapping[str, DenseArtifacts],
    replay_artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, Any]:
    if len(reference) != SMOKE_IMAGES or len(replay) != SMOKE_IMAGES:
        raise ValueError("PSCC-Net smoke comparison requires 35+35")
    expected_ids = {str(row["sample_id"]) for row in reference}
    if (
        set(reference_artifacts) != expected_ids
        or set(replay_artifacts) != expected_ids
    ):
        raise ValueError("PSCC-Net smoke artifact coverage changed")
    applicable = 0
    files_compared = 0
    for left, right in zip(reference, replay, strict=True):
        sample_id = str(left["sample_id"])
        if right.get("sample_id") != sample_id:
            raise ValueError("PSCC-Net smoke row order changed")
        if _smoke_projection(left) != _smoke_projection(right):
            raise ValueError(f"PSCC-Net smoke computational row changed: {sample_id}")
        left_artifact = reference_artifacts[sample_id]
        right_artifact = replay_artifacts[sample_id]
        if left_artifact.t2_applicable != right_artifact.t2_applicable:
            raise ValueError(f"PSCC-Net smoke T2 applicability changed: {sample_id}")
        for left_path, right_path in zip(
            left_artifact.progressive_paths,
            right_artifact.progressive_paths,
            strict=True,
        ):
            if left_path.read_bytes() != right_path.read_bytes():
                raise ValueError(
                    f"PSCC-Net smoke progressive bytes changed: {sample_id}"
                )
            files_compared += 1
        if (
            left_artifact.native_path.read_bytes()
            != right_artifact.native_path.read_bytes()
        ):
            raise ValueError(f"PSCC-Net smoke native map bytes changed: {sample_id}")
        files_compared += 1
        if (left_artifact.mask_path is None) is not (right_artifact.mask_path is None):
            raise ValueError(f"PSCC-Net smoke mask applicability changed: {sample_id}")
        if left_artifact.mask_path is not None and right_artifact.mask_path is not None:
            if (
                left_artifact.mask_path.read_bytes()
                != right_artifact.mask_path.read_bytes()
            ):
                raise ValueError(f"PSCC-Net smoke mask bytes changed: {sample_id}")
            files_compared += 1
        applicable += int(left_artifact.t2_applicable)
    if applicable != SMOKE_T2_IMAGES or files_compared != 195:
        raise ValueError("PSCC-Net smoke comparison coverage changed")
    return {
        "images_compared": SMOKE_IMAGES,
        "t2_applicable_images_compared": SMOKE_T2_IMAGES,
        "t2_not_applicable_images_compared": SMOKE_T2_NOT_APPLICABLE,
        "artifact_files_compared_exact": files_compared,
        "computational_result_projection_exact": True,
        "four_progressive_NPY_files_exact_per_image": True,
        "native_NPY_file_exact_per_image": True,
        "applicable_threshold_PNG_files_exact": True,
        "fullframe_masks_absent_in_both_runs": True,
        "detection_head_logits_probabilities_scores_exact": True,
        "localization_metrics_exact": True,
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
    expected_sha256 = _json_sha256(value)
    atomic_write_json(path, dict(value))
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
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
        / (f"{reference_run_id}__vs__{replay_run_id}" "_comparison.json")
    )


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    psccnet_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        reference_run_id != DEFAULT_SMOKE_RUN_ID_A
        or replay_run_id != DEFAULT_SMOKE_RUN_ID_B
        or reference_run_id == replay_run_id
    ):
        raise ValueError("PSCC-Net smoke comparison requires frozen A then B")
    reference = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=reference_run_id,
        psccnet_root=psccnet_root,
    )
    replay = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=replay_run_id,
        psccnet_root=psccnet_root,
    )
    if (
        [row["sample_id"] for row in reference.selected]
        != [row["sample_id"] for row in replay.selected]
        or reference.contract.as_dict() != replay.contract.as_dict()
        or _smoke_immutable_projection(reference.immutable)
        != _smoke_immutable_projection(replay.immutable)
        or reference.immutable.get("runtime") != replay.immutable.get("runtime")
    ):
        raise ValueError("PSCC-Net A/B smoke immutable contract changed")
    structural_golden = independent_structural_golden(
        psccnet_root=psccnet_root,
        recorded_checkpoint_audit=_require_mapping(
            reference.immutable.get("checkpoint_audit"),
            "reference checkpoint audit",
        ),
        recorded_model_audit=_require_mapping(
            reference.immutable.get("model_audit"),
            "reference model audit",
        ),
    )
    if replay.immutable.get("checkpoint_audit") != reference.immutable.get(
        "checkpoint_audit"
    ) or replay.immutable.get("model_audit") != reference.immutable.get("model_audit"):
        raise ValueError("PSCC-Net smoke structural audits differ")
    comparison = compare_computational_results(
        reference.latest_results,
        replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
    )
    _verify_bundle_unchanged(reference)
    _verify_bundle_unchanged(replay)
    report: dict[str, Any] = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "exact_reproduction_passed",
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "compared_at": utc_now(),
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            "evidence_snapshot": dict(reference.evidence_snapshot),
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            "evidence_snapshot": dict(replay.evidence_snapshot),
        },
        "selection": {
            "selected_images": SMOKE_IMAGES,
            "per_condition": SMOKE_PER_CONDITION,
            "selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
            "t2_applicable_images": SMOKE_T2_IMAGES,
            "t2_not_applicable_images": SMOKE_T2_NOT_APPLICABLE,
        },
        "recorded_runtime_exact": True,
        "independent_structural_golden": structural_golden,
        "comparison": comparison,
        "analyzer_source": {
            "path": "eval/opensource/analyze_psccnet_balanced.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
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
        label="PSCC-Net smoke comparison output",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_verified(
        output,
        report,
        label="PSCC-Net smoke comparison",
    )
    return report


def _configure_recorded_runtime(
    *,
    recorded: Mapping[str, Any],
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    expected = _validate_runtime(recorded, label="recorded runtime")
    if device_text != expected.get("device"):
        raise ValueError(
            "PSCC-Net replay device must equal the recorded runtime device"
        )
    device, current = runner.configure_runtime(device_text)
    validated = _validate_runtime(current, label="current replay runtime")
    if validated != expected:
        raise ValueError("PSCC-Net current runtime differs from recorded")
    return device, validated


def replay_model(
    bundle: RunBundle,
    *,
    psccnet_root: Path,
    device: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Freshly replay all 1,775 model forwards and demand exact persisted bytes."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("PSCC-Net fresh replay requires formal 1775")
    models: tuple[Any, Any, Any] | None = None
    fresh_rows: list[dict[str, Any]] = []
    progressive_maps_compared = 0
    native_maps_compared = 0
    masks_derived = 0
    fullframe_masks_absent = 0
    try:
        models, loaded_device = legacy.load_model(
            psccnet_root=psccnet_root,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("PSCC-Net fresh model loaded on wrong device")
        for index, (input_row, persisted) in enumerate(
            zip(bundle.selected, bundle.latest_results, strict=True),
            start=1,
        ):
            sample_id = str(input_row["sample_id"])
            if persisted.get("sample_id") != sample_id:
                raise ValueError("PSCC-Net fresh replay order changed")
            width = int(input_row["width"])
            height = int(input_row["height"])
            input_path = _safe_repo_file(
                input_row.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                expected_path=(
                    bundle.release.repo_root / str(input_row["canonical_path"])
                ),
                label=f"{sample_id} replay input",
            )
            tensor, preprocess = _independent_preprocess_tensor(input_path)
            if tensor.shape != (3, height, width) or preprocess != persisted.get(
                "preprocess"
            ):
                raise ValueError(f"{sample_id} fresh preprocessing changed")
            (
                progressive_maps,
                native_map,
                logits,
                probabilities,
                _peak_bytes,
                _latency_ms,
            ) = legacy.infer_one(
                models,
                device,
                tensor,
                native_width=width,
                native_height=height,
            )
            if len(progressive_maps) != 4:
                raise ValueError(f"{sample_id} fresh progressive output count changed")
            artifact = bundle.artifacts[sample_id]
            for stage, (fresh_map, stored_path) in enumerate(
                zip(
                    progressive_maps,
                    artifact.progressive_paths,
                    strict=True,
                ),
                start=1,
            ):
                stored_map = np.load(
                    stored_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                fresh_array = np.ascontiguousarray(
                    fresh_map,
                    dtype=np.float32,
                )
                if not np.array_equal(fresh_array, stored_map):
                    raise ValueError(
                        f"{sample_id} fresh progressive map{stage} changed"
                    )
                progressive_maps_compared += 1
                del stored_map, fresh_array
            stored_native = np.load(
                artifact.native_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            fresh_native = np.ascontiguousarray(
                native_map,
                dtype=np.float32,
            )
            if not np.array_equal(fresh_native, stored_native):
                raise ValueError(f"{sample_id} fresh native map changed")
            native_maps_compared += 1

            fresh_logits = np.ascontiguousarray(logits, dtype=np.float32)
            fresh_probabilities = np.ascontiguousarray(
                probabilities,
                dtype=np.float32,
            )
            recorded_logits = np.asarray(
                persisted.get("classification_logits"),
                dtype=np.float32,
            )
            recorded_probabilities = np.asarray(
                persisted.get("classification_probabilities"),
                dtype=np.float32,
            )
            if (
                fresh_logits.shape != (2,)
                or fresh_probabilities.shape != (2,)
                or not np.array_equal(fresh_logits, recorded_logits)
                or not np.array_equal(
                    fresh_probabilities,
                    recorded_probabilities,
                )
            ):
                raise ValueError(f"{sample_id} fresh DetectionHead output changed")
            independent_probabilities = _float32_softmax(fresh_logits)
            if not np.allclose(
                fresh_probabilities,
                independent_probabilities,
                rtol=0.0,
                atol=STATIC_SOFTMAX_ABS_TOLERANCE,
            ):
                raise ValueError(f"{sample_id} fresh DetectionHead softmax changed")
            if float(fresh_probabilities[1]) != persisted.get("ai_score") or (
                float(fresh_probabilities[1]) > CLASSIFICATION_THRESHOLD
            ) != (persisted.get("classification_decision") == "forged"):
                raise ValueError(f"{sample_id} fresh T1 decision changed")

            if artifact.t2_applicable:
                if artifact.mask_path is None:
                    raise ValueError(f"{sample_id} applicable replay mask missing")
                with Image.open(artifact.mask_path) as opened:
                    stored_mask = np.asarray(opened, dtype=np.uint8)
                fresh_mask = np.where(
                    fresh_native > MASK_THRESHOLD,
                    255,
                    0,
                ).astype(np.uint8)
                if not np.array_equal(fresh_mask, stored_mask):
                    raise ValueError(f"{sample_id} fresh strict threshold mask changed")
                masks_derived += 1
                del stored_mask, fresh_mask
            else:
                if (
                    artifact.mask_path is not None
                    or persisted.get("mask_path") is not None
                    or persisted.get("localization") is not None
                ):
                    raise ValueError(f"{sample_id} full-frame replay fabricated T2")
                fullframe_masks_absent += 1

            # All metric-bearing fields and all native maps have just been
            # proven exact.  Keep the immutable persisted row so timing and
            # resource diagnostics cannot accidentally enter the reducers.
            fresh_rows.append(dict(persisted))
            del (
                tensor,
                progressive_maps,
                native_map,
                logits,
                probabilities,
                stored_native,
                fresh_native,
                fresh_logits,
                fresh_probabilities,
                recorded_logits,
                recorded_probabilities,
                independent_probabilities,
            )
            if index % 25 == 0 or index == FORMAL_IMAGES:
                print(
                    f"[fresh replay {index}/{FORMAL_IMAGES}] " f"exact {sample_id}",
                    flush=True,
                )
            gc.collect()
            if getattr(device, "type", None) == "cuda":
                import torch

                torch.cuda.empty_cache()
    finally:
        models = None
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            import torch

            torch.cuda.empty_cache()
    if (
        len(fresh_rows) != FORMAL_IMAGES
        or progressive_maps_compared != FORMAL_IMAGES * 4
        or native_maps_compared != FORMAL_IMAGES
        or masks_derived != FORMAL_T2_IMAGES
        or fullframe_masks_absent != FORMAL_T2_NOT_APPLICABLE
    ):
        raise ValueError("PSCC-Net fresh replay coverage changed")
    return (
        {
            "status": "fresh_full_model_replay_exact",
            "selected_images_freshly_reopened": FORMAL_IMAGES,
            "selected_images_freshly_preprocessed": FORMAL_IMAGES,
            "model_forwards": FORMAL_IMAGES,
            "t1_logits_compared_exact": FORMAL_IMAGES,
            "t1_probabilities_compared_exact": FORMAL_IMAGES,
            "t1_scores_compared_exact": FORMAL_IMAGES,
            "progressive_probability_maps_compared_exact": (progressive_maps_compared),
            "native_probability_maps_compared_exact": (native_maps_compared),
            "applicable_masks_rederived_exact": masks_derived,
            "fullframe_masks_not_created": fullframe_masks_absent,
            "maximum_t1_logit_abs_diff": 0.0,
            "maximum_t1_probability_abs_diff": 0.0,
            "maximum_progressive_map_abs_diff": 0.0,
            "maximum_native_map_abs_diff": 0.0,
        },
        tuple(fresh_rows),
    )


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    psccnet_root: Path,
    device_text: str,
    metrics_output_path: Path,
    audit_output_path: Path,
    replay: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError(f"PSCC-Net formal audit requires {DEFAULT_FORMAL_RUN_ID}")
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        psccnet_root=psccnet_root,
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
                *artifact.progressive_paths,
                artifact.native_path,
                artifact.mask_path,
            )
            if path is not None
        ],
    ]
    metrics_output = _validate_output_path(
        metrics_output_path,
        expected_path=bundle.run_dir / "balanced250_metrics.json",
        protected=protected,
        label="PSCC-Net Balanced250 metrics output",
    )
    audit_output = _validate_output_path(
        audit_output_path,
        expected_path=bundle.run_dir / "independent_audit.json",
        protected=[*protected, metrics_output],
        label="PSCC-Net independent audit output",
    )

    structural_golden = independent_structural_golden(
        psccnet_root=psccnet_root,
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
    fresh_metrics_exact: bool | None = None
    if replay:
        device, current_runtime = _configure_recorded_runtime(
            recorded=recorded_runtime,
            device_text=device_text,
        )
        fresh_replay, fresh_rows = replay_model(
            bundle,
            psccnet_root=psccnet_root,
            device=device,
        )
        if fresh_rows != bundle.latest_results:
            raise ValueError("PSCC-Net fresh replay metric rows changed")
        # T1 consumes only the 1,775 exact result rows.  T2 consumes only the
        # 1,025 exact applicable native maps.  Re-running AP/bootstrap on
        # byte-identical inputs cannot add evidence, so equivalence follows
        # from the exhaustive gates above.
        fresh_metrics_exact = True

    source_after = _independent_source_record(psccnet_root)
    assets_after = _independent_assets_record(psccnet_root)
    environment_after = _independent_environment_record()
    artifact_ignore_after = _independent_artifact_ignore(repo_root)
    if (
        source_after != bundle.immutable.get("source")
        or assets_after != bundle.immutable.get("assets")
        or environment_after != bundle.immutable.get("environment")
        or artifact_ignore_after != bundle.immutable.get("artifact_ignore")
    ):
        raise ValueError("PSCC-Net source/assets/environment changed during analysis")
    _verify_bundle_unchanged(bundle)

    metrics_sha = _json_sha256(metrics)
    analyzer_path = Path(__file__).resolve()
    static_softmax_max = max(
        artifact.static_softmax_max_abs_diff for artifact in bundle.artifacts.values()
    )
    static_native_max = max(
        artifact.static_native_max_abs_diff for artifact in bundle.artifacts.values()
    )
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": ("replay_audit_passed" if replay else "artifact_audit_passed"),
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
            "fullframe_t2_not_applicable_images": (FORMAL_T2_NOT_APPLICABLE),
        },
        "coverage": dict(bundle.coverage),
        "evidence_snapshot": dict(bundle.evidence_snapshot),
        "independent_provenance": {
            "source_exact_and_clean": True,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_bound_files": len(runner.SOURCE_BOUND_FILES),
            "official_assets_git_tracked": 4,
            "checkpoint_bundle_sha256": (legacy.CHECKPOINT_BUNDLE_SHA256),
            "analysis_environment_exact": True,
            "gitignore_contract_exact": True,
            "source_assets_environment_unchanged_after_replay": True,
        },
        "independent_structural_golden": structural_golden,
        "recorded_runtime": recorded_runtime,
        "recorded_runtime_reproduced": current_runtime,
        "artifact_audit": {
            "progressive_mask1_float32_verified": FORMAL_IMAGES,
            "progressive_mask2_float32_verified": FORMAL_IMAGES,
            "progressive_mask3_float32_verified": FORMAL_IMAGES,
            "progressive_mask4_float32_verified": FORMAL_IMAGES,
            "progressive_maps_total_verified": FORMAL_IMAGES * 4,
            "native_probability_maps_float32_verified": FORMAL_IMAGES,
            "native_threshold_png_masks_verified": FORMAL_T2_IMAGES,
            "fullframe_masks_absent": FORMAL_T2_NOT_APPLICABLE,
            "all_paths_canonical": True,
            "all_file_hashes_exact": True,
            "all_array_hashes_exact": True,
            "all_shapes_and_dtypes_exact": True,
            "all_values_finite_and_in_unit_interval": True,
            "persisted_detection_head_softmax_replayed": True,
            "persisted_detection_head_softmax_max_abs_diff": (static_softmax_max),
            "persisted_detection_head_softmax_abs_tolerance": (
                STATIC_SOFTMAX_ABS_TOLERANCE
            ),
            "persisted_native_align_corners_true_replayed": True,
            "persisted_native_map_max_abs_diff": static_native_max,
            "persisted_native_map_rtol": STATIC_NATIVE_REPLAY_RTOL,
            "persisted_native_map_atol": STATIC_NATIVE_REPLAY_ATOL,
            "applicable_png_exact_score_map_strict_gt_0_5": True,
            "fullframe_localization_metrics_absent": True,
        },
        "metrics": {
            "schema_version": metrics["schema_version"],
            "t1_schema_version": metrics["t1"]["schema_version"],
            "t2_schema_version": metrics["t2"]["schema_version"],
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "shared_balanced250_metrics_only": True,
            "official_t2_threshold_operator": (
                metrics["official_t2_threshold_operator"]
            ),
            "shared_t2_threshold_operator": (metrics["shared_t2_threshold_operator"]),
            "shared_t2_operator_equivalent_to_official": (
                metrics["shared_t2_operator_equivalent_to_official"]
            ),
            "metrics_sha256": metrics_sha,
        },
        "fresh_model_replay": fresh_replay,
        "fresh_model_metrics_exact": fresh_metrics_exact,
        "fresh_model_metrics_equivalence_proof": (
            {
                "t1_inputs": "all_1775_metric_result_rows_exact",
                "t2_inputs": ("all_1025_applicable_native_score_maps_exact"),
                "fullframe_maps_excluded_from_t2": True,
                "metric_reducer_rerun_on_identical_inputs": False,
            }
            if replay
            else None
        ),
        "scientific_boundaries": {
            "t1_uses_all_1775_images": True,
            "t2_uses_only_real_and_local_1025_images": True,
            "fullframe_750_t2_is_not_applicable": True,
            "fullframe_diagnostic_maps_are_not_t2_results": True,
            "t1_is_independent_detection_head_class1_softmax": True,
            "t2_is_stage1_NLC_sigmoid_probability": True,
            "strict_threshold_operator_is_greater_than": True,
            "official_mask_audit_and_fresh_replay_use_strict_gt": True,
            "shared_t2_reducer_uses_frozen_gte": True,
            "threshold_operators_not_generally_equivalent_for_float32": True,
            "author_published_numeric_golden_available": False,
            "no_numeric_golden_was_fabricated": True,
            "fresh_full_model_replay_is_default": True,
        },
        "license": {
            "identifier": "MIT",
            "commercial_use_permission": True,
            "redistribution_permission": True,
            "official_checkpoint_separate_terms_present": False,
            "trained_data_rights_not_audited": True,
            "dependency_compliance_not_legal_advice": True,
            "benchmark_use_does_not_establish_product_clearance": True,
        },
        "analyzer_source": {
            "path": "eval/opensource/analyze_psccnet_balanced.py",
            "bytes": analyzer_path.stat().st_size,
            "sha256": sha256_file(analyzer_path),
        },
        "artifacts": {
            "metrics_path": repo_relative(metrics_output, repo_root),
            "metrics_sha256": metrics_sha,
            "audit_path": repo_relative(audit_output, repo_root),
        },
    }
    _write_json_verified(
        metrics_output,
        metrics,
        label="PSCC-Net Balanced250 metrics",
    )
    _write_json_verified(
        audit_output,
        report,
        label="PSCC-Net independent audit",
    )
    if sha256_file(metrics_output) != metrics_sha:
        raise ValueError("PSCC-Net metrics changed after write")
    return report


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
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
    parser.add_argument("--run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument(
        "--psccnet-root",
        type=Path,
        default=DEFAULT_PSCCNET_ROOT,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="must exactly reproduce the recorded formal runtime device",
    )
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
    repo_root = args.repo_root.resolve()
    results_dir = _safe_standard_root(
        args.results_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="PSCC-Net results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="PSCC-Net artifacts root",
    )
    run_id = _valid_run_id(args.run_id)
    psccnet_root = _anchored(args.psccnet_root, repo_root)
    if args.compare_smoke_run_id is not None:
        compare_id = _valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
        ):
            raise ValueError("PSCC-Net smoke comparison accepts no formal options")
        expected_output = _comparison_output_path(
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
        )
        comparison_output = (
            _anchored(args.comparison_output, repo_root)
            if args.comparison_output is not None
            else expected_output
        )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
            psccnet_root=psccnet_root,
            output_path=comparison_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError("--comparison-output requires --compare-smoke-run-id")
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError("PSCC-Net formal run ID is frozen")
    run_dir = _resolve_run_dir(
        results_dir,
        run_id,
        "PSCC-Net formal run directory",
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
        psccnet_root=psccnet_root,
        device_text=str(args.device),
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
