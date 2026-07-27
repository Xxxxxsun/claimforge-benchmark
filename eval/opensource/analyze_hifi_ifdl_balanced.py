#!/usr/bin/env python3
"""Fail-closed Balanced250 audit, metrics, smoke comparison, and replay.

HiFi-IFDL exposes two genuinely different outputs.  T1 is the frozen fine
14-class hierarchy head, scored as ``1 - P(authentic)`` with a strict
``> 0.5`` benchmark decision.  T2 is the raw nonnegative hypersphere
distance from the released authentic center, thresholded inclusively at
``>= 2.3``.  Real and local-insertion inputs are T2-applicable; full-frame
conditional edits are explicitly localization-N/A.

This analyzer treats every run envelope and raw artifact as untrusted.  It
rebuilds the canonical selection, verifies source/assets/runtime/license and
the CPU strict-load evidence, independently replays preprocessing, reopens
all persisted embeddings/maps/masks, recomputes shared Balanced250 metrics,
requires exact frozen A/B smoke reproduction, and by default performs a fresh
ordered 1,775-image model replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
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

from eval.opensource import run_hifi_ifdl as legacy
from eval.opensource import run_hifi_ifdl_balanced as runner
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


AUDIT_SCHEMA_VERSION = "hifi_ifdl_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "hifi_ifdl_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "hifi_ifdl_balanced250_summary_v2"
T1_METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
T2_METRICS_SCHEMA_VERSION = "balanced250_t2_summary_v1"

DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = runner.DEFAULT_ARTIFACTS_DIR
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
DEFAULT_HIFI_ROOT = legacy.DEFAULT_HIFI_ROOT
DEFAULT_HRNET_CHECKPOINT = legacy.DEFAULT_HRNET_CHECKPOINT
DEFAULT_NLC_CHECKPOINT = legacy.DEFAULT_NLC_CHECKPOINT

FORMAL_IMAGES = 1_775
FORMAL_T2_IMAGES = 1_025
SMOKE_IMAGES = 35
SMOKE_T2_IMAGES = 20
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_726
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 2.3
PAIRWISE_DISTANCE_EPSILON = 1e-6
DISTANCE_ABSOLUTE_TOLERANCE = 2e-5
NATIVE_RESTORE_ABSOLUTE_TOLERANCE = 3e-5
SIMPLEX_SUM_ABSOLUTE_TOLERANCE = runner.SIMPLEX_SUM_ABS_TOLERANCE
STATIC_SOFTMAX_ABSOLUTE_TOLERANCE = runner.STATIC_CPU_SOFTMAX_ABS_TOLERANCE
ARTIFACT_AUDIT_WORKERS = 8

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

EXPECTED_ADAPTER_SOURCE_PATHS = runner.ADAPTER_SOURCE_PATHS
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
_RUN_SPECIFIC_RESULT_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "artifact_paths",
        "embedding_artifact",
        "distance_model_artifact",
        "distance_native_artifact",
        "score_map_path",
        "mask_path",
    }
)


@dataclass(frozen=True)
class DenseArtifacts:
    """Validated persisted tensors for one T2-applicable result."""

    sample_id: str
    embedding_path: Path
    embedding_file_sha256: str
    embedding_array_sha256: str
    model_distance_path: Path
    model_distance_file_sha256: str
    model_distance_array_sha256: str
    native_distance_path: Path
    native_distance_file_sha256: str
    native_distance_array_sha256: str
    mask_path: Path
    mask_file_sha256: str
    mask_array_sha256: str
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
    history: Mapping[str, Any]
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


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
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                row = _require_mapping(value, f"{label}:{line_number}")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{label}:{line_number} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return rows


def _lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    candidate = path if path.is_absolute() else (base or Path.cwd()) / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_standard_root(
    path: Path,
    *,
    repo_root: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    candidate = _lexical_absolute(path, base=repo_root)
    expected = _lexical_absolute(expected_relative, base=repo_root)
    _reject_symlink_components(candidate, label)
    _reject_symlink_components(expected, f"expected {label}")
    if candidate.resolve() != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected_relative}")
    return expected.resolve()


def _resolve_run_dir(root: Path, run_id: Any, label: str) -> Path:
    safe_id = runner._valid_run_id(run_id)
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


def _absolute_regular_file(
    path: Path,
    *,
    expected_path: Path,
    label: str,
) -> Path:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, label)
    if (
        not path.is_absolute()
        or candidate != path
        or path.resolve() != expected_path.resolve()
    ):
        raise ValueError(f"{label} path changed")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    return path.resolve()


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _score_spec() -> ScoreSpec:
    score = runner.SCORE_SPEC
    expected = {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": ">",
    }
    if not isinstance(score, ScoreSpec) or score.as_dict() != expected:
        raise ValueError("HiFi-IFDL runner SCORE_SPEC changed")
    return score


def _selection_for_mode(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
) -> tuple[SelectionSpec, tuple[dict[str, Any], ...]]:
    if mode not in {"formal", "smoke"}:
        raise ValueError("analyzer only accepts formal or smoke runs")
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
        raise ValueError("HiFi-IFDL selection capability changed")
    rows = tuple(dict(row) for row in selected)
    counts = Counter(str(row["condition"]) for row in rows)
    t2_count = sum(runner._t2_semantics(row)[0] for row in rows)
    if mode == "formal":
        expected = (
            FORMAL_IMAGES,
            FORMAL_T2_IMAGES,
            FORMAL_SELECTED_ROWS_SHA256,
            FORMAL_SELECTED_IDS_SHA256,
        )
        if (
            len(rows),
            t2_count,
            _rows_sha256(rows),
            selected_ids_sha256(str(row["sample_id"]) for row in rows),
        ) != expected or dict(counts) != FORMAL_COUNTS:
            raise ValueError("formal HiFi-IFDL selection drifted")
    elif mode == "smoke":
        expected_counts = Counter(
            {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
        )
        if (
            len(rows) != SMOKE_IMAGES
            or t2_count != SMOKE_T2_IMAGES
            or counts != expected_counts
            or _rows_sha256(rows) != SMOKE_SELECTED_ROWS_SHA256
            or selected_ids_sha256(str(row["sample_id"]) for row in rows)
            != SMOKE_SELECTED_IDS_SHA256
        ):
            raise ValueError("smoke HiFi-IFDL selection drifted")
    else:
        raise ValueError("analyzer only accepts formal or smoke runs")
    return spec, rows


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    sources = _require_mapping(value, "immutable.adapter_sources")
    if tuple(sources) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("HiFi-IFDL adapter source order/set changed")
    verified: dict[str, dict[str, Any]] = {}
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
        verified[relative] = expected
    return verified


def _independent_environment_record() -> dict[str, Any]:
    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    pyvenv = prefix / "pyvenv.cfg"
    versions: dict[str, str | None] = {}
    for name in runner.EXPECTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    expected_cache = runner.FROZEN_PYTHONPYCACHEPREFIX.resolve()
    actual_cache = (
        Path(sys.pycache_prefix).resolve()
        if isinstance(sys.pycache_prefix, str)
        else None
    )
    if (
        executable != runner.EXPECTED_PYTHON_EXECUTABLE
        or prefix != runner.EXPECTED_VENV_ROOT
        or platform.python_version() != "3.12.3"
        or not pyvenv.is_file()
        or pyvenv.is_symlink()
        or pyvenv.stat().st_size != runner.EXPECTED_PYVENV_BYTES
        or sha256_file(pyvenv) != runner.EXPECTED_PYVENV_SHA256
        or versions != runner.EXPECTED_PACKAGES
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or sys.dont_write_bytecode is not True
        or actual_cache != expected_cache
        or not expected_cache.is_dir()
        or expected_cache.is_symlink()
        or any(expected_cache.iterdir())
    ):
        raise ValueError("live HiFi-IFDL analysis environment changed")
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
        "PYTHONDONTWRITEBYTECODE": "1",
        "sys_dont_write_bytecode": True,
        "PYTHONPYCACHEPREFIX": str(expected_cache),
        "sys_pycache_prefix": str(expected_cache),
        "pycache_prefix_initially_empty": True,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _independent_source_record(hifi_root: Path) -> dict[str, Any]:
    root = _absolute_regular_directory(
        hifi_root,
        expected_path=DEFAULT_HIFI_ROOT,
        label="HiFi-IFDL source root",
    )
    commit = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    origin = _git_value(root, "remote", "get-url", "origin")
    if (
        commit != legacy.MODEL_SOURCE_COMMIT
        or tree != runner.MODEL_TREE
        or origin != runner.MODEL_GIT_ORIGIN
    ):
        raise ValueError("live HiFi-IFDL source identity changed")
    status_text = _git_value(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status_text is None:
        raise ValueError("cannot inspect live HiFi-IFDL source")
    status = [line for line in status_text.splitlines() if line]
    cache_pattern = re.compile(r"^\?\? (?:[^/]+/)*__pycache__/[A-Za-z0-9_.-]+\.pyc$")
    if any(cache_pattern.fullmatch(line) is None for line in status):
        raise ValueError("live HiFi-IFDL source inventory drifted")
    files: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha) in runner.SOURCE_BOUND_FILES.items():
        path = root / relative
        _reject_symlink_components(path, f"HiFi-IFDL source {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
            or _git_value(root, "ls-files", "--error-unmatch", relative) != relative
        ):
            raise ValueError(f"HiFi-IFDL source-bound file changed: {relative}")
        files[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
            "git_tracked": True,
        }
    value = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tree": tree,
        "origin": origin,
        "tracked_and_non_cache_untracked_clean": True,
        "untracked_bytecode_caches_ignored": len(status),
        "bytecode_cache_execution": False,
        "loader": "verified_source_with_empty_external_pycache_prefix",
        "source_bound_files": files,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


def _absolute_regular_directory(
    path: Path,
    *,
    expected_path: Path,
    label: str,
) -> Path:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, label)
    if (
        not path.is_absolute()
        or candidate != path
        or path.resolve() != expected_path.resolve()
        or not path.is_dir()
        or path.is_symlink()
    ):
        raise ValueError(f"{label} path changed")
    return path.resolve()


def _asset_record(
    path: Path,
    *,
    expected_path: Path,
    expected_bytes: int,
    expected_sha256: str,
    provider: str,
    label: str,
) -> dict[str, Any]:
    resolved = _absolute_regular_file(
        path,
        expected_path=expected_path,
        label=label,
    )
    if (
        resolved.stat().st_size != expected_bytes
        or sha256_file(resolved) != expected_sha256
    ):
        raise ValueError(f"{label} content changed")
    return {
        "path": str(resolved),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "provider": provider,
    }


def _independent_assets_record(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    root = hifi_root.resolve()
    assets = {
        "initialization_weight": _asset_record(
            root / str(legacy.INITIALIZATION_WEIGHT["path"]),
            expected_path=(
                DEFAULT_HIFI_ROOT / str(legacy.INITIALIZATION_WEIGHT["path"])
            ),
            expected_bytes=int(legacy.INITIALIZATION_WEIGHT["bytes"]),
            expected_sha256=str(legacy.INITIALIZATION_WEIGHT["sha256"]),
            provider="official_author_git_repository",
            label="HiFi-IFDL initialization weight",
        ),
        "feature_extractor": _asset_record(
            hrnet_checkpoint,
            expected_path=DEFAULT_HRNET_CHECKPOINT,
            expected_bytes=int(legacy.CHECKPOINTS["feature_extractor"]["bytes"]),
            expected_sha256=str(legacy.CHECKPOINTS["feature_extractor"]["sha256"]),
            provider="official_author_google_drive",
            label="HiFi-IFDL feature-extractor checkpoint",
        ),
        "hierarchical_localizer_classifier": _asset_record(
            nlc_checkpoint,
            expected_path=DEFAULT_NLC_CHECKPOINT,
            expected_bytes=int(
                legacy.CHECKPOINTS["hierarchical_localizer_classifier"]["bytes"]
            ),
            expected_sha256=str(
                legacy.CHECKPOINTS["hierarchical_localizer_classifier"]["sha256"]
            ),
            provider="official_author_google_drive",
            label="HiFi-IFDL hierarchical-head checkpoint",
        ),
        "center_radius": _asset_record(
            root / str(legacy.CENTER_RADIUS["path"]),
            expected_path=DEFAULT_HIFI_ROOT / str(legacy.CENTER_RADIUS["path"]),
            expected_bytes=int(legacy.CENTER_RADIUS["bytes"]),
            expected_sha256=str(legacy.CENTER_RADIUS["sha256"]),
            provider="official_author_git_repository",
            label="HiFi-IFDL center/radius asset",
        ),
    }
    value = {
        "released_identifier": "750001",
        "bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "assets": assets,
    }
    return {**value, "contract_sha256": _fingerprint(value)}


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
    cuda = device.startswith("cuda:")
    if device != "cpu":
        suffix = device.removeprefix("cuda:")
        if not cuda or not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError(f"{label}.device is not canonical")
    if set(runtime) != base_keys | ({"cuda"} if cuda else set()):
        raise ValueError(f"{label} key set changed")
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
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(f"{label}.{key} changed")
    _require_string(runtime.get("torch_version"), f"{label}.torch_version")
    cuda_version = runtime.get("torch_cuda_version")
    if cuda_version is not None:
        _require_string(cuda_version, f"{label}.torch_cuda_version")
    if cuda:
        record = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        if set(record) != {
            "logical_device_index",
            "device_name",
            "total_memory_bytes",
            "compute_capability",
        }:
            raise ValueError(f"{label}.cuda key set changed")
        if (
            record.get("logical_device_index") != int(device[5:])
            or not isinstance(record.get("device_name"), str)
            or not record["device_name"]
            or _require_nonnegative_int(
                record.get("total_memory_bytes"),
                f"{label}.cuda.total_memory_bytes",
            )
            <= 0
            or not isinstance(record.get("compute_capability"), list)
            or len(record["compute_capability"]) != 2
            or any(
                _require_nonnegative_int(item, f"{label}.cuda.compute_capability") < 0
                for item in record["compute_capability"]
            )
        ):
            raise ValueError(f"{label}.cuda content changed")
    contract = {
        key: nested for key, nested in runtime.items() if key != "contract_sha256"
    }
    if runtime.get("contract_sha256") != _fingerprint(contract):
        raise ValueError(f"{label}.contract_sha256 changed")
    return dict(runtime)


def _validate_record_fingerprint(value: Any, label: str) -> dict[str, Any]:
    record = _require_mapping(value, label)
    digest = _require_sha256(record.get("contract_sha256"), f"{label}.contract_sha256")
    body = {key: nested for key, nested in record.items() if key != "contract_sha256"}
    if _fingerprint(body) != digest:
        raise ValueError(f"{label} fingerprint changed")
    return record


def _validate_checkpoint_model_audits(
    checkpoint_value: Any,
    model_value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _validate_record_fingerprint(
        checkpoint_value,
        "immutable.checkpoint_audit",
    )
    if checkpoint.get("bundle_sha256") != legacy.CHECKPOINT_BUNDLE_SHA256 or set(
        _require_mapping(checkpoint.get("task_components"), "task components")
    ) != set(legacy.CHECKPOINTS):
        raise ValueError("HiFi-IFDL checkpoint audit identity changed")
    for role, expected in runner.CHECKPOINT_AUDIT.items():
        record = _validate_record_fingerprint(
            checkpoint["task_components"].get(role),
            f"checkpoint audit {role}",
        )
        expected_fields = {
            "state_dict_tensors": int(expected["state_keys"]),
            "state_dict_elements": int(expected["state_elements"]),
            "dtype_counts": expected["dtype_counts"],
            "ordered_keys_sha256": expected["ordered_keys_sha256"],
            "tensor_schema_sha256": expected["tensor_schema_sha256"],
            "all_floating_tensors_finite": True,
            "weights_only": True,
            "map_location": "cpu",
            "unsafe_globals": [],
        }
        for key, expected_value in expected_fields.items():
            if record.get(key) != expected_value:
                raise ValueError(f"HiFi-IFDL checkpoint audit {role}.{key} changed")
    _validate_record_fingerprint(
        checkpoint.get("initialization_weight"),
        "initialization-weight audit",
    )
    center_radius = _validate_record_fingerprint(
        checkpoint.get("center_radius"),
        "center/radius audit",
    )
    if (
        center_radius.get("center", {}).get("tensor_sha256")
        != runner.CENTER_TENSOR_SHA256
        or center_radius.get("radius", {}).get("tensor_sha256")
        != runner.RADIUS_TENSOR_SHA256
        or center_radius.get("radius", {}).get("value")
        != float(legacy.CENTER_RADIUS["radius_value"])
    ):
        raise ValueError("HiFi-IFDL center/radius audit changed")

    model = _validate_record_fingerprint(model_value, "immutable.model_audit")
    if (
        model.get("construction_device") != "cpu"
        or model.get("parameter_count") != runner.EXPECTED_MODEL_PARAMETERS
        or model.get("trainable_parameter_count")
        != sum(runner.EXPECTED_TRAINABLE_PARAMETERS.values())
        or model.get("buffer_elements") != runner.EXPECTED_MODEL_BUFFERS
        or model.get("module_count") != runner.EXPECTED_MODEL_MODULES
        or model.get("forward_performed") is not False
    ):
        raise ValueError("HiFi-IFDL strict CPU model audit changed")
    return checkpoint, model


def _validate_provenance(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    adapters = _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    source = _independent_source_record(hifi_root)
    assets = _independent_assets_record(
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    environment = _independent_environment_record()
    if immutable.get("source") != source:
        raise ValueError("recorded/live HiFi-IFDL source evidence differs")
    if immutable.get("assets") != assets:
        raise ValueError("recorded/live HiFi-IFDL asset evidence differs")
    if immutable.get("environment") != environment:
        raise ValueError("recorded/live HiFi-IFDL environment evidence differs")
    checkpoint, model = _validate_checkpoint_model_audits(
        immutable.get("checkpoint_audit"),
        immutable.get("model_audit"),
    )
    if immutable.get("license") != runner.LICENSE_RECORD:
        raise ValueError("HiFi-IFDL license declaration changed")
    if immutable.get("resource_expectation") != runner.RESOURCE_EXPECTATION:
        raise ValueError("HiFi-IFDL resource expectation changed")
    ignore = _validate_record_fingerprint(
        immutable.get("artifact_ignore"),
        "immutable.artifact_ignore",
    )
    if (
        ignore.get("ignored") is not True
        or ignore.get("probe")
        != "outputs/opensource/hifi_ifdl/_contract_probe/embeddings_model_256/sample.npy"
    ):
        raise ValueError("HiFi-IFDL artifact-ignore evidence changed")
    preflight = _require_mapping(
        immutable.get("cpu_preflight"),
        "immutable.cpu_preflight",
    )
    if (
        set(preflight) != {"performed_before_accelerator_configuration", "report"}
        or preflight.get("performed_before_accelerator_configuration") is not True
    ):
        raise ValueError("HiFi-IFDL CPU-preflight ordering changed")
    report = _validate_record_fingerprint(
        preflight.get("report"),
        "immutable.cpu_preflight.report",
    )
    if (
        report.get("schema_version") != runner.CPU_PREFLIGHT_SCHEMA
        or report.get("cuda_initialized_before") is not False
        or report.get("cuda_initialized_after") is not False
        or report.get("environment") != environment
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("adapter_sources") != adapters
        or report.get("artifact_ignore") != ignore
        or report.get("checkpoint_audit") != checkpoint
        or report.get("model_audit") != model
        or report.get("balanced250_forward_performed") is not False
        or report.get("balanced250_score_computed") is not False
    ):
        raise ValueError("HiFi-IFDL recorded CPU preflight changed")
    return {
        "adapter_sources": adapters,
        "source": source,
        "assets": assets,
        "environment": environment,
        "checkpoint_audit": checkpoint,
        "model_audit": model,
        "license": runner.LICENSE_RECORD,
        "commercial_use_clearance_established": False,
        "cpu_preflight_ordering": "before_accelerator_configuration",
    }


def independent_structural_golden(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
    recorded_checkpoint_audit: Mapping[str, Any],
    recorded_model_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat the complete safe CPU strict load without a model forward."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "HiFi-IFDL structural golden started after CUDA initialization"
        )
    checkpoint, model = runner._construct_cpu_model_audit(
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("HiFi-IFDL structural golden initialized CUDA")
    if checkpoint != dict(recorded_checkpoint_audit):
        raise ValueError("HiFi-IFDL checkpoint structural replay changed")
    if model != dict(recorded_model_audit):
        raise ValueError("HiFi-IFDL strict CPU model replay changed")
    return {
        "status": "independent_cpu_structural_golden_passed",
        "kind": "all_assets_checkpoint_schema_and_strict_model_load_no_forward",
        "author_published_numerical_golden": None,
        "author_published_numerical_golden_available": False,
        "reason": (
            "the official HiFi-IFDL release provides checkpoint assets but "
            "no frozen numerical output fixture"
        ),
        "construction_device": "cpu",
        "model_forwards": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "checkpoint_audit_sha256": _fingerprint(checkpoint),
        "model_audit_sha256": _fingerprint(model),
        "executable_numeric_gates": [
            "frozen_smoke_A_B_exact_reproduction",
            "formal_full_selection_exact_fresh_replay",
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
        raise ValueError("HiFi-IFDL canonical Balanced250 release changed")
    selection = _require_mapping(raw.get("selection"), "dataset contract selection")
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
        raise ValueError("HiFi-IFDL dataset contract does not rebuild exactly")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("HiFi-IFDL selected-row hash changed")
    if immutable.get("selected_ids_sha256") != contract.selection.selected_ids_sha256:
        raise ValueError("HiFi-IFDL selected-ID hash changed")
    return release, selected, contract


def _expected_model_contract() -> dict[str, Any]:
    return {
        "name": runner.MODEL_NAME,
        "slug": runner.MODEL_SLUG,
        "architecture": runner.MODEL_ARCHITECTURE,
        "repository": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "source_tree": runner.MODEL_TREE,
        "checkpoint_id": runner.CHECKPOINT_ID,
        "checkpoint_bundle_sha256": legacy.CHECKPOINT_BUNDLE_SHA256,
        "checkpoint_release": dict(legacy.CHECKPOINT_RELEASE),
        "checkpoint_components": {
            role: dict(contract) for role, contract in legacy.CHECKPOINTS.items()
        },
        "center_radius": dict(legacy.CENTER_RADIUS),
        "initialization_weight": dict(legacy.INITIALIZATION_WEIGHT),
        "fine_class_names": list(legacy.FINE_CLASS_NAMES),
        "variant": "official_general_detection_and_localization_750001_not_retrained",
    }


def _expected_preprocess_contract() -> dict[str, Any]:
    return {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "geometry": "direct_stretch_to_256x256",
        "resize": "Pillow_bicubic",
        "aspect_ratio_preserved": False,
        "input_crop": None,
        "input_reencode": False,
        "scale": "uint8_divide_255_float32",
        "normalization_mean_std": None,
        "tensor_layout": "CHW",
        "tensor_dtype": "float32",
        "batch_size": 1,
    }


def _expected_inference_contract() -> dict[str, Any]:
    return {
        "feature_extractor": "HighResolutionNet",
        "head": "NLCDetection",
        "classification_output": ("one_minus_float32_softmax_fine14_authentic_index_0"),
        "official_classification_decision": "argmax_fine14_index_not_equal_to_0",
        "localization_output": "PairwiseDistance_p2_eps1e-6_to_released_center",
        "native_restore": "torch_bilinear_raw_distance_align_corners_false",
        "threshold_after_native_restore": True,
        "auxiliary_sigmoid_mask_is_primary": False,
        "test_time_augmentation": False,
        "ensemble": False,
        "autocast": False,
        "forward_passes_per_image": 1,
        "static_cpu_softmax_sanity": {
            "classes": 14,
            "reference": "numpy_float32_exp_sum_div_from_fine14_logits",
            "comparison": "recorded_device_torch_softmax_float32",
            "simplex_sum_absolute_tolerance": (SIMPLEX_SUM_ABSOLUTE_TOLERANCE),
            "cross_device_absolute_tolerance": (STATIC_SOFTMAX_ABSOLUTE_TOLERANCE),
            "roundoff_basis": (
                "float32 unit roundoff u=eps/2; 14u is approximately "
                "7eps; next binary integer bound is 8eps"
            ),
            "scope": "static_cross_device_sanity_only",
            "recorded_device_smoke_and_fresh_replay_tolerance": 0.0,
            "affects_score_decision_or_artifacts": False,
        },
    }


def _validate_immutable_static(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> None:
    if set(immutable) != EXPECTED_IMMUTABLE_KEYS:
        raise ValueError("HiFi-IFDL immutable key set changed")
    if (
        immutable.get("schema_version") != runner.RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
        or immutable.get("model") != _expected_model_contract()
        or immutable.get("preprocess") != _expected_preprocess_contract()
        or immutable.get("inference") != _expected_inference_contract()
        or immutable.get("score_spec") != _score_spec().as_dict()
        or immutable.get("t2_spec") != runner.T2_SPEC
        or immutable.get("task_scope") != runner.TASK_SCOPE
        or immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT
        or immutable.get("license") != runner.LICENSE_RECORD
        or immutable.get("resource_expectation") != runner.RESOURCE_EXPECTATION
    ):
        raise ValueError("HiFi-IFDL immutable scientific contract changed")
    _validate_runtime(immutable.get("runtime"), label="immutable.runtime")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
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
        raise ValueError("HiFi-IFDL finalized manifest key set changed")
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("HiFi-IFDL manifest is not a complete v2 run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest.fingerprint",
    )
    immutable = _require_mapping(manifest.get("immutable"), "manifest.immutable")
    if _fingerprint(immutable) != fingerprint:
        raise ValueError("HiFi-IFDL manifest fingerprint does not bind immutable")
    _validate_immutable_static(
        immutable,
        repo_root=repo_root,
        run_id=run_id,
        expected_mode=expected_mode,
    )
    disk = _require_mapping(manifest.get("disk_preflight"), "manifest.disk_preflight")
    if set(disk) != {
        "free_bytes_before_inference",
        "conservative_pending_bytes_plus_reserve",
        "fixed_reserve_bytes",
    }:
        raise ValueError("HiFi-IFDL disk-preflight key set changed")
    for key, value in disk.items():
        _require_nonnegative_int(value, f"manifest.disk_preflight.{key}")
    if disk["fixed_reserve_bytes"] not in (0, runner.MIN_DISK_RESERVE_BYTES):
        raise ValueError("HiFi-IFDL disk reserve changed")
    if (disk["conservative_pending_bytes_plus_reserve"] == 0) is not (
        disk["fixed_reserve_bytes"] == 0
    ):
        raise ValueError("HiFi-IFDL disk reserve accounting is inconsistent")
    execution = _require_mapping(manifest.get("execution"), "manifest.execution")
    if set(execution) != EXPECTED_EXECUTION_KEYS:
        raise ValueError("HiFi-IFDL execution key set changed")
    for key in EXPECTED_EXECUTION_KEYS:
        _require_nonnegative_int(execution.get(key), f"manifest.execution.{key}")
    if execution["physical_result_rows"] < execution["latest_result_rows"]:
        raise ValueError("HiFi-IFDL execution row accounting is impossible")
    _reject_nonfinite(manifest, "manifest")
    return fingerprint, immutable


def _independent_preprocess(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    import imageio.v2 as imageio

    decoded = np.asarray(imageio.imread(path))
    if decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.uint8:
        raise ValueError("HiFi-IFDL independent RGB decode changed")
    height, width = decoded.shape[:2]
    resized = np.asarray(
        Image.fromarray(decoded).resize(
            (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE),
            resample=Image.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    tensor = np.ascontiguousarray(
        resized.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    )
    if (
        tensor.shape != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise ValueError("HiFi-IFDL independent preprocessing changed")
    audit = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [width, height],
        "model_size": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_sha256": _array_sha256(tensor),
    }
    return tensor, (width, height), audit


def _stable_softmax_float32(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32)
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted, dtype=np.float32)
    return np.ascontiguousarray(
        exponentials / np.sum(exponentials, dtype=np.float32),
        dtype=np.float32,
    )


def _score_projection(row: Mapping[str, Any], *, sample_id: str) -> dict[str, Any]:
    hierarchy = _require_mapping(
        row.get("classification_hierarchy"),
        f"{sample_id}.classification_hierarchy",
    )
    if tuple(hierarchy) != tuple(name for name, _ in runner.HIERARCHY_SPECS):
        raise ValueError(f"{sample_id} hierarchy order/set changed")
    logits_by_name: dict[str, np.ndarray] = {}
    for name, classes in runner.HIERARCHY_SPECS:
        record = _require_mapping(hierarchy.get(name), f"{sample_id}.{name}")
        if set(record) != {"values", "shape", "dtype", "array_sha256"}:
            raise ValueError(f"{sample_id}.{name} metadata changed")
        try:
            logits = np.ascontiguousarray(record.get("values"), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{sample_id}.{name} values are not numeric") from error
        if (
            logits.shape != (classes,)
            or not np.isfinite(logits).all()
            or record.get("shape") != [classes]
            or record.get("dtype") != "float32"
            or record.get("array_sha256") != _array_sha256(logits)
        ):
            raise ValueError(f"{sample_id}.{name} payload changed")
        logits_by_name[name] = logits
    try:
        probabilities = np.ascontiguousarray(
            row.get("fine_probabilities"),
            dtype=np.float32,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sample_id} fine probabilities are not numeric") from error
    expected_probabilities = _stable_softmax_float32(
        logits_by_name["out3_fine_14class"]
    )
    if (
        probabilities.shape != (14,)
        or not np.isfinite(probabilities).all()
        or float(probabilities.min()) < 0.0
        or float(probabilities.max()) > 1.0
        or not math.isclose(
            float(probabilities.sum(dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=SIMPLEX_SUM_ABSOLUTE_TOLERANCE,
        )
        or row.get("fine_probabilities_shape") != [14]
        or row.get("fine_probabilities_dtype") != "float32"
        or row.get("fine_probabilities_array_sha256") != _array_sha256(probabilities)
        or not np.allclose(
            probabilities,
            expected_probabilities,
            rtol=0.0,
            atol=STATIC_SOFTMAX_ABSOLUTE_TOLERANCE,
        )
    ):
        raise ValueError(f"{sample_id} fine softmax payload changed")
    score = float(np.float32(1.0) - probabilities[0])
    fine_index = int(np.argmax(probabilities))
    benchmark = score > CLASSIFICATION_THRESHOLD
    official = fine_index != 0
    expected_scalars = {
        "ai_score": score,
        "score_semantics": "one_minus_softmax_probability_fine_class_0_authentic",
        "classification_decision": "forged" if benchmark else "authentic",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": ">",
        "official_fine_class_index": fine_index,
        "official_fine_class_name": legacy.FINE_CLASS_NAMES[fine_index],
        "official_binary_decision": official,
        "official_decision": "forged" if official else "authentic",
        "official_decision_rule": "argmax_fine_14class_index_not_equal_to_0",
    }
    for key, expected in expected_scalars.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id}.{key} changed")
    return {
        "hierarchy_logits": logits_by_name,
        "fine_probabilities": probabilities,
        "ai_score": score,
        "benchmark_binary_decision": benchmark,
        "official_fine_class_index": fine_index,
        "official_binary_decision": official,
    }


def _validate_auxiliary_mask(value: Any, *, sample_id: str) -> None:
    record = _require_mapping(value, f"{sample_id}.auxiliary_learned_mask")
    if set(record) != {
        "shape",
        "dtype",
        "minimum",
        "maximum",
        "mean",
        "primary_output",
        "reason",
    }:
        raise ValueError(f"{sample_id} auxiliary-mask key set changed")
    minimum = _require_finite(record.get("minimum"), f"{sample_id}.aux.minimum")
    maximum = _require_finite(record.get("maximum"), f"{sample_id}.aux.maximum")
    mean = _require_finite(record.get("mean"), f"{sample_id}.aux.mean")
    if (
        record.get("shape") != [256, 256]
        or record.get("dtype") != "float32"
        or not 0.0 <= minimum <= mean <= maximum <= 1.0
        or record.get("primary_output") is not False
        or record.get("reason")
        != (
            "the official public localize API ignores this sigmoid mask "
            "and thresholds hypersphere distance instead"
        )
    ):
        raise ValueError(f"{sample_id} auxiliary-mask contract changed")


def _pairwise_distance_float32(
    embedding: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    features = np.asarray(embedding)
    center_array = np.asarray(center)
    if (
        features.dtype != np.float32
        or features.shape != (legacy.EMBEDDING_CHANNELS, 256, 256)
        or center_array.dtype != np.float32
        or center_array.shape not in {(legacy.EMBEDDING_CHANNELS,), (1, 18)}
        or not np.isfinite(features).all()
        or not np.isfinite(center_array).all()
    ):
        raise ValueError("HiFi-IFDL embedding/center contract changed")
    vectors = np.ascontiguousarray(features.transpose(1, 2, 0).reshape(-1, 18))
    difference = (
        vectors - center_array.reshape(1, 18) + np.float32(PAIRWISE_DISTANCE_EPSILON)
    )
    squared = np.multiply(difference, difference, dtype=np.float32)
    distance = np.sqrt(
        np.sum(squared, axis=1, dtype=np.float32),
        dtype=np.float32,
    )
    return np.ascontiguousarray(distance.reshape(256, 256), dtype=np.float32)


def _bilinear_align_corners_false(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(score_map, dtype=np.float32)
    if (
        source.ndim != 2
        or source.size == 0
        or not np.isfinite(source).all()
        or width <= 0
        or height <= 0
    ):
        raise ValueError("HiFi-IFDL native restore input changed")
    source_height, source_width = source.shape
    if (height, width) == source.shape:
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
    horizontal = source[:, x0] * (np.float32(1.0) - wx) + source[:, x1] * wx
    restored = horizontal[y0, :] * (np.float32(1.0) - wy) + horizontal[y1, :] * wy
    return np.ascontiguousarray(restored, dtype=np.float32)


def _nearest_resize_mask(
    target: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(target, dtype=bool)
    if source.ndim != 2 or source.size == 0 or width <= 0 or height <= 0:
        raise ValueError("HiFi-IFDL target-resize input changed")
    if source.shape == (height, width):
        return np.ascontiguousarray(source)
    source_height, source_width = source.shape
    y = np.floor(
        (np.arange(height, dtype=np.float64) + 0.5) * source_height / height
    ).astype(np.int64)
    x = np.floor(
        (np.arange(width, dtype=np.float64) + 0.5) * source_width / width
    ).astype(np.int64)
    y = np.minimum(y, source_height - 1)
    x = np.minimum(x, source_width - 1)
    return np.ascontiguousarray(source[y[:, None], x[None, :]])


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _independent_distance_metrics(
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
    ):
        raise ValueError("HiFi-IFDL distance/target contract changed")
    prediction = scores >= np.float32(MASK_THRESHOLD)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    pixel_ap: float | None = None
    if include_ap and truth.any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )
    return {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": ">=",
        "score_semantics": "hifi_hypersphere_euclidean_distance",
        "score_dtype": "float32",
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


def _load_center(hifi_root: Path) -> np.ndarray:
    import torch

    path = hifi_root / str(legacy.CENTER_RADIUS["path"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or list(payload) != ["center", "radius"]:
        raise ValueError("HiFi-IFDL center/radius payload changed")
    center = payload["center"]
    radius = payload["radius"]
    if (
        not isinstance(center, torch.Tensor)
        or center.dtype != torch.float32
        or list(center.shape) != [18]
        or _array_sha256(center.numpy()) != runner.CENTER_TENSOR_SHA256
        or not isinstance(radius, torch.Tensor)
        or radius.dtype != torch.float32
        or list(radius.shape) != []
        or float(radius.item()) != float(legacy.CENTER_RADIUS["radius_value"])
        or _array_sha256(radius.numpy()) != runner.RADIUS_TENSOR_SHA256
    ):
        raise ValueError("HiFi-IFDL released center/radius tensor changed")
    return np.ascontiguousarray(center.numpy(), dtype=np.float32)


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
    allowed = set(identity) | (
        set(runner._OK_ONLY_KEYS) if status == "ok" else set(runner._ERROR_ONLY_KEYS)
    )
    if set(row) != allowed:
        raise ValueError(f"{sample_id} result key set changed")
    for key, value in identity.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id} identity field {key} changed")
    _require_string(row.get("completed_at"), f"{sample_id}.completed_at")
    if status == "error":
        for field in ("error_type", "error", "traceback"):
            _require_string(row.get(field), f"{sample_id}.{field}")
        _reject_nonfinite(row, f"result.{sample_id}")
        return
    input_path = _safe_repo_file(
        expected.get("canonical_path"),
        repo_root=repo_root,
        expected_path=repo_root / str(expected["canonical_path"]),
        label=f"{sample_id} canonical input",
    )
    _, native_size, preprocess = _independent_preprocess(input_path)
    if native_size != (int(expected["width"]), int(expected["height"])):
        raise ValueError(f"{sample_id} native dimensions changed")
    if row.get("preprocess") != preprocess:
        raise ValueError(f"{sample_id} independent preprocessing changed")
    _score_projection(row, sample_id=sample_id)
    _validate_auxiliary_mask(row.get("auxiliary_learned_mask"), sample_id=sample_id)
    if (
        row.get("mask_threshold") != MASK_THRESHOLD
        or row.get("mask_threshold_operator") != ">="
    ):
        raise ValueError(f"{sample_id} localization threshold changed")
    latency = _require_finite(row.get("latency_ms"), f"{sample_id}.latency_ms")
    if latency < 0.0:
        raise ValueError(f"{sample_id} latency is negative")
    _require_nonnegative_int(
        row.get("peak_cuda_memory_bytes"),
        f"{sample_id}.peak_cuda_memory_bytes",
    )
    _reject_nonfinite(row, f"result.{sample_id}")


def _load_npy_record(
    record_value: Any,
    *,
    repo_root: Path,
    expected_path: Path,
    expected_shape: tuple[int, ...],
    expected_semantics: str,
    nonnegative: bool,
    label: str,
) -> tuple[Path, str, str, np.ndarray]:
    record = _require_mapping(record_value, label)
    if set(record) != {
        "path",
        "sha256",
        "bytes",
        "shape",
        "dtype",
        "array_sha256",
        "semantics",
    }:
        raise ValueError(f"{label} metadata key set changed")
    path = _safe_repo_file(
        record.get("path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=label,
    )
    file_sha = _require_sha256(record.get("sha256"), f"{label}.sha256")
    array_sha = _require_sha256(
        record.get("array_sha256"),
        f"{label}.array_sha256",
    )
    expected_bytes = (
        int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        + runner.NPY_HEADER_BYTES
    )
    if (
        sha256_file(path) != file_sha
        or record.get("bytes") != expected_bytes
        or path.stat().st_size != expected_bytes
        or record.get("shape") != list(expected_shape)
        or record.get("dtype") != "float32"
        or record.get("semantics") != expected_semantics
    ):
        raise ValueError(f"{label} metadata/content changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label}") from error
    if (
        not isinstance(array, np.ndarray)
        or array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or (nonnegative and float(array.min()) < 0.0)
        or _array_sha256(array) != array_sha
    ):
        raise ValueError(f"{label} array changed")
    return path, file_sha, array_sha, array


def _load_mask(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_path: Path,
    native_distance: np.ndarray,
    sample_id: str,
) -> tuple[Path, str, str, np.ndarray]:
    path = _safe_repo_file(
        row.get("mask_path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=f"{sample_id} native mask",
    )
    file_sha = _require_sha256(
        row.get("mask_sha256"),
        f"{sample_id}.mask_sha256",
    )
    expected_bytes = _require_nonnegative_int(
        row.get("mask_bytes"),
        f"{sample_id}.mask_bytes",
    )
    if (
        sha256_file(path) != file_sha
        or expected_bytes <= 0
        or path.stat().st_size != expected_bytes
        or row.get("mask_shape") != list(native_distance.shape)
        or row.get("mask_dtype") != "uint8"
        or row.get("mask_semantics")
        != "raw_hypersphere_distance_greater_than_or_equal_to_2_3"
    ):
        raise ValueError(f"{sample_id} native-mask metadata changed")
    try:
        with Image.open(path) as opened:
            if opened.mode != "L" or opened.size != (
                native_distance.shape[1],
                native_distance.shape[0],
            ):
                raise ValueError(f"{sample_id} native-mask image contract changed")
            pixels = np.ascontiguousarray(np.asarray(opened, dtype=np.uint8))
    except OSError as error:
        raise ValueError(f"cannot load {sample_id} native mask") from error
    if pixels.shape != native_distance.shape or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} native-mask pixels changed")
    expected = np.where(
        native_distance >= np.float32(MASK_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)
    if not np.array_equal(pixels, expected):
        raise ValueError(f"{sample_id} mask is not native distance >= 2.3")
    return path, file_sha, _array_sha256(pixels), pixels


def _validate_fullframe_artifact_absence(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    expected = runner._not_applicable_artifact_fields()
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id} full-frame artifact field {key} changed")
    if row.get("localization") is not None:
        raise ValueError(f"{sample_id} full-frame localization must be N/A")
    if row.get("t2_applicable") is not False:
        raise ValueError(f"{sample_id} full-frame T2 applicability changed")


def _validate_artifact_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    center: np.ndarray,
) -> DenseArtifacts | None:
    sample_id = str(expected["sample_id"])
    applicable, semantics = runner._t2_semantics(expected)
    if row.get("t2_applicable") is not applicable:
        raise ValueError(f"{sample_id} T2 applicability changed")
    if row.get("t2_target_semantics") != semantics:
        raise ValueError(f"{sample_id} T2 target semantics changed")
    if not applicable:
        _validate_fullframe_artifact_absence(row, sample_id=sample_id)
        return None
    paths = runner.artifact_paths(artifact_root, sample_id)
    embedding_path, embedding_file_sha, embedding_array_sha, embedding = (
        _load_npy_record(
            row.get("embedding_artifact"),
            repo_root=repo_root,
            expected_path=paths["embedding_model_256"],
            expected_shape=(legacy.EMBEDDING_CHANNELS, 256, 256),
            expected_semantics="official_18d_pixel_embedding",
            nonnegative=False,
            label=f"{sample_id} embedding",
        )
    )
    model_path, model_file_sha, model_array_sha, model_distance = _load_npy_record(
        row.get("distance_model_artifact"),
        repo_root=repo_root,
        expected_path=paths["distance_model_256"],
        expected_shape=(256, 256),
        expected_semantics=(
            "raw_pairwise_distance_to_released_authentic_center_model_256"
        ),
        nonnegative=True,
        label=f"{sample_id} model distance",
    )
    native_shape = (int(expected["height"]), int(expected["width"]))
    native_path, native_file_sha, native_array_sha, native_distance = _load_npy_record(
        row.get("distance_native_artifact"),
        repo_root=repo_root,
        expected_path=paths["distance_native"],
        expected_shape=native_shape,
        expected_semantics=("raw_distance_bilinear_align_corners_false_native_restore"),
        nonnegative=True,
        label=f"{sample_id} native distance",
    )
    artifact_paths = _require_mapping(
        row.get("artifact_paths"),
        f"{sample_id}.artifact_paths",
    )
    if artifact_paths != {
        "embedding_model_256": row["embedding_artifact"]["path"],
        "distance_model_256": row["distance_model_artifact"]["path"],
        "distance_native": row["distance_native_artifact"]["path"],
        "native_mask": row.get("mask_path"),
    }:
        raise ValueError(f"{sample_id} artifact path aliases changed")
    native_record = row["distance_native_artifact"]
    aliases = {
        "score_map_path": native_record["path"],
        "score_map_sha256": native_record["sha256"],
        "score_map_bytes": native_record["bytes"],
        "score_map_shape": native_record["shape"],
        "score_map_dtype": native_record["dtype"],
        "score_map_array_sha256": native_record["array_sha256"],
        "score_map_semantics": native_record["semantics"],
        "dense_output_disposition": ("saved_and_scored_with_applicable_ground_truth"),
    }
    for key, value in aliases.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id} artifact alias {key} changed")
    replayed_model = _pairwise_distance_float32(embedding, center)
    model_diff = float(np.max(np.abs(model_distance - replayed_model)))
    if model_diff > DISTANCE_ABSOLUTE_TOLERANCE:
        raise ValueError(f"{sample_id} model distance/embedding replay changed")
    replayed_native = _bilinear_align_corners_false(
        model_distance,
        width=native_shape[1],
        height=native_shape[0],
    )
    native_diff = float(np.max(np.abs(native_distance - replayed_native)))
    if native_diff > NATIVE_RESTORE_ABSOLUTE_TOLERANCE:
        raise ValueError(f"{sample_id} native bilinear restore changed")
    mask_path, mask_file_sha, mask_array_sha, _ = _load_mask(
        row,
        repo_root=repo_root,
        expected_path=paths["native_mask"],
        native_distance=native_distance,
        sample_id=sample_id,
    )
    target_native = load_ground_truth(expected, repo_root)
    if target_native is None or target_native.shape != native_shape:
        raise ValueError(f"{sample_id} T2 ground truth changed")
    target_model = _nearest_resize_mask(target_native, width=256, height=256)
    include_ap = str(expected["condition"]) != "real"
    expected_localization = {
        "model_256": _independent_distance_metrics(
            model_distance,
            target_model,
            include_ap=include_ap,
        ),
        "native": _independent_distance_metrics(
            native_distance,
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
        embedding_path=embedding_path,
        embedding_file_sha256=embedding_file_sha,
        embedding_array_sha256=embedding_array_sha,
        model_distance_path=model_path,
        model_distance_file_sha256=model_file_sha,
        model_distance_array_sha256=model_array_sha,
        native_distance_path=native_path,
        native_distance_file_sha256=native_file_sha,
        native_distance_array_sha256=native_array_sha,
        mask_path=mask_path,
        mask_file_sha256=mask_file_sha,
        mask_array_sha256=mask_array_sha,
        width=native_shape[1],
        height=native_shape[0],
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
    center: np.ndarray,
) -> dict[str, DenseArtifacts]:
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe HiFi-IFDL artifact root: {artifact_root}"
        )
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != set(EXPECTED_ARTIFACT_INVENTORY) or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("HiFi-IFDL artifact-root inventory changed")
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
        raise ValueError("HiFi-IFDL artifact/result coverage changed")
    applicable = {
        sample_id
        for sample_id, row in expected_by_id.items()
        if runner._t2_semantics(row)[0]
    }
    expected_npy = {f"{sample_id}.npy" for sample_id in applicable}
    expected_names = {
        "embeddings_model_256": expected_npy,
        "distance_maps_model_256": expected_npy,
        "distance_maps_native": expected_npy,
        "masks_native": {f"{sample_id}.png" for sample_id in applicable},
    }
    for directory_name, names in expected_names.items():
        _exact_directory_inventory(
            artifact_root / directory_name,
            expected_names=names,
            label=f"HiFi-IFDL {directory_name} directory",
        )
    artifacts: dict[str, DenseArtifacts] = {}
    applicable_rows: list[Mapping[str, Any]] = []
    for expected in selected:
        sample_id = str(expected["sample_id"])
        if runner._t2_semantics(expected)[0]:
            applicable_rows.append(expected)
        else:
            artifact = _validate_artifact_row(
                result_by_id[sample_id],
                expected=expected,
                repo_root=repo_root,
                artifact_root=artifact_root,
                center=center,
            )
            if artifact is not None:
                raise ValueError(f"{sample_id} invented a full-frame artifact")
    workers = min(ARTIFACT_AUDIT_WORKERS, max(1, len(applicable_rows)))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="hifi-artifact-audit",
    ) as executor:
        futures = {
            executor.submit(
                _validate_artifact_row,
                result_by_id[str(expected["sample_id"])],
                expected=expected,
                repo_root=repo_root,
                artifact_root=artifact_root,
                center=center,
            ): str(expected["sample_id"])
            for expected in applicable_rows
        }
        for future in concurrent.futures.as_completed(futures):
            sample_id = futures[future]
            artifact = future.result()
            if artifact is None or artifact.sample_id != sample_id:
                raise ValueError(f"{sample_id} applicable artifact disappeared")
            artifacts[sample_id] = artifact
    if set(artifacts) != applicable:
        raise ValueError("HiFi-IFDL validated artifact coverage changed")
    return artifacts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, DenseArtifacts],
) -> str:
    records = []
    for sample_id, artifact in sorted(artifacts.items()):
        records.append(
            {
                "sample_id": sample_id,
                "embedding_path": artifact.embedding_path.as_posix(),
                "embedding_file_sha256": artifact.embedding_file_sha256,
                "embedding_array_sha256": artifact.embedding_array_sha256,
                "model_distance_path": artifact.model_distance_path.as_posix(),
                "model_distance_file_sha256": (artifact.model_distance_file_sha256),
                "model_distance_array_sha256": (artifact.model_distance_array_sha256),
                "native_distance_path": artifact.native_distance_path.as_posix(),
                "native_distance_file_sha256": (artifact.native_distance_file_sha256),
                "native_distance_array_sha256": (artifact.native_distance_array_sha256),
                "mask_path": artifact.mask_path.as_posix(),
                "mask_file_sha256": artifact.mask_file_sha256,
                "mask_array_sha256": artifact.mask_array_sha256,
                "width": artifact.width,
                "height": artifact.height,
            }
        )
    return _fingerprint(records)


def _validate_history(
    selected: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = {str(row["sample_id"]) for row in selected}
    histories: dict[str, list[str]] = {}
    for line_number, attempt in enumerate(attempts, start=1):
        sample_id = attempt.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError(
                f"HiFi-IFDL history row {line_number} has unexpected sample_id"
            )
        status = attempt.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"HiFi-IFDL history row {line_number} has invalid status")
        prior = histories.setdefault(sample_id, [])
        if "ok" in prior:
            raise ValueError("HiFi-IFDL history contains an attempt after success")
        prior.append(str(status))
    return {
        "policy": "zero_or_more_errors_then_at_most_one_terminal_success_per_id",
        "physical_attempts": len(attempts),
        "ids_with_attempts": len(histories),
        "errors": sum(
            status == "error" for statuses in histories.values() for status in statuses
        ),
        "recovered_error_to_ok": sum(
            statuses[-1] == "ok" and "error" in statuses[:-1]
            for statuses in histories.values()
        ),
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
    coverage: Mapping[str, Any],
    history: Mapping[str, Any],
    artifacts: Mapping[str, DenseArtifacts],
) -> None:
    run_id = str(immutable["run_id"])
    expected_inputs = _read_jsonl(expected_path, "HiFi-IFDL expected inputs")
    if expected_inputs != list(selected):
        raise ValueError("HiFi-IFDL expected-input ledger changed")
    outputs = _require_mapping(immutable.get("outputs"), "immutable.outputs")
    expected_output_paths = {
        "results_path": repo_relative(results_path, repo_root),
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "summary_path": repo_relative(summary_path, repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        **{
            f"{name}_dir": repo_relative(artifact_root / name, repo_root)
            for name in runner.ARTIFACT_DIRECTORIES
        },
    }
    if outputs != expected_output_paths:
        raise ValueError("HiFi-IFDL immutable output paths changed")
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    expected_dataset = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
        "t1_applicable_images": len(selected),
        "t2_applicable_images": len(artifacts),
    }
    if dataset != expected_dataset:
        raise ValueError("HiFi-IFDL dataset envelope changed")
    inventory_counts = {name: len(artifacts) for name in runner.ARTIFACT_DIRECTORIES}
    manifest_outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    if set(manifest_outputs) != set(outputs) | {
        "results_sha256",
        "summary_sha256",
        "artifact_inventory",
    }:
        raise ValueError("HiFi-IFDL manifest output key set changed")
    if any(manifest_outputs.get(key) != value for key, value in outputs.items()):
        raise ValueError("HiFi-IFDL manifest output paths changed")
    if (
        manifest_outputs.get("results_sha256") != sha256_file(results_path)
        or manifest_outputs.get("summary_sha256") != sha256_file(summary_path)
        or manifest_outputs.get("artifact_inventory") != inventory_counts
    ):
        raise ValueError("HiFi-IFDL manifest output evidence changed")
    if set(summary) != {
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
    }:
        raise ValueError("HiFi-IFDL runtime-summary key set changed")
    if (
        summary.get("schema_version") != runner.RUNTIME_SUMMARY_SCHEMA
        or summary.get("summary_kind") != "runtime_coverage_and_artifact_inventory_only"
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner") != "analyze_hifi_ifdl_balanced.py"
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != manifest["fingerprint"]
        or summary.get("status") != "complete"
        or summary.get("mode") != immutable["mode"]
        or summary.get("model") != runner.MODEL_NAME
        or summary.get("model_slug") != runner.MODEL_SLUG
        or summary.get("score_spec") != _score_spec().as_dict()
        or summary.get("t2_spec") != runner.T2_SPEC
        or summary.get("dataset_contract") != contract.as_dict()
        or summary.get("coverage") != coverage
        or summary.get("attempt_history") != history
        or summary.get("artifact_inventory") != inventory_counts
    ):
        raise ValueError("HiFi-IFDL runtime summary changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    execution = manifest["execution"]
    if (
        execution["new_successes"] + execution["resume_skips"] != len(selected)
        or execution["physical_result_rows"] != len(physical_results)
        or execution["latest_result_rows"] != len(selected)
        or execution["superseded_attempts"] != len(physical_results) - len(selected)
        or execution["new_errors"] > history["errors"]
    ):
        raise ValueError("HiFi-IFDL execution accounting changed")


def _validate_run_directory(run_dir: Path) -> None:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError(f"missing/unsafe HiFi-IFDL run dir: {run_dir}")
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
                f"HiFi-IFDL run directory contains unsafe entry: {entry.name}"
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
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> RunBundle:
    run_dir = _resolve_run_dir(results_dir, run_id, "HiFi-IFDL run directory")
    artifact_root = _resolve_run_dir(
        artifacts_dir,
        run_id,
        "HiFi-IFDL artifact directory",
    )
    _validate_run_directory(run_dir)
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe HiFi-IFDL artifact directory: {artifact_root}"
        )
    manifest_path = run_dir / "manifest.json"
    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "HiFi-IFDL manifest")
    summary = _load_json(summary_path, "HiFi-IFDL runtime summary")
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
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    physical = tuple(_read_jsonl(results_path, "HiFi-IFDL physical results"))
    history = _validate_history(selected, physical)
    if history != runner._validate_physical_attempt_history(selected, physical):
        raise ValueError("HiFi-IFDL independent/runner history audits disagree")
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    for row in physical:
        sample_id = _require_string(row.get("sample_id"), "result.sample_id")
        expected = expected_by_id.get(sample_id)
        if expected is None:
            raise ValueError(f"unexpected HiFi-IFDL result {sample_id}")
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
    if any(row.get("status") != "ok" for row in latest):
        raise ValueError("HiFi-IFDL latest coverage is not all successful")
    coverage = coverage_object.as_dict()
    center = _load_center(hifi_root)
    artifacts = validate_artifact_inventory(
        latest_results=latest,
        selected=selected,
        repo_root=repo_root,
        artifact_root=artifact_root,
        center=center,
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
        coverage=coverage,
        history=history,
        artifacts=artifacts,
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
        history=history,
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="formal",
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="smoke",
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )


def _distance_to_shared_probability(distance: np.ndarray) -> np.ndarray:
    """Monotone compatibility map for the shared [0,1], >=0.5 reducer.

    ``d / (d + 2.3)`` is strictly monotone for nonnegative finite ``d`` and
    maps the official inclusive ``d >= 2.3`` decision to ``p >= 0.5``.  The
    raw distance remains the scientific artifact and is audited separately.
    """

    raw = np.asarray(distance)
    if (
        raw.dtype != np.float32
        or raw.ndim != 2
        or raw.size == 0
        or not np.isfinite(raw).all()
        or float(raw.min()) < 0.0
    ):
        raise ValueError("HiFi-IFDL shared-metric distance input changed")
    converted = np.ascontiguousarray(
        np.asarray(raw, dtype=np.float64)
        / (np.asarray(raw, dtype=np.float64) + float(MASK_THRESHOLD)),
        dtype=np.float32,
    )
    if (
        not np.isfinite(converted).all()
        or float(converted.min()) < 0.0
        or float(converted.max()) > 1.0
        or not np.array_equal(
            converted >= np.float32(0.5),
            raw >= np.float32(MASK_THRESHOLD),
        )
    ):
        raise ValueError(
            "HiFi-IFDL shared-metric transform changed the official threshold mask"
        )
    return converted


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
            or not runner._t2_semantics(expected)[0]
        ):
            raise ValueError(f"T2 callback requested non-applicable map {sample_id}")
        if sha256_file(artifact.native_distance_path) != (
            artifact.native_distance_file_sha256
        ):
            raise ValueError(f"T2 callback artifact changed for {sample_id}")
        distance = np.load(artifact.native_distance_path, allow_pickle=False)
        if (
            distance.shape != (artifact.height, artifact.width)
            or distance.dtype != np.float32
            or _array_sha256(distance) != artifact.native_distance_array_sha256
        ):
            raise ValueError(f"T2 callback native map changed for {sample_id}")
        return _distance_to_shared_probability(distance)

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
            "HiFi-IFDL formal metrics require iterations=1000 and seed=20260726"
        )
    if (
        bundle.mode != "formal"
        or len(bundle.selected) != FORMAL_IMAGES
        or len(bundle.artifacts) != FORMAL_T2_IMAGES
    ):
        raise ValueError("HiFi-IFDL metrics require formal 1775/1025 coverage")
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
        raise ValueError("HiFi-IFDL shared T1 metrics are incomplete")
    t2 = summarize_balanced250_t2(
        bundle.release.inputs,
        selected_results,
        repo_root=bundle.release.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name=("hifi_raw_distance_monotone_d_over_d_plus_2_3_compatibility"),
        threshold=0.5,
        threshold_operator=">=",
        iterations=iterations,
        seed=seed,
    )
    if (
        t2.get("schema_version") != T2_METRICS_SCHEMA_VERSION
        or t2.get("coverage", {}).get("is_complete") is not True
        or t2.get("coverage", {}).get("native_maps_evaluated") != FORMAL_T2_IMAGES
    ):
        raise ValueError("HiFi-IFDL shared T2 metrics are incomplete")
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
        raise ValueError("HiFi-IFDL T2 full-frame exclusion evidence changed")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "formal_images_t1": FORMAL_IMAGES,
        "formal_images_t2": FORMAL_T2_IMAGES,
        "primary_t1": {
            "source": "fine_14class_head",
            "score": "one_minus_softmax_authentic_class_0",
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": ">",
        },
        "primary_t2": {
            "source": "raw_hypersphere_euclidean_distance",
            "threshold": MASK_THRESHOLD,
            "threshold_operator": ">=",
            "probability": False,
        },
        "shared_t2_compatibility": {
            "required_by_shared_reducer_range": [0.0, 1.0],
            "transform": "p=d/(d+2.3)",
            "strictly_monotone_on_nonnegative_finite_distance": True,
            "raw_threshold_equivalence": "d>=2.3 iff p>=0.5",
            "threshold_mask_checked_bit_exact_per_image": True,
            "raw_artifacts_remain_primary": True,
            "ranking_metrics_invariant_under_exact_monotone_transform": True,
        },
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
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
            raise ValueError(f"HiFi-IFDL evidence changed during audit: {key}")
    release = load_canonical_release(
        bundle.release.repo_root,
        bundle.release.manifest_path,
        verify_files=True,
    )
    if release.manifest_sha256 != bundle.evidence_snapshot["dataset_manifest_sha256"]:
        raise ValueError("Balanced250 release changed during HiFi-IFDL audit")
    for sample_id, artifact in bundle.artifacts.items():
        for path, digest in (
            (artifact.embedding_path, artifact.embedding_file_sha256),
            (artifact.model_distance_path, artifact.model_distance_file_sha256),
            (artifact.native_distance_path, artifact.native_distance_file_sha256),
            (artifact.mask_path, artifact.mask_file_sha256),
        ):
            if sha256_file(path) != digest:
                raise ValueError(
                    f"HiFi-IFDL artifact changed during audit: {sample_id}"
                )
    if (
        _artifact_inventory_sha256(bundle.artifacts)
        != bundle.evidence_snapshot["artifact_inventory_sha256"]
    ):
        raise ValueError("HiFi-IFDL artifact inventory changed during audit")


def _smoke_result_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        key: value
        for key, value in row.items()
        if key not in _RUN_SPECIFIC_RESULT_FIELDS
    }
    for key in (
        "embedding_artifact",
        "distance_model_artifact",
        "distance_native_artifact",
    ):
        record = row.get(key)
        if isinstance(record, Mapping):
            projection[key] = {
                nested_key: nested
                for nested_key, nested in record.items()
                if nested_key != "path"
            }
        else:
            projection[key] = record
    return projection


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in immutable.items()
        if key not in {"run_id", "outputs"}
    }


def compare_computational_results(
    first: RunBundle,
    second: RunBundle,
) -> dict[str, Any]:
    if (
        first.mode != "smoke"
        or second.mode != "smoke"
        or len(first.selected) != SMOKE_IMAGES
        or len(second.selected) != SMOKE_IMAGES
        or tuple(row["sample_id"] for row in first.selected)
        != tuple(row["sample_id"] for row in second.selected)
    ):
        raise ValueError("HiFi-IFDL smoke bundles are not the frozen 5x7 pair")
    if _smoke_immutable_projection(first.immutable) != _smoke_immutable_projection(
        second.immutable
    ):
        raise ValueError("HiFi-IFDL smoke immutable contracts differ")
    max_embedding = 0.0
    max_model_distance = 0.0
    max_native_distance = 0.0
    artifact_rows = 0
    for first_row, second_row in zip(
        first.latest_results,
        second.latest_results,
        strict=True,
    ):
        sample_id = str(first_row["sample_id"])
        if second_row.get("sample_id") != sample_id:
            raise ValueError("HiFi-IFDL smoke result order changed")
        if stable_json(_smoke_result_projection(first_row)) != stable_json(
            _smoke_result_projection(second_row)
        ):
            raise ValueError(f"HiFi-IFDL smoke result differs for {sample_id}")
        first_artifact = first.artifacts.get(sample_id)
        second_artifact = second.artifacts.get(sample_id)
        if (first_artifact is None) is not (second_artifact is None):
            raise ValueError(f"HiFi-IFDL smoke T2 scope differs for {sample_id}")
        if first_artifact is None:
            continue
        assert second_artifact is not None
        artifact_rows += 1
        arrays = (
            (
                first_artifact.embedding_path,
                second_artifact.embedding_path,
                "embedding",
            ),
            (
                first_artifact.model_distance_path,
                second_artifact.model_distance_path,
                "model_distance",
            ),
            (
                first_artifact.native_distance_path,
                second_artifact.native_distance_path,
                "native_distance",
            ),
        )
        for first_path, second_path, kind in arrays:
            first_array = np.load(first_path, allow_pickle=False)
            second_array = np.load(second_path, allow_pickle=False)
            difference = float(
                np.max(np.abs(first_array.astype(np.float64) - second_array))
            )
            if not np.array_equal(first_array, second_array):
                raise ValueError(f"HiFi-IFDL smoke {kind} differs for {sample_id}")
            if kind == "embedding":
                max_embedding = max(max_embedding, difference)
            elif kind == "model_distance":
                max_model_distance = max(max_model_distance, difference)
            else:
                max_native_distance = max(max_native_distance, difference)
        if (
            first_artifact.mask_file_sha256 != second_artifact.mask_file_sha256
            or first_artifact.mask_array_sha256 != second_artifact.mask_array_sha256
        ):
            raise ValueError(f"HiFi-IFDL smoke mask differs for {sample_id}")
    if artifact_rows != SMOKE_T2_IMAGES:
        raise ValueError("HiFi-IFDL smoke artifact count changed")
    _verify_bundle_unchanged(first)
    _verify_bundle_unchanged(second)
    return {
        "exact": True,
        "images_compared": SMOKE_IMAGES,
        "t2_artifact_sets_compared": artifact_rows,
        "result_computational_projection_exact": True,
        "immutable_projection_exact": True,
        "classification_hierarchy_exact": True,
        "fine_probabilities_exact": True,
        "ai_scores_exact": True,
        "embeddings_exact": True,
        "model_distance_maps_exact": True,
        "native_distance_maps_exact": True,
        "native_masks_exact": True,
        "max_abs_embedding_difference": max_embedding,
        "max_abs_model_distance_difference": max_model_distance,
        "max_abs_native_distance_difference": max_native_distance,
    }


def _configure_recorded_runtime(bundle: RunBundle) -> tuple[Any, dict[str, Any]]:
    recorded = _validate_runtime(
        bundle.immutable.get("runtime"),
        label="recorded replay runtime",
    )
    device, live = runner.configure_runtime(str(recorded["device"]))
    if live != recorded:
        raise ValueError("HiFi-IFDL fresh-replay runtime differs from recorded runtime")
    return device, live


def replay_model(
    bundle: RunBundle,
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    """Reload once and exactly replay every frozen formal forward."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("HiFi-IFDL fresh replay requires formal 1775 coverage")
    import torch

    device, runtime = _configure_recorded_runtime(bundle)
    models = None
    center = None
    radius = None
    max_head_diff = 0.0
    max_probability_diff = 0.0
    max_embedding_diff = 0.0
    max_model_distance_diff = 0.0
    max_native_distance_diff = 0.0
    applicable_replayed = 0
    try:
        models, center, radius, loaded_device = legacy.load_model(
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("HiFi-IFDL fresh model loaded on the wrong device")
        for index, (input_row, result_row) in enumerate(
            zip(bundle.selected, bundle.latest_results, strict=True),
            start=1,
        ):
            sample_id = str(input_row["sample_id"])
            input_path = _safe_repo_file(
                input_row.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                expected_path=(
                    bundle.release.repo_root / str(input_row["canonical_path"])
                ),
                label=f"{sample_id} fresh-replay input",
            )
            tensor, (width, height), preprocess = _independent_preprocess(input_path)
            if preprocess != result_row.get("preprocess"):
                raise ValueError(f"{sample_id} fresh preprocessing changed")
            processed, _, _ = legacy.infer_one(
                models,
                center,
                loaded_device,
                tensor,
                native_width=width,
                native_height=height,
            )
            hierarchy = _require_mapping(
                result_row.get("classification_hierarchy"),
                f"{sample_id} recorded hierarchy",
            )
            processed_hierarchy = _require_mapping(
                processed.get("hierarchy_logits"),
                f"{sample_id} fresh hierarchy",
            )
            if set(processed_hierarchy) != {name for name, _ in runner.HIERARCHY_SPECS}:
                raise ValueError(f"{sample_id} fresh hierarchy key set changed")
            for name, classes in runner.HIERARCHY_SPECS:
                recorded_logits = np.asarray(
                    hierarchy[name]["values"],
                    dtype=np.float32,
                )
                fresh_logits = np.ascontiguousarray(
                    processed_hierarchy[name],
                    dtype=np.float32,
                )
                if fresh_logits.shape != (classes,):
                    raise ValueError(f"{sample_id} fresh {name} shape changed")
                difference = float(
                    np.max(
                        np.abs(
                            fresh_logits.astype(np.float64)
                            - recorded_logits.astype(np.float64)
                        )
                    )
                )
                max_head_diff = max(max_head_diff, difference)
                if not np.array_equal(fresh_logits, recorded_logits):
                    raise ValueError(f"{sample_id} fresh {name} is not exact")
            recorded_probabilities = np.asarray(
                result_row.get("fine_probabilities"),
                dtype=np.float32,
            )
            fresh_probabilities = np.ascontiguousarray(
                processed.get("fine_probabilities"),
                dtype=np.float32,
            )
            probability_diff = float(
                np.max(
                    np.abs(
                        fresh_probabilities.astype(np.float64)
                        - recorded_probabilities.astype(np.float64)
                    )
                )
            )
            max_probability_diff = max(
                max_probability_diff,
                probability_diff,
            )
            if not np.array_equal(fresh_probabilities, recorded_probabilities):
                raise ValueError(f"{sample_id} fresh fine probabilities are not exact")
            score = float(np.float32(1.0) - fresh_probabilities[0])
            if (
                score != result_row.get("ai_score")
                or float(processed.get("score")) != score
                or bool(processed.get("benchmark_binary_decision"))
                is not (score > CLASSIFICATION_THRESHOLD)
                or int(processed.get("official_fine_class_index"))
                != result_row.get("official_fine_class_index")
                or str(processed.get("official_fine_class_name"))
                != result_row.get("official_fine_class_name")
                or bool(processed.get("official_binary_decision"))
                is not result_row.get("official_binary_decision")
                or processed.get("auxiliary_learned_mask_stats")
                != result_row.get("auxiliary_learned_mask")
            ):
                raise ValueError(f"{sample_id} fresh T1/auxiliary output changed")
            embedding = np.ascontiguousarray(
                processed.get("embedding"),
                dtype=np.float32,
            )
            model_distance = np.ascontiguousarray(
                processed.get("distance_model_256"),
                dtype=np.float32,
            )
            native_distance = np.ascontiguousarray(
                processed.get("distance_native"),
                dtype=np.float32,
            )
            if (
                embedding.shape != (18, 256, 256)
                or model_distance.shape != (256, 256)
                or native_distance.shape != (height, width)
                or not np.isfinite(embedding).all()
                or not np.isfinite(model_distance).all()
                or not np.isfinite(native_distance).all()
            ):
                raise ValueError(f"{sample_id} fresh dense output changed")
            artifact = bundle.artifacts.get(sample_id)
            applicable = runner._t2_semantics(input_row)[0]
            if applicable:
                if artifact is None:
                    raise ValueError(f"{sample_id} fresh applicable artifact missing")
                applicable_replayed += 1
                recorded_arrays = (
                    (
                        embedding,
                        np.load(artifact.embedding_path, allow_pickle=False),
                        "embedding",
                    ),
                    (
                        model_distance,
                        np.load(artifact.model_distance_path, allow_pickle=False),
                        "model distance",
                    ),
                    (
                        native_distance,
                        np.load(artifact.native_distance_path, allow_pickle=False),
                        "native distance",
                    ),
                )
                for fresh, recorded_array, kind in recorded_arrays:
                    difference = float(
                        np.max(
                            np.abs(
                                fresh.astype(np.float64)
                                - recorded_array.astype(np.float64)
                            )
                        )
                    )
                    if kind == "embedding":
                        max_embedding_diff = max(max_embedding_diff, difference)
                    elif kind == "model distance":
                        max_model_distance_diff = max(
                            max_model_distance_diff,
                            difference,
                        )
                    else:
                        max_native_distance_diff = max(
                            max_native_distance_diff,
                            difference,
                        )
                    if not np.array_equal(fresh, recorded_array):
                        raise ValueError(f"{sample_id} fresh {kind} is not bit-exact")
            elif artifact is not None:
                raise ValueError(f"{sample_id} full-frame artifact was invented")
            if index % 100 == 0 or index == FORMAL_IMAGES:
                print(
                    f"[HiFi-IFDL fresh replay {index}/{FORMAL_IMAGES}] {sample_id}",
                    flush=True,
                )
    finally:
        del models
        del center
        del radius
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if applicable_replayed != FORMAL_T2_IMAGES:
        raise ValueError("HiFi-IFDL fresh T2 replay coverage changed")
    _verify_bundle_unchanged(bundle)
    return {
        "status": "passed",
        "mode": "fresh_model_full_selection_exact_replay",
        "images_replayed": FORMAL_IMAGES,
        "t1_heads_replayed": FORMAL_IMAGES,
        "t2_dense_artifact_sets_replayed": FORMAL_T2_IMAGES,
        "fullframe_dense_outputs_checked_shape_finite_but_not_persisted": 750,
        "single_model_instance_for_ordered_selection": True,
        "recorded_runtime_reused_exactly": True,
        "runtime": runtime,
        "classification_hierarchy_exact": True,
        "fine_probabilities_exact": True,
        "ai_scores_exact": True,
        "applicable_embeddings_exact": True,
        "applicable_model_distance_maps_exact": True,
        "applicable_native_distance_maps_exact": True,
        "max_abs_classification_logit_difference": max_head_diff,
        "max_abs_fine_probability_difference": max_probability_diff,
        "max_abs_embedding_difference": max_embedding_diff,
        "max_abs_model_distance_difference": max_model_distance_diff,
        "max_abs_native_distance_difference": max_native_distance_diff,
        "comparison_tolerance": 0.0,
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_verified(path: Path, value: Mapping[str, Any]) -> str:
    _reject_symlink_components(path, f"output {path.name}")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"unsafe output path: {path}")
    atomic_write_json(path, dict(value))
    expected = _json_sha256(value)
    if sha256_file(path) != expected or _load_json(path, f"output {path.name}") != dict(
        value
    ):
        raise ValueError(f"HiFi-IFDL output verification failed: {path}")
    return expected


def _comparison_output_path(
    *,
    repo_root: Path,
    first_run_id: str,
    second_run_id: str,
) -> Path:
    reports = (repo_root / DEFAULT_RESULTS_DIR / "_reports").resolve()
    _reject_symlink_components(reports, "HiFi-IFDL report directory")
    reports.mkdir(parents=True, exist_ok=True)
    return reports / f"{first_run_id}__vs__{second_run_id}_comparison.json"


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    first_run_id: str,
    second_run_id: str,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if first_run_id != DEFAULT_SMOKE_RUN_ID_A or second_run_id != (
        DEFAULT_SMOKE_RUN_ID_B
    ):
        raise ValueError("HiFi-IFDL smoke comparison requires frozen A then B IDs")
    first = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=first_run_id,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    second = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=second_run_id,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    comparison = compare_computational_results(first, second)
    output = (
        _comparison_output_path(
            repo_root=repo_root,
            first_run_id=first_run_id,
            second_run_id=second_run_id,
        )
        if output_path is None
        else output_path.resolve()
    )
    expected_output = _comparison_output_path(
        repo_root=repo_root,
        first_run_id=first_run_id,
        second_run_id=second_run_id,
    )
    if output != expected_output:
        raise ValueError("HiFi-IFDL smoke comparison output path changed")
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "passed",
        "generated_at": utc_now(),
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "first_run_id": first.run_id,
        "second_run_id": second.run_id,
        "selection": {
            "images": SMOKE_IMAGES,
            "t2_applicable_images": SMOKE_T2_IMAGES,
            "per_condition": SMOKE_PER_CONDITION,
            "selected_rows_sha256": SMOKE_SELECTED_ROWS_SHA256,
            "selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
        },
        "first_evidence": dict(first.evidence_snapshot),
        "second_evidence": dict(second.evidence_snapshot),
        "comparison": comparison,
        "gate": "exact_A_B_reproduction_required",
    }
    digest = _write_json_verified(output, report)
    return {
        **report,
        "output_path": repo_relative(output, repo_root),
        "output_sha256": digest,
    }


def _validate_smoke_gate(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> dict[str, Any]:
    first = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=DEFAULT_SMOKE_RUN_ID_A,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    second = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=DEFAULT_SMOKE_RUN_ID_B,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    comparison = compare_computational_results(first, second)
    path = _comparison_output_path(
        repo_root=repo_root,
        first_run_id=DEFAULT_SMOKE_RUN_ID_A,
        second_run_id=DEFAULT_SMOKE_RUN_ID_B,
    )
    report = _load_json(path, "HiFi-IFDL smoke comparison")
    if (
        set(report)
        != {
            "schema_version",
            "status",
            "generated_at",
            "method",
            "model_slug",
            "first_run_id",
            "second_run_id",
            "selection",
            "first_evidence",
            "second_evidence",
            "comparison",
            "gate",
        }
        or report.get("schema_version") != SMOKE_COMPARISON_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("method") != runner.MODEL_NAME
        or report.get("model_slug") != runner.MODEL_SLUG
        or report.get("first_run_id") != DEFAULT_SMOKE_RUN_ID_A
        or report.get("second_run_id") != DEFAULT_SMOKE_RUN_ID_B
        or report.get("selection")
        != {
            "images": SMOKE_IMAGES,
            "t2_applicable_images": SMOKE_T2_IMAGES,
            "per_condition": SMOKE_PER_CONDITION,
            "selected_rows_sha256": SMOKE_SELECTED_ROWS_SHA256,
            "selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
        }
        or report.get("first_evidence") != dict(first.evidence_snapshot)
        or report.get("second_evidence") != dict(second.evidence_snapshot)
        or report.get("comparison") != comparison
        or report.get("gate") != "exact_A_B_reproduction_required"
    ):
        raise ValueError("HiFi-IFDL persisted smoke gate changed")
    _require_string(report.get("generated_at"), "smoke comparison generated_at")
    return {
        "status": "passed",
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "first_run_id": DEFAULT_SMOKE_RUN_ID_A,
        "second_run_id": DEFAULT_SMOKE_RUN_ID_B,
        "comparison": comparison,
    }


def _formal_output_paths(
    bundle: RunBundle,
    *,
    metrics_output: Path | None,
    audit_output: Path | None,
) -> tuple[Path, Path]:
    expected_metrics = (bundle.run_dir / "balanced250_metrics.json").resolve()
    expected_audit = (bundle.run_dir / "independent_audit.json").resolve()
    requested_metrics = (
        expected_metrics if metrics_output is None else metrics_output.resolve()
    )
    requested_audit = expected_audit if audit_output is None else audit_output.resolve()
    if requested_metrics != expected_metrics or requested_audit != expected_audit:
        raise ValueError("HiFi-IFDL formal outputs must use frozen run-local paths")
    if requested_metrics == requested_audit:
        raise ValueError("HiFi-IFDL formal output paths overlap")
    protected = {
        bundle.manifest_path.resolve(),
        bundle.results_path.resolve(),
        bundle.expected_path.resolve(),
        bundle.summary_path.resolve(),
        bundle.release.manifest_path.resolve(),
    }
    if requested_metrics in protected or requested_audit in protected:
        raise ValueError("HiFi-IFDL output overlaps protected evidence")
    for path in (requested_metrics, requested_audit):
        _reject_symlink_components(path, f"HiFi-IFDL output {path.name}")
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"unsafe HiFi-IFDL output path: {path}")
    return requested_metrics, requested_audit


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
    metrics_output: Path | None = None,
    audit_output: Path | None = None,
    fresh_replay: bool = True,
) -> dict[str, Any]:
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError(
            f"HiFi-IFDL formal analysis run-id must be {DEFAULT_FORMAL_RUN_ID}"
        )
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    metrics_path, audit_path = _formal_output_paths(
        bundle,
        metrics_output=metrics_output,
        audit_output=audit_output,
    )
    smoke_gate = _validate_smoke_gate(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    provenance = _validate_provenance(
        bundle.immutable,
        repo_root=repo_root,
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    structural = independent_structural_golden(
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
        recorded_checkpoint_audit=provenance["checkpoint_audit"],
        recorded_model_audit=provenance["model_audit"],
    )
    metrics = {
        **recompute_metrics(bundle),
        "generated_at": utc_now(),
    }
    fresh = (
        replay_model(
            bundle,
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
        )
        if fresh_replay
        else {
            "status": "explicitly_skipped",
            "publishable": False,
            "required_default": True,
            "images_replayed": 0,
        }
    )
    _verify_bundle_unchanged(bundle)
    if _independent_source_record(hifi_root) != bundle.immutable["source"]:
        raise ValueError("HiFi-IFDL source changed during analysis")
    if (
        _independent_assets_record(
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
        )
        != bundle.immutable["assets"]
    ):
        raise ValueError("HiFi-IFDL assets changed during analysis")
    metrics_sha = _write_json_verified(metrics_path, metrics)
    analyzer_path = Path(__file__).resolve()
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if fresh_replay else "diagnostic_only",
        "publishable": fresh_replay,
        "generated_at": utc_now(),
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "contract_checks": {
            "formal_images_t1": FORMAL_IMAGES,
            "formal_images_t2": FORMAL_T2_IMAGES,
            "formal_selected_rows_sha256": FORMAL_SELECTED_ROWS_SHA256,
            "formal_selected_ids_sha256": FORMAL_SELECTED_IDS_SHA256,
            "fine_head_classes": 14,
            "simplex_sum_absolute_tolerance": (SIMPLEX_SUM_ABSOLUTE_TOLERANCE),
            "static_cross_device_softmax_absolute_tolerance": (
                STATIC_SOFTMAX_ABSOLUTE_TOLERANCE
            ),
            "static_softmax_roundoff_basis": (
                "float32 unit roundoff u=eps/2; 14u is approximately "
                "7eps; next binary integer bound is 8eps"
            ),
            "recorded_device_smoke_and_fresh_replay_tolerance": 0.0,
            "t1_score": "one_minus_softmax_fine_class_0_authentic",
            "t1_threshold": CLASSIFICATION_THRESHOLD,
            "t1_threshold_operator": ">",
            "t2_score": "raw_hypersphere_euclidean_distance",
            "t2_threshold": MASK_THRESHOLD,
            "t2_threshold_operator": ">=",
            "fullframe_t2": "not_applicable",
            "fullframe_t2_images_excluded": 750,
            "persisted_t2_artifact_sets": len(bundle.artifacts),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "coverage": dict(bundle.coverage),
        "attempt_history": dict(bundle.history),
        "provenance": provenance,
        "structural_golden": structural,
        "smoke_reproducibility_gate": smoke_gate,
        "fresh_replay": fresh,
        "metrics": {
            "path": repo_relative(metrics_path, repo_root),
            "sha256": metrics_sha,
            "schema_version": METRICS_SCHEMA_VERSION,
            "t1_schema_version": T1_METRICS_SCHEMA_VERSION,
            "t2_schema_version": T2_METRICS_SCHEMA_VERSION,
        },
        "evidence_snapshot": dict(bundle.evidence_snapshot),
        "artifact_audit": {
            "inventory_sha256": bundle.evidence_snapshot["artifact_inventory_sha256"],
            "embedding_model_256": len(bundle.artifacts),
            "distance_map_model_256": len(bundle.artifacts),
            "distance_map_native": len(bundle.artifacts),
            "native_threshold_mask": len(bundle.artifacts),
            "pairwise_distance_absolute_tolerance": (DISTANCE_ABSOLUTE_TOLERANCE),
            "native_restore_absolute_tolerance": (NATIVE_RESTORE_ABSOLUTE_TOLERANCE),
            "persisted_artifact_hashes_exact": True,
        },
        "license": runner.LICENSE_RECORD,
        "commercial_use_clearance_established": False,
        "resource_expectation": runner.RESOURCE_EXPECTATION,
        "analyzer": {
            "path": repo_relative(analyzer_path, repo_root),
            "bytes": analyzer_path.stat().st_size,
            "sha256": sha256_file(analyzer_path),
            "python_executable": str(Path(sys.executable)),
            "numpy": np.__version__,
        },
    }
    audit_sha = _write_json_verified(audit_path, audit)
    return {
        "status": audit["status"],
        "publishable": audit["publishable"],
        "run_id": bundle.run_id,
        "metrics_path": repo_relative(metrics_path, repo_root),
        "metrics_sha256": metrics_sha,
        "audit_path": repo_relative(audit_path, repo_root),
        "audit_sha256": audit_sha,
        "fresh_replay": fresh["status"],
        "smoke_gate": smoke_gate["status"],
    }


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument("--hifi-root", type=Path, default=DEFAULT_HIFI_ROOT)
    parser.add_argument(
        "--hrnet-checkpoint",
        type=Path,
        default=DEFAULT_HRNET_CHECKPOINT,
    )
    parser.add_argument(
        "--nlc-checkpoint",
        type=Path,
        default=DEFAULT_NLC_CHECKPOINT,
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--skip-fresh-replay", action="store_true")
    parser.add_argument("--compare-smoke", action="store_true")
    parser.add_argument(
        "--smoke-run-id-a",
        default=DEFAULT_SMOKE_RUN_ID_A,
    )
    parser.add_argument(
        "--smoke-run-id-b",
        default=DEFAULT_SMOKE_RUN_ID_B,
    )
    parser.add_argument("--comparison-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    results_dir = _safe_standard_root(
        args.results_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="HiFi-IFDL results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="HiFi-IFDL artifacts root",
    )
    hifi_root = _absolute_regular_directory(
        _anchored(args.hifi_root, repo_root),
        expected_path=DEFAULT_HIFI_ROOT,
        label="HiFi-IFDL source root",
    )
    hrnet_checkpoint = _absolute_regular_file(
        _anchored(args.hrnet_checkpoint, repo_root),
        expected_path=DEFAULT_HRNET_CHECKPOINT,
        label="HiFi-IFDL HRNet checkpoint",
    )
    nlc_checkpoint = _absolute_regular_file(
        _anchored(args.nlc_checkpoint, repo_root),
        expected_path=DEFAULT_NLC_CHECKPOINT,
        label="HiFi-IFDL NLC checkpoint",
    )
    if args.compare_smoke:
        if (
            args.run_id != DEFAULT_FORMAL_RUN_ID
            or args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_fresh_replay
        ):
            raise ValueError(
                "smoke comparison accepts no formal output/replay overrides"
            )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            first_run_id=args.smoke_run_id_a,
            second_run_id=args.smoke_run_id_b,
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
            output_path=(
                _anchored(args.comparison_output, repo_root)
                if args.comparison_output is not None
                else None
            ),
        )
    else:
        if (
            args.smoke_run_id_a != DEFAULT_SMOKE_RUN_ID_A
            or args.smoke_run_id_b != DEFAULT_SMOKE_RUN_ID_B
            or args.comparison_output is not None
        ):
            raise ValueError("formal analysis accepts no smoke overrides")
        report = analyze(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            run_id=runner._valid_run_id(args.run_id),
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
            metrics_output=(
                _anchored(args.metrics_output, repo_root)
                if args.metrics_output is not None
                else None
            ),
            audit_output=(
                _anchored(args.audit_output, repo_root)
                if args.audit_output is not None
                else None
            ),
            fresh_replay=not args.skip_fresh_replay,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
