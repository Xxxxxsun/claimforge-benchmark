#!/usr/bin/env python3
"""Fail-closed audit, metrics, smoke comparison, and replay for TruFor.

The TruFor Balanced250 runner persists two native-resolution float32 maps for
every successful image: the forged-probability map and the TCP reliability
map.  Only authentic and local-insertion images are T2-applicable and have a
threshold PNG.  Full-frame edits retain both raw maps as diagnostics but must
never acquire a fabricated T2 target, mask, or localization metric.

This analyzer treats the manifest, append-only JSONL, runtime summary, and all
large raster artifacts as untrusted.  It independently rebuilds the canonical
selection, validates source/assets/runtime/strict-load evidence, reopens every
input, verifies every raw map and applicable mask, recomputes the shared
Balanced250 T1 and T2 metrics, compares the frozen A/B smoke runs byte-for-byte,
and by default reloads the official model for a complete 1,775-image replay.

TruFor does not publish a numerical golden output for this checkpoint.  The
audit therefore reports that fact instead of inventing one.  Its executable
numeric gates are exact A/B smoke reproduction and exact full-selection replay;
the independent CPU strict-load audit is the structural checkpoint golden.
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

from eval.opensource import run_trufor as legacy
from eval.opensource import run_trufor_balanced as runner
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


AUDIT_SCHEMA_VERSION = "trufor_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "trufor_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "trufor_balanced250_summary_v2"
T1_METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
T2_METRICS_SCHEMA_VERSION = "balanced250_t2_summary_v1"

DEFAULT_RESULTS_DIR = Path("results/opensource/trufor")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/trufor")
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
DEFAULT_TRUFOR_ROOT = legacy.DEFAULT_TRUFOR_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_ARCHIVE = runner.DEFAULT_ARCHIVE

FORMAL_IMAGES = 1_775
FORMAL_T2_IMAGES = 1_025
SMOKE_IMAGES = 35
SMOKE_T2_IMAGES = 20
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_726
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
SCORE_ABS_TOLERANCE = 1e-7

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
        "model",
        "preprocess",
        "score_spec",
        "t2_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "source",
        "assets",
        "environment",
        "checkpoint_audit",
        "model_audit",
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
    }
)
EXPECTED_ARTIFACT_INVENTORY = (
    "score_maps_native",
    "reliability_maps_native",
    "masks_native",
)
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
        "score_map_native_path",
        "reliability_map_native_path",
        "mask_path",
        "artifact_paths",
    }
)


@dataclass(frozen=True)
class DenseArtifacts:
    """Validated raw artifacts for one terminal successful attempt."""

    sample_id: str
    score_path: Path
    score_file_sha256: str
    score_array_sha256: str
    reliability_path: Path
    reliability_file_sha256: str
    reliability_array_sha256: str
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
    artifacts: Mapping[str, DenseArtifacts]
    evidence_snapshot: Mapping[str, str]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _stable_result_row(row: Mapping[str, Any]) -> str:
    """Serialize a result row even when the contract exposes it read-only."""

    return stable_json(dict(row))


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = nested
    return value


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
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                row = _require_mapping(
                    value,
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


def _absolute_regular_file(
    value: Any,
    *,
    expected_path: Path,
    label: str,
) -> Path:
    path = Path(_require_string(value, label))
    if not path.is_absolute() or _lexical_absolute(path) != path:
        raise ValueError(f"{label} is not a canonical absolute path")
    _reject_symlink_components(path, label)
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"{label} differs from configured asset path")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/unsafe {label}: {path}")
    return path.resolve()


def _score_spec() -> ScoreSpec:
    value = runner.SCORE_SPEC
    if not isinstance(value, ScoreSpec) or value.as_dict() != {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": ">=",
    }:
        raise ValueError("TruFor runner SCORE_SPEC changed")
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
        raise ValueError("TruFor selection capability changed")
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
            raise ValueError("formal TruFor selection drifted")
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
            raise ValueError("smoke TruFor selection drifted")
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
        raise ValueError("TruFor adapter source order/set changed")
    independently_verified: dict[str, dict[str, Any]] = {}
    for relative in EXPECTED_ADAPTER_SOURCE_PATHS:
        record = _require_mapping(
            sources.get(relative),
            f"adapter source {relative}",
        )
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
    return independently_verified


def _validate_runtime(value: Any, *, label: str) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    base_keys = {
        "device",
        "seed",
        "precision",
        "batch_size",
        "autocast",
        "inference_mode",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
        "float32_matmul_precision",
        "torch_cuda_runtime",
        "cuda_initialized_before_configuration",
        "cuda_initialized_after_configuration",
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
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "deterministic_algorithms_enabled": False,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": None,
        "matmul_allow_tf32": True,
        "float32_matmul_precision": "high",
        "cuda_initialized_before_configuration": False,
        "cuda_initialized_after_configuration": cuda,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(f"{label}.{key} changed")
    cuda_runtime = runtime.get("torch_cuda_runtime")
    if cuda_runtime is not None:
        _require_string(cuda_runtime, f"{label}.torch_cuda_runtime")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    if cudnn != {
        "enabled": False,
        "benchmark": False,
        "deterministic": False,
        "allow_tf32": True,
        "source": "official_trufor_ph3_config",
    }:
        raise ValueError(f"{label}.cudnn contract changed")
    if cuda:
        cuda_record = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
        if set(cuda_record) != {
            "device_index",
            "device_name",
            "total_memory_bytes",
            "capability",
        }:
            raise ValueError(f"{label}.cuda key set changed")
        if cuda_record.get("device_index") != int(device.split(":", 1)[1]):
            raise ValueError(f"{label}.cuda device index changed")
        _require_string(
            cuda_record.get("device_name"),
            f"{label}.cuda.device_name",
        )
        if (
            _require_nonnegative_int(
                cuda_record.get("total_memory_bytes"),
                f"{label}.cuda.total_memory_bytes",
            )
            == 0
        ):
            raise ValueError(f"{label}.cuda total memory is zero")
        capability = cuda_record.get("capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{label}.cuda capability changed")
    _reject_nonfinite(runtime, label)
    return runtime


def _independent_source_record(trufor_root: Path) -> dict[str, Any]:
    root = trufor_root.resolve()
    if root.name != "TruFor_train_test" or not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe TruFor source root: {root}")
    repository = root.parent
    if legacy._git_value(repository, "rev-parse", "HEAD") != (
        legacy.MODEL_SOURCE_COMMIT
    ):
        raise ValueError("live TruFor source commit changed")
    status = legacy._git_value(
        repository,
        "status",
        "--short",
        "--untracked-files=all",
    )
    if status is None or status:
        raise ValueError("live TruFor source tree is not clean")
    files: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha) in {
        **runner.TRUFOR_SOURCE_FILES,
        **runner.SOURCE_BOUND_ASSETS,
    }.items():
        path = root / relative
        _reject_symlink_components(path, f"TruFor source {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"TruFor source-bound file changed: {relative}")
        if (
            legacy._git_value(
                repository,
                "ls-files",
                "--error-unmatch",
                f"TruFor_train_test/{relative}",
            )
            is None
        ):
            raise ValueError(f"TruFor source file is untracked: {relative}")
        files[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
            "git_tracked": True,
        }
    return {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "git_root": str(repository),
        "commit": legacy.MODEL_SOURCE_COMMIT,
        "tracked_and_untracked_clean": True,
        "source_bound_files": files,
    }


def _asset_record(
    path: Path,
    *,
    expected_name: str,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    _reject_symlink_components(path, label)
    if (
        resolved.name != expected_name
        or not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size != expected_bytes
        or sha256_file(resolved) != expected_sha256
    ):
        raise ValueError(f"{label} changed")
    return {
        "path": str(resolved),
        "filename": expected_name,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
    }


def _independent_assets_record(
    *,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    archive = _asset_record(
        archive_path,
        expected_name="TruFor_weights.zip",
        expected_bytes=runner.ARCHIVE_BYTES,
        expected_sha256=legacy.CHECKPOINT_ZIP_SHA256,
        label="TruFor official archive",
    )
    if runner._md5_file(archive_path.resolve()) != legacy.CHECKPOINT_ZIP_MD5:
        raise ValueError("TruFor archive published MD5 changed")
    archive.update(
        {
            "url": legacy.CHECKPOINT_URL,
            "published_md5": legacy.CHECKPOINT_ZIP_MD5,
            "members": ["weights/", "weights/trufor.pth.tar"],
            "inner_checkpoint_uncompressed_bytes": runner.CHECKPOINT_BYTES,
        }
    )
    checkpoint = _asset_record(
        checkpoint_path,
        expected_name="trufor.pth.tar",
        expected_bytes=runner.CHECKPOINT_BYTES,
        expected_sha256=legacy.CHECKPOINT_SHA256,
        label="TruFor checkpoint",
    )
    checkpoint.update(
        {
            "id": runner.CHECKPOINT_ID,
            "epoch": legacy.CHECKPOINT_EPOCH,
            "weights_only": True,
            "strict_model_load": True,
        }
    )
    root = trufor_root.resolve()
    return {
        "archive": archive,
        "checkpoint": checkpoint,
        "configuration": {
            "path": str(root / "lib/config/trufor_ph3.yaml"),
            "bytes": runner.SOURCE_BOUND_ASSETS["lib/config/trufor_ph3.yaml"][0],
            "sha256": legacy.MODEL_CONFIG_SHA256,
            "constructor_pretrained": None,
            "noiseprint_weights": None,
            "constructor_external_weight_files_used": False,
        },
        "licenses": {
            "overall": {
                **runner.LICENSE_RECORD["overall"],
                "absolute_path": str(root / "LICENSE.txt"),
                "bytes": runner.SOURCE_BOUND_ASSETS["LICENSE.txt"][0],
            },
            "cmx_component": {
                **runner.LICENSE_RECORD["cmx_component"],
                "absolute_path": str(root / "LICENSE_CMX.txt"),
                "bytes": runner.SOURCE_BOUND_ASSETS["LICENSE_CMX.txt"][0],
            },
        },
    }


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
    if (
        executable != runner.EXPECTED_PYTHON_EXECUTABLE
        or prefix != runner.EXPECTED_VENV_ROOT
        or platform.python_version() != "3.12.3"
        or not pyvenv.is_file()
        or pyvenv.is_symlink()
        or sha256_file(pyvenv) != runner.EXPECTED_PYVENV_SHA256
        or versions != runner.EXPECTED_PACKAGES
    ):
        raise ValueError("live TruFor analysis environment changed")
    return {
        "python_executable": str(executable),
        "python_prefix": str(prefix),
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "bytes": pyvenv.stat().st_size,
            "sha256": runner.EXPECTED_PYVENV_SHA256,
            "include_system_site_packages": True,
        },
        "packages": versions,
    }


def _validate_checkpoint_model_audits(
    *,
    checkpoint_audit: Any,
    model_audit: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _require_mapping(
        checkpoint_audit,
        "immutable.checkpoint_audit",
    )
    expected_checkpoint = {
        "outer_type": "builtins.dict",
        "outer_keys": list(runner.CHECKPOINT_OUTER_KEYS),
        "epoch": legacy.CHECKPOINT_EPOCH,
        "best_key": "loss",
        "best_value": 0.14553078470670194,
        "state_dict_type": "collections.OrderedDict",
        "state_dict_tensors": runner.CHECKPOINT_STATE_KEYS,
        "state_dict_elements": runner.CHECKPOINT_STATE_ELEMENTS,
        "dtype_counts": {
            "torch.float32": runner.CHECKPOINT_FLOAT32_TENSORS,
            "torch.int64": runner.CHECKPOINT_INT64_TENSORS,
        },
        "ordered_keys_sha256": runner.CHECKPOINT_ORDERED_KEYS_SHA256,
        "tensor_schema_sha256": runner.CHECKPOINT_TENSOR_SCHEMA_SHA256,
        "all_floating_tensors_finite": True,
        "static_unsafe_globals": list(sorted(runner.CHECKPOINT_UNSAFE_GLOBALS)),
        "safe_globals_allowlist": [
            "numpy.core.multiarray.scalar",
            "numpy.dtype",
            "numpy.dtype[float64]_class",
        ],
        "weights_only": True,
        "map_location": "cpu",
    }
    if checkpoint != expected_checkpoint:
        raise ValueError("TruFor checkpoint structural audit changed")
    model = _require_mapping(model_audit, "immutable.model_audit")
    expected_model = {
        "construction_device": "cpu",
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "eval_mode": True,
        "parameters": runner.EXPECTED_MODEL_PARAMETERS,
        "trainable_parameters": runner.EXPECTED_TRAINABLE_PARAMETERS,
        "buffers": runner.EXPECTED_MODEL_BUFFERS,
        "modules": runner.EXPECTED_MODEL_MODULES,
        "constructor_external_weight_files_used": False,
        "model_forwards": 0,
    }
    if model != expected_model:
        raise ValueError("TruFor strict CPU model audit changed")
    return checkpoint, model


def _validate_provenance(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    source = _independent_source_record(trufor_root)
    if immutable.get("source") != source:
        raise ValueError("recorded/live TruFor source evidence differs")
    assets = _independent_assets_record(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    if immutable.get("assets") != assets:
        raise ValueError("recorded/live TruFor asset evidence differs")
    environment = _independent_environment_record()
    if immutable.get("environment") != environment:
        raise ValueError("recorded/live TruFor environment evidence differs")
    checkpoint, model = _validate_checkpoint_model_audits(
        checkpoint_audit=immutable.get("checkpoint_audit"),
        model_audit=immutable.get("model_audit"),
    )
    if immutable.get("license") != runner.LICENSE_RECORD:
        raise ValueError("TruFor license declaration changed")
    preflight_wrapper = _require_mapping(
        immutable.get("cpu_preflight"),
        "immutable.cpu_preflight",
    )
    if (
        set(preflight_wrapper)
        != {
            "performed_before_accelerator_configuration",
            "report",
        }
        or preflight_wrapper.get("performed_before_accelerator_configuration")
        is not True
    ):
        raise ValueError("TruFor CPU preflight ordering changed")
    report = _require_mapping(
        preflight_wrapper.get("report"),
        "immutable.cpu_preflight.report",
    )
    if (
        set(report)
        != {
            "schema_version",
            "status",
            "environment",
            "source",
            "assets",
            "checkpoint_audit",
            "model_audit",
            "license",
            "accelerator_model_forwards",
            "balanced250_model_scores_computed",
            "cuda_initialized_before",
            "cuda_initialized_after",
        }
        or report.get("schema_version") != runner.CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("environment") != environment
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("checkpoint_audit") != checkpoint
        or report.get("model_audit") != model
        or report.get("license") != runner.LICENSE_RECORD
        or report.get("accelerator_model_forwards") != 0
        or report.get("balanced250_model_scores_computed") != 0
        or report.get("cuda_initialized_before") is not False
        or report.get("cuda_initialized_after") is not False
    ):
        raise ValueError("TruFor recorded CPU preflight changed")
    return {
        "adapter_sources": _verify_adapter_sources(
            immutable.get("adapter_sources"),
            repo_root=repo_root,
        ),
        "source": source,
        "assets": assets,
        "environment": environment,
        "checkpoint_audit": checkpoint,
        "model_audit": model,
        "license": runner.LICENSE_RECORD,
        "cpu_preflight_ordering": "before_accelerator_configuration",
    }


def independent_structural_golden(
    *,
    checkpoint_path: Path,
    trufor_root: Path,
    recorded_checkpoint_audit: Mapping[str, Any],
    recorded_model_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat the safe CPU strict load without performing a model forward."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("TruFor structural golden started after CUDA initialization")
    checkpoint, model = runner._build_cpu_model_audit(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("TruFor structural golden initialized CUDA")
    if checkpoint != dict(recorded_checkpoint_audit) or model != dict(
        recorded_model_audit
    ):
        raise ValueError("TruFor independent CPU strict-load replay changed")
    return {
        "status": "independent_cpu_structural_golden_passed",
        "kind": "checkpoint_schema_and_strict_model_load_no_forward",
        "author_published_numerical_golden": None,
        "author_published_numerical_golden_available": False,
        "reason": (
            "official TruFor release provides checkpoint/config but no "
            "frozen numerical output fixture"
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
    manifest_relative = release_binding.get("manifest_path")
    manifest_path = _safe_repo_file(
        manifest_relative,
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
        raise ValueError("TruFor canonical Balanced250 release changed")
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
        raise ValueError("TruFor dataset contract does not rebuild exactly")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("TruFor selected-row SHA-256 changed")
    if immutable.get("selected_ids_sha256") != (contract.selection.selected_ids_sha256):
        raise ValueError("TruFor selected-ID SHA-256 changed")
    return release, selected, contract


def _validate_immutable_static(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    expected_mode: str,
) -> None:
    if set(immutable) != EXPECTED_IMMUTABLE_KEYS:
        raise ValueError("TruFor immutable key set changed")
    if (
        immutable.get("schema_version") != runner.RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
        or immutable.get("score_spec") != _score_spec().as_dict()
        or immutable.get("t2_spec") != runner.T2_SPEC
        or immutable.get("task_scope") != runner.TASK_SCOPE
        or immutable.get("artifact_contract") != runner.ARTIFACT_CONTRACT
    ):
        raise ValueError("TruFor immutable scientific contract changed")
    expected_model = {
        "name": runner.MODEL_NAME,
        "slug": runner.MODEL_SLUG,
        "architecture": runner.MODEL_ARCHITECTURE,
        "repository": legacy.MODEL_REPO_URL,
        "source_commit": legacy.MODEL_SOURCE_COMMIT,
        "checkpoint_id": runner.CHECKPOINT_ID,
        "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
        "checkpoint_bytes": runner.CHECKPOINT_BYTES,
        "positive_class_index": 1,
    }
    if immutable.get("model") != expected_model:
        raise ValueError("TruFor immutable model contract changed")
    expected_preprocess = {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "PIL_convert_RGB",
        "tensor_layout": "CHW",
        "tensor_dtype": "float32",
        "input_scale_divisor": 256.0,
        "input_resize": None,
        "input_crop": None,
        "network_map_upsample": ("bilinear_align_corners_false_to_native_input_size"),
        "batch_size": 1,
        "autocast": False,
    }
    if immutable.get("preprocess") != expected_preprocess:
        raise ValueError("TruFor immutable preprocessing changed")
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
        raise ValueError("TruFor finalized manifest key set changed")
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("TruFor manifest is not a complete v2 run")
    _require_string(manifest.get("started_at"), "manifest.started_at")
    _require_string(manifest.get("completed_at"), "manifest.completed_at")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest.fingerprint",
    )
    immutable = _require_mapping(
        manifest.get("immutable"),
        "manifest.immutable",
    )
    if _fingerprint(immutable) != fingerprint:
        raise ValueError("TruFor manifest fingerprint does not bind immutable")
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
            "estimated_pending_map_bytes_plus_reserve",
            "fixed_reserve_bytes",
        }
        or any(
            _require_nonnegative_int(value, f"disk_preflight.{key}") < 0
            for key, value in disk.items()
        )
        or disk.get("fixed_reserve_bytes") != runner.MIN_DISK_RESERVE_BYTES
    ):
        raise ValueError("TruFor disk preflight evidence changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    if set(execution) != EXPECTED_EXECUTION_KEYS:
        raise ValueError("TruFor execution accounting key set changed")
    for key in EXPECTED_EXECUTION_KEYS:
        _require_nonnegative_int(
            execution.get(key),
            f"manifest.execution.{key}",
        )
    if execution["physical_result_rows"] < execution["latest_result_rows"]:
        raise ValueError("TruFor execution row accounting is impossible")
    _reject_nonfinite(manifest, "manifest")
    return fingerprint, immutable


def _independent_preprocess_tensor(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as opened:
        rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB"), dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("TruFor independent RGB decode changed")
    height, width = rgb.shape[:2]
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1),
        dtype=np.float32,
    )
    tensor /= np.float32(256.0)
    if (
        tensor.shape != (3, height, width)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 255.0 / 256.0
    ):
        raise ValueError("TruFor independent preprocessing changed")
    audit = {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "PIL_convert_RGB",
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(rgb.shape),
        "decoded_rgb_dtype": "uint8",
        "decoded_rgb_array_sha256": _array_sha256(rgb),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_array_sha256": _array_sha256(tensor),
        "input_scale": "float32_divide_by_256",
        "input_scale_divisor": 256.0,
        "input_resize": None,
        "input_crop": None,
        "network_map_upsample": ("bilinear_align_corners_false_to_native_input_size"),
        "post_network_map_restore": None,
    }
    return tensor, audit


def _independent_preprocess(path: Path) -> dict[str, Any]:
    _, audit = _independent_preprocess_tensor(path)
    return audit


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    score = _require_finite(row.get("ai_score"), f"{sample_id}.ai_score")
    logit = _require_finite(
        row.get("raw_detection_logit"),
        f"{sample_id}.raw_detection_logit",
    )
    expected_probability = _stable_sigmoid(logit)
    if (
        not 0.0 <= score <= 1.0
        or not math.isclose(
            score,
            expected_probability,
            rel_tol=0.0,
            abs_tol=SCORE_ABS_TOLERANCE,
        )
        or row.get("raw_outputs") != {"binary_forged_logit": logit}
        or row.get("class_probabilities") != {"real": 1.0 - score, "forged": score}
        or row.get("probability") != score
        or row.get("score") != score
        or row.get("score_margin") != logit
        or row.get("score_semantics") != "sigmoid_binary_logit_probability_of_forged"
        or row.get("calibrated_probability") is not False
        or row.get("classification_decision") is not (score >= CLASSIFICATION_THRESHOLD)
        or row.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator") != ">="
    ):
        raise ValueError(f"{sample_id} T1 score contract changed")


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
    common_keys = set(identity) | {"status", "completed_at"}
    expected_keys = (
        common_keys | set(runner._OK_ONLY_KEYS)
        if status == "ok"
        else common_keys | {"error_type", "error", "traceback"}
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
    if row.get("preprocess") != _independent_preprocess(input_path):
        raise ValueError(f"{sample_id} independent preprocessing changed")
    latency = _require_finite(
        row.get("latency_ms"),
        f"{sample_id}.latency_ms",
    )
    if latency < 0.0:
        raise ValueError(f"{sample_id} latency is negative")
    peak = _require_nonnegative_int(
        row.get("peak_cuda_memory_bytes"),
        f"{sample_id}.peak_cuda_memory_bytes",
    )
    del peak
    if (
        row.get("mask_threshold") != MASK_THRESHOLD
        or row.get("mask_threshold_operator") != ">="
    ):
        raise ValueError(f"{sample_id} mask threshold changed")
    _reject_nonfinite(row, f"result.{sample_id}")


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
    if scores.shape != truth.shape:
        raise ValueError("TruFor score/target dimensions differ")
    if (
        scores.size == 0
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
    ):
        raise ValueError("TruFor score map is invalid")
    prediction = scores >= MASK_THRESHOLD
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


def _load_probability_map(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_path: Path,
    expected_shape: tuple[int, int],
    expected_semantics: str,
    repo_root: Path,
    sample_id: str,
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
    expected_bytes = (
        int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        + runner.NPY_HEADER_BYTES
    )
    if (
        sha256_file(path) != file_sha
        or row.get(f"{prefix}_bytes") != expected_bytes
        or path.stat().st_size != expected_bytes
        or row.get(f"{prefix}_shape") != list(expected_shape)
        or row.get(f"{prefix}_dtype") != "float32"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"{sample_id} {prefix} file metadata changed")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} cannot load {prefix}") from error
    if (
        array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or float(np.min(array)) < 0.0
        or float(np.max(array)) > 1.0
    ):
        raise ValueError(f"{sample_id} {prefix} array contract changed")
    array_sha = _array_sha256(array)
    if row.get(f"{prefix}_array_sha256") != array_sha:
        raise ValueError(f"{sample_id} {prefix} array SHA-256 changed")
    return path, file_sha, array_sha, array


def _load_mask(
    row: Mapping[str, Any],
    *,
    expected_path: Path,
    expected_shape: tuple[int, int],
    repo_root: Path,
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
    if (
        sha256_file(path) != file_sha
        or row.get("mask_bytes") != path.stat().st_size
        or row.get("mask_shape") != list(expected_shape)
        or row.get("mask_dtype") != "uint8"
        or row.get("mask_semantics")
        != "native_probability_map_ge_0_5_encoded_L_0_or_255"
    ):
        raise ValueError(f"{sample_id} native mask metadata changed")
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise ValueError(f"{sample_id} mask encoding changed")
            pixels = np.asarray(opened, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"{sample_id} mask cannot be decoded") from error
    if pixels.shape != expected_shape or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} mask pixels changed")
    array_sha = _array_sha256(pixels)
    if row.get("mask_array_sha256") != array_sha:
        raise ValueError(f"{sample_id} mask array SHA-256 changed")
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
    shape = (height, width)
    paths = runner.artifact_paths(artifact_root, sample_id)
    score_path, score_file_sha, score_array_sha, score_map = _load_probability_map(
        row,
        prefix="score_map_native",
        expected_path=paths["score"],
        expected_shape=shape,
        expected_semantics=("softmax_localization_logits_channel_1_forged_probability"),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    (
        reliability_path,
        reliability_file_sha,
        reliability_array_sha,
        reliability_map,
    ) = _load_probability_map(
        row,
        prefix="reliability_map_native",
        expected_path=paths["reliability"],
        expected_shape=shape,
        expected_semantics=("sigmoid_TCP_localization_reliability_not_anomaly"),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    expected_reliability = {
        "semantics": ("TCP_localization_reliability_not_forged_probability"),
        "used_for_primary_metrics": False,
        "multiplied_into_score_map": False,
        "min": float(np.min(reliability_map)),
        "mean": float(np.mean(reliability_map)),
        "median": float(np.median(reliability_map)),
        "p05": float(np.quantile(reliability_map, 0.05)),
        "p95": float(np.quantile(reliability_map, 0.95)),
        "max": float(np.max(reliability_map)),
    }
    _assert_nested_close(
        row.get("reliability"),
        expected_reliability,
        label=f"{sample_id}.reliability",
    )
    applicable = expected.get("gt_mask_kind") in _T2_GT_KINDS
    expected_paths = {
        "score_map_native": repo_relative(score_path, repo_root),
        "reliability_map_native": repo_relative(
            reliability_path,
            repo_root,
        ),
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
            score_path=score_path,
            score_file_sha256=score_file_sha,
            score_array_sha256=score_array_sha,
            reliability_path=reliability_path,
            reliability_file_sha256=reliability_file_sha,
            reliability_array_sha256=reliability_array_sha,
            mask_path=None,
            mask_file_sha256=None,
            mask_array_sha256=None,
            t2_applicable=False,
            width=width,
            height=height,
        )

    mask_path, mask_file_sha, mask_array_sha, mask = _load_mask(
        row,
        expected_path=paths["mask"],
        expected_shape=shape,
        repo_root=repo_root,
        sample_id=sample_id,
    )
    if not np.array_equal(mask == 255, score_map >= MASK_THRESHOLD):
        raise ValueError(f"{sample_id} mask is not raw score map >= 0.5")
    target = load_ground_truth(expected, repo_root)
    if target is None or target.shape != shape:
        raise ValueError(f"{sample_id} T2 ground truth changed")
    expected_localization = {
        "native": _independent_pixel_metrics(
            score_map,
            target,
            include_ap=expected.get("gt_mask_kind") == "exact_diff",
        )
    }
    _assert_nested_close(
        row.get("localization"),
        expected_localization,
        label=f"{sample_id}.localization",
    )
    if (
        expected.get("gt_mask_kind") == "all_zero"
        and expected_localization["native"]["pixel_ap"] is not None
    ):
        raise ValueError("real T2 row incorrectly reports pixel AP")
    return DenseArtifacts(
        sample_id=sample_id,
        score_path=score_path,
        score_file_sha256=score_file_sha,
        score_array_sha256=score_array_sha,
        reliability_path=reliability_path,
        reliability_file_sha256=reliability_file_sha,
        reliability_array_sha256=reliability_array_sha,
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
        raise FileNotFoundError(f"missing/unsafe TruFor artifact root: {artifact_root}")
    entries = list(artifact_root.iterdir())
    if {entry.name for entry in entries} != set(EXPECTED_ARTIFACT_INVENTORY) or any(
        not entry.is_dir() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("TruFor artifact-root inventory changed")
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for row in latest_results:
        sample_id = _require_string(
            row.get("sample_id"),
            "artifact result sample_id",
        )
        if sample_id in result_by_id:
            raise ValueError(f"duplicate latest result {sample_id}")
        if row.get("status") != "ok":
            raise ValueError(f"latest result {sample_id} is not successful")
        result_by_id[sample_id] = row
    if set(result_by_id) != set(expected_by_id):
        raise ValueError("TruFor artifact/result coverage changed")
    all_npy = {f"{sample_id}.npy" for sample_id in expected_by_id}
    applicable = {
        sample_id
        for sample_id, row in expected_by_id.items()
        if row.get("gt_mask_kind") in _T2_GT_KINDS
    }
    _exact_directory_inventory(
        artifact_root / "score_maps_native",
        expected_names=all_npy,
        label="TruFor score-map directory",
    )
    _exact_directory_inventory(
        artifact_root / "reliability_maps_native",
        expected_names=all_npy,
        label="TruFor reliability-map directory",
    )
    _exact_directory_inventory(
        artifact_root / "masks_native",
        expected_names={f"{sample_id}.png" for sample_id in applicable},
        label="TruFor mask directory",
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
        raise ValueError("TruFor validated artifact coverage changed")
    return artifacts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, DenseArtifacts],
) -> str:
    records = []
    for sample_id, artifact in sorted(artifacts.items()):
        records.append(
            {
                "sample_id": sample_id,
                "score_path": artifact.score_path.as_posix(),
                "score_file_sha256": artifact.score_file_sha256,
                "score_array_sha256": artifact.score_array_sha256,
                "reliability_path": artifact.reliability_path.as_posix(),
                "reliability_file_sha256": (artifact.reliability_file_sha256),
                "reliability_array_sha256": (artifact.reliability_array_sha256),
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


def _validate_dataset_manifest_outputs(
    *,
    manifest: Mapping[str, Any],
    immutable: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
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
) -> None:
    snapshot = _read_jsonl(expected_path, "TruFor expected inputs")
    if snapshot != list(selected):
        raise ValueError("TruFor expected-input snapshot changed")
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
        raise ValueError("TruFor manifest dataset binding changed")
    expected_base_outputs = {
        "results_path": repo_relative(results_path, repo_root),
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "summary_path": repo_relative(summary_path, repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        "score_maps_native_dir": repo_relative(
            artifact_root / "score_maps_native",
            repo_root,
        ),
        "reliability_maps_native_dir": repo_relative(
            artifact_root / "reliability_maps_native",
            repo_root,
        ),
        "masks_native_dir": repo_relative(
            artifact_root / "masks_native",
            repo_root,
        ),
    }
    if immutable.get("outputs") != expected_base_outputs:
        raise ValueError("TruFor immutable output bindings changed")
    inventory = {
        "score_maps_native": len(selected),
        "reliability_maps_native": len(selected),
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
        raise ValueError("TruFor finalized output bindings changed")
    expected_summary = {
        "schema_version": runner.RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": ("runtime_coverage_and_artifact_inventory_only"),
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_trufor_balanced.py",
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
        "artifact_inventory": inventory,
    }
    if set(summary) != set(expected_summary) | {"generated_at"}:
        raise ValueError("TruFor runtime summary key set changed")
    _require_string(summary.get("generated_at"), "summary.generated_at")
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"TruFor runtime summary {key} changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    latest_index = index_latest_attempts(
        selected,
        physical_results,
        run_id=str(manifest["run_id"]),
        run_manifest_fingerprint=str(manifest["fingerprint"]),
        score_spec=_score_spec(),
    )
    if (
        execution.get("physical_result_rows") != len(physical_results)
        or execution.get("latest_result_rows") != len(latest_results)
        or execution.get("superseded_attempts") != latest_index.superseded_attempts
        or execution.get("new_successes") + execution.get("resume_skips")
        != len(selected)
        or execution.get("new_successes") > len(selected)
        or execution.get("resume_skips") > len(selected)
        or execution.get("new_errors")
        > sum(row.get("status") == "error" for row in physical_results)
    ):
        raise ValueError("TruFor execution accounting changed")
    del run_dir


def _validate_run_directory(run_dir: Path) -> None:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError(f"missing/unsafe TruFor run dir: {run_dir}")
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
                f"TruFor run directory contains unsafe entry: {entry.name}"
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
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> RunBundle:
    run_dir = _resolve_run_dir(results_dir, run_id, "TruFor run directory")
    artifact_root = _resolve_run_dir(
        artifacts_dir,
        run_id,
        "TruFor artifact directory",
    )
    _validate_run_directory(run_dir)
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(
            f"missing/unsafe TruFor artifact directory: {artifact_root}"
        )
    manifest_path = run_dir / "manifest.json"
    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = _load_json(manifest_path, "TruFor manifest")
    summary = _load_json(summary_path, "TruFor runtime summary")
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
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    physical = tuple(_read_jsonl(results_path, "TruFor physical results"))
    runner._validate_physical_attempt_history(selected, physical)
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    for row in physical:
        sample_id = _require_string(
            row.get("sample_id"),
            "result.sample_id",
        )
        expected = expected_by_id.get(sample_id)
        if expected is None:
            raise ValueError(f"unexpected TruFor result {sample_id}")
        _validate_attempt(
            row,
            expected=expected,
            repo_root=repo_root,
            run_id=run_id,
            fingerprint=fingerprint,
        )
        runner._validate_runner_attempt(
            row,
            input_row=expected,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=False,
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
        raise ValueError("TruFor latest coverage contains an error")
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
        run_dir=run_dir,
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
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="formal",
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="smoke",
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )


def _native_map_loader(
    bundle: RunBundle,
):
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
        array = np.load(
            artifact.score_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            array.shape != (artifact.height, artifact.width)
            or array.dtype != np.float32
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
            or _array_sha256(array) != artifact.score_array_sha256
        ):
            raise ValueError(f"T2 callback map changed for {sample_id}")
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
            "TruFor formal metrics require iterations=1000 and " "seed=20260726"
        )
    if (
        bundle.mode != "formal"
        or len(bundle.selected) != FORMAL_IMAGES
        or sum(row.get("gt_mask_kind") in _T2_GT_KINDS for row in bundle.selected)
        != FORMAL_T2_IMAGES
    ):
        raise ValueError("TruFor metrics require formal 1775/1025 coverage")
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
        raise ValueError("TruFor shared T1 metrics are incomplete")
    t2 = summarize_balanced250_t2(
        bundle.release.inputs,
        selected_results,
        repo_root=bundle.release.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name="trufor_native_forged_probability_map",
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
        raise ValueError("TruFor shared T2 metrics are incomplete")
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
        raise ValueError("TruFor T2 fullframe exclusion evidence changed")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "formal_images_t1": FORMAL_IMAGES,
        "formal_images_t2": FORMAL_T2_IMAGES,
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_operator": ">=",
        "reliability_used_for_primary_metrics": False,
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
            raise ValueError(f"TruFor evidence changed during audit: {key}")
    release = load_canonical_release(
        bundle.release.repo_root,
        bundle.release.manifest_path,
        verify_files=True,
    )
    if release.manifest_sha256 != expected["dataset_manifest_sha256"]:
        raise ValueError("Balanced250 release changed during TruFor audit")
    records = []
    for sample_id, artifact in sorted(bundle.artifacts.items()):
        if (
            sha256_file(artifact.score_path) != artifact.score_file_sha256
            or sha256_file(artifact.reliability_path)
            != artifact.reliability_file_sha256
            or (
                artifact.mask_path is not None
                and sha256_file(artifact.mask_path) != artifact.mask_file_sha256
            )
        ):
            raise ValueError(f"TruFor artifact changed during audit: {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "score_path": artifact.score_path.as_posix(),
                "score_file_sha256": artifact.score_file_sha256,
                "score_array_sha256": artifact.score_array_sha256,
                "reliability_path": artifact.reliability_path.as_posix(),
                "reliability_file_sha256": (artifact.reliability_file_sha256),
                "reliability_array_sha256": (artifact.reliability_array_sha256),
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
    if _fingerprint(records) != expected["artifact_inventory_sha256"]:
        raise ValueError("TruFor artifact inventory changed during audit")


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
        raise ValueError("TruFor smoke comparison requires 35+35 rows")
    if set(reference_artifacts) != {str(row["sample_id"]) for row in reference} or set(
        replay_artifacts
    ) != {str(row["sample_id"]) for row in replay}:
        raise ValueError("TruFor smoke artifact coverage changed")
    applicable = 0
    for left, right in zip(reference, replay, strict=True):
        sample_id = str(left["sample_id"])
        if right.get("sample_id") != sample_id:
            raise ValueError("TruFor smoke result order changed")
        if _smoke_projection(left) != _smoke_projection(right):
            raise ValueError(f"TruFor smoke computational row changed: {sample_id}")
        left_artifact = reference_artifacts[sample_id]
        right_artifact = replay_artifacts[sample_id]
        if left_artifact.t2_applicable != right_artifact.t2_applicable:
            raise ValueError(f"TruFor smoke T2 applicability changed: {sample_id}")
        for label, left_path, right_path in (
            (
                "score map",
                left_artifact.score_path,
                right_artifact.score_path,
            ),
            (
                "reliability map",
                left_artifact.reliability_path,
                right_artifact.reliability_path,
            ),
            (
                "mask",
                left_artifact.mask_path,
                right_artifact.mask_path,
            ),
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
        raise ValueError("TruFor smoke T2 comparison coverage changed")
    return {
        "images_compared": SMOKE_IMAGES,
        "t2_applicable_images_compared": SMOKE_T2_IMAGES,
        "t2_not_applicable_images_compared": (SMOKE_IMAGES - SMOKE_T2_IMAGES),
        "computational_result_projection_exact": True,
        "raw_score_map_file_bytes_exact": True,
        "raw_reliability_map_file_bytes_exact": True,
        "applicable_threshold_png_file_bytes_exact": True,
        "fullframe_masks_absent_in_both_runs": True,
        "t1_scores_and_logits_exact": True,
        "localization_and_reliability_summaries_exact": True,
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
        / (f"{reference_run_id}__vs__{replay_run_id}" "_comparison.json")
    )


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        reference_run_id != DEFAULT_SMOKE_RUN_ID_A
        or replay_run_id != DEFAULT_SMOKE_RUN_ID_B
        or reference_run_id == replay_run_id
    ):
        raise ValueError("TruFor smoke comparison requires frozen A then B")
    reference = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=reference_run_id,
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    replay = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=replay_run_id,
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    if (
        [row["sample_id"] for row in reference.selected]
        != [row["sample_id"] for row in replay.selected]
        or reference.contract.as_dict() != replay.contract.as_dict()
        or _smoke_immutable_projection(reference.immutable)
        != _smoke_immutable_projection(replay.immutable)
    ):
        raise ValueError("TruFor A/B smoke immutable contract changed")
    comparison = compare_computational_results(
        reference.latest_results,
        replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
    )
    if reference.immutable.get("runtime") != replay.immutable.get("runtime"):
        raise ValueError("TruFor smoke recorded runtimes differ")
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
            "t2_not_applicable_images": (SMOKE_IMAGES - SMOKE_T2_IMAGES),
        },
        "recorded_runtime_exact": True,
        "comparison": comparison,
        "numerical_golden_boundary": {
            "author_published_numerical_golden": None,
            "smoke_A_B_is_executable_reproduction_gate": True,
            "no_author_golden_claim_fabricated": True,
        },
        "analyzer_source": {
            "path": "eval/opensource/analyze_trufor_balanced.py",
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
        label="TruFor smoke comparison output",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_verified(
        output,
        report,
        label="TruFor smoke comparison",
    )
    return report


def _configure_recorded_runtime(
    *,
    recorded: Mapping[str, Any],
    device_text: str,
) -> tuple[Any, dict[str, Any]]:
    expected = _validate_runtime(recorded, label="recorded runtime")
    if device_text != expected.get("device"):
        raise ValueError("TruFor replay device must equal the recorded runtime device")
    device, current = runner.configure_runtime(device_text)
    validated = _validate_runtime(current, label="current replay runtime")
    if validated != expected:
        raise ValueError("TruFor current runtime differs from recorded")
    return device, validated


def replay_model(
    bundle: RunBundle,
    *,
    trufor_root: Path,
    checkpoint_path: Path,
    device: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("TruFor fresh replay requires formal 1775")
    model = None
    fresh_rows: list[dict[str, Any]] = []
    maps_compared = 0
    reliability_compared = 0
    masks_derived = 0
    try:
        model, loaded_device = legacy.load_model(
            trufor_root=trufor_root,
            checkpoint_path=checkpoint_path,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("TruFor fresh model loaded on wrong device")
        for index, (input_row, persisted) in enumerate(
            zip(
                bundle.selected,
                bundle.latest_results,
                strict=True,
            ),
            start=1,
        ):
            sample_id = str(input_row["sample_id"])
            if persisted.get("sample_id") != sample_id:
                raise ValueError("TruFor fresh replay order changed")
            input_path = _safe_repo_file(
                input_row.get("canonical_path"),
                repo_root=bundle.release.repo_root,
                expected_path=(
                    bundle.release.repo_root / str(input_row["canonical_path"])
                ),
                label=f"{sample_id} replay input",
            )
            tensor, preprocess = _independent_preprocess_tensor(input_path)
            if preprocess != persisted.get("preprocess"):
                raise ValueError(f"{sample_id} fresh preprocessing changed")
            (
                score,
                logit,
                score_map,
                reliability_map,
                _peak_bytes,
                _latency_ms,
            ) = legacy.infer_one(model, device, tensor)
            if float(score) != persisted.get("ai_score") or float(
                logit
            ) != persisted.get("raw_detection_logit"):
                raise ValueError(f"{sample_id} fresh T1 score/logit changed")
            artifact = bundle.artifacts[sample_id]
            stored_score = np.load(
                artifact.score_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            stored_reliability = np.load(
                artifact.reliability_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            score_array = np.ascontiguousarray(
                score_map,
                dtype=np.float32,
            )
            reliability_array = np.ascontiguousarray(
                reliability_map,
                dtype=np.float32,
            )
            if not np.array_equal(score_array, stored_score):
                raise ValueError(f"{sample_id} fresh raw score map changed")
            if not np.array_equal(
                reliability_array,
                stored_reliability,
            ):
                raise ValueError(f"{sample_id} fresh reliability map changed")
            maps_compared += 1
            reliability_compared += 1
            if artifact.t2_applicable:
                if artifact.mask_path is None:
                    raise ValueError(f"{sample_id} applicable replay mask missing")
                with Image.open(artifact.mask_path) as opened:
                    mask = np.asarray(opened, dtype=np.uint8)
                if not np.array_equal(
                    mask == 255,
                    score_array >= MASK_THRESHOLD,
                ):
                    raise ValueError(f"{sample_id} fresh derived mask changed")
                masks_derived += 1
            elif artifact.mask_path is not None:
                raise ValueError(f"{sample_id} fullframe replay fabricates a mask")
            fresh_row = dict(persisted)
            fresh_row.update(runner._score_payload(score, logit))
            if _stable_result_row(fresh_row) != _stable_result_row(persisted):
                raise ValueError(f"{sample_id} fresh metric result row changed")
            fresh_rows.append(fresh_row)
            del (
                tensor,
                score_map,
                reliability_map,
                score_array,
                reliability_array,
                stored_score,
                stored_reliability,
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
        del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            import torch

            torch.cuda.empty_cache()
    if (
        len(fresh_rows) != FORMAL_IMAGES
        or maps_compared != FORMAL_IMAGES
        or reliability_compared != FORMAL_IMAGES
        or masks_derived != FORMAL_T2_IMAGES
    ):
        raise ValueError("TruFor fresh replay coverage changed")
    return (
        {
            "status": "fresh_full_model_replay_exact",
            "selected_images_freshly_reopened": FORMAL_IMAGES,
            "selected_images_freshly_preprocessed": FORMAL_IMAGES,
            "model_forwards": FORMAL_IMAGES,
            "t1_scores_compared_exact": FORMAL_IMAGES,
            "detection_logits_compared_exact": FORMAL_IMAGES,
            "raw_score_maps_compared_exact": maps_compared,
            "raw_reliability_maps_compared_exact": (reliability_compared),
            "applicable_masks_rederived_exact": masks_derived,
            "fullframe_masks_not_created": (FORMAL_IMAGES - FORMAL_T2_IMAGES),
            "maximum_t1_score_abs_diff": 0.0,
            "maximum_detection_logit_abs_diff": 0.0,
            "maximum_score_map_abs_diff": 0.0,
            "maximum_reliability_map_abs_diff": 0.0,
        },
        tuple(fresh_rows),
    )


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    trufor_root: Path,
    checkpoint_path: Path,
    archive_path: Path,
    device_text: str,
    metrics_output_path: Path,
    audit_output_path: Path,
    replay: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError(f"TruFor formal audit requires {DEFAULT_FORMAL_RUN_ID}")
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
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
                artifact.score_path,
                artifact.reliability_path,
                artifact.mask_path,
            )
            if path is not None
        ],
    ]
    metrics_output = _validate_output_path(
        metrics_output_path,
        expected_path=bundle.run_dir / "balanced250_metrics.json",
        protected=protected,
        label="TruFor Balanced250 metrics output",
    )
    audit_output = _validate_output_path(
        audit_output_path,
        expected_path=bundle.run_dir / "independent_audit.json",
        protected=[*protected, metrics_output],
        label="TruFor independent audit output",
    )

    structural_golden = independent_structural_golden(
        checkpoint_path=checkpoint_path,
        trufor_root=trufor_root,
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
            trufor_root=trufor_root,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        if [_stable_result_row(row) for row in fresh_rows] != [
            _stable_result_row(row) for row in bundle.latest_results
        ]:
            raise ValueError("TruFor fresh replay metric rows changed")
        # T1 consumes only the now-exact result rows.  T2 consumes only the
        # now-exact native score maps; reliability never enters a primary
        # metric.  Re-running the expensive AP/bootstrap reducer on identical
        # bytes cannot add evidence, so metric equivalence follows exactly
        # from the complete input-equivalence gate above.
        fresh_metrics_exact = True
    source_after = _independent_source_record(trufor_root)
    assets_after = _independent_assets_record(
        trufor_root=trufor_root,
        checkpoint_path=checkpoint_path,
        archive_path=archive_path,
    )
    if source_after != bundle.immutable.get(
        "source"
    ) or assets_after != bundle.immutable.get("assets"):
        raise ValueError("TruFor source/assets changed during analysis")
    _verify_bundle_unchanged(bundle)

    metrics_sha = _json_sha256(metrics)
    analyzer_path = Path(__file__).resolve()
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
            "fullframe_t2_not_applicable_images": (FORMAL_IMAGES - FORMAL_T2_IMAGES),
        },
        "coverage": dict(bundle.coverage),
        "evidence_snapshot": dict(bundle.evidence_snapshot),
        "independent_provenance": {
            "source_exact_and_clean": True,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "source_bound_files": len(
                {
                    **runner.TRUFOR_SOURCE_FILES,
                    **runner.SOURCE_BOUND_ASSETS,
                }
            ),
            "archive_sha256": legacy.CHECKPOINT_ZIP_SHA256,
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": runner.CHECKPOINT_BYTES,
            "configuration_sha256": legacy.MODEL_CONFIG_SHA256,
            "overall_license_sha256": legacy.MODEL_LICENSE_SHA256,
            "analysis_environment_exact": True,
            "source_and_assets_unchanged_after_replay": True,
        },
        "independent_structural_golden": structural_golden,
        "recorded_runtime": recorded_runtime,
        "recorded_runtime_reproduced": current_runtime,
        "artifact_audit": {
            "raw_score_maps_float32_verified": FORMAL_IMAGES,
            "raw_reliability_maps_float32_verified": FORMAL_IMAGES,
            "native_threshold_png_masks_verified": FORMAL_T2_IMAGES,
            "fullframe_masks_absent": (FORMAL_IMAGES - FORMAL_T2_IMAGES),
            "all_paths_canonical": True,
            "all_file_hashes_exact": True,
            "all_array_hashes_exact": True,
            "all_shapes_and_dtypes_exact": True,
            "all_values_finite_and_in_unit_interval": True,
            "applicable_png_exact_score_map_ge_0_5": True,
            "reliability_used_for_primary_metrics": False,
            "reliability_not_multiplied_into_score_map": True,
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
        },
        "fresh_model_replay": fresh_replay,
        "fresh_model_metrics_exact": fresh_metrics_exact,
        "fresh_model_metrics_equivalence_proof": (
            {
                "t1_inputs": ("all_1775_metric_result_rows_exact"),
                "t2_inputs": ("all_1025_applicable_native_score_maps_exact"),
                "reliability_excluded_from_primary_metrics": True,
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
            "tcp_reliability_is_diagnostic_only": True,
            "author_published_numeric_golden_available": False,
            "no_numeric_golden_was_fabricated": True,
            "fresh_full_model_replay_is_default": True,
        },
        "license": {
            "identifier": ("TruFor_custom_informational_nonprofit_only"),
            "commercial_use_cleared": False,
            "commercial_use_requires_separate_authorization": True,
            "cmx_mit_does_not_override_overall_restriction": True,
        },
        "analyzer_source": {
            "path": "eval/opensource/analyze_trufor_balanced.py",
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
        label="TruFor Balanced250 metrics",
    )
    _write_json_verified(
        audit_output,
        report,
        label="TruFor independent audit",
    )
    if sha256_file(metrics_output) != metrics_sha:
        raise ValueError("TruFor metrics changed after write")
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
        "--trufor-root",
        type=Path,
        default=DEFAULT_TRUFOR_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
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
        label="TruFor results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="TruFor artifacts root",
    )
    run_id = _valid_run_id(args.run_id)
    trufor_root = _anchored(args.trufor_root, repo_root)
    checkpoint = _anchored(args.checkpoint, repo_root)
    archive = _anchored(args.archive, repo_root)
    if args.compare_smoke_run_id is not None:
        compare_id = _valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
        ):
            raise ValueError("TruFor smoke comparison accepts no formal options")
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
            trufor_root=trufor_root,
            checkpoint_path=checkpoint,
            archive_path=archive,
            output_path=comparison_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError("--comparison-output requires --compare-smoke-run-id")
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError("TruFor formal run ID is frozen")
    run_dir = _resolve_run_dir(
        results_dir,
        run_id,
        "TruFor formal run directory",
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
        trufor_root=trufor_root,
        checkpoint_path=checkpoint,
        archive_path=archive,
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
