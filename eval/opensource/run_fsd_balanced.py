#!/usr/bin/env python3
"""Run the pinned official FSD v1.2 inference release on Balanced250.

This v2 orchestration layer keeps the legacy Mouse runner unchanged.  It
selects the exact whole-image T1 Balanced250 inputs, performs an exact
CPU-only golden preflight before configuring the requested accelerator,
records append-only result attempts, and persists each official raw
960-dimensional float64 descriptor without computing scientific metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_fsd as legacy
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
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
    select_inputs,
)
from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


RUN_MANIFEST_SCHEMA = "fsd_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "fsd_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "fsd_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "fsd_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/balanced250_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/fsd")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/fsd")
DEFAULT_FORMAL_RUN_ID = (
    "fsd_v1_2_0_official_balanced250_v1_full1775_20260726"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_WEIGHTS_DIR = Path(
    "/root/.cache/claimforge/checkpoints/fsd-v1.2.0"
)
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = 20260726
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 12 * 1024**3

FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.10.0+cu128",
    "torch_distribution": "2.10.0",
    "numpy": "2.4.3",
    "Pillow": "12.1.1",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
}
FROZEN_PYTHON_EXECUTABLE = Path(
    "/root/.cache/claimforge/venvs/fsd-v1.2.0/bin/python"
)

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=legacy.AI_SCORE_THRESHOLD,
    threshold_operator=legacy.THRESHOLD_OPERATOR,
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

PREPROCESS_CONTRACT = {
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
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_CONTRACT = {
    "descriptor": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.FSD_DIMENSION],
        "dtype": "float64",
        "nbytes": legacy.FSD_DIMENSION * np.dtype(np.float64).itemsize,
        "finite": True,
        "semantics": "official_compute_fsd_before_released_transforms",
        "allow_pickle": False,
    }
}

TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}

MODEL_CONTRACT = {
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

ADAPTER_SOURCE_PATHS = (
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

IMMUTABLE_CONFIG_KEYS = frozenset(
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

CPU_GOLDEN_SAMPLE_ID = "5f7535f0b957874982b1b080"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/"
    f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
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
EXPECTED_WEIGHTS_BUNDLE_SHA256 = (
    "3c1959f0092fdbe681e41c96f12c1b6d3762e46f21b37b08d4a7e617d1acdfce"
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(
            "run-id must be one safe ASCII path component (max 160 chars)"
        )
    return value


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local source participating in inference and audit."""

    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"missing or unsafe FSD Balanced adapter source: {path}"
            )
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
        or any("pair_rank" in row for row in selected)
    ):
        raise ValueError("formal FSD Balanced250 selection drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        isinstance(per_condition_limit, bool)
        or not isinstance(per_condition_limit, int)
        or not 1 <= per_condition_limit <= 250
    ):
        raise ValueError("smoke per-condition-limit must be in [1, 250]")
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=per_condition_limit,
    )
    inputs_by_id = {
        str(row["sample_id"]): row for row in release.inputs
    }
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if (
            condition in Capability.WHOLE_IMAGE_T1.conditions
            and counts[condition] < per_condition_limit
        ):
            sample_id = str(panel_row["sample_id"])
            source = inputs_by_id.get(sample_id)
            if source is None or source.get("panel") is not True:
                raise ValueError("smoke panel has a dangling/non-panel input")
            selected.append(source)
            counts[condition] += 1
    expected = {
        condition: per_condition_limit for condition in BALANCED_CONDITIONS
    }
    if dict(counts) != expected or any("pair_rank" in row for row in selected):
        raise ValueError("smoke panel does not cover the frozen T1 conditions")
    selected.sort(key=lambda row: int(row["rank"]))
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    """Create the exact formal, panel-smoke, or diagnostic single selection."""

    if release.release_kind != "balanced250":
        raise ValueError("FSD v2 requires a Balanced250 release")
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode does not accept input selectors")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode does not accept sample-id")
        return _smoke_selection(
            release,
            DEFAULT_SMOKE_LIMIT
            if per_condition_limit is None
            else per_condition_limit,
        )
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode does not accept per-condition-limit")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            sample_id=sample_id,
        )
        selected = select_inputs(release, spec)
        if len(selected) != 1 or "pair_rank" in selected[0]:
            raise ValueError("single selection drifted")
        return spec, selected
    raise ValueError(f"unsupported inference mode {mode!r}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_cublas_workspace() -> str:
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_CONFIG}, got {current!r}"
        )
    return CUBLAS_WORKSPACE_CONFIG


def _frozen_runtime_versions() -> dict[str, str]:
    import torch

    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_distribution": str(_package_version("torch")),
        "numpy": str(np.__version__),
        "Pillow": str(_package_version("Pillow")),
        "scipy": str(_package_version("scipy")),
        "scikit-learn": str(_package_version("scikit-learn")),
    }
    if actual != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            "FSD dedicated runtime version drifted: "
            f"expected {FROZEN_RUNTIME_VERSIONS}, got {actual}"
        )
    executable = Path(sys.executable).resolve()
    if executable != FROZEN_PYTHON_EXECUTABLE.resolve():
        raise RuntimeError(
            "FSD must run in its dedicated Python environment: "
            f"{FROZEN_PYTHON_EXECUTABLE}"
        )
    return actual


def configure_runtime(
    device_text: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the exact deterministic FSD runtime and requested device."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"FSD runtime seed must be exactly {DEFAULT_SEED}")
    cublas_workspace = _configure_cublas_workspace()
    import torch

    versions = _frozen_runtime_versions()
    if device_text == "cpu":
        device = torch.device("cpu")
    elif re.fullmatch(r"cuda:[0-9]+", device_text):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_text)
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device_text}")
        torch.cuda.set_device(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        if int(free_bytes) < MINIMUM_CUDA_FREE_BYTES:
            raise RuntimeError(
                f"{device} has only {int(free_bytes)} free bytes; FSD requires "
                f"at least {MINIMUM_CUDA_FREE_BYTES}"
            )
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    runtime: dict[str, Any] = {
        "device": str(device),
        "python": {
            "implementation": platform.python_implementation(),
            "version": versions["python"],
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "packages": {
            "torch": {
                "version": versions["torch"],
                "distribution_version": versions["torch_distribution"],
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
            },
            "numpy": versions["numpy"],
            "Pillow": versions["Pillow"],
            "scipy": versions["scipy"],
            "scikit-learn": versions["scikit-learn"],
        },
        "seed": seed,
        "descriptor_dtype": "float64",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": cublas_workspace,
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "minimum_cuda_free_bytes": MINIMUM_CUDA_FREE_BYTES,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    validate_runtime_contract(runtime, label="configured runtime")
    return device, runtime


def validate_runtime_contract(
    value: Mapping[str, Any],
    *,
    label: str = "runtime",
) -> Mapping[str, Any]:
    """Validate a persisted runtime without reconfiguring global state."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    device = value.get("device")
    if not isinstance(device, str) or (
        device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None
    ):
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
    if device.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = value.get("python")
    if not isinstance(python_record, Mapping) or set(python_record) != {
        "implementation",
        "version",
        "executable",
    }:
        raise ValueError(f"{label}.python key set changed")
    if (
        python_record.get("implementation") != "CPython"
        or python_record.get("version") != FROZEN_RUNTIME_VERSIONS["python"]
        or Path(str(python_record.get("executable"))).resolve()
        != FROZEN_PYTHON_EXECUTABLE.resolve()
    ):
        raise ValueError(f"{label}.python dedicated runtime changed")
    packages = value.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "torch",
        "numpy",
        "Pillow",
        "scipy",
        "scikit-learn",
    }:
        raise ValueError(f"{label}.packages key set changed")
    torch_record = packages.get("torch")
    if not isinstance(torch_record, Mapping) or set(torch_record) != {
        "version",
        "distribution_version",
        "cuda_runtime",
        "cudnn_version",
    }:
        raise ValueError(f"{label}.packages.torch key set changed")
    if (
        torch_record.get("version") != FROZEN_RUNTIME_VERSIONS["torch"]
        or torch_record.get("distribution_version")
        != FROZEN_RUNTIME_VERSIONS["torch_distribution"]
        or torch_record.get("cuda_runtime") != "12.8"
        or packages.get("numpy") != FROZEN_RUNTIME_VERSIONS["numpy"]
        or packages.get("Pillow") != FROZEN_RUNTIME_VERSIONS["Pillow"]
        or packages.get("scipy") != FROZEN_RUNTIME_VERSIONS["scipy"]
        or packages.get("scikit-learn")
        != FROZEN_RUNTIME_VERSIONS["scikit-learn"]
    ):
        raise ValueError(f"{label}.packages frozen versions changed")
    if not isinstance(value.get("platform"), str) or not value["platform"]:
        raise ValueError(f"{label}.platform is invalid")
    if (
        value.get("seed") != DEFAULT_SEED
        or value.get("descriptor_dtype") != "float64"
        or value.get("batch_size") != 1
        or value.get("autocast") is not False
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("minimum_cuda_free_bytes") != MINIMUM_CUDA_FREE_BYTES
    ):
        raise ValueError(f"{label} deterministic numerical contract changed")
    cudnn = value.get("cudnn")
    if not isinstance(cudnn, Mapping) or set(cudnn) != {
        "enabled",
        "benchmark",
        "deterministic",
        "allow_tf32",
    }:
        raise ValueError(f"{label}.cudnn key set changed")
    if (
        type(cudnn.get("enabled")) is not bool
        or cudnn.get("benchmark") is not False
        or cudnn.get("deterministic") is not True
        or cudnn.get("allow_tf32") is not False
    ):
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if device.startswith("cuda:"):
        cuda = value.get("cuda")
        if not isinstance(cuda, Mapping) or set(cuda) != {
            "runtime",
            "device_index",
            "device_name",
            "total_memory_bytes",
            "capability",
        }:
            raise ValueError(f"{label}.cuda key set changed")
        index = int(device.split(":", 1)[1])
        capability = cuda.get("capability")
        if (
            cuda.get("runtime") != "12.8"
            or cuda.get("device_index") != index
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
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    return path


def _validate_golden_forward(
    processed: Mapping[str, Any],
    descriptor: np.ndarray,
    *,
    label: str,
) -> tuple[bytes, str]:
    if (
        not isinstance(descriptor, np.ndarray)
        or descriptor.shape != (legacy.FSD_DIMENSION,)
        or descriptor.dtype != np.float64
        or not descriptor.flags.c_contiguous
        or not np.isfinite(descriptor).all()
    ):
        raise ValueError(f"{label} descriptor changed")
    expected = {
        "raw_likelihood": CPU_GOLDEN_RAW_LIKELIHOOD,
        "released_z_score": CPU_GOLDEN_RELEASED_Z_SCORE,
        "ai_score": CPU_GOLDEN_AI_SCORE,
        "released_is_fake": False,
        "classification_decision": False,
    }
    for key, expected_value in expected.items():
        if processed.get(key) != expected_value:
            raise ValueError(f"{label} {key} changed")
    manual = processed.get("manual_replay")
    if manual != {
        "raw_likelihood": CPU_GOLDEN_RAW_LIKELIHOOD,
        "released_z_score": CPU_GOLDEN_RELEASED_Z_SCORE,
        "ai_score": CPU_GOLDEN_AI_SCORE,
        "released_is_fake": False,
        "classification_decision": False,
        "official_raw_exact_match": True,
        "official_z_exact_match": True,
        "compute_fsd_calls": 1,
    }:
        raise ValueError(f"{label} manual full-forward evidence changed")
    array_sha = _array_sha256(descriptor)
    if array_sha != CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256:
        raise ValueError(f"{label} descriptor array SHA-256 changed")
    payload = _npy_bytes(descriptor)
    if hashlib.sha256(payload).hexdigest() != CPU_GOLDEN_DESCRIPTOR_FILE_SHA256:
        raise ValueError(f"{label} descriptor NPY SHA-256 changed")
    return payload, array_sha


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    weights_dir: Path,
) -> dict[str, Any]:
    """Run the pinned full-image CPU golden twice before accelerator setup."""

    root = repo_root.resolve()
    source_root = source_root.resolve()
    weights_dir = weights_dir.resolve()
    image_path = root / CPU_GOLDEN_INPUT_PATH
    _require_regular_file(image_path, "CPU golden image")
    if sha256_file(image_path) != CPU_GOLDEN_IMAGE_SHA256:
        raise ValueError("CPU golden input SHA-256 changed")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    detector = None
    try:
        detector, loaded_device, load_audit = legacy.load_detector(
            source_root=source_root,
            weights_dir=weights_dir,
            device_name="cpu",
        )
        if str(loaded_device) != "cpu" or str(device) != "cpu":
            raise ValueError("CPU golden detector did not load on CPU")
        first, first_descriptor, _first_peak, _first_latency = legacy.infer_one(
            detector,
            loaded_device,
            image_path,
        )
        second, second_descriptor, _second_peak, _second_latency = (
            legacy.infer_one(
                detector,
                loaded_device,
                image_path,
            )
        )
        first_bytes, first_array_sha = _validate_golden_forward(
            first,
            first_descriptor,
            label="CPU golden first forward",
        )
        second_bytes, second_array_sha = _validate_golden_forward(
            second,
            second_descriptor,
            label="CPU golden repeat forward",
        )
        if (
            first != second
            or not np.array_equal(first_descriptor, second_descriptor)
            or first_bytes != second_bytes
        ):
            raise ValueError("CPU golden repeated full forwards are not exact")
        source = load_audit.get("source")
        weights = load_audit.get("weights")
        if not isinstance(source, dict) or not isinstance(weights, dict):
            raise ValueError("CPU golden load audit lacks provenance")
        golden = {
            "sample_id": CPU_GOLDEN_SAMPLE_ID,
            "input_path": CPU_GOLDEN_INPUT_PATH,
            "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
            "input_width": 1800,
            "input_height": 1350,
            "preprocess": legacy.compute_preprocess_geometry(1800, 1350),
            "descriptor_file_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "descriptor_file_bytes": len(first_bytes),
            "descriptor_array_sha256": first_array_sha,
            "descriptor_shape": [legacy.FSD_DIMENSION],
            "descriptor_dtype": "float64",
            "descriptor_nbytes": int(first_descriptor.nbytes),
            "raw_likelihood": first["raw_likelihood"],
            "released_z_score": first["released_z_score"],
            "ai_score": first["ai_score"],
            "classification_decision": first["classification_decision"],
            "released_is_fake": first["released_is_fake"],
            "full_image_forward": True,
            "compute_fsd_calls": 1,
            "repeat_descriptor_file_sha256": hashlib.sha256(
                second_bytes
            ).hexdigest(),
            "repeat_descriptor_file_bytes": len(second_bytes),
            "repeat_descriptor_array_sha256": second_array_sha,
            "repeat_raw_likelihood": second["raw_likelihood"],
            "repeat_released_z_score": second["released_z_score"],
            "repeat_ai_score": second["ai_score"],
            "repeat_classification_decision": second[
                "classification_decision"
            ],
            "repeat_released_is_fake": second["released_is_fake"],
            "repeat_full_image_forward": True,
            "repeat_compute_fsd_calls": 1,
            "repeat_byte_exact": True,
        }
        return {
            "schema_version": CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "weights": weights,
            "runtime": runtime,
            "golden": golden,
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "dataset_manifest_loaded": False,
        }
    finally:
        if detector is not None:
            del detector
        gc.collect()


_FALSE_DECLARATIONS = frozenset(
    {"valid_for_t2", "native_dense_output", "t2_applicable"}
)
_NULL_DECLARATIONS = frozenset(
    {
        "localization_output",
        "localisation_output",
        "dense_output",
        "score_map",
        "predicted_mask",
        "t2",
        "joint",
        "joint_score",
    }
)
_FORBIDDEN_CLAIM_KEYS = frozenset(
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
_FORBIDDEN_CLAIM_PREFIXES = (
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


def _reject_unsupported_claims(value: Any, label: str = "payload") -> None:
    """Reject pair-rank, localization, T2, dense, and joint-score claims."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if normalized in _ALLOWED_DIAGNOSTIC_KEYS:
                _reject_unsupported_claims(child, child_label)
                continue
            if normalized == "pair_rank":
                raise ValueError(f"{child_label} is an unsupported FSD claim")
            if normalized in _FALSE_DECLARATIONS:
                if child is not False:
                    raise ValueError(
                        f"{child_label} is an unsupported FSD claim"
                    )
                continue
            if normalized in _NULL_DECLARATIONS:
                if child is not None:
                    raise ValueError(
                        f"{child_label} is an unsupported FSD claim"
                    )
                continue
            if normalized in _FORBIDDEN_CLAIM_KEYS or normalized.startswith(
                _FORBIDDEN_CLAIM_PREFIXES
            ):
                raise ValueError(f"{child_label} is an unsupported FSD claim")
            _reject_unsupported_claims(child, child_label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _reject_unsupported_claims(child, f"{label}[{index}]")


def _visibility_diagnostic(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Describe how much local exact-diff GT survives FSD's center crop."""

    gt_kind = row.get("gt_mask_kind")
    preprocess = legacy.compute_preprocess_geometry(
        int(row["width"]),
        int(row["height"]),
    )
    if gt_kind == "exact_diff":
        gt = legacy._validate_gt_row(row, repo_root)
        if gt is None:
            raise ValueError("local exact-diff input has no GT mask")
        edit_region = row.get("edit_region_xyxy")
        if (
            not isinstance(edit_region, list)
            or len(edit_region) != 4
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in edit_region)
        ):
            raise ValueError("local exact-diff input has invalid edit region")
        gt_diagnostic = legacy._gt_visibility(row, gt, preprocess)
        box_diagnostic = legacy._box_visibility(
            edit_region,
            list(preprocess["effective_native_crop_xyxy"]),
        )
        return {
            "edit_visibility": gt_diagnostic["category"],
            "edit_visible_gt_fraction": gt_diagnostic["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": gt_diagnostic,
                "edit_box": box_diagnostic,
            },
        }
    if gt_kind not in ("all_zero", "not_applicable"):
        raise ValueError("Balanced250 input has unsupported GT kind")
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


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    weights_bundle_sha256: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    """Build the exact FSD extension of the shared Balanced250 v2 identity."""

    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    if not isinstance(weights_bundle_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        weights_bundle_sha256,
    ):
        raise ValueError("weights bundle SHA-256 is invalid")
    if weights_bundle_sha256 != EXPECTED_WEIGHTS_BUNDLE_SHA256:
        raise ValueError("weights bundle SHA-256 changed")
    identity = build_result_identity(
        row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
    )
    input_path = _anchored(Path(str(row["canonical_path"])), repo_root)
    if repo_relative(input_path, repo_root) != identity["input_path"]:
        raise ValueError("Balanced250 canonical input path is not repository-local")
    return {
        **identity,
        "valid_for_metrics": valid_for_metrics,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "model_source_commit": legacy.MODEL_SOURCE_COMMIT,
        "release_tag": legacy.RELEASE_TAG,
        "weights_bundle_sha256": weights_bundle_sha256,
        "config_fingerprint": run_manifest_fingerprint,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        **_visibility_diagnostic(row, repo_root=repo_root),
    }


_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "descriptor",
        "raw_descriptor_path",
        "raw_descriptor_sha256",
        "raw_descriptor_array_sha256",
        "raw_descriptor_shape",
        "raw_descriptor_dtype",
        "raw_descriptor_nbytes",
        "raw_descriptor_semantics",
        "artifact_paths",
        "raw_likelihood",
        "released_z_score",
        "ai_score",
        "score",
        "score_semantics",
        "released_is_fake",
        "released_threshold",
        "released_threshold_operator",
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


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _safe_repo_file(value: Any, *, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a canonical repository path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} is not a canonical repository path")
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
        raise ValueError(f"{label} escapes the repository") from error
    _require_regular_file(resolved, label)
    return resolved


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    raw = _finite_number(
        row.get("raw_likelihood"),
        f"{sample_id} raw_likelihood",
    )
    z_score = _finite_number(
        row.get("released_z_score"),
        f"{sample_id} released_z_score",
    )
    ai_score = _finite_number(row.get("ai_score"), f"{sample_id} ai_score")
    if z_score != (raw - legacy.TRAIN_MEAN) / legacy.TRAIN_STD:
        raise ValueError(f"{sample_id} released calibration changed")
    if ai_score != -z_score or row.get("score") != ai_score:
        raise ValueError(f"{sample_id} score orientation/alias changed")
    released_decision = z_score < legacy.RELEASED_Z_THRESHOLD
    decision = ai_score > legacy.AI_SCORE_THRESHOLD
    if released_decision != decision:
        raise ValueError(f"{sample_id} released/AI decisions differ")
    expected_classification = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": decision,
        "threshold": legacy.AI_SCORE_THRESHOLD,
        "threshold_operator": legacy.THRESHOLD_OPERATOR,
        "semantics": "higher_is_more_AI_negative_released_z",
    }
    expected_t1 = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": decision,
        "threshold": legacy.AI_SCORE_THRESHOLD,
        "threshold_operator": legacy.THRESHOLD_OPERATOR,
        "policy": "released_FSD_whole_image_score_sign_inverted",
    }
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
    if (
        row.get("score_semantics") != "negative_released_FSD_z_score"
        or row.get("released_is_fake") is not released_decision
        or row.get("released_threshold") != legacy.RELEASED_Z_THRESHOLD
        or row.get("released_threshold_operator")
        != legacy.RELEASED_THRESHOLD_OPERATOR
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold") != legacy.AI_SCORE_THRESHOLD
        or row.get("classification_threshold_operator")
        != legacy.THRESHOLD_OPERATOR
        or row.get("classification") != expected_classification
        or row.get("t1") != expected_t1
        or row.get("manual_replay") != expected_manual
    ):
        raise ValueError(f"{sample_id} score aliases/manual replay changed")


def _validate_descriptor_artifact(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    repo_root: Path,
    run_id: str,
) -> None:
    descriptor = row.get("descriptor")
    expected_descriptor_keys = {
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
    if not isinstance(descriptor, Mapping) or set(descriptor) != (
        expected_descriptor_keys
    ):
        raise ValueError(f"{sample_id} descriptor key set changed")
    expected_relative = (
        DEFAULT_ARTIFACTS_DIR
        / run_id
        / "raw_descriptors"
        / f"{sample_id}.npy"
    ).as_posix()
    if descriptor.get("relative_path") != expected_relative:
        raise ValueError(f"{sample_id} descriptor path is not canonical")
    path = _safe_repo_file(
        descriptor["relative_path"],
        repo_root=repo_root,
        label=f"{sample_id} descriptor",
    )
    payload = path.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    if (
        descriptor.get("sha256") != file_sha
        or descriptor.get("file_bytes") != len(payload)
        or descriptor.get("dtype") != "float64"
        or descriptor.get("shape") != [legacy.FSD_DIMENSION]
        or descriptor.get("nbytes")
        != legacy.FSD_DIMENSION * np.dtype(np.float64).itemsize
        or descriptor.get("finite") is not True
        or descriptor.get("semantics")
        != "official_compute_fsd_before_released_transforms"
    ):
        raise ValueError(f"{sample_id} descriptor metadata/hash changed")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} descriptor NPY is invalid") from error
    if (
        not isinstance(array, np.ndarray)
        or array.shape != (legacy.FSD_DIMENSION,)
        or array.dtype != np.float64
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or array.nbytes
        != legacy.FSD_DIMENSION * np.dtype(np.float64).itemsize
        or payload != _npy_bytes(array)
    ):
        raise ValueError(f"{sample_id} descriptor array changed")
    array_sha = _array_sha256(array)
    if descriptor.get("array_sha256") != array_sha:
        raise ValueError(f"{sample_id} descriptor array SHA-256 changed")
    aliases = {
        "raw_descriptor_path": expected_relative,
        "raw_descriptor_sha256": file_sha,
        "raw_descriptor_array_sha256": array_sha,
        "raw_descriptor_shape": [legacy.FSD_DIMENSION],
        "raw_descriptor_dtype": "float64",
        "raw_descriptor_nbytes": int(array.nbytes),
        "raw_descriptor_semantics": (
            "official_compute_fsd_before_released_transforms"
        ),
        "artifact_paths": {"raw_descriptor_npy": expected_relative},
    }
    for key, expected in aliases.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id} descriptor alias {key} changed")


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
) -> None:
    """Validate one physical append-only attempt, including its artifact."""

    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        weights_bundle_sha256=str(attempt.get("weights_bundle_sha256")),
        valid_for_metrics=status == "ok",
    )
    expected_keys = set(expected) | {"status", "completed_at"}
    if status == "ok":
        expected_keys |= _OK_RESULT_FIELDS
    else:
        expected_keys |= {"error_type", "error", "traceback"}
    if set(attempt) != expected_keys:
        raise ValueError(
            "result attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, expected_value in expected.items():
        if attempt.get(key) != expected_value:
            raise ValueError(f"result attempt field {key} drifted")
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("result attempt completed_at is invalid")
    _reject_unsupported_claims(attempt, "result attempt")
    if status == "error":
        if (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt["traceback"]
        ):
            raise ValueError("error result payload is invalid")
        return
    sample_id = str(input_row["sample_id"])
    _validate_score_payload(attempt, sample_id=sample_id)
    expected_preprocess = legacy.compute_preprocess_geometry(
        int(input_row["width"]),
        int(input_row["height"]),
    )
    if attempt.get("preprocess") != expected_preprocess:
        raise ValueError(f"{sample_id} preprocessing record changed")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _finite_number(attempt.get(field), f"{sample_id} {field}") < 0.0:
            raise ValueError(f"{sample_id} {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError(f"{sample_id} peak memory is invalid")
    _validate_descriptor_artifact(
        attempt,
        sample_id=sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
    source: Mapping[str, Any],
    weights: Mapping[str, Any],
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    run_dir: Path,
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    descriptor_dir: Path,
) -> dict[str, Any]:
    """Build the exact immutable config bound by the manifest fingerprint."""

    run_id = _valid_run_id(run_id)
    validate_runtime_contract(runtime, label="immutable runtime")
    if source.get("commit") != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError("source audit commit changed")
    if (
        weights.get("bundle_sha256") != EXPECTED_WEIGHTS_BUNDLE_SHA256
        or weights.get("release_tag") != legacy.RELEASE_TAG
    ):
        raise ValueError("weight audit identity changed")
    immutable = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": MODEL_CONTRACT,
        "source_tag_drift": legacy.SOURCE_TAG_DRIFT,
        "paper_release_drift": legacy.PAPER_RELEASE_DRIFT,
        "preprocess": PREPROCESS_CONTRACT,
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "source": dict(source),
        "weights": dict(weights),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "outputs": {
            "run_dir": repo_relative(run_dir, repo_root),
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "descriptor_dir": repo_relative(descriptor_dir, repo_root),
        },
    }
    if set(immutable) != IMMUTABLE_CONFIG_KEYS:
        raise AssertionError("internal immutable FSD config key set drifted")
    _reject_unsupported_claims(immutable, "immutable config")
    return immutable


def _ensure_repo_child(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    require_directory: bool = False,
) -> Path:
    """Resolve a repository child while rejecting every symlink component."""

    root = repo_root.resolve()
    raw = path if path.is_absolute() else root / path
    absolute = Path(os.path.abspath(raw))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if require_directory and not absolute.is_dir():
        raise FileNotFoundError(f"missing {label}: {absolute}")
    return absolute


def _prepare_output_directories(
    *,
    repo_root: Path,
    run_dir: Path,
    descriptor_root: Path,
    resume: bool,
) -> Path:
    run_dir = _ensure_repo_child(
        run_dir,
        repo_root=repo_root,
        label="FSD run directory",
    )
    descriptor_root = _ensure_repo_child(
        descriptor_root,
        repo_root=repo_root,
        label="FSD descriptor artifact root",
    )
    descriptor_dir = descriptor_root / "raw_descriptors"
    if not resume:
        if run_dir.exists() and (
            not run_dir.is_dir() or any(run_dir.iterdir())
        ):
            raise FileExistsError(
                f"run directory is non-empty; pass --resume: {run_dir}"
            )
        if descriptor_root.exists() and (
            not descriptor_root.is_dir() or any(descriptor_root.iterdir())
        ):
            raise FileExistsError(
                "descriptor artifact root is non-empty; pass --resume: "
                f"{descriptor_root}"
            )
    else:
        if not run_dir.is_dir() or not descriptor_dir.is_dir():
            raise FileNotFoundError(
                "resume requires the run and raw-descriptor directories"
            )
        root_entries = list(descriptor_root.iterdir())
        if (
            len(root_entries) != 1
            or root_entries[0].name != "raw_descriptors"
            or root_entries[0].is_symlink()
            or not root_entries[0].is_dir()
        ):
            raise ValueError("resume descriptor artifact root inventory changed")
    run_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_child(
        descriptor_dir,
        repo_root=repo_root,
        label="FSD raw descriptor directory",
        require_directory=True,
    )
    return descriptor_dir


def _validate_descriptor_inventory(
    *,
    descriptor_dir: Path,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
) -> int:
    descriptor_dir = _ensure_repo_child(
        descriptor_dir,
        repo_root=repo_root,
        label="FSD raw descriptor directory",
        require_directory=True,
    )
    entries = list(descriptor_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"descriptor inventory contains unsafe entry {entry.name}"
            )
    expected = {
        f"{sample_id}.npy"
        for sample_id, row in latest_by_sample_id.items()
        if row.get("status") == "ok"
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "FSD descriptor inventory mismatch: "
            f"missing={sorted(expected - actual)[:1]}, "
            f"extra={sorted(actual - expected)[:1]}"
        )
    for sample_id, row in latest_by_sample_id.items():
        if row.get("status") == "ok":
            _validate_descriptor_artifact(
                row,
                sample_id=sample_id,
                repo_root=repo_root,
                run_id=run_id,
            )
    return len(actual)


def _descriptor_record(
    *,
    descriptor: np.ndarray,
    descriptor_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(descriptor, np.ndarray)
        or descriptor.shape != (legacy.FSD_DIMENSION,)
        or descriptor.dtype != np.float64
        or not descriptor.flags.c_contiguous
        or not np.isfinite(descriptor).all()
    ):
        raise ValueError("official FSD descriptor array contract changed")
    payload = descriptor_path.read_bytes()
    if payload != _npy_bytes(descriptor):
        raise ValueError("persisted descriptor is not canonical NumPy bytes")
    return {
        "relative_path": repo_relative(descriptor_path, repo_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "array_sha256": _array_sha256(descriptor),
        "dtype": "float64",
        "shape": [legacy.FSD_DIMENSION],
        "nbytes": int(descriptor.nbytes),
        "finite": True,
        "semantics": "official_compute_fsd_before_released_transforms",
    }


def _build_ok_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    weights_bundle_sha256: str,
    descriptor_dir: Path,
    processed: Mapping[str, Any],
    descriptor: np.ndarray,
    preprocess: Mapping[str, Any],
    preprocess_latency_ms: float,
    latency_ms: float,
    peak_cuda_memory_bytes: int | None,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    descriptor_path = descriptor_dir / f"{sample_id}.npy"
    legacy._atomic_save_npy(descriptor_path, descriptor)
    descriptor_record = _descriptor_record(
        descriptor=descriptor,
        descriptor_path=descriptor_path,
        repo_root=repo_root,
    )
    raw = processed["raw_likelihood"]
    z_score = processed["released_z_score"]
    ai_score = processed["ai_score"]
    decision = processed["classification_decision"]
    relative = descriptor_record["relative_path"]
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            weights_bundle_sha256=weights_bundle_sha256,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "completed_at": utc_now(),
        "preprocess": dict(preprocess),
        "preprocess_latency_ms": float(preprocess_latency_ms),
        "descriptor": descriptor_record,
        "raw_descriptor_path": relative,
        "raw_descriptor_sha256": descriptor_record["sha256"],
        "raw_descriptor_array_sha256": descriptor_record["array_sha256"],
        "raw_descriptor_shape": [legacy.FSD_DIMENSION],
        "raw_descriptor_dtype": "float64",
        "raw_descriptor_nbytes": descriptor_record["nbytes"],
        "raw_descriptor_semantics": descriptor_record["semantics"],
        "artifact_paths": {"raw_descriptor_npy": relative},
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "ai_score": ai_score,
        "score": ai_score,
        "score_semantics": "negative_released_FSD_z_score",
        "released_is_fake": processed["released_is_fake"],
        "released_threshold": legacy.RELEASED_Z_THRESHOLD,
        "released_threshold_operator": legacy.RELEASED_THRESHOLD_OPERATOR,
        "classification_decision": decision,
        "classification_threshold": legacy.AI_SCORE_THRESHOLD,
        "classification_threshold_operator": legacy.THRESHOLD_OPERATOR,
        "classification": {
            "score": ai_score,
            "raw_likelihood": raw,
            "released_z_score": z_score,
            "decision": decision,
            "threshold": legacy.AI_SCORE_THRESHOLD,
            "threshold_operator": legacy.THRESHOLD_OPERATOR,
            "semantics": "higher_is_more_AI_negative_released_z",
        },
        "t1": {
            "score": ai_score,
            "raw_likelihood": raw,
            "released_z_score": z_score,
            "decision": decision,
            "threshold": legacy.AI_SCORE_THRESHOLD,
            "threshold_operator": legacy.THRESHOLD_OPERATOR,
            "policy": "released_FSD_whole_image_score_sign_inverted",
        },
        "manual_replay": processed["manual_replay"],
        "latency_ms": float(latency_ms),
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    return result


def _build_error_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    weights_bundle_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            weights_bundle_sha256=weights_bundle_sha256,
            valid_for_metrics=False,
        ),
        "status": "error",
        "completed_at": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


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
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(row, dict):
                    raise ValueError(f"{row_label} is not a JSON object")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


def _validate_preflight_report(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> None:
    expected_keys = {
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
    if set(report) != expected_keys:
        raise ValueError("CPU preflight report key set changed")
    if (
        report.get("schema_version") != CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("weights") != weights
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("CPU preflight report/provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    golden = report.get("golden")
    expected_golden_keys = {
        "sample_id",
        "input_path",
        "image_sha256",
        "input_width",
        "input_height",
        "preprocess",
        "descriptor_file_sha256",
        "descriptor_file_bytes",
        "descriptor_array_sha256",
        "descriptor_shape",
        "descriptor_dtype",
        "descriptor_nbytes",
        "raw_likelihood",
        "released_z_score",
        "ai_score",
        "classification_decision",
        "released_is_fake",
        "full_image_forward",
        "compute_fsd_calls",
        "repeat_descriptor_file_sha256",
        "repeat_descriptor_file_bytes",
        "repeat_descriptor_array_sha256",
        "repeat_raw_likelihood",
        "repeat_released_z_score",
        "repeat_ai_score",
        "repeat_classification_decision",
        "repeat_released_is_fake",
        "repeat_full_image_forward",
        "repeat_compute_fsd_calls",
        "repeat_byte_exact",
    }
    if not isinstance(golden, Mapping) or set(golden) != expected_golden_keys:
        raise ValueError("CPU golden key set changed")
    expected_values = {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": legacy.compute_preprocess_geometry(1800, 1350),
        "descriptor_file_sha256": CPU_GOLDEN_DESCRIPTOR_FILE_SHA256,
        "descriptor_file_bytes": 7808,
        "descriptor_array_sha256": CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256,
        "descriptor_shape": [legacy.FSD_DIMENSION],
        "descriptor_dtype": "float64",
        "descriptor_nbytes": 7680,
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
    if dict(golden) != expected_values:
        raise ValueError("CPU golden exact two-forward evidence changed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Explicit pinned v1.2.0 asset directory; no downloader is used.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke", "single", "preflight"),
        default="formal",
    )
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--device",
        help="explicit cpu or cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute one append-only FSD Balanced250 run."""

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(Path(args.source_root), repo_root)
    weights_dir = _anchored(Path(args.weights_dir), repo_root)
    mode = str(args.mode)
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed != DEFAULT_SEED
    ):
        raise ValueError(f"FSD seed must be exactly {DEFAULT_SEED}")
    if mode == "preflight":
        if (
            bool(args.resume)
            or bool(args.fail_fast)
            or args.run_id is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError(
                "preflight accepts no run/selection/resume/CUDA options"
            )
        report = run_cpu_preflight(
            repo_root=repo_root,
            source_root=source_root,
            weights_dir=weights_dir,
        )
        source, weights = legacy._verify_static_contract(
            source_root=source_root,
            weights_dir=weights_dir,
        )
        _validate_preflight_report(
            report,
            source=source,
            weights=weights,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if mode != "formal" and args.run_id is None:
        raise ValueError("smoke and single modes require explicit --run-id")
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _anchored(
        Path(args.dataset_manifest),
        repo_root,
    )
    results_root = _ensure_repo_child(
        Path(args.results_dir),
        repo_root=repo_root,
        label="FSD results root",
    )
    artifacts_root = _ensure_repo_child(
        Path(args.artifacts_dir),
        repo_root=repo_root,
        label="FSD artifacts root",
    )
    if repo_relative(artifacts_root, repo_root) != DEFAULT_ARTIFACTS_DIR.as_posix():
        raise ValueError(
            f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}"
        )
    run_dir = results_root / run_id
    descriptor_root = artifacts_root / run_id
    descriptor_dir = descriptor_root / "raw_descriptors"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    selection_spec, selected = select_mode_inputs(
        release,
        mode=mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    )

    # This exact CPU gate deliberately precedes requested accelerator setup.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        weights_dir=weights_dir,
    )
    source, weights = legacy._verify_static_contract(
        source_root=source_root,
        weights_dir=weights_dir,
    )
    _validate_preflight_report(
        cpu_preflight,
        source=source,
        weights=weights,
    )
    device, runtime = configure_runtime(device_text, seed=DEFAULT_SEED)
    adapter_sources = adapter_source_contract(repo_root)

    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        adapter_sources=adapter_sources,
        source=source,
        weights=weights,
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        run_dir=run_dir,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        descriptor_dir=descriptor_dir,
    )
    fingerprint = _fingerprint(immutable)
    descriptor_dir = _prepare_output_directories(
        repo_root=repo_root,
        run_dir=run_dir,
        descriptor_root=descriptor_root,
        resume=bool(args.resume),
    )

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )
        prior_manifest = _load_json_strict(manifest_path, "prior manifest")
        if (
            prior_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
            or prior_manifest.get("run_id") != run_id
            or prior_manifest.get("fingerprint") != fingerprint
            or prior_manifest.get("immutable") != immutable
        ):
            raise ValueError("resume run manifest fingerprint/config drifted")
        if _read_jsonl_strict(expected_path, "expected inputs") != selected:
            raise ValueError("resume expected-input snapshot drifted")
        started_at = prior_manifest.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError("resume manifest started_at is invalid")
    else:
        atomic_write_jsonl(expected_path, selected)
        started_at = utc_now()

    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "fingerprint": fingerprint,
        "immutable": immutable,
        "dataset": {
            "contract": dataset_contract.as_dict(),
            "manifest_path": repo_relative(
                dataset_manifest_path,
                repo_root,
            ),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        },
        "outputs": dict(immutable["outputs"]),
    }
    atomic_write_json(manifest_path, manifest)

    physical_before = (
        _read_jsonl_strict(results_path, "prior physical results")
        if results_path.is_file()
        else []
    )
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_before:
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("prior result is outside the selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    _validate_descriptor_inventory(
        descriptor_dir=descriptor_dir,
        latest_by_sample_id=latest_before.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )

    pending_ids = set(latest_before.pending_sample_ids(retry_errors=True))
    new_successes = 0
    resume_skips = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    detector = None
    try:
        if pending_ids:
            detector, loaded_device, load_audit = legacy.load_detector(
                source_root=source_root,
                weights_dir=weights_dir,
                device_name=str(device),
            )
            if loaded_device != device:
                raise ValueError("loaded detector device differs from runtime")
            if (
                load_audit.get("source") != source
                or load_audit.get("weights") != weights
            ):
                raise ValueError("loaded detector provenance differs")
        else:
            loaded_device = device

        for index, input_row in enumerate(selected, start=1):
            sample_id = str(input_row["sample_id"])
            if sample_id not in pending_ids:
                resume_skips += 1
                print(
                    f"[{index}/{len(selected)}] resume {sample_id}",
                    flush=True,
                )
                continue
            descriptor_path = descriptor_dir / f"{sample_id}.npy"
            try:
                input_path = _safe_repo_file(
                    str(input_row["canonical_path"]),
                    repo_root=repo_root,
                    label=f"{sample_id} canonical input",
                )
                if sha256_file(input_path) != input_row["canonical_sha256"]:
                    raise ValueError(
                        f"{sample_id} canonical input SHA-256 changed"
                    )
                preprocess_started = time.perf_counter()
                preprocess = legacy.compute_preprocess_geometry(
                    int(input_row["width"]),
                    int(input_row["height"]),
                )
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if detector is None:
                    raise RuntimeError("FSD detector was not loaded")
                processed, descriptor, peak, latency = legacy.infer_one(
                    detector,
                    loaded_device,
                    input_path,
                )
                result = _build_ok_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    weights_bundle_sha256=EXPECTED_WEIGHTS_BUNDLE_SHA256,
                    descriptor_dir=descriptor_dir,
                    processed=processed,
                    descriptor=descriptor,
                    preprocess=preprocess,
                    preprocess_latency_ms=preprocess_latency_ms,
                    latency_ms=latency,
                    peak_cuda_memory_bytes=peak,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"ai_score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                if descriptor_path.exists():
                    if descriptor_path.is_symlink() or not descriptor_path.is_file():
                        raise ValueError(
                            f"unsafe failed descriptor path: {descriptor_path}"
                        ) from error
                    descriptor_path.unlink()
                result = _build_error_result(
                    input_row=input_row,
                    repo_root=repo_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    weights_bundle_sha256=EXPECTED_WEIGHTS_BUNDLE_SHA256,
                    error=error,
                )
                append_jsonl(results_path, result)
                new_errors += 1
                print(
                    f"[{index}/{len(selected)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
                    break
            finally:
                gc.collect()
                if device.type == "cuda":
                    __import__("torch").cuda.empty_cache()
    finally:
        if detector is not None:
            del detector
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()

    physical_results = _read_jsonl_strict(
        results_path,
        "physical results",
    )
    for attempt in physical_results:
        sample_id = str(attempt.get("sample_id"))
        if sample_id not in inputs_by_id:
            raise ValueError("result is outside the selection")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    coverage = summarize_coverage(latest)
    descriptor_files = _validate_descriptor_inventory(
        descriptor_dir=descriptor_dir,
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
        run_id=run_id,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_fsd_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "score_spec": SCORE_SPEC.as_dict(),
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": coverage.as_dict(),
        "generated_at": utc_now(),
    }
    _reject_unsupported_claims(summary, "runtime summary")
    atomic_write_json(summary_path, summary)

    manifest["status"] = summary["status"]
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": new_successes,
        "resume_skips": resume_skips,
        "new_errors": new_errors,
        "physical_result_rows": len(physical_results),
        "latest_result_rows": len(latest.latest_by_sample_id),
        "superseded_attempts": latest.superseded_attempts,
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "descriptor_files": descriptor_files,
        }
    )
    _reject_unsupported_claims(manifest, "run manifest")
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "mode": mode,
                "coverage": coverage.as_dict(),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError("FSD fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
