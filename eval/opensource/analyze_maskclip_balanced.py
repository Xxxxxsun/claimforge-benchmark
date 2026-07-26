#!/usr/bin/env python3
"""Audit, summarize, compare, and replay MaskCLIP Balanced250 runs.

The MaskCLIP adapter produces one native T1 score and a 512x512 dense map for
every selected image.  Native dense artifacts exist only for the four T2
applicable conditions (real plus the three local insertions).  This analyzer
rebuilds the frozen selection, validates every append-only physical attempt,
checks the complete latest-attempt coverage, independently reconstructs every
native map and threshold mask, recomputes the shared T1/T2 statistics, and can
freshly reload the official model for a complete replay.

No pair is inferred from ``task_id`` or a legacy Mouse ``pair_rank``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from eval.opensource import run_maskclip as legacy_runner
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
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
    Capability,
    CanonicalRelease,
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

try:
    from eval.opensource import run_maskclip_balanced as balanced_runner
except ImportError:  # Allows isolated helper tests while the paired runner lands.
    balanced_runner = None  # type: ignore[assignment]

try:
    from eval.opensource.balanced250_localization_metrics import (
        summarize_balanced250_t2,
    )
except ImportError:  # Allows isolated helper tests while the shared T2 API lands.
    summarize_balanced250_t2 = None  # type: ignore[assignment]


AUDIT_SCHEMA_VERSION = "maskclip_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "maskclip_balanced_smoke_comparison_v2"
T1_METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
T2_METRICS_SCHEMA_VERSION = "balanced250_t2_summary_v1"
DEFAULT_RESULTS_DIR = Path("results/opensource/maskclip")
DEFAULT_FORMAL_RUN_ID = "maskclip_sd15_balanced250_v1_full1775_20260726"
DEFAULT_SOURCE_ROOT = legacy_runner.DEFAULT_OPENSDI_ROOT
DEFAULT_CHECKPOINT = legacy_runner.DEFAULT_CHECKPOINT
DEFAULT_CLIP_CHECKPOINT = legacy_runner.DEFAULT_CLIP_CHECKPOINT
MODEL_SIZE = legacy_runner.MODEL_INPUT_SIZE
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
LOGIT_ABS_TOLERANCE = 1e-5
SCORE_ABS_TOLERANCE = 1e-7
SOFTMAX_ABS_TOLERANCE = 1e-6
MODEL_MAP_ABS_TOLERANCE = 1e-6

_T2_GT_KINDS = frozenset({"all_zero", "exact_diff"})
_FULLFRAME_GT_KIND = "not_applicable"
_ARTIFACT_PATH_FIELDS = frozenset(
    {
        "score_map_model_path",
        "score_map_native_path",
        "mask_path",
    }
)
_SMOKE_IGNORED_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "latency_ms",
        "peak_cuda_memory_bytes",
        *_ARTIFACT_PATH_FIELDS,
    }
)


@dataclass(frozen=True)
class DenseArtifacts:
    """Validated paths and immutable metadata for one successful result."""

    sample_id: str
    model_path: Path
    model_sha256: str
    model_bytes: int
    native_path: Path | None
    native_sha256: str | None
    native_bytes: int | None
    mask_path: Path | None
    mask_sha256: str | None
    mask_bytes: int | None
    t2_applicable: bool
    width: int
    height: int


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    fingerprint: str
    run_dir: Path
    artifact_root: Path
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
    artifacts: Mapping[str, DenseArtifacts]


def _require_runner() -> Any:
    if balanced_runner is None:
        raise RuntimeError("run_maskclip_balanced.py is not available")
    return balanced_runner


def _require_t2_metrics() -> Callable[..., dict[str, Any]]:
    if summarize_balanced250_t2 is None:
        raise RuntimeError("balanced250_localization_metrics.py is not available")
    return summarize_balanced250_t2


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
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} is not boolean")
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
                raise ValueError(f"{label}:{line_number} is invalid JSON") from error
            row = _require_mapping(value, f"{label}:{line_number}")
            if line != f"{stable_json(row)}\n":
                raise ValueError(f"{label}:{line_number} is not canonical JSONL")
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


def _valid_run_id(value: str) -> str:
    runner = _require_runner()
    validator = getattr(runner, "_valid_run_id", None)
    if callable(validator):
        return str(validator(value))
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
            for character in value
        )
        or value in (".", "..")
    ):
        raise ValueError("run-id contains unsafe characters")
    return value


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    valid = _valid_run_id(run_id)
    root = results_root.resolve()
    candidate = (root / valid).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("resolved run directory escapes results root") from error
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
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite")


def _verify_adapter_sources(value: Any, *, repo_root: Path) -> None:
    runner = _require_runner()
    sources = _require_mapping(value, "immutable.adapter_sources")
    expected_paths = set(getattr(runner, "ADAPTER_SOURCE_PATHS"))
    if set(sources) != expected_paths:
        missing = sorted(expected_paths - set(sources))
        extra = sorted(set(sources) - expected_paths)
        raise ValueError(
            "immutable.adapter_sources key set mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative, raw in sources.items():
        record = _require_mapping(raw, f"adapter source {relative}")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"adapter source {relative} metadata keys changed")
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


def _softmax_two(logits: Sequence[float]) -> tuple[float, float]:
    left, right = (float(value) for value in logits)
    maximum = max(left, right)
    exp_left = math.exp(left - maximum)
    exp_right = math.exp(right - maximum)
    denominator = exp_left + exp_right
    return exp_left / denominator, exp_right / denominator


def _validate_score_payload(row: Mapping[str, Any], *, sample_id: str) -> None:
    logits = _require_mapping(
        row.get("class_logits"),
        f"{sample_id} class_logits",
    )
    probabilities = _require_mapping(
        row.get("class_probabilities"),
        f"{sample_id} class_probabilities",
    )
    if set(logits) != {"real", "forged"}:
        raise ValueError(f"{sample_id} class_logits keys changed")
    if set(probabilities) != {"real", "forged"}:
        raise ValueError(f"{sample_id} class_probabilities keys changed")
    real_logit = _require_finite(logits.get("real"), f"{sample_id} real logit")
    forged_logit = _require_finite(
        logits.get("forged"),
        f"{sample_id} forged logit",
    )
    real_probability = _require_finite(
        probabilities.get("real"),
        f"{sample_id} real probability",
    )
    forged_probability = _require_finite(
        probabilities.get("forged"),
        f"{sample_id} forged probability",
    )
    if not (0.0 <= real_probability <= 1.0 and 0.0 <= forged_probability <= 1.0):
        raise ValueError(f"{sample_id} class probability falls outside [0, 1]")
    expected_real, expected_forged = _softmax_two((real_logit, forged_logit))
    if not math.isclose(
        real_probability,
        expected_real,
        rel_tol=0.0,
        abs_tol=SOFTMAX_ABS_TOLERANCE,
    ) or not math.isclose(
        forged_probability,
        expected_forged,
        rel_tol=0.0,
        abs_tol=SOFTMAX_ABS_TOLERANCE,
    ):
        raise ValueError(f"{sample_id} logits/softmax relationship changed")
    score = _require_finite(row.get("ai_score"), f"{sample_id} ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{sample_id} ai_score falls outside [0, 1]")
    if (
        row.get("probability") != score
        or row.get("score") != score
        or forged_probability != score
    ):
        raise ValueError(f"{sample_id} score aliases differ")
    margin = _require_finite(
        row.get("score_margin"),
        f"{sample_id} score_margin",
    )
    expected_margin = float(np.float32(forged_logit) - np.float32(real_logit))
    if margin != expected_margin:
        raise ValueError(f"{sample_id} score margin changed")
    if row.get("classification_threshold") != CLASSIFICATION_THRESHOLD:
        raise ValueError(f"{sample_id} classification threshold changed")
    if row.get("classification_threshold_operator") != ">=":
        raise ValueError(f"{sample_id} classification operator changed")
    decision = _require_bool(
        row.get("classification_decision"),
        f"{sample_id} classification_decision",
    )
    if decision is not (score >= CLASSIFICATION_THRESHOLD):
        raise ValueError(f"{sample_id} classification decision changed")
    if row.get("calibrated_probability") is not False:
        raise ValueError(f"{sample_id} score is incorrectly marked calibrated")
    if row.get("score_semantics") != ("softmax_probability_of_class_1_forged"):
        raise ValueError(f"{sample_id} score semantics changed")


def _validate_runtime_evidence(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    runtime = _require_mapping(value, label)
    base_keys = {
        "device",
        "python",
        "platform",
        "torch",
        "torchvision",
        "numpy",
        "opencv",
        "pillow",
        "imdlbenco",
        "timm",
        "seed",
        "precision",
        "batch_size",
        "autocast",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
    }
    device = _require_string(runtime.get("device"), f"{label}.device")
    cuda_device = device.startswith("cuda:")
    if device != "cpu":
        suffix = device.removeprefix("cuda:")
        if not cuda_device or not suffix.isdigit():
            raise ValueError(f"{label}.device must be cpu or explicit cuda:N")
    expected_keys = base_keys | ({"cuda"} if cuda_device else set())
    if set(runtime) != expected_keys:
        raise ValueError(f"{label} key set changed")
    for field in (
        "python",
        "platform",
        "torch",
        "numpy",
        "opencv",
        "pillow",
    ):
        _require_string(runtime.get(field), f"{label}.{field}")
    for field in ("torchvision", "imdlbenco", "timm"):
        version = runtime.get(field)
        if version is not None:
            _require_string(version, f"{label}.{field}")
    expected_values = {
        "seed": getattr(_require_runner(), "MODEL_SEED", 42),
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "matmul_allow_tf32": False,
    }
    for field, expected_value in expected_values.items():
        if runtime.get(field) != expected_value:
            raise ValueError(f"{label}.{field} changed")
    cudnn = _require_mapping(runtime.get("cudnn"), f"{label}.cudnn")
    expected_cudnn = {
        "benchmark": False,
        "deterministic": True,
        "allow_tf32": False,
    }
    if cudnn != expected_cudnn:
        raise ValueError(f"{label}.cudnn deterministic contract changed")
    if cuda_device:
        cuda = _require_mapping(runtime.get("cuda"), f"{label}.cuda")
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
        device_index = _require_nonnegative_int(
            cuda.get("device_index"),
            f"{label}.cuda.device_index",
        )
        if device_index != int(device.removeprefix("cuda:")):
            raise ValueError(f"{label}.cuda.device_index disagrees with device")
        memory = _require_nonnegative_int(
            cuda.get("total_memory_bytes"),
            f"{label}.cuda.total_memory_bytes",
        )
        if memory == 0:
            raise ValueError(f"{label}.cuda.total_memory_bytes is zero")
        capability = cuda.get("capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{label}.cuda.capability changed")
    _reject_nonfinite_numbers(runtime, label)
    return runtime


def _independent_restore_native(
    model_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    source = np.asarray(model_map)
    if source.shape != (MODEL_SIZE, MODEL_SIZE) or source.dtype != np.float32:
        raise ValueError("model score map has invalid shape or dtype")
    restored = cv2.resize(
        source,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def _independent_resize_target(target: np.ndarray) -> np.ndarray:
    import cv2

    truth = np.asarray(target, dtype=bool)
    if truth.ndim != 2 or not truth.size:
        raise ValueError("ground-truth mask is not a non-empty 2-D array")
    resized = cv2.resize(
        truth.astype(np.uint8),
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


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
        raise ValueError("score/target shape mismatch")
    if (
        scores.size == 0
        or not np.isfinite(scores).all()
        or float(scores.min()) < 0.0
        or float(scores.max()) > 1.0
    ):
        raise ValueError("score map values are invalid")
    prediction = scores >= MASK_THRESHOLD
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    pixel_ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        pixel_ap = float(average_precision_score(truth.reshape(-1), scores.reshape(-1)))
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
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
        "pixel_ap": pixel_ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


def _assert_nested_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} mapping keys changed")
        for key, expected_value in expected.items():
            _assert_nested_close(
                actual[key],
                expected_value,
                label=f"{label}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list changed")
        for index, expected_value in enumerate(expected):
            _assert_nested_close(
                actual[index],
                expected_value,
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


def _load_validated_npy(
    *,
    row: Mapping[str, Any],
    prefix: str,
    expected_path: Path,
    expected_shape: tuple[int, int],
    repo_root: Path,
    sample_id: str,
) -> tuple[Path, str, int, np.ndarray]:
    path = _safe_repo_path(
        row.get(f"{prefix}_path"),
        repo_root=repo_root,
        label=f"{sample_id} {prefix} path",
    )
    if path != expected_path.resolve():
        raise ValueError(f"{sample_id} {prefix} path is not canonical")
    expected_sha = _require_sha256(
        row.get(f"{prefix}_sha256"),
        f"{sample_id} {prefix} SHA-256",
    )
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{sample_id} {prefix} artifact hash mismatch")
    file_bytes = path.stat().st_size
    if row.get(f"{prefix}_bytes") != file_bytes:
        raise ValueError(f"{sample_id} {prefix} byte-size metadata mismatch")
    if row.get(f"{prefix}_shape") != list(expected_shape):
        raise ValueError(f"{sample_id} {prefix} shape metadata changed")
    if row.get(f"{prefix}_dtype") != "float32":
        raise ValueError(f"{sample_id} {prefix} dtype metadata changed")
    _require_string(
        row.get(f"{prefix}_semantics"),
        f"{sample_id} {prefix} semantics",
    )
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{sample_id} cannot load {prefix}") from error
    if array.shape != expected_shape:
        raise ValueError(f"{sample_id} {prefix} array shape changed")
    if array.dtype != np.float32:
        raise ValueError(f"{sample_id} {prefix} array dtype changed")
    if not array.flags.c_contiguous:
        raise ValueError(f"{sample_id} {prefix} is not C-contiguous")
    if not np.isfinite(array).all():
        raise ValueError(f"{sample_id} {prefix} contains non-finite values")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{sample_id} {prefix} falls outside [0, 1]")
    return path, expected_sha, file_bytes, array


def _load_validated_mask(
    *,
    row: Mapping[str, Any],
    expected_path: Path,
    expected_shape: tuple[int, int],
    repo_root: Path,
    sample_id: str,
) -> tuple[Path, str, int, np.ndarray]:
    path = _safe_repo_path(
        row.get("mask_path"),
        repo_root=repo_root,
        label=f"{sample_id} mask path",
    )
    if path != expected_path.resolve():
        raise ValueError(f"{sample_id} mask path is not canonical")
    expected_sha = _require_sha256(
        row.get("mask_sha256"),
        f"{sample_id} mask SHA-256",
    )
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{sample_id} mask artifact hash mismatch")
    file_bytes = path.stat().st_size
    if row.get("mask_bytes") != file_bytes:
        raise ValueError(f"{sample_id} mask byte-size metadata mismatch")
    if row.get("mask_shape") != list(expected_shape):
        raise ValueError(f"{sample_id} mask shape metadata changed")
    if row.get("mask_dtype") != "uint8":
        raise ValueError(f"{sample_id} mask dtype metadata changed")
    _require_string(row.get("mask_semantics"), f"{sample_id} mask semantics")
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG":
                raise ValueError(f"{sample_id} mask is not PNG")
            if opened.mode != "L":
                raise ValueError(f"{sample_id} mask is not L mode")
            pixels = np.asarray(opened, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"{sample_id} mask cannot be decoded") from error
    if pixels.shape != expected_shape:
        raise ValueError(f"{sample_id} mask dimensions changed")
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} mask is not binary 0/255")
    return path, expected_sha, file_bytes, pixels


def _nullable_artifact_fields(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_path",
        f"{prefix}_sha256",
        f"{prefix}_bytes",
        f"{prefix}_shape",
        f"{prefix}_dtype",
        f"{prefix}_semantics",
    )


def _validate_t2_declaration(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> bool:
    sample_id = str(expected["sample_id"])
    gt_kind = expected.get("gt_mask_kind")
    applicable = gt_kind in _T2_GT_KINDS
    if row.get("t2_applicable") is not applicable:
        raise ValueError(f"{sample_id} t2_applicable changed")
    task_scope = _require_mapping(
        row.get("task_scope"),
        f"{sample_id} task_scope",
    )
    required = {
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "native_dense_output": True,
        "model_512_output_role": (
            "t2_and_diagnostic" if applicable else "diagnostic_only"
        ),
    }
    for field, expected_value in required.items():
        if task_scope.get(field) != expected_value:
            raise ValueError(f"{sample_id} task_scope.{field} changed")
    if set(task_scope) != set(required):
        raise ValueError(f"{sample_id} task_scope key set changed")
    semantics = _require_string(
        row.get("t2_target_semantics"),
        f"{sample_id} t2_target_semantics",
    )
    expected_semantics = {
        "not_applicable": "not_applicable_fullframe",
        "all_zero": "all_zero_real_false_positive_area",
        "exact_diff": "exact_diff_local_insertion",
    }.get(str(gt_kind))
    if expected_semantics is None:
        raise ValueError(f"{sample_id} has unsupported GT kind {gt_kind!r}")
    if semantics != expected_semantics:
        raise ValueError(f"{sample_id} T2 target semantics changed")
    return applicable


def _validate_artifact_row(
    *,
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> DenseArtifacts:
    sample_id = str(expected["sample_id"])
    if row.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id} result identity changed")
    _validate_score_payload(row, sample_id=sample_id)
    applicable = _validate_t2_declaration(row, expected=expected)
    if row.get("mask_threshold") != MASK_THRESHOLD:
        raise ValueError(f"{sample_id} mask threshold changed")
    if row.get("mask_threshold_operator") != ">=":
        raise ValueError(f"{sample_id} mask threshold operator changed")

    width = int(expected["width"])
    height = int(expected["height"])
    model_path, model_sha, model_bytes, model_map = _load_validated_npy(
        row=row,
        prefix="score_map_model",
        expected_path=(artifact_root / "score_maps_model_512" / f"{sample_id}.npy"),
        expected_shape=(MODEL_SIZE, MODEL_SIZE),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    if row.get("score_map_model_semantics") != ("released_sigmoid_forged_probability"):
        raise ValueError(f"{sample_id} model-map semantics changed")
    native_path: Path | None = None
    native_sha: str | None = None
    native_bytes: int | None = None
    mask_path: Path | None = None
    mask_sha: str | None = None
    mask_bytes: int | None = None

    if not applicable:
        for field in (
            *_nullable_artifact_fields("score_map_native"),
            *_nullable_artifact_fields("mask"),
        ):
            if row.get(field) is not None:
                raise ValueError(f"{sample_id} fullframe field {field} must be null")
        if row.get("localization") is not None:
            raise ValueError(f"{sample_id} fullframe localization must be null")
        return DenseArtifacts(
            sample_id=sample_id,
            model_path=model_path,
            model_sha256=model_sha,
            model_bytes=model_bytes,
            native_path=None,
            native_sha256=None,
            native_bytes=None,
            mask_path=None,
            mask_sha256=None,
            mask_bytes=None,
            t2_applicable=False,
            width=width,
            height=height,
        )

    native_path, native_sha, native_bytes, native_map = _load_validated_npy(
        row=row,
        prefix="score_map_native",
        expected_path=(artifact_root / "score_maps_native" / f"{sample_id}.npy"),
        expected_shape=(height, width),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    if row.get("score_map_native_semantics") != (
        "model_512_probability_map_restored_opencv_inter_linear"
    ):
        raise ValueError(f"{sample_id} native-map semantics changed")
    reconstructed = _independent_restore_native(
        model_map,
        width=width,
        height=height,
    )
    if not np.array_equal(reconstructed, native_map):
        difference = float(
            np.max(
                np.abs(
                    reconstructed.astype(np.float64)
                    - np.asarray(native_map, dtype=np.float64)
                )
            )
        )
        raise ValueError(
            f"{sample_id} saved native map is not exact INTER_LINEAR "
            f"restoration (max difference {difference})"
        )

    mask_path, mask_sha, mask_bytes, mask_pixels = _load_validated_mask(
        row=row,
        expected_path=(artifact_root / "masks_native" / f"{sample_id}.png"),
        expected_shape=(height, width),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    if row.get("mask_semantics") != (
        "native_probability_map_ge_0_5_encoded_L_0_or_255"
    ):
        raise ValueError(f"{sample_id} mask semantics changed")
    expected_mask = np.asarray(native_map) >= MASK_THRESHOLD
    if not np.array_equal(mask_pixels == 255, expected_mask):
        raise ValueError(f"{sample_id} mask is not native score map >= 0.5")

    target_native = load_ground_truth(expected, repo_root)
    if target_native is None:
        raise ValueError(f"{sample_id} applicable row has no ground truth")
    target_model = _independent_resize_target(target_native)
    include_ap = expected.get("gt_mask_kind") == "exact_diff"
    expected_localization = {
        "model_512": _independent_pixel_metrics(
            model_map,
            target_model,
            include_ap=include_ap,
        ),
        "native": _independent_pixel_metrics(
            native_map,
            target_native,
            include_ap=include_ap,
        ),
    }
    if expected.get("gt_mask_kind") == "all_zero":
        if (
            expected_localization["model_512"]["pixel_ap"] is not None
            or expected_localization["native"]["pixel_ap"] is not None
        ):
            raise ValueError(f"{sample_id} real row received pixel AP")
    _assert_nested_close(
        row.get("localization"),
        expected_localization,
        label=f"{sample_id} localization",
    )
    return DenseArtifacts(
        sample_id=sample_id,
        model_path=model_path,
        model_sha256=model_sha,
        model_bytes=model_bytes,
        native_path=native_path,
        native_sha256=native_sha,
        native_bytes=native_bytes,
        mask_path=mask_path,
        mask_sha256=mask_sha,
        mask_bytes=mask_bytes,
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
        raise FileNotFoundError(f"missing or unsafe {label}: {directory}")
    entries = list(directory.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError(f"{label} contains a non-regular entry")
    actual = {entry.name for entry in entries}
    if actual != expected_names:
        missing = sorted(expected_names - actual)
        extra = sorted(actual - expected_names)
        raise ValueError(
            f"{label} inventory mismatch: " f"missing={missing[:3]}, extra={extra[:3]}"
        )


def validate_artifact_inventory(
    *,
    selected: Sequence[Mapping[str, Any]],
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, DenseArtifacts]:
    if len(selected) != len(latest_results):
        raise ValueError("selected/result artifact coverage differs")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(f"missing or unsafe artifact root: {artifact_root}")
    expected_root_entries = {
        "score_maps_model_512",
        "score_maps_native",
        "masks_native",
    }
    root_entries = list(artifact_root.iterdir())
    if {entry.name for entry in root_entries} != expected_root_entries or any(
        not entry.is_dir() or entry.is_symlink() for entry in root_entries
    ):
        raise ValueError(
            "artifact-root top-level inventory is not the exact three "
            "safe artifact directories"
        )
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    if len(expected_by_id) != len(selected):
        raise ValueError("selection contains duplicate sample_id")
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for row in latest_results:
        sample_id = _require_string(row.get("sample_id"), "result sample_id")
        if sample_id in result_by_id:
            raise ValueError(f"results contain duplicate sample_id {sample_id}")
        result_by_id[sample_id] = row
    if set(result_by_id) != set(expected_by_id):
        raise ValueError("result artifact coverage differs from selection")
    applicable_ids = {
        sample_id
        for sample_id, row in expected_by_id.items()
        if row.get("gt_mask_kind") in _T2_GT_KINDS
    }
    all_names = {f"{sample_id}.npy" for sample_id in expected_by_id}
    applicable_npy = {f"{sample_id}.npy" for sample_id in applicable_ids}
    applicable_png = {f"{sample_id}.png" for sample_id in applicable_ids}
    _exact_directory_inventory(
        artifact_root / "score_maps_model_512",
        expected_names=all_names,
        label="model-map directory",
    )
    _exact_directory_inventory(
        artifact_root / "score_maps_native",
        expected_names=applicable_npy,
        label="native-map directory",
    )
    _exact_directory_inventory(
        artifact_root / "masks_native",
        expected_names=applicable_png,
        label="mask directory",
    )
    artifacts: dict[str, DenseArtifacts] = {}
    for expected in selected:
        sample_id = str(expected["sample_id"])
        artifacts[sample_id] = _validate_artifact_row(
            row=result_by_id[sample_id],
            expected=expected,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
    return artifacts


def _runner_value(name: str, fallback: Any = None) -> Any:
    runner = _require_runner()
    return getattr(runner, name, fallback)


def _score_spec() -> ScoreSpec:
    value = _runner_value(
        "SCORE_SPEC",
        ScoreSpec(
            key="ai_score",
            direction="higher_means_fake",
            fixed_threshold=CLASSIFICATION_THRESHOLD,
            threshold_operator=">=",
        ),
    )
    if not isinstance(value, ScoreSpec):
        raise ValueError("MaskCLIP runner SCORE_SPEC has the wrong type")
    expected = {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": ">=",
    }
    if value.as_dict() != expected:
        raise ValueError("MaskCLIP runner SCORE_SPEC changed")
    return value


def _selection_for_mode(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
) -> tuple[SelectionSpec, tuple[dict[str, Any], ...]]:
    runner = _require_runner()
    selector = getattr(runner, "select_mode_inputs", None)
    if callable(selector):
        spec, rows = selector(
            release,
            mode=mode,
            per_condition_limit=per_condition_limit,
            sample_id=None,
        )
    else:
        if mode == "formal":
            spec = SelectionSpec(capability=Capability.LOCAL_T1_T2)
        elif mode == "smoke":
            spec = SelectionSpec(
                capability=Capability.LOCAL_T1_T2,
                per_condition_limit=per_condition_limit,
            )
        else:
            raise ValueError(f"unsupported mode {mode!r}")
        rows = select_inputs(release, spec)
    if not isinstance(spec, SelectionSpec):
        raise ValueError("runner returned an invalid selection spec")
    if spec.capability is not Capability.LOCAL_T1_T2:
        raise ValueError("MaskCLIP selection capability changed")
    return spec, tuple(dict(row) for row in rows)


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[CanonicalRelease, tuple[dict[str, Any], ...], RunDatasetContract]:
    raw_contract = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_binding = _require_mapping(
        raw_contract.get("release"),
        "dataset contract release",
    )
    manifest_path = _safe_repo_path(
        release_binding.get("manifest_path"),
        repo_root=repo_root,
        label="dataset manifest",
    )
    release = load_canonical_release(repo_root, manifest_path, verify_files=True)
    selection = _require_mapping(
        raw_contract.get("selection"),
        "dataset contract selection",
    )
    selection_spec = _require_mapping(
        selection.get("spec"),
        "dataset contract selection spec",
    )
    limit = selection_spec.get("per_condition_limit")
    if expected_mode == "formal" and limit is not None:
        raise ValueError("formal selection has a per-condition limit")
    if expected_mode == "smoke":
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("smoke selection limit is invalid")
    spec, selected = _selection_for_mode(
        release,
        mode=expected_mode,
        per_condition_limit=limit,
    )
    rebuilt = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=_score_spec(),
    )
    if rebuilt.as_dict() != raw_contract:
        raise ValueError("immutable dataset contract does not rebuild exactly")
    if immutable.get("mode") != expected_mode:
        raise ValueError(f"analyzer requires immutable.mode={expected_mode}")
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score_spec changed")
    if immutable.get("selected_rows_sha256") != _rows_sha256(selected):
        raise ValueError("immutable selected-row SHA-256 changed")
    persisted_ids = _require_sha256(
        immutable.get("selected_ids_sha256"),
        "immutable.selected_ids_sha256",
    )
    if persisted_ids != rebuilt.selection.selected_ids_sha256:
        raise ValueError("immutable selected-ID SHA-256 changed")
    expected_images = 1775 if expected_mode == "formal" else 7 * int(limit)
    if len(selected) != expected_images:
        raise ValueError(
            f"{expected_mode} selection has {len(selected)} images, "
            f"not {expected_images}"
        )
    return release, selected, rebuilt


def _validate_model_provenance(immutable: Mapping[str, Any]) -> None:
    runner = _require_runner()
    model = _require_mapping(immutable.get("model"), "immutable.model")
    expected_model = {
        "name": "MaskCLIP",
        "slug": "opensdi_maskclip_sd15",
        "architecture": getattr(
            runner,
            "MODEL_ARCHITECTURE",
            "OpenSDI_MaskCLIP_ViTL",
        ),
        "repository": legacy_runner.MODEL_REPO_URL,
        "source_commit": legacy_runner.MODEL_SOURCE_COMMIT,
        "model_setting_name": "ViTL",
        "checkpoint_id": getattr(
            runner,
            "CHECKPOINT_ID",
            "MaskCLIP_sd15_20241109_08_53_19_epoch13",
        ),
        "checkpoint_sha256": legacy_runner.CHECKPOINT_SHA256,
        "class_names": ["real", "forged"],
        "positive_class_index": 1,
    }
    if model != expected_model:
        raise ValueError("immutable.model contract changed")
    source = _require_mapping(immutable.get("source"), "immutable.source")
    if set(source) != {
        "repository",
        "root",
        "commit",
        "tracked_dirty",
        "core_source_files",
    }:
        raise ValueError("immutable.source key set changed")
    if source.get("repository") != legacy_runner.MODEL_REPO_URL:
        raise ValueError("immutable.source repository changed")
    if (
        source.get("commit") != legacy_runner.MODEL_SOURCE_COMMIT
        or source.get("tracked_dirty") is not False
    ):
        raise ValueError("immutable.source revision/cleanliness changed")
    source_root = Path(_require_string(source.get("root"), "immutable.source.root"))
    if (
        not source_root.is_absolute()
        or source_root.resolve() != source_root
        or not source_root.is_dir()
    ):
        raise ValueError("immutable.source.root is invalid")
    if (
        legacy_runner._git_value(source_root, "rev-parse", "HEAD")
        != legacy_runner.MODEL_SOURCE_COMMIT
    ):
        raise ValueError("live OpenSDI source commit changed")
    tracked_status = legacy_runner._git_value(
        source_root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if tracked_status is None:
        raise ValueError("cannot inspect live OpenSDI tracked source status")
    if tracked_status:
        raise ValueError("live OpenSDI tracked source is dirty")

    source_files = _require_mapping(
        source.get("core_source_files"),
        "immutable.source.core_source_files",
    )
    expected_source_files = set(getattr(runner, "OPENSDI_SOURCE_FILES"))
    if set(source_files) != expected_source_files:
        raise ValueError("immutable.source core source-file set changed")
    for relative in sorted(expected_source_files):
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise ValueError("runner OpenSDI source path contract is unsafe")
        record = _require_mapping(
            source_files.get(relative),
            f"immutable.source.core_source_files.{relative}",
        )
        if set(record) != {"bytes", "sha256"}:
            raise ValueError(f"OpenSDI source metadata changed for {relative}")
        path = source_root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/unsafe OpenSDI source file: {path}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"OpenSDI source byte size changed for {relative}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"OpenSDI source SHA-256 changed for {relative}")

    assets = _require_mapping(immutable.get("assets"), "immutable.assets")
    expected_assets = {
        "maskclip": {
            "filename": legacy_runner.CHECKPOINT_FILENAME,
            "bytes": getattr(runner, "MASKCLIP_CHECKPOINT_BYTES"),
            "sha256": legacy_runner.CHECKPOINT_SHA256,
            "id": getattr(runner, "CHECKPOINT_ID"),
            "repository": legacy_runner.CHECKPOINT_REPO,
            "revision": legacy_runner.CHECKPOINT_REVISION,
            "epoch": 13,
            "weights_only": True,
            "strict_model_load": True,
        },
        "mae_initialization": {
            "filename": "mae_pretrain_vit_base.pth",
            "bytes": getattr(runner, "MAE_CHECKPOINT_BYTES"),
            "sha256": legacy_runner.MAE_SHA256,
            "role": "OpenSDI_MaskCLIP_constructor_initialization",
        },
        "clip": {
            "filename": "ViT-L-14.pt",
            "bytes": getattr(runner, "CLIP_CHECKPOINT_BYTES"),
            "sha256": legacy_runner.CLIP_SHA256,
            "git_commit": legacy_runner.CLIP_GIT_COMMIT,
            "model": "ViT-L/14",
        },
    }
    if set(assets) != set(expected_assets):
        raise ValueError("immutable.assets role set changed")
    for role, expected_static in expected_assets.items():
        record = _require_mapping(
            assets.get(role),
            f"immutable.assets.{role}",
        )
        if set(record) != {"path", *expected_static}:
            raise ValueError(f"immutable.assets.{role} metadata keys changed")
        for field, expected_value in expected_static.items():
            if record.get(field) != expected_value:
                raise ValueError(f"immutable.assets.{role}.{field} changed")
        path = Path(
            _require_string(
                record.get("path"),
                f"immutable.assets.{role}.path",
            )
        )
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not path.is_file()
            or path.is_symlink()
        ):
            raise FileNotFoundError(
                f"missing/unsafe immutable.assets.{role}.path: {path}"
            )
        if path.name != expected_static["filename"]:
            raise ValueError(f"immutable.assets.{role} filename/path mismatch")
        if path.stat().st_size != expected_static["bytes"]:
            raise ValueError(f"immutable.assets.{role} live byte size changed")
        if sha256_file(path) != expected_static["sha256"]:
            raise ValueError(f"immutable.assets.{role} live SHA-256 changed")


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    run_id: str,
    expected_mode: str | None = None,
) -> tuple[str, dict[str, Any]]:
    runner = _require_runner()
    expected_top = {
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
    if set(manifest) != expected_top:
        raise ValueError("run manifest key set changed")
    schema = _runner_value(
        "RUN_MANIFEST_SCHEMA",
        _runner_value("RUN_MANIFEST_SCHEMA_VERSION"),
    )
    if manifest.get("schema_version") != schema:
        raise ValueError("unsupported MaskCLIP Balanced run manifest")
    if manifest.get("run_id") != run_id:
        raise ValueError("manifest run_id mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("analyzer requires manifest status complete")
    _require_string(manifest.get("started_at"), "manifest started_at")
    _require_string(manifest.get("completed_at"), "manifest completed_at")
    immutable = _require_mapping(manifest.get("immutable"), "manifest immutable")
    expected_immutable = {
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
        "runtime",
        "preflight",
        "artifact_contract",
        "outputs",
    }
    if set(immutable) != expected_immutable:
        raise ValueError("manifest immutable key set changed")
    config_schema = _runner_value(
        "RUN_CONFIG_SCHEMA",
        _runner_value("RUN_CONFIG_SCHEMA_VERSION"),
    )
    if immutable.get("schema_version") != config_schema:
        raise ValueError("immutable schema_version changed")
    if immutable.get("run_id") != run_id:
        raise ValueError("immutable run_id mismatch")
    mode = immutable.get("mode")
    if mode not in ("formal", "smoke"):
        raise ValueError("immutable mode is unsupported")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"analyzer requires immutable.mode={expected_mode}")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint")
    calculated = hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()
    if fingerprint != calculated:
        raise ValueError("manifest fingerprint does not bind immutable config")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    if immutable.get("score_spec") != _score_spec().as_dict():
        raise ValueError("immutable score_spec changed")
    expected_t2 = _runner_value("T2_SPEC")
    if expected_t2 is not None:
        expected_t2_value = (
            expected_t2.as_dict() if hasattr(expected_t2, "as_dict") else expected_t2
        )
        if immutable.get("t2_spec") != expected_t2_value:
            raise ValueError("immutable t2_spec changed")
    else:
        t2 = _require_mapping(immutable.get("t2_spec"), "immutable.t2_spec")
        if (
            t2.get("threshold") != MASK_THRESHOLD
            or t2.get("threshold_operator") != ">="
        ):
            raise ValueError("immutable t2_spec threshold changed")
    expected_scope = _runner_value("TASK_SCOPE")
    if expected_scope is not None and immutable.get("task_scope") != expected_scope:
        raise ValueError("immutable task_scope changed")
    else:
        task_scope = _require_mapping(
            immutable.get("task_scope"),
            "immutable.task_scope",
        )
        if (
            task_scope.get("valid_for_t1") is not True
            or task_scope.get("valid_for_t2") is not True
            or task_scope.get("native_dense_output") is not True
        ):
            raise ValueError("immutable task_scope changed")
    _validate_model_provenance(immutable)
    preprocess = _require_mapping(
        immutable.get("preprocess"),
        "immutable.preprocess",
    )
    expected_preprocess = {
        "profile": _runner_value(
            "PREPROCESS_PROFILE",
            "official_opensdi_512_stretch_clip_normalize",
        ),
        "precision": "float32",
        "batch_size": 1,
        "model_input_size": [MODEL_SIZE, MODEL_SIZE],
        "input_resize": "opencv_inter_linear_stretch",
        "normalization_mean": legacy_runner.CLIP_MEAN.tolist(),
        "normalization_std": legacy_runner.CLIP_STD.tolist(),
    }
    if preprocess != expected_preprocess:
        raise ValueError("immutable preprocessing contract changed")
    artifact_contract = _require_mapping(
        immutable.get("artifact_contract"),
        "immutable.artifact_contract",
    )
    expected_artifact_contract = _runner_value("ARTIFACT_CONTRACT")
    if (
        expected_artifact_contract is not None
        and artifact_contract != expected_artifact_contract
    ):
        raise ValueError("immutable artifact contract changed")
    _validate_runtime_evidence(
        immutable.get("runtime"),
        label="immutable.runtime",
    )
    preflight = _require_mapping(
        immutable.get("preflight"),
        "immutable.preflight",
    )
    expected_preflight = {
        "performed_before_accelerator_configuration": True,
        "dataset_files_verified": True,
        "source_commit_and_cleanliness_verified": True,
        "source_file_hashes_verified": True,
        "all_weight_file_hashes_and_sizes_verified": True,
        "cuda_used": False,
    }
    if preflight != expected_preflight:
        raise ValueError("immutable preflight evidence changed")
    _require_mapping(immutable.get("outputs"), "immutable.outputs")
    _require_mapping(manifest.get("dataset"), "manifest dataset")
    _require_mapping(manifest.get("outputs"), "manifest outputs")
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    for field in (
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
    ):
        _require_nonnegative_int(
            execution.get(field),
            f"manifest execution {field}",
        )
    _reject_nonfinite_numbers(manifest, "manifest")
    del runner
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
    expected = _read_jsonl_strict(expected_path, "expected inputs")
    if expected != list(selected):
        raise ValueError("expected-input snapshot is not the exact selection")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    required = {
        "contract": contract.as_dict(),
        "manifest_path": repo_relative(release.manifest_path, repo_root),
        "manifest_sha256": release.manifest_sha256,
        "expected_inputs_path": repo_relative(expected_path, repo_root),
        "expected_inputs_sha256": sha256_file(expected_path),
        "selected_images": len(selected),
    }
    if dataset != required:
        raise ValueError("manifest dataset exact contract mismatch")


def _validate_physical_results(
    *,
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    runner = _require_runner()
    validator = getattr(runner, "_validate_runner_attempt", None)
    if not callable(validator):
        raise RuntimeError("MaskCLIP runner lacks _validate_runner_attempt")
    expected = {str(row["sample_id"]): row for row in selected}
    for index, row in enumerate(rows):
        sample_id = _require_string(
            row.get("sample_id"),
            f"physical result row {index} sample_id",
        )
        if sample_id not in expected:
            raise ValueError(f"physical result row {index} has unexpected sample_id")
        if "pair_rank" in row:
            raise ValueError(f"physical result row {index} contains legacy pair_rank")
        validator(
            row,
            input_row=expected[sample_id],
            repo_root=repo_root,
            run_dir=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=False,
            recompute_preprocess=False,
        )
        _reject_nonfinite_numbers(row, f"physical result row {index}")


def _latest_in_selection_order(
    *,
    selected: Sequence[Mapping[str, Any]],
    physical_results: Sequence[Mapping[str, Any]],
    run_id: str,
    fingerprint: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=_score_spec(),
    )
    coverage = summarize_coverage(latest)
    require_complete_coverage(coverage)
    rows = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])]) for row in selected
    )
    return rows, coverage.as_dict()


def _validate_execution_accounting(
    manifest: Mapping[str, Any],
    *,
    selected_images: int,
    physical_rows: int,
    latest_rows: int,
) -> None:
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    expected = {
        "physical_result_rows": physical_rows,
        "latest_result_rows": latest_rows,
        "superseded_attempts": physical_rows - latest_rows,
        "new_errors": 0,
    }
    if set(execution) != {
        "new_successes",
        "resume_skips",
        "new_errors",
        "physical_result_rows",
        "latest_result_rows",
        "superseded_attempts",
    }:
        raise ValueError("manifest execution key set changed")
    for field, expected_value in expected.items():
        if execution.get(field) != expected_value:
            raise ValueError(f"manifest execution {field} mismatch")
    if (
        execution.get("new_successes", 0)
        + execution.get(
            "resume_skips",
            0,
        )
        != selected_images
    ):
        raise ValueError("manifest successful work accounting changed")


def _expected_artifact_root(
    *,
    repo_root: Path,
    run_id: str,
    persisted_artifact_root: Path | None = None,
) -> Path:
    resolver = getattr(_require_runner(), "resolve_artifact_root", None)
    if callable(resolver):
        return Path(
            resolver(
                repo_root=repo_root,
                run_id=run_id,
                artifact_root=persisted_artifact_root,
            )
        ).resolve()
    base = _runner_value(
        "DEFAULT_ARTIFACTS_DIR",
        Path("outputs/opensource/maskclip"),
    )
    base_path = Path(base)
    resolved_base = (
        base_path.resolve()
        if base_path.is_absolute()
        else (repo_root.resolve() / base_path).resolve()
    )
    candidate = (
        resolved_base / _valid_run_id(run_id)
        if persisted_artifact_root is None
        else persisted_artifact_root
    ).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("artifact root escapes repository root") from error
    if candidate.name != _valid_run_id(run_id):
        raise ValueError("artifact root is not bound to the exact run ID")
    return candidate


def _expected_output_paths(
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
) -> dict[str, str]:
    return {
        "results_path": repo_relative(run_dir / "results.jsonl", repo_root),
        "expected_inputs_path": repo_relative(
            run_dir / "expected_inputs.jsonl",
            repo_root,
        ),
        "summary_path": repo_relative(run_dir / "summary.json", repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        "score_maps_model_512_dir": repo_relative(
            artifact_root / "score_maps_model_512",
            repo_root,
        ),
        "score_maps_native_dir": repo_relative(
            artifact_root / "score_maps_native",
            repo_root,
        ),
        "masks_native_dir": repo_relative(
            artifact_root / "masks_native",
            repo_root,
        ),
    }


def _validate_output_bindings(
    *,
    manifest: Mapping[str, Any],
    immutable: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    selected: Sequence[Mapping[str, Any]],
) -> None:
    actual_paths = _expected_output_paths(repo_root, run_dir, artifact_root)
    immutable_outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable.outputs",
    )
    if immutable_outputs != actual_paths:
        raise ValueError("immutable.outputs exact path contract mismatch")
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    required_hashes = {
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
    }
    expected_applicable = sum(
        row.get("gt_mask_kind") in _T2_GT_KINDS for row in selected
    )
    expected_inventory = {
        "model_512": len(selected),
        "native": expected_applicable,
        "masks": expected_applicable,
    }
    expected_outputs = {
        **actual_paths,
        **required_hashes,
        "artifact_inventory": expected_inventory,
    }
    if outputs != expected_outputs:
        raise ValueError("manifest outputs exact contract mismatch")


def _validate_runtime_summary(
    *,
    summary: Mapping[str, Any],
    run_id: str,
    fingerprint: str,
    mode: str,
    contract: RunDatasetContract,
    coverage: Mapping[str, Any],
) -> None:
    schema = _runner_value(
        "RUNTIME_SUMMARY_SCHEMA",
        _runner_value("RUNTIME_SUMMARY_SCHEMA_VERSION"),
    )
    required = {
        "schema_version": schema,
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete",
        "mode": mode,
        "model": "MaskCLIP",
        "model_slug": "opensdi_maskclip_sd15",
        "score_spec": _score_spec().as_dict(),
        "t2_spec": _runner_value("T2_SPEC"),
        "dataset_contract": contract.as_dict(),
        "coverage": dict(coverage),
        "artifact_inventory": {
            "model_512": coverage["valid_images"],
            "native": sum(
                counts["valid_images"]
                for condition, counts in coverage["counts_by_condition"].items()
                if condition in ("real", "local_mouse", "local_cat", "local_trash_can")
            ),
            "masks": sum(
                counts["valid_images"]
                for condition, counts in coverage["counts_by_condition"].items()
                if condition in ("real", "local_mouse", "local_cat", "local_trash_can")
            ),
        },
    }
    for field, expected_value in required.items():
        if summary.get(field) != expected_value:
            raise ValueError(f"stored run summary {field} mismatch")
    owner = summary.get("scientific_metrics_owner")
    if owner != "analyze_maskclip_balanced.py":
        raise ValueError("stored run summary scientific metrics owner changed")
    _require_string(summary.get("generated_at"), "run summary generated_at")
    if set(summary) != {
        *required,
        "scientific_metrics_owner",
        "generated_at",
    }:
        raise ValueError("stored run summary key set changed")
    _reject_nonfinite_numbers(summary, "run summary")


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    mode: str,
) -> RunBundle:
    root = repo_root.resolve()
    run_id = _valid_run_id(run_id)
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
        expected_mode=mode,
    )
    immutable_outputs = _require_mapping(
        immutable.get("outputs"),
        "immutable.outputs",
    )
    persisted_artifact_root = _safe_repo_path(
        immutable_outputs.get("artifact_root"),
        repo_root=root,
        label="immutable.outputs.artifact_root",
        require_file=False,
    )
    artifact_root = _expected_artifact_root(
        repo_root=root,
        run_id=run_id,
        persisted_artifact_root=persisted_artifact_root,
    )
    if persisted_artifact_root != artifact_root:
        raise ValueError("immutable artifact root is not canonical")
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
    physical = tuple(_read_jsonl_strict(results_path, "physical results"))
    _validate_physical_results(
        rows=physical,
        selected=selected,
        repo_root=root,
        artifact_root=artifact_root,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    latest, coverage = _latest_in_selection_order(
        selected=selected,
        physical_results=physical,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    _validate_execution_accounting(
        manifest,
        selected_images=len(selected),
        physical_rows=len(physical),
        latest_rows=len(latest),
    )
    if mode == "smoke" and len(physical) != len(selected):
        raise ValueError("smoke comparison requires one physical attempt per input")
    summary = _load_json(summary_path, "runtime summary")
    _validate_output_bindings(
        manifest=manifest,
        immutable=immutable,
        repo_root=root,
        run_dir=run_dir,
        artifact_root=artifact_root,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        selected=selected,
    )
    _validate_runtime_summary(
        summary=summary,
        run_id=run_id,
        fingerprint=fingerprint,
        mode=mode,
        contract=contract,
        coverage=coverage,
    )
    artifacts = validate_artifact_inventory(
        selected=selected,
        latest_results=latest,
        repo_root=root,
        artifact_root=artifact_root,
    )
    return RunBundle(
        run_id=run_id,
        fingerprint=fingerprint,
        run_dir=run_dir,
        artifact_root=artifact_root,
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


def _native_map_loader(
    bundle: RunBundle,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], np.ndarray]:
    expected_by_id = {str(row["sample_id"]): row for row in bundle.selected}

    def load(
        input_row: Mapping[str, Any],
        result_row: Mapping[str, Any],
    ) -> np.ndarray:
        sample_id = _require_string(
            input_row.get("sample_id"),
            "T2 callback input sample_id",
        )
        if result_row.get("sample_id") != sample_id:
            raise ValueError("T2 callback input/result identity mismatch")
        expected = expected_by_id.get(sample_id)
        if expected is None or dict(input_row) != dict(expected):
            raise ValueError("T2 callback received a non-selected input")
        artifact = bundle.artifacts.get(sample_id)
        if (
            artifact is None
            or not artifact.t2_applicable
            or artifact.native_path is None
        ):
            raise ValueError(
                f"T2 callback requested a non-applicable map for {sample_id}"
            )
        array = np.load(
            artifact.native_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            array.shape != (artifact.height, artifact.width)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
            or float(array.min()) < 0.0
            or float(array.max()) > 1.0
        ):
            raise ValueError(f"T2 callback map changed for {sample_id}")
        return array

    return load


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    t1 = summarize_balanced250_t1(
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
    if (
        t1.get("schema_version") != T1_METRICS_SCHEMA_VERSION
        or t1.get("coverage", {}).get("is_complete") is not True
    ):
        raise ValueError("shared Balanced250 T1 metrics are incomplete")
    t2_function = _require_t2_metrics()
    t2 = t2_function(
        bundle.release.inputs,
        bundle.latest_results,
        repo_root=bundle.release.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle),
        score_map_name="maskclip_native_probability_map",
        threshold=MASK_THRESHOLD,
        threshold_operator=">=",
        iterations=iterations,
        seed=seed,
    )
    if t2.get("schema_version") != T2_METRICS_SCHEMA_VERSION:
        raise ValueError("shared Balanced250 T2 metrics schema changed")
    coverage = _require_mapping(t2.get("coverage"), "T2 metrics coverage")
    if coverage.get("is_complete") is not True:
        raise ValueError("shared Balanced250 T2 metrics are incomplete")
    excluded = t2.get("excluded_not_applicable")
    if not isinstance(excluded, Mapping):
        raise ValueError("T2 metrics lacks excluded_not_applicable evidence")
    serialized_excluded = stable_json(excluded)
    for condition in (
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ):
        if condition not in serialized_excluded:
            raise ValueError(f"T2 metrics does not exclude {condition}")
    return {
        "schema_version": "maskclip_balanced250_summary_v2",
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
        },
        "t1": t1,
        "t2": t2,
    }


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
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


def _smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("smoke projection requires a successful result")
    sample_id = _require_string(row.get("sample_id"), "smoke sample_id")
    _validate_score_payload(row, sample_id=sample_id)
    return {
        key: value for key, value in row.items() if key not in _SMOKE_IGNORED_FIELDS
    }


def _compare_artifact_bytes(
    left: DenseArtifacts,
    right: DenseArtifacts,
) -> None:
    if left.t2_applicable is not right.t2_applicable:
        raise ValueError(f"{left.sample_id} T2 artifact applicability differs")
    for label, left_path, right_path in (
        ("model map", left.model_path, right.model_path),
        ("native map", left.native_path, right.native_path),
        ("mask", left.mask_path, right.mask_path),
    ):
        if (left_path is None) is not (right_path is None):
            raise ValueError(f"{left.sample_id} {label} presence differs")
        if left_path is not None and right_path is not None:
            if left_path.read_bytes() != right_path.read_bytes():
                raise ValueError(f"{left.sample_id} {label} bytes differ")


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_artifacts: Mapping[str, DenseArtifacts],
    replay_artifacts: Mapping[str, DenseArtifacts],
) -> dict[str, Any]:
    reference = _unique_rows(reference_rows, label="reference results")
    replay = _unique_rows(replay_rows, label="replay results")
    if set(reference) != set(replay):
        raise ValueError("reference/replay result coverage differs")
    if set(reference_artifacts) != set(reference):
        raise ValueError("reference artifact coverage differs")
    if set(replay_artifacts) != set(replay):
        raise ValueError("replay artifact coverage differs")
    applicable = 0
    for sample_id in sorted(reference):
        left = _smoke_projection(reference[sample_id])
        right = _smoke_projection(replay[sample_id])
        if left != right:
            differing = sorted(
                {
                    *(set(left) ^ set(right)),
                    *(key for key in set(left) & set(right) if left[key] != right[key]),
                }
            )
            raise ValueError(
                f"{sample_id} computational row differs at {differing[:3]}"
            )
        _compare_artifact_bytes(
            reference_artifacts[sample_id],
            replay_artifacts[sample_id],
        )
        applicable += int(reference_artifacts[sample_id].t2_applicable)
    return {
        "images_compared": len(reference),
        "t2_applicable_images_compared": applicable,
        "identity_timing_path_fields_ignored": sorted(_SMOKE_IGNORED_FIELDS),
        "exact_computational_projection": True,
        "max_class_logit_abs_difference": 0.0,
        "max_class_probability_abs_difference": 0.0,
        "max_ai_score_abs_difference": 0.0,
        "max_model_map_abs_difference": 0.0,
        "model_map_file_bytes_exact": True,
        "applicable_native_map_file_bytes_exact": True,
        "applicable_mask_file_bytes_exact": True,
    }


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    output_path: Path | None,
) -> dict[str, Any]:
    reference_run_id = _valid_run_id(reference_run_id)
    replay_run_id = _valid_run_id(replay_run_id)
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
    comparison = compare_computational_results(
        reference_rows=reference.latest_results,
        replay_rows=replay.latest_results,
        reference_artifacts=reference.artifacts,
        replay_artifacts=replay.artifacts,
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


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def independent_preprocess_image(
    input_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    with Image.open(input_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    resized = cv2.resize(
        rgb,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    normalized = (
        resized / np.float32(255.0) - legacy_runner.CLIP_MEAN
    ) / legacy_runner.CLIP_STD
    tensor = np.ascontiguousarray(
        normalized.transpose(2, 0, 1),
        dtype=np.float32,
    )
    if tensor.shape != (3, MODEL_SIZE, MODEL_SIZE) or not np.isfinite(tensor).all():
        raise ValueError("independent preprocessing produced an invalid tensor")
    audit = {
        "profile": _runner_value(
            "PREPROCESS_PROFILE",
            "official_opensdi_512_stretch_clip_normalize",
        ),
        "decoded_size": [width, height],
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_sha256": _array_sha256(tensor),
        "input_resize": "opencv_inter_linear_stretch",
        "normalization_mean": legacy_runner.CLIP_MEAN.tolist(),
        "normalization_std": legacy_runner.CLIP_STD.tolist(),
    }
    return tensor, audit


def replay_model(
    bundle: RunBundle,
    *,
    source_root: Path,
    checkpoint_path: Path,
    clip_checkpoint_path: Path,
    device_text: str,
) -> dict[str, Any]:
    runner = _require_runner()
    source, assets = runner.verify_assets(
        opensdi_root=source_root,
        checkpoint_path=checkpoint_path,
        clip_checkpoint_path=clip_checkpoint_path,
    )
    immutable = _require_mapping(
        bundle.manifest.get("immutable"),
        "manifest immutable",
    )
    if immutable.get("source") != source or immutable.get("assets") != assets:
        raise ValueError("fresh replay source/assets differ from manifest")
    device, runtime = runner.configure_runtime(device_text)
    model, loaded_device = legacy_runner.load_model(
        opensdi_root=source_root,
        checkpoint_path=checkpoint_path,
        clip_checkpoint_path=clip_checkpoint_path,
        device_name=str(device),
    )
    if str(loaded_device) != str(device):
        raise ValueError("fresh MaskCLIP model loaded on an unexpected device")
    capture = legacy_runner.LogitCapture(model)
    replayed = 0
    applicable_replayed = 0
    max_logit = 0.0
    max_probability = 0.0
    max_score = 0.0
    max_model_map = 0.0
    max_native_map = 0.0
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
            tensor, preprocess = independent_preprocess_image(input_path)
            if row.get("preprocess") != preprocess:
                raise ValueError(f"{sample_id} preprocessing fresh replay changed")
            (
                logits,
                probabilities,
                _peak_bytes,
                _latency_ms,
                model_map,
            ) = legacy_runner.infer_one(
                model,
                capture,
                device,
                tensor,
            )
            stored_logits = np.asarray(
                [
                    row["class_logits"]["real"],
                    row["class_logits"]["forged"],
                ],
                dtype=np.float64,
            )
            stored_probabilities = np.asarray(
                [
                    row["class_probabilities"]["real"],
                    row["class_probabilities"]["forged"],
                ],
                dtype=np.float64,
            )
            logits_difference = float(
                np.max(np.abs(stored_logits - np.asarray(logits, dtype=np.float64)))
            )
            probability_difference = float(
                np.max(
                    np.abs(
                        stored_probabilities
                        - np.asarray(probabilities, dtype=np.float64)
                    )
                )
            )
            score_difference = abs(float(row["ai_score"]) - float(probabilities[1]))
            stored_model_map = np.load(
                bundle.artifacts[sample_id].model_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            model_difference = float(
                np.max(
                    np.abs(
                        np.asarray(stored_model_map, dtype=np.float64)
                        - np.asarray(model_map, dtype=np.float64)
                    )
                )
            )
            max_logit = max(max_logit, logits_difference)
            max_probability = max(max_probability, probability_difference)
            max_score = max(max_score, score_difference)
            max_model_map = max(max_model_map, model_difference)
            if logits_difference > LOGIT_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} class logits replay mismatch")
            if probability_difference > SCORE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} class probabilities replay mismatch")
            if score_difference > SCORE_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} ai_score replay mismatch")
            if model_difference > MODEL_MAP_ABS_TOLERANCE:
                raise ValueError(f"{sample_id} model map replay mismatch")
            if row.get("classification_decision") is not (
                float(probabilities[1]) >= CLASSIFICATION_THRESHOLD
            ):
                raise ValueError(f"{sample_id} replay decision mismatch")
            artifact = bundle.artifacts[sample_id]
            if artifact.t2_applicable:
                assert (
                    artifact.native_path is not None and artifact.mask_path is not None
                )
                replay_native = _independent_restore_native(
                    np.asarray(model_map, dtype=np.float32),
                    width=artifact.width,
                    height=artifact.height,
                )
                stored_native = np.load(
                    artifact.native_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                native_difference = float(
                    np.max(
                        np.abs(
                            replay_native.astype(np.float64)
                            - np.asarray(stored_native, dtype=np.float64)
                        )
                    )
                )
                max_native_map = max(max_native_map, native_difference)
                if native_difference > MODEL_MAP_ABS_TOLERANCE:
                    raise ValueError(f"{sample_id} native map replay mismatch")
                with Image.open(artifact.mask_path) as opened:
                    stored_mask = np.asarray(opened, dtype=np.uint8)
                if not np.array_equal(
                    stored_mask == 255,
                    replay_native >= MASK_THRESHOLD,
                ):
                    raise ValueError(
                        f"{sample_id} native threshold-mask replay mismatch"
                    )
                applicable_replayed += 1
            replayed += 1
    finally:
        capture.close()
        del capture, model
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()
    if replayed != len(bundle.selected):
        raise ValueError("fresh replay did not cover every formal input")
    expected_applicable = sum(
        row.get("gt_mask_kind") in _T2_GT_KINDS for row in bundle.selected
    )
    if applicable_replayed != expected_applicable:
        raise ValueError("fresh replay T2-applicable coverage changed")
    return {
        "images_replayed": replayed,
        "t2_applicable_images_replayed": applicable_replayed,
        "source_commit": source["commit"],
        "checkpoint_sha256": assets["maskclip"]["sha256"],
        "mae_checkpoint_sha256": assets["mae_initialization"]["sha256"],
        "clip_checkpoint_sha256": assets["clip"]["sha256"],
        "runtime": runtime,
        "class_logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "class_probability_abs_tolerance": SCORE_ABS_TOLERANCE,
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "model_map_abs_tolerance": MODEL_MAP_ABS_TOLERANCE,
        "max_class_logit_abs_difference": max_logit,
        "max_class_probability_abs_difference": max_probability,
        "max_ai_score_abs_difference": max_score,
        "max_model_map_abs_difference": max_model_map,
        "max_applicable_native_map_abs_difference": max_native_map,
        "independent_preprocess_recomputed": True,
        "independent_native_restore_recomputed": True,
        "derived_native_threshold_masks_exact": True,
    }


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    checkpoint_path: Path,
    clip_checkpoint_path: Path,
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
    metrics = recompute_metrics(
        bundle,
        iterations=iterations,
        seed=seed,
    )
    if metrics_output_path is not None:
        atomic_write_json(metrics_output_path, metrics)
    replay_report = (
        replay_model(
            bundle,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            clip_checkpoint_path=clip_checkpoint_path,
            device_text=device_text,
        )
        if replay
        else None
    )
    applicable = sum(artifact.t2_applicable for artifact in bundle.artifacts.values())
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": ("replay_audit_passed" if replay else "artifact_audit_passed"),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": len(bundle.selected),
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": bundle.coverage,
        "artifact_inventory": {
            "model_512_maps": len(bundle.artifacts),
            "native_maps": applicable,
            "native_masks": applicable,
            "fullframe_model_maps_diagnostic_only": (
                len(bundle.artifacts) - applicable
            ),
        },
        "metrics_schema_version": metrics["schema_version"],
        "t1_metrics_schema_version": metrics["t1"]["schema_version"],
        "t2_metrics_schema_version": metrics["t2"]["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "fresh_model_replay": replay_report,
        "contract_checks": {
            "exact_formal_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "result_identity_run_id_fingerprint_status_validated": True,
            "adapter_source_and_model_asset_hashes_validated": True,
            "model_512_inventory_shape_dtype_hash_finite_range_validated": True,
            "applicable_native_maps_exact_inter_linear_restore": True,
            "applicable_masks_exact_native_ge_0_5": True,
            "exact_diff_model_target_uses_inter_nearest": True,
            "real_t2_restricted_to_false_positive_area": True,
            "fullframe_t2_not_applicable": True,
            "balanced250_t1_and_t2_metrics_recomputed": True,
            "legacy_pair_rank_used": False,
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
    runner = _require_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=getattr(runner, "DEFAULT_RESULTS_DIR", DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--run-id",
        default=getattr(
            runner,
            "DEFAULT_FORMAL_RUN_ID",
            DEFAULT_FORMAL_RUN_ID,
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=DEFAULT_CLIP_CHECKPOINT,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="fresh replay device; use explicit cpu or cuda:N",
    )
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument(
        "--compare-smoke-run-id",
        help=(
            "validate --run-id and this second smoke run, then write an "
            "exact computational comparison"
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
    run_id = _valid_run_id(args.run_id)
    results_dir = _resolve_results_root(args.results_dir, repo_root)
    run_dir = _resolve_run_dir(results_dir, run_id)
    if args.compare_smoke_run_id is not None:
        compare_run_id = _valid_run_id(args.compare_smoke_run_id)
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
            else results_dir / f"{run_id}__vs__{compare_run_id}_comparison.json"
        )
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
        raise ValueError("--comparison-output requires --compare-smoke-run-id")
    if args.device != "cpu" and not str(args.device).startswith("cuda:"):
        raise ValueError("--device must be explicit cpu or cuda:N")
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
        clip_checkpoint_path=_anchored(args.clip_checkpoint, repo_root),
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
