#!/usr/bin/env python3
"""Independently audit one frozen NPR whole-image detection run.

The analyzer treats the runner JSON, summary, manifest, and persisted
512-dimensional ``fc1`` input as untrusted artifacts.  It verifies the pinned
source tree and checkpoint, repeats Pillow decoding and native-size
preprocessing, runs the complete official model again on the recorded
runtime/device, captures the independently recomputed ``fc1`` input, replays
the final linear layer from the persisted feature, and checks every score and
decision alias.

NPR is a T1-only detector.  Claimed dense maps, masks, localization metrics,
T2 outputs, or joint scores are rejected.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import PIL
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.npr_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_npr_raw_logit_diagnostic,
    summarize_npr_results,
)


DEFAULT_RESULTS_DIR = Path("results/opensource/npr")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_RUN_ID = "npr_aigcdetect_progan4class_mouse_canonical_v1_full275_20260725"
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/NPR-DeepfakeDetection-781ced3f"
)
DEFAULT_HF_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/NPR-HF-Space-522a9f10"
)

FROZEN_SOURCE_COMMIT = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"
FROZEN_HF_SPACE_COMMIT = "522a9f1020f7454d486f28a0d5c148ec37919b32"
FROZEN_HF_SOURCE_FILES = {
    "app.py": "06da679323935bb7f6c8387f18d2d9ce58b488d33d0d1e67286c6ab8d8a7b35a",
    "README.md": ("d73e2354f53c45238a831ed18cecede4e9c3d4da13b9cfd57baf327a502430df"),
    "requirements.txt": (
        "f32a2c183d5e1974b0447ff9262cdb289020dd3c1118854e8a3a95cb3f0ba66c"
    ),
}
FROZEN_CHECKPOINT_SHA256 = (
    "b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a"
)
FROZEN_CHECKPOINT_BYTES = 5_842_385
FROZEN_PROFILE = "official_aigcdetect_native_even_trim"
FEATURE_DIMENSION = 512
FEATURE_DTYPE = np.dtype("float32")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
SCORE_SEMANTICS = "official_float32_sigmoid_probability_higher_is_fake"
T1_POLICY = "official_NPR_AIGC_float32_sigmoid"
FEATURE_SEMANTICS = "official_fc1_input_after_adaptive_global_average_pool"
RAW_LOGIT_ABSOLUTE_TOLERANCE = 1e-5
SIGMOID_ABSOLUTE_TOLERANCE = 1e-7
FULL_MOUSE_VISIBILITY_CENSUS = {"full": 275}

_T2_OR_LOCALIZATION_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "score_map",
        "score_map_path",
        "heatmap",
        "heatmap_path",
        "predicted_mask",
        "predicted_mask_path",
        "mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "miou",
        "dice",
        "pixel_f1",
        "s_joint",
        "joint_score",
        "joint_metrics",
    }
)

_PREFIX_EXACT_FIELDS = (
    "preprocess_profile",
    "checkpoint_id",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
    "task_scope",
    "preprocess",
    "npr_feature_sha256",
    "npr_feature_shape",
    "npr_feature_dtype",
    "npr_feature_semantics",
    "raw_logit",
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
)

_RESULT_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "sample_id",
        "rank",
        "pair_rank",
        "task_id",
        "kind",
        "label",
        "domain",
        "candidate",
        "dataset_id",
        "input_path",
        "input_sha256",
        "input_width",
        "input_height",
        "preprocess_profile",
        "checkpoint_id",
        "config_fingerprint",
        "edit_visibility",
        "edit_visible_gt_fraction",
        "edit_visibility_evidence",
        "task_scope",
    }
)

_SUCCESS_PAYLOAD_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "preprocess",
        "preprocess_latency_ms",
        "npr_feature_path",
        "npr_feature_sha256",
        "npr_feature_shape",
        "npr_feature_dtype",
        "npr_feature_semantics",
        "raw_logit",
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

_ERROR_PAYLOAD_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "error_type",
        "error",
        "traceback",
    }
)


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent result of the frozen NPR input transform."""

    tensor: Any
    decoded_rgb: np.ndarray
    decoded_rgb_sha256: str
    tensor_sha256: str
    residual: Any
    residual_sha256: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """Torch/device pair bound to the recorded numerical runtime."""

    torch: ModuleType
    device: Any
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RunFiles:
    """Physical files belonging to one NPR run directory."""

    run_dir: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest_path: Path
    rows: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    summary: dict[str, Any]
    manifest: dict[str, Any]


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON array")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


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


def _require_probability(value: Any, label: str) -> float:
    result = _require_finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} falls outside [0, 1]")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    digest = _require_sha256(expected, f"{label} expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {digest}")
    return actual


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    return _array_sha256(tensor.detach().cpu().contiguous().numpy())


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_pin(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise RuntimeError(f"NPR runner lacks required audit pin {name}")
    return copy.deepcopy(getattr(module, name))


def _load_runner_pins() -> SimpleNamespace:
    """Import only immutable constants from the runner."""

    from eval.opensource import run_npr as runner

    return SimpleNamespace(
        MODEL_NAME=_module_pin(runner, "MODEL_NAME"),
        MODEL_SLUG=_module_pin(runner, "MODEL_SLUG"),
        MODEL_ARCH=_module_pin(runner, "MODEL_ARCH"),
        MODEL_REPO_URL=_module_pin(runner, "MODEL_REPO_URL"),
        MODEL_SOURCE_COMMIT=_module_pin(runner, "MODEL_SOURCE_COMMIT"),
        PAPER_URL=_module_pin(runner, "PAPER_URL"),
        HF_SPACE_URL=_module_pin(runner, "HF_SPACE_URL"),
        HF_SPACE_COMMIT=_module_pin(runner, "HF_SPACE_COMMIT"),
        HF_SOURCE_FILES=_module_pin(runner, "HF_SOURCE_FILES"),
        PREPROCESS_PROFILE=_module_pin(runner, "PREPROCESS_PROFILE"),
        MODEL_SEED=int(_module_pin(runner, "MODEL_SEED")),
        CLASSIFICATION_THRESHOLD=float(_module_pin(runner, "CLASSIFICATION_THRESHOLD")),
        CLASSIFICATION_THRESHOLD_OPERATOR=str(
            _module_pin(runner, "CLASSIFICATION_THRESHOLD_OPERATOR")
        ),
        FEATURE_DIMENSION=int(_module_pin(runner, "FEATURE_DIMENSION")),
        IMAGE_MEAN=tuple(_module_pin(runner, "IMAGE_MEAN")),
        IMAGE_STD=tuple(_module_pin(runner, "IMAGE_STD")),
        LICENSE_RECORD=_module_pin(runner, "LICENSE_RECORD"),
        CHECKPOINT=_module_pin(runner, "CHECKPOINT"),
        EXCLUDED_RELEASE_ASSETS=_module_pin(
            runner,
            "EXCLUDED_RELEASE_ASSETS",
        ),
        SOURCE_FILES=_module_pin(runner, "SOURCE_FILES"),
    )


def _compare_nested(
    actual: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 0.0,
    exact_mapping_keys: bool = False,
) -> None:
    if isinstance(expected, Mapping):
        mapping = _require_mapping(actual, label)
        if exact_mapping_keys and set(mapping) != set(expected):
            raise ValueError(
                f"{label} keys mismatch: "
                f"{sorted(map(str, mapping))!r} != "
                f"{sorted(map(str, expected))!r}"
            )
        for key, nested in expected.items():
            if key not in mapping:
                raise ValueError(f"{label} lacks {key}")
            _compare_nested(
                mapping[key],
                nested,
                label=f"{label}.{key}",
                float_tolerance=float_tolerance,
                exact_mapping_keys=exact_mapping_keys,
            )
        return
    if isinstance(expected, list):
        sequence = _require_list(actual, label)
        if len(sequence) != len(expected):
            raise ValueError(
                f"{label} length mismatch: {len(sequence)} != {len(expected)}"
            )
        for index, (left, right) in enumerate(zip(sequence, expected)):
            _compare_nested(
                left,
                right,
                label=f"{label}[{index}]",
                float_tolerance=float_tolerance,
                exact_mapping_keys=exact_mapping_keys,
            )
        return
    if isinstance(expected, float):
        number = _require_finite(actual, label)
        if not math.isclose(
            number,
            expected,
            rel_tol=0.0,
            abs_tol=float_tolerance,
        ):
            raise ValueError(f"{label} mismatch: {number!r} != {expected!r}")
        return
    _require_equal(actual, expected, label)


def _reject_t2_localization_or_joint(value: Any, *, label: str) -> None:
    """Recursively reject output types the official NPR model lacks."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            if key in _T2_OR_LOCALIZATION_KEYS:
                if key == "t2" and nested is False:
                    continue
                raise ValueError(
                    f"{label} contains unsupported T2/localization/S_joint "
                    f"field {raw_key!r}"
                )
            _reject_t2_localization_or_joint(
                nested,
                label=f"{label}.{raw_key}",
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_t2_localization_or_joint(
                nested,
                label=f"{label}[{index}]",
            )


def effective_native_size(width: int, height: int) -> tuple[int, int]:
    if width <= 1 or height <= 1:
        raise ValueError("NPR input dimensions must exceed one pixel")
    return width - width % 2, height - height % 2


def preprocess_image(
    path: Path,
    *,
    torch_module: ModuleType,
    profile_id: str = FROZEN_PROFILE,
) -> PreprocessedImage:
    """Independently implement RGB→tensor→normalize→bottom/right trim."""

    functional = torch_module.nn.functional
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.asarray(rgb, dtype=np.uint8).copy()
    if decoded.shape != (height, width, 3):
        raise ValueError(f"decoded RGB shape is invalid: {decoded.shape}")
    channel_first = np.ascontiguousarray(decoded.transpose(2, 0, 1))
    tensor = torch_module.from_numpy(channel_first).to(dtype=torch_module.float32)
    tensor = tensor.div(255.0)
    mean = torch_module.tensor(
        IMAGE_MEAN,
        dtype=torch_module.float32,
    )[:, None, None]
    std = torch_module.tensor(
        IMAGE_STD,
        dtype=torch_module.float32,
    )[:, None, None]
    tensor = tensor.sub(mean).div(std)
    effective_width, effective_height = effective_native_size(width, height)
    tensor = tensor[:, :effective_height, :effective_width].contiguous()
    down = functional.interpolate(
        tensor.unsqueeze(0),
        scale_factor=0.5,
        mode="nearest",
        recompute_scale_factor=True,
    )
    reconstructed = functional.interpolate(
        down,
        scale_factor=2.0,
        mode="nearest",
        recompute_scale_factor=True,
    )
    residual = (tensor.unsqueeze(0) - reconstructed).squeeze(0).contiguous()
    residual64 = residual.to(dtype=torch_module.float64)
    residual_stats = {
        "minimum": float(residual.min().item()),
        "maximum": float(residual.max().item()),
        "mean": float(residual64.mean().item()),
        "mean_absolute": float(residual64.abs().mean().item()),
        "l2": float(torch_module.linalg.vector_norm(residual64).item()),
        "nonzero_elements": int(torch_module.count_nonzero(residual).item()),
        "elements": int(residual.numel()),
    }
    audit = {
        "profile": profile_id,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
            "trim_last_row_if_height_odd",
            "trim_last_column_if_width_odd",
        ],
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.detach().cpu().numpy().dtype),
        "tensor_sha256": _tensor_sha256(tensor),
        "npr_residual_shape": list(residual.shape),
        "npr_residual_dtype": str(residual.detach().cpu().numpy().dtype),
        "npr_residual_sha256": _tensor_sha256(residual),
        "npr_residual_stats": residual_stats,
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }
    return PreprocessedImage(
        tensor=tensor,
        decoded_rgb=decoded,
        decoded_rgb_sha256=audit["decoded_rgb_sha256"],
        tensor_sha256=audit["tensor_sha256"],
        residual=residual,
        residual_sha256=audit["npr_residual_sha256"],
        audit=audit,
    )


def _load_gt_mask(
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
) -> np.ndarray | None:
    sample_id = str(canonical.get("sample_id"))
    width = int(canonical["width"])
    height = int(canonical["height"])
    if canonical.get("kind") == "real":
        expected = {
            "label": 0,
            "gt_mask_kind": "all_zero",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": 0,
        }
        for key, value in expected.items():
            _require_equal(
                canonical.get(key),
                value,
                f"canonical real {sample_id} {key}",
            )
        return None
    _require_equal(canonical.get("kind"), "forged", f"{sample_id} kind")
    _require_equal(canonical.get("label"), 1, f"{sample_id} label")
    _require_equal(
        canonical.get("gt_mask_kind"),
        "exact_diff",
        f"{sample_id} GT kind",
    )
    value = canonical.get("gt_mask_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"forged canonical {sample_id} has no GT path")
    path = _anchored(Path(value), repo_root)
    _verify_hash(path, canonical.get("gt_mask_sha256"), f"{sample_id} GT mask")
    with Image.open(path) as opened:
        pixels = np.asarray(opened).copy()
    _require_equal(pixels.shape, (height, width), f"{sample_id} GT shape")
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"{sample_id} GT is not binary 0/255")
    positive = int(np.count_nonzero(pixels == 255))
    _require_equal(
        positive,
        int(canonical.get("gt_positive_pixels", -1)),
        f"{sample_id} GT positive pixels",
    )
    if positive <= 0:
        raise ValueError(f"{sample_id} forged GT is empty")
    return np.asarray(pixels, dtype=np.uint8)


def _visibility_from_exact_gt(
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    mask = _load_gt_mask(canonical, repo_root=repo_root)
    if mask is None:
        raise ValueError("visibility must be derived from the forged row")
    width = int(canonical["width"])
    height = int(canonical["height"])
    effective_width, effective_height = effective_native_size(width, height)
    total = int(np.count_nonzero(mask == 255))
    visible = int(np.count_nonzero(mask[:effective_height, :effective_width] == 255))
    fraction = visible / total
    category = "none" if visible == 0 else "full" if visible == total else "partial"
    return {
        "edit_visibility": category,
        "edit_visible_gt_fraction": fraction,
        "edit_visible_gt_pixels": visible,
        "edit_total_gt_pixels": total,
        "effective_native_xyxy": [0, 0, effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "evidence": "exact_diff_mask_intersection_with_native_even_trim",
    }


def _pair_visibility(
    canonical_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    forged: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        if row.get("kind") == "forged":
            task_id = str(row.get("task_id"))
            if task_id in forged:
                raise ValueError(f"duplicate forged canonical row {task_id}")
            forged[task_id] = row
    return {
        task_id: _visibility_from_exact_gt(row, repo_root=repo_root)
        for task_id, row in forged.items()
    }


def _float32_sigmoid(value: float, torch_module: ModuleType) -> float:
    tensor = torch_module.tensor(np.float32(value), dtype=torch_module.float32)
    return float(torch_module.sigmoid(tensor).item())


def _compare_float(
    actual: Any,
    expected: float,
    *,
    label: str,
    tolerance: float,
) -> float:
    value = _require_finite(actual, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{label} mismatch: {value!r} != {expected!r} "
            f"(absolute tolerance {tolerance})"
        )
    return value


def _audit_score_fields(
    row: Mapping[str, Any],
    *,
    replay_raw_logit: float,
    replay_probability: float,
    raw_tolerance: float = RAW_LOGIT_ABSOLUTE_TOLERANCE,
    probability_tolerance: float = SIGMOID_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Validate the raw float32 logit, sigmoid, aliases, and strict decision."""

    row_id = row.get("id")
    recorded_raw = _compare_float(
        row.get("raw_logit"),
        replay_raw_logit,
        label=f"row {row_id} raw_logit",
        tolerance=raw_tolerance,
    )
    probability = _require_probability(
        row.get("probability"),
        f"row {row_id} probability",
    )
    _compare_float(
        probability,
        replay_probability,
        label=f"row {row_id} probability replay",
        tolerance=probability_tolerance,
    )
    for key in ("ai_score", "score"):
        _compare_float(
            row.get(key),
            probability,
            label=f"row {row_id} {key}",
            tolerance=0.0,
        )
    decision = probability > FIXED_THRESHOLD
    raw_logit_decision = float(replay_raw_logit) > 0.0
    _require_equal(
        row.get("score_semantics"),
        SCORE_SEMANTICS,
        f"row {row_id} score semantics",
    )
    _require_equal(
        row.get("classification_decision"),
        decision,
        f"row {row_id} classification decision",
    )
    _require_equal(
        row.get("classification_threshold"),
        FIXED_THRESHOLD,
        f"row {row_id} classification threshold",
    )
    _require_equal(
        row.get("classification_threshold_operator"),
        THRESHOLD_OPERATOR,
        f"row {row_id} classification threshold operator",
    )
    classification = _require_mapping(
        row.get("classification"),
        f"row {row_id} classification",
    )
    _compare_nested(
        classification,
        {
            "raw_logit": recorded_raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "decision": decision,
            "semantics": SCORE_SEMANTICS,
        },
        label=f"row {row_id} classification",
        exact_mapping_keys=True,
    )
    t1 = _require_mapping(row.get("t1"), f"row {row_id} t1")
    _compare_nested(
        t1,
        {
            "raw_logit": recorded_raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "decision": decision,
            "policy": T1_POLICY,
        },
        label=f"row {row_id} t1",
        exact_mapping_keys=True,
    )
    manual = _require_mapping(
        row.get("manual_replay"),
        f"row {row_id} manual replay",
    )
    _compare_nested(
        manual,
        {
            "raw_logit": recorded_raw,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
        label=f"row {row_id} manual replay",
        exact_mapping_keys=True,
    )
    return {
        "raw_logit": float(replay_raw_logit),
        "probability": float(replay_probability),
        "decision": bool(decision),
        "raw_logit_decision": bool(raw_logit_decision),
        "decision_equivalent": bool(decision == raw_logit_decision),
    }


def _load_feature(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_dir: Path | None = None,
) -> tuple[np.ndarray, Path]:
    value = row.get("npr_feature_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row.get('id')} has no NPR feature path")
    path = _anchored(Path(value), repo_root)
    if not path.is_file() and run_dir is not None:
        path = (run_dir / value).resolve()
    if run_dir is not None:
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"row {row.get('id')} feature is outside its run directory"
            ) from exc
    _verify_hash(
        path,
        row.get("npr_feature_sha256"),
        f"row {row.get('id')} NPR feature",
    )
    try:
        feature = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"row {row.get('id')} NPR feature is not a safe NPY") from exc
    if not isinstance(feature, np.ndarray):
        raise ValueError(f"row {row.get('id')} feature is not an ndarray")
    _require_equal(
        feature.shape,
        (FEATURE_DIMENSION,),
        f"row {row.get('id')} feature shape",
    )
    _require_equal(
        feature.dtype,
        FEATURE_DTYPE,
        f"row {row.get('id')} feature dtype",
    )
    if not feature.flags.c_contiguous:
        raise ValueError(f"row {row.get('id')} feature is not C-contiguous")
    if not np.isfinite(feature).all():
        raise ValueError(f"row {row.get('id')} feature is not finite")
    _require_equal(
        row.get("npr_feature_shape"),
        [FEATURE_DIMENSION],
        f"row {row.get('id')} recorded feature shape",
    )
    _require_equal(
        row.get("npr_feature_dtype"),
        "float32",
        f"row {row.get('id')} recorded feature dtype",
    )
    _require_equal(
        row.get("npr_feature_semantics"),
        FEATURE_SEMANTICS,
        f"row {row.get('id')} feature semantics",
    )
    return feature, path


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    physical = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {line_number} has no id")
        histories[row_id].append((line_number, row))
        physical[str(row.get("status"))] += 1
    duplicates: list[dict[str, Any]] = []
    recovered: list[str] = []
    latest = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest[statuses[-1]] += 1
        if len(entries) > 1:
            duplicates.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line for line, _ in entries],
                    "statuses": statuses,
                }
            )
        if statuses[-1] == "ok" and "error" in statuses[:-1]:
            recovered.append(row_id)
    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_rows": len(result_rows) - len(histories),
        "ids_with_multiple_rows": len(duplicates),
        "recovered_error_to_ok": len(recovered),
        "recovered_ids": recovered,
        "historical_status_counts": dict(sorted(physical.items())),
        "latest_status_counts": dict(sorted(latest.items())),
        "duplicate_histories": duplicates,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {index} has no id")
        latest[row_id] = row
    return latest


def _selected_rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replay_input_selection(
    rows: list[dict[str, Any]],
    *,
    pair_limit: Any,
    sample_id: Any,
) -> list[dict[str, Any]]:
    """Independently reproduce the runner's mutually exclusive selection."""

    if pair_limit is not None and sample_id is not None:
        raise ValueError("pair_limit and sample_id are mutually exclusive")
    if sample_id is not None:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("config sample_id is invalid")
        selected = [row for row in rows if str(row.get("sample_id")) == sample_id]
        if len(selected) != 1:
            raise ValueError(
                f"config sample_id must select exactly one row: {sample_id}"
            )
        return selected
    if pair_limit is not None and (
        isinstance(pair_limit, bool)
        or not isinstance(pair_limit, int)
        or pair_limit <= 0
    ):
        raise ValueError("config pair_limit is invalid")
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        pair_ranks = pair_ranks[:pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [row for row in rows if int(row["pair_rank"]) in selected_ranks]
    kinds: dict[int, set[str]] = defaultdict(set)
    for row in selected:
        kinds[int(row["pair_rank"])].add(str(row["kind"]))
    invalid = {
        rank: values for rank, values in kinds.items() if values != {"real", "forged"}
    }
    if invalid:
        raise ValueError(f"canonical selection contains incomplete pairs: {invalid}")
    return selected


def _verify_source_tree(
    source_root: Path,
    *,
    pins: SimpleNamespace,
) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing NPR source root: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_SOURCE_COMMIT, "frozen NPR source commit")
    _require_equal(
        commit,
        pins.MODEL_SOURCE_COMMIT,
        "runner/source commit pin",
    )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty is None:
        raise ValueError("NPR source is not a readable git repository")
    if dirty:
        raise ValueError("NPR source has tracked modifications")
    for relative, digest in pins.SOURCE_FILES.items():
        _verify_hash(
            source_root / str(relative),
            digest,
            f"NPR source {relative}",
        )
    checkpoint = source_root / str(pins.CHECKPOINT["repo_relative_path"])
    _verify_hash(
        checkpoint,
        pins.CHECKPOINT["sha256"],
        "repository-bundled NPR checkpoint",
    )
    _require_equal(
        checkpoint.stat().st_size,
        int(pins.CHECKPOINT["bytes"]),
        "repository-bundled NPR checkpoint bytes",
    )
    history = _git_value(
        source_root,
        "log",
        "--format=%H",
        "--",
        str(pins.CHECKPOINT["repo_relative_path"]),
    )
    _require_equal(
        history,
        pins.CHECKPOINT["introduced_commit"],
        "NPR checkpoint introduction history",
    )
    license_files = [
        name
        for name in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE")
        if (source_root / name).exists()
    ]
    _require_equal(license_files, [], "NPR root license-file census")
    return {
        "commit": commit,
        "tracked_dirty": False,
        "checkpoint_history": history,
        "root_license_files": license_files,
    }


def _verify_hf_source_tree(
    hf_source_root: Path,
    *,
    pins: SimpleNamespace,
) -> dict[str, Any]:
    """Audit the pinned HF Space as corroboration, never as inference code."""

    if not hf_source_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned NPR Hugging Face Space source: {hf_source_root}"
        )
    _require_equal(
        pins.HF_SPACE_COMMIT,
        FROZEN_HF_SPACE_COMMIT,
        "analyzer/runner HF Space commit pin",
    )
    _require_equal(
        pins.HF_SOURCE_FILES,
        FROZEN_HF_SOURCE_FILES,
        "analyzer/runner HF Space source-file pins",
    )
    commit = _git_value(hf_source_root, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_HF_SPACE_COMMIT, "frozen HF Space commit")
    dirty = _git_value(
        hf_source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty is None:
        raise ValueError("NPR Hugging Face source is not a readable git repository")
    if dirty:
        raise ValueError("NPR Hugging Face source has tracked modifications")
    for relative, digest in FROZEN_HF_SOURCE_FILES.items():
        _verify_hash(
            hf_source_root / relative,
            digest,
            f"NPR Hugging Face source {relative}",
        )

    app_text = (hf_source_root / "app.py").read_text(encoding="utf-8")
    required_evidence = (
        "model_epoch_last_3090.pth",
        "transforms.ToTensor()",
        "transforms.Normalize(mean=[0.485, 0.456, 0.406]",
        "if w%2 == 1: img = img[:, :, :-1,:  ]",
        "if h%2 == 1: img = img[:, :, :  ,:-1]",
        "NPR = img - interpolate(img, 0.5)",
        "x.sigmoid()",
    )
    missing = [text for text in required_evidence if text not in app_text]
    if missing:
        raise ValueError(
            "pinned NPR Hugging Face app contract changed: " f"missing {missing}"
        )
    if "NPRmodel.eval()" in app_text:
        raise ValueError(
            "frozen finding changed: NPR Hugging Face app now calls eval()"
        )
    return {
        "commit": commit,
        "tracked_dirty": False,
        "source_files_validated": len(FROZEN_HF_SOURCE_FILES),
        "calls_model_eval": False,
        "role": "corroborating_evidence_only_not_executable_reference",
    }


def _expected_hf_source_record(
    hf_source_root: Path,
    *,
    pins: SimpleNamespace,
) -> dict[str, Any]:
    return {
        "space_url": pins.HF_SPACE_URL,
        "root": str(hf_source_root.resolve()),
        "commit": pins.HF_SPACE_COMMIT,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str((hf_source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in pins.HF_SOURCE_FILES.items()
        },
        "role": (
            "corroborating checkpoint and native-size preprocessing only; "
            "official GitHub test.py eval-mode remains the inference contract"
        ),
        "deployment_mode_defect": {
            "calls_model_eval": False,
            "impact": (
                "BatchNorm remains in train mode with batch-one statistics; "
                "the output is batch/input-composition dependent, mutates "
                "running buffers, and materially differs from checkpoint "
                "eval semantics; it is not reproduced or used as a "
                "sensitivity condition"
            ),
        },
        "supply_chain_note": (
            "original app downloads a mutable GitHub-main pickle checkpoint; "
            "this adapter instead hashes and weights-only loads the pinned "
            "GitHub checkpoint"
        ),
    }


def _import_official_resnet(source_root: Path, source_commit: str) -> Any:
    path = source_root / "networks" / "resnet.py"
    name = f"_claimforge_npr_audit_resnet_{source_commit[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official NPR model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_schema(
    state: Mapping[str, Any],
    *,
    torch_module: ModuleType,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    elements = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(
            tensor,
            torch_module.Tensor,
        ):
            raise ValueError("NPR checkpoint must map string keys to tensors")
        if tensor.is_complex():
            raise ValueError(f"NPR checkpoint tensor {key} is complex")
        if tensor.is_floating_point() and not bool(
            torch_module.isfinite(tensor).all().item()
        ):
            raise ValueError(f"NPR checkpoint tensor {key} is not finite")
        elements += int(tensor.numel())
        items.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
            }
        )
    return {
        "container": f"{type(state).__module__}.{type(state).__name__}",
        "entries": len(items),
        "elements": elements,
        "items_sha256": hashlib.sha256(stable_json(items).encode("utf-8")).hexdigest(),
    }


def _validate_adapter_contract(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    contract = _require_mapping(
        config.get("adapter_contract"),
        "config adapter contract",
    )
    required = {
        "eval/opensource/run_npr.py",
        "eval/opensource/npr_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
    }
    _require_equal(set(contract), required, "NPR adapter filenames")
    for relative, raw in contract.items():
        record = _require_mapping(raw, f"adapter {relative}")
        _require_equal(
            set(record),
            {"path", "bytes", "sha256"},
            f"adapter {relative} keys",
        )
        path = Path(str(record.get("path"))).resolve()
        _require_equal(
            path,
            (repo_root / relative).resolve(),
            f"adapter {relative} path",
        )
        _require_equal(
            record.get("bytes"),
            path.stat().st_size,
            f"adapter {relative} bytes",
        )
        _verify_hash(path, record.get("sha256"), f"adapter {relative}")
    return {"files_validated": len(contract), "paths": sorted(contract)}


def _checkpoint_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    source_root: Path,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> tuple[Path, OrderedDict[str, Any], Any, dict[str, Any], str]:
    assets = _require_mapping(manifest.get("assets"), "manifest assets")
    _require_equal(
        set(assets),
        {"checkpoint", "excluded_release_assets", "bundle_sha256"},
        "manifest asset keys",
    )
    record = _require_mapping(assets.get("checkpoint"), "checkpoint asset")
    _require_equal(
        set(record),
        set(pins.CHECKPOINT)
        | {
            "path",
            "actual_bytes",
            "actual_sha256",
            "serialization_safety",
            "schema",
        },
        "manifest checkpoint keys",
    )
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("manifest checkpoint has no explicit path")
    path = _anchored(Path(path_value), repo_root)
    _require_equal(
        path.name,
        pins.CHECKPOINT["filename"],
        "checkpoint filename",
    )
    _verify_hash(path, FROZEN_CHECKPOINT_SHA256, "NPR checkpoint")
    _require_equal(path.stat().st_size, FROZEN_CHECKPOINT_BYTES, "checkpoint bytes")
    _require_equal(
        record.get("actual_sha256"),
        FROZEN_CHECKPOINT_SHA256,
        "manifest checkpoint actual SHA-256",
    )
    _require_equal(
        record.get("actual_bytes"),
        FROZEN_CHECKPOINT_BYTES,
        "manifest checkpoint actual bytes",
    )
    for key, expected in pins.CHECKPOINT.items():
        _compare_nested(
            record.get(key),
            expected,
            label=f"manifest checkpoint {key}",
        )
    unsafe_api = getattr(
        getattr(torch_module, "serialization", None),
        "get_unsafe_globals_in_checkpoint",
        None,
    )
    if unsafe_api is None:
        raise RuntimeError("torch lacks safe checkpoint global inspection")
    unsafe = sorted(unsafe_api(path))
    _require_equal(unsafe, [], "NPR checkpoint unsafe globals")
    payload = torch_module.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, OrderedDict):
        raise ValueError("NPR checkpoint is not a flat OrderedDict")
    state = OrderedDict(payload)
    schema = _checkpoint_schema(state, torch_module=torch_module)
    _require_equal(
        schema["entries"],
        int(pins.CHECKPOINT["state_entries"]),
        "checkpoint state entries",
    )
    _require_equal(
        schema["elements"],
        int(pins.CHECKPOINT["state_elements"]),
        "checkpoint state elements",
    )
    _require_equal(record.get("schema"), schema, "manifest checkpoint schema")
    _require_equal(
        record.get("serialization_safety"),
        {
            "unsafe_globals": [],
            "weights_only": True,
            "map_location": "cpu",
        },
        "manifest checkpoint serialization safety",
    )
    module = _import_official_resnet(source_root, pins.MODEL_SOURCE_COMMIT)
    model = module.resnet50(num_classes=1)
    _require_equal(
        list(state),
        list(model.state_dict()),
        "checkpoint/model key order",
    )
    model.load_state_dict(state, strict=True)
    _require_equal(model.fc1.in_features, FEATURE_DIMENSION, "fc1 input width")
    _require_equal(model.fc1.out_features, 1, "fc1 output width")
    parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    _require_equal(
        parameters,
        int(pins.CHECKPOINT["trainable_parameters"]),
        "NPR model parameter count",
    )
    expected_bundle = hashlib.sha256(
        stable_json(
            {
                "source_commit": pins.MODEL_SOURCE_COMMIT,
                "source_files": pins.SOURCE_FILES,
                "hf_space_commit": pins.HF_SPACE_COMMIT,
                "hf_source_files": pins.HF_SOURCE_FILES,
                "checkpoint_sha256": pins.CHECKPOINT["sha256"],
                "checkpoint_schema": schema,
            }
        ).encode("utf-8")
    ).hexdigest()
    _require_equal(
        assets.get("bundle_sha256"),
        expected_bundle,
        "manifest asset bundle SHA-256",
    )
    _require_equal(
        assets.get("excluded_release_assets"),
        pins.EXCLUDED_RELEASE_ASSETS,
        "manifest excluded release assets",
    )
    return path, state, module, schema, expected_bundle


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _replay_runtime(manifest: Mapping[str, Any]) -> ReplayRuntime:
    """Recreate and verify the exact runner runtime and numerical flags."""

    runtime = _require_mapping(manifest.get("runtime"), "manifest runtime")
    config = _require_mapping(manifest.get("config"), "manifest config")
    _require_equal(
        config.get("runtime_evidence"),
        runtime,
        "config/manifest runtime evidence",
    )
    _require_equal(
        config.get("runtime_evidence_fingerprint"),
        _fingerprint(runtime),
        "runtime evidence fingerprint",
    )
    contract = _require_mapping(
        config.get("runtime_contract"),
        "config runtime contract",
    )
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace_config != ":4096:8":
        raise ValueError(
            "NPR deterministic replay requires " "CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    import torch

    _require_equal(runtime.get("python"), sys.version, "runtime Python")
    _require_equal(
        Path(str(runtime.get("executable"))).resolve(),
        Path(sys.executable).resolve(),
        "runtime executable",
    )
    _require_equal(runtime.get("platform"), platform.platform(), "runtime platform")
    _require_equal(runtime.get("torch"), torch.__version__, "runtime torch")
    _require_equal(
        runtime.get("torchvision"),
        _package_version("torchvision"),
        "runtime torchvision",
    )
    _require_equal(
        runtime.get("pillow"),
        _package_version("Pillow"),
        "runtime Pillow",
    )
    _require_equal(runtime.get("numpy"), np.__version__, "runtime NumPy")
    _require_equal(
        runtime.get("scikit_learn"),
        _package_version("scikit-learn"),
        "runtime scikit-learn",
    )
    _require_equal(runtime.get("seed"), int(contract["seed"]), "runtime seed")
    _require_equal(runtime.get("inference_dtype"), "torch.float32", "runtime dtype")
    _require_equal(runtime.get("autocast"), False, "runtime autocast")
    _require_equal(runtime.get("grad_enabled"), False, "runtime grad flag")
    requested = contract.get("device")
    _require_equal(runtime.get("device"), requested, "runtime device")
    if not isinstance(requested, str):
        raise ValueError("runtime device is not a string")
    device = torch.device(requested)
    expected_runtime_keys = {
        "python",
        "executable",
        "platform",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "scikit_learn",
        "device",
        "seed",
        "inference_dtype",
        "autocast",
        "grad_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
    }
    if device.type == "cuda":
        expected_runtime_keys.add("cuda")
    _require_equal(set(runtime), expected_runtime_keys, "runtime evidence keys")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recorded NPR CUDA runtime is unavailable")
        if device.index is None:
            raise ValueError("recorded NPR CUDA device lacks explicit index")
        if int(device.index) >= torch.cuda.device_count():
            raise RuntimeError(f"recorded CUDA device {device} is unavailable")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        cuda = _require_mapping(runtime.get("cuda"), "runtime CUDA")
        _require_equal(cuda.get("runtime"), torch.version.cuda, "CUDA runtime")
        _require_equal(cuda.get("device_index"), int(device.index), "CUDA index")
        _require_equal(cuda.get("device_name"), properties.name, "CUDA device name")
        _require_equal(
            cuda.get("total_memory_bytes"),
            int(properties.total_memory),
            "CUDA total memory",
        )
        _require_equal(
            cuda.get("capability"),
            [int(properties.major), int(properties.minor)],
            "CUDA capability",
        )
    elif device.type != "cpu":
        raise ValueError("recorded NPR device is neither CPU nor CUDA")
    random.seed(int(contract["seed"]))
    np.random.seed(int(contract["seed"]))
    torch.manual_seed(int(contract["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(contract["seed"]))
        torch.cuda.manual_seed_all(int(contract["seed"]))
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    _require_equal(
        runtime.get("cudnn"),
        {
            "enabled": False,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "runtime cuDNN flags",
    )
    _require_equal(
        runtime.get("matmul_allow_tf32"),
        False,
        "runtime matmul TF32",
    )
    _require_equal(
        runtime.get("deterministic_algorithms_enabled"),
        True,
        "runtime deterministic algorithms",
    )
    _require_equal(
        runtime.get("deterministic_algorithms_warn_only"),
        False,
        "runtime deterministic warn-only flag",
    )
    _require_equal(
        runtime.get("cublas_workspace_config"),
        ":4096:8",
        "runtime cuBLAS workspace configuration",
    )
    _require_equal(
        contract,
        {
            "device": str(device),
            "seed": 100,
            "dtype": "float32",
            "autocast": False,
            "cudnn_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "frozen runtime contract",
    )
    return ReplayRuntime(
        torch=torch,
        device=device,
        evidence={
            "device": str(device),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "pillow": _package_version("Pillow"),
            "numpy": np.__version__,
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32": False,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    )


@contextlib.contextmanager
def _loaded_model(
    *,
    module: Any,
    state: Mapping[str, Any],
    runtime: ReplayRuntime,
):
    model = module.resnet50(num_classes=1)
    model.load_state_dict(state, strict=True)
    model.to(device=runtime.device, dtype=runtime.torch.float32)
    model.eval()
    try:
        yield model
    finally:
        del model
        if runtime.device.type == "cuda":
            runtime.torch.cuda.empty_cache()


def _audit_preprocess_record(
    row: Mapping[str, Any],
    prepared: PreprocessedImage,
) -> None:
    _require_equal(
        row.get("preprocess_profile"),
        FROZEN_PROFILE,
        f"row {row.get('id')} preprocess profile",
    )
    _require_equal(
        row.get("preprocess"),
        prepared.audit,
        f"row {row.get('id')} independent preprocess",
    )


def _validate_result_identity(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
    config_fingerprint: str,
    checkpoint_id: str,
    visibility: Mapping[str, Any],
    row_label: str,
) -> None:
    expected = {
        "schema_version": "npr_detection_result_v1",
        "id": canonical["sample_id"],
        "sample_id": canonical["sample_id"],
        "rank": int(canonical["rank"]),
        "pair_rank": int(canonical["pair_rank"]),
        "task_id": canonical["task_id"],
        "kind": canonical["kind"],
        "label": int(canonical["label"]),
        "domain": canonical["domain"],
        "candidate": str(canonical.get("candidate", "mouse")),
        "dataset_id": str(canonical.get("dataset_id")),
        "input_sha256": canonical["canonical_sha256"],
        "input_width": int(canonical["width"]),
        "input_height": int(canonical["height"]),
        "preprocess_profile": FROZEN_PROFILE,
        "checkpoint_id": checkpoint_id,
        "config_fingerprint": config_fingerprint,
        "edit_visibility": visibility["edit_visibility"],
        "edit_visible_gt_fraction": visibility["edit_visible_gt_fraction"],
        "edit_visibility_evidence": dict(visibility),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
    }
    for key, value in expected.items():
        _compare_nested(
            row.get(key),
            value,
            label=f"{row_label} {key}",
            exact_mapping_keys=True,
        )
    value = row.get("input_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row_label} has no input_path")
    path = _anchored(Path(value), repo_root)
    canonical_path = _anchored(Path(str(canonical["canonical_path"])), repo_root)
    _require_equal(path, canonical_path, f"{row_label} input path")
    _verify_hash(path, canonical["canonical_sha256"], f"{row_label} input")
    with Image.open(path) as opened:
        _require_equal(
            list(opened.size),
            [int(canonical["width"]), int(canonical["height"])],
            f"{row_label} decoded size",
        )


def _validate_result_payload(
    row: Mapping[str, Any],
    *,
    row_label: str,
    repo_root: Path,
    run_dir: Path,
    torch_module: ModuleType,
) -> None:
    status = row.get("status")
    if status not in {"ok", "error"}:
        raise ValueError(f"{row_label} has invalid status {status!r}")
    expected_keys = _RESULT_IDENTITY_KEYS | (
        _SUCCESS_PAYLOAD_KEYS if status == "ok" else _ERROR_PAYLOAD_KEYS
    )
    _require_equal(set(row), expected_keys, f"{row_label} schema keys")
    _reject_t2_localization_or_joint(row, label=row_label)
    if status == "error":
        for key in ("completed_at", "error_type", "error", "traceback"):
            if not isinstance(row.get(key), str) or not row.get(key):
                raise ValueError(f"{row_label} has invalid {key}")
        forbidden = {
            "preprocess",
            "npr_feature_path",
            "raw_logit",
            "probability",
            "ai_score",
            "score",
            "classification",
            "t1",
            "manual_replay",
            "latency_ms",
        }.intersection(row)
        if forbidden:
            raise ValueError(
                f"{row_label} error payload claims success fields "
                f"{sorted(forbidden)}"
            )
        return
    if not isinstance(row.get("completed_at"), str) or not row.get("completed_at"):
        raise ValueError(f"{row_label} has invalid completed_at")
    raw = _require_finite(row.get("raw_logit"), f"{row_label} raw logit")
    _audit_score_fields(
        row,
        replay_raw_logit=raw,
        replay_probability=_float32_sigmoid(raw, torch_module),
        raw_tolerance=0.0,
    )
    _load_feature(row, repo_root=repo_root, run_dir=run_dir)
    for key in ("preprocess_latency_ms", "latency_ms"):
        latency = _require_finite(row.get(key), f"{row_label} {key}")
        if latency < 0.0:
            raise ValueError(f"{row_label} {key} is negative")
    peak = row.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError(f"{row_label} peak CUDA memory is invalid")


def _compare_summary_payload(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    for key, expected in recomputed.items():
        if key not in recorded:
            raise ValueError(f"summary lacks recomputed field {key}")
        _compare_nested(
            recorded[key],
            expected,
            label=f"summary.{key}",
            float_tolerance=1e-12,
            exact_mapping_keys=True,
        )


def recompute_summary(
    *,
    result_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    recorded_summary: Mapping[str, Any],
    independent_result_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = _require_mapping(manifest.get("config"), "manifest config")
    metrics = _require_mapping(config.get("metrics"), "config metrics")
    threshold = metrics.get("fixed_threshold")
    operator = metrics.get("threshold_operator")
    _require_equal(threshold, FIXED_THRESHOLD, "summary threshold")
    _require_equal(operator, THRESHOLD_OPERATOR, "summary threshold operator")
    iterations = metrics.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)
    seed = metrics.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("bootstrap sample count is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed is invalid")
    recomputed = summarize_npr_results(
        result_rows,
        expected_rows,
        threshold=float(threshold),
        bootstrap_samples=iterations,
        seed=seed,
    )
    _require_equal(
        set(recorded_summary),
        set(recomputed)
        | {
            "raw_logit_numerical_diagnostic",
            "run_id",
            "model",
            "model_slug",
            "checkpoint_id",
            "preprocess_profile",
            "config_fingerprint",
            "generated_at",
        },
        "summary keys",
    )
    _compare_summary_payload(recorded_summary, recomputed)
    diagnostic = summarize_npr_raw_logit_diagnostic(
        result_rows,
        expected_rows,
        bootstrap_samples=iterations,
        seed=seed,
    )
    _compare_nested(
        recorded_summary.get("raw_logit_numerical_diagnostic"),
        diagnostic,
        label="summary.raw_logit_numerical_diagnostic",
        float_tolerance=1e-12,
        exact_mapping_keys=True,
    )
    if independent_result_rows is not None:
        independent_diagnostic = summarize_npr_raw_logit_diagnostic(
            independent_result_rows,
            expected_rows,
            bootstrap_samples=iterations,
            seed=seed,
        )
        _compare_nested(
            recorded_summary.get("raw_logit_numerical_diagnostic"),
            independent_diagnostic,
            label=(
                "summary.raw_logit_numerical_diagnostic/"
                "independent_full_model_replay"
            ),
            float_tolerance=1e-12,
            exact_mapping_keys=True,
        )
    return {
        **recomputed,
        "raw_logit_numerical_diagnostic": diagnostic,
        "raw_logit_diagnostic_independent_full_model_match": (
            independent_result_rows is not None
        ),
    }


def _load_run_files(
    *,
    results_dir: Path,
    run_id: str,
) -> RunFiles:
    run_dir = results_dir / run_id
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    for path in (results_path, expected_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return RunFiles(
        run_dir=run_dir,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        rows=read_jsonl(results_path),
        expected=read_jsonl(expected_path),
        summary=_require_mapping(
            json.loads(summary_path.read_text(encoding="utf-8")),
            f"{run_id} summary",
        ),
        manifest=_require_mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            f"{run_id} manifest",
        ),
    )


def validate_provenance(
    *,
    repo_root: Path,
    source_root: Path,
    hf_source_root: Path,
    inputs_path: Path,
    all_inputs: list[dict[str, Any]],
    files: RunFiles,
) -> dict[str, Any]:
    """Verify immutable source/assets/config, rows, and physical file hashes."""

    pins = _load_runner_pins()
    _require_equal(
        pins.MODEL_SOURCE_COMMIT,
        FROZEN_SOURCE_COMMIT,
        "analyzer/runner source pin",
    )
    _require_equal(
        pins.CHECKPOINT["sha256"],
        FROZEN_CHECKPOINT_SHA256,
        "analyzer/runner checkpoint pin",
    )
    _require_equal(
        int(pins.CHECKPOINT["bytes"]),
        FROZEN_CHECKPOINT_BYTES,
        "analyzer/runner checkpoint bytes",
    )
    _require_equal(
        pins.PREPROCESS_PROFILE,
        FROZEN_PROFILE,
        "analyzer/runner profile pin",
    )
    _require_equal(
        pins.HF_SPACE_COMMIT,
        FROZEN_HF_SPACE_COMMIT,
        "analyzer/runner HF Space commit pin",
    )
    _require_equal(
        pins.HF_SOURCE_FILES,
        FROZEN_HF_SOURCE_FILES,
        "analyzer/runner HF source-file pins",
    )
    _require_equal(
        pins.FEATURE_DIMENSION,
        FEATURE_DIMENSION,
        "analyzer/runner feature dimension",
    )
    _require_equal(tuple(pins.IMAGE_MEAN), IMAGE_MEAN, "ImageNet mean")
    _require_equal(tuple(pins.IMAGE_STD), IMAGE_STD, "ImageNet std")

    manifest = files.manifest
    _require_equal(
        set(manifest),
        {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "repo_root",
            "config_fingerprint",
            "config",
            "source",
            "assets",
            "runtime",
            "dataset",
            "visibility_census",
            "outputs",
            "execution",
        },
        "manifest keys",
    )
    _require_equal(
        manifest.get("schema_version"),
        "npr_detection_run_manifest_v1",
        "manifest schema",
    )
    _require_equal(manifest.get("run_id"), files.run_dir.name, "manifest run ID")
    _require_equal(
        Path(str(manifest.get("repo_root"))).resolve(),
        repo_root.resolve(),
        "manifest repository root",
    )
    config = _require_mapping(manifest.get("config"), "manifest config")
    _require_equal(
        set(config),
        {
            "model",
            "model_slug",
            "model_arch",
            "adapter_contract",
            "source_commit",
            "source_files",
            "checkpoint_id",
            "checkpoint_sha256",
            "checkpoint_schema_sha256",
            "preprocess_profile",
            "preprocess_contract",
            "model_contract",
            "runtime_contract",
            "runtime_evidence",
            "runtime_evidence_fingerprint",
            "dataset",
            "metrics",
            "license",
            "checkpoint_selection_frozen_before_scores",
            "excluded_release_assets",
        },
        "config keys",
    )
    config_fingerprint = _require_sha256(
        manifest.get("config_fingerprint"),
        "config fingerprint",
    )
    _require_equal(
        config_fingerprint,
        _fingerprint(config),
        "config fingerprint",
    )
    for key, expected in {
        "model": pins.MODEL_NAME,
        "model_slug": pins.MODEL_SLUG,
        "model_arch": pins.MODEL_ARCH,
        "source_commit": pins.MODEL_SOURCE_COMMIT,
        "source_files": pins.SOURCE_FILES,
        "checkpoint_id": pins.CHECKPOINT["id"],
        "checkpoint_sha256": pins.CHECKPOINT["sha256"],
        "preprocess_profile": pins.PREPROCESS_PROFILE,
        "license": pins.LICENSE_RECORD,
        "checkpoint_selection_frozen_before_scores": True,
        "excluded_release_assets": pins.EXCLUDED_RELEASE_ASSETS,
    }.items():
        _compare_nested(config.get(key), expected, label=f"config {key}")
    _require_equal(
        config.get("preprocess_contract"),
        {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "resize": None,
            "crop": None,
            "batch_size": 1,
            "odd_dimension_policy": (
                "drop_last_bottom_row_and_or_right_column_before_NPR"
            ),
            "normalization_mean": list(IMAGE_MEAN),
            "normalization_std": list(IMAGE_STD),
        },
        "frozen preprocess contract",
    )
    _require_equal(
        config.get("model_contract"),
        {
            "npr": "x - nearest_upsample_2x(nearest_downsample_0.5x(x))",
            "npr_scale": 2.0 / 3.0,
            "feature_dimension": FEATURE_DIMENSION,
            "output": "one_raw_logit",
            "model_mode": "eval",
            "model_mode_source": "official_GitHub_test.py",
            "score": "torch_float32_sigmoid",
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "score_direction": "higher_means_fake",
            "valid_for_t2": False,
        },
        "frozen model contract",
    )
    metrics_contract = _require_mapping(config.get("metrics"), "config metrics")
    _require_equal(
        set(metrics_contract),
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "fixed_threshold",
            "threshold_operator",
            "raw_logit_numerical_diagnostic",
        },
        "config metrics keys",
    )
    _require_equal(
        metrics_contract.get("raw_logit_numerical_diagnostic"),
        {
            "preregistered_before_full_mouse_run": True,
            "preregistered_at": "2026-07-25",
            "trigger": (
                "CUDA pair-1 smoke produced finite raw logits near -170 "
                "whose official float32 sigmoid saturated to exact zero"
            ),
            "policy": (
                "always report raw-logit AUROC/AP, real-only 5% FPR, "
                "paired ranking/delta and pair bootstrap beside the "
                "official probability metrics; never choose whichever "
                "looks better and never replace the released >0.5 rule"
            ),
        },
        "pre-registered raw-logit diagnostic contract",
    )
    adapter = _validate_adapter_contract(config, repo_root=repo_root)
    source_local = _verify_source_tree(source_root, pins=pins)
    hf_source_local = _verify_hf_source_tree(hf_source_root, pins=pins)
    source = _require_mapping(manifest.get("source"), "manifest source")
    _require_equal(
        set(source),
        {
            "repo_url",
            "root",
            "commit",
            "tracked_dirty",
            "source_files",
            "checkpoint_history",
            "root_license_files",
            "license_record",
            "hf_space",
        },
        "manifest source keys",
    )
    _require_equal(source.get("repo_url"), pins.MODEL_REPO_URL, "source URL")
    _require_equal(
        Path(str(source.get("root"))).resolve(),
        source_root.resolve(),
        "source root",
    )
    _require_equal(source.get("commit"), pins.MODEL_SOURCE_COMMIT, "source commit")
    _require_equal(source.get("tracked_dirty"), False, "source dirty flag")
    expected_source_records = {
        relative: {
            "path": str((source_root / relative).resolve()),
            "sha256": digest,
        }
        for relative, digest in pins.SOURCE_FILES.items()
    }
    _require_equal(
        source.get("source_files"),
        expected_source_records,
        "source-file records",
    )
    _require_equal(
        source.get("checkpoint_history"),
        {
            "path": str(
                (source_root / pins.CHECKPOINT["repo_relative_path"]).resolve()
            ),
            "introduced_commit": pins.CHECKPOINT["introduced_commit"],
        },
        "source checkpoint history",
    )
    _require_equal(source.get("root_license_files"), [], "source license files")
    _require_equal(
        source.get("license_record"),
        pins.LICENSE_RECORD,
        "source license record",
    )
    _require_equal(
        source.get("hf_space"),
        _expected_hf_source_record(hf_source_root, pins=pins),
        "Hugging Face corroborating source record",
    )

    runtime = _replay_runtime(manifest)
    checkpoint_path, state, module, schema, bundle = _checkpoint_from_manifest(
        manifest,
        repo_root=repo_root,
        source_root=source_root,
        pins=pins,
        torch_module=runtime.torch,
    )
    _require_equal(
        config.get("checkpoint_schema_sha256"),
        schema["items_sha256"],
        "config checkpoint schema SHA-256",
    )

    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    _require_equal(
        set(dataset),
        {
            "manifest_path",
            "manifest_sha256",
            "inputs_path",
            "inputs_sha256",
            "expected_inputs_path",
            "expected_inputs_sha256",
            "selected_images",
            "selected_tasks",
        },
        "manifest dataset keys",
    )
    _require_equal(
        _anchored(Path(str(dataset.get("inputs_path"))), repo_root),
        inputs_path.resolve(),
        "manifest canonical inputs path",
    )
    inputs_digest = _verify_hash(
        inputs_path,
        dataset.get("inputs_sha256"),
        "canonical inputs JSONL",
    )
    release_path = _anchored(
        Path(str(dataset.get("manifest_path"))),
        repo_root,
    )
    release_digest = _verify_hash(
        release_path,
        dataset.get("manifest_sha256"),
        "canonical dataset manifest",
    )
    release = _require_mapping(
        json.loads(release_path.read_text(encoding="utf-8")),
        "canonical dataset manifest",
    )
    _require_equal(
        release.get("schema_version"),
        "claimforge_mouse_canonical_v1",
        "canonical release schema",
    )
    _require_equal(
        release.get("images"),
        len(all_inputs),
        "canonical release image count",
    )
    ranks = [int(row["rank"]) for row in all_inputs]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("canonical inputs are not in unique rank order")
    _require_equal(
        release.get("inputs_sha256"),
        inputs_digest,
        "canonical release input SHA-256",
    )
    _require_equal(
        _anchored(Path(str(release.get("inputs_path"))), repo_root),
        inputs_path.resolve(),
        "canonical release input path",
    )
    _require_equal(
        _anchored(Path(str(dataset.get("expected_inputs_path"))), repo_root),
        files.expected_path.resolve(),
        "expected-input snapshot path",
    )
    _verify_hash(
        files.expected_path,
        dataset.get("expected_inputs_sha256"),
        "expected-input snapshot",
    )
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(all_inputs):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"canonical input {index} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"canonical inputs repeat {sample_id}")
        by_id[sample_id] = row
    config_dataset = _require_mapping(config.get("dataset"), "config dataset")
    _require_equal(
        set(config_dataset),
        {
            "schema_version",
            "dataset_id",
            "inputs_sha256",
            "selected_ids",
            "selected_rows_sha256",
            "pair_limit",
            "sample_id",
        },
        "config dataset keys",
    )
    _require_equal(
        config_dataset.get("schema_version"),
        release.get("schema_version"),
        "config dataset schema",
    )
    _require_equal(
        config_dataset.get("dataset_id"),
        release.get("dataset_id"),
        "config dataset ID",
    )
    selected_ids = config_dataset.get("selected_ids")
    _require_equal(
        selected_ids,
        [str(row["sample_id"]) for row in files.expected],
        "config selected IDs",
    )
    expected_from_canonical = [by_id[str(sample_id)] for sample_id in selected_ids]
    _require_equal(
        files.expected,
        expected_from_canonical,
        "expected-input snapshot/canonical rows",
    )
    replayed_selection = _replay_input_selection(
        all_inputs,
        pair_limit=config_dataset.get("pair_limit"),
        sample_id=config_dataset.get("sample_id"),
    )
    _require_equal(
        files.expected,
        replayed_selection,
        "expected-input snapshot/replayed selection",
    )
    for canonical in files.expected:
        _load_gt_mask(canonical, repo_root=repo_root)
    _require_equal(
        config_dataset.get("selected_rows_sha256"),
        _selected_rows_sha256(files.expected),
        "selected-row fingerprint",
    )
    _require_equal(
        config_dataset.get("inputs_sha256"),
        inputs_digest,
        "config input SHA-256",
    )
    _require_equal(
        dataset.get("selected_images"),
        len(files.expected),
        "selected image count",
    )
    selected_tasks = {str(row["task_id"]) for row in files.expected}
    _require_equal(
        dataset.get("selected_tasks"),
        len(selected_tasks),
        "selected task count",
    )

    visibility = _pair_visibility(all_inputs, repo_root=repo_root)
    census = dict(
        Counter(visibility[task]["edit_visibility"] for task in selected_tasks)
    )
    _require_equal(manifest.get("visibility_census"), census, "visibility census")
    if len(files.expected) == 550 and len(selected_tasks) == 275:
        _require_equal(census, FULL_MOUSE_VISIBILITY_CENSUS, "full Mouse census")

    latest = _latest_by_id(files.rows)
    _require_equal(
        set(latest),
        {str(row["sample_id"]) for row in files.expected},
        "latest result IDs",
    )
    selected_by_id = {str(row["sample_id"]): row for row in files.expected}
    for line_number, row in enumerate(files.rows, start=1):
        row_id = row.get("id")
        if row_id not in selected_by_id:
            raise ValueError(f"physical result row {line_number} has extra ID")
        canonical = selected_by_id[str(row_id)]
        task_visibility = visibility[str(canonical["task_id"])]
        _validate_result_identity(
            row,
            canonical,
            repo_root=repo_root,
            config_fingerprint=config_fingerprint,
            checkpoint_id=str(pins.CHECKPOINT["id"]),
            visibility=task_visibility,
            row_label=f"physical result row {line_number}",
        )
        _validate_result_payload(
            row,
            row_label=f"physical result row {line_number}",
            repo_root=repo_root,
            run_dir=files.run_dir,
            torch_module=runtime.torch,
        )

    summary = files.summary
    for key, expected in {
        "run_id": files.run_dir.name,
        "model": pins.MODEL_NAME,
        "model_slug": pins.MODEL_SLUG,
        "checkpoint_id": pins.CHECKPOINT["id"],
        "preprocess_profile": FROZEN_PROFILE,
        "config_fingerprint": config_fingerprint,
    }.items():
        _require_equal(summary.get(key), expected, f"summary {key}")
    _reject_t2_localization_or_joint(summary, label="summary")
    _reject_t2_localization_or_joint(manifest, label="manifest")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    _require_equal(
        set(outputs),
        {
            "results_path",
            "summary_path",
            "feature_dir",
            "results_sha256",
            "summary_sha256",
            "feature_files",
        },
        "manifest output keys",
    )
    for key, path in {
        "results_path": files.results_path,
        "summary_path": files.summary_path,
        "feature_dir": files.run_dir / "features",
    }.items():
        _require_equal(
            _anchored(Path(str(outputs.get(key))), repo_root),
            path.resolve(),
            f"manifest output {key}",
        )
    results_digest = _verify_hash(
        files.results_path,
        outputs.get("results_sha256"),
        "NPR physical results",
    )
    summary_digest = _verify_hash(
        files.summary_path,
        outputs.get("summary_sha256"),
        "NPR runner summary",
    )
    feature_count = sum(
        1 for path in (files.run_dir / "features").glob("*.npy") if path.is_file()
    )
    _require_equal(outputs.get("feature_files"), feature_count, "feature count")
    expected_feature_names = {
        f"{sample_id}.npy"
        for sample_id, row in latest.items()
        if row.get("status") == "ok"
    }
    actual_feature_names = {
        path.name
        for path in (files.run_dir / "features").glob("*.npy")
        if path.is_file()
    }
    _require_equal(
        actual_feature_names,
        expected_feature_names,
        "successful-result feature file set",
    )
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    _require_equal(
        set(execution),
        {
            "new_successes",
            "resume_skips",
            "new_errors",
            "physical_result_rows",
        },
        "manifest execution keys",
    )
    _require_equal(
        execution.get("physical_result_rows"),
        len(files.rows),
        "execution physical result rows",
    )
    action_counts: dict[str, int] = {}
    for key in ("new_successes", "resume_skips", "new_errors"):
        value = execution.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"execution {key} is not a non-negative integer")
        action_counts[key] = value
    _require_equal(
        sum(action_counts.values()),
        len(files.expected),
        "execution selected-image action count",
    )
    expected_status = (
        "complete"
        if bool(
            _require_mapping(summary.get("coverage"), "summary coverage").get(
                "is_complete"
            )
        )
        else "incomplete"
    )
    _require_equal(manifest.get("status"), expected_status, "manifest status")
    return {
        "source_commit": pins.MODEL_SOURCE_COMMIT,
        "source_files_validated": len(pins.SOURCE_FILES),
        "source_audit": source_local,
        "hf_space_commit": pins.HF_SPACE_COMMIT,
        "hf_source_files_validated": len(pins.HF_SOURCE_FILES),
        "hf_source_audit": hf_source_local,
        "checkpoint_path": _relative_or_absolute(checkpoint_path, repo_root),
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "checkpoint_schema": schema,
        "asset_bundle_sha256": bundle,
        "adapter_contract": adapter,
        "config_fingerprint": config_fingerprint,
        "runtime": runtime.evidence,
        "canonical_inputs_sha256": inputs_digest,
        "canonical_manifest_sha256": release_digest,
        "expected_inputs_sha256": sha256_file(files.expected_path),
        "results_sha256": results_digest,
        "summary_sha256": summary_digest,
        "manifest_sha256": sha256_file(files.manifest_path),
        "physical_result_rows_validated": len(files.rows),
        "latest_rows_validated": len(latest),
        "selected_inputs_validated": len(files.expected),
        "visibility_census": census,
        "_state": state,
        "_module": module,
        "_runtime_object": runtime,
    }


def _strip_private_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value for key, value in provenance.items() if not str(key).startswith("_")
    }


def audit_artifacts(
    *,
    repo_root: Path,
    source_root: Path,
    all_inputs: list[dict[str, Any]],
    files: RunFiles,
    runtime: ReplayRuntime | None = None,
    module: Any | None = None,
    state: Mapping[str, Any] | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Independently preprocess and run the complete model for every row."""

    pins = _load_runner_pins()
    active_runtime = runtime or _replay_runtime(files.manifest)
    if model is None and (module is None or state is None):
        _, loaded_state, loaded_module, _, _ = _checkpoint_from_manifest(
            files.manifest,
            repo_root=repo_root,
            source_root=source_root,
            pins=pins,
            torch_module=active_runtime.torch,
        )
        module = loaded_module
        state = loaded_state
    model_context = (
        contextlib.nullcontext(model)
        if model is not None
        else _loaded_model(
            module=module,
            state=state,
            runtime=active_runtime,
        )
    )
    latest = _latest_by_id(files.rows)
    visibility = _pair_visibility(all_inputs, repo_root=repo_root)
    feature_differences: list[float] = []
    raw_differences: list[float] = []
    probability_differences: list[float] = []
    replay_probabilities: list[float] = []
    replay_raw_logits: list[float] = []
    replay_ids: list[str] = []
    independent_result_rows: list[dict[str, Any]] = []
    trim_census: Counter[str] = Counter()
    replayed = 0
    with model_context as active_model:
        if not hasattr(active_model, "fc1"):
            raise ValueError("NPR replay model has no fc1 layer")
        _require_equal(
            active_model.fc1.in_features,
            FEATURE_DIMENSION,
            "replay fc1 input dimension",
        )
        for canonical in files.expected:
            sample_id = str(canonical["sample_id"])
            row = latest[sample_id]
            _require_equal(row.get("status"), "ok", f"row {sample_id} status")
            path = _anchored(Path(str(canonical["canonical_path"])), repo_root)
            prepared = preprocess_image(
                path,
                torch_module=active_runtime.torch,
                profile_id=FROZEN_PROFILE,
            )
            _audit_preprocess_record(row, prepared)
            trim_census[
                f"bottom_{prepared.audit['trim_bottom']}_right_"
                f"{prepared.audit['trim_right']}"
            ] += 1
            persisted, _ = _load_feature(
                row,
                repo_root=repo_root,
                run_dir=files.run_dir,
            )
            captured: list[Any] = []

            def capture_fc1(_layer: Any, arguments: tuple[Any, ...]) -> None:
                if len(arguments) != 1:
                    raise RuntimeError("NPR fc1 hook received invalid arguments")
                captured.append(arguments[0].detach().clone())

            hook = active_model.fc1.register_forward_pre_hook(capture_fc1)
            try:
                with active_runtime.torch.inference_mode():
                    output = active_model(
                        prepared.tensor.unsqueeze(0).to(
                            device=active_runtime.device,
                            dtype=active_runtime.torch.float32,
                            non_blocking=False,
                        )
                    )
            finally:
                hook.remove()
            _require_equal(
                list(output.shape),
                [1, 1],
                f"row {sample_id} full-model output shape",
            )
            _require_equal(
                output.dtype,
                active_runtime.torch.float32,
                f"row {sample_id} full-model output dtype",
            )
            _require_equal(
                len(captured),
                1,
                f"row {sample_id} fc1 hook calls",
            )
            feature_tensor = captured[0]
            _require_equal(
                list(feature_tensor.shape),
                [1, FEATURE_DIMENSION],
                f"row {sample_id} full-model feature shape",
            )
            _require_equal(
                feature_tensor.dtype,
                active_runtime.torch.float32,
                f"row {sample_id} full-model feature dtype",
            )
            recomputed = np.ascontiguousarray(
                feature_tensor.squeeze(0).detach().cpu().numpy(),
                dtype=np.float32,
            )
            if not np.isfinite(recomputed).all():
                raise ValueError(f"row {sample_id} recomputed feature is non-finite")
            difference = float(np.max(np.abs(recomputed - persisted)))
            feature_differences.append(difference)
            if not np.array_equal(recomputed, persisted):
                raise ValueError(
                    f"row {sample_id} persisted feature differs from complete "
                    f"model replay (max abs {difference})"
                )

            artifact_tensor = active_runtime.torch.from_numpy(persisted.copy()).to(
                device=active_runtime.device,
                dtype=active_runtime.torch.float32,
            )[None, :]
            with active_runtime.torch.inference_mode():
                artifact_output = active_runtime.torch.nn.functional.linear(
                    artifact_tensor,
                    active_model.fc1.weight,
                    active_model.fc1.bias,
                )
                full_probability = active_runtime.torch.sigmoid(output)
                artifact_probability = active_runtime.torch.sigmoid(artifact_output)
            if not active_runtime.torch.equal(output, artifact_output):
                raise ValueError(
                    f"row {sample_id} persisted-feature fc1 replay differs "
                    "from complete model output"
                )
            if not active_runtime.torch.equal(
                full_probability,
                artifact_probability,
            ):
                raise ValueError(f"row {sample_id} persisted-feature sigmoid differs")
            raw = float(artifact_output.reshape(()).item())
            probability = float(artifact_probability.reshape(()).item())
            audited = _audit_score_fields(
                row,
                replay_raw_logit=raw,
                replay_probability=probability,
            )
            independent_row = copy.deepcopy(dict(row))
            independent_row.update(
                {
                    "raw_logit": raw,
                    "probability": probability,
                    "ai_score": probability,
                    "score": probability,
                    "classification_decision": bool(probability > FIXED_THRESHOLD),
                }
            )
            for key in ("classification", "t1"):
                nested = _require_mapping(
                    independent_row.get(key),
                    f"row {sample_id} independent {key}",
                )
                nested.update(
                    {
                        "raw_logit": raw,
                        "probability": probability,
                        "ai_score": probability,
                        "score": probability,
                        "decision": bool(probability > FIXED_THRESHOLD),
                    }
                )
            manual = _require_mapping(
                independent_row.get("manual_replay"),
                f"row {sample_id} independent manual replay",
            )
            manual.update(
                {
                    "raw_logit": raw,
                    "probability": probability,
                    "ai_score": probability,
                    "classification_decision": bool(probability > FIXED_THRESHOLD),
                }
            )
            independent_result_rows.append(independent_row)
            raw_differences.append(abs(float(row["raw_logit"]) - raw))
            probability_differences.append(abs(float(row["probability"]) - probability))
            replay_raw_logits.append(raw)
            replay_probabilities.append(probability)
            replay_ids.append(sample_id)
            expected_visibility = visibility[str(canonical["task_id"])]
            _compare_nested(
                row.get("edit_visibility_evidence"),
                expected_visibility,
                label=f"row {sample_id} visibility replay",
            )
            _require_equal(
                audited["decision"],
                bool(probability > 0.5),
                f"row {sample_id} strict decision",
            )
            replayed += 1
    raw_array = np.asarray(replay_raw_logits, dtype=np.float64)
    probability_array = np.asarray(replay_probabilities, dtype=np.float64)
    if not np.isfinite(raw_array).all():
        raise ValueError("independently replayed raw logits are non-finite")
    probability_decisions = probability_array > FIXED_THRESHOLD
    raw_decisions = raw_array > 0.0
    decision_disagreement = probability_decisions != raw_decisions
    disagreement_ids = [
        sample_id
        for sample_id, differs in zip(
            replay_ids,
            decision_disagreement.tolist(),
            strict=True,
        )
        if differs
    ]
    exact_zero = int(np.count_nonzero(probability_array == 0.0))
    exact_one = int(np.count_nonzero(probability_array == 1.0))
    return {
        "images_redecoded": replayed,
        "images_preprocessed_independently": replayed,
        "complete_model_forward_passes": replayed,
        "fc1_features_captured": replayed,
        "persisted_features_validated": replayed,
        "feature_shape": [FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_semantics": FEATURE_SEMANTICS,
        "maximum_feature_absolute_difference": max(
            feature_differences,
            default=0.0,
        ),
        "maximum_raw_logit_absolute_difference": max(
            raw_differences,
            default=0.0,
        ),
        "maximum_probability_absolute_difference": max(
            probability_differences,
            default=0.0,
        ),
        "persisted_feature_fc1_replayed": True,
        "raw_logit_dtype": "float32",
        "sigmoid_dtype": "float32",
        "strict_decision_replayed": "probability > 0.5",
        "raw_logit_decision_equivalence": {
            "probability_threshold": FIXED_THRESHOLD,
            "raw_logit_threshold": 0.0,
            "threshold_operator": THRESHOLD_OPERATOR,
            "all_images_equal": not disagreement_ids,
            "disagreement_images": len(disagreement_ids),
            "disagreement_ids": disagreement_ids,
            "note": (
                "a tiny positive float32 logit can round to sigmoid 0.5; "
                "released decisions always use probability > 0.5"
            ),
        },
        "probability_saturation": {
            "exact_zero_images": exact_zero,
            "exact_one_images": exact_one,
            "non_saturated_images": (replayed - exact_zero - exact_one),
            "unique_probability_values": int(np.unique(probability_array).size),
        },
        "preprocess_profile": FROZEN_PROFILE,
        "decoded_rgb_hash_replayed": True,
        "normalized_trimmed_tensor_hash_replayed": True,
        "npr_residual_hash_and_stats_replayed": True,
        "trim_census": dict(sorted(trim_census.items())),
        "runtime": active_runtime.evidence,
        "_independent_result_rows": independent_result_rows,
    }


def _config_without_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    dataset = _require_mapping(result.get("dataset"), "config dataset")
    for key in ("selected_ids", "selected_rows_sha256", "pair_limit", "sample_id"):
        dataset.pop(key, None)
    return result


def audit_prefix_reproducibility(
    *,
    repo_root: Path,
    full: RunFiles,
    prefix: RunFiles,
) -> dict[str, Any]:
    """Require a physically independent, exact deterministic smoke prefix."""

    if full.run_dir == prefix.run_dir:
        raise ValueError("prefix and full runs must have independent directories")
    full_config = _require_mapping(full.manifest.get("config"), "full config")
    prefix_config = _require_mapping(prefix.manifest.get("config"), "prefix config")
    _require_equal(
        _config_without_selection(prefix_config),
        _config_without_selection(full_config),
        "prefix/full non-selection config",
    )
    _require_equal(
        prefix.manifest.get("source"),
        full.manifest.get("source"),
        "prefix/full source",
    )
    _require_equal(
        prefix.manifest.get("assets"),
        full.manifest.get("assets"),
        "prefix/full assets",
    )
    _require_equal(
        prefix.manifest.get("runtime"),
        full.manifest.get("runtime"),
        "prefix/full runtime",
    )
    if not prefix.expected:
        raise ValueError("prefix expected-input snapshot is empty")
    _require_equal(
        prefix.expected,
        full.expected[: len(prefix.expected)],
        "prefix ordered input rows",
    )
    full_latest = _latest_by_id(full.rows)
    prefix_latest = _latest_by_id(prefix.rows)
    prefix_ids = [str(row["sample_id"]) for row in prefix.expected]
    _require_equal(set(prefix_latest), set(prefix_ids), "prefix latest IDs")
    for sample_id in prefix_ids:
        full_row = full_latest[sample_id]
        prefix_row = prefix_latest[sample_id]
        _require_equal(full_row.get("status"), "ok", f"full {sample_id} status")
        _require_equal(prefix_row.get("status"), "ok", f"prefix {sample_id} status")
        full_feature, full_path = _load_feature(
            full_row,
            repo_root=repo_root,
            run_dir=full.run_dir,
        )
        prefix_feature, prefix_path = _load_feature(
            prefix_row,
            repo_root=repo_root,
            run_dir=prefix.run_dir,
        )
        if full_path == prefix_path:
            raise ValueError(f"prefix {sample_id} reuses full feature file")
        if not np.array_equal(full_feature, prefix_feature):
            raise ValueError(f"prefix/full {sample_id} features differ")
        for field in _PREFIX_EXACT_FIELDS:
            _require_equal(
                prefix_row.get(field),
                full_row.get(field),
                f"prefix/full {sample_id} {field}",
            )
    return {
        "policy": (
            "independent run directories and feature files; exact ordered "
            "prefix; byte-identical float32 fc1 inputs, preprocessing, raw "
            "logits, sigmoid probabilities, aliases, and strict decisions"
        ),
        "full_run_id": full.run_dir.name,
        "prefix_run_id": prefix.run_dir.name,
        "prefix_images": len(prefix_ids),
        "samples_compared": len(prefix_ids),
        "full_results_sha256": sha256_file(full.results_path),
        "prefix_results_sha256": sha256_file(prefix.results_path),
        "full_manifest_sha256": sha256_file(full.manifest_path),
        "prefix_manifest_sha256": sha256_file(prefix.manifest_path),
        "copied_full_artifacts_rejected": True,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    source_root = _anchored(Path(args.npr_root), repo_root)
    hf_source_root = _anchored(Path(args.hf_source_root), repo_root)
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    all_inputs = read_jsonl(inputs_path)
    files = _load_run_files(results_dir=results_dir, run_id=args.run_id)
    provenance_internal = validate_provenance(
        repo_root=repo_root,
        source_root=source_root,
        hf_source_root=hf_source_root,
        inputs_path=inputs_path,
        all_inputs=all_inputs,
        files=files,
    )
    artifacts_internal = audit_artifacts(
        repo_root=repo_root,
        source_root=source_root,
        all_inputs=all_inputs,
        files=files,
        runtime=provenance_internal["_runtime_object"],
        module=provenance_internal["_module"],
        state=provenance_internal["_state"],
    )
    artifacts = _strip_private_provenance(artifacts_internal)
    recomputed = recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=files.manifest,
        recorded_summary=files.summary,
        independent_result_rows=artifacts_internal["_independent_result_rows"],
    )

    prefix_provenance: dict[str, Any] | None = None
    prefix_artifacts: dict[str, Any] | None = None
    prefix_recomputed: dict[str, Any] | None = None
    prefix_audit: dict[str, Any] | None = None
    if args.prefix_run_id:
        prefix_results_dir = _anchored(
            Path(args.prefix_results_dir or args.results_dir),
            repo_root,
        )
        prefix = _load_run_files(
            results_dir=prefix_results_dir,
            run_id=args.prefix_run_id,
        )
        prefix_internal = validate_provenance(
            repo_root=repo_root,
            source_root=source_root,
            hf_source_root=hf_source_root,
            inputs_path=inputs_path,
            all_inputs=all_inputs,
            files=prefix,
        )
        prefix_artifacts_internal = audit_artifacts(
            repo_root=repo_root,
            source_root=source_root,
            all_inputs=all_inputs,
            files=prefix,
            runtime=prefix_internal["_runtime_object"],
            module=prefix_internal["_module"],
            state=prefix_internal["_state"],
        )
        prefix_artifacts = _strip_private_provenance(prefix_artifacts_internal)
        prefix_recomputed = recompute_summary(
            result_rows=prefix.rows,
            expected_rows=prefix.expected,
            manifest=prefix.manifest,
            recorded_summary=prefix.summary,
            independent_result_rows=prefix_artifacts_internal[
                "_independent_result_rows"
            ],
        )
        prefix_provenance = _strip_private_provenance(prefix_internal)
        prefix_audit = audit_prefix_reproducibility(
            repo_root=repo_root,
            full=files,
            prefix=prefix,
        )

    analysis = {
        "schema_version": "npr_independent_analysis_v1",
        "run_id": args.run_id,
        "generated_at": utc_now(),
        "method": (
            "physical retry-history and file-hash audit; frozen source, "
            "checkpoint, adapter, config, and runtime verification; "
            "independent Pillow RGB decoding, ToTensor/ImageNet normalization, "
            "bottom/right even trim, NPR residual hash/stat replay, complete "
            "official model forward, 512d fc1-input comparison, persisted "
            "feature fc1 replay, float32 sigmoid, strict >0.5 decision, all "
            "score-alias validation, and npr_metrics summary recomputation"
        ),
        "result_history": summarize_result_history(files.rows),
        "provenance": _strip_private_provenance(provenance_internal),
        "artifact_replay": artifacts,
        "recomputed_summary": recomputed,
        "prefix_provenance": prefix_provenance,
        "prefix_artifact_replay": prefix_artifacts,
        "prefix_recomputed_summary": prefix_recomputed,
        "prefix_reproducibility": prefix_audit,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_outputs_rejected": True,
            "joint_outputs_rejected": True,
        },
        "status": "audited",
    }
    output = (
        _anchored(Path(args.output), repo_root)
        if args.output
        else files.run_dir / "analysis.json"
    )
    atomic_write_json(output, analysis)
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    parser.add_argument("--npr-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--hf-source-root", default=str(DEFAULT_HF_SOURCE_ROOT))
    parser.add_argument("--prefix-run-id")
    parser.add_argument("--prefix-results-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
