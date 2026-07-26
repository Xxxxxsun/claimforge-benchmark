#!/usr/bin/env python3
"""Run the frozen OpenSDI MaskCLIP adapter on Balanced250.

This v2 runner preserves the audited Mouse-v1 implementation.  It delegates
preprocessing, model loading, forward inference, native map restoration, and
fixed-threshold pixel metrics to the frozen MaskCLIP sources while adding the
strict Balanced250 selection, identity, provenance, resume, and artifact
contracts.
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
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_maskclip as legacy
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
    load_ground_truth,
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
from eval.opensource.maskclip_metrics import binary_pixel_metrics


RUN_MANIFEST_SCHEMA = "maskclip_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "maskclip_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "maskclip_balanced_runtime_summary_v2"

MODEL_NAME = "MaskCLIP"
MODEL_SLUG = "opensdi_maskclip_sd15"
MODEL_ARCHITECTURE = "OpenSDI_MaskCLIP_ViTL"
PREPROCESS_PROFILE = "official_opensdi_512_stretch_clip_normalize"
CHECKPOINT_ID = "MaskCLIP_sd15_20241109_08_53_19_epoch13"

CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">="
MASK_THRESHOLD = 0.5
MASK_THRESHOLD_OPERATOR = ">="
MODEL_SEED = 42
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/balanced250_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/maskclip")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/maskclip")
DEFAULT_FORMAL_RUN_ID = "maskclip_sd15_balanced250_v1_full1775_20260726"
DEFAULT_SMOKE_A_RUN_ID = "maskclip_sd15_balanced250_v1_smoke5x7_a_20260726"
DEFAULT_SMOKE_B_RUN_ID = "maskclip_sd15_balanced250_v1_smoke5x7_b_20260726"
DEFAULT_SMOKE_LIMIT = 5

MASKCLIP_CHECKPOINT_BYTES = 3_101_582_209
MAE_CHECKPOINT_BYTES = 343_249_461
CLIP_CHECKPOINT_BYTES = 932_768_134

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=CLASSIFICATION_THRESHOLD,
    threshold_operator=CLASSIFICATION_THRESHOLD_OPERATOR,
)

T2_SPEC: dict[str, Any] = {
    "valid_conditions": [
        "real",
        "local_mouse",
        "local_cat",
        "local_trash_can",
    ],
    "not_applicable_conditions": [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ],
    "model_probability_map": {
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "released_sigmoid_forged_probability",
        "saved_for": "all_successful_inputs",
    },
    "native_probability_map": {
        "restore": "opencv_inter_linear_to_native_size",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "saved_for": "t2_applicable_inputs_only",
    },
    "native_binary_mask": {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "encoding": "PNG_L_0_or_255",
        "saved_for": "t2_applicable_inputs_only",
    },
    "ground_truth": {
        "real": "all_zero",
        "local": "exact_diff",
        "fullframe": "not_applicable",
        "fullframe_conditioning_box_is_not_ground_truth": True,
    },
}

TASK_SCOPE: dict[str, Any] = {
    "primary_task": "T1_whole_image_AIGC_detection_and_T2_localization",
    "valid_for_t1": True,
    "valid_for_t2": True,
    "fullframe_t2_not_applicable": True,
    "native_dense_output": True,
}

ARTIFACT_CONTRACT: dict[str, Any] = {
    "score_map_model_512": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "released_sigmoid_forged_probability",
        "inventory": "one_per_successful_input",
    },
    "score_map_native": {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": "native_height_by_native_width",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "model_512_probability_map_restored_opencv_inter_linear",
        "inventory": "one_per_successful_t2_applicable_input",
    },
    "mask_native": {
        "format": "PNG",
        "mode": "L",
        "values": [0, 255],
        "shape": "native_height_by_native_width",
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "inventory": "one_per_successful_t2_applicable_input",
    },
}

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}

OPENSDI_SOURCE_FILES = (
    "model/MaskCLIP.py",
    "model/clip_utils.py",
    "model/mae.py",
    "model/prompt_learner.py",
)

ADAPTER_SOURCE_PATHS = (
    "eval/opensource/run_maskclip_balanced.py",
    "eval/opensource/analyze_maskclip_balanced.py",
    "eval/opensource/run_maskclip.py",
    "eval/opensource/analyze_maskclip_run.py",
    "eval/opensource/maskclip_metrics.py",
    "eval/opensource/balanced250_localization_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
)

_OK_ONLY_KEYS = frozenset(
    {
        "preprocess",
        "class_logits",
        "class_probabilities",
        "ai_score",
        "probability",
        "score",
        "score_margin",
        "score_semantics",
        "calibrated_probability",
        "classification_decision",
        "classification_threshold",
        "classification_threshold_operator",
        "score_map_model_path",
        "score_map_model_sha256",
        "score_map_model_bytes",
        "score_map_model_shape",
        "score_map_model_dtype",
        "score_map_model_semantics",
        "score_map_native_path",
        "score_map_native_sha256",
        "score_map_native_bytes",
        "score_map_native_shape",
        "score_map_native_dtype",
        "score_map_native_semantics",
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


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _valid_run_id(value: Any) -> str:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_.-"
    )
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(character not in allowed for character in value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(
            "run-id must be one safe ASCII path component (max 160 chars)"
        )
    return value


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


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local source participating in inference or later audit."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = (repo_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing MaskCLIP adapter source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_assets(
    *,
    opensdi_root: Path,
    checkpoint_path: Path,
    clip_checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify source provenance and all three weight files without CUDA."""

    root = opensdi_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing OpenSDI source root: {root}")
    commit = legacy._git_value(root, "rev-parse", "HEAD")
    if commit != legacy.MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"OpenSDI source commit mismatch: {commit} != "
            f"{legacy.MODEL_SOURCE_COMMIT}"
        )
    tracked_status = legacy._git_value(
        root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if tracked_status is None:
        raise ValueError("cannot inspect OpenSDI source status")
    if tracked_status:
        raise ValueError("OpenSDI tracked source tree is dirty")
    source_files: dict[str, dict[str, Any]] = {}
    for relative in OPENSDI_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing OpenSDI source file: {path}")
        source_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    source = {
        "repository": legacy.MODEL_REPO_URL,
        "root": str(root),
        "commit": commit,
        "tracked_dirty": False,
        "core_source_files": source_files,
    }

    mae_path = root / "weights/mae_pretrain_vit_base.pth"

    def asset(
        path: Path,
        *,
        label: str,
        expected_name: str,
        expected_sha256: str,
        expected_bytes: int,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved.name != expected_name:
            raise ValueError(f"{label} filename changed")
        if not resolved.is_file():
            raise FileNotFoundError(f"missing {label}: {resolved}")
        if resolved.stat().st_size != expected_bytes:
            raise ValueError(f"{label} byte size changed")
        digest = sha256_file(resolved)
        if digest != expected_sha256:
            raise ValueError(f"{label} SHA-256 changed")
        return {
            "path": str(resolved),
            "filename": expected_name,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            **dict(extra),
        }

    assets = {
        "maskclip": asset(
            checkpoint_path,
            label="MaskCLIP checkpoint",
            expected_name=legacy.CHECKPOINT_FILENAME,
            expected_sha256=legacy.CHECKPOINT_SHA256,
            expected_bytes=MASKCLIP_CHECKPOINT_BYTES,
            extra={
                "id": CHECKPOINT_ID,
                "repository": legacy.CHECKPOINT_REPO,
                "revision": legacy.CHECKPOINT_REVISION,
                "epoch": 13,
                "weights_only": True,
                "strict_model_load": True,
            },
        ),
        "mae_initialization": asset(
            mae_path,
            label="MAE initialization checkpoint",
            expected_name="mae_pretrain_vit_base.pth",
            expected_sha256=legacy.MAE_SHA256,
            expected_bytes=MAE_CHECKPOINT_BYTES,
            extra={"role": "OpenSDI_MaskCLIP_constructor_initialization"},
        ),
        "clip": asset(
            clip_checkpoint_path,
            label="OpenAI CLIP checkpoint",
            expected_name="ViT-L-14.pt",
            expected_sha256=legacy.CLIP_SHA256,
            expected_bytes=CLIP_CHECKPOINT_BYTES,
            extra={
                "git_commit": legacy.CLIP_GIT_COMMIT,
                "model": "ViT-L/14",
            },
        ),
    }
    return source, assets


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    """Freeze deterministic float32 inference before constructing the model."""

    cublas_workspace_config = _configure_cublas_workspace()
    import cv2
    import torch

    if device_text == "cpu":
        device = torch.device("cpu")
    elif device_text.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_text)
        torch.cuda.set_device(device)
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

    runtime: dict[str, Any] = {
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": _package_version("Pillow"),
        "imdlbenco": _package_version("IMDLBenCo"),
        "timm": _package_version("timm"),
        "seed": MODEL_SEED,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": cublas_workspace_config,
        "cudnn": {
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    return device, runtime


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.LOCAL_T1_T2)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
    ):
        raise ValueError("formal MaskCLIP Balanced250 selection drifted")
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
        capability=Capability.LOCAL_T1_T2,
        per_condition_limit=per_condition_limit,
    )
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    expected = {
        condition: per_condition_limit for condition in BALANCED_CONDITIONS
    }
    if (
        dict(counts) != expected
        or len(selected) != 7 * per_condition_limit
        or not all(row.get("panel") is True for row in selected)
    ):
        raise ValueError("MaskCLIP smoke selection drifted")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if release.release_kind != "balanced250":
        raise ValueError("MaskCLIP v2 requires a Balanced250 release")
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
            capability=Capability.LOCAL_T1_T2,
            sample_id=sample_id,
        )
        return spec, select_inputs(release, spec)
    raise ValueError(f"unsupported inference mode {mode!r}")


def _t2_semantics(row: Mapping[str, Any]) -> tuple[bool, str]:
    kind = row.get("gt_mask_kind")
    if kind == "all_zero":
        return True, "all_zero_real_false_positive_area"
    if kind == "exact_diff":
        return True, "exact_diff_local_insertion"
    if kind == "not_applicable":
        return False, "not_applicable_fullframe"
    raise ValueError("unsupported Balanced250 GT kind")


def result_task_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    applicable, _ = _t2_semantics(row)
    return {
        "valid_for_t1": True,
        "valid_for_t2": applicable,
        "native_dense_output": True,
        "model_512_output_role": (
            "t2_and_diagnostic" if applicable else "diagnostic_only"
        ),
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    applicable, semantics = _t2_semantics(row)
    return {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "checkpoint_id": CHECKPOINT_ID,
        "config_fingerprint": run_manifest_fingerprint,
        "task_scope": result_task_scope(row),
        "t2_applicable": applicable,
        "t2_target_semantics": semantics,
    }


def _preprocess_with_audit(
    input_path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    tensor, (width, height) = legacy.preprocess_image(input_path)
    value = np.ascontiguousarray(tensor)
    if (
        value.shape
        != (3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("MaskCLIP preprocessing emitted an invalid tensor")
    audit = {
        "profile": PREPROCESS_PROFILE,
        "decoded_size": [width, height],
        "tensor_shape": list(value.shape),
        "tensor_dtype": str(value.dtype),
        "tensor_sha256": _array_sha256(value),
        "input_resize": "opencv_inter_linear_stretch",
        "normalization_mean": legacy.CLIP_MEAN.tolist(),
        "normalization_std": legacy.CLIP_STD.tolist(),
    }
    return value, (width, height), audit


def _score_payload(
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    raw_logits = np.asarray(logits, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)
    if (
        raw_logits.shape != (2,)
        or probs.shape != (2,)
        or not np.isfinite(raw_logits).all()
        or not np.isfinite(probs).all()
        or float(probs.min()) < 0.0
        or float(probs.max()) > 1.0
        or not math.isclose(
            float(probs.sum(dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("MaskCLIP class logits/probabilities are invalid")
    score = float(probs[1])
    return {
        "class_logits": {
            "real": float(raw_logits[0]),
            "forged": float(raw_logits[1]),
        },
        "class_probabilities": {
            "real": float(probs[0]),
            "forged": score,
        },
        "ai_score": score,
        "probability": score,
        "score": score,
        "score_margin": float(
            np.float32(raw_logits[1]) - np.float32(raw_logits[0])
        ),
        "score_semantics": "softmax_probability_of_class_1_forged",
        "calibrated_probability": False,
        "classification_decision": score >= CLASSIFICATION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
    }


def artifact_paths(run_dir: Path, sample_id: str) -> dict[str, Path]:
    return {
        "model": run_dir / "score_maps_model_512" / f"{sample_id}.npy",
        "native": run_dir / "score_maps_native" / f"{sample_id}.npy",
        "mask": run_dir / "masks_native" / f"{sample_id}.png",
    }


def resolve_artifact_root(
    *,
    repo_root: Path,
    run_id: str,
    artifact_root: Path | None,
) -> Path:
    """Resolve one repository-local artifact root bound to the run ID."""

    valid_run_id = _valid_run_id(run_id)
    root = repo_root.resolve()
    candidate = _anchored(
        (
            DEFAULT_ARTIFACTS_DIR / valid_run_id
            if artifact_root is None
            else artifact_root
        ),
        root,
    )
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact root escapes repository") from error
    if candidate.name != valid_run_id:
        raise ValueError("artifact root must end with the exact run-id")
    if candidate == root:
        raise ValueError("artifact root must be below repository root")
    return candidate


def _artifact_fields(
    *,
    repo_root: Path,
    model_path: Path,
    model_map: np.ndarray,
    native_path: Path | None,
    native_map: np.ndarray | None,
    mask_path: Path | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "score_map_model_path": repo_relative(model_path, repo_root),
        "score_map_model_sha256": sha256_file(model_path),
        "score_map_model_bytes": model_path.stat().st_size,
        "score_map_model_shape": list(model_map.shape),
        "score_map_model_dtype": str(model_map.dtype),
        "score_map_model_semantics": (
            "released_sigmoid_forged_probability"
        ),
    }
    if native_path is None or native_map is None or mask_path is None:
        fields.update(
            {
                "score_map_native_path": None,
                "score_map_native_sha256": None,
                "score_map_native_bytes": None,
                "score_map_native_shape": None,
                "score_map_native_dtype": None,
                "score_map_native_semantics": None,
                "mask_path": None,
                "mask_sha256": None,
                "mask_bytes": None,
                "mask_shape": None,
                "mask_dtype": None,
                "mask_semantics": None,
            }
        )
    else:
        fields.update(
            {
                "score_map_native_path": repo_relative(
                    native_path,
                    repo_root,
                ),
                "score_map_native_sha256": sha256_file(native_path),
                "score_map_native_bytes": native_path.stat().st_size,
                "score_map_native_shape": list(native_map.shape),
                "score_map_native_dtype": str(native_map.dtype),
                "score_map_native_semantics": (
                    "model_512_probability_map_restored_opencv_inter_linear"
                ),
                "mask_path": repo_relative(mask_path, repo_root),
                "mask_sha256": sha256_file(mask_path),
                "mask_bytes": mask_path.stat().st_size,
                "mask_shape": list(native_map.shape),
                "mask_dtype": "uint8",
                "mask_semantics": (
                    "native_probability_map_ge_0_5_encoded_L_0_or_255"
                ),
            }
        )
    return fields


def _localization_payload(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    model_map: np.ndarray,
    native_map: np.ndarray,
) -> dict[str, Any]:
    target_native = load_ground_truth(row, repo_root)
    if target_native is None:
        raise ValueError("T2-applicable input has no ground truth")
    target_model = legacy.resize_target(
        target_native,
        legacy.MODEL_INPUT_SIZE,
        legacy.MODEL_INPUT_SIZE,
    )
    include_ap = row.get("gt_mask_kind") == "exact_diff"
    return {
        "model_512": binary_pixel_metrics(
            model_map,
            target_model,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
        "native": binary_pixel_metrics(
            native_map,
            target_native,
            MASK_THRESHOLD,
            include_ap=include_ap,
        ),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_score_payload(row: Mapping[str, Any], sample_id: str) -> None:
    logits = row.get("class_logits")
    probabilities = row.get("class_probabilities")
    if (
        not isinstance(logits, Mapping)
        or set(logits) != {"real", "forged"}
        or not isinstance(probabilities, Mapping)
        or set(probabilities) != {"real", "forged"}
    ):
        raise ValueError(f"{sample_id} class score payload changed")
    real_logit = _finite_number(logits["real"], f"{sample_id} real logit")
    forged_logit = _finite_number(
        logits["forged"],
        f"{sample_id} forged logit",
    )
    real_probability = _finite_number(
        probabilities["real"],
        f"{sample_id} real probability",
    )
    forged_probability = _finite_number(
        probabilities["forged"],
        f"{sample_id} forged probability",
    )
    shifted = np.asarray(
        [real_logit, forged_logit],
        dtype=np.float64,
    )
    shifted -= float(np.max(shifted))
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= float(expected_probabilities.sum())
    if (
        not 0.0 <= real_probability <= 1.0
        or not 0.0 <= forged_probability <= 1.0
        or not math.isclose(
            real_probability + forged_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not np.allclose(
            [real_probability, forged_probability],
            expected_probabilities,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise ValueError(f"{sample_id} softmax relationship changed")
    score = _finite_number(row.get("ai_score"), f"{sample_id} ai_score")
    if (
        score != forged_probability
        or row.get("probability") != score
        or row.get("score") != score
        or row.get("classification_decision")
        is not (score >= CLASSIFICATION_THRESHOLD)
        or row.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
        or row.get("score_semantics")
        != "softmax_probability_of_class_1_forged"
        or row.get("calibrated_probability") is not False
    ):
        raise ValueError(f"{sample_id} T1 score contract changed")
    expected_margin = float(
        np.float32(forged_logit) - np.float32(real_logit)
    )
    if row.get("score_margin") != expected_margin:
        raise ValueError(f"{sample_id} score margin changed")


def _resolve_exact_artifact(
    value: Any,
    *,
    repo_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    path = _anchored(Path(value), repo_root)
    if path != expected_path.resolve():
        raise ValueError(f"{label} path is not canonical")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing/non-regular {label}: {path}")
    return path


def _load_probability_map(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_path: Path,
    expected_shape: tuple[int, int],
    expected_semantics: str,
    repo_root: Path,
    sample_id: str,
) -> np.ndarray:
    path = _resolve_exact_artifact(
        row.get(f"{prefix}_path"),
        repo_root=repo_root,
        expected_path=expected_path,
        label=f"{sample_id} {prefix}",
    )
    if (
        row.get(f"{prefix}_sha256") != sha256_file(path)
        or row.get(f"{prefix}_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{sample_id} {prefix} file metadata changed")
    array = np.load(path, allow_pickle=False)
    if (
        array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
        or row.get(f"{prefix}_shape") != list(expected_shape)
        or row.get(f"{prefix}_dtype") != "float32"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"{sample_id} {prefix} array contract changed")
    return np.ascontiguousarray(array)


def _validate_preprocess(
    row: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    recompute: bool,
) -> None:
    sample_id = str(input_row["sample_id"])
    preprocess = row.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ValueError(f"{sample_id} preprocess audit is missing")
    expected_static = {
        "profile": PREPROCESS_PROFILE,
        "decoded_size": [
            int(input_row["width"]),
            int(input_row["height"]),
        ],
        "tensor_shape": [3, legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "tensor_dtype": "float32",
        "input_resize": "opencv_inter_linear_stretch",
        "normalization_mean": legacy.CLIP_MEAN.tolist(),
        "normalization_std": legacy.CLIP_STD.tolist(),
    }
    for key, value in expected_static.items():
        if preprocess.get(key) != value:
            raise ValueError(f"{sample_id} preprocess {key} changed")
    digest = preprocess.get("tensor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{sample_id} tensor SHA-256 is invalid")
    if recompute:
        input_path = _anchored(
            Path(str(input_row["canonical_path"])),
            repo_root,
        )
        _, _, expected = _preprocess_with_audit(input_path)
        if dict(preprocess) != expected:
            raise ValueError(f"{sample_id} preprocessing replay changed")


def _validate_ok_artifacts(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
) -> None:
    sample_id = str(input_row["sample_id"])
    paths = artifact_paths(run_dir, sample_id)
    model_map = _load_probability_map(
        attempt,
        prefix="score_map_model",
        expected_path=paths["model"],
        expected_shape=(legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE),
        expected_semantics="released_sigmoid_forged_probability",
        repo_root=repo_root,
        sample_id=sample_id,
    )
    applicable, _ = _t2_semantics(input_row)
    if not applicable:
        null_fields = {
            "score_map_native_path",
            "score_map_native_sha256",
            "score_map_native_bytes",
            "score_map_native_shape",
            "score_map_native_dtype",
            "score_map_native_semantics",
            "mask_path",
            "mask_sha256",
            "mask_bytes",
            "mask_shape",
            "mask_dtype",
            "mask_semantics",
            "localization",
        }
        if any(attempt.get(field) is not None for field in null_fields):
            raise ValueError(f"{sample_id} fullframe result claims T2 output")
        return

    expected_shape = (int(input_row["height"]), int(input_row["width"]))
    native_map = _load_probability_map(
        attempt,
        prefix="score_map_native",
        expected_path=paths["native"],
        expected_shape=expected_shape,
        expected_semantics=(
            "model_512_probability_map_restored_opencv_inter_linear"
        ),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    restored = legacy.restore_score_map(
        model_map,
        int(input_row["width"]),
        int(input_row["height"]),
    )
    if not np.array_equal(native_map, restored):
        raise ValueError(f"{sample_id} native probability restoration changed")

    mask_path = _resolve_exact_artifact(
        attempt.get("mask_path"),
        repo_root=repo_root,
        expected_path=paths["mask"],
        label=f"{sample_id} native mask",
    )
    if (
        attempt.get("mask_sha256") != sha256_file(mask_path)
        or attempt.get("mask_bytes") != mask_path.stat().st_size
        or attempt.get("mask_shape") != list(expected_shape)
        or attempt.get("mask_dtype") != "uint8"
        or attempt.get("mask_semantics")
        != "native_probability_map_ge_0_5_encoded_L_0_or_255"
        or attempt.get("mask_threshold") != MASK_THRESHOLD
        or attempt.get("mask_threshold_operator") != MASK_THRESHOLD_OPERATOR
    ):
        raise ValueError(f"{sample_id} native mask metadata changed")
    with Image.open(mask_path) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError(f"{sample_id} native mask encoding changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    if (
        pixels.shape != expected_shape
        or not np.isin(pixels, (0, 255)).all()
        or not np.array_equal(pixels == 255, native_map >= MASK_THRESHOLD)
    ):
        raise ValueError(f"{sample_id} native threshold mask changed")
    expected_localization = _localization_payload(
        row=input_row,
        repo_root=repo_root,
        model_map=model_map,
        native_map=native_map,
    )
    if attempt.get("localization") != expected_localization:
        raise ValueError(f"{sample_id} localization metrics changed")


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    verify_artifacts: bool,
    recompute_preprocess: bool = False,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    common_keys = set(expected) | {"status", "completed_at"}
    expected_keys = (
        common_keys | _OK_ONLY_KEYS
        if status == "ok"
        else common_keys
        | {"error_type", "error", "traceback"}
    )
    actual_keys = set(attempt)
    if actual_keys != expected_keys:
        raise ValueError(
            f"result attempt key set changed: "
            f"missing={sorted(expected_keys - actual_keys)[:1]}, "
            f"extra={sorted(actual_keys - expected_keys)[:1]}"
        )
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"result attempt field {key} drifted")
    sample_id = str(input_row["sample_id"])
    completed_at = attempt.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError(f"{sample_id} completed_at is invalid")
    if status == "error":
        error_type = attempt.get("error_type")
        error_text = attempt.get("error")
        error_traceback = attempt.get("traceback")
        if (
            not isinstance(error_type, str)
            or not error_type
            or not isinstance(error_text, str)
            or not isinstance(error_traceback, str)
            or not error_traceback
        ):
            raise ValueError(f"error result {sample_id} payload is invalid")
        return
    _validate_score_payload(attempt, sample_id)
    _validate_preprocess(
        attempt,
        input_row=input_row,
        repo_root=repo_root,
        recompute=recompute_preprocess,
    )
    latency = _finite_number(
        attempt.get("latency_ms"),
        f"{sample_id} latency_ms",
    )
    if latency < 0.0:
        raise ValueError(f"{sample_id} latency is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{sample_id} peak memory is invalid")
    if attempt.get("mask_threshold") != MASK_THRESHOLD or attempt.get(
        "mask_threshold_operator"
    ) != MASK_THRESHOLD_OPERATOR:
        raise ValueError(f"{sample_id} mask threshold changed")
    if verify_artifacts:
        _validate_ok_artifacts(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            run_dir=run_dir,
        )


def validate_artifact_inventory(
    *,
    run_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    expected_directories = {
        "score_maps_model_512",
        "score_maps_native",
        "masks_native",
    }
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError(
            f"missing or unsafe MaskCLIP artifact root: {run_dir}"
        )
    root_entries = list(run_dir.iterdir())
    actual_directories = {entry.name for entry in root_entries}
    if (
        actual_directories != expected_directories
        or any(not entry.is_dir() or entry.is_symlink() for entry in root_entries)
    ):
        raise ValueError(
            "MaskCLIP artifact root inventory mismatch: "
            f"missing={sorted(expected_directories - actual_directories)[:1]}, "
            f"extra={sorted(actual_directories - expected_directories)[:1]}"
        )
    by_id = {str(row["sample_id"]): row for row in selected}
    expected_model = {
        f"{sample_id}.npy"
        for sample_id, result in latest_by_sample_id.items()
        if result.get("status") == "ok"
    }
    expected_native = {
        f"{sample_id}.npy"
        for sample_id, result in latest_by_sample_id.items()
        if result.get("status") == "ok"
        and _t2_semantics(by_id[sample_id])[0]
    }
    expected_masks = {
        f"{sample_id}.png"
        for sample_id, result in latest_by_sample_id.items()
        if result.get("status") == "ok"
        and _t2_semantics(by_id[sample_id])[0]
    }
    directories = {
        "model_512": (
            run_dir / "score_maps_model_512",
            expected_model,
        ),
        "native": (
            run_dir / "score_maps_native",
            expected_native,
        ),
        "masks": (
            run_dir / "masks_native",
            expected_masks,
        ),
    }
    counts: dict[str, int] = {}
    for name, (directory, expected) in directories.items():
        actual = (
            {path.name for path in directory.iterdir() if path.is_file()}
            if directory.is_dir()
            else set()
        )
        if actual != expected:
            raise ValueError(
                f"MaskCLIP {name} inventory mismatch: "
                f"missing={sorted(expected - actual)[:1]}, "
                f"extra={sorted(actual - expected)[:1]}"
            )
        if directory.is_dir() and any(
            not path.is_file() for path in directory.iterdir()
        ):
            raise ValueError(f"MaskCLIP {name} inventory contains non-files")
        counts[name] = len(actual)
    return counts


def _prepare_artifact_root(artifact_root: Path) -> None:
    """Create the three fixed directories without accepting extra entries."""

    expected_directories = {
        "score_maps_model_512",
        "score_maps_native",
        "masks_native",
    }
    if artifact_root.exists():
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise ValueError(
                f"MaskCLIP artifact root is not a safe directory: "
                f"{artifact_root}"
            )
        for entry in artifact_root.iterdir():
            if (
                entry.name not in expected_directories
                or not entry.is_dir()
                or entry.is_symlink()
            ):
                raise ValueError(
                    "MaskCLIP artifact root contains an unexpected or unsafe "
                    f"entry: {entry}"
                )
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected_directories):
        (artifact_root / name).mkdir(exist_ok=True)


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "architecture": MODEL_ARCHITECTURE,
            "repository": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "model_setting_name": "ViTL",
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_sha256": legacy.CHECKPOINT_SHA256,
            "class_names": ["real", "forged"],
            "positive_class_index": 1,
        },
        "preprocess": {
            "profile": PREPROCESS_PROFILE,
            "precision": "float32",
            "batch_size": 1,
            "model_input_size": [
                legacy.MODEL_INPUT_SIZE,
                legacy.MODEL_INPUT_SIZE,
            ],
            "input_resize": "opencv_inter_linear_stretch",
            "normalization_mean": legacy.CLIP_MEAN.tolist(),
            "normalization_std": legacy.CLIP_STD.tolist(),
        },
        "score_spec": SCORE_SPEC.as_dict(),
        "t2_spec": T2_SPEC,
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "preflight": {
            "performed_before_accelerator_configuration": True,
            "dataset_files_verified": True,
            "source_commit_and_cleanliness_verified": True,
            "source_file_hashes_verified": True,
            "all_weight_file_hashes_and_sizes_verified": True,
            "cuda_used": False,
        },
        "artifact_contract": ARTIFACT_CONTRACT,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
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
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--opensdi-root",
        type=Path,
        default=legacy.DEFAULT_OPENSDI_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=legacy.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=legacy.DEFAULT_CLIP_CHECKPOINT,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "repository-local run artifact root; defaults to "
            "outputs/opensource/maskclip/<run-id>"
        ),
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    opensdi_root = _anchored(args.opensdi_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    clip_checkpoint_path = _anchored(args.clip_checkpoint, repo_root)

    # All release/source/weight verification intentionally precedes runtime
    # accelerator configuration.
    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    source, assets = verify_assets(
        opensdi_root=opensdi_root,
        checkpoint_path=checkpoint_path,
        clip_checkpoint_path=clip_checkpoint_path,
    )
    adapter_sources = adapter_source_contract(repo_root)
    if args.mode == "preflight":
        if (
            args.resume
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError("preflight accepts no selection/resume/CUDA options")
        report = {
            "schema_version": "maskclip_balanced_preflight_v2",
            "status": "passed",
            "dataset_id": release.dataset_id,
            "dataset_manifest_sha256": release.manifest_sha256,
            "verified_images": len(release.inputs),
            "source": source,
            "assets": assets,
            "adapter_sources": adapter_sources,
            "cuda_used": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if args.mode != "formal" and args.run_id is None:
        raise ValueError("smoke and single modes require an explicit --run-id")
    selection_spec, selected = select_mode_inputs(
        release,
        mode=args.mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    )

    device_text = args.device or "cuda:0"
    device, runtime = configure_runtime(device_text)
    import torch

    results_root = _anchored(args.results_dir, repo_root)
    run_dir = (results_root / run_id).resolve()
    try:
        run_dir.relative_to(results_root.resolve())
    except ValueError as error:
        raise ValueError("run directory escapes results root") from error
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    artifact_root = resolve_artifact_root(
        repo_root=repo_root,
        run_id=run_id,
        artifact_root=args.artifact_root,
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    if artifact_root.exists() and any(artifact_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"artifact root is non-empty; pass --resume: {artifact_root}"
        )

    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=args.mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        adapter_sources=adapter_sources,
        source=source,
        assets=assets,
        runtime=runtime,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    fingerprint = _fingerprint(immutable)

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
            or prior_manifest.get("run_id") != run_id
            or prior_manifest.get("fingerprint") != fingerprint
            or prior_manifest.get("immutable") != immutable
        ):
            raise ValueError("resume run manifest fingerprint/config drifted")
        if read_jsonl(expected_path) != selected:
            raise ValueError("resume expected input snapshot drifted")
        started_at = prior_manifest.get("started_at")
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
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        },
        "outputs": dict(immutable["outputs"]),
    }

    physical_before = read_jsonl(results_path) if results_path.is_file() else []
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_before:
        input_row = inputs_by_id[str(attempt["sample_id"])]
        _validate_runner_attempt(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            run_dir=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=attempt.get("status") == "ok",
            recompute_preprocess=False,
        )

    pending = [
        row
        for row in selected
        if latest_before.latest_by_sample_id.get(
            str(row["sample_id"]),
            {},
        ).get("status")
        != "ok"
    ]
    for row in selected:
        sample_id = str(row["sample_id"])
        prior = latest_before.latest_by_sample_id.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
            _validate_runner_attempt(
                prior,
                input_row=row,
                repo_root=repo_root,
                run_dir=artifact_root,
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
                verify_artifacts=True,
                recompute_preprocess=True,
            )
    _prepare_artifact_root(artifact_root)
    validate_artifact_inventory(
        run_dir=artifact_root,
        selected=selected,
        latest_by_sample_id=latest_before.latest_by_sample_id,
    )
    # Resume is fail-closed and non-mutating until all reusable evidence has
    # passed replay validation.  In particular, a corrupted artifact must not
    # downgrade a previously complete manifest to ``running``.
    atomic_write_json(manifest_path, manifest)

    model = None
    capture = None
    new_successes = 0
    resume_skips = len(selected) - len(pending)
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        if pending:
            model, loaded_device = legacy.load_model(
                opensdi_root=opensdi_root,
                checkpoint_path=checkpoint_path,
                clip_checkpoint_path=clip_checkpoint_path,
                device_name=str(device),
            )
            if str(loaded_device) != str(device):
                raise ValueError("MaskCLIP loaded on an unexpected device")
            capture = legacy.LogitCapture(model)
        for index, input_row in enumerate(pending, start=1):
            sample_id = str(input_row["sample_id"])
            paths = artifact_paths(artifact_root, sample_id)
            expected_ok = result_identity(
                input_row,
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
                valid_for_metrics=True,
            )
            try:
                input_path = _anchored(
                    Path(str(input_row["canonical_path"])),
                    repo_root,
                )
                tensor, (width, height), preprocess = _preprocess_with_audit(
                    input_path
                )
                if (width, height) != (
                    int(input_row["width"]),
                    int(input_row["height"]),
                ):
                    raise ValueError("canonical image dimensions changed")
                assert model is not None and capture is not None
                (
                    logits,
                    probabilities,
                    peak_bytes,
                    latency_ms,
                    raw_model_map,
                ) = legacy.infer_one(model, capture, device, tensor)
                model_map = np.ascontiguousarray(
                    raw_model_map,
                    dtype=np.float32,
                )
                if (
                    model_map.shape
                    != (legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE)
                    or not np.isfinite(model_map).all()
                    or float(model_map.min()) < 0.0
                    or float(model_map.max()) > 1.0
                ):
                    raise ValueError("MaskCLIP model probability map is invalid")
                legacy._atomic_save_npy(paths["model"], model_map)

                applicable, _ = _t2_semantics(input_row)
                native_map: np.ndarray | None = None
                localization: dict[str, Any] | None = None
                native_path: Path | None = None
                mask_path: Path | None = None
                if applicable:
                    native_map = np.ascontiguousarray(
                        legacy.restore_score_map(model_map, width, height),
                        dtype=np.float32,
                    )
                    native_path = paths["native"]
                    mask_path = paths["mask"]
                    legacy._atomic_save_npy(native_path, native_map)
                    legacy._atomic_save_mask(
                        mask_path,
                        native_map >= MASK_THRESHOLD,
                    )
                    localization = _localization_payload(
                        row=input_row,
                        repo_root=repo_root,
                        model_map=model_map,
                        native_map=native_map,
                    )
                artifact_fields = _artifact_fields(
                    repo_root=repo_root,
                    model_path=paths["model"],
                    model_map=model_map,
                    native_path=native_path,
                    native_map=native_map,
                    mask_path=mask_path,
                )
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    **_score_payload(logits, probabilities),
                    **artifact_fields,
                    "mask_threshold": MASK_THRESHOLD,
                    "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
                    "localization": localization,
                    "latency_ms": float(latency_ms),
                    "peak_cuda_memory_bytes": int(peak_bytes),
                }
                _validate_runner_attempt(
                    result,
                    input_row=input_row,
                    repo_root=repo_root,
                    run_dir=artifact_root,
                    run_id=run_id,
                    run_manifest_fingerprint=fingerprint,
                    verify_artifacts=True,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(pending)}] ok {sample_id} "
                    f"score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                new_errors += 1
                for path in paths.values():
                    path.unlink(missing_ok=True)
                error_result = {
                    **result_identity(
                        input_row,
                        run_id=run_id,
                        run_manifest_fingerprint=fingerprint,
                        valid_for_metrics=False,
                    ),
                    "status": "error",
                    "completed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(results_path, error_result)
                print(
                    f"[{index}/{len(pending)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
                    break
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if capture is not None:
            capture.close()
        del capture, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    physical_results = read_jsonl(results_path) if results_path.is_file() else []
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    for sample_id, attempt in latest.latest_by_sample_id.items():
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            run_dir=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=attempt.get("status") == "ok",
        )
    coverage = summarize_coverage(latest)
    inventories = validate_artifact_inventory(
        run_dir=artifact_root,
        selected=selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_and_artifact_inventory_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_maskclip_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": args.mode,
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "score_spec": SCORE_SPEC.as_dict(),
        "t2_spec": T2_SPEC,
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": coverage.as_dict(),
        "artifact_inventory": inventories,
        "generated_at": utc_now(),
    }
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
            "artifact_inventory": inventories,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "mode": args.mode,
                "coverage": coverage.as_dict(),
                "artifact_inventory": inventories,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError("MaskCLIP fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
