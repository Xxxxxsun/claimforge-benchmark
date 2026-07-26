#!/usr/bin/env python3
"""Independently audit a UniversalFakeDetect (UFD) inference run.

The runner persists the 768-dimensional OpenAI CLIP ViT-L/14 image feature.
This analyzer does not trust that artifact or any score in the result JSONL:
it verifies the immutable source/checkpoints and runtime, decodes and
preprocesses every selected image again, runs the pinned visual encoder on the
recorded CUDA device, and independently replays the released linear head,
float32 sigmoid, and strict ``score > 0.5`` decision.

UniversalFakeDetect is an image-level T1 detector.  Any claimed T2,
localization, pixel-mask, or S_joint output is rejected.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageOps

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.ufd_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_ufd_results,
)


DEFAULT_RUN_ID = (
    "universalfakedetect_clip_vit_l14_mouse_canonical_v1_full275_20260724"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/universalfakedetect")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_UFD_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "UniversalFakeDetect-76a0e3e60a8a"
)

FEATURE_DIMENSION = 768
FEATURE_DTYPE = np.dtype("<f4")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CROP_SIZE = 224
RESIZE_SHORT_SIDE = 256
RAW_LOGIT_ABSOLUTE_TOLERANCE = 1e-5
SIGMOID_ABSOLUTE_TOLERANCE = 1e-7

CURRENT_HEAD_VISIBILITY_CENSUS = {
    "none": 247,
    "partial": 14,
    "full": 14,
}
CHECKPOINT_ERA_VISIBILITY_CENSUS = {
    "none": 80,
    "partial": 33,
    "full": 162,
}
CURRENT_HEAD_CANONICAL_CROP_EQUALITY_CENSUS = {
    "pairs": 275,
    "crop_equal_pairs": 246,
    "crop_different_pairs": 29,
    "by_edit_visibility": {
        "none": {
            "pairs": 247,
            "crop_equal_pairs": 246,
            "crop_different_pairs": 1,
        },
        "partial": {
            "pairs": 14,
            "crop_equal_pairs": 0,
            "crop_different_pairs": 14,
        },
        "full": {
            "pairs": 14,
            "crop_equal_pairs": 0,
            "crop_different_pairs": 14,
        },
    },
}

_T2_OR_LOCALIZATION_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "score_map",
        "score_map_path",
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
    "clip_feature_sha256",
    "clip_feature_shape",
    "clip_feature_dtype",
    "clip_feature_semantics",
    "preprocess",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
)


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent preprocessing result before CUDA transfer."""

    tensor: Any
    crop_rgb: np.ndarray
    decoded_rgb_sha256: str
    crop_rgb_sha256: str
    tensor_sha256: str
    geometry: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """Exact torch/device runtime selected from the run manifest."""

    torch: ModuleType
    device: Any
    recorded_device: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class DetectionPair:
    """A complete real/forged result pair."""

    task_id: str
    real: dict[str, Any]
    forged: dict[str, Any]


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


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    digest = _require_sha256(expected, f"{label} expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {digest}")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint", "created_at", "adapter"}
    }
    return hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_bytes(repository: Path, revision_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "show", revision_path],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot read frozen UFD git object {revision_path}"
        ) from exc


def _module_pin(module: ModuleType, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return copy.deepcopy(getattr(module, name))
    raise RuntimeError(
        "UniversalFakeDetect runner lacks required audit pin "
        f"(one of {names})"
    )


def _load_runner_pins() -> SimpleNamespace:
    """Import only the benchmark runner's immutable public constants."""

    from eval.opensource import run_universalfakedetect as runner

    head_introduction = _module_pin(runner, "HEAD_INTRO_COMMIT")
    head_era_validate = _module_pin(runner, "HEAD_ERA_VALIDATE_SHA256")
    checkpoint_profile = _module_pin(runner, "CHECKPOINT_ERA_PROFILE")
    resize_removal = _module_pin(runner, "RESIZE_REMOVAL_COMMIT")
    current_profile = _module_pin(runner, "CURRENT_PROFILE")
    source_commit = _module_pin(runner, "MODEL_SOURCE_COMMIT")
    source_checkpoint_drift = {
        "head_introduction_commit": head_introduction,
        "head_era_validate_sha256": head_era_validate,
        "checkpoint_era_profile": checkpoint_profile,
        "resize_removal_commit": resize_removal,
        "current_profile": current_profile,
        "current_source_commit": source_commit,
        "ambiguity": (
            "the release does not state whether the linear head should be "
            "evaluated with checkpoint-era Resize(256) or current native "
            "CenterCrop(224)"
        ),
        "resolution": (
            "profile is mandatory and immutable; the two profiles are "
            "reported as separate experimental conditions"
        ),
    }
    return SimpleNamespace(
        MODEL_NAME=_module_pin(runner, "MODEL_NAME"),
        MODEL_SLUG=_module_pin(runner, "MODEL_SLUG"),
        MODEL_REPO_URL=_module_pin(runner, "MODEL_REPO_URL"),
        MODEL_SOURCE_COMMIT=source_commit,
        MODEL_ARCH=_module_pin(runner, "MODEL_ARCH"),
        PAPER_URL=_module_pin(runner, "PAPER_URL"),
        LICENSE_RECORD=_module_pin(runner, "LICENSE_RECORD"),
        SOURCE_FILES=_module_pin(runner, "SOURCE_FILES"),
        HEAD_CHECKPOINT=_module_pin(
            runner,
            "HEAD_CHECKPOINT",
            "HEAD_CHECKPOINT_FILE",
        ),
        BACKBONE_CHECKPOINT=_module_pin(
            runner,
            "BACKBONE_CHECKPOINT",
            "BACKBONE_CHECKPOINT_FILE",
        ),
        PREPROCESS_PROFILES=_module_pin(runner, "PREPROCESS_PROFILES"),
        SOURCE_CHECKPOINT_DRIFT=source_checkpoint_drift,
        HEAD_INTRO_COMMIT=head_introduction,
        HEAD_ERA_VALIDATE_SHA256=head_era_validate,
        RESIZE_REMOVAL_COMMIT=resize_removal,
        CURRENT_PROFILE=current_profile,
        CHECKPOINT_ERA_PROFILE=checkpoint_profile,
        FEATURE_DIMENSION=int(
            getattr(
                runner,
                "FEATURE_DIMENSION",
                getattr(runner, "CLIP_FEATURE_DIMENSION", FEATURE_DIMENSION),
            )
        ),
        CLASSIFICATION_THRESHOLD=float(
            getattr(
                runner,
                "CLASSIFICATION_THRESHOLD",
                getattr(runner, "FIXED_THRESHOLD", FIXED_THRESHOLD),
            )
        ),
        CLASSIFICATION_THRESHOLD_OPERATOR=str(
            getattr(
                runner,
                "CLASSIFICATION_THRESHOLD_OPERATOR",
                THRESHOLD_OPERATOR,
            )
        ),
    )


def _normalise_source_files(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is not a non-empty mapping")
    result: dict[str, str] = {}
    for path, raw in value.items():
        digest = raw.get("sha256") if isinstance(raw, Mapping) else raw
        result[str(path)] = _require_sha256(
            digest,
            f"{label} {path} SHA-256",
        )
    return result


def _verify_source_tree(
    source_root: Path,
    *,
    expected_commit: str,
    expected_files: Mapping[str, str],
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing UFD source root: {source_root}")
    _require_equal(
        _git_value(source_root, "rev-parse", "HEAD"),
        expected_commit,
        "UFD checked-out source commit",
    )
    status = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if status is None:
        raise ValueError("UFD source root is not a readable git repository")
    if status:
        raise ValueError("UFD source root has tracked modifications")
    for relative, digest in expected_files.items():
        _verify_hash(source_root / relative, digest, f"UFD source {relative}")


def _profile_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("runner PREPROCESS_PROFILES is not a mapping")
    result: dict[str, dict[str, Any]] = {}
    for key, raw in value.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"preprocess profile {key} is not a mapping")
        profile_id = str(raw.get("profile_id", raw.get("id", key)))
        if profile_id in result:
            raise ValueError(f"duplicate preprocess profile {profile_id}")
        result[profile_id] = copy.deepcopy(dict(raw))
    return result


def _profile_kind(profile_id: str, pins: SimpleNamespace) -> str:
    profiles = _profile_mapping(pins.PREPROCESS_PROFILES)
    if profile_id not in profiles:
        raise ValueError(f"unknown preprocess profile {profile_id!r}")
    searchable = (
        profile_id + " " + stable_json(profiles[profile_id])
    ).lower()
    if any(token in searchable for token in ("checkpoint", "resize256", "resize_256")):
        return "checkpoint_era_resize256"
    if any(token in searchable for token in ("native", "current_head", "current-head")):
        return "current_head_native"
    resize = profiles[profile_id].get("resize")
    if isinstance(resize, Mapping) and resize.get("enabled") is True:
        return "checkpoint_era_resize256"
    if resize is False or (
        isinstance(resize, Mapping) and resize.get("enabled") is False
    ):
        return "current_head_native"
    raise ValueError(
        f"preprocess profile {profile_id!r} does not identify its geometry"
    )


def _center_crop_start(length: int, size: int = CROP_SIZE) -> int:
    return int(round((int(length) - int(size)) / 2.0))


def _resize_short_side_dimensions(
    width: int,
    height: int,
    short_side: int = RESIZE_SHORT_SIDE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    short = min(width, height)
    long = max(width, height)
    new_long = int(int(short_side) * long / short)
    if width <= height:
        return int(short_side), new_long
    return new_long, int(short_side)


def _preprocess_geometry(
    width: int,
    height: int,
    *,
    profile_kind: str,
    profile_id: str | None = None,
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if profile_kind == "current_head_native":
        resized_width, resized_height = int(width), int(height)
        resize_enabled = False
        default_profile_id = "current_head_native_center_crop224"
    elif profile_kind == "checkpoint_era_resize256":
        resized_width, resized_height = _resize_short_side_dimensions(
            int(width),
            int(height),
        )
        resize_enabled = True
        default_profile_id = "checkpoint_era_resize256_center_crop224"
    else:
        raise ValueError(f"unsupported profile kind {profile_kind!r}")
    pad_left = (
        (CROP_SIZE - resized_width) // 2
        if CROP_SIZE > resized_width
        else 0
    )
    pad_top = (
        (CROP_SIZE - resized_height) // 2
        if CROP_SIZE > resized_height
        else 0
    )
    pad_right = (
        (CROP_SIZE - resized_width + 1) // 2
        if CROP_SIZE > resized_width
        else 0
    )
    pad_bottom = (
        (CROP_SIZE - resized_height + 1) // 2
        if CROP_SIZE > resized_height
        else 0
    )
    padded_width = resized_width + pad_left + pad_right
    padded_height = resized_height + pad_top + pad_bottom
    left = _center_crop_start(padded_width)
    top = _center_crop_start(padded_height)
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE
    visible_left = max(0.0, float(left - pad_left))
    visible_top = max(0.0, float(top - pad_top))
    visible_right = min(
        float(resized_width),
        float(right - pad_left),
    )
    visible_bottom = min(
        float(resized_height),
        float(bottom - pad_top),
    )
    effective_native_crop = [
        visible_left * int(width) / resized_width,
        visible_top * int(height) / resized_height,
        visible_right * int(width) / resized_width,
        visible_bottom * int(height) / resized_height,
    ]
    return {
        "profile_id": profile_id or default_profile_id,
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "native_size": [int(width), int(height)],
        "resize": {
            "enabled": resize_enabled,
            "source_size": [int(width), int(height)],
            "destination_size": [resized_width, resized_height],
            "short_side": RESIZE_SHORT_SIDE if resize_enabled else None,
            "interpolation": "PIL_BILINEAR" if resize_enabled else None,
            "antialias": True if resize_enabled else None,
            "rounding": (
                "torchvision_int_truncation_for_long_side"
                if resize_enabled
                else None
            ),
        },
        "center_crop": {
            "input_size": [resized_width, resized_height],
            "padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
            "padding_fill": 0,
            "padded_size": [padded_width, padded_height],
            "start_xy": [left, top],
            "size": [CROP_SIZE, CROP_SIZE],
            "end_xy": [right, bottom],
            "rounding": "int(round((dimension-crop)/2.0))",
        },
        "effective_native_crop_xyxy": effective_native_crop,
        "pixel_center_mapping": (
            "if resized: d=(native_index+0.5)*resized_size/native_size-0.5; "
            "else d=native_index; then d+=center_crop_padding_left_or_top"
        ),
        "normalize": {
            "to_tensor_scale": "uint8_div_255_to_float32",
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
        },
    }


def preprocess_image(
    image_path: Path,
    *,
    profile_kind: str,
    torch_module: ModuleType,
    profile_id: str | None = None,
) -> PreprocessedImage:
    """Independently reproduce one of the two frozen UFD profiles."""

    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded_rgb = np.ascontiguousarray(
            np.asarray(rgb, dtype=np.uint8)
        )
        geometry = _preprocess_geometry(
            width,
            height,
            profile_kind=profile_kind,
            profile_id=profile_id,
        )
        resize = geometry["resize"]
        if resize["enabled"]:
            rgb = rgb.resize(
                tuple(resize["destination_size"]),
                resample=Image.Resampling.BILINEAR,
            )
        crop = geometry["center_crop"]
        padding = crop["padding_ltrb"]
        if any(padding):
            rgb = ImageOps.expand(rgb, border=tuple(padding), fill=0)
        left, top = crop["start_xy"]
        rgb = rgb.crop(
            (left, top, left + CROP_SIZE, top + CROP_SIZE)
        )
        crop_rgb = np.asarray(rgb, dtype=np.uint8).copy()
    if crop_rgb.shape != (CROP_SIZE, CROP_SIZE, 3):
        raise ValueError(
            f"preprocess produced invalid crop shape {crop_rgb.shape}"
        )
    tensor = (
        torch_module.from_numpy(crop_rgb)
        .permute(2, 0, 1)
        .to(dtype=torch_module.float32)
        .div(255.0)
    )
    mean = torch_module.tensor(CLIP_MEAN, dtype=torch_module.float32)[:, None, None]
    std = torch_module.tensor(CLIP_STD, dtype=torch_module.float32)[:, None, None]
    tensor = tensor.sub(mean).div(std).contiguous()
    tensor_array = tensor.detach().cpu().numpy()
    return PreprocessedImage(
        tensor=tensor,
        crop_rgb=crop_rgb,
        decoded_rgb_sha256=_array_sha256(decoded_rgb),
        crop_rgb_sha256=_array_sha256(crop_rgb),
        tensor_sha256=_array_sha256(tensor_array),
        geometry=geometry,
    )


def _edit_box_visibility(
    edit_region: list[int],
    native_crop: list[float],
) -> dict[str, Any]:
    box = [float(value) for value in edit_region]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("edit_region_xyxy has non-positive area")
    left = max(box[0], native_crop[0])
    top = max(box[1], native_crop[1])
    right = min(box[2], native_crop[2])
    bottom = min(box[3], native_crop[3])
    intersection = (
        [left, top, right, bottom]
        if right > left and bottom > top
        else None
    )
    area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    fraction = min(1.0, max(0.0, visible_area / area))
    category = (
        "none"
        if fraction == 0.0
        else "full"
        if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else "partial"
    )
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": native_crop,
        "intersection_xyxy": intersection,
        "edit_area": area,
        "visible_area": visible_area,
        "visible_fraction": fraction,
        "category": category,
        "basis": (
            "continuous_edit_box_area_intersection_with_effective_native_crop"
        ),
    }


def _visibility_from_exact_gt(
    forged_input: Mapping[str, Any],
    *,
    repo_root: Path,
    profile_kind: str,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Recompute pre-canonical GT pixel-center visibility.

    Mouse exact-diff GT is measured on decoded source RGB before the real and
    forged images are independently canonicalized as JPEG quality 95.  This
    visibility is therefore an input-location condition, not evidence that two
    decoded canonical crops are byte-identical.
    """

    width = int(forged_input["width"])
    height = int(forged_input["height"])
    mask_value = forged_input.get("gt_mask_path")
    if not isinstance(mask_value, str) or not mask_value:
        raise ValueError(
            f"forged input {forged_input.get('sample_id')} has no GT mask"
        )
    mask_path = _anchored(Path(mask_value), repo_root)
    _verify_hash(
        mask_path,
        forged_input.get("gt_mask_sha256"),
        f"forged input {forged_input.get('sample_id')} GT mask",
    )
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened)
    if mask.ndim == 3:
        if not np.array_equal(mask, mask[..., :1]):
            raise ValueError("GT mask channels are not identical")
        mask = mask[..., 0]
    if mask.shape != (height, width):
        raise ValueError(
            f"GT mask shape {mask.shape} != canonical {(height, width)}"
        )
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("paired forged exact-difference mask is not binary 0/255")
    ys, xs = np.nonzero(mask == 255)
    total = int(xs.size)
    if total <= 0:
        raise ValueError("paired forged exact-difference mask is empty")
    recorded_total = forged_input.get("gt_positive_pixels")
    if recorded_total is not None:
        _require_equal(
            total,
            int(recorded_total),
            "forged GT positive-pixel count",
        )

    geometry = _preprocess_geometry(
        width,
        height,
        profile_kind=profile_kind,
        profile_id=profile_id,
    )
    crop = geometry["center_crop"]
    left, top = crop["start_xy"]
    pad_left, pad_top, _, _ = crop["padding_ltrb"]
    if profile_kind == "current_head_native":
        mapped_x = xs.astype(np.float64) + pad_left
        mapped_y = ys.astype(np.float64) + pad_top
        influence_claim = (
            "pre_canonical_input_condition_only; independent JPEG_q95 "
            "canonicalization can diffuse a source edit across the crop "
            "boundary, so none does not imply canonical crop equality"
        )
    elif profile_kind == "checkpoint_era_resize256":
        new_width, new_height = geometry["resize"]["destination_size"]
        mapped_x = (
            (xs.astype(np.float64) + 0.5) * new_width / width
            - 0.5
            + pad_left
        )
        mapped_y = (
            (ys.astype(np.float64) + 0.5) * new_height / height
            - 0.5
            + pad_top
        )
        influence_claim = (
            "pre_canonical_pixel_center_diagnostic_only; independent JPEG_q95 "
            "canonicalization and Pillow bilinear antialiasing may diffuse "
            "changes across the crop boundary"
        )
    else:
        raise ValueError(f"unsupported profile kind {profile_kind!r}")

    visible_mask = (
        (mapped_x >= left)
        & (mapped_x < left + CROP_SIZE)
        & (mapped_y >= top)
        & (mapped_y < top + CROP_SIZE)
    )
    visible = int(np.count_nonzero(visible_mask))
    fraction = visible / total
    category = (
        "none" if visible == 0 else "full" if visible == total else "partial"
    )
    edit_region = forged_input.get("edit_region_xyxy")
    if (
        not isinstance(edit_region, list)
        or len(edit_region) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in edit_region)
    ):
        raise ValueError("paired forged input has invalid edit_region_xyxy")
    edit_box = _edit_box_visibility(
        edit_region,
        list(geometry["effective_native_crop_xyxy"]),
    )
    return {
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_through_"
            "selected_UFD_resize_padding_and_center_crop"
        ),
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": str(forged_input["sample_id"]),
        "profile_id": str(geometry["profile_id"]),
        "profile_kind": profile_kind,
        "formula": str(geometry["pixel_center_mapping"]),
        "influence_interpretation": influence_claim,
        "geometry": geometry,
        "edit_box": edit_box,
    }


def _compare_nested_subset(
    recorded: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 0.0,
) -> None:
    """Require all independently expected fields without trusting extras."""

    if isinstance(expected, Mapping):
        actual = _require_mapping(recorded, label)
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{label} lacks {key}")
            _compare_nested_subset(
                actual[key],
                value,
                label=f"{label}.{key}",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(expected, list):
        actual = _require_list(recorded, label)
        if len(actual) != len(expected):
            raise ValueError(
                f"{label} length mismatch: {len(actual)} != {len(expected)}"
            )
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_nested_subset(
                left,
                right,
                label=f"{label}[{index}]",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(expected, float):
        actual = _require_finite(recorded, label)
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=float_tolerance,
        ):
            raise ValueError(
                f"{label} mismatch: {actual!r} != {expected!r}"
            )
        return
    _require_equal(recorded, expected, label)


def _reject_t2_localization_or_joint(value: Any, *, label: str) -> None:
    """Recursively reject invented outputs unsupported by UFD."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            if key in _T2_OR_LOCALIZATION_KEYS:
                if not (key == "t2" and nested is False):
                    raise ValueError(
                        f"{label} contains unsupported T2/localization/S_joint "
                        f"field {raw_key!r}"
                    )
                continue
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


def _selection_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


def _select_manifest_inputs(
    input_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(input_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"canonical row {line_number} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"canonical inputs repeat sample_id {sample_id}")
        by_id[sample_id] = row
    selection = _require_mapping(manifest.get("selection"), "manifest selection")
    ordered = _require_list(selection.get("rows"), "manifest ordered inputs")
    if not ordered:
        raise ValueError("manifest ordered inputs are empty")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(ordered):
        item = _require_mapping(raw, f"ordered input {index}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ordered input {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"ordered inputs repeat sample_id {sample_id}")
        if sample_id not in by_id:
            raise ValueError(f"ordered inputs select unknown ID {sample_id}")
        seen.add(sample_id)
        selected.append(by_id[sample_id])
    _require_equal(
        ordered,
        _selection_contract(selected),
        "manifest ordered input contract",
    )
    return selected


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    historical: Counter[str] = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {line_number} has no id")
        histories[row_id].append((line_number, row))
        historical[str(row.get("status"))] += 1
    duplicate_histories: list[dict[str, Any]] = []
    recovered: list[str] = []
    latest_counts: Counter[str] = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicate_histories.append(
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
        "ids_with_multiple_rows": len(duplicate_histories),
        "recovered_error_to_ok": len(recovered),
        "recovered_ids": recovered,
        "historical_status_counts": dict(sorted(historical.items())),
        "latest_status_counts": dict(sorted(latest_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {line_number} has no id")
        latest[row_id] = row
    return latest


def _float32_sigmoid(value: float, torch_module: ModuleType) -> float:
    tensor = torch_module.tensor(
        np.float32(value),
        dtype=torch_module.float32,
    )
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
    """Validate every score/decision alias against an independent replay."""

    recorded_raw = _compare_float(
        row.get("raw_logit"),
        replay_raw_logit,
        label=f"row {row.get('id')} raw_logit",
        tolerance=raw_tolerance,
    )
    probability = _require_probability(
        row.get("probability"),
        f"row {row.get('id')} probability",
    )
    if not math.isclose(
        probability,
        replay_probability,
        rel_tol=0.0,
        abs_tol=probability_tolerance,
    ):
        raise ValueError(
            f"row {row.get('id')} probability replay mismatch: "
            f"{probability} != {replay_probability}"
        )
    for key in ("ai_score", "score"):
        _compare_float(
            row.get(key),
            probability,
            label=f"row {row.get('id')} {key}",
            tolerance=0.0,
        )
    decision = bool(probability > FIXED_THRESHOLD)
    _require_equal(
        decision,
        bool(recorded_raw > 0.0),
        f"row {row.get('id')} probability/logit decision equivalence",
    )
    for key in ("classification_decision",):
        _require_equal(
            row.get(key),
            decision,
            f"row {row.get('id')} {key}",
        )
    _require_equal(
        row.get("classification_threshold"),
        FIXED_THRESHOLD,
        f"row {row.get('id')} classification threshold",
    )
    _require_equal(
        row.get("classification_threshold_operator"),
        THRESHOLD_OPERATOR,
        f"row {row.get('id')} classification operator",
    )
    _require_equal(
        row.get("score_semantics"),
        "official_sigmoid_probability_higher_is_fake",
        f"row {row.get('id')} score semantics",
    )
    classification = _require_mapping(
        row.get("classification"),
        f"row {row.get('id')} classification",
    )
    for key, expected in {
        "raw_logit": recorded_raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": FIXED_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "decision": decision,
        "semantics": "official_sigmoid_probability_higher_is_fake",
    }.items():
        if key not in classification:
            raise ValueError(
                f"row {row.get('id')} classification lacks {key}"
            )
        _compare_nested_subset(
            classification[key],
            expected,
            label=f"row {row.get('id')} classification.{key}",
        )
    t1 = _require_mapping(row.get("t1"), f"row {row.get('id')} t1")
    for key, expected in {
        "raw_logit": recorded_raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": FIXED_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "decision": decision,
        "policy": "official_UFD_CLIP_linear_probe_probability",
    }.items():
        _compare_nested_subset(
            t1.get(key),
            expected,
            label=f"row {row.get('id')} t1.{key}",
        )
    manual = _require_mapping(
        row.get("manual_replay"),
        f"row {row.get('id')} manual replay",
    )
    for key, expected, tolerance in (
        ("raw_logit", recorded_raw, 0.0),
        ("probability", probability, 0.0),
        ("ai_score", probability, 0.0),
        ("classification_decision", decision, 0.0),
        ("model_forward_calls", 1, 0.0),
        ("fc_hook_calls", 1, 0.0),
    ):
        if key not in manual:
            raise ValueError(f"row {row.get('id')} manual replay lacks {key}")
        if isinstance(expected, float):
            _compare_float(
                manual[key],
                expected,
                label=f"row {row.get('id')} manual replay {key}",
                tolerance=tolerance,
            )
        else:
            _require_equal(
                manual[key],
                expected,
                f"row {row.get('id')} manual replay {key}",
            )
    for key in (
        "official_logit_exact_match",
        "official_probability_exact_match",
    ):
        _require_equal(
            manual.get(key),
            True,
            f"row {row.get('id')} manual replay {key}",
        )
    return {
        "raw_logit": replay_raw_logit,
        "probability": replay_probability,
        "decision": decision,
    }


def _asset_pin(value: Any, label: str) -> dict[str, Any]:
    """Normalize a runner asset constant to one exact file record."""

    if not isinstance(value, Mapping):
        raise ValueError(f"runner {label} pin is not a mapping")
    raw = dict(value)
    if "sha256" not in raw:
        candidates = [
            dict(item)
            for item in raw.values()
            if isinstance(item, Mapping) and "sha256" in item
        ]
        if len(candidates) != 1:
            raise ValueError(f"runner {label} pin is ambiguous")
        raw = candidates[0]
    digest = _require_sha256(raw.get("sha256"), f"runner {label} SHA-256")
    size = raw.get("bytes", raw.get("size"))
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"runner {label} byte size is invalid")
    path = raw.get(
        "path",
        raw.get("relative_path", raw.get("filename", raw.get("name"))),
    )
    if not isinstance(path, str) or not path:
        raise ValueError(f"runner {label} pin has no path/name")
    return {
        **raw,
        "path": path,
        "bytes": int(size),
        "sha256": digest,
    }


def _asset_bundle_sha256(
    head_pin: Mapping[str, Any],
    backbone_pin: Mapping[str, Any],
) -> str:
    payload = [
        {
            "role": "linear_head",
            "bytes": int(head_pin["bytes"]),
            "sha256": str(head_pin["sha256"]),
        },
        {
            "role": "clip_backbone",
            "bytes": int(backbone_pin["bytes"]),
            "sha256": str(backbone_pin["sha256"]),
        },
    ]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _recorded_asset(
    value: Any,
    *,
    label: str,
    pin: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    record = _require_mapping(value, f"manifest {label}")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"manifest {label} has no path")
    path = _anchored(Path(path_value), repo_root)
    _require_equal(
        path.name,
        Path(str(pin["path"])).name,
        f"manifest {label} filename",
    )
    _require_equal(
        record.get("bytes", record.get("size")),
        int(pin["bytes"]),
        f"manifest {label} byte size",
    )
    _require_equal(
        record.get("sha256"),
        pin["sha256"],
        f"manifest {label} SHA-256",
    )
    _verify_hash(path, pin["sha256"], label)
    _require_equal(path.stat().st_size, int(pin["bytes"]), f"{label} file size")
    return path, record


def _validate_head_checkpoint(
    *,
    path: Path,
    record: Mapping[str, Any],
    torch_module: ModuleType,
) -> dict[str, Any]:
    unsafe_api = getattr(
        getattr(torch_module, "serialization", None),
        "get_unsafe_globals_in_checkpoint",
        None,
    )
    if unsafe_api is None:
        raise RuntimeError(
            "torch lacks get_unsafe_globals_in_checkpoint for UFD head audit"
        )
    unsafe = list(unsafe_api(path))
    if unsafe:
        raise ValueError(f"UFD head checkpoint has unsafe globals: {unsafe}")
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("UFD head checkpoint is not a state-dict mapping")
    _require_equal(
        list(payload.keys()),
        ["weight", "bias"],
        "UFD head state-dict keys/order",
    )
    expected = {
        "weight": ((1, FEATURE_DIMENSION), torch_module.float32),
        "bias": ((1,), torch_module.float32),
    }
    for key, (shape, dtype) in expected.items():
        tensor = payload[key]
        _require_equal(tuple(tensor.shape), shape, f"UFD head {key} shape")
        _require_equal(tensor.dtype, dtype, f"UFD head {key} dtype")
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise ValueError(f"UFD head {key} is not finite")
    safety = record.get("serialization_safety")
    if safety is not None:
        safety_record = _require_mapping(
            safety,
            "manifest UFD head serialization safety",
        )
        _require_equal(
            safety_record.get("unsafe_globals"),
            [],
            "manifest UFD head unsafe globals",
        )
        _require_equal(
            safety_record.get("weights_only"),
            True,
            "manifest UFD head weights_only",
        )
    return {
        "unsafe_globals": unsafe,
        "weights_only": True,
        "keys": ["weight", "bias"],
        "weight_shape": [1, FEATURE_DIMENSION],
        "bias_shape": [1],
        "dtype": "float32",
    }


def _runtime_package_version(
    runtime: Mapping[str, Any],
    name: str,
) -> str:
    packages = _require_mapping(runtime.get("packages"), "runtime packages")
    raw = packages.get(name)
    if isinstance(raw, Mapping):
        value = raw.get("full_version", raw.get("version"))
    else:
        value = raw
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime package {name} has no version")
    return value


def _distribution_or_module_version(
    distribution: str,
    module: ModuleType,
) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        value = getattr(module, "__version__", None)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"cannot determine {distribution} version")
        return value


def _kernel_replay_runtime(manifest: Mapping[str, Any]) -> ReplayRuntime:
    """Require and reproduce the exact recorded CUDA numerical runtime."""

    runtime = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime contract",
    )
    _require_equal(
        manifest.get("environment"),
        runtime,
        "manifest environment/runtime compatibility copy",
    )
    import torch

    recorded_torch = _runtime_package_version(runtime, "torch")
    _require_equal(torch.__version__, recorded_torch, "torch runtime version")
    _require_equal(
        np.__version__,
        _runtime_package_version(runtime, "numpy"),
        "NumPy runtime version",
    )
    _require_equal(
        _distribution_or_module_version("Pillow", PIL),
        _runtime_package_version(runtime, "Pillow"),
        "Pillow runtime version",
    )
    packages = _require_mapping(runtime.get("packages"), "runtime packages")
    if "torchvision" in packages:
        import torchvision

        _require_equal(
            torchvision.__version__,
            _runtime_package_version(runtime, "torchvision"),
            "torchvision runtime version",
        )
    for package_name in ("ftfy", "regex"):
        _require_equal(
            importlib.metadata.version(package_name),
            _runtime_package_version(runtime, package_name),
            f"{package_name} runtime version",
        )

    python_record = _require_mapping(runtime.get("python"), "runtime Python")
    _require_equal(
        platform.python_implementation(),
        python_record.get("implementation"),
        "Python implementation",
    )
    _require_equal(
        platform.python_version(),
        python_record.get("version"),
        "Python version",
    )
    _require_equal(
        Path(sys.executable).resolve(),
        Path(str(python_record.get("executable"))).resolve(),
        "Python executable",
    )

    accelerator = _require_mapping(
        runtime.get("accelerator"),
        "runtime accelerator",
    )
    _require_equal(
        accelerator.get("device_type"),
        "cuda",
        "UFD replay device type",
    )
    requested = accelerator.get("requested_device")
    if not isinstance(requested, str) or not requested.startswith("cuda:"):
        raise ValueError("runtime requested_device is not an explicit CUDA index")
    index = accelerator.get("device_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("runtime CUDA device index is invalid")
    _require_equal(requested, f"cuda:{index}", "runtime requested CUDA device")
    _require_equal(
        platform.machine(),
        accelerator.get("machine"),
        "runtime machine architecture",
    )
    _require_equal(
        platform.processor(),
        accelerator.get("processor"),
        "runtime processor",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("recorded CUDA runtime is unavailable")
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"recorded CUDA device {index} is unavailable")
    torch.cuda.set_device(index)
    device = torch.device(requested)
    _require_equal(
        torch.cuda.get_device_name(index),
        accelerator.get("gpu_name"),
        "recorded CUDA GPU name",
    )
    if "gpu_capability" in accelerator:
        _require_equal(
            list(torch.cuda.get_device_capability(index)),
            accelerator.get("gpu_capability"),
            "recorded CUDA capability",
        )
    if "torch_cuda" in accelerator:
        _require_equal(
            torch.version.cuda,
            accelerator.get("torch_cuda"),
            "recorded torch CUDA version",
        )
    if "cudnn_version" in accelerator:
        _require_equal(
            torch.backends.cudnn.version(),
            accelerator.get("cudnn_version"),
            "recorded cuDNN version",
        )

    flags = _require_mapping(
        runtime.get("numerical_flags"),
        "runtime numerical flags",
    )
    required_flags = {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    for key, expected in required_flags.items():
        _require_equal(flags.get(key), expected, f"runtime numerical flag {key}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    evidence = {
        "recorded_device": requested,
        "actual_device": str(device),
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_capability": list(torch.cuda.get_device_capability(index)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numerical_flags": required_flags,
        "silent_cpu_fallback_rejected": True,
    }
    return ReplayRuntime(
        torch=torch,
        device=device,
        recorded_device=requested,
        evidence=evidence,
    )


@contextlib.contextmanager
def _pinned_clip_module(source_root: Path):
    """Import the bundled CLIP fork under an isolated synthetic package."""

    package_name = (
        "_claimforge_ufd_audit_"
        + hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()[:12]
    )
    package = ModuleType(package_name)
    package.__path__ = [str(source_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    before = set(sys.modules)
    try:
        clip_api = importlib.import_module(
            f"{package_name}.models.clip.clip"
        )
        yield clip_api
    finally:
        for name in list(sys.modules):
            if name == package_name or (
                name.startswith(f"{package_name}.") and name not in before
            ):
                sys.modules.pop(name, None)


def _load_feature(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str | None = None,
) -> tuple[np.ndarray, Path]:
    path_value = row.get(
        "clip_feature_path",
        _require_mapping(
            row.get("artifact_paths", {}),
            f"row {row.get('id')} artifact paths",
        ).get("clip_feature_npy"),
    )
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"row {row.get('id')} has no CLIP feature path")
    path = _anchored(Path(path_value), repo_root)
    if run_id is not None and run_id not in path.parts:
        raise ValueError(
            f"row {row.get('id')} feature artifact is outside its run directory"
        )
    _verify_hash(
        path,
        row.get("clip_feature_sha256"),
        f"row {row.get('id')} CLIP feature artifact",
    )
    try:
        feature = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"row {row.get('id')} CLIP feature is not a safe NPY"
        ) from exc
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
        row.get("clip_feature_shape"),
        [FEATURE_DIMENSION],
        f"row {row.get('id')} recorded feature shape",
    )
    _require_equal(
        row.get("clip_feature_dtype"),
        "float32",
        f"row {row.get('id')} recorded feature dtype",
    )
    return feature, path


def _validate_row_identity(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
    pins: SimpleNamespace,
    row_label: str,
) -> None:
    expected = {
        "id": canonical["sample_id"],
        "rank": canonical["rank"],
        "pair_rank": canonical["pair_rank"],
        "task_id": canonical["task_id"],
        "domain": canonical["domain"],
        "kind": canonical["kind"],
        "label": canonical["label"],
        "gt_mask_kind": canonical["gt_mask_kind"],
        "gt_mask_sha256": canonical.get("gt_mask_sha256"),
        "edit_region_xyxy": canonical.get("edit_region_xyxy"),
    }
    for key, value in expected.items():
        _require_equal(row.get(key), value, f"{row_label} {key}")
    image_value = row.get("image_path")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{row_label} has no image path")
    image_path = _anchored(Path(image_value), repo_root)
    canonical_path = _anchored(Path(str(canonical["canonical_path"])), repo_root)
    _require_equal(image_path, canonical_path, f"{row_label} image path")
    _verify_hash(
        image_path,
        canonical["canonical_sha256"],
        f"{row_label} canonical image",
    )
    _require_equal(
        row.get("image_sha256"),
        canonical["canonical_sha256"],
        f"{row_label} image SHA-256",
    )
    _require_equal(
        row.get("image_size"),
        [int(canonical["width"]), int(canonical["height"])],
        f"{row_label} image size",
    )
    with Image.open(image_path) as image:
        _require_equal(
            list(image.size),
            [int(canonical["width"]), int(canonical["height"])],
            f"{row_label} decoded dimensions",
        )
    _require_equal(row.get("model"), pins.MODEL_NAME, f"{row_label} model")
    _require_equal(
        row.get("model_slug"),
        pins.MODEL_SLUG,
        f"{row_label} model slug",
    )
    _require_equal(
        row.get("model_arch"),
        pins.MODEL_ARCH,
        f"{row_label} model architecture",
    )


def _audit_preprocess_record(
    row: Mapping[str, Any],
    preprocessed: PreprocessedImage,
    *,
    profile_id: str,
) -> None:
    row_id = row.get("id")
    _require_equal(
        row.get("preprocess_profile"),
        profile_id,
        f"row {row_id} preprocess profile",
    )
    record = _require_mapping(
        row.get("preprocess"),
        f"row {row_id} preprocess",
    )
    geometry = _require_mapping(
        record.get("geometry"),
        f"row {row_id} preprocess geometry",
    )
    _require_equal(
        geometry,
        preprocessed.geometry,
        f"row {row_id} preprocess geometry",
    )
    _require_equal(
        geometry.get("profile_id"),
        profile_id,
        f"row {row_id} preprocess geometry profile",
    )
    _require_equal(
        record.get("decoded_rgb_sha256"),
        preprocessed.decoded_rgb_sha256,
        f"row {row_id} decoded RGB SHA-256",
    )
    _require_equal(
        record.get("crop_rgb_sha256"),
        preprocessed.crop_rgb_sha256,
        f"row {row_id} crop RGB SHA-256",
    )
    _require_equal(
        record.get("crop_rgb_shape"),
        [CROP_SIZE, CROP_SIZE, 3],
        f"row {row_id} crop RGB shape",
    )
    _require_equal(
        record.get("crop_rgb_dtype"),
        "uint8",
        f"row {row_id} crop RGB dtype",
    )
    _require_equal(
        record.get("tensor_shape"),
        [3, CROP_SIZE, CROP_SIZE],
        f"row {row_id} tensor shape",
    )
    _require_equal(
        record.get("tensor_dtype"),
        "float32",
        f"row {row_id} tensor dtype",
    )
    _require_equal(
        record.get("tensor_sha256"),
        preprocessed.tensor_sha256,
        f"row {row_id} tensor SHA-256",
    )


def _audit_visibility_record(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    row_id = row.get("id")
    _require_equal(
        row.get("edit_visibility"),
        expected["category"],
        f"row {row_id} edit visibility",
    )
    _compare_float(
        row.get("edit_visible_gt_fraction"),
        float(expected["visible_fraction"]),
        label=f"row {row_id} visible GT fraction",
        tolerance=0.0,
    )
    evidence = _require_mapping(
        row.get("edit_visibility_evidence"),
        f"row {row_id} edit visibility evidence",
    )
    recorded_gt = _require_mapping(
        evidence.get("gt"),
        f"row {row_id} exact-GT visibility evidence",
    )
    expected_gt = {
        key: expected[key]
        for key in (
            "basis",
            "category",
            "visible_fraction",
            "positive_pixels",
            "visible_positive_pixel_centers",
            "forged_sample_id",
            "profile_id",
            "formula",
        )
    }
    _require_equal(
        recorded_gt,
        expected_gt,
        f"row {row_id} exact-GT visibility evidence",
    )
    _require_equal(
        evidence.get("edit_box"),
        expected["edit_box"],
        f"row {row_id} edit-box visibility evidence",
    )


def _validate_physical_result_payload(
    row: Mapping[str, Any],
    *,
    row_label: str,
    repo_root: Path,
    run_id: str,
    profile_id: str,
    torch_module: ModuleType,
) -> None:
    status = row.get("status")
    if status not in {"ok", "error"}:
        raise ValueError(f"{row_label} has invalid status {status!r}")
    visibility = row.get("edit_visibility")
    if visibility not in {"none", "partial", "full"}:
        raise ValueError(f"{row_label} has invalid edit_visibility")
    fraction = _require_probability(
        row.get("edit_visible_gt_fraction"),
        f"{row_label} edit_visible_gt_fraction",
    )
    expected_category = (
        "none" if fraction == 0.0 else "full" if fraction == 1.0 else "partial"
    )
    _require_equal(visibility, expected_category, f"{row_label} visibility")
    _require_mapping(
        row.get("edit_visibility_evidence"),
        f"{row_label} edit visibility evidence",
    )
    _require_equal(row.get("valid_for_t1"), True, f"{row_label} valid_for_t1")
    _require_equal(row.get("valid_for_t2"), False, f"{row_label} valid_for_t2")
    _require_equal(
        row.get("t1_policy"),
        "official_UFD_CLIP_linear_probe_probability",
        f"{row_label} T1 policy",
    )
    _require_equal(
        row.get("t2_policy"),
        "unsupported_whole_image_detector",
        f"{row_label} T2 policy",
    )
    _require_equal(
        row.get("preprocess_profile"),
        profile_id,
        f"{row_label} preprocess profile",
    )
    _reject_t2_localization_or_joint(row, label=row_label)

    if status == "error":
        _require_equal(
            row.get("valid_for_metrics"),
            False,
            f"{row_label} valid_for_metrics",
        )
        for key in ("error_type", "error_message", "traceback", "completed_at"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{row_label} has invalid {key}")
        forbidden = {
            "raw_logit",
            "probability",
            "ai_score",
            "score",
            "classification",
            "t1",
            "manual_replay",
            "clip_feature_path",
            "preprocess",
            "latency_ms",
            "peak_cuda_memory_bytes",
        }.intersection(row)
        if forbidden:
            raise ValueError(
                f"{row_label} error payload claims success fields "
                f"{sorted(forbidden)}"
            )
        return

    _require_equal(
        row.get("valid_for_metrics"),
        True,
        f"{row_label} valid_for_metrics",
    )
    raw = _require_finite(row.get("raw_logit"), f"{row_label} raw logit")
    probability = _float32_sigmoid(raw, torch_module)
    _audit_score_fields(
        row,
        replay_raw_logit=raw,
        replay_probability=probability,
        raw_tolerance=0.0,
        probability_tolerance=SIGMOID_ABSOLUTE_TOLERANCE,
    )
    _load_feature(row, repo_root=repo_root, run_id=run_id)
    _require_equal(
        row.get("clip_feature_semantics"),
        "official_CLIP_encode_image_output_before_linear_head",
        f"{row_label} feature semantics",
    )
    feature_path = row.get("clip_feature_path")
    _require_equal(
        row.get("artifact_paths"),
        {"clip_feature_npy": feature_path},
        f"{row_label} feature artifact alias",
    )
    preprocess = _require_mapping(row.get("preprocess"), f"{row_label} preprocess")
    for key in (
        "geometry",
        "decoded_rgb_sha256",
        "crop_rgb_sha256",
        "crop_rgb_shape",
        "crop_rgb_dtype",
        "tensor_shape",
        "tensor_dtype",
        "tensor_sha256",
    ):
        if key not in preprocess:
            raise ValueError(f"{row_label} preprocess lacks {key}")
    latency = _require_finite(row.get("latency_ms"), f"{row_label} latency")
    if latency < 0.0:
        raise ValueError(f"{row_label} latency is negative")
    peak = row.get("peak_cuda_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"{row_label} peak CUDA memory is invalid")
    if not isinstance(row.get("completed_at"), str) or not row.get(
        "completed_at"
    ):
        raise ValueError(f"{row_label} completed_at is invalid")


def _row_provenance_identity(
    row: Mapping[str, Any],
    *,
    row_label: str,
    run_id: str,
    fingerprint: str,
    source_commit: str,
    asset_bundle_sha256: str,
    inputs_sha256: str,
) -> None:
    _require_equal(row.get("run_id"), run_id, f"{row_label} run ID")
    _require_equal(
        row.get("run_manifest_fingerprint"),
        fingerprint,
        f"{row_label} run fingerprint",
    )
    _require_equal(
        row.get("input_manifest_sha256"),
        inputs_sha256,
        f"{row_label} input manifest SHA-256",
    )
    _require_equal(
        row.get("model_source_commit", row.get("source_commit")),
        source_commit,
        f"{row_label} source commit",
    )
    _require_equal(
        row.get("asset_bundle_sha256", row.get("weights_bundle_sha256")),
        asset_bundle_sha256,
        f"{row_label} asset bundle SHA-256",
    )


def _validate_adapter_contract(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    records = _require_list(
        manifest.get("adapter_contract"),
        "manifest adapter contract",
    )
    paths: set[Path] = set()
    for index, raw in enumerate(records):
        item = _require_mapping(raw, f"adapter contract entry {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"adapter contract entry {index} has no path")
        path = _anchored(Path(path_value), repo_root)
        if path in paths:
            raise ValueError("adapter contract repeats a file")
        paths.add(path)
        _verify_hash(
            path,
            item.get("sha256"),
            f"adapter contract entry {index}",
        )
    names = {path.name for path in paths}
    required = {"run_universalfakedetect.py", "common.py", "ufd_metrics.py"}
    _require_equal(names, required, "adapter contract filenames")
    return {
        "files_validated": len(paths),
        "filenames": sorted(names),
    }


def validate_provenance(
    *,
    repo_root: Path,
    source_root: Path,
    run_id: str,
    inputs_path: Path,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate physical history, inputs, source, assets, and contracts."""

    pins = _load_runner_pins()
    _require_equal(
        manifest.get("schema_version"),
        "opensource_run_manifest_v1",
        "run manifest schema",
    )
    _require_equal(manifest.get("run_id"), run_id, "run manifest ID")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "run manifest fingerprint",
    )
    _require_equal(
        fingerprint,
        _manifest_fingerprint(manifest),
        "run manifest fingerprint",
    )

    profile_id = manifest.get("preprocess_profile")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("manifest has no preprocess_profile")
    profile_kind = _profile_kind(profile_id, pins)
    profiles = _profile_mapping(pins.PREPROCESS_PROFILES)
    _require_equal(
        manifest.get("preprocess_profile_contract"),
        profiles[profile_id],
        "manifest preprocess profile contract",
    )
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    protocol_profile = protocol.get(
        "preprocess_profile",
        protocol.get("preprocess"),
    )
    if isinstance(protocol_profile, str):
        _require_equal(
            protocol_profile,
            profile_id,
            "manifest protocol preprocess profile",
        )
    elif isinstance(protocol_profile, Mapping):
        recorded_profile_id = protocol_profile.get(
            "id",
            protocol_profile.get("profile_id"),
        )
        _require_equal(
            recorded_profile_id,
            profile_id,
            "manifest protocol preprocess profile ID",
        )
    else:
        raise ValueError("manifest protocol has no preprocess profile")
    _require_equal(
        protocol.get("preprocess_profile_contract"),
        profiles[profile_id],
        "manifest protocol preprocess profile contract",
    )
    recorded_drift = manifest.get("source_checkpoint_drift")
    _require_equal(
        recorded_drift,
        pins.SOURCE_CHECKPOINT_DRIFT,
        "manifest source/checkpoint preprocessing drift",
    )

    model = _require_mapping(manifest.get("model"), "manifest model")
    for actual, expected, label in (
        (model.get("name"), pins.MODEL_NAME, "manifest model name"),
        (model.get("slug"), pins.MODEL_SLUG, "manifest model slug"),
        (model.get("repo_url"), pins.MODEL_REPO_URL, "manifest repository URL"),
        (
            model.get("source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "manifest source commit",
        ),
        (
            model.get("architecture"),
            pins.MODEL_ARCH,
            "manifest model architecture",
        ),
        (model.get("paper_url"), pins.PAPER_URL, "manifest paper URL"),
        (model.get("license"), pins.LICENSE_RECORD, "manifest license record"),
    ):
        _require_equal(actual, expected, label)
    _require_equal(
        model.get("task_support"),
        {"t1": True, "t2": False},
        "manifest task support",
    )
    source_root_value = model.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ValueError("manifest model has no source_root")
    _require_equal(
        Path(source_root_value).resolve(),
        source_root.resolve(),
        "manifest UFD source root",
    )
    expected_source = _normalise_source_files(
        pins.SOURCE_FILES,
        "runner source files",
    )
    _verify_source_tree(
        source_root,
        expected_commit=pins.MODEL_SOURCE_COMMIT,
        expected_files=expected_source,
    )
    source_audit = _require_mapping(
        model.get("source_audit"),
        "manifest source audit",
    )
    for actual, expected, label in (
        (source_audit.get("repo_url"), pins.MODEL_REPO_URL, "source audit URL"),
        (
            Path(str(source_audit.get("root"))).resolve(),
            source_root.resolve(),
            "source audit root",
        ),
        (
            source_audit.get("commit"),
            pins.MODEL_SOURCE_COMMIT,
            "source audit commit",
        ),
        (
            source_audit.get("tracked_dirty"),
            False,
            "source audit tracked-dirty flag",
        ),
    ):
        _require_equal(actual, expected, label)
    recorded_source_raw = _require_mapping(
        source_audit.get("source_files"),
        "manifest source files",
    )
    recorded_source = _normalise_source_files(
        {
            path: (
                raw.get("sha256")
                if isinstance(raw, Mapping)
                else raw
            )
            for path, raw in recorded_source_raw.items()
        },
        "manifest source files",
    )
    _require_equal(
        recorded_source,
        expected_source,
        "manifest frozen source hashes",
    )
    bundled_head_pin = _asset_pin(
        pins.HEAD_CHECKPOINT,
        "repository-bundled head checkpoint",
    )
    bundled_head_path = source_root / str(
        pins.HEAD_CHECKPOINT["repo_relative_path"]
    )
    _verify_hash(
        bundled_head_path,
        bundled_head_pin["sha256"],
        "repository-bundled UFD head",
    )
    _require_equal(
        bundled_head_path.stat().st_size,
        int(bundled_head_pin["bytes"]),
        "repository-bundled UFD head size",
    )
    bundled_head_audit = _require_mapping(
        source_audit.get("bundled_head"),
        "manifest repository-bundled head",
    )
    _require_equal(
        Path(str(bundled_head_audit.get("path"))).resolve(),
        bundled_head_path.resolve(),
        "manifest repository-bundled head path",
    )
    _require_equal(
        bundled_head_audit.get("bytes"),
        int(bundled_head_pin["bytes"]),
        "manifest repository-bundled head size",
    )
    _require_equal(
        bundled_head_audit.get("sha256"),
        bundled_head_pin["sha256"],
        "manifest repository-bundled head SHA-256",
    )
    historical_validate_sha256 = hashlib.sha256(
        _git_bytes(
            source_root,
            f"{pins.HEAD_INTRO_COMMIT}:validate.py",
        )
    ).hexdigest()
    _require_equal(
        historical_validate_sha256,
        pins.HEAD_ERA_VALIDATE_SHA256,
        "checkpoint-era validate.py SHA-256",
    )
    _require_equal(
        _git_value(
            source_root,
            "log",
            "--format=%H",
            "--",
            str(pins.HEAD_CHECKPOINT["repo_relative_path"]),
        ),
        pins.HEAD_INTRO_COMMIT,
        "released-head introduction history",
    )
    _require_equal(
        source_audit.get("preprocess_drift"),
        {
            "head_introduction_commit": pins.HEAD_INTRO_COMMIT,
            "head_era_validate_sha256": pins.HEAD_ERA_VALIDATE_SHA256,
            "resize_removal_commit": pins.RESIZE_REMOVAL_COMMIT,
            "current_source_commit": pins.MODEL_SOURCE_COMMIT,
            "interpretation": (
                "the repository does not identify whether the released head "
                "should use its checkpoint-era or current validation transform"
            ),
        },
        "manifest source-audit preprocess drift",
    )

    import torch

    head_pin = _asset_pin(pins.HEAD_CHECKPOINT, "head checkpoint")
    backbone_pin = _asset_pin(
        pins.BACKBONE_CHECKPOINT,
        "backbone checkpoint",
    )
    head_path, head_record = _recorded_asset(
        model.get("head_checkpoint"),
        label="UFD linear head",
        pin=head_pin,
        repo_root=repo_root,
    )
    backbone_path, backbone_record = _recorded_asset(
        model.get("backbone_checkpoint"),
        label="OpenAI CLIP ViT-L/14 backbone",
        pin=backbone_pin,
        repo_root=repo_root,
    )
    head_audit = _validate_head_checkpoint(
        path=head_path,
        record=head_record,
        torch_module=torch,
    )
    asset_bundle_sha256 = _require_sha256(
        model.get("asset_bundle_sha256"),
        "manifest asset bundle SHA-256",
    )
    _require_equal(
        asset_bundle_sha256,
        _asset_bundle_sha256(head_pin, backbone_pin),
        "manifest frozen asset bundle SHA-256",
    )
    if not zipfile.is_zipfile(backbone_path):
        raise ValueError("OpenAI CLIP backbone is not a TorchScript ZIP archive")
    explicit_paths = _require_mapping(
        model.get("explicit_paths"),
        "manifest explicit model paths",
    )
    _require_equal(
        Path(str(explicit_paths.get("source_root"))).resolve(),
        source_root.resolve(),
        "manifest explicit source root",
    )
    _require_equal(
        Path(str(explicit_paths.get("head_checkpoint"))).resolve(),
        head_path.resolve(),
        "manifest explicit head checkpoint",
    )
    _require_equal(
        Path(str(explicit_paths.get("backbone_checkpoint"))).resolve(),
        backbone_path.resolve(),
        "manifest explicit backbone checkpoint",
    )

    classification = _require_mapping(
        protocol.get("classification"),
        "manifest classification protocol",
    )
    threshold = classification.get(
        "threshold",
        classification.get(
            "ai_score_threshold",
            classification.get("released_threshold"),
        ),
    )
    _require_equal(
        threshold,
        FIXED_THRESHOLD,
        "manifest fixed UFD threshold",
    )
    operator = classification.get(
        "threshold_operator",
        classification.get(
            "operator",
            classification.get(
                "ai_score_operator",
                classification.get("released_operator"),
            ),
        ),
    )
    _require_equal(operator, THRESHOLD_OPERATOR, "manifest threshold operator")
    if classification.get("strict") is not None:
        _require_equal(
            classification.get("strict"),
            True,
            "manifest strict threshold flag",
        )
    feature_contract = protocol.get("clip_feature", protocol.get("feature"))
    feature_contract = _require_mapping(
        feature_contract,
        "manifest CLIP feature contract",
    )
    _require_equal(
        feature_contract.get("shape"),
        [FEATURE_DIMENSION],
        "manifest CLIP feature shape",
    )
    _require_equal(
        feature_contract.get("dtype"),
        "float32",
        "manifest CLIP feature dtype",
    )
    normalized = feature_contract.get(
        "l2_normalized",
        feature_contract.get("normalization"),
    )
    if normalized not in (None, False, "none", "not_l2_normalized"):
        raise ValueError("manifest does not explicitly forbid feature L2 norm")
    _require_equal(
        feature_contract.get("persisted_for_every_ok_image"),
        True,
        "manifest persisted-feature requirement",
    )
    bootstrap_samples = protocol.get(
        "bootstrap_samples",
        DEFAULT_BOOTSTRAP_SAMPLES,
    )
    bootstrap_seed = protocol.get("seed", DEFAULT_BOOTSTRAP_SEED)
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("manifest bootstrap sample count is invalid")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("manifest bootstrap seed is invalid")

    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    _require_equal(
        _anchored(Path(str(dataset.get("inputs_path"))), repo_root),
        inputs_path.resolve(),
        "manifest canonical inputs path",
    )
    inputs_sha256 = _require_sha256(
        dataset.get("inputs_sha256"),
        "manifest canonical inputs SHA-256",
    )
    _verify_hash(inputs_path, inputs_sha256, "canonical input JSONL")
    dataset_manifest_value = dataset.get("manifest_path")
    if not isinstance(dataset_manifest_value, str) or not dataset_manifest_value:
        raise ValueError("manifest dataset has no canonical manifest path")
    dataset_manifest_path = _anchored(
        Path(dataset_manifest_value),
        repo_root,
    )
    _verify_hash(
        dataset_manifest_path,
        dataset.get("manifest_sha256"),
        "canonical dataset manifest",
    )

    selected = _select_manifest_inputs(input_rows, manifest)
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    latest = _latest_by_id(result_rows)
    if set(latest) != set(selected_by_id):
        missing = sorted(set(selected_by_id) - set(latest))
        extra = sorted(set(latest) - set(selected_by_id))
        raise ValueError(
            f"latest result IDs do not match selection; missing={missing}, "
            f"extra={extra}"
        )
    for line_number, row in enumerate(result_rows, start=1):
        row_label = f"physical result row {line_number}"
        row_id = row.get("id")
        if row_id not in selected_by_id:
            raise ValueError(f"{row_label} has unexpected ID {row_id!r}")
        _require_equal(
            row.get("schema_version"),
            "opensource_result_v1",
            f"{row_label} schema",
        )
        _validate_row_identity(
            row,
            selected_by_id[str(row_id)],
            repo_root=repo_root,
            pins=pins,
            row_label=row_label,
        )
        _row_provenance_identity(
            row,
            row_label=row_label,
            run_id=run_id,
            fingerprint=fingerprint,
            source_commit=pins.MODEL_SOURCE_COMMIT,
            asset_bundle_sha256=asset_bundle_sha256,
            inputs_sha256=inputs_sha256,
        )
        _validate_physical_result_payload(
            row,
            row_label=row_label,
            repo_root=repo_root,
            run_id=run_id,
            profile_id=profile_id,
            torch_module=torch,
        )

    _require_equal(summary.get("run_id"), run_id, "summary run ID")
    _require_equal(
        summary.get("run_manifest_fingerprint"),
        fingerprint,
        "summary run fingerprint",
    )
    if "preprocess_profile" in summary:
        _require_equal(
            summary.get("preprocess_profile"),
            profile_id,
            "summary preprocess profile",
        )
    _reject_t2_localization_or_joint(summary, label="summary")
    _reject_t2_localization_or_joint(manifest, label="manifest")
    adapter = _validate_adapter_contract(manifest, repo_root=repo_root)
    runtime = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime contract",
    )
    _require_equal(
        manifest.get("environment"),
        runtime,
        "manifest environment/runtime copy",
    )
    return {
        "run_manifest_fingerprint": fingerprint,
        "preprocess_profile": profile_id,
        "profile_kind": profile_kind,
        "preprocess_profile_contract": profiles[profile_id],
        "source_commit": pins.MODEL_SOURCE_COMMIT,
        "source_files_validated": len(expected_source),
        "head_checkpoint": _relative_or_absolute(head_path, repo_root),
        "head_checkpoint_audit": head_audit,
        "backbone_checkpoint": _relative_or_absolute(
            backbone_path,
            repo_root,
        ),
        "asset_bundle_sha256": asset_bundle_sha256,
        "adapter_contract": adapter,
        "canonical_inputs_sha256": inputs_sha256,
        "canonical_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "physical_result_rows_validated": len(result_rows),
        "latest_rows_validated": len(latest),
        "selected_inputs_validated": len(selected),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _model_assets(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    pins: SimpleNamespace,
) -> tuple[Path, Path]:
    model = _require_mapping(manifest.get("model"), "manifest model")
    head_path, _ = _recorded_asset(
        model.get("head_checkpoint"),
        label="UFD linear head",
        pin=_asset_pin(pins.HEAD_CHECKPOINT, "head checkpoint"),
        repo_root=repo_root,
    )
    backbone_path, _ = _recorded_asset(
        model.get("backbone_checkpoint"),
        label="OpenAI CLIP ViT-L/14 backbone",
        pin=_asset_pin(pins.BACKBONE_CHECKPOINT, "backbone checkpoint"),
        repo_root=repo_root,
    )
    return head_path, backbone_path


def _load_replay_model(
    *,
    source_root: Path,
    head_path: Path,
    backbone_path: Path,
    runtime: ReplayRuntime,
) -> contextlib.AbstractContextManager[tuple[Any, Any]]:
    """Load bundled visual encoder plus the exact released linear head."""

    @contextlib.contextmanager
    def manager():
        with _pinned_clip_module(source_root) as clip_api:
            backbone, _unused_preprocess = clip_api.load(
                str(backbone_path),
                device="cpu",
            )
            head_state = runtime.torch.load(
                head_path,
                map_location="cpu",
                weights_only=True,
            )
            head = runtime.torch.nn.Linear(FEATURE_DIMENSION, 1)
            head.load_state_dict(head_state, strict=True)
            backbone.eval().to(runtime.device)
            head.eval().to(runtime.device)
            floating_dtypes = {
                parameter.dtype
                for parameter in backbone.parameters()
                if parameter.is_floating_point()
            }
            _require_equal(
                floating_dtypes,
                {runtime.torch.float32},
                "UFD backbone floating dtypes",
            )
            _require_equal(
                {parameter.dtype for parameter in head.parameters()},
                {runtime.torch.float32},
                "UFD head floating dtypes",
            )
            try:
                yield backbone, head
            finally:
                del backbone
                del head
                runtime.torch.cuda.empty_cache()

    return manager()


def _complete_result_pairs(
    selected: list[dict[str, Any]],
    latest: Mapping[str, dict[str, Any]],
) -> list[DetectionPair]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_ranks: dict[str, int] = {}
    for canonical in selected:
        task_id = str(canonical["task_id"])
        by_task[task_id][str(canonical["kind"])] = latest[
            str(canonical["sample_id"])
        ]
        pair_ranks.setdefault(task_id, int(canonical["pair_rank"]))
    pairs: list[DetectionPair] = []
    for task_id, kinds in by_task.items():
        if set(kinds) == {"real", "forged"}:
            pairs.append(
                DetectionPair(
                    task_id=task_id,
                    real=kinds["real"],
                    forged=kinds["forged"],
                )
            )
    pairs.sort(key=lambda pair: pair_ranks[pair.task_id])
    return pairs


def _audit_canonical_crop_pair_equivalence(
    *,
    pairs: list[DetectionPair],
    crop_rgb_arrays: Mapping[str, np.ndarray],
    crop_hashes: Mapping[str, str],
    tensor_hashes: Mapping[str, str],
    artifact_features: Mapping[str, np.ndarray],
    reencoded_features: Mapping[str, np.ndarray],
    replay_scores: Mapping[str, Mapping[str, Any]],
    visibility_by_task: Mapping[str, Mapping[str, Any]],
    profile_kind: str,
    full_selection: bool,
) -> dict[str, Any]:
    """Bind downstream pair equality to canonical decoded crop equality."""

    counts: dict[str, dict[str, int]] = {}
    crop_different_details: list[dict[str, Any]] = []
    crop_equal_pairs = 0
    for pair in pairs:
        real_id = str(pair.real["id"])
        forged_id = str(pair.forged["id"])
        visibility = str(visibility_by_task[pair.task_id]["category"])
        bucket = counts.setdefault(
            visibility,
            {
                "pairs": 0,
                "crop_equal_pairs": 0,
                "crop_different_pairs": 0,
            },
        )
        bucket["pairs"] += 1
        real_crop = crop_rgb_arrays[real_id]
        forged_crop = crop_rgb_arrays[forged_id]
        crop_equal = bool(np.array_equal(real_crop, forged_crop))
        _require_equal(
            crop_hashes[real_id] == crop_hashes[forged_id],
            crop_equal,
            f"pair {pair.task_id} crop array/hash equality",
        )
        if crop_equal:
            crop_equal_pairs += 1
            bucket["crop_equal_pairs"] += 1
            _require_equal(
                tensor_hashes[real_id],
                tensor_hashes[forged_id],
                f"canonical-crop-equal pair {pair.task_id} tensor hash",
            )
            if not np.array_equal(
                artifact_features[real_id],
                artifact_features[forged_id],
            ):
                raise ValueError(
                    f"canonical-crop-equal pair {pair.task_id} persisted "
                    "features differ"
                )
            if not np.array_equal(
                reencoded_features[real_id],
                reencoded_features[forged_id],
            ):
                raise ValueError(
                    f"canonical-crop-equal pair {pair.task_id} re-encoded "
                    "features differ"
                )
            for key in ("raw_logit", "probability", "decision"):
                _require_equal(
                    replay_scores[real_id][key],
                    replay_scores[forged_id][key],
                    f"canonical-crop-equal pair {pair.task_id} {key}",
                )
            continue

        bucket["crop_different_pairs"] += 1
        pixel_difference = np.abs(
            real_crop.astype(np.int16) - forged_crop.astype(np.int16)
        )
        crop_different_details.append(
            {
                "task_id": pair.task_id,
                "edit_visibility": visibility,
                "differing_channel_values": int(
                    np.count_nonzero(pixel_difference)
                ),
                "differing_pixels": int(
                    np.count_nonzero(np.any(pixel_difference != 0, axis=2))
                ),
                "maximum_rgb_absolute_difference": int(
                    pixel_difference.max(initial=0)
                ),
                "tensor_exact": (
                    tensor_hashes[real_id] == tensor_hashes[forged_id]
                ),
                "persisted_feature_exact": bool(
                    np.array_equal(
                        artifact_features[real_id],
                        artifact_features[forged_id],
                    )
                ),
                "reencoded_feature_exact": bool(
                    np.array_equal(
                        reencoded_features[real_id],
                        reencoded_features[forged_id],
                    )
                ),
                "raw_logit_exact": (
                    replay_scores[real_id]["raw_logit"]
                    == replay_scores[forged_id]["raw_logit"]
                ),
                "raw_logit_delta_forged_minus_real": (
                    float(replay_scores[forged_id]["raw_logit"])
                    - float(replay_scores[real_id]["raw_logit"])
                ),
                "probability_exact": (
                    replay_scores[real_id]["probability"]
                    == replay_scores[forged_id]["probability"]
                ),
                "probability_delta_forged_minus_real": (
                    float(replay_scores[forged_id]["probability"])
                    - float(replay_scores[real_id]["probability"])
                ),
                "decision_exact": (
                    replay_scores[real_id]["decision"]
                    == replay_scores[forged_id]["decision"]
                ),
            }
        )

    core_by_visibility = {
        visibility: dict(bucket)
        for visibility, bucket in sorted(counts.items())
    }
    core_census = {
        "pairs": len(pairs),
        "crop_equal_pairs": crop_equal_pairs,
        "crop_different_pairs": len(pairs) - crop_equal_pairs,
        "by_edit_visibility": core_by_visibility,
    }
    expected_census = (
        CURRENT_HEAD_CANONICAL_CROP_EQUALITY_CENSUS
        if profile_kind == "current_head_native"
        else None
    )
    if full_selection and expected_census is not None:
        _require_equal(
            core_census,
            expected_census,
            "full current-head canonical crop equality census",
        )
    by_visibility = {
        visibility: {
            **bucket,
            "crop_equal_fraction": (
                bucket["crop_equal_pairs"] / bucket["pairs"]
                if bucket["pairs"]
                else None
            ),
        }
        for visibility, bucket in core_by_visibility.items()
    }
    different_by_visibility: dict[str, list[str]] = defaultdict(list)
    for detail in crop_different_details:
        different_by_visibility[str(detail["edit_visibility"])].append(
            str(detail["task_id"])
        )
    return {
        "basis": (
            "np.array_equal_on_independently_decoded_and_preprocessed_"
            "canonical_RGB_crop_uint8_[224,224,3]"
        ),
        "gt_visibility_semantics": (
            "pre_canonicalization_input_location_condition_only; exact-diff "
            "GT precedes independent real/forged JPEG_q95 canonicalization"
        ),
        "pairs": len(pairs),
        "crop_equal_pairs": crop_equal_pairs,
        "crop_different_pairs": len(pairs) - crop_equal_pairs,
        "crop_equal_fraction": (
            crop_equal_pairs / len(pairs) if pairs else None
        ),
        "by_edit_visibility": by_visibility,
        "crop_different_task_ids_by_edit_visibility": {
            visibility: task_ids
            for visibility, task_ids in sorted(
                different_by_visibility.items()
            )
        },
        "crop_different_pair_details": crop_different_details,
        "crop_equal_downstream_exact_pairs_validated": crop_equal_pairs,
        "crop_equal_downstream_exact_contract": [
            "normalized_tensor_bytes",
            "persisted_CLIP_feature_float32_array",
            "independently_reencoded_CLIP_feature_float32_array",
            "raw_logit",
            "sigmoid_probability_and_all_score_aliases",
            "strict_probability_greater_than_0.5_decision",
        ],
        "crop_different_pair_downstream_equality_not_assumed": True,
        "full_current_expected_census": expected_census,
        "full_current_expected_census_enforced": bool(
            full_selection and expected_census is not None
        ),
    }


def audit_artifacts(
    *,
    repo_root: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
    all_input_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    latest: Mapping[str, dict[str, Any]],
    runtime: ReplayRuntime | None = None,
    encoder: Any | None = None,
    head: Any | None = None,
) -> dict[str, Any]:
    """Re-decode, re-encode, and independently score every selected image."""

    pins = _load_runner_pins()
    profile_id = str(manifest.get("preprocess_profile"))
    profile_kind = _profile_kind(profile_id, pins)
    replay_runtime = runtime or _kernel_replay_runtime(manifest)
    head_path, backbone_path = _model_assets(
        manifest,
        repo_root=repo_root,
        pins=pins,
    )
    if (encoder is None) != (head is None):
        raise ValueError("test injection must provide both encoder and head")
    model_context: contextlib.AbstractContextManager[tuple[Any, Any]]
    if encoder is None:
        model_context = _load_replay_model(
            source_root=source_root,
            head_path=head_path,
            backbone_path=backbone_path,
            runtime=replay_runtime,
        )
    else:
        model_context = contextlib.nullcontext((encoder, head))

    canonical_by_id = {
        str(row["sample_id"]): row for row in all_input_rows
    }
    forged_by_task = {
        str(row["task_id"]): row
        for row in all_input_rows
        if row.get("kind") == "forged"
    }
    if len(canonical_by_id) != len(all_input_rows):
        raise ValueError("canonical inputs repeat sample IDs")

    crop_rgb_arrays: dict[str, np.ndarray] = {}
    crop_hashes: dict[str, str] = {}
    tensor_hashes: dict[str, str] = {}
    artifact_features: dict[str, np.ndarray] = {}
    reencoded_features: dict[str, np.ndarray] = {}
    replay_scores: dict[str, dict[str, Any]] = {}
    visibility_by_task: dict[str, dict[str, Any]] = {}
    selected_kinds: dict[str, set[str]] = defaultdict(set)
    raw_differences: list[float] = []
    probability_differences: list[float] = []
    feature_max_differences: list[float] = []

    with model_context as (active_encoder, active_head):
        for canonical in selected:
            sample_id = str(canonical["sample_id"])
            if sample_id not in latest:
                raise ValueError(f"missing latest row {sample_id}")
            row = latest[sample_id]
            _require_equal(row.get("status"), "ok", f"row {sample_id} status")
            image_path = _anchored(
                Path(str(canonical["canonical_path"])),
                repo_root,
            )
            preprocessed = preprocess_image(
                image_path,
                profile_kind=profile_kind,
                torch_module=replay_runtime.torch,
                profile_id=profile_id,
            )
            _audit_preprocess_record(
                row,
                preprocessed,
                profile_id=profile_id,
            )
            crop_rgb_arrays[sample_id] = preprocessed.crop_rgb
            crop_hashes[sample_id] = preprocessed.crop_rgb_sha256
            tensor_hashes[sample_id] = preprocessed.tensor_sha256

            artifact_feature, _ = _load_feature(
                row,
                repo_root=repo_root,
                run_id=str(manifest["run_id"]),
            )
            artifact_features[sample_id] = artifact_feature

            input_tensor = preprocessed.tensor.unsqueeze(0).to(
                replay_runtime.device,
                dtype=replay_runtime.torch.float32,
            )
            with replay_runtime.torch.inference_mode():
                feature_tensor = active_encoder.encode_image(input_tensor)
                raw_tensor = active_head(feature_tensor).reshape(())
                probability_tensor = replay_runtime.torch.sigmoid(raw_tensor)
            _require_equal(
                tuple(feature_tensor.shape),
                (1, FEATURE_DIMENSION),
                f"row {sample_id} re-encoded feature shape",
            )
            _require_equal(
                feature_tensor.dtype,
                replay_runtime.torch.float32,
                f"row {sample_id} re-encoded feature dtype",
            )
            feature = (
                feature_tensor.detach()
                .to("cpu", dtype=replay_runtime.torch.float32)
                .numpy()
                .reshape(FEATURE_DIMENSION)
            )
            if not np.isfinite(feature).all():
                raise ValueError(f"row {sample_id} re-encoded feature is non-finite")
            reencoded_features[sample_id] = feature
            feature_difference = float(
                np.max(np.abs(feature - artifact_feature))
            )
            feature_max_differences.append(feature_difference)
            if not np.array_equal(feature, artifact_feature):
                raise ValueError(
                    f"row {sample_id} persisted feature differs from full "
                    f"re-encoding (max abs {feature_difference})"
                )

            # Replay the head from the persisted artifact independently of the
            # just-computed feature.  It remains on the recorded CUDA device so
            # the float32 reduction contract is identical to the runner.
            artifact_tensor = replay_runtime.torch.from_numpy(
                artifact_feature.copy()
            ).to(
                replay_runtime.device,
                dtype=replay_runtime.torch.float32,
            )[None, :]
            with replay_runtime.torch.inference_mode():
                artifact_raw_tensor = active_head(artifact_tensor).reshape(())
                artifact_probability_tensor = replay_runtime.torch.sigmoid(
                    artifact_raw_tensor
                )
            raw = float(raw_tensor.item())
            probability = float(probability_tensor.item())
            artifact_raw = float(artifact_raw_tensor.item())
            artifact_probability = float(artifact_probability_tensor.item())
            _compare_float(
                artifact_raw,
                raw,
                label=f"row {sample_id} artifact/re-encode raw logit",
                tolerance=0.0,
            )
            _compare_float(
                artifact_probability,
                probability,
                label=f"row {sample_id} artifact/re-encode sigmoid",
                tolerance=0.0,
            )
            audited = _audit_score_fields(
                row,
                replay_raw_logit=artifact_raw,
                replay_probability=artifact_probability,
            )
            raw_differences.append(
                abs(float(row["raw_logit"]) - artifact_raw)
            )
            probability_differences.append(
                abs(float(row["probability"]) - artifact_probability)
            )
            replay_scores[sample_id] = audited

            forged_input = forged_by_task.get(str(canonical["task_id"]))
            if forged_input is None:
                raise ValueError(
                    f"row {sample_id} has no paired forged canonical input"
                )
            visibility = _visibility_from_exact_gt(
                forged_input,
                repo_root=repo_root,
                profile_kind=profile_kind,
                profile_id=profile_id,
            )
            prior = visibility_by_task.setdefault(
                str(canonical["task_id"]),
                visibility,
            )
            _require_equal(
                visibility,
                prior,
                f"pair {canonical['task_id']} visibility replay",
            )
            _audit_visibility_record(row, visibility)
            selected_kinds[str(canonical["task_id"])].add(
                str(canonical["kind"])
            )

    visibility_counts = Counter(
        str(value["category"]) for value in visibility_by_task.values()
    )
    complete_counts = Counter(
        str(visibility_by_task[task_id]["category"])
        for task_id, kinds in selected_kinds.items()
        if kinds == {"real", "forged"}
    )
    full_selection = len(selected) == 550 and len(visibility_by_task) == 275
    expected_census = (
        CURRENT_HEAD_VISIBILITY_CENSUS
        if profile_kind == "current_head_native"
        else CHECKPOINT_ERA_VISIBILITY_CENSUS
    )
    if full_selection:
        _require_equal(
            dict(visibility_counts),
            expected_census,
            f"full {profile_id} visibility census",
        )

    pairs = _complete_result_pairs(selected, latest)
    crop_equivalence = _audit_canonical_crop_pair_equivalence(
        pairs=pairs,
        crop_rgb_arrays=crop_rgb_arrays,
        crop_hashes=crop_hashes,
        tensor_hashes=tensor_hashes,
        artifact_features=artifact_features,
        reencoded_features=reencoded_features,
        replay_scores=replay_scores,
        visibility_by_task=visibility_by_task,
        profile_kind=profile_kind,
        full_selection=full_selection,
    )

    return {
        "images_redecoded": len(selected),
        "images_fully_reencoded": len(selected),
        "persisted_features_validated": len(selected),
        "feature_shape": [FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_l2_normalized": False,
        "maximum_feature_absolute_difference": max(
            feature_max_differences,
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
        "head_replayed_from_persisted_feature": True,
        "float32_sigmoid_replayed": True,
        "strict_decision_replayed": "probability > 0.5",
        "preprocess_profile": profile_id,
        "profile_kind": profile_kind,
        "crop_rgb_hash_replayed": True,
        "normalized_tensor_hash_replayed": True,
        "edit_visibility_tasks": dict(sorted(visibility_counts.items())),
        "complete_pair_edit_visibility": dict(sorted(complete_counts.items())),
        "full_selection": full_selection,
        "required_full_visibility_census": expected_census,
        "canonical_crop_pair_equivalence": crop_equivalence,
        "gt_visibility_interpretation": (
            "pre-canonicalization input-location condition only; Mouse "
            "exact-diff GT is computed before independent real/forged "
            "JPEG_q95 canonicalization and cannot establish canonical crop "
            "or downstream equality"
        ),
        "runtime": replay_runtime.evidence,
    }


def _compare_summary_payload(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    label: str = "summary",
) -> None:
    for key, expected in recomputed.items():
        if key not in recorded:
            raise ValueError(f"{label} lacks recomputed field {key}")
        _compare_nested_subset(
            recorded[key],
            expected,
            label=f"{label}.{key}",
            float_tolerance=1e-12,
        )


def recompute_summary(
    *,
    result_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    recorded_summary: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    classification = _require_mapping(
        protocol.get("classification"),
        "manifest classification protocol",
    )
    threshold = classification.get(
        "threshold",
        classification.get(
            "ai_score_threshold",
            classification.get("released_threshold"),
        ),
    )
    _require_equal(threshold, FIXED_THRESHOLD, "fixed UFD threshold")
    iterations = protocol.get(
        "bootstrap_samples",
        DEFAULT_BOOTSTRAP_SAMPLES,
    )
    seed = protocol.get("seed", DEFAULT_BOOTSTRAP_SEED)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("manifest bootstrap sample count is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest bootstrap seed is invalid")
    recomputed = summarize_ufd_results(
        result_rows,
        selected,
        threshold=threshold,
        bootstrap_samples=iterations,
        seed=seed,
    )
    _compare_summary_payload(recorded_summary, recomputed)
    return recomputed


def audit_prefix_reproducibility(
    *,
    repo_root: Path,
    full_run_id: str,
    full_manifest: Mapping[str, Any],
    full_rows: list[dict[str, Any]],
    prefix_run_id: str,
    prefix_manifest: Mapping[str, Any],
    prefix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require an independent exact deterministic prefix run."""

    if prefix_run_id == full_run_id:
        raise ValueError("prefix/full must use independent run IDs")
    _require_equal(
        prefix_manifest.get("run_id"),
        prefix_run_id,
        "prefix manifest run ID",
    )
    full_fingerprint = _require_sha256(
        full_manifest.get("fingerprint"),
        "full manifest fingerprint",
    )
    prefix_fingerprint = _require_sha256(
        prefix_manifest.get("fingerprint"),
        "prefix manifest fingerprint",
    )
    if prefix_fingerprint == full_fingerprint:
        raise ValueError("prefix/full must use independent fingerprints")
    _require_equal(
        prefix_fingerprint,
        _manifest_fingerprint(prefix_manifest),
        "prefix manifest fingerprint",
    )
    full_selection = _require_mapping(
        full_manifest.get("selection"),
        "full selection",
    )
    prefix_selection = _require_mapping(
        prefix_manifest.get("selection"),
        "prefix selection",
    )
    full_ordered = _require_list(
        full_selection.get("rows"),
        "full ordered inputs",
    )
    prefix_ordered = _require_list(
        prefix_selection.get("rows"),
        "prefix ordered inputs",
    )
    if not prefix_ordered:
        raise ValueError("prefix selection is empty")
    if full_ordered[: len(prefix_ordered)] != prefix_ordered:
        raise ValueError("prefix selection is not an exact full-run prefix")
    for key in (
        "condition",
        "preprocess_profile",
        "preprocess_profile_contract",
        "source_checkpoint_drift",
        "runtime_contract",
        "environment",
        "protocol",
        "dataset",
        "adapter_contract",
    ):
        _require_equal(
            prefix_manifest.get(key),
            full_manifest.get(key),
            f"prefix/full manifest {key}",
        )
    full_model = _require_mapping(full_manifest.get("model"), "full model")
    prefix_model = _require_mapping(prefix_manifest.get("model"), "prefix model")
    _require_equal(prefix_model, full_model, "prefix/full model contract")

    full_latest = _latest_by_id(full_rows)
    prefix_latest = _latest_by_id(prefix_rows)
    prefix_ids = [str(item["sample_id"]) for item in prefix_ordered]
    _require_equal(
        set(prefix_latest),
        set(prefix_ids),
        "prefix latest result IDs",
    )
    if any(sample_id not in full_latest for sample_id in prefix_ids):
        raise ValueError("full run is missing a prefix sample")

    compared = 0
    for sample_id in prefix_ids:
        prefix = prefix_latest[sample_id]
        full = full_latest[sample_id]
        _require_equal(prefix.get("status"), "ok", f"prefix {sample_id} status")
        _require_equal(full.get("status"), "ok", f"full {sample_id} status")
        _require_equal(
            prefix.get("run_id"),
            prefix_run_id,
            f"prefix {sample_id} own run ID",
        )
        _require_equal(
            prefix.get("run_manifest_fingerprint"),
            prefix_fingerprint,
            f"prefix {sample_id} own fingerprint",
        )
        _require_equal(
            full.get("run_id"),
            full_run_id,
            f"full {sample_id} own run ID",
        )
        _require_equal(
            full.get("run_manifest_fingerprint"),
            full_fingerprint,
            f"full {sample_id} own fingerprint",
        )
        prefix_feature, prefix_path = _load_feature(
            prefix,
            repo_root=repo_root,
            run_id=prefix_run_id,
        )
        full_feature, full_path = _load_feature(
            full,
            repo_root=repo_root,
            run_id=full_run_id,
        )
        if prefix_path == full_path:
            raise ValueError(
                f"prefix {sample_id} reuses full-run feature artifact"
            )
        _require_equal(
            prefix.get("clip_feature_sha256"),
            full.get("clip_feature_sha256"),
            f"prefix/full {sample_id} feature file SHA-256",
        )
        if not np.array_equal(prefix_feature, full_feature):
            raise ValueError(f"prefix/full {sample_id} features differ")
        for field in _PREFIX_EXACT_FIELDS:
            if field == "clip_feature_sha256":
                continue
            _require_equal(
                prefix.get(field),
                full.get(field),
                f"prefix/full {sample_id} {field}",
            )
        compared += 1
    return {
        "policy": (
            "independent run identities and feature paths; exact ordered "
            "prefix; byte-identical float32 features, preprocessing, scores, "
            "and decisions"
        ),
        "prefix_run_id": prefix_run_id,
        "full_run_id": full_run_id,
        "prefix_manifest_fingerprint": prefix_fingerprint,
        "full_manifest_fingerprint": full_fingerprint,
        "prefix_images": len(prefix_ids),
        "prefix_pairs": len(prefix_ids) // 2,
        "samples_compared": compared,
        "copied_full_rows_rejected": True,
    }


def _load_run_files(
    *,
    results_dir: Path,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    result_path = results_dir / f"{run_id}.jsonl"
    manifest_path = results_dir / f"{run_id}.run_manifest.json"
    summary_path = results_dir / f"{run_id}.summary.json"
    for path in (result_path, manifest_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_jsonl(result_path)
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        f"{run_id} run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        f"{run_id} summary",
    )
    return rows, manifest, summary


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    source_root = Path(args.ufd_root).resolve()
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    all_inputs = read_jsonl(inputs_path)
    result_rows, manifest, summary = _load_run_files(
        results_dir=results_dir,
        run_id=args.run_id,
    )
    selected = _select_manifest_inputs(all_inputs, manifest)
    provenance = validate_provenance(
        repo_root=repo_root,
        source_root=source_root,
        run_id=args.run_id,
        inputs_path=inputs_path,
        input_rows=all_inputs,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
    )
    runtime = _kernel_replay_runtime(manifest)
    latest = _latest_by_id(result_rows)
    artifacts = audit_artifacts(
        repo_root=repo_root,
        source_root=source_root,
        manifest=manifest,
        all_input_rows=all_inputs,
        selected=selected,
        latest=latest,
        runtime=runtime,
    )
    recomputed = recompute_summary(
        result_rows=result_rows,
        selected=selected,
        manifest=manifest,
        recorded_summary=summary,
    )
    prefix_audit: dict[str, Any] | None = None
    prefix_provenance: dict[str, Any] | None = None
    prefix_artifacts: dict[str, Any] | None = None
    prefix_recomputed: dict[str, Any] | None = None
    if args.prefix_run_id:
        prefix_dir = _anchored(
            Path(args.prefix_results_dir or args.results_dir),
            repo_root,
        )
        prefix_rows, prefix_manifest, prefix_summary = _load_run_files(
            results_dir=prefix_dir,
            run_id=args.prefix_run_id,
        )
        prefix_selected = _select_manifest_inputs(all_inputs, prefix_manifest)
        prefix_provenance = validate_provenance(
            repo_root=repo_root,
            source_root=source_root,
            run_id=args.prefix_run_id,
            inputs_path=inputs_path,
            input_rows=all_inputs,
            result_rows=prefix_rows,
            manifest=prefix_manifest,
            summary=prefix_summary,
        )
        prefix_runtime = _kernel_replay_runtime(prefix_manifest)
        prefix_artifacts = audit_artifacts(
            repo_root=repo_root,
            source_root=source_root,
            manifest=prefix_manifest,
            all_input_rows=all_inputs,
            selected=prefix_selected,
            latest=_latest_by_id(prefix_rows),
            runtime=prefix_runtime,
        )
        prefix_recomputed = recompute_summary(
            result_rows=prefix_rows,
            selected=prefix_selected,
            manifest=prefix_manifest,
            recorded_summary=prefix_summary,
        )
        prefix_audit = audit_prefix_reproducibility(
            repo_root=repo_root,
            full_run_id=args.run_id,
            full_manifest=manifest,
            full_rows=result_rows,
            prefix_run_id=args.prefix_run_id,
            prefix_manifest=prefix_manifest,
            prefix_rows=prefix_rows,
        )

    analysis = {
        "schema_version": "ufd_independent_analysis_v1",
        "run_id": args.run_id,
        "generated_at": utc_now(),
        "method": (
            "physical-history/latest audit; immutable source/head/CLIP and "
            "runtime verification; independent Pillow preprocessing, crop "
            "RGB/tensor hashing, pre-canonical exact-GT visibility, canonical "
            "crop pair-equality census, conditional downstream equality, full "
            "CUDA CLIP re-encoding, persisted-feature comparison, linear-head "
            "replay, float32 sigmoid, strict >0.5 decision, and ufd_metrics "
            "recomputation"
        ),
        "result_history": summarize_result_history(result_rows),
        "provenance": provenance,
        "artifact_replay": artifacts,
        "recomputed_summary": recomputed,
        "prefix_provenance": prefix_provenance,
        "prefix_artifact_replay": prefix_artifacts,
        "prefix_recomputed_summary": prefix_recomputed,
        "prefix_reproducibility": prefix_audit,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_outputs_rejected": True,
            "s_joint_outputs_rejected": True,
        },
        "status": "audited",
    }
    output_path = (
        _anchored(Path(args.output), repo_root)
        if args.output
        else results_dir / f"{args.run_id}.analysis.json"
    )
    atomic_write_json(output_path, analysis)
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a UniversalFakeDetect run",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    parser.add_argument("--ufd-root", default=str(DEFAULT_UFD_ROOT))
    parser.add_argument("--prefix-run-id")
    parser.add_argument("--prefix-results-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    analysis = analyze(parse_args())
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
