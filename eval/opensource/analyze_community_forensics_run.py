#!/usr/bin/env python3
"""Independently audit a frozen Community Forensics inference run.

The analyzer treats runner JSON and feature files as untrusted evidence.  It
revalidates the pinned Git and Hugging Face assets, independently decodes and
preprocesses every selected image, runs a fresh timm ViT-S/16 model, captures
the 384-dimensional classifier-head input, and replays the released linear
head, float32 sigmoid, and strict ``probability > 0.5`` decision.  It then
recomputes the benchmark summary from the physical JSONL history.

Community Forensics is an image-level detector.  T2, localization, pixel-mask,
and S_joint claims are rejected recursively.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

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
from eval.opensource.community_forensics_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_community_forensics_results,
)


DEFAULT_RESULTS_DIR = Path("results/opensource/community_forensics")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_RUN_ID = (
    "community_forensics_highres_vit_s16_384_"
    "mouse_canonical_v1_full275_20260725"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/Community-Forensics-ee5b71d4"
)
DEFAULT_MODEL_ROOT = Path(
    "/root/.cache/claimforge/models/community_forensics/"
    "commfor-model-384-6076002b"
)
DEFAULT_PROCESSOR_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "commfor-data-preprocessor-3540a3f0"
)

FROZEN_SOURCE_COMMIT = "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"
FROZEN_EVAL_SINGLE_COMMIT = "5e52ed690bdbd609f9bb1705c4c80d11872a05bd"
FROZEN_MODEL_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
FROZEN_PROCESSOR_REVISION = "3540a3f0d688f8bf492a8aed48613b891f88047e"
FROZEN_CHECKPOINT_SHA256 = (
    "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
)
FROZEN_CHECKPOINT_BYTES = 87_262_324
FROZEN_CHECKPOINT_TENSORS = 152
FROZEN_CHECKPOINT_ELEMENTS = 21_811_969
FROZEN_MODEL_PARAMETER_COUNT = 21_811_969
FROZEN_TIMM_MODEL = "vit_small_patch16_384.augreg_in21k_ft_in1k"
FROZEN_CANONICAL_RELEASE = {
    "schema_version": "claimforge_mouse_canonical_v1",
    "dataset_id": "claimforge-mouse-good275-canonical-jpeg-q95-v1",
    "pairs": 275,
    "images": 550,
    "inputs_sha256": (
        "e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a"
    ),
    "pairs_sha256": (
        "bb6328be7cc7d4ae74b1e5b0b132f7fb6133c6fe73f294ebb46aebeda4f8f4b8"
    ),
    "contract_sha256": (
        "c419e24d6f9d69822ca575e00e30f2c769ba7a28a2fcea1f6634466caf540757"
    ),
}
FROZEN_GOLDEN_CASES = (
    (
        "00000274.png",
        "43a7b37d75eac6a04ab0b7b75655b7f12ea97f5de5055f6025b078a6cf36ba09",
        0.9988338351,
    ),
    (
        "00000420.png",
        "069af2f62865b68c3c0d5a7feb98667306b8b13836d136a7a1c304a2eb172f8b",
        0.9878403544,
    ),
    (
        "00000845.png",
        "e3c1d98ae99c54e58546c92db007071913c698803435351a4bcf80efc223247e",
        0.9568564892,
    ),
    (
        "00000916.png",
        "a4c434f68a4b8bf2ed6bb6f15b25c72ff8e159fd6c669ec0c35c29a96d0b05be",
        0.9516021609,
    ),
    (
        "00000989.png",
        "cfe5ba7592ee9f0b3e04f80fc1f9df5081418a93e12bb1dc32d6d64eb361393a",
        0.7860031724,
    ),
)
GOLDEN_ABSOLUTE_TOLERANCE = 1e-5

FROZEN_MAIN_SOURCE_FILES = {
    "LICENSE": "7ca7b8f7aaf663c7941e3bb851a4c1bbfdef51d5503be93711614f528569a5c6",
    "README.md": "c4673b4eeb21b52fb38b116439073abca9724423f7c48b16aa2956f160f491f5",
    "models.py": "988f2f566aa4c177dd845bfccdab9eac2b3fa5174649332b09ee8cc840c8bb97",
    "dataloader.py": "e33032c6bf5e18c5ae3ae0a0391a8571a7bb2cfa4b140c0eb80f5ce86f14562b",
    "custom_transforms.py": (
        "85e49fee816b20859d6f1f96fbe51d296c2d4c7b6c371326b764f57a58a25bdb"
    ),
    "dataprocessor_hf.py": (
        "3cec0839d2683694651187439f6e137928084aa0b49a80a692085fad9f5d9d92"
    ),
    "eval.py": "ce7c2671c0a66bdf810d633021b1c95913f54adff1ce8b6958a4bfaa4c5c8815",
    "eval_using_huggingface.ipynb": (
        "dde770009f581fc51e2e234f7559d38b4dccaf2f604ba58c86a67263572e8d3c"
    ),
}
FROZEN_EVAL_SINGLE_FILES = {
    "main.py": "1aabf779060c343f60d38d7f2600c42a489a39cc93702bd754e1fa5b26ea4758",
    "models.py": "8d96f826344802ca55e2e1382550778136c3865df617b68951f3a9e80ff304d0",
    "LICENSE": "7ca7b8f7aaf663c7941e3bb851a4c1bbfdef51d5503be93711614f528569a5c6",
    "README.md": "4f0fc3272e8acc645ed1bc96935c480a3ee799a326dce707046cfd193736cd9f",
    "requirements.txt": (
        "54c71f1353465013e6df0c04563052a0d4ad13c02bb46174d629708f843ca04b"
    ),
}
FROZEN_MODEL_FILES = {
    "README.md": "47fc83b44bdc02e73e0c0497598668c8cf4a9fb03842a6f64849eb8736644485",
    "config.json": "877416e73aac0fdbc1be723f5ddf674e78a2350305865ad9031564b287b60147",
    "model.safetensors": FROZEN_CHECKPOINT_SHA256,
}
FROZEN_PROCESSOR_FILES = {
    ".gitattributes": (
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"
    ),
    "README.md": "13bd1a9bc6406fc7cb6b81fee48509912ee6b20d4798034fb1c1154fedf4bcc6",
    "custom_transforms.py": (
        "85e49fee816b20859d6f1f96fbe51d296c2d4c7b6c371326b764f57a58a25bdb"
    ),
    "dataloader.py": "eae0d9a33eaebbf26b0813f7bd7c8cf5fe136e4053502ee888f5ae37d379d4aa",
    "dataprocessor_hf.py": (
        "4ea40fa1c24e8d620252342254263ede5a25b2717bd3d931b81bc3a72ffa558b"
    ),
    "preprocessor_config.json": (
        "f498f9ef77d9f4e80dc6aab5ec4a9e5147420f81b3b650cedfc78a063fa96f9b"
    ),
}

FEATURE_DIMENSION = 384
FEATURE_DTYPE = np.dtype("float32")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
RESIZE_SHORT_SIDE = 440
CROP_SIZE = 384
RAW_LOGIT_ABSOLUTE_TOLERANCE = 1e-5
PROBABILITY_ABSOLUTE_TOLERANCE = 1e-7

# These values are cross-checked against the runner constants at audit time.
# The exact profile/semantics names are release-facing schema values.
FROZEN_PROFILE = "official_highres_resize440_centercrop384"
FEATURE_SEMANTICS = "timm_vit_forward_head_pre_logits_classifier_input"
SCORE_SEMANTICS = "torch_float32_sigmoid_of_single_raw_logit"
T1_POLICY = "released_probability_strictly_greater_than_0_5"

_T2_OR_LOCALIZATION_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "score_map",
        "score_map_path",
        "attention_map",
        "attention_map_path",
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
    "commfor_feature_sha256",
    "commfor_feature_array_sha256",
    "feature_array_sha256",
    "commfor_feature_shape",
    "commfor_feature_dtype",
    "commfor_feature_semantics",
    "valid_for_metrics",
    "preprocess",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
)


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent CPU preprocessing evidence."""

    tensor: Any
    decoded_rgb: np.ndarray
    crop_rgb: np.ndarray
    decoded_rgb_sha256: str
    resized_rgb_sha256: str
    crop_rgb_sha256: str
    tensor_sha256: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """Torch and device used for the independent numerical replay."""

    torch: ModuleType
    device: Any
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RunFiles:
    """One immutable-on-disk runner output bundle."""

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
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _require_probability(value: Any, label: str) -> float:
    number = _require_finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} falls outside [0, 1]")
    return number


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _safe_component(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise ValueError(f"{label} must be one safe non-empty path component")
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
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    return _array_sha256(tensor.detach().cpu().contiguous().numpy())


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _selected_rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        raise ValueError(f"cannot read frozen git object {revision_path}") from exc


def _module_pin(module: ModuleType, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return copy.deepcopy(getattr(module, name))
    raise RuntimeError(
        "Community Forensics runner lacks required audit pin "
        f"(one of {names})"
    )


def _load_runner_pins() -> SimpleNamespace:
    """Import only immutable public constants from the runner."""

    from eval.opensource import run_community_forensics as runner

    return SimpleNamespace(
        MODEL_NAME=_module_pin(runner, "MODEL_NAME"),
        MODEL_SLUG=_module_pin(runner, "MODEL_SLUG"),
        MODEL_ARCH=_module_pin(runner, "MODEL_ARCH"),
        MODEL_REPO_URL=_module_pin(runner, "MODEL_REPO_URL"),
        PAPER_URL=_module_pin(runner, "PAPER_URL"),
        MODEL_SOURCE_COMMIT=_module_pin(runner, "MODEL_SOURCE_COMMIT"),
        EVAL_SINGLE_COMMIT=_module_pin(runner, "EVAL_SINGLE_COMMIT"),
        MODEL_HF_REVISION=_module_pin(runner, "MODEL_HF_REVISION"),
        PROCESSOR_HF_REVISION=_module_pin(runner, "PROCESSOR_HF_REVISION"),
        SOURCE_FILES=_module_pin(runner, "SOURCE_FILES"),
        EVAL_SINGLE_FILES=_module_pin(runner, "EVAL_SINGLE_FILES"),
        MODEL_FILES=_module_pin(
            runner,
            "MODEL_ASSET_FILES",
            "MODEL_FILES",
        ),
        PROCESSOR_FILES=_module_pin(
            runner,
            "PROCESSOR_ASSET_FILES",
            "PROCESSOR_FILES",
        ),
        CHECKPOINT=_module_pin(runner, "CHECKPOINT"),
        PREPROCESS_PROFILE=_module_pin(runner, "PREPROCESS_PROFILE"),
        FEATURE_DIMENSION=int(_module_pin(runner, "FEATURE_DIMENSION")),
        FEATURE_SEMANTICS=_module_pin(
            runner,
            "FEATURE_SEMANTICS",
        ),
        SCORE_SEMANTICS=_module_pin(runner, "SCORE_SEMANTICS"),
        T1_POLICY=_module_pin(runner, "T1_POLICY"),
        CANONICAL_RELEASE=_module_pin(runner, "CANONICAL_RELEASE"),
        GOLDEN_CASES=_module_pin(runner, "GOLDEN_CASES"),
        GOLDEN_ABS_TOLERANCE=float(
            _module_pin(runner, "GOLDEN_ABS_TOLERANCE")
        ),
        IMAGE_MEAN=tuple(_module_pin(runner, "IMAGE_MEAN")),
        IMAGE_STD=tuple(_module_pin(runner, "IMAGE_STD")),
        LICENSE_RECORD=_module_pin(runner, "LICENSE_RECORD"),
        CLASSIFICATION_THRESHOLD=float(
            _module_pin(runner, "CLASSIFICATION_THRESHOLD")
        ),
        CLASSIFICATION_THRESHOLD_OPERATOR=str(
            _module_pin(runner, "CLASSIFICATION_THRESHOLD_OPERATOR")
        ),
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
    """Recursively reject output types absent from the released detector."""

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


def resize_short_side_dimensions(
    width: int,
    height: int,
    short_side: int = RESIZE_SHORT_SIDE,
) -> tuple[int, int]:
    """Match torchvision's integer-size PIL Resize geometry."""

    if width <= 0 or height <= 0 or short_side <= 0:
        raise ValueError("image dimensions and short side must be positive")
    if width <= height:
        return int(short_side), int(short_side * height / width)
    return int(short_side * width / height), int(short_side)


def _center_crop_geometry(
    width: int,
    height: int,
    size: int = CROP_SIZE,
) -> dict[str, Any]:
    """Match torchvision CenterCrop, including symmetric zero padding."""

    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("center-crop dimensions must be positive")
    pad_left = max((size - width) // 2, 0)
    pad_top = max((size - height) // 2, 0)
    pad_right = max((size - width + 1) // 2, 0)
    pad_bottom = max((size - height + 1) // 2, 0)
    padded_width = width + pad_left + pad_right
    padded_height = height + pad_top + pad_bottom
    left = int(round((padded_width - size) / 2.0))
    top = int(round((padded_height - size) / 2.0))
    return {
        "input_size": [width, height],
        "padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
        "padding_fill": 0,
        "padded_size": [padded_width, padded_height],
        "start_xy": [left, top],
        "size": [size, size],
        "end_xy": [left + size, top + size],
        "rounding": "int(round((dimension-crop)/2.0))",
    }


def _center_crop_rgb(image: Image.Image, size: int = CROP_SIZE) -> Image.Image:
    geometry = _center_crop_geometry(*image.size, size=size)
    padding = geometry["padding_ltrb"]
    if any(padding):
        image = ImageOps.expand(image, border=tuple(padding), fill=0)
    left, top = geometry["start_xy"]
    return image.crop((left, top, left + size, top + size))


def compute_preprocess_geometry(width: int, height: int) -> dict[str, Any]:
    """Independently reproduce the released torchvision pixel geometry."""

    resized_width, resized_height = resize_short_side_dimensions(width, height)
    crop_left = int(round((resized_width - CROP_SIZE) / 2.0))
    crop_top = int(round((resized_height - CROP_SIZE) / 2.0))
    crop_right = crop_left + CROP_SIZE
    crop_bottom = crop_top + CROP_SIZE
    return {
        "profile_id": FROZEN_PROFILE,
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "native_size": [width, height],
        "resize": {
            "enabled": True,
            "source_size": [width, height],
            "destination_size": [resized_width, resized_height],
            "short_side": RESIZE_SHORT_SIDE,
            "interpolation": "PIL_BILINEAR",
            "antialias": True,
            "rounding": "torchvision_int_truncation_for_long_side",
        },
        "center_crop": {
            "input_size": [resized_width, resized_height],
            "start_xy": [crop_left, crop_top],
            "size": [CROP_SIZE, CROP_SIZE],
            "end_xy": [crop_right, crop_bottom],
            "rounding": "int(round((dimension-crop)/2.0))",
        },
        "effective_native_crop_xyxy": [
            crop_left * width / resized_width,
            crop_top * height / resized_height,
            crop_right * width / resized_width,
            crop_bottom * height / resized_height,
        ],
        "pixel_center_mapping": (
            "d=(native_index+0.5)*resized_size/native_size-0.5"
        ),
        "normalize": {
            "to_tensor_scale": "uint8_div_255_to_float32",
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }


def preprocess_image(
    path: Path,
    *,
    torch_module: ModuleType,
    profile_id: str = FROZEN_PROFILE,
) -> PreprocessedImage:
    """Independent Pillow/torch implementation of the released 384 profile."""

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.asarray(rgb, dtype=np.uint8).copy()
        resized_width, resized_height = resize_short_side_dimensions(
            width,
            height,
        )
        resized_image = rgb.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        resized = np.asarray(resized_image, dtype=np.uint8).copy()
        crop_image = _center_crop_rgb(resized_image, CROP_SIZE)
        crop = np.asarray(crop_image, dtype=np.uint8).copy()
    _require_equal(
        crop.shape,
        (CROP_SIZE, CROP_SIZE, 3),
        "independent preprocessed crop shape",
    )
    tensor = (
        torch_module.from_numpy(np.ascontiguousarray(crop))
        .permute(2, 0, 1)
        .to(dtype=torch_module.float32)
        .div(255.0)
    )
    mean = torch_module.tensor(
        IMAGE_MEAN,
        dtype=torch_module.float32,
    )[:, None, None]
    std = torch_module.tensor(
        IMAGE_STD,
        dtype=torch_module.float32,
    )[:, None, None]
    tensor = tensor.sub(mean).div(std).contiguous()
    geometry = compute_preprocess_geometry(width, height)
    audit = {
        "profile": profile_id,
        "geometry": geometry,
        "decoded_rgb_sha256": _array_sha256(decoded),
        "resized_rgb_sha256": _array_sha256(resized),
        "resized_rgb_shape": list(resized.shape),
        "resized_rgb_dtype": str(resized.dtype),
        "crop_rgb_sha256": _array_sha256(crop),
        "crop_rgb_shape": list(crop.shape),
        "crop_rgb_dtype": str(crop.dtype),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.detach().cpu().numpy().dtype),
        "tensor_sha256": _tensor_sha256(tensor),
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }
    return PreprocessedImage(
        tensor=tensor,
        decoded_rgb=decoded,
        crop_rgb=crop,
        decoded_rgb_sha256=audit["decoded_rgb_sha256"],
        resized_rgb_sha256=audit["resized_rgb_sha256"],
        crop_rgb_sha256=audit["crop_rgb_sha256"],
        tensor_sha256=audit["tensor_sha256"],
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
    if pixels.ndim == 3:
        if not np.array_equal(pixels, pixels[..., :1]):
            raise ValueError(f"{sample_id} GT channels differ")
        pixels = pixels[..., 0]
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
    """Map exact-GT pixel centers through Resize(440)+CenterCrop(384)."""

    mask = _load_gt_mask(canonical, repo_root=repo_root)
    if mask is None:
        raise ValueError("visibility must be derived from the forged row")
    width = int(canonical["width"])
    height = int(canonical["height"])
    geometry = compute_preprocess_geometry(width, height)
    resized_width, resized_height = geometry["resize"]["destination_size"]
    left, top = geometry["center_crop"]["start_xy"]
    right, bottom = geometry["center_crop"]["end_xy"]
    ys, xs = np.nonzero(mask == 255)
    mapped_x = (
        (xs.astype(np.float64) + 0.5) * resized_width / width
        - 0.5
    )
    mapped_y = (
        (ys.astype(np.float64) + 0.5) * resized_height / height
        - 0.5
    )
    visible_mask = (
        (mapped_x >= left)
        & (mapped_x < right)
        & (mapped_y >= top)
        & (mapped_y < bottom)
    )
    total = int(xs.size)
    visible = int(np.count_nonzero(visible_mask))
    fraction = visible / total
    category = "none" if visible == 0 else "full" if visible == total else "partial"
    return {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": str(canonical["sample_id"]),
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_through_"
            "official_resize440_and_center_crop384"
        ),
        "profile_id": FROZEN_PROFILE,
        "formula": str(geometry["pixel_center_mapping"]),
    }


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
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": native_crop,
        "intersection_xyxy": intersection,
        "edit_area": area,
        "visible_area": visible_area,
        "visible_fraction": fraction,
        "category": (
            "none"
            if fraction == 0.0
            else "full"
            if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
            else "partial"
        ),
        "basis": (
            "continuous_edit_box_area_intersection_with_effective_native_crop"
        ),
    }


def _pair_visibility(
    canonical_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in canonical_rows:
        task_id = str(row.get("task_id"))
        kind = str(row.get("kind"))
        if kind in pairs.setdefault(task_id, {}):
            raise ValueError(f"duplicate canonical {kind} row: {task_id}")
        pairs[task_id][kind] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in pairs.items():
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"canonical task is incomplete: {task_id}")
        real, forged = pair["real"], pair["forged"]
        for key in ("width", "height", "domain", "edit_region_xyxy"):
            _require_equal(
                real.get(key),
                forged.get(key),
                f"canonical pair {task_id} {key}",
            )
        gt = _visibility_from_exact_gt(forged, repo_root=repo_root)
        edit_region = forged.get("edit_region_xyxy")
        if (
            not isinstance(edit_region, list)
            or len(edit_region) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in edit_region
            )
        ):
            raise ValueError(f"invalid edit region for {task_id}")
        geometry = compute_preprocess_geometry(
            int(forged["width"]),
            int(forged["height"]),
        )
        result[task_id] = {
            "edit_visibility": gt["category"],
            "edit_visible_gt_fraction": gt["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": gt,
                "edit_box": _edit_box_visibility(
                    edit_region,
                    list(geometry["effective_native_crop_xyxy"]),
                ),
            },
        }
    return result


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
    number = _require_finite(actual, label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{label} mismatch: {number!r} != {expected!r} "
            f"(absolute tolerance {tolerance})"
        )
    return number


def _audit_score_fields(
    row: Mapping[str, Any],
    *,
    replay_raw_logit: float,
    replay_probability: float,
    raw_tolerance: float = RAW_LOGIT_ABSOLUTE_TOLERANCE,
    probability_tolerance: float = PROBABILITY_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Validate float32 scores, aliases, and released strict threshold."""

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
        f"row {row_id} classification operator",
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
            "classifier_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
        label=f"row {row_id} manual replay",
        exact_mapping_keys=True,
    )
    raw_decision = float(replay_raw_logit) > 0.0
    return {
        "raw_logit": float(replay_raw_logit),
        "probability": float(replay_probability),
        "decision": bool(decision),
        "raw_logit_decision": bool(raw_decision),
        "decision_equivalent": bool(decision == raw_decision),
    }


def _load_feature(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_dir: Path | None = None,
) -> tuple[np.ndarray, Path]:
    value = row.get("commfor_feature_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row.get('id')} has no Community feature path")
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
        expected_path = (
            run_dir / "features" / f"{row.get('id')}.npy"
        ).resolve()
        _require_equal(
            path.resolve(),
            expected_path,
            f"row {row.get('id')} exact feature path",
        )
    _verify_hash(
        path,
        row.get("commfor_feature_sha256"),
        f"row {row.get('id')} Community feature",
    )
    try:
        feature = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"row {row.get('id')} Community feature is not a safe NPY"
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
        row.get("commfor_feature_shape"),
        [FEATURE_DIMENSION],
        f"row {row.get('id')} recorded feature shape",
    )
    _require_equal(
        row.get("commfor_feature_dtype"),
        "float32",
        f"row {row.get('id')} recorded feature dtype",
    )
    _require_equal(
        row.get("commfor_feature_semantics"),
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
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {line_number} has no id")
        latest[row_id] = row
    return latest


def _verify_source_tree(
    source_root: Path,
    *,
    pins: SimpleNamespace,
) -> dict[str, Any]:
    """Verify main HEAD plus the official eval_single Git objects."""

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"missing Community Forensics source root: {source_root}"
        )
    _require_equal(
        pins.MODEL_SOURCE_COMMIT,
        FROZEN_SOURCE_COMMIT,
        "analyzer/runner main source commit",
    )
    _require_equal(
        pins.EVAL_SINGLE_COMMIT,
        FROZEN_EVAL_SINGLE_COMMIT,
        "analyzer/runner eval_single commit",
    )
    _require_equal(
        pins.SOURCE_FILES,
        FROZEN_MAIN_SOURCE_FILES,
        "analyzer/runner main source hashes",
    )
    _require_equal(
        pins.EVAL_SINGLE_FILES,
        FROZEN_EVAL_SINGLE_FILES,
        "analyzer/runner eval_single source hashes",
    )
    commit = _git_value(source_root, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_SOURCE_COMMIT, "checked-out source commit")
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty is None:
        raise ValueError("source root is not a readable Git repository")
    if dirty:
        raise ValueError("Community Forensics source has tracked modifications")
    for relative, digest in FROZEN_MAIN_SOURCE_FILES.items():
        _verify_hash(
            source_root / relative,
            digest,
            f"Community Forensics main source {relative}",
        )
    eval_single_records: dict[str, dict[str, Any]] = {}
    for relative, digest in FROZEN_EVAL_SINGLE_FILES.items():
        blob = _git_bytes(
            source_root,
            f"{FROZEN_EVAL_SINGLE_COMMIT}:{relative}",
        )
        actual = hashlib.sha256(blob).hexdigest()
        _require_equal(actual, digest, f"eval_single {relative} SHA-256")
        eval_single_records[relative] = {
            "git_object": f"{FROZEN_EVAL_SINGLE_COMMIT}:{relative}",
            "bytes": len(blob),
            "sha256": actual,
        }

    main_model = (source_root / "models.py").read_text(encoding="utf-8")
    main_loader = (source_root / "dataloader.py").read_text(encoding="utf-8")
    single_main = _git_bytes(
        source_root,
        f"{FROZEN_EVAL_SINGLE_COMMIT}:main.py",
    ).decode("utf-8")
    single_model = _git_bytes(
        source_root,
        f"{FROZEN_EVAL_SINGLE_COMMIT}:models.py",
    ).decode("utf-8")
    required = {
        "main models.py": (
            (
                "vit_small_patch16_384."
                "augreg_in21k_ft_in1k"
            ),
            "self.vit.head = nn.Linear(in_features=384, out_features=1",
        ),
        "main dataloader.py": (
            "resize_size=440",
            "crop_size=384",
            "transforms.Resize(resize_size)",
            "transforms.CenterCrop(crop_size)",
        ),
        "eval_single models.py": (
            "transforms.Resize(resize_size)",
            "transforms.CenterCrop(crop_size)",
            "transforms.ToTensor()",
            "norm_mean = [0.485, 0.456, 0.406]",
            "torch.nn.functional.sigmoid(x)",
        ),
        "eval_single main.py": ("if prob > 0.5:",),
    }
    texts = {
        "main models.py": main_model,
        "main dataloader.py": main_loader,
        "eval_single models.py": single_model,
        "eval_single main.py": single_main,
    }
    missing = [
        f"{label}:{needle}"
        for label, needles in required.items()
        for needle in needles
        if needle not in texts[label]
    ]
    if missing:
        raise ValueError(f"official source semantic evidence changed: {missing}")
    return {
        "repo_url": pins.MODEL_REPO_URL,
        "root": str(source_root.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str((source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in FROZEN_MAIN_SOURCE_FILES.items()
        },
        "eval_single": {
            "commit": FROZEN_EVAL_SINGLE_COMMIT,
            "branch_relationship": (
                "separate_official_eval_single_branch_not_main_ancestor"
            ),
            "files": eval_single_records,
            "role": (
                "corroborates single-image RGB/resize/crop/normalize/sigmoid/"
                "strict-threshold execution semantics"
            ),
        },
        "license_record": pins.LICENSE_RECORD,
    }


def _verify_processor_tree(
    processor_root: Path,
    *,
    pins: SimpleNamespace,
) -> dict[str, Any]:
    if not processor_root.is_dir():
        raise FileNotFoundError(f"missing processor root: {processor_root}")
    _require_equal(
        pins.PROCESSOR_HF_REVISION,
        FROZEN_PROCESSOR_REVISION,
        "analyzer/runner processor revision",
    )
    _require_equal(
        pins.PROCESSOR_FILES,
        FROZEN_PROCESSOR_FILES,
        "analyzer/runner processor hashes",
    )
    commit = _git_value(processor_root, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_PROCESSOR_REVISION, "processor revision")
    dirty = _git_value(
        processor_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty is None:
        raise ValueError("processor root is not a readable Git repository")
    if dirty:
        raise ValueError("processor source has tracked modifications")
    for relative, digest in FROZEN_PROCESSOR_FILES.items():
        _verify_hash(
            processor_root / relative,
            digest,
            f"processor source {relative}",
        )
    config = _require_mapping(
        json.loads(
            (processor_root / "preprocessor_config.json").read_text(
                encoding="utf-8"
            )
        ),
        "processor config",
    )
    _require_equal(
        config,
        {
            "image_processor_type": "CommForImageProcessor",
            "size": 384,
            "auto_map": {
                "AutoImageProcessor": (
                    "dataprocessor_hf.CommForImageProcessor"
                )
            },
        },
        "processor configuration",
    )
    loader = (processor_root / "dataloader.py").read_text(encoding="utf-8")
    required = (
        "resize_size=440",
        "crop_size=384",
        "transforms.Resize(resize_size)",
        "transforms.CenterCrop(crop_size)",
        "ctrans.ToTensor_range(val_min=0, val_max=1)",
        "norm_mean = [0.485, 0.456, 0.406]",
        "norm_std = [0.229, 0.224, 0.225]",
    )
    missing = [needle for needle in required if needle not in loader]
    if missing:
        raise ValueError(f"processor semantic evidence changed: {missing}")
    return {
        "repository": "OwensLab/commfor-data-preprocessor",
        "root": str(processor_root.resolve()),
        "revision": commit,
        "tracked_dirty": False,
        "files": {
            relative: {
                "path": str((processor_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in FROZEN_PROCESSOR_FILES.items()
        },
        "config": config,
    }


def _checkpoint_schema(
    checkpoint_path: Path,
    *,
    torch_module: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect every safetensors entry without pickle execution."""

    from safetensors import safe_open
    from safetensors.torch import load_file

    items: list[dict[str, Any]] = []
    with safe_open(
        checkpoint_path,
        framework="pt",
        device="cpu",
    ) as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
        for key in keys:
            tensor = handle.get_tensor(key)
            _require_equal(
                tensor.dtype,
                torch_module.float32,
                f"checkpoint tensor {key} dtype",
            )
            if not bool(torch_module.isfinite(tensor).all().item()):
                raise ValueError(f"checkpoint tensor {key} is non-finite")
            items.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                    "sha256": _tensor_sha256(tensor),
                }
            )
    _require_equal(metadata, None, "checkpoint safetensors metadata")
    state = dict(load_file(checkpoint_path, device="cpu"))
    _require_equal(list(state), keys, "safetensors loader key order")
    schema = {
        "format": "safetensors",
        "metadata": metadata,
        "tensor_count": len(items),
        "state_elements": sum(item["numel"] for item in items),
        "all_dtype": "torch.float32",
        "items_sha256": hashlib.sha256(
            stable_json(items).encode("utf-8")
        ).hexdigest(),
        "keys": keys,
        "items": items,
    }
    _require_equal(
        schema["tensor_count"],
        FROZEN_CHECKPOINT_TENSORS,
        "checkpoint tensor count",
    )
    _require_equal(
        schema["state_elements"],
        FROZEN_CHECKPOINT_ELEMENTS,
        "checkpoint state elements",
    )
    return state, schema


def _verify_assets(
    *,
    model_root: Path,
    processor_root: Path,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the selected 384 checkpoint and processor snapshot."""

    _require_equal(
        pins.MODEL_HF_REVISION,
        FROZEN_MODEL_REVISION,
        "analyzer/runner model revision",
    )
    _require_equal(
        pins.MODEL_FILES,
        FROZEN_MODEL_FILES,
        "analyzer/runner model asset hashes",
    )
    _require_equal(
        pins.CHECKPOINT["sha256"],
        FROZEN_CHECKPOINT_SHA256,
        "analyzer/runner checkpoint SHA-256",
    )
    for key, expected in {
        "bytes": FROZEN_CHECKPOINT_BYTES,
        "tensor_count": FROZEN_CHECKPOINT_TENSORS,
        "state_elements": FROZEN_CHECKPOINT_ELEMENTS,
        "trainable_parameters": FROZEN_MODEL_PARAMETER_COUNT,
    }.items():
        _require_equal(
            int(pins.CHECKPOINT[key]),
            expected,
            f"analyzer/runner checkpoint {key}",
        )
    if not model_root.is_dir():
        raise FileNotFoundError(f"missing model root: {model_root}")
    for relative, digest in FROZEN_MODEL_FILES.items():
        _verify_hash(model_root / relative, digest, f"model asset {relative}")
    checkpoint_path = model_root / "model.safetensors"
    _require_equal(
        checkpoint_path.stat().st_size,
        FROZEN_CHECKPOINT_BYTES,
        "checkpoint byte size",
    )
    model_config = _require_mapping(
        json.loads(
            (model_root / "config.json").read_text(encoding="utf-8")
        ),
        "model config",
    )
    _require_equal(
        model_config,
        {
            "device": "cuda",
            "freeze_backbone": False,
            "input_size": 384,
            "model_size": "small",
            "patch_size": 16,
        },
        "model configuration",
    )
    if "license: mit" not in (
        model_root / "README.md"
    ).read_text(encoding="utf-8"):
        raise ValueError("model-card MIT metadata changed")
    state, schema = _checkpoint_schema(
        checkpoint_path,
        torch_module=torch_module,
    )
    processor = _verify_processor_tree(processor_root, pins=pins)
    checkpoint_record = {
        **dict(pins.CHECKPOINT),
        "path": str(checkpoint_path.resolve()),
        "actual_bytes": checkpoint_path.stat().st_size,
        "actual_sha256": sha256_file(checkpoint_path),
        "serialization_safety": {
            "format": "safetensors",
            "pickle_executed": False,
            "loader": "safetensors.torch.load_file",
        },
        "schema": schema,
    }
    assets = {
        "checkpoint": checkpoint_record,
        "model_repository": {
            "repository": "OwensLab/commfor-model-384",
            "revision": FROZEN_MODEL_REVISION,
            "root": str(model_root.resolve()),
            "files": {
                relative: {
                    "path": str((model_root / relative).resolve()),
                    "sha256": digest,
                }
                for relative, digest in FROZEN_MODEL_FILES.items()
            },
            "config": model_config,
        },
        "processor": processor,
        "bundle_sha256": hashlib.sha256(
            stable_json(
                {
                    "model_revision": FROZEN_MODEL_REVISION,
                    "model_sha256": FROZEN_CHECKPOINT_SHA256,
                    "processor_revision": FROZEN_PROCESSOR_REVISION,
                    "processor_files": FROZEN_PROCESSOR_FILES,
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
    return state, assets


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validated_runtime_truth(torch_module: ModuleType) -> dict[str, Any]:
    truth = {
        "cudnn_enabled": bool(torch_module.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(
            torch_module.backends.cudnn.deterministic
        ),
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(
            torch_module.backends.cudnn.allow_tf32
        ),
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": (
            torch_module.get_float32_matmul_precision()
        ),
    }
    _require_equal(
        truth,
        {
            "cudnn_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
        },
        "actual replay runtime truth",
    )
    return truth


def _replay_runtime(manifest: Mapping[str, Any]) -> ReplayRuntime:
    """Verify and recreate the recorded deterministic numerical runtime."""

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
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace != ":4096:8":
        raise ValueError(
            "deterministic replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    import torch

    current = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "safetensors": _package_version("safetensors"),
        "scikit_learn": _package_version("scikit-learn"),
    }
    for key, expected in current.items():
        _require_equal(runtime.get(key), expected, f"runtime {key}")
    _require_equal(runtime.get("timm"), "1.0.15", "runtime timm pin")
    requested = runtime.get("device")
    if not isinstance(requested, str):
        raise ValueError("runtime device is not a string")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recorded CUDA runtime is unavailable")
        if device.index is None:
            raise ValueError("recorded CUDA device lacks explicit index")
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"recorded CUDA device {device} is unavailable")
        torch.cuda.set_device(device)
        _require_equal(
            runtime.get("cuda_device_name"),
            torch.cuda.get_device_name(device),
            "runtime CUDA device name",
        )
    elif device.type != "cpu":
        raise ValueError("recorded device is neither CPU nor CUDA")
    _require_equal(
        runtime.get("cuda_available"),
        bool(torch.cuda.is_available()),
        "runtime CUDA availability",
    )
    _require_equal(runtime.get("cuda_version"), torch.version.cuda, "CUDA version")
    _require_equal(
        runtime.get("cudnn_version"),
        (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "runtime cuDNN version",
    )
    for key, expected in {
        "seed": 100,
        "dtype": "float32",
        "autocast": False,
        "cudnn_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "tf32": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
    }.items():
        _require_equal(runtime.get(key), expected, f"runtime flag {key}")
    _require_equal(
        runtime.get("torch_allow_tf32_cublas_override_env"),
        os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "runtime TF32 override environment",
    )
    _require_equal(
        runtime.get("timm_fused_attention_policy"),
        "official_timm_1.0.15_default_recorded_at_model_load",
        "runtime timm attention policy",
    )
    _require_equal(
        contract,
        {
            "device": requested,
            "seed": 100,
            "dtype": "float32",
            "autocast": False,
            "cudnn_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "network_allowed": False,
        },
        "frozen runtime contract",
    )
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(100)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    runtime_truth = _validated_runtime_truth(torch)
    return ReplayRuntime(
        torch=torch,
        device=device,
        evidence={
            "device": str(device),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "timm": _package_version("timm"),
            "safetensors": _package_version("safetensors"),
            "pillow": _package_version("Pillow"),
            "numpy": np.__version__,
            **runtime_truth,
            "tf32": False,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    )


def _construct_model(
    *,
    state: Mapping[str, Any],
    runtime: ReplayRuntime,
) -> tuple[Any, dict[str, Any]]:
    """Construct a fresh offline timm model and strictly load every tensor."""

    torch = runtime.torch
    import timm

    _require_equal(timm.__version__, "1.0.15", "timm runtime")

    class CommunityForensicsModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vit = timm.create_model(
                FROZEN_TIMM_MODEL,
                pretrained=False,
            )
            self.vit.head = torch.nn.Linear(
                FEATURE_DIMENSION,
                1,
                bias=True,
                dtype=torch.float32,
            )

        def forward(self, value: Any) -> Any:
            return self.vit(value)

    network_attempts = {
        "urllib_urlopen": 0,
        "socket_create_connection": 0,
        "socket_connect": 0,
    }

    def reject(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            network_attempts[name] += 1
            raise RuntimeError(
                "network access is forbidden during audit model construction"
            )

        return blocked

    offline_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, offline_environment))
        stack.enter_context(
            mock.patch.object(
                urllib.request,
                "urlopen",
                side_effect=reject("urllib_urlopen"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket,
                "create_connection",
                side_effect=reject("socket_create_connection"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket.socket,
                "connect",
                side_effect=reject("socket_connect"),
            )
        )
        model = CommunityForensicsModel()
    if any(network_attempts.values()):
        raise RuntimeError(
            f"audit model construction attempted network: {network_attempts}"
        )
    _require_equal(
        set(model.state_dict()),
        set(state),
        "checkpoint/model state keys",
    )
    incompatibility = model.load_state_dict(state, strict=True)
    _require_equal(
        list(incompatibility.missing_keys),
        [],
        "strict-load missing keys",
    )
    _require_equal(
        list(incompatibility.unexpected_keys),
        [],
        "strict-load unexpected keys",
    )
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), state[key]):
            raise ValueError(f"loaded model tensor differs: {key}")
    parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    _require_equal(
        parameter_count,
        FROZEN_MODEL_PARAMETER_COUNT,
        "model parameter count",
    )
    _require_equal(
        (model.vit.head.in_features, model.vit.head.out_features),
        (FEATURE_DIMENSION, 1),
        "model classifier shape",
    )
    fused_flags = [
        bool(getattr(block.attn, "fused_attn", True))
        for block in model.vit.blocks
    ]
    if not fused_flags or len(set(fused_flags)) != 1:
        raise ValueError("timm attention implementation flags are inconsistent")
    model.to(device=runtime.device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audit = {
        "construction": {
            "framework": "timm",
            "version": "1.0.15",
            "architecture": FROZEN_TIMM_MODEL,
            "pretrained": False,
            "wrapper_state_prefix": "vit.",
            "classifier_replacement": "torch.nn.Linear(384,1,bias=True)",
            "fused_attention": fused_flags[0],
            "fused_attention_policy": "official_timm_1.0.15_default",
            "attention_block_flags": fused_flags,
        },
        "load": {
            "format": "safetensors",
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "full_state_coverage": True,
            "loaded_tensor_count": len(state),
            "loaded_state_elements": sum(
                int(value.numel()) for value in state.values()
            ),
            "parameter_count": parameter_count,
        },
        "network": {
            "allowed": False,
            "offline_environment": offline_environment,
            "attempts": network_attempts,
        },
        "model_mode": "eval",
        "requires_grad": False,
        "feature_dimension": FEATURE_DIMENSION,
    }
    return model, audit


@contextlib.contextmanager
def _loaded_model(
    *,
    state: Mapping[str, Any],
    runtime: ReplayRuntime,
):
    model, audit = _construct_model(state=state, runtime=runtime)
    try:
        yield model, audit
    finally:
        del model
        if runtime.device.type == "cuda":
            runtime.torch.cuda.empty_cache()


def _audit_official_golden(
    *,
    model: Any,
    runtime: ReplayRuntime,
    source_root: Path,
) -> dict[str, Any]:
    """Independently replay the five official notebook examples."""

    tensors: list[Any] = []
    cases: list[dict[str, Any]] = []
    for filename, digest, expected in FROZEN_GOLDEN_CASES:
        path = source_root / "test_images" / filename
        _verify_hash(path, digest, f"official golden image {filename}")
        prepared = preprocess_image(path, torch_module=runtime.torch)
        tensors.append(prepared.tensor)
        cases.append(
            {
                "filename": filename,
                "sha256": digest,
                "probability": expected,
                "path": str(path.resolve()),
                "tensor_sha256": prepared.tensor_sha256,
            }
        )
    batch = runtime.torch.stack(tensors).to(
        device=runtime.device,
        dtype=runtime.torch.float32,
    )
    with runtime.torch.inference_mode():
        logits = model(batch)
        probabilities = runtime.torch.sigmoid(logits)
    _require_equal(
        tuple(logits.shape),
        (len(cases), 1),
        "golden logit shape",
    )
    _require_equal(
        logits.dtype,
        runtime.torch.float32,
        "golden logit dtype",
    )
    for case, actual in zip(
        cases,
        probabilities.detach().cpu().reshape(-1).tolist(),
        strict=True,
    ):
        expected = float(case["probability"])
        difference = abs(float(actual) - expected)
        case.update(
            {
                "expected_probability": expected,
                "actual_probability": float(actual),
                "absolute_difference": difference,
                "passed": difference <= GOLDEN_ABSOLUTE_TOLERANCE,
            }
        )
        if difference > GOLDEN_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"official golden mismatch {case['filename']}: "
                f"{actual} != {expected}"
            )
    return {
        "status": "passed",
        "source": (
            "official eval_using_huggingface.ipynb five DALL-E 2 images; "
            "full-precision frozen values independently reproduced before "
            "Mouse scoring"
        ),
        "notebook_sha256": FROZEN_MAIN_SOURCE_FILES[
            "eval_using_huggingface.ipynb"
        ],
        "batch_size": len(cases),
        "dtype": "float32",
        "score": "torch.sigmoid(raw_logit)",
        "absolute_tolerance": GOLDEN_ABSOLUTE_TOLERANCE,
        "cases": cases,
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
    expected = {
        "eval/opensource/run_community_forensics.py",
        "eval/opensource/community_forensics_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
    }
    _require_equal(set(contract), expected, "adapter filenames")
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


def _select_inputs(
    rows: list[dict[str, Any]],
    *,
    pair_limit: int | None,
    sample_id: str | None,
) -> list[dict[str, Any]]:
    if pair_limit is not None and sample_id is not None:
        raise ValueError("pair-limit and sample-id are mutually exclusive")
    if sample_id is not None:
        _safe_component(sample_id, label="sample-id")
        selected = [
            row for row in rows if str(row.get("sample_id")) == sample_id
        ]
        if len(selected) != 1:
            raise ValueError("sample-id does not select exactly one row")
        return selected
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if isinstance(pair_limit, bool) or pair_limit <= 0:
            raise ValueError("pair-limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    keep = set(pair_ranks)
    selected = [row for row in rows if int(row["pair_rank"]) in keep]
    census: dict[int, list[str]] = defaultdict(list)
    tasks: dict[int, set[str]] = defaultdict(set)
    for row in selected:
        rank = int(row["pair_rank"])
        census[rank].append(str(row["kind"]))
        tasks[rank].add(str(row["task_id"]))
    invalid = {
        rank: kinds
        for rank, kinds in census.items()
        if sorted(kinds) != ["forged", "real"]
    }
    if invalid:
        raise ValueError(f"selected canonical pairs are incomplete: {invalid}")
    invalid_tasks = {
        rank: values for rank, values in tasks.items() if len(values) != 1
    }
    if invalid_tasks:
        raise ValueError(
            f"selected pair ranks have mismatched task IDs: {invalid_tasks}"
        )
    return selected


def _validate_canonical_input(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    sample_id = str(row.get("sample_id"))
    path_value = row.get("canonical_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"canonical {sample_id} has no input path")
    path = _anchored(Path(path_value), repo_root)
    _verify_hash(
        path,
        row.get("canonical_sha256"),
        f"canonical input {sample_id}",
    )
    with Image.open(path) as opened:
        _require_equal(
            list(opened.size),
            [int(row["width"]), int(row["height"])],
            f"canonical input {sample_id} dimensions",
        )
    _load_gt_mask(row, repo_root=repo_root)
    return path


def _audit_canonical_pairs(
    pair_rows: list[dict[str, Any]],
    all_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-check the frozen pairs.jsonl against all 550 input rows."""

    _require_equal(len(pair_rows), 275, "canonical pair-row count")
    by_id: dict[str, dict[str, Any]] = {}
    for row in all_inputs:
        sample_id = _safe_component(
            row.get("sample_id"),
            label="canonical sample_id",
        )
        if sample_id in by_id:
            raise ValueError(f"duplicate canonical sample ID {sample_id}")
        by_id[sample_id] = row
    _require_equal(len(by_id), 550, "canonical unique sample IDs")
    pair_ranks: list[int] = []
    task_ids: list[str] = []
    domains: Counter[str] = Counter()
    paired_sample_ids: list[str] = []
    nested_fields = (
        "sample_id",
        "kind",
        "label",
        "canonical_path",
        "canonical_sha256",
        "canonical_bytes",
        "raw_path",
        "raw_sha256",
    )
    for index, pair in enumerate(pair_rows):
        _require_equal(
            pair.get("schema_version"),
            FROZEN_CANONICAL_RELEASE["schema_version"],
            f"canonical pair {index} schema",
        )
        _require_equal(
            pair.get("dataset_id"),
            FROZEN_CANONICAL_RELEASE["dataset_id"],
            f"canonical pair {index} dataset",
        )
        _require_equal(
            pair.get("candidate"),
            "mouse",
            f"canonical pair {index} candidate",
        )
        pair_rank = int(pair.get("pair_rank", -1))
        _require_equal(pair_rank, index, f"canonical pair {index} rank/order")
        pair_ranks.append(pair_rank)
        task_id = _safe_component(
            pair.get("task_id"),
            label=f"canonical pair {index} task_id",
        )
        task_ids.append(task_id)
        domain = str(pair.get("domain"))
        domains[domain] += 1
        for kind, label in (("real", 0), ("forged", 1)):
            nested = _require_mapping(
                pair.get(kind),
                f"canonical pair {index} {kind}",
            )
            sample_id = _safe_component(
                nested.get("sample_id"),
                label=f"canonical pair {index} {kind} sample_id",
            )
            if sample_id not in by_id:
                raise ValueError(
                    f"canonical pair {index} references unknown {sample_id}"
                )
            canonical = by_id[sample_id]
            paired_sample_ids.append(sample_id)
            for field in nested_fields:
                _require_equal(
                    nested.get(field),
                    canonical.get(field),
                    f"canonical pair {index} {kind}.{field}",
                )
            for field, expected in {
                "kind": kind,
                "label": label,
                "pair_rank": pair_rank,
                "task_id": task_id,
                "domain": domain,
                "dataset_id": FROZEN_CANONICAL_RELEASE["dataset_id"],
                "candidate": "mouse",
                "width": int(pair["width"]),
                "height": int(pair["height"]),
                "edit_region_xyxy": pair["edit_region_xyxy"],
                "context_region_xyxy": pair["context_region_xyxy"],
            }.items():
                _require_equal(
                    canonical.get(field),
                    expected,
                    f"canonical input {sample_id} {field}",
                )
        forged = by_id[str(pair["forged"]["sample_id"])]
        for field in (
            "gt_mask_path",
            "gt_mask_sha256",
            "gt_positive_pixels",
        ):
            _require_equal(
                pair.get(field),
                forged.get(field),
                f"canonical pair {index} {field}",
            )
    _require_equal(pair_ranks, list(range(275)), "canonical pair ranks")
    _require_equal(len(set(task_ids)), 275, "canonical unique task IDs")
    _require_equal(
        domains,
        Counter({"lodging": 147, "restaurant": 128}),
        "canonical pair domain census",
    )
    _require_equal(
        set(paired_sample_ids),
        set(by_id),
        "canonical pair/input sample coverage",
    )
    _require_equal(
        len(paired_sample_ids),
        len(set(paired_sample_ids)),
        "canonical pair sample uniqueness",
    )
    return {
        "pairs": len(pair_rows),
        "images": len(by_id),
        "pair_ranks": [0, 274],
        "unique_task_ids": len(set(task_ids)),
        "unique_sample_ids": len(by_id),
        "domain_census": dict(sorted(domains.items())),
        "pairs_to_inputs_exact": True,
    }


def _validate_result_identity(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
    config_fingerprint: str,
    visibility: Mapping[str, Any],
    pins: SimpleNamespace,
) -> None:
    sample_id = str(canonical["sample_id"])
    expected = {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": str(canonical["task_id"]),
        "pair_rank": int(canonical["pair_rank"]),
        "rank": int(canonical["rank"]),
        "kind": str(canonical["kind"]),
        "label": int(canonical["label"]),
        "domain": str(canonical["domain"]),
        "input_sha256": str(canonical["canonical_sha256"]),
        "input_width": int(canonical["width"]),
        "input_height": int(canonical["height"]),
        "model": pins.MODEL_NAME,
        "model_slug": pins.MODEL_SLUG,
        "preprocess_profile": FROZEN_PROFILE,
        "score_semantics": SCORE_SEMANTICS,
        "classification_threshold": FIXED_THRESHOLD,
        "classification_threshold_operator": THRESHOLD_OPERATOR,
        "config_fingerprint": config_fingerprint,
        "edit_visibility": visibility["edit_visibility"],
        "edit_visible_gt_fraction": visibility[
            "edit_visible_gt_fraction"
        ],
        "edit_visibility_evidence": visibility[
            "edit_visibility_evidence"
        ],
    }
    for key, value in expected.items():
        _compare_nested(
            row.get(key),
            value,
            label=f"row {sample_id} {key}",
            exact_mapping_keys=True,
        )
    value = row.get("input_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {sample_id} has no input_path")
    _require_equal(
        _anchored(Path(value), repo_root),
        _anchored(Path(str(canonical["canonical_path"])), repo_root),
        f"row {sample_id} input path",
    )


def _validate_result_payload(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_dir: Path,
    torch_module: ModuleType,
) -> None:
    _reject_t2_localization_or_joint(row, label=f"row {row.get('id')}")
    status = row.get("status")
    if status not in {"ok", "error"}:
        raise ValueError(f"row {row.get('id')} has invalid status {status!r}")
    for key in ("completed_at",):
        if not isinstance(row.get(key), str) or not row.get(key):
            raise ValueError(f"row {row.get('id')} has invalid {key}")
    if status == "error":
        _require_equal(
            row.get("valid_for_metrics"),
            False,
            f"error row {row.get('id')} metric validity",
        )
        for key in (
            "raw_logit",
            "probability",
            "ai_score",
            "score",
            "classification_decision",
        ):
            if key not in row:
                raise ValueError(f"error row {row.get('id')} lacks {key}")
            _require_equal(
                row.get(key),
                None,
                f"error row {row.get('id')} {key}",
            )
        for key in ("error_type", "error", "traceback"):
            if not isinstance(row.get(key), str) or not row.get(key):
                raise ValueError(f"row {row.get('id')} has invalid {key}")
        forbidden = {
            "preprocess",
            "commfor_feature_path",
            "commfor_feature_sha256",
            "commfor_feature_array_sha256",
            "feature_array_sha256",
            "artifact_paths",
            "classification",
            "t1",
            "manual_replay",
        }.intersection(row)
        if forbidden:
            raise ValueError(
                f"error row claims success fields: {sorted(forbidden)}"
            )
        return
    _require_equal(
        row.get("valid_for_metrics"),
        True,
        f"row {row.get('id')} metric validity",
    )
    raw = _require_finite(
        row.get("raw_logit"),
        f"row {row.get('id')} raw logit",
    )
    _audit_score_fields(
        row,
        replay_raw_logit=raw,
        replay_probability=_float32_sigmoid(raw, torch_module),
        raw_tolerance=0.0,
    )
    feature, _ = _load_feature(
        row,
        repo_root=repo_root,
        run_dir=run_dir,
    )
    array_digest = _array_sha256(feature)
    raw_array_digest = _require_sha256(
        row.get("commfor_feature_array_sha256"),
        f"row {row.get('id')} feature array SHA-256",
    )
    _require_equal(
        raw_array_digest,
        array_digest,
        f"row {row.get('id')} feature array SHA-256",
    )
    _require_equal(
        row.get("feature_array_sha256"),
        array_digest,
        f"row {row.get('id')} feature array SHA-256 alias",
    )
    _require_equal(
        row.get("artifact_paths"),
        {"commfor_feature_npy": row.get("commfor_feature_path")},
        f"row {row.get('id')} artifact paths",
    )
    for key in ("preprocess_latency_ms", "latency_ms"):
        latency = _require_finite(
            row.get(key),
            f"row {row.get('id')} {key}",
        )
        if latency < 0.0:
            raise ValueError(f"row {row.get('id')} {key} is negative")
    peak = row.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError(f"row {row.get('id')} peak CUDA memory is invalid")


def _load_run_files(
    *,
    results_dir: Path,
    run_id: str,
) -> RunFiles:
    _safe_component(run_id, label="run-id")
    root = results_dir.resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("run-id escapes the results directory")
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
    model_root: Path,
    processor_root: Path,
    inputs_path: Path,
    all_inputs: list[dict[str, Any]],
    files: RunFiles,
) -> dict[str, Any]:
    """Audit source, weights, runtime, selection, rows, and physical hashes."""

    pins = _load_runner_pins()
    for actual, expected, label in (
        (pins.MODEL_ARCH, FROZEN_TIMM_MODEL, "runner architecture"),
        (pins.PREPROCESS_PROFILE, FROZEN_PROFILE, "runner preprocess profile"),
        (pins.FEATURE_DIMENSION, FEATURE_DIMENSION, "runner feature dimension"),
        (pins.FEATURE_SEMANTICS, FEATURE_SEMANTICS, "runner feature semantics"),
        (pins.SCORE_SEMANTICS, SCORE_SEMANTICS, "runner score semantics"),
        (pins.T1_POLICY, T1_POLICY, "runner T1 policy"),
        (tuple(pins.IMAGE_MEAN), IMAGE_MEAN, "runner ImageNet mean"),
        (tuple(pins.IMAGE_STD), IMAGE_STD, "runner ImageNet std"),
        (
            pins.CLASSIFICATION_THRESHOLD,
            FIXED_THRESHOLD,
            "runner threshold",
        ),
        (
            pins.CLASSIFICATION_THRESHOLD_OPERATOR,
            THRESHOLD_OPERATOR,
            "runner threshold operator",
        ),
        (
            pins.CANONICAL_RELEASE,
            FROZEN_CANONICAL_RELEASE,
            "runner canonical release pin",
        ),
        (
            pins.GOLDEN_ABS_TOLERANCE,
            GOLDEN_ABSOLUTE_TOLERANCE,
            "runner golden tolerance",
        ),
    ):
        _require_equal(actual, expected, label)
    runner_golden = tuple(
        (
            str(case["filename"]),
            str(case["sha256"]),
            float(case["probability"]),
        )
        for case in pins.GOLDEN_CASES
    )
    _require_equal(
        runner_golden,
        FROZEN_GOLDEN_CASES,
        "runner golden cases",
    )

    manifest = files.manifest
    _reject_t2_localization_or_joint(manifest, label="manifest")
    _reject_t2_localization_or_joint(files.summary, label="summary")
    required_manifest_keys = {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "resumed_at",
        "completed_at",
        "repo_root",
        "config_fingerprint",
        "config",
        "source",
        "assets",
        "model_audit",
        "official_golden",
        "runtime",
        "dataset",
        "visibility_census",
        "full_dataset_visibility_audit",
        "outputs",
        "execution",
    }
    _require_equal(set(manifest), required_manifest_keys, "manifest keys")
    _require_equal(
        manifest.get("schema_version"),
        "community_forensics_detection_run_manifest_v1",
        "manifest schema",
    )
    _require_equal(manifest.get("run_id"), files.run_dir.name, "manifest run ID")
    _require_equal(
        Path(str(manifest.get("repo_root"))).resolve(),
        repo_root.resolve(),
        "manifest repository root",
    )
    config = _require_mapping(manifest.get("config"), "manifest config")
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
        "model_arch": FROZEN_TIMM_MODEL,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "eval_single_commit": FROZEN_EVAL_SINGLE_COMMIT,
        "source_files": FROZEN_MAIN_SOURCE_FILES,
        "checkpoint_id": pins.CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_MODEL_REVISION,
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "processor_revision": FROZEN_PROCESSOR_REVISION,
        "processor_files": FROZEN_PROCESSOR_FILES,
        "preprocess_profile": FROZEN_PROFILE,
        "license": pins.LICENSE_RECORD,
        "checkpoint_selection_frozen_before_scores": True,
    }.items():
        _compare_nested(config.get(key), expected, label=f"config {key}")
    _require_equal(
        config.get("preprocess_contract"),
        {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "resize": {
                "short_side": RESIZE_SHORT_SIDE,
                "aspect_preserving": True,
                "interpolation": "PIL_BILINEAR",
                "antialias": True,
            },
            "crop": {
                "kind": "torchvision_CenterCrop",
                "size": CROP_SIZE,
            },
            "to_tensor": "uint8_div_255_to_float32",
            "normalization_mean": list(IMAGE_MEAN),
            "normalization_std": list(IMAGE_STD),
            "batch_size": 1,
        },
        "frozen preprocess contract",
    )
    _require_equal(
        config.get("model_contract"),
        {
            "construction": (
                "timm_1.0.15_create_model_pretrained_false_then_replace_head"
            ),
            "strict_full_safetensors_load": True,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_semantics": FEATURE_SEMANTICS,
            "output": "one_raw_logit",
            "model_mode": "eval",
            "score": SCORE_SEMANTICS,
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "t1_policy": T1_POLICY,
            "score_direction": "higher_means_fake",
            "valid_for_t2": False,
            "attention_or_features_are_localization": False,
        },
        "frozen model contract",
    )
    metrics = _require_mapping(config.get("metrics"), "config metrics")
    _require_equal(metrics.get("fixed_threshold"), FIXED_THRESHOLD, "metric threshold")
    _require_equal(
        metrics.get("threshold_operator"),
        THRESHOLD_OPERATOR,
        "metric threshold operator",
    )
    _require_equal(metrics.get("pair_bootstrap"), True, "metric pair bootstrap")
    _require_equal(
        metrics.get("bootstrap_unit"),
        "task_id_pair",
        "metric bootstrap unit",
    )
    iterations = metrics.get("bootstrap_samples")
    seed = metrics.get("bootstrap_seed")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("bootstrap sample count is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed is invalid")
    adapter = _validate_adapter_contract(config, repo_root=repo_root)
    source_expected = _verify_source_tree(source_root, pins=pins)
    _require_equal(manifest.get("source"), source_expected, "manifest source")
    runtime = _replay_runtime(manifest)
    state, assets_expected = _verify_assets(
        model_root=model_root,
        processor_root=processor_root,
        pins=pins,
        torch_module=runtime.torch,
    )
    _require_equal(manifest.get("assets"), assets_expected, "manifest assets")
    schema = assets_expected["checkpoint"]["schema"]
    _require_equal(
        config.get("checkpoint_schema_sha256"),
        schema["items_sha256"],
        "config checkpoint schema SHA-256",
    )
    model, model_audit = _construct_model(state=state, runtime=runtime)
    _require_equal(manifest.get("model_audit"), model_audit, "manifest model audit")
    _require_equal(
        config.get("model_construction_audit"),
        model_audit,
        "config model construction audit",
    )
    _require_equal(
        config.get("model_construction_audit_fingerprint"),
        _fingerprint(model_audit),
        "model construction audit fingerprint",
    )
    golden = _audit_official_golden(
        model=model,
        runtime=runtime,
        source_root=source_root,
    )
    _compare_nested(
        manifest.get("official_golden"),
        golden,
        label="manifest official golden",
        float_tolerance=GOLDEN_ABSOLUTE_TOLERANCE,
        exact_mapping_keys=True,
    )
    recorded_golden = _require_mapping(
        manifest.get("official_golden"),
        "manifest official golden",
    )
    _require_equal(
        recorded_golden.get("absolute_tolerance"),
        GOLDEN_ABSOLUTE_TOLERANCE,
        "manifest golden absolute tolerance",
    )
    _require_equal(
        config.get("official_golden"),
        manifest.get("official_golden"),
        "config/manifest official golden",
    )
    _require_equal(
        config.get("official_golden_fingerprint"),
        _fingerprint(_require_mapping(manifest["official_golden"], "golden")),
        "official golden fingerprint",
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
        _anchored(Path(str(dataset["inputs_path"])), repo_root),
        inputs_path.resolve(),
        "manifest canonical inputs path",
    )
    _verify_hash(inputs_path, dataset.get("inputs_sha256"), "canonical inputs")
    release_path = _anchored(Path(str(dataset["manifest_path"])), repo_root)
    _verify_hash(
        release_path,
        dataset.get("manifest_sha256"),
        "canonical dataset manifest",
    )
    release = _require_mapping(
        json.loads(release_path.read_text(encoding="utf-8")),
        "canonical release",
    )
    for key, expected in FROZEN_CANONICAL_RELEASE.items():
        _require_equal(
            release.get(key),
            expected,
            f"canonical release {key}",
        )
    _require_equal(
        len(all_inputs),
        FROZEN_CANONICAL_RELEASE["images"],
        "canonical image count",
    )
    _require_equal(
        release.get("inputs_sha256"),
        sha256_file(inputs_path),
        "release canonical input SHA-256",
    )
    pairs_value = release.get("pairs_path")
    if not isinstance(pairs_value, str) or not pairs_value:
        raise ValueError("canonical release has no pairs_path")
    pairs_path = _anchored(Path(pairs_value), repo_root)
    _verify_hash(
        pairs_path,
        FROZEN_CANONICAL_RELEASE["pairs_sha256"],
        "canonical pairs JSONL",
    )
    pair_rows = read_jsonl(pairs_path)
    pairs_audit = _audit_canonical_pairs(pair_rows, all_inputs)
    ranks = [int(row["rank"]) for row in all_inputs]
    if (
        ranks != list(range(FROZEN_CANONICAL_RELEASE["images"]))
        or len(ranks) != len(set(ranks))
    ):
        raise ValueError("canonical inputs are not in unique rank order")
    for canonical in all_inputs:
        _validate_canonical_input(canonical, repo_root=repo_root)
    config_dataset = _require_mapping(config.get("dataset"), "config dataset")
    for key, expected in {
        "schema_version": FROZEN_CANONICAL_RELEASE["schema_version"],
        "dataset_id": FROZEN_CANONICAL_RELEASE["dataset_id"],
        "inputs_sha256": FROZEN_CANONICAL_RELEASE["inputs_sha256"],
    }.items():
        _require_equal(
            config_dataset.get(key),
            expected,
            f"config dataset {key}",
        )
    pair_limit = config_dataset.get("pair_limit")
    sample_id = config_dataset.get("sample_id")
    selected = _select_inputs(
        all_inputs,
        pair_limit=pair_limit,
        sample_id=sample_id,
    )
    _require_equal(files.expected, selected, "expected-input selection replay")
    _require_equal(
        config_dataset.get("selected_ids"),
        [str(row["sample_id"]) for row in selected],
        "selected IDs",
    )
    _require_equal(
        config_dataset.get("selected_rows_sha256"),
        _selected_rows_sha256(selected),
        "selected rows SHA-256",
    )
    expected_on_disk = _anchored(
        Path(str(dataset["expected_inputs_path"])),
        repo_root,
    )
    _require_equal(
        expected_on_disk,
        files.expected_path.resolve(),
        "expected-input snapshot path",
    )
    _verify_hash(
        files.expected_path,
        dataset.get("expected_inputs_sha256"),
        "expected-input snapshot",
    )
    _require_equal(
        dataset.get("selected_images"),
        len(selected),
        "selected image count",
    )
    _require_equal(
        dataset.get("selected_tasks"),
        len({str(row["task_id"]) for row in selected}),
        "selected task count",
    )

    visibility = _pair_visibility(all_inputs, repo_root=repo_root)
    full_fractions = [
        float(value["edit_visible_gt_fraction"])
        for value in visibility.values()
    ]
    full_census = dict(
        Counter(
            str(value["edit_visibility"])
            for value in visibility.values()
        )
    )
    _require_equal(len(visibility), 275, "full visibility pair count")
    _require_equal(
        full_census,
        {"full": 162, "partial": 32, "none": 81},
        "full visibility census",
    )
    full_mean = float(np.mean(full_fractions))
    _require_equal(
        round(full_mean, 6),
        0.646589,
        "full mean edit visibility",
    )
    _require_equal(
        manifest.get("full_dataset_visibility_audit"),
        {
            "pairs": 275,
            "census": {"full": 162, "partial": 32, "none": 81},
            "mean_edit_visible_gt_fraction": full_mean,
            "rounded_mean_6_decimals": 0.646589,
            "basis": (
                "exact_diff_positive_pixel_centers_after_official_resize_crop"
            ),
        },
        "manifest full visibility audit",
    )
    task_ids = sorted({str(row["task_id"]) for row in selected})
    census = dict(
        sorted(
            Counter(
                visibility[task_id]["edit_visibility"]
                for task_id in task_ids
            ).items()
        )
    )
    _require_equal(
        manifest.get("visibility_census"),
        census,
        "manifest visibility census",
    )
    latest = _latest_by_id(files.rows)
    _require_equal(
        set(latest),
        {str(row["sample_id"]) for row in selected},
        "latest result IDs",
    )
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    for physical_index, row in enumerate(files.rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in selected_by_id:
            raise ValueError(
                f"physical result row {physical_index} has unexpected id "
                f"{row_id!r}"
            )
        canonical = selected_by_id[row_id]
        _validate_result_identity(
            row,
            canonical,
            repo_root=repo_root,
            config_fingerprint=config_fingerprint,
            visibility=visibility[str(canonical["task_id"])],
            pins=pins,
        )
        _validate_result_payload(
            row,
            repo_root=repo_root,
            run_dir=files.run_dir,
            torch_module=runtime.torch,
        )

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    for key, path in {
        "results_path": files.results_path,
        "summary_path": files.summary_path,
        "feature_dir": files.run_dir / "features",
    }.items():
        _require_equal(
            _anchored(Path(str(outputs[key])), repo_root),
            path.resolve(),
            f"manifest {key}",
        )
    _verify_hash(
        files.results_path,
        outputs.get("results_sha256"),
        "physical result JSONL",
    )
    _verify_hash(
        files.summary_path,
        outputs.get("summary_sha256"),
        "physical summary JSON",
    )
    _require_equal(
        outputs.get("feature_files"),
        len(list((files.run_dir / "features").glob("*.npy"))),
        "feature file census",
    )
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest execution",
    )
    _require_equal(
        execution.get("physical_result_rows"),
        len(files.rows),
        "physical result row count",
    )
    return {
        "source": source_expected,
        "assets": assets_expected,
        "adapter": adapter,
        "runtime": runtime.evidence,
        "model": model_audit,
        "official_golden": golden,
        "dataset": {
            "canonical_inputs_sha256": sha256_file(inputs_path),
            "canonical_pairs_sha256": sha256_file(pairs_path),
            "canonical_contract_sha256": FROZEN_CANONICAL_RELEASE[
                "contract_sha256"
            ],
            "expected_inputs_sha256": sha256_file(files.expected_path),
            "selected_images": len(selected),
            "selected_tasks": len(task_ids),
            "selection_replayed": True,
            "pairs_audit": pairs_audit,
        },
        "physical_files": {
            "manifest_sha256": sha256_file(files.manifest_path),
            "results_sha256": sha256_file(files.results_path),
            "summary_sha256": sha256_file(files.summary_path),
            "expected_inputs_sha256": sha256_file(files.expected_path),
        },
        "visibility_census": census,
        "_pins": pins,
        "_runtime_object": runtime,
        "_state": state,
        "_model": model,
    }


def _classifier(model: Any) -> Any:
    if hasattr(model, "vit") and hasattr(model.vit, "head"):
        return model.vit.head
    if hasattr(model, "head"):
        return model.head
    raise ValueError("replay model has no auditable linear classifier head")


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


def audit_artifacts(
    *,
    repo_root: Path,
    source_root: Path,
    all_inputs: list[dict[str, Any]],
    files: RunFiles,
    runtime: ReplayRuntime,
    model: Any | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Redecode and run every image through a fresh complete model."""

    if model is None and state is None:
        raise ValueError("artifact replay needs an independent model or state")
    model_context = (
        contextlib.nullcontext((model, None))
        if model is not None
        else _loaded_model(state=state or {}, runtime=runtime)
    )
    latest = _latest_by_id(files.rows)
    visibility = _pair_visibility(all_inputs, repo_root=repo_root)
    feature_differences: list[float] = []
    raw_differences: list[float] = []
    probability_differences: list[float] = []
    replay_raw_logits: list[float] = []
    replay_probabilities: list[float] = []
    replay_ids: list[str] = []
    independent_latest: dict[str, dict[str, Any]] = {}
    decoded_hashes: list[str] = []
    resized_hashes: list[str] = []
    crop_hashes: list[str] = []
    tensor_hashes: list[str] = []
    replayed = 0
    with model_context as (active_model, _model_audit):
        classifier = _classifier(active_model)
        _require_equal(
            classifier.in_features,
            FEATURE_DIMENSION,
            "classifier input dimension",
        )
        _require_equal(classifier.out_features, 1, "classifier output dimension")
        for canonical in files.expected:
            sample_id = str(canonical["sample_id"])
            if sample_id not in latest:
                raise ValueError(f"missing latest result row {sample_id}")
            row = latest[sample_id]
            _require_equal(row.get("status"), "ok", f"row {sample_id} status")
            path = _anchored(
                Path(str(canonical["canonical_path"])),
                repo_root,
            )
            prepared = preprocess_image(
                path,
                torch_module=runtime.torch,
            )
            _audit_preprocess_record(row, prepared)
            decoded_hashes.append(prepared.decoded_rgb_sha256)
            resized_hashes.append(prepared.resized_rgb_sha256)
            crop_hashes.append(prepared.crop_rgb_sha256)
            tensor_hashes.append(prepared.tensor_sha256)
            persisted, _ = _load_feature(
                row,
                repo_root=repo_root,
                run_dir=files.run_dir,
            )
            captured: list[Any] = []

            def capture_head(
                _module: Any,
                arguments: tuple[Any, ...],
            ) -> None:
                if len(arguments) != 1:
                    raise RuntimeError(
                        "classifier pre-hook received invalid arguments"
                    )
                captured.append(arguments[0].detach().clone())

            hook = classifier.register_forward_pre_hook(capture_head)
            try:
                tensor = prepared.tensor.unsqueeze(0).to(
                    device=runtime.device,
                    dtype=runtime.torch.float32,
                    non_blocking=False,
                )
                with runtime.torch.inference_mode():
                    output = active_model(tensor)
            finally:
                hook.remove()
            _require_equal(
                tuple(output.shape),
                (1, 1),
                f"row {sample_id} full-model output shape",
            )
            _require_equal(
                output.dtype,
                runtime.torch.float32,
                f"row {sample_id} full-model output dtype",
            )
            _require_equal(
                len(captured),
                1,
                f"row {sample_id} classifier hook calls",
            )
            feature_tensor = captured[0]
            _require_equal(
                tuple(feature_tensor.shape),
                (1, FEATURE_DIMENSION),
                f"row {sample_id} full-model feature shape",
            )
            _require_equal(
                feature_tensor.dtype,
                runtime.torch.float32,
                f"row {sample_id} full-model feature dtype",
            )
            recomputed = np.ascontiguousarray(
                feature_tensor.squeeze(0).detach().cpu().numpy(),
                dtype=np.float32,
            )
            if not np.isfinite(recomputed).all():
                raise ValueError(f"row {sample_id} replayed feature is non-finite")
            feature_difference = float(
                np.max(np.abs(recomputed - persisted))
            )
            feature_differences.append(feature_difference)
            if not np.array_equal(recomputed, persisted):
                raise ValueError(
                    f"row {sample_id} persisted feature differs from complete "
                    f"model replay (max abs {feature_difference})"
                )

            artifact_tensor = runtime.torch.from_numpy(
                persisted.copy()
            ).to(
                device=runtime.device,
                dtype=runtime.torch.float32,
            )[None, :]
            with runtime.torch.inference_mode():
                artifact_output = runtime.torch.nn.functional.linear(
                    artifact_tensor,
                    classifier.weight,
                    classifier.bias,
                )
                full_probability = runtime.torch.sigmoid(output)
                artifact_probability = runtime.torch.sigmoid(artifact_output)
            if not runtime.torch.equal(output, artifact_output):
                maximum = float(
                    runtime.torch.max(
                        runtime.torch.abs(output - artifact_output)
                    ).item()
                )
                raise ValueError(
                    f"row {sample_id} persisted-feature F.linear replay "
                    f"differs from full output (max abs {maximum})"
                )
            if not runtime.torch.equal(
                full_probability,
                artifact_probability,
            ):
                raise ValueError(
                    f"row {sample_id} persisted-feature sigmoid replay differs"
                )
            raw = float(artifact_output.reshape(()).item())
            probability = float(artifact_probability.reshape(()).item())
            audited = _audit_score_fields(
                row,
                replay_raw_logit=raw,
                replay_probability=probability,
            )
            raw_differences.append(abs(float(row["raw_logit"]) - raw))
            probability_differences.append(
                abs(float(row["probability"]) - probability)
            )
            replay_raw_logits.append(raw)
            replay_probabilities.append(probability)
            replay_ids.append(sample_id)
            expected_visibility = visibility[str(canonical["task_id"])]
            _require_equal(
                row.get("edit_visibility"),
                expected_visibility["edit_visibility"],
                f"row {sample_id} replayed edit visibility",
            )
            _compare_float(
                row.get("edit_visible_gt_fraction"),
                float(expected_visibility["edit_visible_gt_fraction"]),
                label=f"row {sample_id} replayed visible GT fraction",
                tolerance=0.0,
            )
            _require_equal(
                row.get("edit_visibility_evidence"),
                expected_visibility["edit_visibility_evidence"],
                f"row {sample_id} replayed visibility evidence",
            )
            _require_equal(
                audited["decision"],
                probability > FIXED_THRESHOLD,
                f"row {sample_id} strict decision",
            )
            independent = copy.deepcopy(dict(row))
            independent.update(
                {
                    "raw_logit": raw,
                    "probability": probability,
                    "ai_score": probability,
                    "score": probability,
                    "classification_decision": (
                        probability > FIXED_THRESHOLD
                    ),
                }
            )
            for key in ("classification", "t1"):
                nested = _require_mapping(
                    independent.get(key),
                    f"row {sample_id} independent {key}",
                )
                nested.update(
                    {
                        "raw_logit": raw,
                        "probability": probability,
                        "ai_score": probability,
                        "score": probability,
                        "decision": probability > FIXED_THRESHOLD,
                    }
                )
            manual = _require_mapping(
                independent.get("manual_replay"),
                f"row {sample_id} independent manual replay",
            )
            manual.update(
                {
                    "raw_logit": raw,
                    "probability": probability,
                    "ai_score": probability,
                    "classification_decision": (
                        probability > FIXED_THRESHOLD
                    ),
                }
            )
            independent_latest[sample_id] = independent
            replayed += 1

    raw_array = np.asarray(replay_raw_logits, dtype=np.float64)
    probability_array = np.asarray(replay_probabilities, dtype=np.float64)
    if not np.isfinite(raw_array).all():
        raise ValueError("independently replayed raw logits are non-finite")
    if not np.isfinite(probability_array).all():
        raise ValueError("independently replayed probabilities are non-finite")
    disagreements = (probability_array > FIXED_THRESHOLD) != (raw_array > 0.0)
    disagreement_ids = [
        sample_id
        for sample_id, differs in zip(
            replay_ids,
            disagreements.tolist(),
            strict=True,
        )
        if differs
    ]
    exact_zero = int(np.count_nonzero(probability_array == 0.0))
    exact_one = int(np.count_nonzero(probability_array == 1.0))
    independent_physical_rows = copy.deepcopy(files.rows)
    latest_physical_index: dict[str, int] = {}
    for index, row in enumerate(independent_physical_rows):
        latest_physical_index[str(row["id"])] = index
    for sample_id, independent in independent_latest.items():
        independent_physical_rows[
            latest_physical_index[sample_id]
        ] = independent
    return {
        "images_redecoded": replayed,
        "images_preprocessed_independently": replayed,
        "complete_model_forward_passes": replayed,
        "classifier_features_captured": replayed,
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
        "persisted_feature_manual_f_linear_replayed": True,
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
            "non_saturated_images": replayed - exact_zero - exact_one,
            "unique_probability_values": int(
                np.unique(probability_array).size
            ),
        },
        "preprocess_profile": FROZEN_PROFILE,
        "decoded_rgb_hashes_replayed": len(decoded_hashes),
        "resized_rgb_hashes_replayed": len(resized_hashes),
        "crop_rgb_hashes_replayed": len(crop_hashes),
        "normalized_tensor_hashes_replayed": len(tensor_hashes),
        "runtime": runtime.evidence,
        "physically_independent_replay": {
            "runner_scores_trusted_as_input": False,
            "runner_features_trusted_as_model_input": False,
            "fresh_pillow_decode_per_image": True,
            "independent_preprocess_implementation": True,
            "fresh_complete_model_forward_per_image": True,
            "classifier_input_captured_from_fresh_forward": True,
            "persisted_feature_loaded_only_after_fresh_preprocess": True,
            "persisted_feature_manual_head_replay": (
                "torch.nn.functional.linear_then_float32_sigmoid"
            ),
            "all_selected_images_replayed": replayed == len(files.expected),
            "evidence": (
                "each canonical path was reopened; decoded/resized/cropped/"
                "normalized hashes were regenerated; the complete pinned ViT "
                "was forwarded; its 384d head input was compared byte-for-byte "
                "with a run-directory NPY; that NPY was then independently "
                "passed through F.linear and float32 sigmoid"
            ),
        },
        "_independent_result_rows": independent_physical_rows,
    }


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
    iterations = metrics.get(
        "bootstrap_samples",
        DEFAULT_BOOTSTRAP_SAMPLES,
    )
    seed = metrics.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("bootstrap sample count is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed is invalid")
    recomputed = summarize_community_forensics_results(
        result_rows,
        expected_rows,
        threshold=float(threshold),
        bootstrap_samples=iterations,
        seed=seed,
    )
    metadata_keys = {
        "run_id",
        "model",
        "model_slug",
        "checkpoint_id",
        "preprocess_profile",
        "config_fingerprint",
        "official_golden_status",
        "official_golden_fingerprint",
        "generated_at",
    }
    _require_equal(
        set(recorded_summary),
        set(recomputed) | metadata_keys,
        "summary keys",
    )
    _compare_summary_payload(recorded_summary, recomputed)
    _require_equal(
        recorded_summary.get("run_id"),
        manifest.get("run_id"),
        "summary run ID",
    )
    _require_equal(
        recorded_summary.get("config_fingerprint"),
        manifest.get("config_fingerprint"),
        "summary config fingerprint",
    )
    _require_equal(
        recorded_summary.get("preprocess_profile"),
        FROZEN_PROFILE,
        "summary preprocess profile",
    )
    golden = _require_mapping(
        manifest.get("official_golden"),
        "manifest official golden",
    )
    _require_equal(
        recorded_summary.get("official_golden_status"),
        golden.get("status"),
        "summary golden status",
    )
    _require_equal(
        recorded_summary.get("official_golden_fingerprint"),
        _fingerprint(golden),
        "summary golden fingerprint",
    )
    independent_match = False
    if independent_result_rows is not None:
        independent = summarize_community_forensics_results(
            independent_result_rows,
            expected_rows,
            threshold=float(threshold),
            bootstrap_samples=iterations,
            seed=seed,
        )
        _compare_nested(
            independent,
            recomputed,
            label="summary independent complete-model replay",
            float_tolerance=1e-12,
            exact_mapping_keys=True,
        )
        independent_match = True
    return {
        **recomputed,
        "independent_complete_model_summary_match": independent_match,
    }


def _config_without_selection(
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    """Require a physically separate, exact deterministic smoke prefix."""

    if full.run_dir.resolve() == prefix.run_dir.resolve():
        raise ValueError("prefix and full runs must have independent directories")
    full_config = _require_mapping(full.manifest.get("config"), "full config")
    prefix_config = _require_mapping(prefix.manifest.get("config"), "prefix config")
    _require_equal(
        _config_without_selection(prefix_config),
        _config_without_selection(full_config),
        "prefix/full non-selection config",
    )
    for key in (
        "source",
        "assets",
        "model_audit",
        "official_golden",
        "runtime",
        "full_dataset_visibility_audit",
    ):
        _require_equal(
            prefix.manifest.get(key),
            full.manifest.get(key),
            f"prefix/full {key}",
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
        _require_equal(
            prefix_row.get("status"),
            "ok",
            f"prefix {sample_id} status",
        )
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
        if full_path.resolve() == prefix_path.resolve():
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
            "independent run directories and NPY files; exact ordered prefix; "
            "byte-identical float32 ViT head inputs, decoded/resized/crop/"
            "tensor hashes, logits, sigmoid probabilities, aliases, and "
            "strict decisions"
        ),
        "full_run_id": full.run_dir.name,
        "prefix_run_id": prefix.run_dir.name,
        "prefix_images": len(prefix_ids),
        "samples_compared": len(prefix_ids),
        "full_results_sha256": sha256_file(full.results_path),
        "prefix_results_sha256": sha256_file(prefix.results_path),
        "full_manifest_sha256": sha256_file(full.manifest_path),
        "prefix_manifest_sha256": sha256_file(prefix.manifest_path),
        "independent_directories": True,
        "independent_feature_paths": True,
        "copied_full_artifacts_rejected": True,
    }


def _strip_private(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: nested
        for key, nested in value.items()
        if not str(key).startswith("_")
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    source_root = _anchored(Path(args.source_root), repo_root)
    model_root = _anchored(Path(args.model_root), repo_root)
    processor_root = _anchored(Path(args.processor_root), repo_root)
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    all_inputs = read_jsonl(inputs_path)
    files = _load_run_files(results_dir=results_dir, run_id=args.run_id)
    provenance_internal = validate_provenance(
        repo_root=repo_root,
        source_root=source_root,
        model_root=model_root,
        processor_root=processor_root,
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
        model=provenance_internal["_model"],
    )
    artifacts = _strip_private(artifacts_internal)
    recomputed = recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=files.manifest,
        recorded_summary=files.summary,
        independent_result_rows=artifacts_internal[
            "_independent_result_rows"
        ],
    )

    prefix_provenance: dict[str, Any] | None = None
    prefix_artifacts: dict[str, Any] | None = None
    prefix_summary: dict[str, Any] | None = None
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
            model_root=model_root,
            processor_root=processor_root,
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
            model=prefix_internal["_model"],
        )
        prefix_provenance = _strip_private(prefix_internal)
        prefix_artifacts = _strip_private(prefix_artifacts_internal)
        prefix_summary = recompute_summary(
            result_rows=prefix.rows,
            expected_rows=prefix.expected,
            manifest=prefix.manifest,
            recorded_summary=prefix.summary,
            independent_result_rows=prefix_artifacts_internal[
                "_independent_result_rows"
            ],
        )
        prefix_audit = audit_prefix_reproducibility(
            repo_root=repo_root,
            full=files,
            prefix=prefix,
        )

    analysis = {
        "schema_version": (
            "community_forensics_independent_analysis_v1"
        ),
        "run_id": args.run_id,
        "generated_at": utc_now(),
        "method": (
            "physical retry/file-hash and immutable source/assets/runtime "
            "audit; independent Pillow RGB, torchvision-equivalent "
            "Resize(440)/CenterCrop(384)/ToTensor/ImageNet normalization; "
            "complete fresh timm ViT-S/16 forward for every image; exact 384d "
            "head-input comparison; persisted-feature F.linear replay; "
            "float32 sigmoid and strict >0.5 replay; summary recomputation"
        ),
        "result_history": summarize_result_history(files.rows),
        "provenance": _strip_private(provenance_internal),
        "artifact_replay": artifacts,
        "recomputed_summary": recomputed,
        "prefix_provenance": prefix_provenance,
        "prefix_artifact_replay": prefix_artifacts,
        "prefix_recomputed_summary": prefix_summary,
        "prefix_reproducibility": prefix_audit,
        "physically_independent_replay": artifacts[
            "physically_independent_replay"
        ],
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
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--processor-root", default=str(DEFAULT_PROCESSOR_ROOT))
    parser.add_argument("--prefix-run-id")
    parser.add_argument("--prefix-results-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
