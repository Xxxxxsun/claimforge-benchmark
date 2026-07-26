#!/usr/bin/env python3
"""Independently audit a frozen official B-Free inference run.

The analyzer treats runner JSON and NPZ artifacts as untrusted evidence.  It
revalidates the pinned source tree, release ZIP, YAML configuration, and
``weights_only=True`` checkpoint schema; independently reopens and preprocesses
every selected image; runs a fresh official five-crop DINOv2 forward; and
compares the five 768-dimensional head inputs, five crop logits, their
float32 mean, and the released strict ``raw_logit > 0`` decision.  The linear
head is also replayed directly from each persisted feature artifact.

B-Free is an image-level detector.  T2, localization, pixel-mask, and joint
claims are rejected recursively.
"""

from __future__ import annotations

import argparse
import contextlib
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
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.bfree_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    bfree_fake_probability_float32,
    summarize_bfree_results,
)
from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)


DEFAULT_RESULTS_DIR = Path("results/opensource/bfree")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_RUN_ID = (
    "bfree_dino2reg4_mouse_canonical_v1_full275_20260725"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/b-free-c6a9f898"
)
DEFAULT_WEIGHTS_DIR = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4"
)
DEFAULT_WEIGHTS_ZIP = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4.zip"
)

FROZEN_SOURCE_COMMIT = "c6a9f898782fb466b29af01f21960b67415afb0e"
FROZEN_ZIP_SHA256 = (
    "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f"
)
FROZEN_ZIP_MD5 = "f3f53fa647848b16cf81c913f148a198"
FROZEN_ZIP_BYTES = 321_653_488
FROZEN_ZIP_MEMBERS = {
    "BFREE_dino2reg4/": 0,
    "BFREE_dino2reg4/config.yaml": 153,
    "BFREE_dino2reg4/model_epoch_best.pth": 346_171_370,
}
FROZEN_CONFIG_SHA256 = (
    "1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40"
)
FROZEN_CONFIG_BYTES = 153
FROZEN_CHECKPOINT_SHA256 = (
    "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
)
FROZEN_CHECKPOINT_BYTES = 346_171_370
FROZEN_CHECKPOINT_TENSORS = 177
FROZEN_CHECKPOINT_ELEMENTS = 86_526_721
FROZEN_MODEL_PARAMETER_COUNT = 86_526_721
FROZEN_CHECKPOINT_SCHEMA_SHA256 = (
    "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524"
)

FROZEN_MODEL_NAME = "BFREE_dino2reg4"
FROZEN_CONFIG_MODEL_NAME = "BFREE_dino2reg4"
FROZEN_CONFIG_ARCH = (
    "timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m"
)
FROZEN_TIMM_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
FROZEN_NORM_TYPE = "resnet"
FROZEN_WEIGHTS_FILE = "model_epoch_best.pth"
FROZEN_CONFIG = {
    "arch": FROZEN_CONFIG_ARCH,
    "model_name": FROZEN_CONFIG_MODEL_NAME,
    "norm_type": FROZEN_NORM_TYPE,
    "patch_size": None,
    "weights_file": FROZEN_WEIGHTS_FILE,
}

FROZEN_SOURCE_FILES = {
    "LICENSE.txt": (
        "cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27"
    ),
    "README.md": (
        "6633483801daf6574c05afe8d2e892d4f84afb338cde467218899c346366c185"
    ),
    "code/LICENSE.txt": (
        "cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27"
    ),
    "code/README.md": (
        "81386f0127828890f1a8a4126470b9bc311592dbb9ba7f601eb1806adb8cde5f"
    ),
    "code/main_bfree.py": (
        "eea848f57d1415c2c52804ff013a611435cdb38371c297c2de67fb459e4079ce"
    ),
    "code/main_bfree_single.py": (
        "72f43b8999d2b36689dfa77728f2cf51000a1333a62aa2aad65973ef881c24f7"
    ),
    "code/networks/__init__.py": (
        "048638ddd724ebbfbe995c3a735284a0551b8ca5fec74ca6a0ac2a5a4e6dd8cf"
    ),
    "code/networks/wrapper5crops.py": (
        "1f4de65b82b33c3864ab368836bab009a18f9bf0f828335777272645f236d60b"
    ),
    "code/utils/normalization.py": (
        "12a244d489f001ee7f25aae9bbe2b8fe1b1172f365b1ac0b8d632257f6c2354b"
    ),
    "code/utils/dmetrics.py": (
        "fbb4a29d4f7d1d8492a28190f011dc4703a4894b8622d86a2f84ab920f836244"
    ),
    "code/requirements.txt": (
        "8fd3131b5fbe8e16cdcbd0b022dee6f1213a216e989f52dd39005ca31393a168"
    ),
    "code/demo_images/metainfo.csv": (
        "b939eee3b5c8a6a2bf41a687e1be0454aaee7a850a4581d06bbad50fd49496d4"
    ),
    "code/demo_images/results.csv": (
        "81a3c434bfd3ec1aceb667fa82b86f3a55c84c390ab8b97029a4e5de0a1958a3"
    ),
    "code/demo_images/results_metrics.csv": (
        "5f565b8d51606670a5044d5c9928ca6f7966bd336d4f56aca04d2b0fdcb916ea"
    ),
    "code/demo_images/img0000.png": (
        "c7351aee67f37fe5acf1aa7781612b2760b90e0d56010038ec2e48ff9a79360e"
    ),
    "code/demo_images/img0001.png": (
        "34f54d4ee77bea6640b87d2665ff5c3871ee848837776b53ab58fc1ec3cddead"
    ),
    "code/demo_images/img0002.png": (
        "0d54da3b1a23b2a9aa235c7cbddfb5100d9a4d019fc6621e2c1126d910eb5f08"
    ),
    "code/demo_images/img0003.png": (
        "a0947013ca31ac169878892fa6c0efa43e22beb1f069a26028d23604d5f931fa"
    ),
}

FROZEN_GOLDEN_CASES = (
    {
        "filename": "img0000.png",
        "sha256": (
            "c7351aee67f37fe5acf1aa7781612b2760b90e0d56010038ec2e48ff9a79360e"
        ),
        "width": 835,
        "height": 1256,
        "official_raw_logit": -5.9374785,
        "decoded_rgb_sha256": (
            "13f331e6926c61747afb70325fa423408a2cff09405b36a5feb6a24b0723e216"
        ),
        "tensor_sha256": (
            "aa250c75b0da43bc9eafacaef948a87094149579337c37249a432ea6bd70412b"
        ),
        "patch_grid_hw": [89, 59],
    },
    {
        "filename": "img0001.png",
        "sha256": (
            "34f54d4ee77bea6640b87d2665ff5c3871ee848837776b53ab58fc1ec3cddead"
        ),
        "width": 1258,
        "height": 833,
        "official_raw_logit": -4.441922,
        "decoded_rgb_sha256": (
            "425f264c07dc97ec044b32a9a2bcb538e8750ec2b584feb178d5e6d29d09925d"
        ),
        "tensor_sha256": (
            "e205b12364b070d29500ba6733e05b0735f2e37049c251ef9a76b890ef8ee91f"
        ),
        "patch_grid_hw": [59, 89],
    },
    {
        "filename": "img0002.png",
        "sha256": (
            "0d54da3b1a23b2a9aa235c7cbddfb5100d9a4d019fc6621e2c1126d910eb5f08"
        ),
        "width": 1024,
        "height": 1024,
        "official_raw_logit": 4.430519,
        "decoded_rgb_sha256": (
            "32a4651c3efbe85d917a325de56cec5dee78151785a962301d7f29645e328491"
        ),
        "tensor_sha256": (
            "b2d25ae0bebc69f0fa1580031dec8f7cd3a5e088bd8449a2c5276c4cc9c332db"
        ),
        "patch_grid_hw": [73, 73],
    },
    {
        "filename": "img0003.png",
        "sha256": (
            "a0947013ca31ac169878892fa6c0efa43e22beb1f069a26028d23604d5f931fa"
        ),
        "width": 1024,
        "height": 1024,
        "official_raw_logit": 3.8499813,
        "decoded_rgb_sha256": (
            "b17c899e7f66a8e9cb383be47889ec153404bb449db25466c26be6b47ea281f1"
        ),
        "tensor_sha256": (
            "223319e2821751263cc8819cd42f30825cf86b49f75a882be4ed4fbe1ebc7893"
        ),
        "patch_grid_hw": [73, 73],
    },
)
GOLDEN_ABSOLUTE_TOLERANCE = 5e-5

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
FROZEN_VISIBILITY_CENSUS = {
    "pairs": 275,
    "edit_visibility": {"full": 173, "partial": 36, "none": 66},
    "mean_edit_visible_gt_fraction": 0.6891766376903072,
    "wrap_pairs": 26,
    "wrap_edit_visibility": {"full": 17, "partial": 2, "none": 7},
    "distinct_crop_starts": {1: 26, 3: 1, 5: 248},
    "by_domain": {
        "lodging": {"full": 95, "partial": 18, "none": 34},
        "restaurant": {"full": 78, "partial": 18, "none": 32},
    },
}

PATCH_SIZE = 14
CROP_SIZE = 504
CROP_GRID_SIZE = 36
CROPS = 5
FEATURE_DIMENSION = 768
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
FEATURE_DTYPE = np.dtype("float32")
MODEL_SEED = 20260725
RAW_LOGIT_ABSOLUTE_TOLERANCE = 1e-6
FEATURE_ABSOLUTE_TOLERANCE = 0.0

FROZEN_PROFILE = "official_native_rgb_resnet_norm_dinov2_5crop504"
FEATURE_SEMANTICS = "five_dinov2_head_input_vectors_official_crop_order"
CROP_LOGIT_SEMANTICS = "model_head_applied_to_each_of_five_crop_features"
SCORE_SEMANTICS = "official_float32_mean_of_five_crop_raw_logits"
T1_POLICY = "released_mean_raw_logit_strictly_greater_than_0"

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


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent RGB decoding and native-resolution normalization."""

    tensor: Any
    decoded_rgb: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """The independently configured torch runtime."""

    torch: ModuleType
    device: Any
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RunFiles:
    """Physical files and parsed payloads for one run."""

    run_dir: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest_path: Path
    rows: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ForwardEvidence:
    """Independent complete-model and head-replay outputs."""

    features: np.ndarray
    crop_logits: np.ndarray
    raw_logit: float
    wrapper_raw_logit: float


def _anchored(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(value)


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is not a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_component(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f"{label} is not a safe non-empty path component")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    expected_digest = _require_sha256(expected, f"{label} expected SHA-256")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_digest}, got {actual}"
        )
    return actual


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    return _array_sha256(
        tensor.detach().to(device="cpu").contiguous().numpy()
    )


def _decoded_sha256(array: np.ndarray) -> str:
    rgb = np.ascontiguousarray(array, dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(dict(value)).encode("utf-8")).hexdigest()


def _selected_rows_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _module_pin(module: ModuleType, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(
        "B-Free runner lacks required pin " + " or ".join(names)
    )


def _load_runner_pins() -> SimpleNamespace:
    """Cross-check independent constants against the runner at audit time."""

    from eval.opensource import run_bfree as runner

    return SimpleNamespace(
        module=runner,
        MODEL_SOURCE_COMMIT=_module_pin(
            runner, "MODEL_SOURCE_COMMIT", "SOURCE_COMMIT"
        ),
        SOURCE_FILES=_module_pin(runner, "SOURCE_FILES"),
        WEIGHTS_ZIP=_module_pin(
            runner, "OFFICIAL_ZIP", "WEIGHTS_ZIP", "ZIP_ASSET"
        ),
        CONFIG_ASSET=_module_pin(runner, "CONFIG_ASSET", "CONFIG"),
        CHECKPOINT=_module_pin(runner, "CHECKPOINT"),
        GOLDEN_CASES=_module_pin(runner, "GOLDEN_CASES"),
        GOLDEN_ABS_TOLERANCE=_module_pin(
            runner, "GOLDEN_ABS_TOLERANCE", "GOLDEN_ABSOLUTE_TOLERANCE"
        ),
        CANONICAL_RELEASE=_module_pin(runner, "CANONICAL_RELEASE"),
        VISIBILITY_CENSUS=_module_pin(
            runner, "FROZEN_VISIBILITY_CENSUS", "VISIBILITY_CENSUS"
        ),
        FEATURE_DIMENSION=_module_pin(runner, "FEATURE_DIMENSION"),
        CROPS=_module_pin(runner, "CROPS", "NUM_CROPS"),
        PATCH_SIZE=_module_pin(runner, "PATCH_SIZE"),
        CROP_SIZE=_module_pin(runner, "CROP_SIZE"),
        PREPROCESS_PROFILE=_module_pin(
            runner, "PREPROCESS_PROFILE", "FROZEN_PROFILE"
        ),
        FEATURE_SEMANTICS=_module_pin(runner, "FEATURE_SEMANTICS"),
        SCORE_SEMANTICS=_module_pin(runner, "SCORE_SEMANTICS"),
        T1_POLICY=_module_pin(runner, "T1_POLICY"),
    )


def _compare_nested(
    actual: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 0.0,
) -> None:
    """Require all independent fields while allowing recorded metadata extras."""

    if isinstance(expected, Mapping):
        actual_mapping = _require_mapping(actual, label)
        for key, expected_value in expected.items():
            if key not in actual_mapping:
                raise ValueError(f"{label} lacks independent field {key}")
            _compare_nested(
                actual_mapping[key],
                expected_value,
                label=f"{label}.{key}",
                float_tolerance=float_tolerance,
            )
        return
    if isinstance(expected, list):
        actual_list = _require_list(actual, label)
        if len(actual_list) != len(expected):
            raise ValueError(f"{label} length mismatch")
        for index, expected_value in enumerate(expected):
            _compare_nested(
                actual_list[index],
                expected_value,
                label=f"{label}[{index}]",
                float_tolerance=float_tolerance,
            )
        return
    if (
        isinstance(expected, (float, np.floating))
        and not isinstance(expected, (bool, np.bool_))
    ):
        actual_number = _require_finite(actual, label)
        if not math.isclose(
            actual_number,
            float(expected),
            rel_tol=0.0,
            abs_tol=float_tolerance,
        ):
            raise ValueError(
                f"{label} mismatch: expected {expected}, got {actual_number}"
            )
        return
    _require_equal(actual, expected, label)


def _reject_t2_localization_or_joint(value: Any, *, label: str) -> None:
    """Reject positive/ambiguous T2 claims at any nesting depth."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if key in _T2_OR_LOCALIZATION_KEYS:
                if key == "t2" and nested is False:
                    continue
                raise ValueError(
                    f"{label}.{raw_key} makes an unsupported B-Free "
                    "T2/localization/joint claim"
                )
            if key in ("valid_for_t2", "has_localization", "supports_t2"):
                if nested is not False:
                    raise ValueError(
                        f"{label}.{raw_key} must be explicitly false"
                    )
                continue
            _reject_t2_localization_or_joint(
                nested, label=f"{label}.{raw_key}"
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_t2_localization_or_joint(
                nested, label=f"{label}[{index}]"
            )


def _center_start(tokens: int) -> int:
    return max((tokens - CROP_GRID_SIZE) // 2, 0)


def compute_preprocess_geometry(width: int, height: int) -> dict[str, Any]:
    """Reconstruct the official patch projection and five token crops.

    ``Wrapper5crops`` applies the 14-pixel convolution before crop selection.
    If either token-grid dimension is below 36, it repeats the whole grid and
    truncates *both* dimensions to 36.  Consequently a short dimension can
    also discard tokens in the other, otherwise-long dimension.
    """

    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < PATCH_SIZE
        or height < PATCH_SIZE
    ):
        raise ValueError(
            f"B-Free input must be at least {PATCH_SIZE}x{PATCH_SIZE}"
        )
    grid_h = height // PATCH_SIZE
    grid_w = width // PATCH_SIZE
    wrap = grid_h < CROP_GRID_SIZE or grid_w < CROP_GRID_SIZE
    if wrap:
        starts_xy = [[0, 0] for _ in range(CROPS)]
        # The five duplicated rectangles intentionally mirror the released
        # runner's crop-by-crop evidence.  Their union is the native source
        # area referenced by the repeated patch grid.
        rectangle = [
            0,
            0,
            min(grid_w, CROP_GRID_SIZE) * PATCH_SIZE,
            min(grid_h, CROP_GRID_SIZE) * PATCH_SIZE,
        ]
        rects = [list(rectangle) for _ in range(CROPS)]
        effective_wh = [CROP_GRID_SIZE, CROP_GRID_SIZE]
    else:
        center_x = (grid_w - CROP_GRID_SIZE) // 2
        center_y = (grid_h - CROP_GRID_SIZE) // 2
        maximum_x = grid_w - CROP_GRID_SIZE
        maximum_y = grid_h - CROP_GRID_SIZE
        starts_xy = [
            [center_x, center_y],
            [
                0,
                0,
            ],
            [0, maximum_y],
            [maximum_x, maximum_y],
            [maximum_x, 0],
        ]
        rects = [
            [
                start_x * PATCH_SIZE,
                start_y * PATCH_SIZE,
                (start_x + CROP_GRID_SIZE) * PATCH_SIZE,
                (start_y + CROP_GRID_SIZE) * PATCH_SIZE,
            ]
            for start_x, start_y in starts_xy
        ]
        effective_wh = [grid_w, grid_h]
    return {
        "profile_id": FROZEN_PROFILE,
        "decoder": "Pillow.Image.open.convert_RGB",
        "native_size": [width, height],
        "resize": {"enabled": False},
        "to_tensor": "torchvision.transforms.ToTensor_uint8_div_255_float32",
        "normalization": {
            "mean": [float(value) for value in IMAGE_MEAN],
            "std": [float(value) for value in IMAGE_STD],
        },
        "patch_projection": {
            "kernel_size": [PATCH_SIZE, PATCH_SIZE],
            "stride": [PATCH_SIZE, PATCH_SIZE],
            "right_bottom_remainders_dropped": [
                width % PATCH_SIZE,
                height % PATCH_SIZE,
            ],
        },
        "patch_grid_wh": [grid_w, grid_h],
        "replicate_wrap_applied": wrap,
        "replicate_wrap_trigger": "either_patch_grid_dimension_below_36",
        "replicate_wrap_semantics": (
            "repeat_both_grid_dimensions_then_truncate_both_to_36"
            if wrap
            else "not_applicable"
        ),
        "post_wrap_patch_grid_wh": effective_wh,
        "crop_size_pixels": [CROP_SIZE, CROP_SIZE],
        "crop_size_patches": [CROP_GRID_SIZE, CROP_GRID_SIZE],
        "crop_order": [
            "center",
            "top_left",
            "bottom_left",
            "bottom_right",
            "top_right",
        ],
        "crop_starts_patch_xy": starts_xy,
        "distinct_crop_starts": len(
            {tuple(value) for value in starts_xy}
        ),
        "used_native_rectangles_xyxy": rects,
        "used_native_pixel_rule": (
            "union_of_integer_half_open_patch_receptive_field_rectangles"
        ),
    }


def preprocess_image(
    path: Path,
    *,
    torch_module: ModuleType,
) -> PreprocessedImage:
    """Decode RGB and independently reproduce ToTensor + ResNet normalize."""

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        decoded = np.asarray(rgb_image, dtype=np.uint8).copy()
    height, width = decoded.shape[:2]
    geometry = compute_preprocess_geometry(width, height)
    # This is deliberately independent of torchvision and official helpers.
    tensor = (
        torch_module.from_numpy(decoded.copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch_module.float32)
        .div(255.0)
    )
    mean = torch_module.as_tensor(
        np.asarray(IMAGE_MEAN, dtype=np.float32),
        dtype=torch_module.float32,
    ).view(3, 1, 1)
    std = torch_module.as_tensor(
        np.asarray(IMAGE_STD, dtype=np.float32),
        dtype=torch_module.float32,
    ).view(3, 1, 1)
    tensor = tensor.sub(mean).div(std).contiguous()
    audit = {
        "profile": FROZEN_PROFILE,
        "geometry": geometry,
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _decoded_sha256(decoded),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_sha256": _tensor_sha256(tensor),
        "normalization": {
            "mean": [float(value) for value in IMAGE_MEAN],
            "std": [float(value) for value in IMAGE_STD],
        },
        "resize_applied": False,
        "crop_applied_before_patch_projection": False,
    }
    return PreprocessedImage(tensor=tensor, decoded_rgb=decoded, audit=audit)


def _load_gt_mask(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if row.get("kind") != "forged":
        return np.zeros(expected_shape, dtype=bool)
    path_value = row.get("gt_mask_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("forged canonical row lacks gt_mask_path")
    path = _anchored(Path(path_value), repo_root).resolve()
    _verify_hash(path, row.get("gt_mask_sha256"), "forged GT mask")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != expected_shape:
        raise ValueError("forged GT mask shape does not match image")
    binary = mask > 0
    if int(np.count_nonzero(binary)) != int(row.get("gt_positive_pixels")):
        raise ValueError("forged GT mask positive-pixel count mismatch")
    return binary


def _visibility_from_exact_gt(
    mask: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    positive = int(np.count_nonzero(mask))
    if positive <= 0:
        raise ValueError("exact forged GT mask has no positive pixels")
    consumed = np.zeros(mask.shape, dtype=bool)
    rects = _require_list(
        geometry.get("used_native_rectangles_xyxy"),
        "used native rectangles",
    )
    for index, rect in enumerate(rects):
        values = _require_list(rect, f"used native rectangle {index}")
        if len(values) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("used native rectangle is malformed")
        left, top, right, bottom = values
        if not (
            0 <= left < right <= mask.shape[1]
            and 0 <= top < bottom <= mask.shape[0]
        ):
            raise ValueError("used native rectangle falls outside image")
        consumed[top:bottom, left:right] = True
    visible = int(np.count_nonzero(mask & consumed))
    fraction = visible / positive
    category = "none" if visible == 0 else "full" if visible == positive else "partial"
    return {
        "edit_visibility": category,
        "edit_visible_gt_fraction": fraction,
        "edit_visibility_evidence": {
            "definition": (
                "exact_gt_positive_pixels_intersect_union_of_native_pixels_"
                "consumed_by_official_five_token_crops"
            ),
            "gt_positive_pixels": positive,
            "visible_gt_positive_pixels": visible,
            "used_native_rectangles_xyxy": rects,
            "wrap_applied": bool(
                geometry.get("replicate_wrap_applied")
            ),
            "attention_or_localization_used": False,
        },
    }


def _pair_visibility(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        task_id = str(row.get("task_id"))
        kind = str(row.get("kind"))
        if kind not in ("real", "forged") or kind in by_task[task_id]:
            raise ValueError(f"canonical pair {task_id} has invalid {kind} row")
        by_task[task_id][kind] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in by_task.items():
        if set(pair) != {"real", "forged"}:
            # Single-image preflight selections inherit a frozen row-level
            # visibility record and are validated elsewhere.
            continue
        forged = pair["forged"]
        width = int(forged.get("width"))
        height = int(forged.get("height"))
        geometry = compute_preprocess_geometry(width, height)
        mask = _load_gt_mask(
            forged,
            repo_root=repo_root,
            expected_shape=(height, width),
        )
        core = _visibility_from_exact_gt(mask, geometry)
        evidence = {
            "category": core["edit_visibility"],
            "visible_fraction": core["edit_visible_gt_fraction"],
            "positive_pixels": int(np.count_nonzero(mask)),
            "visible_positive_pixels": int(
                core["edit_visibility_evidence"][
                    "visible_gt_positive_pixels"
                ]
            ),
            "forged_sample_id": str(forged["sample_id"]),
            "basis": (
                "exact_diff_positive_pixels_intersecting_union_of_official_"
                "five_patch_crop_receptive_fields"
            ),
            "geometry": geometry,
        }
        result[task_id] = {
            "edit_visibility": core["edit_visibility"],
            "edit_visible_gt_fraction": core[
                "edit_visible_gt_fraction"
            ],
            "edit_visibility_evidence": evidence,
            "geometry": geometry,
            "replicate_wrap_applied": geometry[
                "replicate_wrap_applied"
            ],
            "distinct_crop_starts": geometry[
                "distinct_crop_starts"
            ],
            "domain": str(forged["domain"]),
        }
    return result


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
            f"{label} mismatch: expected {expected}, got {value} "
            f"(absolute tolerance {tolerance})"
        )
    return value


def _audit_score_fields(
    row: Mapping[str, Any],
    *,
    replay_crop_logits: np.ndarray,
    replay_raw_logit: float,
    tolerance: float = RAW_LOGIT_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Validate every raw-score alias and the strict boundary decision."""

    crop_logits = np.asarray(replay_crop_logits, dtype=np.float32)
    if crop_logits.shape != (CROPS,) or not np.isfinite(crop_logits).all():
        raise ValueError("independent crop logits are malformed")
    replay = float(np.float32(replay_raw_logit))
    expected_decision = replay > FIXED_THRESHOLD
    expected_probability = bfree_fake_probability_float32(replay)
    for key in ("raw_logit", "ai_score", "score"):
        if key not in row:
            raise ValueError(f"successful row lacks {key}")
        _compare_float(
            row[key], replay, label=key, tolerance=tolerance
        )
    recorded_crops = np.asarray(
        _require_list(row.get("crop_logits"), "crop_logits"),
        dtype=np.float32,
    )
    if recorded_crops.shape != (CROPS,) or not np.allclose(
        recorded_crops,
        crop_logits,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("crop_logits mismatch")
    _require_equal(
        row.get("classification_decision"),
        expected_decision,
        "classification_decision",
    )
    _compare_float(
        row.get("classification_threshold"),
        FIXED_THRESHOLD,
        label="classification_threshold",
        tolerance=0.0,
    )
    _require_equal(
        row.get("classification_threshold_operator"),
        THRESHOLD_OPERATOR,
        "classification operator",
    )
    if "score_semantics" in row:
        _require_equal(
            row["score_semantics"], SCORE_SEMANTICS, "score_semantics"
        )
    classification = row.get("classification")
    if classification is not None:
        nested = _require_mapping(classification, "classification")
        aliases = {
            "raw_logit": replay,
            "ai_score": replay,
            "fake_probability": expected_probability,
            "decision": expected_decision,
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "semantics": SCORE_SEMANTICS,
        }
        for key, expected in aliases.items():
            if key not in nested:
                raise ValueError(f"classification lacks {key}")
            if isinstance(expected, float):
                _compare_float(
                    nested[key],
                    expected,
                    label=f"classification.{key}",
                    tolerance=tolerance if key != "threshold" else 0.0,
                )
            else:
                _require_equal(
                    nested[key], expected, f"classification.{key}"
                )
    t1 = row.get("t1")
    if t1 is not None:
        nested = _require_mapping(t1, "t1")
        _require_equal(nested.get("policy"), T1_POLICY, "t1.policy")
        _require_equal(nested.get("decision"), expected_decision, "t1.decision")
        for key, expected in (
            ("raw_logit", replay),
            ("ai_score", replay),
            ("fake_probability", expected_probability),
            ("threshold", FIXED_THRESHOLD),
        ):
            _compare_float(
                nested.get(key),
                expected,
                label=f"t1.{key}",
                tolerance=tolerance if key != "threshold" else 0.0,
            )
        _require_equal(
            nested.get("threshold_operator"),
            THRESHOLD_OPERATOR,
            "t1.threshold_operator",
        )
    if "fake_probability" in row:
        _compare_float(
            row["fake_probability"],
            expected_probability,
            label="fake_probability",
            tolerance=1e-7,
        )
    manual = row.get("manual_replay")
    if manual is not None:
        nested = _require_mapping(manual, "manual_replay")
        _compare_float(
            nested.get("raw_logit"),
            replay,
            label="manual_replay.raw_logit",
            tolerance=tolerance,
        )
        recorded_manual_crops = np.asarray(
            _require_list(
                nested.get("crop_logits"),
                "manual_replay.crop_logits",
            ),
            dtype=np.float32,
        )
        if recorded_manual_crops.shape != (CROPS,) or not np.allclose(
            recorded_manual_crops,
            crop_logits,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("manual_replay.crop_logits mismatch")
    return {
        "crop_logits": crop_logits.tolist(),
        "raw_logit": replay,
        "decision": expected_decision,
    }


def _verify_source_tree(source_root: Path) -> dict[str, Any]:
    resolved = source_root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"B-Free source root is missing: {resolved}")
    commit = _git_value(resolved, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_SOURCE_COMMIT, "B-Free source commit")
    hashes: dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_FILES.items():
        hashes[relative] = _verify_hash(
            resolved / relative,
            expected,
            f"B-Free source {relative}",
        )
    status = _git_value(resolved, "status", "--porcelain")
    if status not in ("", None):
        raise ValueError("B-Free pinned source worktree is dirty")
    return {
        "root": str(resolved),
        "commit": commit,
        "files": hashes,
        "clean_worktree": status == "",
    }


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, "r") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_schema(state: Mapping[str, Any], torch_module: ModuleType) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    total = 0
    for key, value in state.items():
        if not isinstance(key, str) or not torch_module.is_tensor(value):
            raise ValueError("B-Free state_dict contains a non-tensor entry")
        if value.dtype != torch_module.float32:
            raise ValueError("B-Free state_dict contains a non-float32 tensor")
        count = int(value.numel())
        total += count
        items.append(
            {
                "k": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "n": count,
            }
        )
    digest = hashlib.sha256(stable_json(items).encode("utf-8")).hexdigest()
    return {
        "tensor_count": len(items),
        "state_elements": total,
        "items_sha256": digest,
        "items": items,
    }


def _verify_assets(
    *,
    weights_dir: Path,
    weights_zip: Path,
    torch_module: ModuleType,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    weights = weights_dir.resolve()
    archive_path = weights_zip.resolve()
    config_path = weights / "config.yaml"
    checkpoint_path = weights / FROZEN_WEIGHTS_FILE
    _verify_hash(archive_path, FROZEN_ZIP_SHA256, "B-Free release ZIP")
    if archive_path.stat().st_size != FROZEN_ZIP_BYTES:
        raise ValueError("B-Free release ZIP byte count mismatch")
    if _md5_file(archive_path) != FROZEN_ZIP_MD5:
        raise ValueError("B-Free release ZIP official MD5 mismatch")
    _verify_hash(config_path, FROZEN_CONFIG_SHA256, "B-Free config")
    _verify_hash(
        checkpoint_path, FROZEN_CHECKPOINT_SHA256, "B-Free checkpoint"
    )
    if config_path.stat().st_size != FROZEN_CONFIG_BYTES:
        raise ValueError("B-Free config byte count mismatch")
    if checkpoint_path.stat().st_size != FROZEN_CHECKPOINT_BYTES:
        raise ValueError("B-Free checkpoint byte count mismatch")

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require_equal(config, FROZEN_CONFIG, "B-Free parsed config")
    with zipfile.ZipFile(archive_path, "r") as archive:
        files = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        )
        config_members = [name for name in files if name.endswith("/config.yaml")]
        checkpoint_members = [
            name for name in files if name.endswith("/model_epoch_best.pth")
        ]
        if len(config_members) != 1 or len(checkpoint_members) != 1:
            raise ValueError("B-Free release ZIP has unexpected member layout")
        if (
            _zip_member_sha256(archive, config_members[0])
            != FROZEN_CONFIG_SHA256
            or _zip_member_sha256(archive, checkpoint_members[0])
            != FROZEN_CHECKPOINT_SHA256
        ):
            raise ValueError(
                "B-Free extracted config/checkpoint do not match release ZIP"
            )

    unsafe_getter = getattr(
        torch_module.serialization,
        "get_unsafe_globals_in_checkpoint",
        None,
    )
    unsafe_globals = (
        list(unsafe_getter(checkpoint_path)) if unsafe_getter is not None else []
    )
    if unsafe_globals:
        raise ValueError(
            f"B-Free checkpoint requires unsafe globals: {unsafe_globals}"
        )
    payload = torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    top = _require_mapping(payload, "B-Free checkpoint")
    if list(top) != ["model"]:
        raise ValueError("B-Free checkpoint top-level schema is not only model")
    state = top["model"]
    if not isinstance(state, Mapping):
        raise ValueError("B-Free checkpoint model is not a state_dict")
    schema = _checkpoint_schema(state, torch_module)
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
    _require_equal(
        schema["items_sha256"],
        FROZEN_CHECKPOINT_SCHEMA_SHA256,
        "checkpoint schema digest",
    )
    evidence = {
        "weights_zip": {
            "path": str(archive_path),
            "sha256": FROZEN_ZIP_SHA256,
            "md5": FROZEN_ZIP_MD5,
            "bytes": FROZEN_ZIP_BYTES,
            "zip_members_match_extracted_assets": True,
        },
        "config": {
            "path": str(config_path),
            "sha256": FROZEN_CONFIG_SHA256,
            "bytes": FROZEN_CONFIG_BYTES,
            "parsed": config,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": FROZEN_CHECKPOINT_SHA256,
            "bytes": FROZEN_CHECKPOINT_BYTES,
            "loaded_with_weights_only": True,
            "unsafe_globals": unsafe_globals,
            "top_level_keys": list(top),
            "tensor_count": schema["tensor_count"],
            "state_elements": schema["state_elements"],
            "schema_items_sha256": schema["items_sha256"],
        },
    }
    return evidence, state


@contextlib.contextmanager
def _official_network_import(source_root: Path) -> Iterator[Any]:
    """Import only the already-hash-verified official network package."""

    code_root = str((source_root / "code").resolve())
    old_path = list(sys.path)
    displaced = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "networks" or name.startswith("networks.")
    }
    for name in displaced:
        del sys.modules[name]
    sys.path.insert(0, code_root)
    try:
        import networks

        yield networks
    finally:
        for name in list(sys.modules):
            if name == "networks" or name.startswith("networks."):
                del sys.modules[name]
        sys.modules.update(displaced)
        sys.path[:] = old_path


def _build_and_load_model(
    *,
    source_root: Path,
    state: Mapping[str, Any],
    torch_module: ModuleType,
) -> tuple[Any, dict[str, Any]]:
    """Construct the official architecture and strict-load both state halves."""

    attempts = {
        "urllib_urlopen": 0,
        "socket_create_connection": 0,
        "socket_connect": 0,
    }

    def reject(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            attempts[name] += 1
            raise RuntimeError("network access is forbidden")

        return blocked

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
        )
        stack.enter_context(
            mock.patch.object(
                __import__("urllib.request", fromlist=["urlopen"]),
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
        with _official_network_import(source_root) as networks:
            model = networks.get_network(
                FROZEN_CONFIG_ARCH, pretrained=False
            )
    if any(attempts.values()):
        raise RuntimeError(
            f"B-Free construction attempted network access: {attempts}"
        )
    patch_state = {
        key[len("patch_embed.") :]: value
        for key, value in state.items()
        if key.startswith("patch_embed.")
    }
    body_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("patch_embed.")
    }
    patch_result = model.patch_embed.load_state_dict(patch_state, strict=True)
    body_result = model.model.load_state_dict(body_state, strict=True)
    if (
        patch_result.missing_keys
        or patch_result.unexpected_keys
        or body_result.missing_keys
        or body_result.unexpected_keys
    ):
        raise ValueError("B-Free checkpoint strict load was not exact")
    parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    _require_equal(
        parameters,
        FROZEN_MODEL_PARAMETER_COUNT,
        "B-Free model parameter count",
    )
    model.eval()
    return model, {
        "architecture": FROZEN_CONFIG_ARCH,
        "timm_architecture": FROZEN_TIMM_ARCH,
        "strict_load": True,
        "patch_embed_missing_keys": [],
        "patch_embed_unexpected_keys": [],
        "body_missing_keys": [],
        "body_unexpected_keys": [],
        "parameter_count": parameters,
        "network": {
            "allowed": False,
            "attempts": attempts,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_truth(torch_module: ModuleType, device: Any) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": sys.version,
        "torch": str(torch_module.__version__),
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "transformers": _package_version("transformers"),
        "numpy": str(np.__version__),
        "pillow": _package_version("Pillow"),
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_version": torch_module.version.cuda,
        "cudnn_enabled": bool(torch_module.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": str(
            torch_module.get_float32_matmul_precision()
        ),
        "dtype": "float32",
        "autocast": False,
        "amp": False,
        "network_allowed": False,
        "seed": MODEL_SEED,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cuda_device_name": (
            torch_module.cuda.get_device_name(device) if cuda else None
        ),
    }


def _replay_runtime(
    manifest: Mapping[str, Any],
    *,
    requested_device: str | None = None,
) -> ReplayRuntime:
    import torch

    recorded = _require_mapping(manifest.get("runtime"), "manifest.runtime")
    device_name = requested_device or str(recorded.get("device", "cpu"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("recorded B-Free CUDA runtime is unavailable")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    # Keep the released framework default on both CPU and CUDA.  It is still
    # recorded on CPU even though no cuDNN kernel is selected there.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    evidence = _runtime_truth(torch, device)
    # Compare every truth field shared by the recorded and independent
    # runtimes.  Paths, timestamps, and runner-only metadata remain harmless.
    for key, value in evidence.items():
        if key in recorded:
            _require_equal(recorded[key], value, f"manifest.runtime.{key}")
    return ReplayRuntime(torch=torch, device=device, evidence=evidence)


def _classifier(model: Any) -> Any:
    body = getattr(model, "model", None)
    head = getattr(body, "head", None)
    if head is None:
        raise ValueError("B-Free model lacks model.head")
    return head


def _forward_with_evidence(
    model: Any,
    tensor: Any,
    runtime: ReplayRuntime,
) -> ForwardEvidence:
    """Run one complete official wrapper forward and capture its head input."""

    torch = runtime.torch
    captured: list[Any] = []

    def capture(_module: Any, arguments: tuple[Any, ...]) -> None:
        if len(arguments) != 1:
            raise ValueError("B-Free head hook received unexpected arguments")
        captured.append(arguments[0].detach().clone())

    hook = _classifier(model).register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            output = model(tensor.unsqueeze(0).to(runtime.device))
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("B-Free classifier hook did not fire exactly once")
    feature_tensor = captured[0]
    if tuple(feature_tensor.shape) != (CROPS, FEATURE_DIMENSION):
        raise ValueError(
            "B-Free head feature shape is not five by 768"
        )
    if feature_tensor.dtype != torch.float32:
        raise ValueError("B-Free head feature dtype is not float32")
    with torch.inference_mode():
        replay_logits_tensor = _classifier(model)(feature_tensor).reshape(CROPS)
        replay_mean_tensor = torch.mean(replay_logits_tensor, dtype=torch.float32)
    wrapper_value = float(output.reshape(()).detach().cpu().item())
    replay_value = float(replay_mean_tensor.detach().cpu().item())
    if wrapper_value != replay_value:
        raise ValueError(
            "B-Free wrapper output differs from five-crop head replay mean"
        )
    features = np.ascontiguousarray(
        feature_tensor.detach().cpu().numpy(), dtype=np.float32
    )
    crop_logits = np.ascontiguousarray(
        replay_logits_tensor.detach().cpu().numpy(), dtype=np.float32
    )
    if not np.isfinite(features).all() or not np.isfinite(crop_logits).all():
        raise ValueError("B-Free fresh forward produced non-finite evidence")
    return ForwardEvidence(
        features=features,
        crop_logits=crop_logits,
        raw_logit=replay_value,
        wrapper_raw_logit=wrapper_value,
    )


def _safe_npz(
    path: Path,
    *,
    expected_keys: set[str],
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise ValueError(
                    "B-Free NPZ artifact keys mismatch: "
                    f"{sorted(archive.files)}"
                )
            arrays = {
                key: np.ascontiguousarray(archive[key])
                for key in sorted(expected_keys)
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("B-Free"):
            raise
        raise ValueError(f"unsafe or malformed B-Free NPZ artifact: {path}") from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("B-Free NPZ artifact contains an object array")
    return arrays


def _load_artifact(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    path_value = row.get("artifact_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("successful B-Free row lacks artifact_path")
    path = _anchored(Path(path_value), repo_root).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "B-Free artifact path falls outside its run directory"
        ) from exc
    if path.suffix.lower() != ".npz":
        raise ValueError("B-Free artifact is not NPZ")
    _verify_hash(path, row.get("artifact_sha256"), "B-Free artifact")
    arrays = _safe_npz(path, expected_keys={"features", "crop_logits"})
    features = arrays["features"]
    crop_logits = arrays["crop_logits"]
    if features.shape != (CROPS, FEATURE_DIMENSION):
        raise ValueError("B-Free artifact features shape mismatch")
    if crop_logits.shape != (CROPS,):
        raise ValueError("B-Free artifact crop_logits shape mismatch")
    if features.dtype != FEATURE_DTYPE or crop_logits.dtype != FEATURE_DTYPE:
        raise ValueError("B-Free artifact dtype is not float32")
    if not np.isfinite(features).all() or not np.isfinite(crop_logits).all():
        raise ValueError("B-Free artifact contains non-finite values")
    _require_equal(
        _array_sha256(features),
        _require_sha256(
            row.get("feature_array_sha256"), "feature_array_sha256"
        ),
        "feature raw-array SHA-256",
    )
    _require_equal(
        _array_sha256(crop_logits),
        _require_sha256(
            row.get("crop_logits_array_sha256"),
            "crop_logits_array_sha256",
        ),
        "crop-logit raw-array SHA-256",
    )
    return features, crop_logits, path


def _replay_artifact_head(
    features: np.ndarray,
    model: Any,
    runtime: ReplayRuntime,
) -> tuple[np.ndarray, float]:
    torch = runtime.torch
    tensor = torch.from_numpy(
        np.ascontiguousarray(features, dtype=np.float32)
    ).to(runtime.device)
    with torch.inference_mode():
        logits_tensor = _classifier(model)(tensor).reshape(CROPS)
        mean_tensor = torch.mean(logits_tensor, dtype=torch.float32)
    logits = np.ascontiguousarray(
        logits_tensor.detach().cpu().numpy(), dtype=np.float32
    )
    return logits, float(mean_tensor.detach().cpu().item())


def _validate_result_identity(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    index: int,
) -> None:
    sample_id = str(expected.get("sample_id"))
    if row.get("id") != sample_id or row.get("sample_id", sample_id) != sample_id:
        raise ValueError(f"result row {index} identity mismatch")
    for key in ("task_id", "kind", "label", "domain"):
        if row.get(key) != expected.get(key):
            raise ValueError(f"result row {index} {key} mismatch")
    status = row.get("status")
    if status not in ("ok", "error"):
        raise ValueError(f"result row {index} has invalid status")
    if row.get("edit_visibility") not in ("none", "partial", "full"):
        raise ValueError(f"result row {index} has invalid edit_visibility")
    fraction = _require_finite(
        row.get("edit_visible_gt_fraction"),
        f"result row {index} visible fraction",
    )
    expected_visibility = (
        "none" if fraction == 0.0 else "full" if fraction == 1.0 else "partial"
    )
    if not 0.0 <= fraction <= 1.0 or row["edit_visibility"] != expected_visibility:
        raise ValueError(f"result row {index} visibility mismatch")
    if status == "ok":
        _require_equal(
            row.get("valid_for_metrics"),
            True,
            f"result row {index} valid_for_metrics",
        )
        latency = _require_finite(
            row.get("latency_ms"), f"result row {index} latency_ms"
        )
        if latency < 0:
            raise ValueError(f"result row {index} has negative latency")
    else:
        _require_equal(
            row.get("valid_for_metrics"),
            False,
            f"result row {index} valid_for_metrics",
        )
        for key in (
            "raw_logit",
            "ai_score",
            "score",
            "crop_logits",
            "classification_decision",
        ):
            if key not in row:
                raise ValueError(f"error result row {index} lacks {key}")
            if row[key] is not None:
                raise ValueError(f"error result row {index} {key} is not null")
        if "artifact_path" in row and row["artifact_path"] is not None:
            raise ValueError(
                f"error result row {index} artifact_path is not null"
            )


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("physical result row lacks a valid id")
        latest[row_id] = row
    return latest


def _load_run_files(*, results_dir: Path, run_id: str) -> RunFiles:
    safe_id = _safe_component(run_id, label="run_id")
    run_dir = (results_dir / safe_id).resolve()
    base = results_dir.resolve()
    try:
        run_dir.relative_to(base)
    except ValueError as exc:
        raise ValueError("run directory escapes results directory") from exc
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file() and (run_dir / "manifest.json").is_file():
        manifest_path = run_dir / "manifest.json"
    for label, path in (
        ("results", results_path),
        ("expected inputs", expected_path),
        ("summary", summary_path),
        ("run manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"B-Free {label} file is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunFiles(
        run_dir=run_dir,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        rows=read_jsonl(results_path),
        expected=read_jsonl(expected_path),
        summary=_require_mapping(summary, "summary"),
        manifest=_require_mapping(manifest, "manifest"),
    )


def _validate_run_manifest_identity(
    *,
    repo_root: Path,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
    files: RunFiles,
) -> dict[str, Any]:
    """Bind the parsed run to its physical files and frozen configuration."""

    manifest = files.manifest
    run_id = files.run_dir.name
    _require_equal(
        manifest.get("schema_version"),
        "bfree_detection_run_manifest_v1",
        "manifest schema_version",
    )
    _require_equal(manifest.get("run_id"), run_id, "manifest run_id")
    _require_equal(manifest.get("status"), "complete", "manifest status")
    if not isinstance(manifest.get("completed_at"), str) or not manifest[
        "completed_at"
    ]:
        raise ValueError("complete B-Free manifest lacks completed_at")

    config = _require_mapping(manifest.get("config"), "manifest.config")
    recorded_fingerprint = _require_sha256(
        manifest.get("config_fingerprint"),
        "manifest config_fingerprint",
    )
    actual_fingerprint = _fingerprint(config)
    _require_equal(
        recorded_fingerprint,
        actual_fingerprint,
        "manifest config_fingerprint",
    )
    for index, row in enumerate(files.rows):
        _require_equal(
            row.get("config_fingerprint"),
            recorded_fingerprint,
            f"result row {index} config_fingerprint",
        )
    if "config_fingerprint" in files.summary:
        _require_equal(
            files.summary["config_fingerprint"],
            recorded_fingerprint,
            "summary config_fingerprint",
        )
    if "run_id" in files.summary:
        _require_equal(files.summary["run_id"], run_id, "summary run_id")

    outputs = _require_mapping(
        manifest.get("outputs"), "manifest.outputs"
    )

    def output_path(key: str, physical: Path) -> None:
        value = outputs.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest.outputs.{key} is not a path")
        observed = _anchored(Path(value), repo_root).resolve()
        _require_equal(
            observed,
            physical.resolve(),
            f"manifest.outputs.{key}",
        )

    output_path("results_path", files.results_path)
    output_path("summary_path", files.summary_path)
    _require_equal(
        outputs.get("results_sha256"),
        sha256_file(files.results_path),
        "manifest outputs results SHA-256",
    )
    _require_equal(
        outputs.get("summary_sha256"),
        sha256_file(files.summary_path),
        "manifest outputs summary SHA-256",
    )
    artifact_dir = files.run_dir / "artifacts"
    output_path("artifact_dir", artifact_dir)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"B-Free artifact directory is missing: {artifact_dir}"
        )
    artifacts = sorted(
        path.resolve()
        for path in artifact_dir.glob("*.npz")
        if path.is_file()
    )
    artifact_count = outputs.get("artifact_files")
    if isinstance(artifact_count, bool) or not isinstance(
        artifact_count, int
    ):
        raise ValueError("manifest.outputs.artifact_files is not an integer")
    _require_equal(
        artifact_count,
        len(artifacts),
        "manifest artifact file count",
    )

    dataset = _require_mapping(
        manifest.get("dataset"), "manifest.dataset"
    )
    expected_value = dataset.get("expected_inputs_path")
    if not isinstance(expected_value, str) or not expected_value:
        raise ValueError("manifest.dataset.expected_inputs_path is invalid")
    _require_equal(
        _anchored(Path(expected_value), repo_root).resolve(),
        files.expected_path.resolve(),
        "manifest expected_inputs_path",
    )
    _require_equal(
        dataset.get("expected_inputs_sha256"),
        sha256_file(files.expected_path),
        "manifest expected_inputs SHA-256",
    )
    _require_equal(
        dataset.get("selected_images"),
        len(files.expected),
        "manifest selected image count",
    )

    source = _require_mapping(manifest.get("source"), "manifest.source")
    _require_equal(
        source.get("commit"), FROZEN_SOURCE_COMMIT, "manifest source commit"
    )
    _require_equal(
        source.get("tracked_dirty"), False, "manifest source tracked_dirty"
    )
    _require_equal(
        Path(str(source.get("root"))).resolve(),
        source_root.resolve(),
        "manifest source root",
    )
    source_files = _require_mapping(
        source.get("files"), "manifest.source.files"
    )
    _require_equal(
        set(source_files),
        set(FROZEN_SOURCE_FILES),
        "manifest source file set",
    )
    for relative, digest in FROZEN_SOURCE_FILES.items():
        recorded = _require_mapping(
            source_files.get(relative),
            f"manifest source file {relative}",
        )
        _require_equal(
            recorded.get("sha256"),
            digest,
            f"manifest source file {relative} SHA-256",
        )
        _require_equal(
            Path(str(recorded.get("path"))).resolve(),
            (source_root / relative).resolve(),
            f"manifest source file {relative} path",
        )

    assets = _require_mapping(manifest.get("assets"), "manifest.assets")
    zip_asset = _require_mapping(assets.get("zip"), "manifest.assets.zip")
    for key, expected in (
        ("sha256", FROZEN_ZIP_SHA256),
        ("verified_sha256", FROZEN_ZIP_SHA256),
        ("md5", FROZEN_ZIP_MD5),
        ("verified_md5", FROZEN_ZIP_MD5),
        ("bytes", FROZEN_ZIP_BYTES),
        ("members", FROZEN_ZIP_MEMBERS),
    ):
        _require_equal(
            zip_asset.get(key), expected, f"manifest ZIP {key}"
        )
    _require_equal(
        Path(str(zip_asset.get("path"))).resolve(),
        weights_zip.resolve(),
        "manifest ZIP path",
    )
    config_asset = _require_mapping(
        assets.get("config"), "manifest.assets.config"
    )
    for key, expected in (
        ("sha256", FROZEN_CONFIG_SHA256),
        ("bytes", FROZEN_CONFIG_BYTES),
        ("parsed", FROZEN_CONFIG),
        ("parsed_actual", FROZEN_CONFIG),
    ):
        _require_equal(
            config_asset.get(key), expected, f"manifest config asset {key}"
        )
    _require_equal(
        Path(str(config_asset.get("path"))).resolve(),
        (weights_dir / "config.yaml").resolve(),
        "manifest config asset path",
    )
    checkpoint = _require_mapping(
        assets.get("checkpoint"), "manifest.assets.checkpoint"
    )
    for key, expected in (
        ("sha256", FROZEN_CHECKPOINT_SHA256),
        ("bytes", FROZEN_CHECKPOINT_BYTES),
        ("tensor_count", FROZEN_CHECKPOINT_TENSORS),
        ("state_elements", FROZEN_CHECKPOINT_ELEMENTS),
        ("schema_sha256", FROZEN_CHECKPOINT_SCHEMA_SHA256),
        ("top_level_keys", ["model"]),
        ("state_container", "collections.OrderedDict"),
        ("dtype", "float32"),
        ("safe_weights_only_load", True),
        ("unsafe_globals", []),
    ):
        _require_equal(
            checkpoint.get(key), expected, f"manifest checkpoint {key}"
        )
    _require_equal(
        Path(str(checkpoint.get("path"))).resolve(),
        (weights_dir / FROZEN_WEIGHTS_FILE).resolve(),
        "manifest checkpoint path",
    )
    checkpoint_schema = _require_mapping(
        checkpoint.get("schema"), "manifest checkpoint schema"
    )
    for key, expected in (
        ("top_level_keys", ["model"]),
        ("state_container", "collections.OrderedDict"),
        ("tensor_count", FROZEN_CHECKPOINT_TENSORS),
        ("state_elements", FROZEN_CHECKPOINT_ELEMENTS),
        ("all_dtype", "torch.float32"),
        ("all_finite", True),
        ("schema_sha256", FROZEN_CHECKPOINT_SCHEMA_SHA256),
    ):
        _require_equal(
            checkpoint_schema.get(key),
            expected,
            f"manifest checkpoint schema {key}",
        )

    frozen_config_fields = {
        "model": "B-Free",
        "model_slug": "bfree_dino2reg4",
        "model_arch": FROZEN_TIMM_ARCH,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "source_files": FROZEN_SOURCE_FILES,
        "official_zip_sha256": FROZEN_ZIP_SHA256,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "checkpoint_schema_sha256": FROZEN_CHECKPOINT_SCHEMA_SHA256,
        "preprocess_profile": FROZEN_PROFILE,
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
    }
    for key, expected in frozen_config_fields.items():
        _require_equal(
            config.get(key), expected, f"manifest.config.{key}"
        )
    model_contract = _require_mapping(
        config.get("model_contract"), "manifest.config.model_contract"
    )
    _compare_nested(
        model_contract,
        {
            "official_wrapper": True,
            "strict_full_checkpoint_load": True,
            "feature_shape": [CROPS, FEATURE_DIMENSION],
            "feature_semantics": FEATURE_SEMANTICS,
            "crop_logits_shape": [CROPS],
            "primary_score": SCORE_SEMANTICS,
            "score_direction": "higher_means_fake",
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        label="manifest.config.model_contract",
    )
    _require_equal(
        config.get("official_golden"),
        manifest.get("official_golden"),
        "manifest/config official_golden",
    )
    _require_equal(
        config.get("runtime_evidence"),
        manifest.get("runtime"),
        "manifest/config runtime evidence",
    )
    _require_equal(
        config.get("model_audit"),
        manifest.get("model_audit"),
        "manifest/config model audit",
    )
    frozen_visibility = _require_mapping(
        config.get("frozen_full_dataset_visibility"),
        "manifest.config frozen visibility",
    )
    _compare_nested(
        _normalize_recorded_visibility_census(frozen_visibility),
        FROZEN_VISIBILITY_CENSUS,
        label="manifest.config frozen visibility",
        float_tolerance=1e-15,
    )
    return {
        "schema_version": "bfree_detection_run_manifest_v1",
        "run_id": run_id,
        "status": "complete",
        "config_fingerprint": recorded_fingerprint,
        "physical_result_rows_bound": len(files.rows),
        "results_sha256_bound": True,
        "summary_sha256_bound": True,
        "expected_inputs_sha256_bound": True,
        "artifact_files": len(artifacts),
        "source_freeze_bound": True,
        "asset_freeze_bound": True,
        "config_freeze_bound": True,
    }


def _validate_canonical_input(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("canonical input lacks sample_id")
    kind = row.get("kind")
    label = row.get("label")
    if kind not in ("real", "forged") or label != int(kind == "forged"):
        raise ValueError(f"canonical input {sample_id} kind/label mismatch")
    path_value = row.get("canonical_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"canonical input {sample_id} lacks canonical_path")
    path = _anchored(Path(path_value), repo_root).resolve()
    _verify_hash(
        path,
        row.get("canonical_sha256"),
        f"canonical input {sample_id}",
    )
    with Image.open(path) as image:
        width, height = image.size
    if int(row.get("width")) != width or int(row.get("height")) != height:
        raise ValueError(f"canonical input {sample_id} dimensions mismatch")
    return path


def _validate_selection(
    *,
    repo_root: Path,
    files: RunFiles,
) -> dict[str, Any]:
    config = _require_mapping(files.manifest.get("config"), "manifest.config")
    dataset = _require_mapping(config.get("dataset"), "manifest.config.dataset")
    ids = [str(row.get("sample_id")) for row in files.expected]
    if len(ids) != len(set(ids)):
        raise ValueError("expected_inputs contains duplicate sample_id")
    if "selected_ids" in dataset:
        _require_equal(dataset["selected_ids"], ids, "selected_ids")
    if "selected_rows_sha256" in dataset:
        _require_equal(
            dataset["selected_rows_sha256"],
            _selected_rows_sha256(files.expected),
            "selected_rows_sha256",
        )
    expected_ids = set(ids)
    for index, row in enumerate(files.rows):
        row_id = row.get("id")
        if row_id not in expected_ids:
            raise ValueError(f"unexpected physical result id {row_id}")
        expected = files.expected[ids.index(str(row_id))]
        _validate_result_identity(row, expected, index=index)
    latest = _latest_by_id(files.rows)
    unknown = set(latest) - expected_ids
    if unknown:
        raise ValueError(f"unexpected latest result id {sorted(unknown)[0]}")
    return {
        "selected_images": len(files.expected),
        "physical_result_rows": len(files.rows),
        "selected_rows_sha256": _selected_rows_sha256(files.expected),
        "selection_order_preserved": True,
    }


def _validate_frozen_release(
    *,
    repo_root: Path,
    files: RunFiles,
) -> dict[str, Any]:
    manifest_dataset = _require_mapping(
        files.manifest.get("dataset_release", {}),
        "manifest.dataset_release",
    )
    is_full = len(files.expected) == FROZEN_CANONICAL_RELEASE["images"]
    release_path = repo_root / "outputs/opensource/mouse_canonical_v1/manifest.json"
    if not is_full or not release_path.is_file():
        return {
            "full_frozen_release": False,
            "selected_images": len(files.expected),
        }
    release = _require_mapping(
        json.loads(release_path.read_text(encoding="utf-8")),
        "canonical release",
    )
    for key in ("schema_version", "dataset_id", "pairs", "images"):
        _require_equal(
            release.get(key), FROZEN_CANONICAL_RELEASE[key], f"release.{key}"
        )
    inputs_path = _anchored(Path(str(release["inputs_path"])), repo_root)
    pairs_path = _anchored(Path(str(release["pairs_path"])), repo_root)
    _verify_hash(
        inputs_path,
        FROZEN_CANONICAL_RELEASE["inputs_sha256"],
        "canonical inputs JSONL",
    )
    _verify_hash(
        pairs_path,
        FROZEN_CANONICAL_RELEASE["pairs_sha256"],
        "canonical pairs JSONL",
    )
    if manifest_dataset:
        for key, expected in FROZEN_CANONICAL_RELEASE.items():
            if key in manifest_dataset:
                _require_equal(
                    manifest_dataset[key], expected, f"manifest dataset {key}"
                )
    canonical_rows = read_jsonl(inputs_path)
    if files.expected != canonical_rows:
        raise ValueError("full expected_inputs is not frozen canonical JSONL")
    return {
        "full_frozen_release": True,
        **FROZEN_CANONICAL_RELEASE,
    }


def _validate_runner_pins() -> dict[str, Any]:
    pins = _load_runner_pins()
    _require_equal(
        pins.MODEL_SOURCE_COMMIT,
        FROZEN_SOURCE_COMMIT,
        "runner source commit",
    )
    _require_equal(pins.SOURCE_FILES, FROZEN_SOURCE_FILES, "runner source files")
    zip_pin = _require_mapping(pins.WEIGHTS_ZIP, "runner weights ZIP")
    for key, expected in (
        ("sha256", FROZEN_ZIP_SHA256),
        ("md5", FROZEN_ZIP_MD5),
        ("bytes", FROZEN_ZIP_BYTES),
    ):
        _require_equal(zip_pin.get(key), expected, f"runner ZIP {key}")
    config_pin = _require_mapping(pins.CONFIG_ASSET, "runner config")
    _require_equal(
        config_pin.get("sha256"), FROZEN_CONFIG_SHA256, "runner config hash"
    )
    checkpoint = _require_mapping(pins.CHECKPOINT, "runner checkpoint")
    for key, expected in (
        ("sha256", FROZEN_CHECKPOINT_SHA256),
        ("bytes", FROZEN_CHECKPOINT_BYTES),
        ("tensor_count", FROZEN_CHECKPOINT_TENSORS),
        ("state_elements", FROZEN_CHECKPOINT_ELEMENTS),
    ):
        _require_equal(checkpoint.get(key), expected, f"runner checkpoint {key}")
    runner_schema = checkpoint.get(
        "schema_items_sha256", checkpoint.get("schema_sha256")
    )
    _require_equal(
        runner_schema,
        FROZEN_CHECKPOINT_SCHEMA_SHA256,
        "runner checkpoint schema SHA-256",
    )
    runner_golden = tuple(dict(case) for case in pins.GOLDEN_CASES)
    if len(runner_golden) != len(FROZEN_GOLDEN_CASES):
        raise ValueError("runner golden case count mismatch")
    for index, (runner_case, frozen_case) in enumerate(
        zip(runner_golden, FROZEN_GOLDEN_CASES, strict=True)
    ):
        expected_fields = {
            "filename": frozen_case["filename"],
            "sha256": frozen_case["sha256"],
            "published_raw_logit": frozen_case["official_raw_logit"],
            "decoded_rgb_sha256": frozen_case["decoded_rgb_sha256"],
            "tensor_sha256": frozen_case["tensor_sha256"],
            "tensor_shape": [
                3,
                frozen_case["height"],
                frozen_case["width"],
            ],
            "patch_grid_wh": list(
                reversed(frozen_case["patch_grid_hw"])
            ),
        }
        for key, expected in expected_fields.items():
            _require_equal(
                runner_case.get(key),
                expected,
                f"runner golden case {index} {key}",
            )
    _require_equal(
        pins.GOLDEN_ABS_TOLERANCE,
        GOLDEN_ABSOLUTE_TOLERANCE,
        "runner golden tolerance",
    )
    _require_equal(
        pins.CANONICAL_RELEASE,
        FROZEN_CANONICAL_RELEASE,
        "runner canonical release",
    )
    _require_equal(
        pins.VISIBILITY_CENSUS,
        FROZEN_VISIBILITY_CENSUS,
        "runner visibility census",
    )
    for actual, expected, label in (
        (pins.FEATURE_DIMENSION, FEATURE_DIMENSION, "feature dimension"),
        (pins.CROPS, CROPS, "crop count"),
        (pins.PATCH_SIZE, PATCH_SIZE, "patch size"),
        (pins.CROP_SIZE, CROP_SIZE, "crop size"),
        (pins.PREPROCESS_PROFILE, FROZEN_PROFILE, "preprocess profile"),
        (pins.FEATURE_SEMANTICS, FEATURE_SEMANTICS, "feature semantics"),
        (pins.SCORE_SEMANTICS, SCORE_SEMANTICS, "score semantics"),
        (pins.T1_POLICY, T1_POLICY, "T1 policy"),
    ):
        _require_equal(actual, expected, f"runner {label}")
    return {
        "status": "ok",
        "source_commit": pins.MODEL_SOURCE_COMMIT,
        "checkpoint_sha256": checkpoint["sha256"],
        "golden_cases": len(pins.GOLDEN_CASES),
    }


def _audit_official_golden(
    *,
    source_root: Path,
    model: Any,
    runtime: ReplayRuntime,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    demo_root = source_root / "code" / "demo_images"
    for frozen in FROZEN_GOLDEN_CASES:
        path = demo_root / str(frozen["filename"])
        _verify_hash(path, frozen["sha256"], f"golden {frozen['filename']}")
        prepared = preprocess_image(path, torch_module=runtime.torch)
        _require_equal(
            prepared.audit["decoded_rgb_sha256"],
            frozen["decoded_rgb_sha256"],
            f"golden {frozen['filename']} decoded hash",
        )
        _require_equal(
            prepared.audit["tensor_sha256"],
            frozen["tensor_sha256"],
            f"golden {frozen['filename']} tensor hash",
        )
        _require_equal(
            prepared.audit["geometry"]["patch_grid_wh"],
            list(reversed(frozen["patch_grid_hw"])),
            f"golden {frozen['filename']} patch grid",
        )
        first = _forward_with_evidence(model, prepared.tensor, runtime)
        second = _forward_with_evidence(model, prepared.tensor, runtime)
        if (
            not np.array_equal(first.features, second.features)
            or not np.array_equal(first.crop_logits, second.crop_logits)
            or first.raw_logit != second.raw_logit
        ):
            raise ValueError(
                f"golden {frozen['filename']} is not bit-reproducible"
            )
        difference = abs(
            first.raw_logit - float(frozen["official_raw_logit"])
        )
        if difference > GOLDEN_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"golden {frozen['filename']} differs from official output"
            )
        cases.append(
            {
                "filename": frozen["filename"],
                "sha256": frozen["sha256"],
                "official_raw_logit": frozen["official_raw_logit"],
                "observed_raw_logit": first.raw_logit,
                "absolute_difference": difference,
                "within_official_tolerance": True,
                "two_runs_bit_identical": True,
                "feature_array_sha256": _array_sha256(first.features),
                "crop_logits": first.crop_logits.tolist(),
                "tensor_sha256": prepared.audit["tensor_sha256"],
            }
        )
    result = {
        "status": "passed",
        "official_tolerance": GOLDEN_ABSOLUTE_TOLERANCE,
        "cases": cases,
        "all_cases_within_official_tolerance": True,
        "all_cases_two_runs_bit_identical": True,
    }
    recorded_mapping = _require_mapping(recorded, "manifest.official_golden")
    if recorded_mapping.get("status") not in ("passed", "ok"):
        raise ValueError("recorded official golden gate did not pass")
    recorded_cases = recorded_mapping.get("cases")
    if isinstance(recorded_cases, list):
        by_name = {
            str(case.get("filename")): case
            for case in recorded_cases
            if isinstance(case, Mapping)
        }
        for case in cases:
            other = _require_mapping(
                by_name.get(str(case["filename"])),
                f"recorded golden {case['filename']}",
            )
            recorded_raw_logit = other.get(
                "actual_raw_logit",
                other.get("observed_raw_logit"),
            )
            _compare_float(
                recorded_raw_logit,
                float(case["observed_raw_logit"]),
                label=f"recorded golden {case['filename']} raw logit",
                tolerance=1e-6,
            )
    return result


def _maximum_absolute_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.shape != second.shape:
        raise ValueError("arrays have different shapes")
    return float(
        np.max(
            np.abs(
                first.astype(np.float64, copy=False)
                - second.astype(np.float64, copy=False)
            ),
            initial=0.0,
        )
    )


def audit_artifacts(
    *,
    repo_root: Path,
    all_inputs: list[dict[str, Any]],
    files: RunFiles,
    runtime: ReplayRuntime,
    model: Any,
) -> dict[str, Any]:
    """Fresh-forward every selected image and validate all persisted evidence."""

    expected_by_id = {
        str(row["sample_id"]): row for row in all_inputs
    }
    if len(expected_by_id) != len(all_inputs):
        raise ValueError("selected input IDs are not unique")
    latest = _latest_by_id(files.rows)
    pair_visibility = _pair_visibility(all_inputs, repo_root=repo_root)
    replay_by_id: dict[str, dict[str, Any]] = {}
    maximum_feature = 0.0
    maximum_crop_logit = 0.0
    maximum_raw = 0.0
    maximum_artifact_head_logit = 0.0
    wrap_images = 0
    artifact_paths: set[Path] = set()

    model = model.to(runtime.device).eval()
    for sample_id, expected in expected_by_id.items():
        path = _validate_canonical_input(expected, repo_root=repo_root)
        prepared = preprocess_image(path, torch_module=runtime.torch)
        if prepared.audit["geometry"]["replicate_wrap_applied"]:
            wrap_images += 1
        forward = _forward_with_evidence(model, prepared.tensor, runtime)
        row = latest.get(sample_id)
        if row is None:
            replay_by_id[sample_id] = {
                "status": "missing",
                "raw_logit": forward.raw_logit,
            }
            continue
        visibility = pair_visibility.get(str(expected["task_id"]))
        if visibility is not None:
            _require_equal(
                row.get("edit_visibility"),
                visibility["edit_visibility"],
                f"{sample_id} edit_visibility",
            )
            _compare_float(
                row.get("edit_visible_gt_fraction"),
                float(visibility["edit_visible_gt_fraction"]),
                label=f"{sample_id} edit_visible_gt_fraction",
                tolerance=0.0,
            )
            if "edit_visibility_evidence" in row:
                _compare_nested(
                    row["edit_visibility_evidence"],
                    visibility["edit_visibility_evidence"],
                    label=f"{sample_id}.edit_visibility_evidence",
                )
        if row["status"] == "error":
            replay_by_id[sample_id] = {
                "status": "error",
                "raw_logit": forward.raw_logit,
            }
            continue
        _compare_nested(
            row.get("preprocess"),
            prepared.audit,
            label=f"{sample_id}.preprocess",
        )
        if "preprocess_profile" in row:
            _require_equal(
                row["preprocess_profile"],
                FROZEN_PROFILE,
                f"{sample_id}.preprocess_profile",
            )
        features, artifact_logits, artifact_path = _load_artifact(
            row,
            repo_root=repo_root,
            run_dir=files.run_dir,
        )
        if artifact_path in artifact_paths:
            raise ValueError("two B-Free rows reuse one artifact path")
        artifact_paths.add(artifact_path)
        feature_diff = _maximum_absolute_difference(
            features, forward.features
        )
        crop_diff = _maximum_absolute_difference(
            artifact_logits, forward.crop_logits
        )
        maximum_feature = max(maximum_feature, feature_diff)
        maximum_crop_logit = max(maximum_crop_logit, crop_diff)
        if feature_diff > FEATURE_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"{sample_id} feature artifact differs from fresh full forward"
            )
        if crop_diff > RAW_LOGIT_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"{sample_id} crop-logit artifact differs from fresh forward"
            )
        head_logits, head_mean = _replay_artifact_head(
            features, model, runtime
        )
        head_diff = _maximum_absolute_difference(
            head_logits, artifact_logits
        )
        maximum_artifact_head_logit = max(
            maximum_artifact_head_logit, head_diff
        )
        if head_diff > RAW_LOGIT_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"{sample_id} artifact head replay differs from crop logits"
            )
        score = _audit_score_fields(
            row,
            replay_crop_logits=forward.crop_logits,
            replay_raw_logit=forward.raw_logit,
        )
        _compare_float(
            head_mean,
            forward.raw_logit,
            label=f"{sample_id} artifact head mean",
            tolerance=RAW_LOGIT_ABSOLUTE_TOLERANCE,
        )
        maximum_raw = max(
            maximum_raw,
            abs(float(row["raw_logit"]) - forward.raw_logit),
        )
        replay_by_id[sample_id] = {
            "status": "ok",
            **score,
            "feature_array_sha256": _array_sha256(forward.features),
            "preprocess": prepared.audit,
        }

    independent_rows: list[dict[str, Any]] = []
    for row in files.rows:
        copied = dict(row)
        replay = replay_by_id[str(row["id"])]
        if row["status"] == "ok":
            copied.update(
                {
                    "raw_logit": replay["raw_logit"],
                    "ai_score": replay["raw_logit"],
                    "score": replay["raw_logit"],
                    "crop_logits": replay["crop_logits"],
                    "classification_decision": replay["decision"],
                }
            )
        independent_rows.append(copied)

    return {
        "selected_images_freshly_reopened": len(all_inputs),
        "selected_images_freshly_preprocessed": len(all_inputs),
        "complete_model_forward_passes": len(all_inputs),
        "successful_artifacts_validated": len(artifact_paths),
        "feature_artifacts_validated": len(artifact_paths),
        "crop_logit_artifacts_validated": len(artifact_paths),
        "artifact_head_replays": len(artifact_paths),
        "maximum_feature_absolute_difference": maximum_feature,
        "maximum_crop_logit_absolute_difference": maximum_crop_logit,
        "maximum_raw_logit_absolute_difference": maximum_raw,
        "maximum_artifact_head_logit_absolute_difference": (
            maximum_artifact_head_logit
        ),
        "wrap_images": wrap_images,
        "strict_decision_replayed": "raw_logit > 0",
        "scope": {
            "T1_whole_image_AIGC_detection": True,
            "T2_localization": False,
            "S_joint": False,
        },
        "physically_independent_replay": {
            "runner_scores_trusted_as_input": False,
            "runner_features_trusted_as_input": False,
            "fresh_image_decode_per_selected_image": True,
            "fresh_preprocess_per_selected_image": True,
            "fresh_complete_model_forward_per_selected_image": True,
            "persisted_feature_head_replay": True,
            "all_selected_images_replayed": True,
        },
        "_independent_result_rows": independent_rows,
    }


def _strip_summary_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "run_id",
        "model",
        "model_slug",
        "checkpoint_id",
        "preprocess_profile",
        "config_fingerprint",
        "official_golden_status",
        "official_golden_fingerprint",
        "generated_at",
        "completed_at",
    }
    return {key: nested for key, nested in value.items() if key not in metadata}


def recompute_summary(
    *,
    result_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    recorded_summary: Mapping[str, Any],
    independent_result_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = _require_mapping(manifest.get("config"), "manifest.config")
    metrics = _require_mapping(config.get("metrics"), "manifest.config.metrics")
    iterations = int(
        metrics.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)
    )
    seed = int(metrics.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    _compare_float(
        metrics.get("fixed_threshold", FIXED_THRESHOLD),
        FIXED_THRESHOLD,
        label="metrics fixed threshold",
        tolerance=0.0,
    )
    _require_equal(
        metrics.get("threshold_operator", THRESHOLD_OPERATOR),
        THRESHOLD_OPERATOR,
        "metrics threshold operator",
    )
    recomputed = summarize_bfree_results(
        result_rows,
        expected_rows,
        threshold=FIXED_THRESHOLD,
        bootstrap_samples=iterations,
        seed=seed,
    )
    _compare_nested(
        _strip_summary_metadata(recorded_summary),
        recomputed,
        label="summary",
    )
    independent_match: bool | None = None
    maximum_independent_score_difference: float | None = None
    if independent_result_rows is not None:
        independent = summarize_bfree_results(
            independent_result_rows,
            expected_rows,
            threshold=FIXED_THRESHOLD,
            bootstrap_samples=iterations,
            seed=seed,
        )
        _compare_nested(
            independent,
            recomputed,
            label="independent_full_model_summary",
            float_tolerance=RAW_LOGIT_ABSOLUTE_TOLERANCE,
        )
        original_latest = _latest_by_id(result_rows)
        independent_latest = _latest_by_id(independent_result_rows)
        differences = [
            abs(
                float(original_latest[sample_id]["ai_score"])
                - float(independent_latest[sample_id]["ai_score"])
            )
            for sample_id in original_latest
            if original_latest[sample_id]["status"] == "ok"
        ]
        maximum_independent_score_difference = max(differences, default=0.0)
        independent_match = True
    return {
        "recorded_summary_exactly_recomputed": True,
        "independent_full_model_summary_within_float_tolerance": (
            independent_match
        ),
        "independent_full_model_summary_float_tolerance": (
            RAW_LOGIT_ABSOLUTE_TOLERANCE
        ),
        "maximum_independent_score_absolute_difference": (
            maximum_independent_score_difference
        ),
        "bootstrap_samples": iterations,
        "bootstrap_seed": seed,
        "recomputed": recomputed,
    }


def _validate_adapter_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    task_scope = _require_mapping(
        manifest.get("task_scope"), "manifest.task_scope"
    )
    if task_scope.get("valid_for_t1") is not True:
        raise ValueError("B-Free manifest does not declare valid T1")
    if task_scope.get("valid_for_t2") is not False:
        raise ValueError("B-Free manifest must declare T2 invalid")
    config = _require_mapping(manifest.get("config"), "manifest.config")
    contract = _require_mapping(
        config.get("adapter_contract"),
        "manifest.config.adapter_contract",
    )
    checks = {
        "feature_dimension": FEATURE_DIMENSION,
        "crop_count": CROPS,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "score_semantics": SCORE_SEMANTICS,
        "feature_semantics": FEATURE_SEMANTICS,
    }
    for key, expected in checks.items():
        if key in contract:
            _require_equal(contract[key], expected, f"adapter contract {key}")
    _reject_t2_localization_or_joint(
        {"task_scope": task_scope, "adapter_contract": contract},
        label="manifest capability",
    )
    return {
        "valid_for_t1": True,
        "valid_for_t2": False,
        "joint_valid": False,
        "score": SCORE_SEMANTICS,
        "decision": "raw_logit > 0",
    }


def _visibility_census(
    rows: list[dict[str, Any]],
    *,
    pair_visibility: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    domains = {
        str(row["task_id"]): str(row["domain"])
        for row in rows
    }
    category = Counter(
        str(value["edit_visibility"]) for value in pair_visibility.values()
    )
    fractions = [
        float(value["edit_visible_gt_fraction"])
        for value in pair_visibility.values()
    ]
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    wrap_category: Counter[str] = Counter()
    distinct_starts: Counter[int] = Counter()
    wrap_pairs = 0
    for task_id, value in pair_visibility.items():
        visibility = str(value["edit_visibility"])
        by_domain[domains[task_id]][visibility] += 1
        distinct_starts[int(value["distinct_crop_starts"])] += 1
        if bool(value["geometry"]["replicate_wrap_applied"]):
            wrap_pairs += 1
            wrap_category[visibility] += 1
    return {
        "pairs": len(pair_visibility),
        "edit_visibility": {
            name: int(category.get(name, 0))
            for name in ("full", "partial", "none")
        },
        "mean_edit_visible_gt_fraction": (
            float(np.mean(fractions)) if fractions else None
        ),
        "wrap_pairs": wrap_pairs,
        "wrap_edit_visibility": {
            name: int(wrap_category.get(name, 0))
            for name in ("full", "partial", "none")
        },
        "distinct_crop_starts": dict(sorted(distinct_starts.items())),
        "by_domain": {
            domain: {
                name: int(counter.get(name, 0))
                for name in ("full", "partial", "none")
            }
            for domain, counter in sorted(by_domain.items())
        },
    }


def _normalize_recorded_visibility_census(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the runner's release-facing census to the frozen pin."""

    recorded = dict(value)
    distinct_raw = recorded.get(
        "distinct_crop_starts",
        recorded.get("distinct_crop_starts_census", {}),
    )
    distinct_mapping = _require_mapping(
        distinct_raw, "recorded distinct crop starts"
    )
    return {
        "pairs": recorded.get("pairs"),
        "edit_visibility": recorded.get(
            "edit_visibility", recorded.get("census")
        ),
        "mean_edit_visible_gt_fraction": recorded.get(
            "mean_edit_visible_gt_fraction"
        ),
        "wrap_pairs": recorded.get(
            "wrap_pairs", recorded.get("wrapped_pairs")
        ),
        "wrap_edit_visibility": recorded.get(
            "wrap_edit_visibility",
            recorded.get("wrapped_visibility_census"),
        ),
        "distinct_crop_starts": {
            int(key): int(count)
            for key, count in distinct_mapping.items()
        },
        "by_domain": recorded.get(
            "by_domain", recorded.get("domain_census")
        ),
    }


def validate_provenance(
    *,
    repo_root: Path,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
    files: RunFiles,
    runtime: ReplayRuntime,
) -> tuple[dict[str, Any], Any]:
    pins = _validate_runner_pins()
    manifest_identity = _validate_run_manifest_identity(
        repo_root=repo_root,
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        files=files,
    )
    source = _verify_source_tree(source_root)
    assets, state = _verify_assets(
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        torch_module=runtime.torch,
    )
    model, load = _build_and_load_model(
        source_root=source_root,
        state=state,
        torch_module=runtime.torch,
    )
    selection = _validate_selection(repo_root=repo_root, files=files)
    release = _validate_frozen_release(repo_root=repo_root, files=files)
    adapter = _validate_adapter_contract(files.manifest)
    pair_visibility = _pair_visibility(files.expected, repo_root=repo_root)
    census = _visibility_census(
        files.expected, pair_visibility=pair_visibility
    )
    if release["full_frozen_release"]:
        _compare_nested(
            census,
            FROZEN_VISIBILITY_CENSUS,
            label="full frozen B-Free visibility census",
            float_tolerance=1e-15,
        )
    recorded_census = files.manifest.get("full_dataset_visibility_audit")
    if isinstance(recorded_census, Mapping) and release["full_frozen_release"]:
        _compare_nested(
            _normalize_recorded_visibility_census(recorded_census),
            FROZEN_VISIBILITY_CENSUS,
            label="recorded full visibility census",
            float_tolerance=1e-15,
        )
    return (
        {
            "runner_pins": pins,
            "run_manifest_identity": manifest_identity,
            "source": source,
            "assets": assets,
            "model_load": load,
            "runtime": runtime.evidence,
            "selection": selection,
            "canonical_release": release,
            "adapter_contract": adapter,
            "visibility_census": census,
        },
        model,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root).resolve()
    source_root = Path(args.source_root).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    weights_zip = Path(args.weights_zip).resolve()
    files = _load_run_files(results_dir=results_dir, run_id=args.run_id)
    _reject_t2_localization_or_joint(files.manifest, label="manifest")
    _reject_t2_localization_or_joint(files.summary, label="summary")
    _reject_t2_localization_or_joint(files.rows, label="results")
    runtime = _replay_runtime(
        files.manifest,
        requested_device=args.device,
    )
    provenance, model = validate_provenance(
        repo_root=repo_root,
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        files=files,
        runtime=runtime,
    )
    golden = _audit_official_golden(
        source_root=source_root,
        model=model.to(runtime.device).eval(),
        runtime=runtime,
        recorded=_require_mapping(
            files.manifest.get("official_golden"),
            "manifest.official_golden",
        ),
    )
    artifacts = audit_artifacts(
        repo_root=repo_root,
        all_inputs=files.expected,
        files=files,
        runtime=runtime,
        model=model,
    )
    independent_rows = artifacts.pop("_independent_result_rows")
    summary = recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=files.manifest,
        recorded_summary=files.summary,
        independent_result_rows=independent_rows,
    )
    result = {
        "schema_version": "bfree_independent_audit_v1",
        "status": "ok",
        "run_id": args.run_id,
        "audited_at": utc_now(),
        "scope": {
            "T1_whole_image_AIGC_detection": True,
            "T2_localization": False,
            "S_joint": False,
        },
        "provenance": provenance,
        "official_golden": golden,
        "artifact_replay": artifacts,
        "summary_replay": summary,
        "files": {
            "results": {
                "path": _relative_or_absolute(
                    files.results_path, repo_root
                ),
                "sha256": sha256_file(files.results_path),
                "bytes": files.results_path.stat().st_size,
            },
            "expected_inputs": {
                "path": _relative_or_absolute(
                    files.expected_path, repo_root
                ),
                "sha256": sha256_file(files.expected_path),
                "bytes": files.expected_path.stat().st_size,
            },
            "summary": {
                "path": _relative_or_absolute(
                    files.summary_path, repo_root
                ),
                "sha256": sha256_file(files.summary_path),
                "bytes": files.summary_path.stat().st_size,
            },
            "manifest": {
                "path": _relative_or_absolute(
                    files.manifest_path, repo_root
                ),
                "sha256": sha256_file(files.manifest_path),
                "bytes": files.manifest_path.stat().st_size,
            },
        },
    }
    output_path = (
        Path(args.output).resolve()
        if args.output
        else files.run_dir / "independent_audit.json"
    )
    try:
        output_path.relative_to(files.run_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "independent audit output must remain inside the run directory"
        ) from exc
    atomic_write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a frozen B-Free run"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--weights-zip", default=str(DEFAULT_WEIGHTS_ZIP))
    parser.add_argument(
        "--device",
        default=None,
        help="audit device; default reuses the recorded run device",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
