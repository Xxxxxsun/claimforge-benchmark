#!/usr/bin/env python3
"""Independently audit a frozen SPAI whole-image detection run.

The runner's JSON and NPY files are treated as untrusted evidence.  This
analyzer verifies the pinned official source and checkpoint, independently
decodes every selected image with Pillow, reconstructs SPAI's native-resolution
224-pixel patch geometry, and performs a fresh complete
FFT -> ViT -> frequency-restoration -> spectral-context-attention -> MLP
forward pass.

Three float32 artifacts are checked for every successful image:

* per-patch frequency-restoration features, shape ``[P, 1096]``;
* the layer-normalized spectral-context feature, shape ``[1096]``; and
* the twelve-head attention weights, shape ``[12, P]``.

The patch features are also replayed through SCA, LayerNorm, and the complete
released MLP head.  The normalized feature is replayed through the complete
MLP head independently.  Logit, float32 sigmoid probability, aliases, and the
strict ``probability > 0.5`` decision must agree.

SPAI is an image-level detector.  Its SCA attention is retained solely as
classifier replay evidence; this analyzer rejects T2, localization, dense-mask,
pixel-metric, and S_joint claims recursively.
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
import os
import platform
import random
import re
import socket
import subprocess
import sys
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

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
from eval.opensource.spai_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_spai_results,
)


DEFAULT_RESULTS_DIR = Path("results/opensource/spai")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_RUN_ID = (
    "spai_any_resolution_spectral_mouse_canonical_v1_full275_20260725"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/spai-8ff7b3b6"
)
DEFAULT_CHECKPOINT = Path("/root/.cache/claimforge/third_party/spai.pth")

FROZEN_SOURCE_COMMIT = "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
FROZEN_CHECKPOINT_SHA256 = (
    "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55"
)
FROZEN_CHECKPOINT_BYTES = 934_865_338
FROZEN_MODEL_STATE_ENTRIES = 324
FROZEN_MODEL_STATE_ELEMENTS = 139_945_243
FROZEN_UNSAFE_GLOBALS = ["yacs.config.CfgNode"]
FROZEN_CONFIG_RELATIVE = "configs/spai.yaml"
FROZEN_CONFIG_SHA256 = (
    "66a3caee07d9af23de453eb604709d4a821aab41a6acfb983a5c9be9d05f3586"
)
FROZEN_SOURCE_FILES = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "README.md": "2de3bf7b8d10fbe9d2990e3a391656d6aaeae599edef0d53589f68f231e1b30c",
    "requirements.txt": (
        "d449bf03fc2e3b0cce3b9985df0ae491f10afb22c4dc948595f9fdd88ff103c4"
    ),
    "configs/spai.yaml": FROZEN_CONFIG_SHA256,
    "spai/__init__.py": (
        "c8cbcf96bb07ad2c517efbea09b7a9fc07f7cb143082caa5b2d3f3f41f3b7b94"
    ),
    "spai/config.py": (
        "7ebda3d1862985870f2d70272dc80933af3ceb66b6df85f2f4d0a901299cb893"
    ),
    "spai/utils.py": (
        "b5eea98152239e60e96ea8cf8a416fb27ce06c9cf9f1ddba61da3a3e79c0f770"
    ),
    "spai/models/__init__.py": (
        "13abb55acc1e51dea398fc097aeb1ed348c8ba8f5f7fc5c5a71677bea4c4f538"
    ),
    "spai/models/build.py": (
        "d89eaade77bcb58d9b0de08ddbd0c7eec94b54dc08958cc169b3a06e47faddcd"
    ),
    "spai/models/filters.py": (
        "0d8006e6b8af445bac7a9edb9c48f69216b92b2cdbd5a2b94dea578666f17c4c"
    ),
    "spai/models/sid.py": (
        "f748716a9d6223d36e5d60966193d5c46b1b26da0d475610b75bfc7ff9192c64"
    ),
    "spai/models/utils.py": (
        "5c9940ecfb5002b3d0b1655f413987ac1a13214394657533719d47193227a507"
    ),
    "spai/models/vision_transformer.py": (
        "1ff53dbf64d9909b852c4727eebf7711265f2d8cdb644801a9ec3ff611c3acc5"
    ),
    "spai/models/backbones.py": (
        "5d048da7d6a43d9a4236f35e2d6746bc29457cc561943fb565764321c63c315d"
    ),
    "spai/models/swin_transformer.py": (
        "b3bcc34ce1ab685d1dcf39cb61d5b76d280bb04a9639dc497c217c1db1b85388"
    ),
    "spai/models/mfm.py": (
        "795933cf358ac0daf3196a20746a646a43005c1a3e4a1ff2a8b58757df46d5be"
    ),
    "spai/models/frequency_loss.py": (
        "6edc01f7cebbfd18fe0a29d50aa0952becb3aa6f53fe733e39498ba7d950db80"
    ),
    "spai/data/__init__.py": (
        "5cf52429272b45a40052857853656787ec67bec7acf26b41bb880473666ea0a9"
    ),
    "spai/data/data_finetune.py": (
        "9b90a95414ec697fb7e9ef465e160624913e16854da679477fa4b440a85d4791"
    ),
    "spai/data/readers.py": (
        "11cbc79598e5d8d96686f3b2e80cbdec10cf0310c124b9827a88b542781b02e9"
    ),
}

FROZEN_RUNTIME_VERSIONS = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "timm": "0.4.12",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "yacs": "0.1.8",
    "einops": "0.8.1",
    "opencv-python-headless": "4.10.0.84",
    "albumentations": "1.4.14",
}
FROZEN_RUNTIME_MODULE_FILES = {
    "torch": "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d",
    "torchvision": "ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73",
    "timm": "f664ef352d89e92a0c681ad812fe9772673d106332b6e1709098146025d202b8",
    "numpy": "22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e",
    "PIL": "7c95303c6848f3f99c07c8cd583fa1530ecc88c2725a0a955ff9c5b73223d59b",
    "yacs": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "einops": "acacaf13ae1b60c38c5c01b811f30c4951b1805674c79d5db7cda946cc389471",
    "cv2": "936bd94c5a5debf0212fc751af79d3a163652f3e850259df2159db6aa3ed8ad8",
    "albumentations": "881857517838ba0e9a8fc90f4ccf224863c5e019685bfac8914fa6608df9a8cf",
}

PREPROCESS_PROFILE = "official_pillow_rgb_native_float32_0_1"
SCORE_SEMANTICS = "torch_float32_sigmoid_of_single_raw_logit"
T1_POLICY = "released_probability_strictly_greater_than_0_5"
PATCH_FEATURE_SEMANTICS = (
    "per_patch_frequency_restoration_features_before_spectral_context_attention"
)
FEATURE_SEMANTICS = (
    "spectral_context_attention_layernorm_output_before_complete_mlp_head"
)
ATTENTION_SEMANTICS = (
    "spectral_context_attention_softmax_weights_classifier_diagnostic_not_localization"
)

PATCH_SIZE = 224
PATCH_STRIDE = 224
MINIMUM_PATCHES = 4
FIVE_CROP_PATCHES = 5
FEATURE_EXTRACTION_BATCH = 400
FEATURE_DIMENSION = 1096
ATTENTION_HEADS = 12
PRIMARY_DEVICE = "cuda:0"
FEATURE_DTYPE = np.dtype("float32")

PATCH_FEATURE_ABSOLUTE_TOLERANCE = 1e-5
FEATURE_ABSOLUTE_TOLERANCE = 1e-5
ATTENTION_ABSOLUTE_TOLERANCE = 1e-6
RAW_LOGIT_ABSOLUTE_TOLERANCE = 1e-5
PROBABILITY_ABSOLUTE_TOLERANCE = 1e-7

FROZEN_MOUSE_VISIBILITY_CENSUS = {
    "none": 18,
    "partial": 14,
    "full": 243,
}
FROZEN_MOUSE_PATCH_MODE_CENSUS = {
    "grid": 262,
    "five_crop": 13,
}

# Independent copy of the executable regression-golden contract.  Do not
# derive these values from the run manifest (or by calling a runner helper):
# the manifest is evidence under audit, not an authority.
FROZEN_GOLDEN_CASES = (
    {
        "relative_path": "midjourney-v6.1/224.png",
        "sha256": (
            "e41a6f0832d363a110f6821a1c6e2120b1f0187345bf652e2fc60125a8c4ea2b"
        ),
        "decoded_rgb_sha256": (
            "71446c3a9d8dbc9450f1071054b44ec092e0c8d51860a0e981114f0bb17c231b"
        ),
        "tensor_sha256": (
            "8b36e5a6b03c646975313b4d5f74d5f04ff6c38d379400aba4a897f3a65815ad"
        ),
        "native_size": [1232, 928],
        "patch_count": 20,
        "raw_logit": 0.9909347295761108,
        "probability": 0.7292724847793579,
        "website_derivative_display": 0.748,
    },
    {
        "relative_path": (
            "stable-diffusion-3/cfg_60/euler/steps_28/000001046_4.webp"
        ),
        "sha256": (
            "bf4c5d6e784346c4dd49adf08c6223f6da93080ca1782f678b29d7cdfed8b386"
        ),
        "decoded_rgb_sha256": (
            "f13b1af78b0dd81c9d9db475ada2863bd1be67da962138218b05f68468ec562a"
        ),
        "tensor_sha256": (
            "d68f370b977c35fed84656c49bd867515d8978fcecd9241035d11ffc950aae7b"
        ),
        "native_size": [1024, 1024],
        "patch_count": 16,
        "raw_logit": 1.6814128160476685,
        "probability": 0.8430914878845215,
        "website_derivative_display": 0.87,
    },
)
FROZEN_GOLDEN_LOGIT_ABSOLUTE_TOLERANCE = 1e-6
FROZEN_GOLDEN_PROBABILITY_ABSOLUTE_TOLERANCE = 1e-7
FROZEN_GOLDEN_SOURCE = (
    "official evaluation-bundle originals, current released source/"
    "checkpoint, pinned highest/no-TF32 float32 runtime"
)
FROZEN_GOLDEN_WEBSITE_DISPLAY_REFERENCE = (
    "official website uses compressed derivatives and displays "
    "0.748/0.87; those values do not match the released executable "
    "regression and are disclosed rather than used as a gate"
)

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
        "attention_map",
        "attention_map_path",
        "attention_mask",
        "attention_mask_path",
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
    "preprocess",
    "edit_visibility",
    "edit_visible_gt_fraction",
    "edit_visibility_evidence",
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
    "valid_for_metrics",
    "valid_for_t2",
    "attention_is_diagnostic_not_T2",
    "spai_patch_features_sha256",
    "spai_patch_features_array_sha256",
    "spai_patch_features_shape",
    "spai_patch_features_dtype",
    "spai_patch_features_semantics",
    "spai_feature_sha256",
    "spai_feature_array_sha256",
    "feature_array_sha256",
    "spai_feature_shape",
    "spai_feature_dtype",
    "spai_feature_semantics",
    "spai_attention_sha256",
    "spai_attention_array_sha256",
    "spai_attention_shape",
    "spai_attention_dtype",
    "spai_attention_semantics",
)


@dataclass(frozen=True)
class PreprocessedImage:
    """Independent CPU preprocessing and patch-geometry evidence."""

    tensor: Any
    decoded_rgb: np.ndarray
    decoded_rgb_sha256: str
    tensor_sha256: str
    geometry: dict[str, Any]
    audit: dict[str, Any]


@dataclass(frozen=True)
class ReplayRuntime:
    """Torch/device pair bound to the recorded numerical runtime."""

    torch: ModuleType
    device: Any
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RunFiles:
    """Physical files belonging to one SPAI run directory."""

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
    """Tensors produced by a fresh complete SPAI forward."""

    patch_features: Any
    normalized_feature: Any
    attention: Any
    raw_logit: Any


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


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_pin(module: ModuleType, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return copy.deepcopy(getattr(module, name))
    raise RuntimeError(
        "SPAI runner lacks required audit pin " + " or ".join(names)
    )


def _optional_module_pin(
    module: ModuleType,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if hasattr(module, name):
            return copy.deepcopy(getattr(module, name))
    return copy.deepcopy(default)


def _load_runner_pins() -> SimpleNamespace:
    """Import only frozen constants from the runner."""

    from eval.opensource import run_spai as runner

    return SimpleNamespace(
        MODEL_NAME=_module_pin(runner, "MODEL_NAME"),
        MODEL_SLUG=_module_pin(runner, "MODEL_SLUG"),
        MODEL_REPO_URL=_module_pin(runner, "MODEL_REPO_URL"),
        MODEL_SOURCE_COMMIT=_module_pin(
            runner,
            "MODEL_SOURCE_COMMIT",
            "SOURCE_COMMIT",
        ),
        PAPER_URL=_module_pin(runner, "PAPER_URL"),
        PREPROCESS_PROFILE=_module_pin(runner, "PREPROCESS_PROFILE"),
        MODEL_SEED=int(_module_pin(runner, "MODEL_SEED")),
        CLASSIFICATION_THRESHOLD=float(
            _module_pin(runner, "CLASSIFICATION_THRESHOLD")
        ),
        CLASSIFICATION_THRESHOLD_OPERATOR=str(
            _module_pin(runner, "CLASSIFICATION_THRESHOLD_OPERATOR")
        ),
        PATCH_SIZE=int(_module_pin(runner, "PATCH_SIZE")),
        PATCH_STRIDE=int(_module_pin(runner, "PATCH_STRIDE")),
        MINIMUM_PATCHES=int(_module_pin(runner, "MINIMUM_PATCHES")),
        FEATURE_EXTRACTION_BATCH=int(
            _module_pin(
                runner,
                "FEATURE_EXTRACTION_BATCH",
                "FEATURE_EXTRACTION_BATCH_SIZE",
            )
        ),
        FEATURE_DIMENSION=int(_module_pin(runner, "FEATURE_DIMENSION")),
        ATTENTION_HEADS=int(_module_pin(runner, "ATTENTION_HEADS")),
        PRIMARY_DEVICE=str(
            _optional_module_pin(
                runner,
                "PRIMARY_DEVICE",
                default=PRIMARY_DEVICE,
            )
        ),
        SOURCE_FILES=_module_pin(runner, "SOURCE_FILES"),
        LICENSE_RECORD=_module_pin(runner, "LICENSE_RECORD"),
        CHECKPOINT=_module_pin(runner, "CHECKPOINT"),
        CONFIG_SHA256=_optional_module_pin(
            runner,
            "CONFIG_SHA256",
            default=FROZEN_CONFIG_SHA256,
        ),
        CHECKPOINT_UNSAFE_GLOBALS=_optional_module_pin(
            runner,
            "CHECKPOINT_UNSAFE_GLOBALS",
            default=FROZEN_UNSAFE_GLOBALS,
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
        value = _require_finite(actual, label)
        if not math.isclose(
            value,
            expected,
            rel_tol=0.0,
            abs_tol=float_tolerance,
        ):
            raise ValueError(f"{label} mismatch: {value!r} != {expected!r}")
        return
    _require_equal(actual, expected, label)


def _reject_t2_localization_or_joint(value: Any, *, label: str) -> None:
    """Reject output classes that SPAI does not natively provide."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            if key == "valid_for_t2":
                if nested is not False:
                    raise ValueError(
                        f"{label} contains unsupported T2/localization/S_joint "
                        f"claim {raw_key!r}={nested!r}"
                    )
                continue
            if key == "attention_is_diagnostic_not_t2":
                if nested is not True:
                    raise ValueError(
                        f"{label} misrepresents SPAI classifier attention as "
                        f"localization evidence via {raw_key!r}={nested!r}"
                    )
                continue
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


def _center_crop_start(length: int, size: int = PATCH_SIZE) -> int:
    """Match torchvision.functional.center_crop's Python-round geometry."""

    if length < size:
        raise ValueError("SPAI five-crop input is smaller than its patch size")
    return int(round((length - size) / 2.0))


def five_crop_boxes(
    width: int,
    height: int,
    size: int = PATCH_SIZE,
) -> list[list[int]]:
    """Return torchvision five-crop boxes in its exact output order."""

    if width < size or height < size:
        raise ValueError("SPAI five-crop input is smaller than its patch size")
    left = _center_crop_start(width, size)
    top = _center_crop_start(height, size)
    return [
        [0, 0, size, size],
        [width - size, 0, width, size],
        [0, height - size, size, height],
        [width - size, height - size, width, height],
        [left, top, left + size, top + size],
    ]


def compute_patch_geometry(
    width: int,
    height: int,
    *,
    patch_size: int = PATCH_SIZE,
    patch_stride: int = PATCH_STRIDE,
    minimum_patches: int = MINIMUM_PATCHES,
) -> dict[str, Any]:
    """Reconstruct official unfold/five-crop selection without model code."""

    if width <= 0 or height <= 0:
        raise ValueError("SPAI native dimensions must be positive")
    pad_left = int((max(patch_size, width) - width) / 2.0)
    pad_right = max(patch_size, width) - width - pad_left
    pad_top = int((max(patch_size, height) - height) / 2.0)
    pad_bottom = max(patch_size, height) - height - pad_top
    processed_width = width + pad_left + pad_right
    processed_height = height + pad_top + pad_bottom
    grid_columns = 1 + (processed_width - patch_size) // patch_stride
    grid_rows = 1 + (processed_height - patch_size) // patch_stride
    initial_count = grid_columns * grid_rows
    covered_width = (grid_columns - 1) * patch_stride + patch_size
    covered_height = (grid_rows - 1) * patch_stride + patch_size
    common = {
        "profile_id": PREPROCESS_PROFILE,
        "native_size": [width, height],
        "model_input_size": [processed_width, processed_height],
        "pad_if_needed": {
            "enabled": any((pad_left, pad_right, pad_top, pad_bottom)),
            "minimum_size": [patch_size, patch_size],
            "left": pad_left,
            "right": pad_right,
            "top": pad_top,
            "bottom": pad_bottom,
            "position": "center",
            "border_mode": "cv2.BORDER_REFLECT_101",
            "border_mode_value": 4,
        },
        "patch_size": [patch_size, patch_size],
        "patch_stride": [patch_stride, patch_stride],
        "minimum_patches": minimum_patches,
        "initial_grid": {
            "rows": grid_rows,
            "columns": grid_columns,
            "count": initial_count,
        },
    }
    if initial_count < minimum_patches:
        boxes = five_crop_boxes(processed_width, processed_height, patch_size)
        return {
            **common,
            "patch_mode": "five_crop",
            "effective_patch_count": len(boxes),
            "grid_covered_xyxy": None,
            "five_crop_boxes_xyxy": boxes,
            "remainder_policy": (
                "torchvision_five_crop_when_initial_grid_has_fewer_than_four"
            ),
        }
    return {
        **common,
        "patch_mode": "grid",
        "effective_patch_count": initial_count,
        "grid_covered_xyxy": [0, 0, covered_width, covered_height],
        "five_crop_boxes_xyxy": None,
        "remainder_policy": (
            "torch_tensor_unfold_discards_nondivisible_right_bottom"
        ),
    }


def _pad_rgb_to_patch(
    decoded: np.ndarray,
    *,
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, list[int]]:
    """Match centered OpenCV ``BORDER_REFLECT_101`` PadIfNeeded."""

    height, width = decoded.shape[:2]
    missing_width = max(0, patch_size - width)
    missing_height = max(0, patch_size - height)
    left = missing_width // 2
    right = missing_width - left
    top = missing_height // 2
    bottom = missing_height - top
    if not any((left, top, right, bottom)):
        return decoded, [0, 0, 0, 0]
    padded = np.pad(
        decoded,
        ((top, bottom), (left, right), (0, 0)),
        # numpy ``reflect`` excludes the edge pixel, matching OpenCV's
        # BORDER_REFLECT_101 used by the frozen Albumentations release.
        mode="reflect",
    )
    return np.ascontiguousarray(padded), [left, top, right, bottom]


def preprocess_image(
    path: Path,
    *,
    torch_module: ModuleType,
    profile_id: str = PREPROCESS_PROFILE,
) -> PreprocessedImage:
    """Independently implement Pillow RGB, native pad, and uint8/255."""

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        decoded = np.asarray(rgb, dtype=np.uint8).copy()
        decoded_width, decoded_height = rgb.size
    padded, padding = _pad_rgb_to_patch(decoded)
    height, width = padded.shape[:2]
    channel_first = np.ascontiguousarray(padded.transpose(2, 0, 1))
    # Albumentations Normalize first casts uint8 to float32 and then
    # multiplies by a float32 reciprocal.  Division is not bit-identical.
    tensor = (
        torch_module.from_numpy(channel_first)
        .to(dtype=torch_module.float32)
        .mul(np.float32(1.0 / 255.0))
        .contiguous()
    )
    geometry = compute_patch_geometry(decoded_width, decoded_height)
    audit = {
        "profile": profile_id,
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "resize": False,
        "crop": False,
        "test_augmentation": False,
        "native_size": [decoded_width, decoded_height],
        "model_input_size": [width, height],
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "padded_rgb_shape": list(padded.shape),
        "padded_rgb_dtype": str(padded.dtype),
        "padded_rgb_sha256": _array_sha256(padded),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.detach().cpu().numpy().dtype),
        "tensor_sha256": _tensor_sha256(tensor),
        "scale": (
            "uint8_to_float32_multiply_float32_reciprocal_1_over_255_"
            "matching_albumentations_1_4_14_normalize_lut"
        ),
        "normalization": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        "value_min": float(tensor.min().item()),
        "value_max": float(tensor.max().item()),
        "geometry": geometry,
    }
    return PreprocessedImage(
        tensor=tensor,
        decoded_rgb=decoded,
        decoded_rgb_sha256=audit["decoded_rgb_sha256"],
        tensor_sha256=audit["tensor_sha256"],
        geometry=geometry,
        audit=audit,
    )


def _patchify_tensor(
    tensor: Any,
    geometry: Mapping[str, Any],
    *,
    torch_module: ModuleType,
) -> Any:
    """Apply official patch selection independently of SPAI utilities."""

    if tensor.ndim != 3:
        raise ValueError("SPAI preprocessed tensor must be CHW")
    if geometry.get("patch_mode") == "grid":
        patches = (
            tensor.unsqueeze(0)
            .unfold(2, PATCH_SIZE, PATCH_STRIDE)
            .unfold(3, PATCH_SIZE, PATCH_STRIDE)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        patches = patches.view(-1, tensor.shape[0], PATCH_SIZE, PATCH_SIZE)
    elif geometry.get("patch_mode") == "five_crop":
        patches = torch_module.stack(
            [
                tensor[:, box[1] : box[3], box[0] : box[2]]
                for box in geometry["five_crop_boxes_xyxy"]
            ],
            dim=0,
        ).contiguous()
    else:
        raise ValueError(f"unknown SPAI patch mode: {geometry.get('patch_mode')}")
    _require_equal(
        int(patches.shape[0]),
        int(geometry["effective_patch_count"]),
        "SPAI effective patch count",
    )
    _require_equal(
        tuple(patches.shape[1:]),
        (3, PATCH_SIZE, PATCH_SIZE),
        "SPAI patch shape",
    )
    return patches


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
    """Intersect exact GT with the union of classifier-visible patches."""

    mask = _load_gt_mask(canonical, repo_root=repo_root)
    if mask is None:
        raise ValueError("SPAI visibility must be derived from the forged row")
    width = int(canonical["width"])
    height = int(canonical["height"])
    geometry = compute_patch_geometry(width, height)
    positive_y, positive_x = np.nonzero(mask == 255)
    total = int(positive_x.size)
    padding = geometry["pad_if_needed"]
    positive_x = positive_x + int(padding["left"])
    positive_y = positive_y + int(padding["top"])
    if geometry["patch_mode"] == "grid":
        _, _, right, bottom = geometry["grid_covered_xyxy"]
        visible_mask = (positive_x < right) & (positive_y < bottom)
    else:
        visible_mask = np.zeros(total, dtype=bool)
        for left, top, right, bottom in geometry["five_crop_boxes_xyxy"]:
            visible_mask |= (
                (positive_x >= left)
                & (positive_x < right)
                & (positive_y >= top)
                & (positive_y < bottom)
            )
    visible = int(np.count_nonzero(visible_mask))
    fraction = visible / total
    category = "none" if visible == 0 else "full" if visible == total else "partial"
    return {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixels": visible,
        "forged_sample_id": str(canonical["sample_id"]),
        "basis": (
            "exact_diff_positive_pixels_in_union_of_official_native_"
            "resolution_patch_receptive_fields"
        ),
        "geometry": geometry,
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
    result: dict[str, dict[str, Any]] = {}
    for task_id, row in forged.items():
        evidence = _visibility_from_exact_gt(row, repo_root=repo_root)
        result[task_id] = {
            "domain": str(row["domain"]),
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "patch_mode": evidence["geometry"]["patch_mode"],
            "edit_visibility_evidence": evidence,
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
    probability_tolerance: float = PROBABILITY_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Validate float32 score fields and the released strict decision."""

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
    recorded_decision = probability > FIXED_THRESHOLD
    decision = replay_probability > FIXED_THRESHOLD
    _require_equal(
        recorded_decision,
        decision,
        f"row {row_id} recorded/replay threshold decision",
    )
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
    expected_t1 = {
        "raw_logit": recorded_raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "threshold": FIXED_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "decision": decision,
        "policy": T1_POLICY,
    }
    _compare_nested(
        t1,
        expected_t1,
        label=f"row {row_id} t1",
        exact_mapping_keys=True,
    )
    return {
        "raw_logit": float(replay_raw_logit),
        "probability": float(replay_probability),
        "decision": bool(decision),
    }


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize physical append-only JSONL history and latest retry state."""

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
    ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        ranks = ranks[:pair_limit]
    selected_ranks = set(ranks)
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
        raise FileNotFoundError(f"missing SPAI source root: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    _require_equal(commit, FROZEN_SOURCE_COMMIT, "frozen SPAI source commit")
    _require_equal(commit, pins.MODEL_SOURCE_COMMIT, "runner/source commit pin")
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty is None:
        raise ValueError("SPAI source is not a readable git repository")
    if dirty:
        raise ValueError("SPAI source has tracked modifications")
    _require_equal(pins.SOURCE_FILES, FROZEN_SOURCE_FILES, "source-file pins")
    for relative, digest in FROZEN_SOURCE_FILES.items():
        _verify_hash(
            source_root / relative,
            digest,
            f"SPAI source {relative}",
        )
    _require_equal(
        pins.CONFIG_SHA256,
        FROZEN_CONFIG_SHA256,
        "runner/analyzer SPAI config SHA-256",
    )
    return {
        "commit": commit,
        "tracked_dirty": False,
        "source_files_validated": len(FROZEN_SOURCE_FILES),
        "config_sha256": FROZEN_CONFIG_SHA256,
    }


def _checkpoint_schema(
    state: Mapping[str, Any],
    *,
    torch_module: ModuleType,
    embedded_minimum_patches: int = 1,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    elements = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(
            tensor,
            torch_module.Tensor,
        ):
            raise ValueError("SPAI model state must map string keys to tensors")
        if tensor.is_complex():
            raise ValueError(f"SPAI checkpoint tensor {key} is complex")
        if tensor.is_floating_point() and not bool(
            torch_module.isfinite(tensor).all().item()
        ):
            raise ValueError(f"SPAI checkpoint tensor {key} is not finite")
        elements += int(tensor.numel())
        items.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "sha256": _tensor_sha256(tensor),
            }
        )
    return {
        "tensor_count": len(items),
        "state_elements": elements,
        "dtype_counts": dict(Counter(item["dtype"] for item in items)),
        "items_sha256": hashlib.sha256(
            stable_json(items).encode("utf-8")
        ).hexdigest(),
        "items": items,
        "embedded_config_minimum_patches": embedded_minimum_patches,
        "released_inference_config_minimum_patches": MINIMUM_PATCHES,
        "embedded_config_is_historical_not_restored": True,
    }


@contextlib.contextmanager
def _official_spai_import(source_root: Path):
    """Import the pinned package while preventing a different SPAI shadow."""

    source = str(source_root.resolve())
    previous_path = list(sys.path)
    previous_modules = {
        key: module
        for key, module in list(sys.modules.items())
        if key == "spai" or key.startswith("spai.")
    }
    for key in previous_modules:
        sys.modules.pop(key, None)
    sys.path.insert(0, source)
    try:
        yield
    finally:
        sys.path[:] = previous_path
        for key in list(sys.modules):
            if key == "spai" or key.startswith("spai."):
                sys.modules.pop(key, None)
        sys.modules.update(previous_modules)


def _safe_checkpoint_payload(
    checkpoint: Path,
    *,
    torch_module: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Safely load the official pickle with one audited CfgNode allowlist."""

    unsafe_api = getattr(
        getattr(torch_module, "serialization", None),
        "get_unsafe_globals_in_checkpoint",
        None,
    )
    if unsafe_api is None:
        raise RuntimeError("torch lacks safe checkpoint global inspection")
    unsafe = sorted(unsafe_api(checkpoint))
    _require_equal(unsafe, FROZEN_UNSAFE_GLOBALS, "SPAI checkpoint unsafe globals")
    from yacs.config import CfgNode

    with torch_module.serialization.safe_globals([CfgNode]):
        payload = torch_module.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, dict):
        raise ValueError("SPAI checkpoint is not a dictionary")
    required = {
        "model",
        "optimizer",
        "lr_scheduler",
        "max_accuracy",
        "epoch",
        "config",
        "amp",
    }
    _require_equal(set(payload), required, "SPAI checkpoint top-level keys")
    if not isinstance(payload.get("model"), OrderedDict):
        raise ValueError("SPAI checkpoint model is not an OrderedDict")
    safety = {
        "weights_only": True,
        "pickle_executed": False,
        "safe_global_allowlist": FROZEN_UNSAFE_GLOBALS,
        "loader": "torch.load(map_location=cpu, weights_only=True)",
    }
    return payload, safety


def _build_and_load_model(
    source_root: Path,
    checkpoint: Path,
    *,
    torch_module: ModuleType,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build current official config and strict-load only its model state."""

    network_attempts = Counter()

    def blocked(name: str):
        def reject(*_args: Any, **_kwargs: Any) -> Any:
            network_attempts[name] += 1
            raise RuntimeError(
                "network access is forbidden during SPAI audit construction"
            )

        return reject

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                urllib.request,
                "urlopen",
                side_effect=blocked("urllib_urlopen"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket,
                "create_connection",
                side_effect=blocked("socket_create_connection"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket.socket,
                "connect",
                side_effect=blocked("socket_connect"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch_module.hub,
                "load",
                side_effect=blocked("torch_hub_load"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch_module.hub,
                "load_state_dict_from_url",
                side_effect=blocked("torch_hub_load_state_dict_from_url"),
            )
        )
        with _official_spai_import(source_root):
            config_module = importlib.import_module("spai.config")
            build_module = importlib.import_module("spai.models.build")
            config = config_module.get_custom_config(
                str(source_root / FROZEN_CONFIG_RELATIVE)
            )
            model = build_module.build_cls_model(config)
    if network_attempts:
        raise RuntimeError(
            f"SPAI audit construction attempted network: {dict(network_attempts)}"
        )
    payload, safety = _safe_checkpoint_payload(
        checkpoint,
        torch_module=torch_module,
    )
    state = payload["model"]
    embedded = int(payload["config"].MODEL.PATCH_VIT.MINIMUM_PATCHES)
    _require_equal(embedded, 1, "embedded historical minimum patches")
    schema = _checkpoint_schema(
        state,
        torch_module=torch_module,
        embedded_minimum_patches=embedded,
    )
    _require_equal(
        schema["tensor_count"],
        FROZEN_MODEL_STATE_ENTRIES,
        "SPAI model-state entries",
    )
    _require_equal(
        schema["state_elements"],
        FROZEN_MODEL_STATE_ELEMENTS,
        "SPAI model-state elements",
    )
    _require_equal(
        list(state),
        list(model.state_dict()),
        "SPAI checkpoint/model key order",
    )
    model.load_state_dict(state, strict=True)
    _require_equal(model.img_patch_size, PATCH_SIZE, "model patch size")
    _require_equal(model.img_patch_stride, PATCH_STRIDE, "model patch stride")
    _require_equal(model.minimum_patches, MINIMUM_PATCHES, "model minimum patches")
    _require_equal(model.cls_vector_dim, FEATURE_DIMENSION, "model feature width")
    _require_equal(model.heads, ATTENTION_HEADS, "model attention heads")
    state_elements = sum(
        int(tensor.numel()) for tensor in model.state_dict().values()
    )
    _require_equal(
        state_elements,
        FROZEN_MODEL_STATE_ELEMENTS,
        "SPAI model parameter/buffer element contract",
    )
    model.eval()
    return model, schema, safety


def _checkpoint_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> tuple[Path, dict[str, Any]]:
    assets = _require_mapping(manifest.get("assets"), "manifest assets")
    record = _require_mapping(assets.get("checkpoint"), "checkpoint asset")
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest checkpoint has no explicit path")
    path = _anchored(Path(value), repo_root)
    _verify_hash(path, FROZEN_CHECKPOINT_SHA256, "SPAI checkpoint")
    _require_equal(path.stat().st_size, FROZEN_CHECKPOINT_BYTES, "checkpoint bytes")
    for key in ("actual_sha256", "sha256"):
        if key in record:
            _require_equal(
                record[key],
                FROZEN_CHECKPOINT_SHA256,
                f"manifest checkpoint {key}",
            )
    for key in ("actual_bytes", "bytes"):
        if key in record:
            _require_equal(
                int(record[key]),
                FROZEN_CHECKPOINT_BYTES,
                f"manifest checkpoint {key}",
            )
    checkpoint_pin = _require_mapping(pins.CHECKPOINT, "runner checkpoint pin")
    for key, expected in checkpoint_pin.items():
        if key in record:
            _compare_nested(
                record[key],
                expected,
                label=f"manifest checkpoint {key}",
            )
    return path, record


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _replay_runtime(manifest: Mapping[str, Any]) -> ReplayRuntime:
    """Verify and recreate the runner's deterministic numerical runtime."""

    # These must precede importing Albumentations or any optional hub client.
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
    runtime_contract = _require_mapping(
        config.get("runtime_contract"),
        "config runtime contract",
    )
    expected_contract = {
        "device": runtime.get("device"),
        "seed": 0,
        "dtype": "float32",
        "autocast": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
        "network_allowed": False,
        "runtime_versions": FROZEN_RUNTIME_VERSIONS,
        "runtime_module_file_hashes": FROZEN_RUNTIME_MODULE_FILES,
    }
    _require_equal(
        runtime_contract,
        expected_contract,
        "runtime contract",
    )
    expected_keys = {
        "python",
        "platform",
        "versions",
        "module_files",
        "device",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "cuda_device_name",
        "seed",
        "dtype",
        "autocast",
        "tf32",
        "network_allowed",
        "cublas_workspace_config",
        "no_albumentations_update",
        "cudnn_enabled",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "deterministic_algorithms",
        "float32_matmul_precision",
    }
    _require_equal(set(runtime), expected_keys, "runtime evidence keys")
    _require_equal(runtime["python"], sys.version, "runtime Python")
    _require_equal(runtime["platform"], platform.platform(), "runtime platform")
    versions = _require_mapping(runtime["versions"], "runtime versions")
    _require_equal(
        versions,
        FROZEN_RUNTIME_VERSIONS,
        "runtime frozen package versions",
    )
    actual_versions = {
        name: _package_version(name) for name in FROZEN_RUNTIME_VERSIONS
    }
    _require_equal(actual_versions, versions, "runtime installed versions")
    module_files = _require_mapping(
        runtime["module_files"],
        "runtime module files",
    )
    _require_equal(
        set(module_files),
        set(FROZEN_RUNTIME_MODULE_FILES),
        "runtime module-file keys",
    )
    for module_name, expected_digest in FROZEN_RUNTIME_MODULE_FILES.items():
        module = importlib.import_module(module_name)
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            raise ValueError(f"runtime module has no file: {module_name}")
        actual_path = Path(value).resolve()
        record = _require_mapping(
            module_files[module_name],
            f"runtime module {module_name}",
        )
        _require_equal(
            set(record),
            {"path", "sha256"},
            f"runtime module {module_name} keys",
        )
        _require_equal(
            Path(str(record["path"])).resolve(),
            actual_path,
            f"runtime module {module_name} path",
        )
        _require_equal(
            record["sha256"],
            expected_digest,
            f"runtime module {module_name} digest pin",
        )
        _verify_hash(
            actual_path,
            expected_digest,
            f"runtime module {module_name}",
        )
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected_workspace = ":4096:8"
    _require_equal(
        runtime["cublas_workspace_config"],
        expected_workspace,
        "runtime CUBLAS workspace config",
    )
    _require_equal(
        runtime["no_albumentations_update"],
        "1",
        "runtime Albumentations update check disabled",
    )
    if workspace_config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    elif workspace_config != expected_workspace:
        raise ValueError(
            "SPAI deterministic replay requires "
            f"CUBLAS_WORKSPACE_CONFIG={expected_workspace}"
        )
    import torch

    requested = runtime.get("device")
    if not isinstance(requested, str):
        raise ValueError("runtime device is not a string")
    _require_equal(
        requested,
        PRIMARY_DEVICE,
        "SPAI executable replay device",
    )
    device = torch.device(requested)
    _require_equal(
        runtime["cuda_available"],
        bool(torch.cuda.is_available()),
        "runtime CUDA availability",
    )
    _require_equal(runtime["cuda_version"], torch.version.cuda, "CUDA version")
    _require_equal(
        runtime["cudnn_version"],
        torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuDNN version",
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recorded SPAI CUDA runtime is unavailable")
        if device.index is None:
            raise ValueError("recorded SPAI CUDA device lacks explicit index")
        torch.cuda.set_device(device)
        _require_equal(
            runtime["cuda_device_name"],
            torch.cuda.get_device_name(device),
            "CUDA device name",
        )
    elif device.type != "cpu":
        raise ValueError("recorded SPAI device is neither CPU nor CUDA")
    else:
        _require_equal(runtime["cuda_device_name"], None, "CPU CUDA device name")
    strict_values = {
        "seed": 0,
        "dtype": "float32",
        "autocast": False,
        "tf32": False,
        "network_allowed": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }
    for key, expected in strict_values.items():
        _require_equal(runtime[key], expected, f"runtime {key}")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    replayed_truth = {
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for key, expected in replayed_truth.items():
        _require_equal(runtime[key], expected, f"replayed runtime {key}")
    return ReplayRuntime(torch=torch, device=device, evidence=dict(runtime))


def _forward_with_evidence(
    model: Any,
    tensor: Any,
    *,
    torch_module: ModuleType,
    feature_extraction_batch: int = FEATURE_EXTRACTION_BATCH,
) -> ForwardEvidence:
    """Fresh full FFT+ViT+SRS+SCA forward with independent orchestration."""

    geometry = compute_patch_geometry(
        int(tensor.shape[-1]),
        int(tensor.shape[-2]),
    )
    patches = _patchify_tensor(
        tensor,
        geometry,
        torch_module=torch_module,
    )
    encoded: list[Any] = []
    for start in range(0, int(patches.shape[0]), feature_extraction_batch):
        encoded.append(model.mfvit(patches[start : start + feature_extraction_batch]))
    patch_features = torch_module.cat(encoded, dim=0)
    _require_equal(
        tuple(patch_features.shape),
        (int(patches.shape[0]), FEATURE_DIMENSION),
        "fresh SPAI patch-feature shape",
    )
    attended, attention = model.patches_attention(
        patch_features.unsqueeze(0),
        return_attn=True,
    )
    normalized = model.norm(attended)
    raw_logit = model.cls_head(normalized)
    attention = attention.squeeze(0).squeeze(1)
    _require_equal(
        tuple(attention.shape),
        (ATTENTION_HEADS, int(patches.shape[0])),
        "fresh SPAI attention shape",
    )
    return ForwardEvidence(
        patch_features=patch_features,
        normalized_feature=normalized.squeeze(0),
        attention=attention,
        raw_logit=raw_logit.reshape(()),
    )


def _replay_patch_artifact(
    model: Any,
    patch_features: Any,
) -> ForwardEvidence:
    """Replay SCA, LayerNorm, and MLP from persisted patch features."""

    attended, attention = model.patches_attention(
        patch_features.unsqueeze(0),
        return_attn=True,
    )
    normalized = model.norm(attended)
    raw_logit = model.cls_head(normalized)
    return ForwardEvidence(
        patch_features=patch_features,
        normalized_feature=normalized.squeeze(0),
        attention=attention.squeeze(0).squeeze(1),
        raw_logit=raw_logit.reshape(()),
    )


def _safe_npy(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_shape: tuple[int, ...],
    semantics: str,
    repo_root: Path,
    run_dir: Path,
) -> tuple[np.ndarray, Path]:
    row_id = row.get("id")
    value = row.get(f"{prefix}_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row_id} has no {prefix} path")
    path = _anchored(Path(value), repo_root)
    if not path.is_file():
        candidate = (run_dir / value).resolve()
        if candidate.is_file():
            path = candidate
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"row {row_id} {prefix} artifact is outside its run directory"
        ) from exc
    _verify_hash(
        path,
        row.get(f"{prefix}_sha256"),
        f"row {row_id} {prefix}",
    )
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"row {row_id} {prefix} is not a safe NPY") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"row {row_id} {prefix} is not an ndarray")
    _require_equal(array.shape, expected_shape, f"row {row_id} {prefix} shape")
    _require_equal(array.dtype, FEATURE_DTYPE, f"row {row_id} {prefix} dtype")
    if not array.flags.c_contiguous:
        raise ValueError(f"row {row_id} {prefix} is not C-contiguous")
    if not np.isfinite(array).all():
        raise ValueError(f"row {row_id} {prefix} is not finite")
    array_digest = _array_sha256(array)
    _require_equal(
        row.get(f"{prefix}_array_sha256"),
        array_digest,
        f"row {row_id} {prefix} array SHA-256",
    )
    _require_equal(
        row.get(f"{prefix}_shape"),
        list(expected_shape),
        f"row {row_id} recorded {prefix} shape",
    )
    _require_equal(
        row.get(f"{prefix}_dtype"),
        "float32",
        f"row {row_id} recorded {prefix} dtype",
    )
    _require_equal(
        row.get(f"{prefix}_semantics"),
        semantics,
        f"row {row_id} {prefix} semantics",
    )
    return array, path


def _load_artifacts(
    row: Mapping[str, Any],
    *,
    patch_count: int,
    repo_root: Path,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Path]]:
    patch, patch_path = _safe_npy(
        row,
        prefix="spai_patch_features",
        expected_shape=(patch_count, FEATURE_DIMENSION),
        semantics=PATCH_FEATURE_SEMANTICS,
        repo_root=repo_root,
        run_dir=run_dir,
    )
    feature, feature_path = _safe_npy(
        row,
        prefix="spai_feature",
        expected_shape=(FEATURE_DIMENSION,),
        semantics=FEATURE_SEMANTICS,
        repo_root=repo_root,
        run_dir=run_dir,
    )
    attention, attention_path = _safe_npy(
        row,
        prefix="spai_attention",
        expected_shape=(ATTENTION_HEADS, patch_count),
        semantics=ATTENTION_SEMANTICS,
        repo_root=repo_root,
        run_dir=run_dir,
    )
    _require_equal(
        row.get("feature_array_sha256"),
        row.get("spai_feature_array_sha256"),
        f"row {row.get('id')} feature array alias",
    )
    paths = _require_mapping(
        row.get("artifact_paths"),
        f"row {row.get('id')} artifact paths",
    )
    expected_paths = {
        "spai_patch_features_npy": _relative_or_absolute(patch_path, repo_root),
        "spai_feature_npy": _relative_or_absolute(feature_path, repo_root),
        "spai_attention_npy": _relative_or_absolute(attention_path, repo_root),
    }
    _require_equal(
        set(paths),
        set(expected_paths),
        f"row {row.get('id')} artifact path keys",
    )
    for key, expected in expected_paths.items():
        recorded = Path(str(paths[key]))
        resolved = _anchored(recorded, repo_root)
        if not resolved.is_file():
            candidate = (run_dir / recorded).resolve()
            if candidate.is_file():
                resolved = candidate
        _require_equal(
            resolved,
            {
                "spai_patch_features_npy": patch_path,
                "spai_feature_npy": feature_path,
                "spai_attention_npy": attention_path,
            }[key],
            f"row {row.get('id')} artifact path {key}",
        )
    return patch, feature, attention, {
        "patch": patch_path,
        "feature": feature_path,
        "attention": attention_path,
    }


def _audit_preprocess_record(
    row: Mapping[str, Any],
    prepared: PreprocessedImage,
) -> None:
    _require_equal(
        row.get("preprocess_profile"),
        PREPROCESS_PROFILE,
        f"row {row.get('id')} preprocess profile",
    )
    _compare_nested(
        row.get("preprocess"),
        prepared.audit,
        label=f"row {row.get('id')} preprocess",
        exact_mapping_keys=True,
    )


def _validate_result_identity(
    row: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    repo_root: Path,
    config_fingerprint: str,
    pins: SimpleNamespace,
) -> Path:
    row_id = str(row.get("id"))
    expected = {
        "id": canonical["sample_id"],
        "sample_id": canonical["sample_id"],
        "task_id": canonical["task_id"],
        "pair_rank": canonical["pair_rank"],
        "rank": canonical["rank"],
        "kind": canonical["kind"],
        "label": canonical["label"],
        "domain": canonical["domain"],
        "model": pins.MODEL_NAME,
        "model_slug": pins.MODEL_SLUG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "config_fingerprint": config_fingerprint,
    }
    for key, value in expected.items():
        _require_equal(row.get(key), value, f"row {row_id} {key}")
    path_value = row.get("input_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"row {row_id} has no input path")
    input_path = _anchored(Path(path_value), repo_root)
    canonical_path = _anchored(Path(str(canonical["canonical_path"])), repo_root)
    _require_equal(input_path, canonical_path, f"row {row_id} input path")
    _verify_hash(input_path, canonical["canonical_sha256"], f"row {row_id} input")
    _require_equal(
        row.get("input_sha256"),
        canonical["canonical_sha256"],
        f"row {row_id} input SHA-256",
    )
    _require_equal(
        row.get("input_width"),
        canonical["width"],
        f"row {row_id} input width",
    )
    _require_equal(
        row.get("input_height"),
        canonical["height"],
        f"row {row_id} input height",
    )
    return input_path


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
        "eval/opensource/run_spai.py",
        "eval/opensource/spai_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/maskclip_metrics.py",
    }
    _require_equal(set(contract), required, "SPAI adapter filenames")
    for relative, raw in contract.items():
        record = _require_mapping(raw, f"adapter {relative}")
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


def _validate_official_golden_contract(
    *,
    golden_record: Mapping[str, Any],
    golden_assets: list[Any],
) -> dict[str, Any]:
    """Bind recorded golden evidence to analyzer-owned frozen constants."""

    expected_record_keys = {
        "status",
        "source",
        "runs_per_case",
        "logit_absolute_tolerance",
        "probability_absolute_tolerance",
        "official_vs_adapter_full_forward",
        "website_display_reference",
        "cases",
    }
    _require_equal(
        set(golden_record),
        expected_record_keys,
        "official golden keys",
    )
    _require_equal(golden_record.get("status"), "passed", "official golden status")
    _require_equal(
        golden_record.get("source"),
        FROZEN_GOLDEN_SOURCE,
        "official golden source",
    )
    _require_equal(
        golden_record.get("runs_per_case"),
        2,
        "official golden runs per case",
    )
    _require_equal(
        golden_record.get("logit_absolute_tolerance"),
        FROZEN_GOLDEN_LOGIT_ABSOLUTE_TOLERANCE,
        "official golden logit tolerance",
    )
    _require_equal(
        golden_record.get("probability_absolute_tolerance"),
        FROZEN_GOLDEN_PROBABILITY_ABSOLUTE_TOLERANCE,
        "official golden probability tolerance",
    )
    if golden_record.get("official_vs_adapter_full_forward") is not True:
        raise ValueError("official golden full-forward status is not true")
    _require_equal(
        golden_record.get("website_display_reference"),
        FROZEN_GOLDEN_WEBSITE_DISPLAY_REFERENCE,
        "official golden website mismatch semantics",
    )

    recorded_cases = _require_list(
        golden_record.get("cases"),
        "official golden cases",
    )
    _require_equal(
        len(recorded_cases),
        len(FROZEN_GOLDEN_CASES),
        "official golden case count",
    )
    _require_equal(
        len(golden_assets),
        len(FROZEN_GOLDEN_CASES),
        "golden asset count",
    )

    golden_root: Path | None = None
    artifact_digest_sets: list[dict[str, str]] = []
    for index, frozen in enumerate(FROZEN_GOLDEN_CASES):
        label = f"official golden case {index}"
        asset = _require_mapping(golden_assets[index], f"golden asset {index}")
        _require_equal(
            set(asset),
            set(frozen) | {"path", "bytes"},
            f"golden asset {index} keys",
        )
        for key, expected in frozen.items():
            _compare_nested(
                asset.get(key),
                expected,
                label=f"golden asset {index} {key}",
                exact_mapping_keys=True,
            )
        asset_bytes = asset.get("bytes")
        if (
            isinstance(asset_bytes, bool)
            or not isinstance(asset_bytes, int)
            or asset_bytes <= 0
        ):
            raise ValueError(f"golden asset {index} bytes is invalid")
        path_value = asset.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"golden asset {index} path is invalid")
        asset_path = Path(path_value).resolve()
        relative = Path(str(frozen["relative_path"]))
        relative_parts = relative.parts
        if (
            not relative_parts
            or len(asset_path.parts) < len(relative_parts)
            or asset_path.parts[-len(relative_parts) :] != relative_parts
        ):
            raise ValueError(
                f"golden asset {index} path does not bind relative_path"
            )
        case_root = asset_path.parents[len(relative_parts) - 1]
        if golden_root is None:
            golden_root = case_root
        else:
            _require_equal(
                case_root,
                golden_root,
                f"golden asset {index} common root",
            )

        case = _require_mapping(recorded_cases[index], label)
        case_only_keys = {
            "path",
            "preprocess",
            "observed_runs",
            "artifact_hashes",
            "bit_identical_repeats",
            "logit_absolute_difference",
            "probability_absolute_difference",
            "passed",
            "website_derivative_display_matches_released_regression",
        }
        _require_equal(
            set(case),
            set(frozen) | case_only_keys,
            f"{label} keys",
        )
        for key, expected in frozen.items():
            _compare_nested(
                case.get(key),
                expected,
                label=f"{label} {key}",
                exact_mapping_keys=True,
            )
        _require_equal(
            Path(str(case.get("path"))).resolve(),
            asset_path,
            f"{label} path",
        )

        preprocess = _require_mapping(
            case.get("preprocess"),
            f"{label} preprocess",
        )
        preprocess_expected = {
            "profile": PREPROCESS_PROFILE,
            "native_size": frozen["native_size"],
            "decoded_rgb_sha256": frozen["decoded_rgb_sha256"],
            "tensor_sha256": frozen["tensor_sha256"],
        }
        for key, expected in preprocess_expected.items():
            _compare_nested(
                preprocess.get(key),
                expected,
                label=f"{label} preprocess {key}",
                exact_mapping_keys=True,
            )
        geometry = _require_mapping(
            preprocess.get("geometry"),
            f"{label} preprocess geometry",
        )
        _require_equal(
            geometry.get("effective_patch_count"),
            frozen["patch_count"],
            f"{label} patch count",
        )

        observed = _require_list(case.get("observed_runs"), f"{label} observed")
        _require_equal(len(observed), 2, f"{label} observed-run count")
        normalized_observed: list[dict[str, float]] = []
        for run_index, raw in enumerate(observed):
            run = _require_mapping(raw, f"{label} observed {run_index}")
            _require_equal(
                set(run),
                {"raw_logit", "probability"},
                f"{label} observed {run_index} keys",
            )
            normalized_observed.append(
                {
                    "raw_logit": _require_finite(
                        run.get("raw_logit"),
                        f"{label} observed {run_index} raw logit",
                    ),
                    "probability": _require_probability(
                        run.get("probability"),
                        f"{label} observed {run_index} probability",
                    ),
                }
            )

        artifact_runs = _require_list(
            case.get("artifact_hashes"),
            f"{label} artifact hashes",
        )
        _require_equal(len(artifact_runs), 2, f"{label} artifact-hash count")
        normalized_artifacts: list[dict[str, str]] = []
        for run_index, raw in enumerate(artifact_runs):
            hashes = _require_mapping(
                raw,
                f"{label} artifact hashes {run_index}",
            )
            _require_equal(
                set(hashes),
                {"patch_features", "feature", "attention"},
                f"{label} artifact hashes {run_index} keys",
            )
            normalized_artifacts.append(
                {
                    key: _require_sha256(
                        hashes.get(key),
                        f"{label} artifact hashes {run_index} {key}",
                    )
                    for key in ("patch_features", "feature", "attention")
                }
            )

        deterministic = (
            normalized_observed[0] == normalized_observed[1]
            and normalized_artifacts[0] == normalized_artifacts[1]
        )
        if case.get("bit_identical_repeats") is not deterministic:
            raise ValueError(f"{label} bit-identical repeat claim mismatch")
        if not deterministic:
            raise ValueError(f"{label} repeated observations are not identical")
        logit_difference = abs(
            normalized_observed[0]["raw_logit"] - float(frozen["raw_logit"])
        )
        probability_difference = abs(
            normalized_observed[0]["probability"]
            - float(frozen["probability"])
        )
        _require_equal(
            _require_finite(
                case.get("logit_absolute_difference"),
                f"{label} logit difference",
            ),
            logit_difference,
            f"{label} recorded logit difference",
        )
        _require_equal(
            _require_finite(
                case.get("probability_absolute_difference"),
                f"{label} probability difference",
            ),
            probability_difference,
            f"{label} recorded probability difference",
        )
        passed = (
            deterministic
            and logit_difference <= FROZEN_GOLDEN_LOGIT_ABSOLUTE_TOLERANCE
            and probability_difference
            <= FROZEN_GOLDEN_PROBABILITY_ABSOLUTE_TOLERANCE
        )
        if case.get("passed") is not passed or not passed:
            raise ValueError(f"{label} pass status mismatch")
        if (
            case.get(
                "website_derivative_display_matches_released_regression"
            )
            is not False
        ):
            raise ValueError(f"{label} website mismatch semantics changed")
        artifact_digest_sets.append(normalized_artifacts[0])

    assert golden_root is not None
    return {
        "status": "passed",
        "cases_validated": len(FROZEN_GOLDEN_CASES),
        "runs_per_case": 2,
        "golden_root": str(golden_root),
        "artifact_digest_sets": artifact_digest_sets,
        "website_derivatives_are_diagnostic_mismatches": True,
    }


def _validate_manifest_paths_and_source(
    *,
    repo_root: Path,
    source_root: Path,
    inputs_path: Path,
    files: RunFiles,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> dict[str, Any]:
    """Bind all manifest paths and hashes to the physical audited files."""

    manifest = files.manifest
    _require_equal(
        manifest.get("run_id"),
        files.run_dir.name,
        "manifest run id",
    )
    config = _require_mapping(manifest.get("config"), "manifest config")
    fingerprint = _fingerprint(config)
    _require_equal(
        manifest.get("config_fingerprint"),
        fingerprint,
        "manifest config fingerprint",
    )
    _require_equal(
        config.get("source_commit"),
        FROZEN_SOURCE_COMMIT,
        "config source commit",
    )
    _require_equal(
        config.get("source_files"),
        FROZEN_SOURCE_FILES,
        "config source files",
    )
    _require_equal(
        config.get("checkpoint_id"),
        pins.CHECKPOINT["id"],
        "config checkpoint id",
    )
    _require_equal(
        config.get("checkpoint_sha256"),
        FROZEN_CHECKPOINT_SHA256,
        "config checkpoint SHA-256",
    )
    _require_equal(
        manifest.get("attention_is_diagnostic_not_T2"),
        True,
        "manifest attention scope",
    )
    _require_equal(
        manifest.get("valid_for_t2"),
        False,
        "manifest valid_for_t2",
    )
    source = _require_mapping(manifest.get("source"), "manifest source")
    _require_equal(
        Path(str(source.get("root"))).resolve(),
        source_root.resolve(),
        "manifest source root",
    )
    _require_equal(
        source.get("commit"),
        FROZEN_SOURCE_COMMIT,
        "manifest source commit",
    )
    _require_equal(
        source.get("tracked_dirty"),
        False,
        "manifest source dirty state",
    )
    records = _require_mapping(
        source.get("source_files"),
        "manifest source files",
    )
    _require_equal(
        set(records),
        set(FROZEN_SOURCE_FILES),
        "manifest source-file keys",
    )
    for relative, digest in FROZEN_SOURCE_FILES.items():
        record = _require_mapping(
            records[relative],
            f"manifest source {relative}",
        )
        _require_equal(
            Path(str(record.get("path"))).resolve(),
            (source_root / relative).resolve(),
            f"manifest source {relative} path",
        )
        _require_equal(
            record.get("sha256"),
            digest,
            f"manifest source {relative} SHA-256",
        )
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    release_manifest = inputs_path.parent / "manifest.json"
    release_value = dataset.get("manifest_path")
    if not isinstance(release_value, str) or not release_value:
        raise ValueError("manifest dataset lacks manifest_path")
    _require_equal(
        _anchored(Path(release_value), repo_root),
        release_manifest.resolve(),
        "dataset manifest path",
    )
    _verify_hash(
        release_manifest,
        dataset.get("manifest_sha256"),
        "dataset release manifest",
    )
    bindings = (
        ("inputs_path", inputs_path, "inputs_sha256"),
        ("expected_inputs_path", files.expected_path, "expected_inputs_sha256"),
    )
    for path_key, expected_path, hash_key in bindings:
        value = dataset.get(path_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest dataset lacks {path_key}")
        path = _anchored(Path(value), repo_root)
        _require_equal(path, expected_path.resolve(), f"dataset {path_key}")
        _verify_hash(path, dataset.get(hash_key), f"dataset {path_key}")
    selected_tasks = {str(row["task_id"]) for row in files.expected}
    _require_equal(
        dataset.get("selected_images"),
        len(files.expected),
        "manifest selected image count",
    )
    _require_equal(
        dataset.get("selected_tasks"),
        len(selected_tasks),
        "manifest selected task count",
    )
    config_dataset = _require_mapping(
        config.get("dataset"),
        "config dataset",
    )
    _require_equal(
        config_dataset.get("selected_ids"),
        [str(row["sample_id"]) for row in files.expected],
        "config selected ids",
    )
    _require_equal(
        config_dataset.get("selected_rows_sha256"),
        _selected_rows_sha256(files.expected),
        "config selected rows SHA-256",
    )
    _require_equal(
        config_dataset.get("inputs_sha256"),
        sha256_file(inputs_path),
        "config inputs SHA-256",
    )
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    artifact_dir = files.run_dir / "artifacts"
    output_bindings = (
        ("results_path", files.results_path),
        ("summary_path", files.summary_path),
        ("artifact_dir", artifact_dir),
    )
    for key, expected in output_bindings:
        value = outputs.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest outputs lacks {key}")
        actual = _anchored(Path(value), repo_root)
        _require_equal(actual, expected.resolve(), f"manifest output {key}")
        try:
            actual.relative_to(files.run_dir)
        except ValueError as exc:
            raise ValueError(f"manifest output {key} escapes run directory") from exc
    for key, path in (
        ("results_sha256", files.results_path),
        ("summary_sha256", files.summary_path),
    ):
        if key in outputs:
            _verify_hash(path, outputs[key], f"manifest output {key}")
    if "artifact_files" in outputs:
        count = sum(1 for path in artifact_dir.glob("*.npy") if path.is_file())
        _require_equal(
            outputs["artifact_files"],
            count,
            "manifest artifact-file count",
        )
    assets = _require_mapping(manifest.get("assets"), "manifest assets")
    golden = _require_list(assets.get("golden_assets"), "golden assets")
    golden_record = _require_mapping(
        manifest.get("official_golden"),
        "manifest official golden",
    )
    golden_contract = _validate_official_golden_contract(
        golden_record=golden_record,
        golden_assets=golden,
    )
    recorded_golden_cases = _require_list(
        golden_record.get("cases"),
        "manifest official golden cases",
    )
    for index, (raw, frozen) in enumerate(
        zip(golden, FROZEN_GOLDEN_CASES)
    ):
        record = _require_mapping(raw, f"golden asset {index}")
        path = Path(str(record.get("path"))).resolve()
        _verify_hash(path, frozen["sha256"], f"golden asset {index}")
        _require_equal(
            record.get("bytes"),
            path.stat().st_size,
            f"golden asset {index} bytes",
        )
        prepared = preprocess_image(path, torch_module=torch_module)
        _require_equal(
            prepared.decoded_rgb_sha256,
            frozen["decoded_rgb_sha256"],
            f"golden asset {index} decoded RGB SHA-256",
        )
        _require_equal(
            prepared.tensor_sha256,
            frozen["tensor_sha256"],
            f"golden asset {index} tensor SHA-256",
        )
        _require_equal(
            prepared.audit["native_size"],
            frozen["native_size"],
            f"golden asset {index} native size",
        )
        _require_equal(
            prepared.geometry["effective_patch_count"],
            frozen["patch_count"],
            f"golden asset {index} patch count",
        )
        _compare_nested(
            _require_mapping(
                recorded_golden_cases[index],
                f"manifest official golden case {index}",
            ).get("preprocess"),
            prepared.audit,
            label=f"manifest official golden case {index} preprocess",
            exact_mapping_keys=True,
        )
    _require_equal(
        config.get("official_golden"),
        golden_record,
        "config/manifest official golden",
    )
    _require_equal(
        config.get("official_golden_fingerprint"),
        _fingerprint(golden_record),
        "config official golden fingerprint",
    )
    summary = files.summary
    expected_summary_identity = {
        "schema_version": "spai_detection_summary_v1",
        "run_id": files.run_dir.name,
        "model": pins.MODEL_NAME,
        "model_slug": pins.MODEL_SLUG,
        "checkpoint_id": pins.CHECKPOINT["id"],
        "preprocess_profile": PREPROCESS_PROFILE,
        "config_fingerprint": fingerprint,
        "official_golden_status": golden_record.get("status"),
        "official_golden_fingerprint": _fingerprint(golden_record),
        "attention_is_diagnostic_not_T2": True,
        "valid_for_t2": False,
    }
    for key, expected in expected_summary_identity.items():
        _require_equal(summary.get(key), expected, f"summary {key}")
    coverage = _require_mapping(summary.get("coverage"), "summary coverage")
    expected_status = "complete" if coverage.get("is_complete") else "incomplete"
    _require_equal(manifest.get("status"), expected_status, "manifest status")
    full_inputs = read_jsonl(inputs_path)
    pair_visibility = _pair_visibility(full_inputs, repo_root=repo_root)
    visibility_census = dict(
        Counter(
            pair_visibility[task]["edit_visibility"]
            for task in selected_tasks
        )
    )
    patch_mode_census = dict(
        Counter(pair_visibility[task]["patch_mode"] for task in selected_tasks)
    )
    _require_equal(
        manifest.get("visibility_census"),
        visibility_census,
        "manifest visibility census",
    )
    _require_equal(
        manifest.get("patch_mode_census"),
        patch_mode_census,
        "manifest patch-mode census",
    )
    return {
        "config_fingerprint": fingerprint,
        "source_paths_validated": len(records),
        "dataset_paths_validated": len(bindings),
        "output_paths_validated": len(output_bindings),
        "golden_assets_validated": len(golden),
        "official_golden_contract": golden_contract,
        "summary_identity_validated": True,
        "selection_census_validated": True,
    }


def _load_run_files(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
) -> RunFiles:
    safe_run_id = _safe_component(run_id, label="run-id")
    results_root = _anchored(results_dir, repo_root)
    run_dir = (results_root / safe_run_id).resolve()
    _require_equal(
        run_dir.parent,
        results_root.resolve(),
        "run directory boundary",
    )
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    for path in (results_path, expected_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing SPAI run artifact: {path}")
    rows = read_jsonl(results_path)
    expected = read_jsonl(expected_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_mapping(summary, "summary")
    _require_mapping(manifest, "manifest")
    return RunFiles(
        run_dir=run_dir.resolve(),
        results_path=results_path.resolve(),
        expected_path=expected_path.resolve(),
        summary=summary,
        summary_path=summary_path.resolve(),
        manifest=manifest,
        manifest_path=manifest_path.resolve(),
        rows=rows,
        expected=expected,
    )


def _validate_physical_rows(
    *,
    rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    repo_root: Path,
    run_dir: Path,
    config_fingerprint: str,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> dict[str, Any]:
    """Validate every physical retry row before applying latest-row policy."""

    canonical_by_id = {
        str(row["sample_id"]): row for row in canonical_rows
    }
    visibility = _pair_visibility(canonical_rows, repo_root=repo_root)
    counts = Counter()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in canonical_by_id:
            raise ValueError(f"physical result row {index} has unknown id")
        canonical = canonical_by_id[row_id]
        input_path = _validate_result_identity(
            row,
            canonical,
            repo_root=repo_root,
            config_fingerprint=config_fingerprint,
            pins=pins,
        )
        _require_equal(
            row.get("valid_for_t2"),
            False,
            f"physical row {index} valid_for_t2",
        )
        _require_equal(
            row.get("attention_is_diagnostic_not_T2"),
            True,
            f"physical row {index} attention scope",
        )
        expected_visibility = visibility[str(canonical["task_id"])]
        _require_equal(
            row.get("edit_visibility"),
            expected_visibility["edit_visibility"],
            f"physical row {index} visibility",
        )
        _compare_float(
            row.get("edit_visible_gt_fraction"),
            expected_visibility["edit_visible_gt_fraction"],
            label=f"physical row {index} visible fraction",
            tolerance=0.0,
        )
        _compare_nested(
            row.get("edit_visibility_evidence"),
            expected_visibility["edit_visibility_evidence"],
            label=f"physical row {index} visibility evidence",
            exact_mapping_keys=True,
        )
        status = row.get("status")
        if status == "ok":
            _require_equal(
                row.get("valid_for_metrics"),
                True,
                f"physical row {index} valid_for_metrics",
            )
            prepared = preprocess_image(input_path, torch_module=torch_module)
            _audit_preprocess_record(row, prepared)
            patch_count = int(prepared.geometry["effective_patch_count"])
            _load_artifacts(
                row,
                patch_count=patch_count,
                repo_root=repo_root,
                run_dir=run_dir,
            )
            raw = _require_finite(
                row.get("raw_logit"),
                f"physical row {index} raw_logit",
            )
            probability = _float32_sigmoid(raw, torch_module)
            _audit_score_fields(
                row,
                replay_raw_logit=raw,
                replay_probability=probability,
                raw_tolerance=0.0,
                # The physical-row precheck reconstructs sigmoid on CPU,
                # while the frozen run emitted the official float32 sigmoid
                # on CUDA. Those kernels can differ by one float32 ULP.
                # The later fresh CUDA full-model replay remains the primary
                # score audit.
                probability_tolerance=PROBABILITY_ABSOLUTE_TOLERANCE,
            )
            _audit_manual_replay(
                row,
                raw_logit=raw,
                # Manual replay was persisted by the runner on CUDA, so its
                # probability alias must exactly equal the recorded CUDA
                # probability, not this precheck's CPU sigmoid reconstruction.
                probability=_require_probability(
                    row.get("probability"),
                    f"physical row {index} probability",
                ),
            )
        elif status == "error":
            _require_equal(
                row.get("valid_for_metrics"),
                False,
                f"physical row {index} valid_for_metrics",
            )
            for key in (
                "raw_logit",
                "probability",
                "ai_score",
                "score",
                "classification_decision",
            ):
                if key not in row:
                    raise ValueError(f"physical error row {index} lacks {key}")
                _require_equal(
                    row[key],
                    None,
                    f"physical error row {index} {key}",
                )
            for key in ("error_type", "error", "traceback"):
                if not isinstance(row.get(key), str):
                    raise ValueError(
                        f"physical error row {index} has invalid {key}"
                    )
            forbidden = [
                key
                for key in row
                if key.startswith("spai_")
                or key in ("artifact_paths", "preprocess", "classification", "t1")
            ]
            if forbidden:
                raise ValueError(
                    f"physical error row {index} has success payload {forbidden}"
                )
        else:
            raise ValueError(f"physical result row {index} has invalid status")
        counts[str(status)] += 1
    return {
        "rows_validated": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "all_physical_retries_validated": True,
    }


def validate_provenance(
    *,
    repo_root: Path,
    source_root: Path,
    inputs_path: Path,
    files: RunFiles,
    pins: SimpleNamespace,
    torch_module: ModuleType,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Validate source, selection, checkpoint, runtime-facing provenance."""

    _reject_t2_localization_or_joint(files.manifest, label="manifest")
    _reject_t2_localization_or_joint(files.summary, label="summary")
    for index, row in enumerate(files.rows):
        _reject_t2_localization_or_joint(row, label=f"result[{index}]")
    _require_equal(
        files.manifest.get("schema_version"),
        "spai_detection_run_manifest_v1",
        "manifest schema",
    )
    config = _require_mapping(files.manifest.get("config"), "manifest config")
    dataset = _require_mapping(config.get("dataset"), "config dataset")
    source_inputs = read_jsonl(_anchored(inputs_path, repo_root))
    selected = _replay_input_selection(
        source_inputs,
        pair_limit=dataset.get("pair_limit"),
        sample_id=dataset.get("sample_id"),
    )
    _require_equal(files.expected, selected, "expected-input selection")
    if "selected_ids" in dataset:
        _require_equal(
            dataset["selected_ids"],
            [row["sample_id"] for row in selected],
            "selected ids",
        )
    if "selected_rows_sha256" in dataset:
        _require_equal(
            dataset["selected_rows_sha256"],
            _selected_rows_sha256(selected),
            "selected rows SHA-256",
        )
    expected_ids = [str(row["sample_id"]) for row in selected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected inputs contain duplicate sample ids")
    latest = _latest_by_id(files.rows)
    unknown = sorted(set(latest) - set(expected_ids))
    if unknown:
        raise ValueError(f"result history contains unknown ids: {unknown}")
    history = summarize_result_history(files.rows)
    if "result_history" in files.summary:
        _compare_nested(
            files.summary["result_history"],
            history,
            label="summary result history",
            exact_mapping_keys=True,
        )
    source = _verify_source_tree(source_root, pins=pins)
    manifest_bindings = _validate_manifest_paths_and_source(
        repo_root=repo_root,
        source_root=source_root,
        inputs_path=_anchored(inputs_path, repo_root),
        files=files,
        pins=pins,
        torch_module=torch_module,
    )
    checkpoint, checkpoint_record = _checkpoint_from_manifest(
        files.manifest,
        repo_root=repo_root,
        pins=pins,
        torch_module=torch_module,
    )
    adapter = _validate_adapter_contract(config, repo_root=repo_root)
    physical_rows = _validate_physical_rows(
        rows=files.rows,
        canonical_rows=source_inputs,
        repo_root=repo_root,
        run_dir=files.run_dir,
        config_fingerprint=str(files.manifest["config_fingerprint"]),
        pins=pins,
        torch_module=torch_module,
    )
    return selected, checkpoint, {
        "source": source,
        "checkpoint_record": checkpoint_record,
        "adapter": adapter,
        "manifest_bindings": manifest_bindings,
        "selection": {
            "images": len(selected),
            "ids_sha256": hashlib.sha256(
                stable_json(expected_ids).encode("utf-8")
            ).hexdigest(),
            "selected_rows_sha256": _selected_rows_sha256(selected),
        },
        "result_history": history,
        "physical_rows": physical_rows,
    }


def _maximum_absolute_difference(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    if actual.shape != expected.shape:
        raise ValueError(
            f"array shape mismatch: {actual.shape} != {expected.shape}"
        )
    if actual.size == 0:
        return 0.0
    return float(
        np.max(
            np.abs(
                actual.astype(np.float64, copy=False)
                - expected.astype(np.float64, copy=False)
            )
        )
    )


def _require_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    tolerance: float,
) -> float:
    difference = _maximum_absolute_difference(actual, expected)
    if difference > tolerance:
        raise ValueError(
            f"{label} maximum absolute difference {difference} "
            f"exceeds {tolerance}"
        )
    return difference


def _audit_manual_replay(
    row: Mapping[str, Any],
    *,
    raw_logit: float,
    probability: float,
) -> None:
    manual = _require_mapping(
        row.get("manual_replay"),
        f"row {row.get('id')} manual replay",
    )
    required = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": probability > FIXED_THRESHOLD,
    }
    for key, expected in required.items():
        if key not in manual:
            raise ValueError(f"row {row.get('id')} manual replay lacks {key}")
        _compare_nested(
            manual[key],
            expected,
            label=f"row {row.get('id')} manual replay {key}",
        )
    exact_flags = [
        key
        for key in manual
        if key.endswith("_exact_match")
    ]
    if not exact_flags or any(manual[key] is not True for key in exact_flags):
        raise ValueError(
            f"row {row.get('id')} manual replay exact-match flags are invalid"
        )


def audit_artifacts(
    *,
    repo_root: Path,
    source_root: Path,
    all_inputs: list[dict[str, Any]],
    visibility_inputs: list[dict[str, Any]] | None = None,
    files: RunFiles,
    runtime: ReplayRuntime,
    model: Any,
    patch_feature_tolerance: float = PATCH_FEATURE_ABSOLUTE_TOLERANCE,
    feature_tolerance: float = FEATURE_ABSOLUTE_TOLERANCE,
    attention_tolerance: float = ATTENTION_ABSOLUTE_TOLERANCE,
    raw_tolerance: float = RAW_LOGIT_ABSOLUTE_TOLERANCE,
    probability_tolerance: float = PROBABILITY_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Audit all latest successful rows with fresh and artifact replays."""

    del source_root  # Source is already bound into the strictly loaded model.
    torch = runtime.torch
    latest = _latest_by_id(files.rows)
    canonical_by_id = {str(row["sample_id"]): row for row in all_inputs}
    visibility_by_task = _pair_visibility(
        visibility_inputs if visibility_inputs is not None else all_inputs,
        repo_root=repo_root,
    )
    config = _require_mapping(files.manifest.get("config"), "manifest config")
    config_fingerprint = str(
        files.manifest.get(
            "config_fingerprint",
            config.get("config_fingerprint", ""),
        )
    )
    if not config_fingerprint:
        candidates = {
            str(row.get("config_fingerprint"))
            for row in latest.values()
            if row.get("config_fingerprint") is not None
        }
        if len(candidates) != 1:
            raise ValueError("cannot resolve one SPAI config fingerprint")
        config_fingerprint = candidates.pop()
    pins = _load_runner_pins()
    maxima = {
        "patch_feature": 0.0,
        "feature": 0.0,
        "attention": 0.0,
        "raw_logit": 0.0,
        "probability": 0.0,
        "artifact_sca_feature": 0.0,
        "artifact_sca_attention": 0.0,
        "artifact_sca_logit": 0.0,
        "artifact_mlp_logit": 0.0,
    }
    mode_counts = Counter()
    visibility_counts = Counter()
    replayed_by_id: dict[str, dict[str, Any]] = {}
    audited = 0
    model = model.to(runtime.device)
    model.eval()
    for sample_id in [str(row["sample_id"]) for row in all_inputs]:
        if sample_id not in latest:
            continue
        row = latest[sample_id]
        if row.get("status") != "ok":
            continue
        canonical = canonical_by_id[sample_id]
        input_path = _validate_result_identity(
            row,
            canonical,
            repo_root=repo_root,
            config_fingerprint=config_fingerprint,
            pins=pins,
        )
        _require_equal(
            row.get("valid_for_metrics"),
            True,
            f"row {sample_id} valid_for_metrics",
        )
        _require_equal(
            row.get("valid_for_t2"),
            False,
            f"row {sample_id} valid_for_t2",
        )
        _require_equal(
            row.get("attention_is_diagnostic_not_T2"),
            True,
            f"row {sample_id} attention diagnostic scope",
        )
        prepared = preprocess_image(input_path, torch_module=torch)
        _audit_preprocess_record(row, prepared)
        expected_visibility = visibility_by_task[str(canonical["task_id"])]
        _require_equal(
            row.get("edit_visibility"),
            expected_visibility["edit_visibility"],
            f"row {sample_id} edit visibility",
        )
        _compare_float(
            row.get("edit_visible_gt_fraction"),
            expected_visibility["edit_visible_gt_fraction"],
            label=f"row {sample_id} visible GT fraction",
            tolerance=0.0,
        )
        _compare_nested(
            row.get("edit_visibility_evidence"),
            expected_visibility["edit_visibility_evidence"],
            label=f"row {sample_id} visibility evidence",
            exact_mapping_keys=True,
        )
        patch_count = int(prepared.geometry["effective_patch_count"])
        patch_np, feature_np, attention_np, _paths = _load_artifacts(
            row,
            patch_count=patch_count,
            repo_root=repo_root,
            run_dir=files.run_dir,
        )
        image = prepared.tensor.to(runtime.device, dtype=torch.float32)
        persisted_patch = torch.from_numpy(patch_np).to(runtime.device)
        persisted_feature = torch.from_numpy(feature_np).to(runtime.device)
        with torch.inference_mode():
            fresh = _forward_with_evidence(
                model,
                image,
                torch_module=torch,
                feature_extraction_batch=FEATURE_EXTRACTION_BATCH,
            )
            artifact_sca = _replay_patch_artifact(model, persisted_patch)
            artifact_mlp_logit = model.cls_head(
                persisted_feature.unsqueeze(0)
            ).reshape(())
        fresh_patch_np = (
            fresh.patch_features.detach().cpu().contiguous().numpy().astype(
                np.float32,
                copy=False,
            )
        )
        fresh_feature_np = (
            fresh.normalized_feature.detach().cpu().contiguous().numpy().astype(
                np.float32,
                copy=False,
            )
        )
        fresh_attention_np = (
            fresh.attention.detach().cpu().contiguous().numpy().astype(
                np.float32,
                copy=False,
            )
        )
        maxima["patch_feature"] = max(
            maxima["patch_feature"],
            _require_array_close(
                patch_np,
                fresh_patch_np,
                label=f"row {sample_id} fresh patch features",
                tolerance=patch_feature_tolerance,
            ),
        )
        maxima["feature"] = max(
            maxima["feature"],
            _require_array_close(
                feature_np,
                fresh_feature_np,
                label=f"row {sample_id} fresh normalized feature",
                tolerance=feature_tolerance,
            ),
        )
        maxima["attention"] = max(
            maxima["attention"],
            _require_array_close(
                attention_np,
                fresh_attention_np,
                label=f"row {sample_id} fresh attention",
                tolerance=attention_tolerance,
            ),
        )
        artifact_feature_np = (
            artifact_sca.normalized_feature.detach().cpu().numpy()
        )
        artifact_attention_np = artifact_sca.attention.detach().cpu().numpy()
        maxima["artifact_sca_feature"] = max(
            maxima["artifact_sca_feature"],
            _require_array_close(
                feature_np,
                artifact_feature_np,
                label=f"row {sample_id} artifact SCA feature",
                tolerance=feature_tolerance,
            ),
        )
        maxima["artifact_sca_attention"] = max(
            maxima["artifact_sca_attention"],
            _require_array_close(
                attention_np,
                artifact_attention_np,
                label=f"row {sample_id} artifact SCA attention",
                tolerance=attention_tolerance,
            ),
        )
        raw = float(fresh.raw_logit.detach().cpu().to(torch.float32).item())
        probability = _float32_sigmoid(raw, torch)
        score_audit = _audit_score_fields(
            row,
            replay_raw_logit=raw,
            replay_probability=probability,
            raw_tolerance=raw_tolerance,
            probability_tolerance=probability_tolerance,
        )
        maxima["raw_logit"] = max(
            maxima["raw_logit"],
            abs(float(row["raw_logit"]) - raw),
        )
        maxima["probability"] = max(
            maxima["probability"],
            abs(float(row["probability"]) - probability),
        )
        artifact_sca_raw = float(
            artifact_sca.raw_logit.detach().cpu().to(torch.float32).item()
        )
        artifact_mlp_raw = float(
            artifact_mlp_logit.detach().cpu().to(torch.float32).item()
        )
        maxima["artifact_sca_logit"] = max(
            maxima["artifact_sca_logit"],
            abs(artifact_sca_raw - raw),
        )
        maxima["artifact_mlp_logit"] = max(
            maxima["artifact_mlp_logit"],
            abs(artifact_mlp_raw - raw),
        )
        if abs(artifact_sca_raw - raw) > raw_tolerance:
            raise ValueError(
                f"row {sample_id} patch-artifact SCA logit mismatch"
            )
        if abs(artifact_mlp_raw - raw) > raw_tolerance:
            raise ValueError(
                f"row {sample_id} feature-artifact MLP logit mismatch"
            )
        _audit_manual_replay(
            row,
            raw_logit=float(row["raw_logit"]),
            probability=float(row["probability"]),
        )
        replayed = copy.deepcopy(row)
        replayed.update(
            {
                "raw_logit": raw,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "classification_decision": probability > FIXED_THRESHOLD,
            }
        )
        replayed["classification"].update(
            {
                "raw_logit": raw,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "decision": probability > FIXED_THRESHOLD,
            }
        )
        replayed["t1"].update(
            {
                "raw_logit": raw,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "decision": probability > FIXED_THRESHOLD,
            }
        )
        replayed_by_id[sample_id] = replayed
        mode_counts[prepared.geometry["patch_mode"]] += 1
        visibility_counts[expected_visibility["edit_visibility"]] += 1
        audited += 1
        del image, persisted_patch, persisted_feature, fresh, artifact_sca
    latest_line: dict[str, int] = {}
    for index, row in enumerate(files.rows):
        latest_line[str(row["id"])] = index
    independent_rows = [
        copy.deepcopy(
            replayed_by_id[str(row["id"])]
            if latest_line[str(row["id"])] == index
            and row.get("status") == "ok"
            and str(row["id"]) in replayed_by_id
            else row
        )
        for index, row in enumerate(files.rows)
    ]
    return {
        "images_audited": audited,
        "fresh_complete_fft_vit_srs_sca_mlp_forwards": audited,
        "patch_artifacts_validated": audited,
        "normalized_feature_artifacts_validated": audited,
        "attention_diagnostic_artifacts_validated": audited,
        "artifact_sca_norm_mlp_replays": audited,
        "artifact_feature_mlp_replays": audited,
        "attention_role": "classifier_diagnostic_not_T2_or_localization",
        "strict_decision_replayed": "probability > 0.5",
        "patch_mode_counts_by_image": dict(sorted(mode_counts.items())),
        "visibility_counts_by_image": dict(sorted(visibility_counts.items())),
        "maximum_absolute_differences": maxima,
        "tolerances": {
            "patch_feature": patch_feature_tolerance,
            "feature": feature_tolerance,
            "attention": attention_tolerance,
            "raw_logit": raw_tolerance,
            "probability": probability_tolerance,
        },
        "_independent_result_rows": independent_rows,
    }


def _compare_summary_payload(
    actual: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 1e-12,
) -> None:
    _compare_nested(
        actual,
        expected,
        label=label,
        float_tolerance=float_tolerance,
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
    iterations = int(
        metrics.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)
    )
    seed = int(metrics.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    _require_equal(
        float(metrics.get("fixed_threshold", FIXED_THRESHOLD)),
        FIXED_THRESHOLD,
        "metric fixed threshold",
    )
    _require_equal(
        metrics.get("threshold_operator", THRESHOLD_OPERATOR),
        THRESHOLD_OPERATOR,
        "metric threshold operator",
    )
    replayed = summarize_spai_results(
        result_rows,
        expected_rows,
        bootstrap_samples=iterations,
        seed=seed,
    )
    for key, expected in replayed.items():
        if key not in recorded_summary:
            raise ValueError(f"recorded summary lacks {key}")
        _compare_summary_payload(
            recorded_summary[key],
            expected,
            label=f"summary {key}",
        )
    result = {
        "recorded_summary_recomputed": True,
        "bootstrap_samples": iterations,
        "bootstrap_seed": seed,
        "recomputed": replayed,
    }
    if independent_result_rows is not None:
        independent = summarize_spai_results(
            independent_result_rows,
            expected_rows,
            bootstrap_samples=iterations,
            seed=seed,
        )
        _compare_summary_payload(
            independent,
            replayed,
            label="independent full-model summary",
            # The independent rows come from a fresh CUDA forward. Per-image
            # probabilities are already checked against the persisted CUDA
            # values at this frozen one-float32-ULP tolerance; aggregate
            # descriptive statistics inherit that harmless difference.
            float_tolerance=PROBABILITY_ABSOLUTE_TOLERANCE,
        )
        result["independent_full_model_summary_within_probability_tolerance"] = (
            True
        )
        result["independent_full_model_summary_float_tolerance"] = (
            PROBABILITY_ABSOLUTE_TOLERANCE
        )
    return result


def _config_without_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    dataset = _require_mapping(value.get("dataset"), "config dataset")
    for key in (
        "pair_limit",
        "sample_id",
        "selected_ids",
        "selected_rows_sha256",
        "selected_images",
        "selected_pairs",
    ):
        dataset.pop(key, None)
    return value


def audit_prefix_reproducibility(
    *,
    repo_root: Path,
    full: RunFiles,
    prefix: RunFiles,
) -> dict[str, Any]:
    """Require a separately produced prefix to match full-run evidence."""

    full_config = _require_mapping(full.manifest.get("config"), "full config")
    prefix_config = _require_mapping(prefix.manifest.get("config"), "prefix config")
    _require_equal(
        _config_without_selection(prefix_config),
        _config_without_selection(full_config),
        "prefix/full config outside selection",
    )
    for key in ("source", "assets", "runtime"):
        _require_equal(
            prefix.manifest.get(key),
            full.manifest.get(key),
            f"prefix/full manifest {key}",
        )
    prefix_expected = prefix.expected
    _require_equal(
        full.expected[: len(prefix_expected)],
        prefix_expected,
        "prefix/full expected inputs",
    )
    full_latest = _latest_by_id(full.rows)
    prefix_latest = _latest_by_id(prefix.rows)
    compared = 0
    for canonical in prefix_expected:
        row_id = str(canonical["sample_id"])
        left = full_latest[row_id]
        right = prefix_latest[row_id]
        _require_equal(left.get("status"), "ok", f"full row {row_id} status")
        _require_equal(right.get("status"), "ok", f"prefix row {row_id} status")
        for field in _PREFIX_EXACT_FIELDS:
            _require_equal(
                right.get(field),
                left.get(field),
                f"prefix/full row {row_id} {field}",
            )
        patch_count = int(
            _require_mapping(
                left.get("preprocess"),
                f"full row {row_id} preprocess",
            )["geometry"]["effective_patch_count"]
        )
        _, _, _, full_paths = _load_artifacts(
            left,
            patch_count=patch_count,
            repo_root=repo_root,
            run_dir=full.run_dir,
        )
        _, _, _, prefix_paths = _load_artifacts(
            right,
            patch_count=patch_count,
            repo_root=repo_root,
            run_dir=prefix.run_dir,
        )
        for name in full_paths:
            if full_paths[name].resolve() == prefix_paths[name].resolve():
                raise ValueError(
                    f"prefix row {row_id} reuses full-run {name} artifact"
                )
        compared += 1
    return {
        "samples_compared": compared,
        "independent_artifact_paths": True,
        "all_frozen_outputs_exact": True,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    source_root = _anchored(Path(args.source_root), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    files = _load_run_files(
        repo_root=repo_root,
        results_dir=Path(args.results_dir),
        run_id=args.run_id,
    )
    pins = _load_runner_pins()
    _require_equal(
        pins.MODEL_SOURCE_COMMIT,
        FROZEN_SOURCE_COMMIT,
        "runner/analyzer source commit",
    )
    _require_equal(
        pins.PREPROCESS_PROFILE,
        PREPROCESS_PROFILE,
        "runner/analyzer preprocess profile",
    )
    _require_equal(pins.PATCH_SIZE, PATCH_SIZE, "runner/analyzer patch size")
    _require_equal(pins.PATCH_STRIDE, PATCH_STRIDE, "runner/analyzer patch stride")
    _require_equal(
        pins.MINIMUM_PATCHES,
        MINIMUM_PATCHES,
        "runner/analyzer minimum patches",
    )
    _require_equal(
        pins.FEATURE_EXTRACTION_BATCH,
        FEATURE_EXTRACTION_BATCH,
        "runner/analyzer feature-extraction batch",
    )
    _require_equal(
        pins.FEATURE_DIMENSION,
        FEATURE_DIMENSION,
        "runner/analyzer feature dimension",
    )
    _require_equal(
        pins.ATTENTION_HEADS,
        ATTENTION_HEADS,
        "runner/analyzer attention heads",
    )
    _require_equal(
        pins.PRIMARY_DEVICE,
        PRIMARY_DEVICE,
        "runner/analyzer primary device",
    )
    runtime = _replay_runtime(files.manifest)
    selected, checkpoint, provenance = validate_provenance(
        repo_root=repo_root,
        source_root=source_root,
        inputs_path=inputs_path,
        files=files,
        pins=pins,
        torch_module=runtime.torch,
    )
    model, checkpoint_schema, checkpoint_safety = _build_and_load_model(
        source_root,
        checkpoint,
        torch_module=runtime.torch,
    )
    checkpoint_record = _require_mapping(
        provenance["checkpoint_record"],
        "checkpoint record",
    )
    if "schema" in checkpoint_record:
        _require_equal(
            checkpoint_record["schema"],
            checkpoint_schema,
            "manifest checkpoint schema",
        )
    if "serialization_safety" in checkpoint_record:
        _compare_nested(
            checkpoint_record["serialization_safety"],
            checkpoint_safety,
            label="checkpoint serialization safety",
        )
    artifact = audit_artifacts(
        repo_root=repo_root,
        source_root=source_root,
        all_inputs=selected,
        visibility_inputs=read_jsonl(inputs_path),
        files=files,
        runtime=runtime,
        model=model,
        patch_feature_tolerance=args.patch_feature_atol,
        feature_tolerance=args.feature_atol,
        attention_tolerance=args.attention_atol,
        raw_tolerance=args.raw_logit_atol,
        probability_tolerance=args.probability_atol,
    )
    independent_rows = artifact.pop("_independent_result_rows")
    summary = recompute_summary(
        result_rows=files.rows,
        expected_rows=selected,
        manifest=files.manifest,
        recorded_summary=files.summary,
        independent_result_rows=independent_rows,
    )
    result: dict[str, Any] = {
        "schema_version": "spai_independent_audit_v1",
        "run_id": args.run_id,
        "audited_at": utc_now(),
        "status": "ok",
        "scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "attention_is_diagnostic_not_T2": True,
            "joint_score_available": False,
        },
        "provenance": provenance,
        "checkpoint": {
            "path": _relative_or_absolute(checkpoint, repo_root),
            "sha256": FROZEN_CHECKPOINT_SHA256,
            "bytes": FROZEN_CHECKPOINT_BYTES,
            "schema": checkpoint_schema,
            "serialization_safety": checkpoint_safety,
            "strict_model_load": True,
        },
        "artifact_replay": artifact,
        "summary_replay": summary,
    }
    if args.prefix_run_id:
        prefix = _load_run_files(
            repo_root=repo_root,
            results_dir=Path(args.results_dir),
            run_id=args.prefix_run_id,
        )
        result["prefix_reproducibility"] = audit_prefix_reproducibility(
            repo_root=repo_root,
            full=files,
            prefix=prefix,
        )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else files.run_dir / "independent_audit.json"
    )
    atomic_write_json(output_path, result)
    result["output_path"] = _relative_or_absolute(output_path, repo_root)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--prefix-run-id")
    parser.add_argument("--output")
    parser.add_argument(
        "--patch-feature-atol",
        type=float,
        default=PATCH_FEATURE_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--feature-atol",
        type=float,
        default=FEATURE_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--attention-atol",
        type=float,
        default=ATTENTION_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--raw-logit-atol",
        type=float,
        default=RAW_LOGIT_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--probability-atol",
        type=float,
        default=PROBABILITY_ABSOLUTE_TOLERANCE,
    )
    args = parser.parse_args()
    for name in (
        "patch_feature_atol",
        "feature_atol",
        "attention_atol",
        "raw_logit_atol",
        "probability_atol",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def main() -> None:
    result = analyze(parse_args())
    print(stable_json(result))


if __name__ == "__main__":
    main()
