#!/usr/bin/env python3
"""Run the pinned official SPAI any-resolution spectral detector.

The primary release was frozen before inspecting any Mouse model score.  This
adapter verifies the official source/config/checkpoint, loads the checkpoint
with PyTorch's restricted weights-only unpickler, executes the released
native-resolution path offline in float32, and persists sufficient internal
evidence to replay the complete SCA/norm/MLP scoring chain.  SPAI is a
whole-image classifier (T1); its SCA weights are diagnostic evidence, not a
manipulation-localization prediction (T2).
"""

from __future__ import annotations

import argparse
import gc
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
import tempfile
import time
import traceback
import urllib.request
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    read_latest_by_id,
    repo_relative,
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


MODEL_NAME = "SPAI"
MODEL_SLUG = "spai_any_resolution_spectral"
MODEL_ARCH = "official PatchBasedMFViT ViT-B/16 frequency restoration"
MODEL_REPO_URL = "https://github.com/mever-team/spai"
MODEL_SOURCE_COMMIT = "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
WEBSITE_SOURCE_COMMIT = "861a85e554e24fd5079e1ad9a37d141f92714da8"
PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/html/"
    "Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_"
    "Spectral_Learning_CVPR_2025_paper.html"
)

PREPROCESS_PROFILE = "official_pillow_rgb_native_float32_0_1"
MODEL_SEED = 0
PRIMARY_DEVICE = "cuda:0"
PATCH_SIZE = 224
PATCH_STRIDE = 224
MINIMUM_PATCHES = 4
FEATURE_EXTRACTION_BATCH = 400
FEATURE_DIMENSION = 1096
ATTENTION_HEADS = 12
ATTENTION_EMBED_DIMENSION = 1536
FEATURE_SEMANTICS = (
    "spectral_context_attention_layernorm_output_before_complete_mlp_head"
)
PATCH_FEATURE_SEMANTICS = (
    "per_patch_frequency_restoration_features_before_"
    "spectral_context_attention"
)
ATTENTION_SEMANTICS = (
    "spectral_context_attention_softmax_weights_classifier_diagnostic_"
    "not_localization"
)
SCORE_SEMANTICS = "torch_float32_sigmoid_of_single_raw_logit"
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
T1_POLICY = "released_probability_strictly_greater_than_0_5"

CHECKPOINT = {
    "id": "official-google-drive-1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI",
    "filename": "spai.pth",
    "bytes": 934_865_338,
    "sha256": "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55",
    "format": "torch_checkpoint_weights_only_with_yacs_cfgnode_allowlist",
    "tensor_count": 324,
    "state_elements": 139_945_243,
    "schema_items_sha256": (
        "ffe751246ec65936d5583a1db62bf617697484e6185f1bfad7c678f1dad36ef8"
    ),
    "embedded_minimum_patches": 1,
}

SOURCE_FILES = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "README.md": "2de3bf7b8d10fbe9d2990e3a391656d6aaeae599edef0d53589f68f231e1b30c",
    "requirements.txt": "d449bf03fc2e3b0cce3b9985df0ae491f10afb22c4dc948595f9fdd88ff103c4",
    "configs/spai.yaml": "66a3caee07d9af23de453eb604709d4a821aab41a6acfb983a5c9be9d05f3586",
    "spai/__init__.py": "c8cbcf96bb07ad2c517efbea09b7a9fc07f7cb143082caa5b2d3f3f41f3b7b94",
    "spai/config.py": "7ebda3d1862985870f2d70272dc80933af3ceb66b6df85f2f4d0a901299cb893",
    "spai/utils.py": "b5eea98152239e60e96ea8cf8a416fb27ce06c9cf9f1ddba61da3a3e79c0f770",
    "spai/models/__init__.py": "13abb55acc1e51dea398fc097aeb1ed348c8ba8f5f7fc5c5a71677bea4c4f538",
    "spai/models/build.py": "d89eaade77bcb58d9b0de08ddbd0c7eec94b54dc08958cc169b3a06e47faddcd",
    "spai/models/sid.py": "f748716a9d6223d36e5d60966193d5c46b1b26da0d475610b75bfc7ff9192c64",
    "spai/models/vision_transformer.py": "1ff53dbf64d9909b852c4727eebf7711265f2d8cdb644801a9ec3ff611c3acc5",
    "spai/models/filters.py": "0d8006e6b8af445bac7a9edb9c48f69216b92b2cdbd5a2b94dea578666f17c4c",
    "spai/models/utils.py": "5c9940ecfb5002b3d0b1655f413987ac1a13214394657533719d47193227a507",
    "spai/models/backbones.py": "5d048da7d6a43d9a4236f35e2d6746bc29457cc561943fb565764321c63c315d",
    "spai/models/swin_transformer.py": "b3bcc34ce1ab685d1dcf39cb61d5b76d280bb04a9639dc497c217c1db1b85388",
    "spai/models/mfm.py": "795933cf358ac0daf3196a20746a646a43005c1a3e4a1ff2a8b58757df46d5be",
    "spai/models/frequency_loss.py": "6edc01f7cebbfd18fe0a29d50aa0952becb3aa6f53fe733e39498ba7d950db80",
    "spai/data/__init__.py": "5cf52429272b45a40052857853656787ec67bec7acf26b41bb880473666ea0a9",
    "spai/data/data_finetune.py": "9b90a95414ec697fb7e9ef465e160624913e16854da679477fa4b440a85d4791",
    "spai/data/readers.py": "11cbc79598e5d8d96686f3b2e80cbdec10cf0310c124b9827a88b542781b02e9",
}

RUNTIME_VERSIONS = {
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
RUNTIME_MODULE_FILES = {
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

WEBSITE_INDEX_SHA256 = (
    "0f028c7f9860065d63fb757be27142ad426360d6f2895c3a54b244891f5a2891"
)
WEBSITE_DISPLAY_REFERENCES = (
    {
        "filename": "mj61_224-0.748.jpg",
        "git_path": "static/images/carousel_all/mj61_224-0.748.jpg",
        "sha256": "972a99c300260cdee0fd97d0b7fbc6f666d85744cd2ae566f7012db39186ed2f",
        "displayed_probability": 0.748,
        "displayed_decimals": 3,
    },
    {
        "filename": "dalle2_r1db5be20t-0.99.jpg",
        "git_path": "static/images/carousel_all/dalle2_r1db5be20t-0.99.jpg",
        "sha256": "096090cf54407b596c524ac2456d6899d1ac43580a90cdaa20f728ec1a649035",
        "displayed_probability": 0.99,
        "displayed_decimals": 2,
    },
    {
        "filename": "coco_000000019115-0.0.jpg",
        "git_path": "static/images/carousel_all/coco_000000019115-0.0.jpg",
        "sha256": "e7a1b0fff7a918e6830920fa79025f505621525aa8e21d8650466e689ff6f1ab",
        "displayed_probability": 0.0,
        "displayed_decimals": 1,
    },
    {
        "filename": "openimages_65b410839323ec3d-0.09.jpg",
        "git_path": "static/images/carousel_all/openimages_65b410839323ec3d-0.09.jpg",
        "sha256": "7e80a4011c5a5bcd8dbdc3aaa640de514b05d4818b8b9993de7788f8917976a1",
        "displayed_probability": 0.09,
        "displayed_decimals": 2,
    },
)
GOLDEN_CASES = (
    {
        "relative_path": "midjourney-v6.1/224.png",
        "sha256": "e41a6f0832d363a110f6821a1c6e2120b1f0187345bf652e2fc60125a8c4ea2b",
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
        "sha256": "bf4c5d6e784346c4dd49adf08c6223f6da93080ca1782f678b29d7cdfed8b386",
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
GOLDEN_LOGIT_ABS_TOLERANCE = 1e-6
GOLDEN_PROBABILITY_ABS_TOLERANCE = 1e-7

LICENSE_RECORD = {
    "source_and_weights": {
        "spdx": "Apache-2.0",
        "license_sha256": SOURCE_FILES["LICENSE"],
        "official_readme_statement": (
            "source code and model weights released under Apache License 2.0"
        ),
    },
    "overall_commercial_clearance": (
        "Apache-2.0 release metadata present for official code and weights"
    ),
}

CANONICAL_RELEASE = {
    "schema_version": "claimforge_mouse_canonical_v1",
    "dataset_id": "claimforge-mouse-good275-canonical-jpeg-q95-v1",
    "pairs": 275,
    "images": 550,
    "inputs_sha256": "e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a",
    "pairs_sha256": "bb6328be7cc7d4ae74b1e5b0b132f7fb6133c6fe73f294ebb46aebeda4f8f4b8",
    "contract_sha256": "c419e24d6f9d69822ca575e00e30f2c769ba7a28a2fcea1f6634466caf540757",
}

FROZEN_VISIBILITY = {
    "pairs": 275,
    "census": {"full": 243, "partial": 14, "none": 18},
    "mean_edit_visible_gt_fraction": 0.9096355444251016,
    "patch_modes": {"grid": 262, "five_crop": 13},
    "by_domain": {
        "lodging": {"full": 132, "partial": 3, "none": 12},
        "restaurant": {"full": 111, "partial": 11, "none": 6},
    },
}

DEFAULT_SOURCE_ROOT = Path("/root/.cache/claimforge/third_party/spai-8ff7b3b6")
DEFAULT_CHECKPOINT = Path("/root/.cache/claimforge/third_party/spai.pth")
DEFAULT_GOLDEN_ROOT = Path(
    "/root/.cache/claimforge/third_party/spai-official-originals"
)
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/spai")
DEFAULT_RUN_ID = (
    "spai_any_resolution_spectral_"
    "mouse_canonical_v1_full275_20260725"
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_bytes(repo: Path, object_name: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", object_name],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot read pinned git object {object_name}") from exc


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _tensor_sha256(value: Any) -> str:
    return _array_sha256(value.detach().cpu().contiguous().numpy())


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npy",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def adapter_contract(repo_root: Path) -> dict[str, Any]:
    relative_paths = (
        "eval/opensource/run_spai.py",
        "eval/opensource/spai_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
    )
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing SPAI adapter component: {path}")
        result[relative] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def load_release(
    repo_root: Path,
    dataset_manifest_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    release = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    for key, expected in CANONICAL_RELEASE.items():
        if release.get(key) != expected:
            raise ValueError(
                f"canonical release field {key} changed: "
                f"{release.get(key)!r} != {expected!r}"
            )
    inputs_value = release.get("inputs_path")
    pairs_value = release.get("pairs_path")
    if not isinstance(inputs_value, str) or not isinstance(pairs_value, str):
        raise ValueError("canonical release is missing inputs/pairs path")
    inputs_path = _anchored(Path(inputs_value), repo_root)
    pairs_path = _anchored(Path(pairs_value), repo_root)
    _verify_file(
        inputs_path,
        str(release["inputs_sha256"]),
        "canonical inputs.jsonl",
    )
    _verify_file(
        pairs_path,
        str(release["pairs_sha256"]),
        "canonical pairs.jsonl",
    )
    rows = read_jsonl(inputs_path)
    if (
        len(rows) != int(release["images"])
        or len(read_jsonl(pairs_path)) != int(release["pairs"])
    ):
        raise ValueError("canonical release row count changed")
    ranks = [int(row["rank"]) for row in rows]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("canonical inputs are not in unique rank order")
    sample_ids = [
        _safe_component(row.get("sample_id"), label="canonical sample_id")
        for row in rows
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("canonical inputs contain duplicate sample IDs")
    return release, inputs_path, rows


def select_inputs(
    rows: list[dict[str, Any]],
    pair_limit: int | None,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    if pair_limit is not None and sample_id is not None:
        raise ValueError("pair-limit and sample-id are mutually exclusive")
    if sample_id is not None:
        _safe_component(sample_id, label="sample-id")
        selected = [
            row for row in rows if str(row.get("sample_id")) == sample_id
        ]
        if len(selected) != 1:
            raise ValueError(
                f"sample-id must select exactly one row: {sample_id}"
            )
        return selected
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair-limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    chosen = set(pair_ranks)
    selected = [
        row for row in rows if int(row["pair_rank"]) in chosen
    ]
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        by_rank.setdefault(int(row["pair_rank"]), []).append(row)
    for rank, pair in by_rank.items():
        if (
            sorted(str(row["kind"]) for row in pair) != ["forged", "real"]
            or len({str(row["task_id"]) for row in pair}) != 1
        ):
            raise ValueError(f"canonical selection has invalid pair {rank}")
    return selected


def _load_gt_mask(
    row: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    sample_id = str(row.get("sample_id"))
    width, height = int(row["width"]), int(row["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions for {sample_id}")
    if row.get("kind") == "real":
        if (
            row.get("label") != 0
            or row.get("gt_mask_kind") != "all_zero"
            or row.get("gt_mask_path") is not None
            or row.get("gt_mask_sha256") is not None
            or int(row.get("gt_positive_pixels", -1)) != 0
        ):
            raise ValueError(f"invalid real GT contract for {sample_id}")
        return None
    if (
        row.get("kind") != "forged"
        or row.get("label") != 1
        or row.get("gt_mask_kind") != "exact_diff"
    ):
        raise ValueError(f"invalid forged GT contract for {sample_id}")
    path_value = row.get("gt_mask_path")
    digest = row.get("gt_mask_sha256")
    if not isinstance(path_value, str) or not _valid_sha256(digest):
        raise ValueError(f"invalid forged GT path/hash for {sample_id}")
    path = _anchored(Path(path_value), repo_root)
    _verify_file(path, str(digest), f"GT mask {sample_id}")
    with Image.open(path) as opened:
        pixels = np.asarray(opened)
    if (
        pixels.ndim != 2
        or pixels.shape != (height, width)
        or not np.isin(pixels, (0, 255)).all()
    ):
        raise ValueError(f"invalid exact-diff GT for {sample_id}")
    positive = int(np.count_nonzero(pixels == 255))
    if positive <= 0 or positive != int(row["gt_positive_pixels"]):
        raise ValueError(f"GT positive-pixel count changed for {sample_id}")
    return np.asarray(pixels, dtype=np.uint8)


def validate_selected_inputs(
    selected: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    for row in selected:
        sample_id = str(row["sample_id"])
        path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_file(
            path,
            str(row["canonical_sha256"]),
            f"canonical input {sample_id}",
        )
        with Image.open(path) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"canonical dimensions changed for {sample_id}")
        _load_gt_mask(row, repo_root)


def compute_patch_geometry(width: int, height: int) -> dict[str, Any]:
    """Return exact official unfold/five-crop geometry for one native image."""

    if width <= 0 or height <= 0:
        raise ValueError("SPAI native dimensions must be positive")
    pad_left = int((max(PATCH_SIZE, width) - width) / 2.0)
    pad_right = max(PATCH_SIZE, width) - width - pad_left
    pad_top = int((max(PATCH_SIZE, height) - height) / 2.0)
    pad_bottom = max(PATCH_SIZE, height) - height - pad_top
    processed_width = width + pad_left + pad_right
    processed_height = height + pad_top + pad_bottom
    grid_columns = 1 + (processed_width - PATCH_SIZE) // PATCH_STRIDE
    grid_rows = 1 + (processed_height - PATCH_SIZE) // PATCH_STRIDE
    grid_count = grid_columns * grid_rows
    common = {
        "profile_id": PREPROCESS_PROFILE,
        "native_size": [width, height],
        "model_input_size": [processed_width, processed_height],
        "pad_if_needed": {
            "enabled": any((pad_left, pad_right, pad_top, pad_bottom)),
            "minimum_size": [PATCH_SIZE, PATCH_SIZE],
            "left": pad_left,
            "right": pad_right,
            "top": pad_top,
            "bottom": pad_bottom,
            "position": "center",
            "border_mode": "cv2.BORDER_REFLECT_101",
            "border_mode_value": 4,
        },
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "patch_stride": [PATCH_STRIDE, PATCH_STRIDE],
        "minimum_patches": MINIMUM_PATCHES,
        "initial_grid": {
            "rows": grid_rows,
            "columns": grid_columns,
            "count": grid_count,
        },
    }
    if grid_count >= MINIMUM_PATCHES:
        return {
            **common,
            "patch_mode": "grid",
            "effective_patch_count": grid_count,
            "grid_covered_xyxy": [
                0,
                0,
                (grid_columns - 1) * PATCH_STRIDE + PATCH_SIZE,
                (grid_rows - 1) * PATCH_STRIDE + PATCH_SIZE,
            ],
            "five_crop_boxes_xyxy": None,
            "remainder_policy": (
                "torch_tensor_unfold_discards_nondivisible_right_bottom"
            ),
        }
    center_left = int(round((processed_width - PATCH_SIZE) / 2.0))
    center_top = int(round((processed_height - PATCH_SIZE) / 2.0))
    boxes = [
        [0, 0, PATCH_SIZE, PATCH_SIZE],
        [
            processed_width - PATCH_SIZE,
            0,
            processed_width,
            PATCH_SIZE,
        ],
        [
            0,
            processed_height - PATCH_SIZE,
            PATCH_SIZE,
            processed_height,
        ],
        [
            processed_width - PATCH_SIZE,
            processed_height - PATCH_SIZE,
            processed_width,
            processed_height,
        ],
        [
            center_left,
            center_top,
            center_left + PATCH_SIZE,
            center_top + PATCH_SIZE,
        ],
    ]
    return {
        **common,
        "patch_mode": "five_crop",
        "effective_patch_count": 5,
        "grid_covered_xyxy": None,
        "five_crop_boxes_xyxy": boxes,
        "remainder_policy": (
            "torchvision_five_crop_when_initial_grid_has_fewer_than_four"
        ),
    }


def _gt_visibility(
    forged_row: Mapping[str, Any],
    gt: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    positive_y, positive_x = np.nonzero(gt == 255)
    total = int(positive_x.size)
    if total <= 0:
        raise ValueError("forged exact-diff GT has no positive pixels")
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
    return {
        "category": (
            "none" if visible == 0 else "full" if visible == total else "partial"
        ),
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixels": visible,
        "forged_sample_id": str(forged_row["sample_id"]),
        "basis": (
            "exact_diff_positive_pixels_in_union_of_official_native_"
            "resolution_patch_receptive_fields"
        ),
        "geometry": dict(geometry),
    }


def build_pair_visibility(
    all_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        task_id, kind = str(row["task_id"]), str(row["kind"])
        if kind in pairs.setdefault(task_id, {}):
            raise ValueError(f"duplicate canonical {kind} row: {task_id}")
        pairs[task_id][kind] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in pairs.items():
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"canonical task is incomplete: {task_id}")
        real, forged = pair["real"], pair["forged"]
        if (
            int(real["width"]) != int(forged["width"])
            or int(real["height"]) != int(forged["height"])
            or real.get("domain") != forged.get("domain")
            or real.get("edit_region_xyxy") != forged.get("edit_region_xyxy")
        ):
            raise ValueError(f"canonical pair geometry mismatch: {task_id}")
        gt = _load_gt_mask(forged, repo_root)
        assert gt is not None
        evidence = _gt_visibility(
            forged,
            gt,
            compute_patch_geometry(
                int(forged["width"]),
                int(forged["height"]),
            ),
        )
        result[task_id] = {
            "domain": str(forged["domain"]),
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "patch_mode": evidence["geometry"]["patch_mode"],
            "edit_visibility_evidence": evidence,
        }
    return result


def validate_frozen_visibility_census(
    visibility: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    census = dict(
        Counter(
            str(value["edit_visibility"]) for value in visibility.values()
        )
    )
    patch_modes = dict(
        Counter(str(value["patch_mode"]) for value in visibility.values())
    )
    by_domain: dict[str, dict[str, int]] = {}
    for value in visibility.values():
        domain = str(value["domain"])
        category = str(value["edit_visibility"])
        by_domain.setdefault(
            domain,
            {"full": 0, "partial": 0, "none": 0},
        )[category] += 1
    fractions = [
        float(value["edit_visible_gt_fraction"])
        for value in visibility.values()
    ]
    mean_fraction = float(np.mean(fractions))
    if (
        len(visibility) != FROZEN_VISIBILITY["pairs"]
        or census != FROZEN_VISIBILITY["census"]
        or patch_modes != FROZEN_VISIBILITY["patch_modes"]
        or by_domain != FROZEN_VISIBILITY["by_domain"]
        or not math.isclose(
            mean_fraction,
            float(FROZEN_VISIBILITY["mean_edit_visible_gt_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError(
            "SPAI frozen visibility census changed: "
            f"pairs={len(visibility)}, census={census}, "
            f"patch_modes={patch_modes}, by_domain={by_domain}, "
            f"mean={mean_fraction}"
        )
    return {
        **FROZEN_VISIBILITY,
        "mean_edit_visible_gt_fraction": mean_fraction,
        "basis": (
            "exact_diff_positive_pixels_in_union_of_official_native_"
            "resolution_patch_receptive_fields"
        ),
        "role": "input_condition_stratum_not_model_localization",
    }


def _verify_source_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing SPAI source-root: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"SPAI source commit mismatch: {commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"SPAI tracked source tree is dirty: {dirty[:1000]}")
    for relative, digest in SOURCE_FILES.items():
        _verify_file(
            source_root / relative,
            digest,
            f"SPAI source {relative}",
        )

    website_index = _git_bytes(
        source_root,
        f"{WEBSITE_SOURCE_COMMIT}:index.html",
    )
    if hashlib.sha256(website_index).hexdigest() != WEBSITE_INDEX_SHA256:
        raise ValueError("SPAI website index blob changed")
    website_text = website_index.decode("utf-8")
    website_assets: list[dict[str, Any]] = []
    for case in WEBSITE_DISPLAY_REFERENCES:
        blob = _git_bytes(
            source_root,
            f"{WEBSITE_SOURCE_COMMIT}:{case['git_path']}",
        )
        actual = hashlib.sha256(blob).hexdigest()
        if actual != case["sha256"]:
            raise ValueError(f"SPAI website image changed: {case['git_path']}")
        displayed = str(case["displayed_probability"])
        if (
            str(case["git_path"]) not in website_text
            or f">{displayed}<" not in website_text
        ):
            raise ValueError("SPAI website displayed-score evidence changed")
        website_assets.append(
            {
                **case,
                "git_object": (
                    f"{WEBSITE_SOURCE_COMMIT}:{case['git_path']}"
                ),
                "bytes": len(blob),
            }
        )

    readme = (source_root / "README.md").read_text(encoding="utf-8")
    config_text = (
        source_root / "configs" / "spai.yaml"
    ).read_text(encoding="utf-8")
    sid_text = (
        source_root / "spai" / "models" / "sid.py"
    ).read_text(encoding="utf-8")
    required = {
        "README.md": (
            "python -m spai infer",
            "should be placed under the `weights` directory",
            "Apache 2 License",
        ),
        "configs/spai.yaml": (
            'SID_APPROACH: "freq_restoration"',
            'RESOLUTION_MODE: "arbitrary"',
            "FEATURE_EXTRACTION_BATCH: 400",
            "MINIMUM_PATCHES: 4",
            "ORIGINAL_RESOLUTION: True",
        ),
        "spai/models/sid.py": (
            "forward_arbitrary_resolution_batch",
            "patched.size(1) < self.minimum_patches",
            "five_crop(",
            "x = self.norm(x)",
            "x = self.cls_head(x)",
        ),
    }
    missing: dict[str, list[str]] = {}
    for name, text, needles in (
        ("README.md", readme, required["README.md"]),
        ("configs/spai.yaml", config_text, required["configs/spai.yaml"]),
        ("spai/models/sid.py", sid_text, required["spai/models/sid.py"]),
    ):
        absent = [needle for needle in needles if needle not in text]
        if absent:
            missing[name] = absent
    if missing:
        raise ValueError(f"official SPAI semantic evidence changed: {missing}")

    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str((source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in SOURCE_FILES.items()
        },
        "website": {
            "commit": WEBSITE_SOURCE_COMMIT,
            "index_sha256": WEBSITE_INDEX_SHA256,
            "display_references": website_assets,
            "role": (
                "weak human-facing display reference; compressed derivative "
                "scores are disclosed but are not the executable golden gate"
            ),
        },
        "license_record": LICENSE_RECORD,
    }


def _checkpoint_state(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from yacs.config import CfgNode

    with torch.serialization.safe_globals([CfgNode]):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, dict) or tuple(payload) != (
        "model",
        "optimizer",
        "lr_scheduler",
        "max_accuracy",
        "epoch",
        "config",
        "amp",
    ):
        raise ValueError("SPAI checkpoint top-level schema changed")
    state = payload.get("model")
    embedded_config = payload.get("config")
    if not isinstance(state, Mapping) or not isinstance(embedded_config, CfgNode):
        raise ValueError("SPAI checkpoint model/config payload changed")
    embedded_minimum = int(
        embedded_config.MODEL.PATCH_VIT.MINIMUM_PATCHES
    )
    if embedded_minimum != int(CHECKPOINT["embedded_minimum_patches"]):
        raise ValueError("SPAI embedded historical minimum-patches changed")

    items: list[dict[str, Any]] = []
    for key, tensor in state.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
        ):
            raise ValueError(f"SPAI state entry is invalid: {key!r}")
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"SPAI state tensor is non-finite: {key}")
        if tensor.dtype not in (torch.float32, torch.int64):
            raise ValueError(f"SPAI state tensor has unexpected dtype: {key}")
        items.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "sha256": _tensor_sha256(tensor),
            }
        )
    total = sum(item["numel"] for item in items)
    if (
        len(items) != int(CHECKPOINT["tensor_count"])
        or total != int(CHECKPOINT["state_elements"])
        or sum(item["dtype"] == "torch.int64" for item in items) != 1
    ):
        raise ValueError("SPAI checkpoint tensor schema changed")
    items_sha256 = hashlib.sha256(
        stable_json(items).encode("utf-8")
    ).hexdigest()
    if items_sha256 != CHECKPOINT["schema_items_sha256"]:
        raise ValueError("SPAI checkpoint per-tensor schema/hash changed")
    schema = {
        "tensor_count": len(items),
        "state_elements": total,
        "dtype_counts": dict(Counter(item["dtype"] for item in items)),
        "items_sha256": items_sha256,
        "items": items,
        "embedded_config_minimum_patches": embedded_minimum,
        "released_inference_config_minimum_patches": MINIMUM_PATCHES,
        "embedded_config_is_historical_not_restored": True,
    }
    del payload
    return dict(state), schema


def verify_assets(
    *,
    source_root: Path,
    checkpoint_path: Path,
    golden_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _verify_source_contract(source_root)
    _verify_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "official SPAI checkpoint",
    )
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("SPAI checkpoint byte size changed")
    state, schema = _checkpoint_state(checkpoint_path)

    golden_assets: list[dict[str, Any]] = []
    for case in GOLDEN_CASES:
        path = golden_root / str(case["relative_path"])
        _verify_file(path, str(case["sha256"]), "SPAI regression golden")
        with Image.open(path) as opened:
            size = list(opened.size)
        if size != case["native_size"]:
            raise ValueError(f"SPAI golden native size changed: {path}")
        golden_assets.append(
            {
                **case,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
            }
        )
    assets = {
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path.resolve()),
            "actual_bytes": checkpoint_path.stat().st_size,
            "actual_sha256": sha256_file(checkpoint_path),
            "serialization_safety": {
                "weights_only": True,
                "pickle_executed": False,
                "safe_global_allowlist": ["yacs.config.CfgNode"],
                "loader": "torch.load(map_location=cpu, weights_only=True)",
            },
            "schema": schema,
        },
        "golden_assets": golden_assets,
    }
    return source, assets, state


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import torch

    device = torch.device(device_text)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("SPAI supports only cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("pinned SPAI runtime requires Python 3.12")

    actual_versions = {
        name: _package_version(name) for name in RUNTIME_VERSIONS
    }
    if actual_versions != RUNTIME_VERSIONS:
        raise ValueError(
            f"SPAI runtime version mismatch: {actual_versions} "
            f"!= {RUNTIME_VERSIONS}"
        )
    runtime_files: dict[str, Any] = {}
    for module_name, expected in RUNTIME_MODULE_FILES.items():
        module = importlib.import_module(module_name)
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            raise ValueError(f"runtime module has no file: {module_name}")
        path = Path(value).resolve()
        _verify_file(path, expected, f"runtime module {module_name}")
        runtime_files[module_name] = {
            "path": str(path),
            "sha256": expected,
        }

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    runtime_truth = {
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
    expected_truth = {
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }
    if runtime_truth != expected_truth:
        raise RuntimeError(f"SPAI deterministic runtime failed: {runtime_truth}")
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": actual_versions,
        "module_files": runtime_files,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "seed": MODEL_SEED,
        "dtype": "float32",
        "autocast": False,
        "tf32": False,
        "network_allowed": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "no_albumentations_update": os.environ[
            "NO_ALBUMENTATIONS_UPDATE"
        ],
        **runtime_truth,
    }
    return device, runtime


def _official_modules(source_root: Path) -> tuple[Any, Any]:
    resolved = source_root.resolve()
    existing = sys.modules.get("spai")
    if existing is not None:
        value = getattr(existing, "__file__", None)
        if not isinstance(value, str) or resolved not in Path(value).resolve().parents:
            raise RuntimeError("a non-pinned spai module is already imported")
    source_text = str(resolved)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    config_module = importlib.import_module("spai.config")
    models_module = importlib.import_module("spai.models")
    for name in ("spai", "spai.config", "spai.models", "spai.models.sid"):
        module = sys.modules.get(name)
        value = getattr(module, "__file__", None)
        if not isinstance(value, str) or resolved not in Path(value).resolve().parents:
            raise RuntimeError(f"SPAI module escaped pinned source: {name}")
    return config_module, models_module


def load_model(
    *,
    state: Mapping[str, Any],
    source_root: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    network_attempts = {
        "urllib_urlopen": 0,
        "socket_create_connection": 0,
        "socket_connect": 0,
        "torch_hub_load": 0,
        "torch_hub_load_state_dict_from_url": 0,
    }

    def reject(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            network_attempts[name] += 1
            raise RuntimeError(
                "network access is forbidden during SPAI construction"
            )

        return blocked

    with ExitStack() as stack:
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
        stack.enter_context(
            mock.patch.object(
                torch.hub,
                "load",
                side_effect=reject("torch_hub_load"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch.hub,
                "load_state_dict_from_url",
                side_effect=reject("torch_hub_load_state_dict_from_url"),
            )
        )
        config_module, models_module = _official_modules(source_root)
        config = config_module.get_custom_config(
            str(source_root / "configs" / "spai.yaml")
        )
        model = models_module.build_cls_model(config)
    if any(network_attempts.values()):
        raise RuntimeError(
            f"SPAI construction attempted network: {network_attempts}"
        )
    frozen = {
        "sid_approach": str(config.MODEL.SID_APPROACH),
        "resolution_mode": str(config.MODEL.RESOLUTION_MODE),
        "required_normalization": str(config.MODEL.REQUIRED_NORMALIZATION),
        "image_patch_size": int(config.DATA.IMG_SIZE),
        "patch_stride": int(config.MODEL.PATCH_VIT.PATCH_STRIDE),
        "minimum_patches": int(config.MODEL.PATCH_VIT.MINIMUM_PATCHES),
        "feature_extraction_batch": int(
            config.MODEL.FEATURE_EXTRACTION_BATCH
        ),
        "num_classes": int(config.MODEL.NUM_CLASSES),
        "attention_heads": int(config.MODEL.PATCH_VIT.NUM_HEADS),
        "attention_embed_dimension": int(
            config.MODEL.PATCH_VIT.ATTN_EMBED_DIM
        ),
        "original_resolution": bool(config.TEST.ORIGINAL_RESOLUTION),
    }
    expected = {
        "sid_approach": "freq_restoration",
        "resolution_mode": "arbitrary",
        "required_normalization": "positive_0_1",
        "image_patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "minimum_patches": MINIMUM_PATCHES,
        "feature_extraction_batch": FEATURE_EXTRACTION_BATCH,
        "num_classes": 2,
        "attention_heads": ATTENTION_HEADS,
        "attention_embed_dimension": ATTENTION_EMBED_DIMENSION,
        "original_resolution": True,
    }
    if frozen != expected:
        raise ValueError(f"released SPAI model config changed: {frozen}")
    model_keys = list(model.state_dict())
    state_keys = list(state)
    if set(model_keys) != set(state_keys):
        raise ValueError(
            "SPAI full-state keys changed: "
            f"missing={sorted(set(model_keys) - set(state_keys))[:10]}, "
            f"unexpected={sorted(set(state_keys) - set(model_keys))[:10]}"
        )
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("strict SPAI state load reported incompatibilities")
    for key, loaded in model.state_dict().items():
        if not torch.equal(loaded.detach().cpu(), state[key]):
            raise ValueError(f"loaded SPAI tensor differs: {key}")
    if (
        model.minimum_patches != MINIMUM_PATCHES
        or model.img_patch_size != PATCH_SIZE
        or model.img_patch_stride != PATCH_STRIDE
        or model.cls_vector_dim != FEATURE_DIMENSION
        or model.heads != ATTENTION_HEADS
    ):
        raise ValueError("constructed SPAI model contract changed")
    model.eval()
    if model.training or any(
        module.training for module in model.modules()
    ):
        raise RuntimeError("SPAI eval mode did not propagate")
    model.to(device)
    return model, {
        "config": frozen,
        "load": {
            "strict": True,
            "full_state_coverage": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "loaded_tensor_exact_match": True,
        },
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "state_tensors": len(model_keys),
            "state_elements": sum(
                int(tensor.numel()) for tensor in model.state_dict().values()
            ),
            "feature_dimension": FEATURE_DIMENSION,
            "attention_heads": ATTENTION_HEADS,
            "eval": True,
        },
        "network": {
            "allowed": False,
            "attempts": network_attempts,
        },
    }


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Mirror official Pillow RGB + Albumentations Normalize(0, 1)."""

    import cv2

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.ascontiguousarray(
            np.asarray(rgb, dtype=np.uint8)
        )
    geometry = compute_patch_geometry(width, height)
    padding = geometry["pad_if_needed"]
    if padding["enabled"]:
        processed = cv2.copyMakeBorder(
            decoded,
            int(padding["top"]),
            int(padding["bottom"]),
            int(padding["left"]),
            int(padding["right"]),
            cv2.BORDER_REFLECT_101,
        )
    else:
        processed = decoded
    processed = np.ascontiguousarray(processed, dtype=np.uint8)
    # Albumentations 1.4.14's uint8 normalize LUT multiplies by the float32
    # reciprocal instead of performing one float32 division per value.
    hwc = np.ascontiguousarray(
        processed.astype(np.float32) * np.float32(1.0 / 255.0)
    )
    tensor = np.ascontiguousarray(hwc.transpose(2, 0, 1))
    processed_width, processed_height = geometry["model_input_size"]
    if (
        decoded.shape != (height, width, 3)
        or decoded.dtype != np.uint8
        or processed.shape != (processed_height, processed_width, 3)
        or tensor.shape != (3, processed_height, processed_width)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise ValueError("SPAI preprocessing contract changed")
    return tensor, {
        "profile": PREPROCESS_PROFILE,
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "resize": False,
        "crop": False,
        "test_augmentation": False,
        "native_size": [width, height],
        "model_input_size": [processed_width, processed_height],
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "padded_rgb_shape": list(processed.shape),
        "padded_rgb_dtype": str(processed.dtype),
        "padded_rgb_sha256": _array_sha256(processed),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_sha256": _array_sha256(tensor),
        "scale": (
            "uint8_to_float32_multiply_float32_reciprocal_1_over_255_"
            "matching_albumentations_1_4_14_normalize_lut"
        ),
        "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "value_min": float(tensor.min()),
        "value_max": float(tensor.max()),
        "geometry": geometry,
    }


def validate_official_preprocess_equivalence() -> dict[str, Any]:
    """Byte-compare the adapter with the pinned official test transform."""

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=PATCH_SIZE,
                min_width=PATCH_SIZE,
            ),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ]
    )
    shapes = ((13, 17), (231, 449))
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="spai-preprocess-") as directory:
        root = Path(directory)
        for index, (height, width) in enumerate(shapes):
            values = (
                np.arange(height * width * 3, dtype=np.uint64)
                .reshape(height, width, 3)
                * 37
                + 19
            ) % 256
            decoded = np.ascontiguousarray(values.astype(np.uint8))
            path = root / f"case-{index}.png"
            Image.fromarray(decoded, mode="RGB").save(path)
            adapter, audit = preprocess_image(path)
            official = (
                transform(image=decoded)["image"]
                .detach()
                .cpu()
                .numpy()
            )
            official = np.ascontiguousarray(
                official.astype(np.float32, copy=False)
            )
            exact = np.array_equal(adapter, official)
            if not exact:
                raise ValueError(
                    "SPAI adapter preprocessing differs from official "
                    f"Albumentations transform for {height}x{width}"
                )
            cases.append(
                {
                    "native_size": [width, height],
                    "adapter_shape": list(adapter.shape),
                    "adapter_sha256": _array_sha256(adapter),
                    "official_sha256": _array_sha256(official),
                    "exact_match": exact,
                    "pad_if_needed": audit["geometry"]["pad_if_needed"],
                }
            )
    return {
        "status": "passed",
        "official_transform": (
            "Albumentations_1.4.14_PadIfNeeded_224_center_"
            "BORDER_REFLECT_101_Normalize_mean0_std1_ToTensorV2"
        ),
        "cases": cases,
    }


def replay_sca_norm_head(
    *,
    model: Any,
    patch_features: Any,
    official_output: Any | None = None,
    expected_attention: Any | None = None,
    expected_aggregated: Any | None = None,
    expected_feature: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the complete official SCA, norm, MLP, and sigmoid chain."""

    import torch

    if (
        not isinstance(patch_features, torch.Tensor)
        or patch_features.ndim != 2
        or patch_features.shape[0] <= 0
        or patch_features.shape[1] != FEATURE_DIMENSION
        or patch_features.dtype != torch.float32
        or not bool(torch.isfinite(patch_features).all())
    ):
        raise ValueError("SPAI patch features violate [P,1096] float32")
    x = patch_features.unsqueeze(0)
    with torch.inference_mode():
        key, value = model.to_kv(x).chunk(2, dim=-1)
        key = key.reshape(
            1,
            x.shape[1],
            ATTENTION_HEADS,
            ATTENTION_EMBED_DIMENSION // ATTENTION_HEADS,
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            1,
            x.shape[1],
            ATTENTION_HEADS,
            ATTENTION_EMBED_DIMENSION // ATTENTION_HEADS,
        ).permute(0, 2, 1, 3)
        aggregator = model.patch_aggregator.expand(1, -1, -1, -1)
        dots = torch.matmul(aggregator, key.transpose(-1, -2)) * model.scale
        attention = model.attend(dots)
        dropped_attention = model.dropout(attention)
        attended = torch.matmul(dropped_attention, value)
        attended = (
            attended.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(1, 1, ATTENTION_EMBED_DIMENSION)
        )
        aggregated = model.to_out(attended).squeeze(dim=1)
        feature = model.norm(aggregated)
        output = model.cls_head(feature)
        probability_tensor = torch.sigmoid(output)

    expected_shapes = {
        "attention": (1, ATTENTION_HEADS, 1, patch_features.shape[0]),
        "aggregated": (1, FEATURE_DIMENSION),
        "feature": (1, FEATURE_DIMENSION),
        "output": (1, 1),
        "probability": (1, 1),
    }
    values = {
        "attention": attention,
        "aggregated": aggregated,
        "feature": feature,
        "output": output,
        "probability": probability_tensor,
    }
    for name, value_tensor in values.items():
        if (
            tuple(value_tensor.shape) != expected_shapes[name]
            or value_tensor.dtype != torch.float32
            or not bool(torch.isfinite(value_tensor).all())
        ):
            raise ValueError(f"SPAI replay produced invalid {name}")

    comparisons = {
        "official_attention_exact_match": None,
        "official_aggregated_exact_match": None,
        "official_feature_exact_match": None,
        "official_logit_exact_match": None,
    }
    for name, expected_value, replay_value, comparison_key in (
        (
            "attention",
            expected_attention,
            attention,
            "official_attention_exact_match",
        ),
        (
            "aggregated",
            expected_aggregated,
            aggregated,
            "official_aggregated_exact_match",
        ),
        (
            "feature",
            expected_feature,
            feature,
            "official_feature_exact_match",
        ),
        (
            "logit",
            official_output,
            output,
            "official_logit_exact_match",
        ),
    ):
        if expected_value is not None:
            if not isinstance(expected_value, torch.Tensor):
                raise ValueError(f"expected SPAI {name} is not a tensor")
            exact = torch.equal(replay_value, expected_value)
            comparisons[comparison_key] = exact
            if not exact:
                raise ValueError(
                    f"manual SPAI SCA/norm/head replay differs at {name}"
                )
    probability = float(probability_tensor.reshape(()).item())
    raw_logit = float(output.reshape(()).item())
    if not math.isfinite(raw_logit) or not 0.0 <= probability <= 1.0:
        raise ValueError("SPAI produced an invalid score")
    decision = probability > CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = T1_POLICY
    scoring = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            **comparisons,
            "official_probability_exact_match": (
                comparisons["official_logit_exact_match"]
            ),
            "sca_replay": True,
            "norm_replay": True,
            "complete_mlp_replay": True,
            "model_forward_calls": 1,
            "to_kv_hook_calls": 1,
            "attention_hook_calls": 1,
            "norm_hook_calls": 1,
        },
    }
    tensors = {
        "attention": attention,
        "aggregated": aggregated,
        "feature": feature,
        "output": output,
        "probability": probability_tensor,
    }
    return scoring, tensors


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int | None,
    float,
]:
    """Execute one official arbitrary-resolution forward and capture audit state."""

    import torch

    if (
        image.ndim != 3
        or image.shape[0] != 3
        or image.shape[1] < PATCH_SIZE
        or image.shape[2] < PATCH_SIZE
        or image.dtype != np.float32
        or not image.flags.c_contiguous
        or not np.isfinite(image).all()
    ):
        raise ValueError("SPAI input array contract changed")
    if torch.is_autocast_enabled():
        raise RuntimeError("SPAI adapter forbids autocast")
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    captured_patch: list[Any] = []
    captured_attention: list[Any] = []
    captured_aggregated: list[Any] = []
    captured_feature: list[Any] = []

    def capture_to_kv(
        _module: Any,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if len(inputs) != 1:
            raise RuntimeError("SPAI to_kv received unexpected inputs")
        captured_patch.append(inputs[0].detach().clone())

    def capture_attention(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        captured_attention.append(output.detach().clone())

    def capture_norm(
        _module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if len(inputs) != 1:
            raise RuntimeError("SPAI norm received unexpected inputs")
        captured_aggregated.append(inputs[0].detach().clone())
        captured_feature.append(output.detach().clone())

    hooks = (
        model.to_kv.register_forward_hook(capture_to_kv),
        model.attend.register_forward_hook(capture_attention),
        model.norm.register_forward_hook(capture_norm),
    )
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            official_output = model(
                [tensor],
                feature_extraction_batch_size=FEATURE_EXTRACTION_BATCH,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    finally:
        for hook in hooks:
            hook.remove()
    if not all(
        len(values) == 1
        for values in (
            captured_patch,
            captured_attention,
            captured_aggregated,
            captured_feature,
        )
    ):
        raise RuntimeError("SPAI audit hooks did not each fire exactly once")
    if (
        not isinstance(official_output, torch.Tensor)
        or tuple(official_output.shape) != (1, 1)
        or official_output.dtype != torch.float32
        or not bool(torch.isfinite(official_output).all())
    ):
        raise ValueError("official SPAI output violates [1,1] float32")

    patch_tensor = captured_patch[0].squeeze(0)
    attention_tensor = captured_attention[0]
    aggregated_tensor = captured_aggregated[0]
    feature_tensor = captured_feature[0]
    scoring, _ = replay_sca_norm_head(
        model=model,
        patch_features=patch_tensor,
        official_output=official_output,
        expected_attention=attention_tensor,
        expected_aggregated=aggregated_tensor,
        expected_feature=feature_tensor,
    )
    patch = np.ascontiguousarray(
        patch_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    feature = np.ascontiguousarray(
        feature_tensor.reshape(FEATURE_DIMENSION)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    attention = np.ascontiguousarray(
        attention_tensor.reshape(ATTENTION_HEADS, patch.shape[0])
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if (
        patch.ndim != 2
        or patch.shape[1] != FEATURE_DIMENSION
        or patch.dtype != np.float32
        or feature.shape != (FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or attention.shape != (ATTENTION_HEADS, patch.shape[0])
        or attention.dtype != np.float32
        or not np.isfinite(patch).all()
        or not np.isfinite(feature).all()
        or not np.isfinite(attention).all()
    ):
        raise ValueError("captured SPAI artifact contract changed")
    return scoring, patch, feature, attention, peak_bytes, latency_ms


def validate_official_golden(
    *,
    model: Any,
    device: Any,
    golden_root: Path,
) -> dict[str, Any]:
    """Gate Mouse execution on pinned official evaluation-bundle originals."""

    cases: list[dict[str, Any]] = []
    for frozen in GOLDEN_CASES:
        path = golden_root / str(frozen["relative_path"])
        _verify_file(path, str(frozen["sha256"]), "SPAI regression golden")
        image, preprocess = preprocess_image(path)
        if (
            preprocess["decoded_rgb_sha256"] != frozen["decoded_rgb_sha256"]
            or preprocess["tensor_sha256"] != frozen["tensor_sha256"]
            or preprocess["native_size"] != frozen["native_size"]
            or preprocess["geometry"]["effective_patch_count"]
            != frozen["patch_count"]
        ):
            raise ValueError("SPAI golden preprocessing regression")
        observed: list[dict[str, float]] = []
        artifact_hashes: list[dict[str, str]] = []
        for _ in range(2):
            scoring, patch, feature, attention, _, _ = infer_one(
                model,
                device,
                image,
            )
            observed.append(
                {
                    "raw_logit": float(scoring["raw_logit"]),
                    "probability": float(scoring["probability"]),
                }
            )
            artifact_hashes.append(
                {
                    "patch_features": _array_sha256(patch),
                    "feature": _array_sha256(feature),
                    "attention": _array_sha256(attention),
                }
            )
        deterministic = (
            observed[0] == observed[1]
            and artifact_hashes[0] == artifact_hashes[1]
        )
        logit_difference = abs(
            observed[0]["raw_logit"] - float(frozen["raw_logit"])
        )
        probability_difference = abs(
            observed[0]["probability"] - float(frozen["probability"])
        )
        passed = (
            deterministic
            and logit_difference <= GOLDEN_LOGIT_ABS_TOLERANCE
            and probability_difference <= GOLDEN_PROBABILITY_ABS_TOLERANCE
        )
        case = {
            **frozen,
            "path": str(path.resolve()),
            "preprocess": preprocess,
            "observed_runs": observed,
            "artifact_hashes": artifact_hashes,
            "bit_identical_repeats": deterministic,
            "logit_absolute_difference": logit_difference,
            "probability_absolute_difference": probability_difference,
            "passed": passed,
            "website_derivative_display_matches_released_regression": False,
        }
        cases.append(case)
        if not passed:
            raise ValueError(
                "SPAI official-original regression golden mismatch: "
                f"{frozen['relative_path']}"
            )
    return {
        "status": "passed",
        "source": (
            "official evaluation-bundle originals, current released source/"
            "checkpoint, pinned highest/no-TF32 float32 runtime"
        ),
        "runs_per_case": 2,
        "logit_absolute_tolerance": GOLDEN_LOGIT_ABS_TOLERANCE,
        "probability_absolute_tolerance": (
            GOLDEN_PROBABILITY_ABS_TOLERANCE
        ),
        "official_vs_adapter_full_forward": True,
        "website_display_reference": (
            "official website uses compressed derivatives and displays "
            "0.748/0.87; those values do not match the released executable "
            "regression and are disclosed rather than used as a gate"
        ),
        "cases": cases,
    }


def _result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    visibility: Mapping[str, Any],
    config_fingerprint: str,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    canonical_path = _anchored(
        Path(str(row["canonical_path"])),
        repo_root,
    )
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": str(row["task_id"]),
        "pair_rank": int(row["pair_rank"]),
        "rank": int(row["rank"]),
        "kind": str(row["kind"]),
        "label": int(row["label"]),
        "domain": str(row["domain"]),
        "input_path": repo_relative(canonical_path, repo_root),
        "input_sha256": str(row["canonical_sha256"]),
        "input_width": int(row["width"]),
        "input_height": int(row["height"]),
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "score_semantics": SCORE_SEMANTICS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "config_fingerprint": config_fingerprint,
        "edit_visibility": str(visibility["edit_visibility"]),
        "edit_visible_gt_fraction": float(
            visibility["edit_visible_gt_fraction"]
        ),
        "edit_visibility_evidence": visibility[
            "edit_visibility_evidence"
        ],
        "valid_for_t2": False,
        "attention_is_diagnostic_not_T2": True,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _load_artifact(
    *,
    row: Mapping[str, Any],
    prefix: str,
    expected_shape: tuple[int, ...],
    expected_semantics: str,
    expected_path: Path,
    repo_root: Path,
    run_dir: Path,
) -> np.ndarray:
    value = row.get(f"{prefix}_path")
    file_digest = row.get(f"{prefix}_sha256")
    array_digest = row.get(f"{prefix}_array_sha256")
    if (
        not isinstance(value, str)
        or not _valid_sha256(file_digest)
        or not _valid_sha256(array_digest)
        or row.get(f"{prefix}_shape") != list(expected_shape)
        or row.get(f"{prefix}_dtype") != "float32"
        or row.get(f"{prefix}_semantics") != expected_semantics
    ):
        raise ValueError(f"resume {prefix} metadata is invalid")
    path = _anchored(Path(value), repo_root)
    if not path.is_file():
        path = (run_dir / value).resolve()
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"resume {prefix} path changed")
    _verify_file(path, str(file_digest), f"resume {prefix}")
    array = np.load(path, allow_pickle=False)
    if (
        array.shape != expected_shape
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or _array_sha256(array) != array_digest
    ):
        raise ValueError(f"resume {prefix} artifact is invalid")
    return array


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
    config_fingerprint: str,
    model: Any,
    device: Any,
) -> None:
    import torch

    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume identity field {key} changed")
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("only successful valid rows may be resume-skipped")
    if row.get("config_fingerprint") != config_fingerprint:
        raise ValueError("resume config fingerprint changed")
    raw_logit = _finite_number(row.get("raw_logit"), "resume raw_logit")
    probability = _finite_number(
        row.get("probability"),
        "resume probability",
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("resume probability is outside [0,1]")
    decision = probability > CLASSIFICATION_THRESHOLD
    aliases = {
        "ai_score": probability,
        "score": probability,
        "score_semantics": SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "valid_for_t2": False,
        "attention_is_diagnostic_not_T2": True,
    }
    for key, value in aliases.items():
        if row.get(key) != value:
            raise ValueError(f"resume scoring field {key} changed")
    classification = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = T1_POLICY
    if row.get("classification") != classification or row.get("t1") != t1:
        raise ValueError("resume classification/T1 aliases changed")
    replay = row.get("manual_replay")
    if not isinstance(replay, Mapping) or any(
        replay.get(key) is not True
        for key in (
            "official_attention_exact_match",
            "official_aggregated_exact_match",
            "official_feature_exact_match",
            "official_logit_exact_match",
            "official_probability_exact_match",
            "sca_replay",
            "norm_replay",
            "complete_mlp_replay",
        )
    ):
        raise ValueError("resume manual replay evidence changed")
    replay_scalars = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "model_forward_calls": 1,
        "to_kv_hook_calls": 1,
        "attention_hook_calls": 1,
        "norm_hook_calls": 1,
    }
    if any(replay.get(key) != value for key, value in replay_scalars.items()):
        raise ValueError("resume manual replay scalar/call evidence changed")

    input_path = _anchored(Path(str(expected["input_path"])), repo_root)
    _verify_file(
        input_path,
        str(expected["input_sha256"]),
        f"resume SPAI input {row.get('id')}",
    )
    _, preprocess = preprocess_image(input_path)
    if row.get("preprocess") != preprocess:
        raise ValueError("resume SPAI preprocessing does not replay exactly")
    if (
        _finite_number(
            row.get("preprocess_latency_ms"),
            "resume preprocess latency",
        )
        < 0.0
        or _finite_number(row.get("latency_ms"), "resume latency") < 0.0
    ):
        raise ValueError("resume latency is negative")
    peak = row.get("peak_cuda_memory_bytes")
    if (
        peak is not None
        and (
            isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak < 0
        )
    ):
        raise ValueError("resume CUDA peak-memory value is invalid")

    patch_count = int(
        preprocess["geometry"]["effective_patch_count"]
    )
    artifact_dir = run_dir / "artifacts"
    sample_id = str(expected["id"])
    patch = _load_artifact(
        row=row,
        prefix="spai_patch_features",
        expected_shape=(patch_count, FEATURE_DIMENSION),
        expected_semantics=PATCH_FEATURE_SEMANTICS,
        expected_path=artifact_dir / f"{sample_id}.patch_features.npy",
        repo_root=repo_root,
        run_dir=run_dir,
    )
    feature = _load_artifact(
        row=row,
        prefix="spai_feature",
        expected_shape=(FEATURE_DIMENSION,),
        expected_semantics=FEATURE_SEMANTICS,
        expected_path=artifact_dir / f"{sample_id}.feature.npy",
        repo_root=repo_root,
        run_dir=run_dir,
    )
    attention = _load_artifact(
        row=row,
        prefix="spai_attention",
        expected_shape=(ATTENTION_HEADS, patch_count),
        expected_semantics=ATTENTION_SEMANTICS,
        expected_path=artifact_dir / f"{sample_id}.attention.npy",
        repo_root=repo_root,
        run_dir=run_dir,
    )
    artifact_paths = {
        "spai_patch_features_npy": row["spai_patch_features_path"],
        "spai_feature_npy": row["spai_feature_path"],
        "spai_attention_npy": row["spai_attention_path"],
    }
    if (
        row.get("artifact_paths") != artifact_paths
        or row.get("feature_array_sha256")
        != row.get("spai_feature_array_sha256")
    ):
        raise ValueError("resume SPAI artifact aliases changed")
    with torch.inference_mode():
        patch_tensor = torch.from_numpy(patch).to(device)
        expected_attention = (
            torch.from_numpy(attention).unsqueeze(0).unsqueeze(2).to(device)
        )
        expected_feature = torch.from_numpy(feature).unsqueeze(0).to(device)
        expected_output = torch.tensor(
            [[raw_logit]],
            dtype=torch.float32,
            device=device,
        )
        scoring, _ = replay_sca_norm_head(
            model=model,
            patch_features=patch_tensor,
            official_output=expected_output,
            expected_attention=expected_attention,
            expected_feature=expected_feature,
        )
    if (
        scoring["raw_logit"] != raw_logit
        or scoring["probability"] != probability
        or scoring["classification_decision"] is not decision
    ):
        raise ValueError(
            "resume SPAI persisted artifacts do not exactly replay score"
        )
    forbidden = {
        "t2",
        "localization",
        "localization_metrics",
        "attention_map",
        "attention_map_path",
        "score_map",
        "mask",
        "mask_path",
    }
    present = sorted(forbidden.intersection(row))
    if present:
        raise ValueError(f"resume SPAI row invents T2 fields: {present}")


def _run_config(
    *,
    adapter: Mapping[str, Any],
    runtime: Mapping[str, Any],
    release: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    preprocess_equivalence: Mapping[str, Any],
    golden: Mapping[str, Any],
    device_text: str,
    pair_limit: int | None,
    sample_id: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected_serialization = "".join(
        f"{stable_json(row)}\n" for row in selected
    ).encode("utf-8")
    return {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "model_arch": MODEL_ARCH,
        "adapter_contract": dict(adapter),
        "source_commit": source["commit"],
        "source_files": SOURCE_FILES,
        "checkpoint_id": CHECKPOINT["id"],
        "checkpoint_sha256": assets["checkpoint"]["actual_sha256"],
        "checkpoint_schema_sha256": assets["checkpoint"]["schema"][
            "items_sha256"
        ],
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "pad_if_needed": {
                "minimum_height": PATCH_SIZE,
                "minimum_width": PATCH_SIZE,
                "position": "center",
                "border_mode": "cv2.BORDER_REFLECT_101",
            },
            "resize": False,
            "crop": False,
            "test_augmentation": False,
            "to_tensor": (
                "Albumentations_1.4.14_uint8_normalize_LUT_float32_"
                "multiply_by_float32_1_over_255"
            ),
            "batch_size": 1,
        },
        "official_preprocess_equivalence": dict(preprocess_equivalence),
        "official_preprocess_equivalence_fingerprint": (
            _manifest_fingerprint(preprocess_equivalence)
        ),
        "model_contract": {
            "construction": "official_get_custom_config_and_build_cls_model",
            "strict_full_state_load": True,
            "checkpoint_loader": (
                "torch_weights_only_with_yacs_CfgNode_allowlist"
            ),
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "minimum_patches": MINIMUM_PATCHES,
            "feature_extraction_batch": FEATURE_EXTRACTION_BATCH,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_semantics": FEATURE_SEMANTICS,
            "patch_feature_semantics": PATCH_FEATURE_SEMANTICS,
            "attention_heads": ATTENTION_HEADS,
            "attention_semantics": ATTENTION_SEMANTICS,
            "attention_is_diagnostic_not_T2": True,
            "output": "one_raw_logit",
            "model_mode": "eval",
            "score": SCORE_SEMANTICS,
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "t1_policy": T1_POLICY,
            "score_direction": "higher_means_fake",
            "valid_for_t2": False,
        },
        "runtime_contract": {
            "device": device_text,
            "seed": MODEL_SEED,
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
            "runtime_versions": RUNTIME_VERSIONS,
            "runtime_module_file_hashes": RUNTIME_MODULE_FILES,
        },
        "runtime_evidence": dict(runtime),
        "runtime_evidence_fingerprint": _manifest_fingerprint(runtime),
        "model_construction_audit": dict(model_audit),
        "model_construction_audit_fingerprint": _manifest_fingerprint(
            model_audit
        ),
        "official_golden": dict(golden),
        "official_golden_fingerprint": _manifest_fingerprint(golden),
        "dataset": {
            "schema_version": release["schema_version"],
            "dataset_id": release["dataset_id"],
            "inputs_sha256": release["inputs_sha256"],
            "selected_ids": [str(row["sample_id"]) for row in selected],
            "selected_rows_sha256": hashlib.sha256(
                selected_serialization
            ).hexdigest(),
            "pair_limit": pair_limit,
            "sample_id": sample_id,
        },
        "metrics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "pair_bootstrap": True,
            "bootstrap_unit": "task_id_pair",
            "real_only_fpr_target": 0.05,
        },
        "license": LICENSE_RECORD,
        "checkpoint_selection_frozen_before_scores": True,
        "primary_checkpoint_reason": (
            "sole checkpoint linked by the official current README"
        ),
        "embedded_checkpoint_minimum_patches_not_restored": {
            "embedded_historical_value": 1,
            "released_inference_config_value": MINIMUM_PATCHES,
        },
        "frozen_full_dataset_visibility_census": {
            **FROZEN_VISIBILITY,
            "role": "input_condition_stratum_not_model_localization",
        },
    }


def _persist_artifacts(
    *,
    artifact_dir: Path,
    sample_id: str,
    patch: np.ndarray,
    feature: np.ndarray,
    attention: np.ndarray,
    repo_root: Path,
) -> dict[str, Any]:
    specifications = (
        (
            "spai_patch_features",
            "spai_patch_features_npy",
            artifact_dir / f"{sample_id}.patch_features.npy",
            patch,
            PATCH_FEATURE_SEMANTICS,
        ),
        (
            "spai_feature",
            "spai_feature_npy",
            artifact_dir / f"{sample_id}.feature.npy",
            feature,
            FEATURE_SEMANTICS,
        ),
        (
            "spai_attention",
            "spai_attention_npy",
            artifact_dir / f"{sample_id}.attention.npy",
            attention,
            ATTENTION_SEMANTICS,
        ),
    )
    fields: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for prefix, alias, path, array, semantics in specifications:
        if path.resolve().parent != artifact_dir.resolve():
            raise ValueError("SPAI artifact path escapes artifact directory")
        _atomic_save_npy(path, array)
        persisted = np.load(path, allow_pickle=False)
        array_digest = _array_sha256(array)
        if (
            persisted.shape != array.shape
            or persisted.dtype != np.float32
            or not persisted.flags.c_contiguous
            or not np.isfinite(persisted).all()
            or not np.array_equal(persisted, array)
            or _array_sha256(persisted) != array_digest
        ):
            raise ValueError(f"persisted SPAI artifact failed readback: {prefix}")
        relative = repo_relative(path, repo_root)
        fields.update(
            {
                f"{prefix}_path": relative,
                f"{prefix}_sha256": sha256_file(path),
                f"{prefix}_array_sha256": array_digest,
                f"{prefix}_shape": list(array.shape),
                f"{prefix}_dtype": str(array.dtype),
                f"{prefix}_semantics": semantics,
            }
        )
        paths[alias] = relative
    fields["feature_array_sha256"] = fields[
        "spai_feature_array_sha256"
    ]
    fields["artifact_paths"] = paths
    return fields


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=DEFAULT_GOLDEN_ROOT,
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source_root = _anchored(args.source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    golden_root = _anchored(args.golden_root, repo_root)
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    _safe_component(args.run_id, label="run-id")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.preflight_only and args.resume:
        raise ValueError("preflight-only and resume are mutually exclusive")
    if args.device != PRIMARY_DEVICE:
        raise ValueError(
            "SPAI executable regression golden is frozen on cuda:0; "
            "CPU remains supported only for unit/static component tests"
        )

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(all_rows, args.pair_limit, args.sample_id)
    validate_selected_inputs(selected, repo_root)
    visibility = build_pair_visibility(all_rows, repo_root)
    visibility_audit = validate_frozen_visibility_census(visibility)
    source, assets, state = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        golden_root=golden_root,
    )
    device, runtime = configure_runtime(args.device)
    preprocess_equivalence = validate_official_preprocess_equivalence()
    model, model_audit = load_model(
        state=state,
        source_root=source_root,
        device=device,
    )
    del state
    gc.collect()
    golden = validate_official_golden(
        model=model,
        device=device,
        golden_root=golden_root,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model": MODEL_NAME,
                    "source_commit": source["commit"],
                    "checkpoint_sha256": assets["checkpoint"][
                        "actual_sha256"
                    ],
                    "checkpoint_schema": {
                        key: assets["checkpoint"]["schema"][key]
                        for key in (
                            "tensor_count",
                            "state_elements",
                            "items_sha256",
                        )
                    },
                    "runtime": runtime,
                    "model_audit": model_audit,
                    "official_preprocess_equivalence": (
                        preprocess_equivalence
                    ),
                    "official_golden": golden,
                    "full_dataset_visibility_audit": visibility_audit,
                    "selected_images": len(selected),
                    "mouse_model_scores_computed": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    config = _run_config(
        adapter=adapter_contract(repo_root),
        runtime=runtime,
        release=release,
        selected=selected,
        source=source,
        assets=assets,
        model_audit=model_audit,
        preprocess_equivalence=preprocess_equivalence,
        golden=golden,
        device_text=str(device),
        pair_limit=args.pair_limit,
        sample_id=args.sample_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    config_fingerprint = _manifest_fingerprint(config)
    results_root = results_root.resolve()
    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root:
        raise ValueError("run-id escapes results directory")
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    artifact_dir = run_dir / "artifacts"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    prior_manifest: dict[str, Any] | None = None
    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires an existing manifest and expected snapshot"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            prior_manifest.get("schema_version")
            != "spai_detection_run_manifest_v1"
            or prior_manifest.get("run_id") != args.run_id
            or prior_manifest.get("config") != config
            or prior_manifest.get("config_fingerprint")
            != config_fingerprint
            or _manifest_fingerprint(
                prior_manifest.get("config", {})
            )
            != config_fingerprint
            or prior_manifest.get("runtime") != runtime
        ):
            raise ValueError("resume SPAI manifest identity/config changed")
        if read_jsonl(expected_path) != selected:
            raise ValueError("resume expected-input snapshot changed")
        prior_dataset = prior_manifest.get("dataset")
        if (
            not isinstance(prior_dataset, Mapping)
            or prior_dataset.get("expected_inputs_sha256")
            != sha256_file(expected_path)
        ):
            raise ValueError("resume expected-input snapshot hash changed")
        prior_outputs = prior_manifest.get("outputs")
        if isinstance(prior_outputs, Mapping):
            for key, path in (
                ("results_sha256", results_path),
                ("summary_sha256", summary_path),
            ):
                digest = prior_outputs.get(key)
                if digest is not None:
                    if not _valid_sha256(digest):
                        raise ValueError(f"resume manifest {key} is invalid")
                    _verify_file(path, str(digest), f"resume {path.name}")
    else:
        atomic_write_jsonl(expected_path, selected)

    selected_tasks = sorted({str(row["task_id"]) for row in selected})
    visibility_census = dict(
        Counter(
            visibility[task_id]["edit_visibility"]
            for task_id in selected_tasks
        )
    )
    patch_mode_census = dict(
        Counter(
            visibility[task_id]["patch_mode"]
            for task_id in selected_tasks
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": "spai_detection_run_manifest_v1",
        "run_id": args.run_id,
        "status": "running",
        "started_at": (
            prior_manifest.get("started_at")
            if prior_manifest is not None
            else utc_now()
        ),
        "resumed_at": utc_now() if prior_manifest is not None else None,
        "completed_at": None,
        "repo_root": str(repo_root),
        "config_fingerprint": config_fingerprint,
        "config": config,
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "official_preprocess_equivalence": preprocess_equivalence,
        "official_golden": golden,
        "runtime": runtime,
        "attention_is_diagnostic_not_T2": True,
        "valid_for_t2": False,
        "dataset": {
            "manifest_path": repo_relative(
                dataset_manifest_path,
                repo_root,
            ),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "inputs_path": repo_relative(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
            "expected_inputs_path": repo_relative(
                expected_path,
                repo_root,
            ),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
            "selected_tasks": len(selected_tasks),
        },
        "visibility_census": visibility_census,
        "patch_mode_census": patch_mode_census,
        "full_dataset_visibility_audit": visibility_audit,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_dir": repo_relative(artifact_dir, repo_root),
        },
    }
    atomic_write_json(manifest_path, manifest)

    latest = read_latest_by_id(results_path)
    completed = 0
    skipped = 0
    errors = 0
    for index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        identity = _result_identity(
            row,
            repo_root=repo_root,
            visibility=visibility[str(row["task_id"])],
            config_fingerprint=config_fingerprint,
        )
        prior = latest.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
            _validate_resume_row(
                prior,
                expected=identity,
                repo_root=repo_root,
                run_dir=run_dir,
                config_fingerprint=config_fingerprint,
                model=model,
                device=device,
            )
            skipped += 1
            print(f"[{index}/{len(selected)}] resume {sample_id}", flush=True)
            continue

        input_path = _anchored(
            Path(str(row["canonical_path"])),
            repo_root,
        )
        try:
            preprocess_started = time.perf_counter()
            image, preprocess = preprocess_image(input_path)
            preprocess_latency_ms = (
                time.perf_counter() - preprocess_started
            ) * 1000.0
            scoring, patch, feature, attention, peak_bytes, latency_ms = (
                infer_one(model, device, image)
            )
            expected_patch_count = int(
                preprocess["geometry"]["effective_patch_count"]
            )
            if patch.shape[0] != expected_patch_count:
                raise ValueError(
                    "SPAI captured patch count differs from geometry"
                )
            artifacts = _persist_artifacts(
                artifact_dir=artifact_dir,
                sample_id=sample_id,
                patch=patch,
                feature=feature,
                attention=attention,
                repo_root=repo_root,
            )
            result = {
                **identity,
                "status": "ok",
                "valid_for_metrics": True,
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_latency_ms,
                **artifacts,
                "latency_ms": latency_ms,
                "peak_cuda_memory_bytes": peak_bytes,
                **scoring,
            }
            append_jsonl(results_path, result)
            latest[sample_id] = result
            completed += 1
            print(
                f"[{index}/{len(selected)}] ok {sample_id} "
                f"score={result['ai_score']:.9f} "
                f"patches={expected_patch_count}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            error_row = {
                **identity,
                "status": "error",
                "valid_for_metrics": False,
                "completed_at": utc_now(),
                "raw_logit": None,
                "probability": None,
                "ai_score": None,
                "score": None,
                "classification_decision": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(results_path, error_row)
            latest[sample_id] = error_row
            print(
                f"[{index}/{len(selected)}] error {sample_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                __import__("torch").cuda.empty_cache()

    physical_results = read_jsonl(results_path)
    summary = summarize_spai_results(
        physical_results,
        selected,
        threshold=CLASSIFICATION_THRESHOLD,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_id": CHECKPOINT["id"],
            "preprocess_profile": PREPROCESS_PROFILE,
            "config_fingerprint": config_fingerprint,
            "official_golden_status": golden["status"],
            "official_golden_fingerprint": _manifest_fingerprint(golden),
            "attention_is_diagnostic_not_T2": True,
            "valid_for_t2": False,
            "generated_at": utc_now(),
        }
    )
    atomic_write_json(summary_path, summary)
    manifest["status"] = (
        "complete" if summary["coverage"]["is_complete"] else "incomplete"
    )
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": completed,
        "resume_skips": skipped,
        "new_errors": errors,
        "physical_result_rows": len(physical_results),
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": sum(
                1
                for path in artifact_dir.glob("*.npy")
                if path.is_file()
            ),
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": manifest["status"],
                "coverage": summary["coverage"],
                "paired_coverage": summary["paired_coverage"],
                "detection": summary["detection"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
