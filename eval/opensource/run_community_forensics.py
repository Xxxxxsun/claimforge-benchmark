#!/usr/bin/env python3
"""Run the pinned Community Forensics High-res ViT-S/16 detector.

The primary checkpoint and preprocessing profile were frozen before any Mouse
score was inspected.  The adapter constructs the complete timm architecture
with ``pretrained=False``, strictly loads a pinned safetensors state, forbids
network access, and gates execution on the five DALL-E 2 examples embedded in
the official notebook.  Community Forensics is an image classifier (T1); it
does not provide a native localization prediction (T2).
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
from eval.opensource.community_forensics_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_community_forensics_results,
)


MODEL_NAME = "Community Forensics"
MODEL_SLUG = "community_forensics_highres_vit_s16_384"
MODEL_ARCH = "vit_small_patch16_384.augreg_in21k_ft_in1k"
MODEL_REPO_URL = "https://github.com/JeongsooP/Community-Forensics"
MODEL_SOURCE_COMMIT = "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"
EVAL_SINGLE_COMMIT = "5e52ed690bdbd609f9bb1705c4c80d11872a05bd"
PAPER_URL = "https://arxiv.org/abs/2411.04125"
MODEL_HF_REPO = "OwensLab/commfor-model-384"
MODEL_HF_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
PROCESSOR_HF_REPO = "OwensLab/commfor-data-preprocessor"
PROCESSOR_HF_REVISION = "3540a3f0d688f8bf492a8aed48613b891f88047e"

PREPROCESS_PROFILE = "official_highres_resize440_centercrop384"
MODEL_SEED = 100
MODEL_INPUT_SIZE = 384
RESIZE_SHORT_SIDE = 440
FEATURE_DIMENSION = 384
FEATURE_SEMANTICS = "timm_vit_forward_head_pre_logits_classifier_input"
SCORE_SEMANTICS = "torch_float32_sigmoid_of_single_raw_logit"
T1_POLICY = "released_probability_strictly_greater_than_0_5"
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
TIMM_VERSION = "1.0.15"

LICENSE_RECORD = {
    "github_code": {
        "spdx": "MIT",
        "license_file_sha256": (
            "7ca7b8f7aaf663c7941e3bb851a4c1bbfdef51d5503be93711614f528569a5c6"
        ),
    },
    "hf_model_card": {"license": "mit"},
    "hf_processor_card": {"license": "mit"},
    "overall_commercial_clearance": (
        "MIT metadata present for code, selected model, and processor"
    ),
}

CHECKPOINT = {
    "id": "OwensLab/commfor-model-384@6076002b:model.safetensors",
    "repository": MODEL_HF_REPO,
    "revision": MODEL_HF_REVISION,
    "filename": "model.safetensors",
    "bytes": 87_262_324,
    "sha256": "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387",
    "format": "safetensors",
    "tensor_count": 152,
    "state_elements": 21_811_969,
    "trainable_parameters": 21_811_969,
    "dtype": "float32",
}

MODEL_ASSET_FILES = {
    "README.md": "47fc83b44bdc02e73e0c0497598668c8cf4a9fb03842a6f64849eb8736644485",
    "config.json": "877416e73aac0fdbc1be723f5ddf674e78a2350305865ad9031564b287b60147",
    "model.safetensors": str(CHECKPOINT["sha256"]),
}

PROCESSOR_ASSET_FILES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "README.md": "13bd1a9bc6406fc7cb6b81fee48509912ee6b20d4798034fb1c1154fedf4bcc6",
    "custom_transforms.py": "85e49fee816b20859d6f1f96fbe51d296c2d4c7b6c371326b764f57a58a25bdb",
    "dataloader.py": "eae0d9a33eaebbf26b0813f7bd7c8cf5fe136e4053502ee888f5ae37d379d4aa",
    "dataprocessor_hf.py": "4ea40fa1c24e8d620252342254263ede5a25b2717bd3d931b81bc3a72ffa558b",
    "preprocessor_config.json": "f498f9ef77d9f4e80dc6aab5ec4a9e5147420f81b3b650cedfc78a063fa96f9b",
}

SOURCE_FILES = {
    "LICENSE": "7ca7b8f7aaf663c7941e3bb851a4c1bbfdef51d5503be93711614f528569a5c6",
    "README.md": "c4673b4eeb21b52fb38b116439073abca9724423f7c48b16aa2956f160f491f5",
    "models.py": "988f2f566aa4c177dd845bfccdab9eac2b3fa5174649332b09ee8cc840c8bb97",
    "dataloader.py": "e33032c6bf5e18c5ae3ae0a0391a8571a7bb2cfa4b140c0eb80f5ce86f14562b",
    "custom_transforms.py": "85e49fee816b20859d6f1f96fbe51d296c2d4c7b6c371326b764f57a58a25bdb",
    "dataprocessor_hf.py": "3cec0839d2683694651187439f6e137928084aa0b49a80a692085fad9f5d9d92",
    "eval.py": "ce7c2671c0a66bdf810d633021b1c95913f54adff1ce8b6958a4bfaa4c5c8815",
    "eval_using_huggingface.ipynb": "dde770009f581fc51e2e234f7559d38b4dccaf2f604ba58c86a67263572e8d3c",
}

EVAL_SINGLE_FILES = {
    "main.py": "1aabf779060c343f60d38d7f2600c42a489a39cc93702bd754e1fa5b26ea4758",
    "models.py": "8d96f826344802ca55e2e1382550778136c3865df617b68951f3a9e80ff304d0",
    "LICENSE": "7ca7b8f7aaf663c7941e3bb851a4c1bbfdef51d5503be93711614f528569a5c6",
    "README.md": "4f0fc3272e8acc645ed1bc96935c480a3ee799a326dce707046cfd193736cd9f",
    "requirements.txt": "54c71f1353465013e6df0c04563052a0d4ad13c02bb46174d629708f843ca04b",
}

GOLDEN_CASES = (
    {
        "filename": "00000274.png",
        "sha256": "43a7b37d75eac6a04ab0b7b75655b7f12ea97f5de5055f6025b078a6cf36ba09",
        "probability": 0.9988338351,
    },
    {
        "filename": "00000420.png",
        "sha256": "069af2f62865b68c3c0d5a7feb98667306b8b13836d136a7a1c304a2eb172f8b",
        "probability": 0.9878403544,
    },
    {
        "filename": "00000845.png",
        "sha256": "e3c1d98ae99c54e58546c92db007071913c698803435351a4bcf80efc223247e",
        "probability": 0.9568564892,
    },
    {
        "filename": "00000916.png",
        "sha256": "a4c434f68a4b8bf2ed6bb6f15b25c72ff8e159fd6c669ec0c35c29a96d0b05be",
        "probability": 0.9516021609,
    },
    {
        "filename": "00000989.png",
        "sha256": "cfe5ba7592ee9f0b3e04f80fc1f9df5081418a93e12bb1dc32d6d64eb361393a",
        "probability": 0.7860031724,
    },
)
# The official notebook publishes four decimal places.  The independently
# reproduced full-precision references are much tighter than that, but CUDA
# attention kernels can differ from the CPU reference by several float32
# ULPs.  Keep a strict implementation-drift gate without rejecting that
# execution-level variation.
GOLDEN_ABS_TOLERANCE = 1e-5

CANONICAL_RELEASE = {
    "schema_version": "claimforge_mouse_canonical_v1",
    "dataset_id": "claimforge-mouse-good275-canonical-jpeg-q95-v1",
    "pairs": 275,
    "images": 550,
    "inputs_sha256": "e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a",
    "pairs_sha256": "bb6328be7cc7d4ae74b1e5b0b132f7fb6133c6fe73f294ebb46aebeda4f8f4b8",
    "contract_sha256": "c419e24d6f9d69822ca575e00e30f2c769ba7a28a2fcea1f6634466caf540757",
}

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
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/community_forensics")
DEFAULT_RUN_ID = (
    "community_forensics_highres_vit_s16_384_"
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


def _verify_runtime_file(path: Path, expected: str, label: str) -> None:
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
        "eval/opensource/run_community_forensics.py",
        "eval/opensource/community_forensics_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
    )
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing Community Forensics adapter component: {path}"
            )
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
    value = release.get("inputs_path")
    if not isinstance(value, str):
        raise ValueError("canonical release has no inputs_path")
    inputs_path = _anchored(Path(value), repo_root)
    _verify_runtime_file(
        inputs_path,
        str(release.get("inputs_sha256")),
        "canonical inputs.jsonl",
    )
    rows = read_jsonl(inputs_path)
    if len(rows) != int(release.get("images", -1)):
        raise ValueError("canonical input count does not match release manifest")
    ranks = [int(row["rank"]) for row in rows]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("canonical inputs are not in unique rank order")
    sample_ids = [
        _safe_component(row.get("sample_id"), label="canonical sample_id")
        for row in rows
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("canonical inputs contain duplicate sample IDs")
    pairs_value = release.get("pairs_path")
    if not isinstance(pairs_value, str):
        raise ValueError("canonical release has no pairs_path")
    pairs_path = _anchored(Path(pairs_value), repo_root)
    _verify_runtime_file(
        pairs_path,
        str(release.get("pairs_sha256")),
        "canonical pairs.jsonl",
    )
    if len(read_jsonl(pairs_path)) != int(release["pairs"]):
        raise ValueError("canonical pair count does not match release manifest")
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
    selected_ranks = set(pair_ranks)
    selected = [
        row for row in rows if int(row["pair_rank"]) in selected_ranks
    ]
    kinds: dict[int, list[str]] = {}
    task_ids: dict[int, set[str]] = {}
    for row in selected:
        pair_rank = int(row["pair_rank"])
        kinds.setdefault(pair_rank, []).append(str(row["kind"]))
        task_ids.setdefault(pair_rank, set()).add(str(row["task_id"]))
    invalid = {
        rank: values
        for rank, values in kinds.items()
        if sorted(values) != ["forged", "real"]
    }
    if invalid:
        raise ValueError(
            f"canonical selection contains incomplete pairs: {invalid}"
        )
    invalid_tasks = {
        rank: values
        for rank, values in task_ids.items()
        if len(values) != 1
    }
    if invalid_tasks:
        raise ValueError(
            "canonical pair ranks contain mismatched task IDs: "
            f"{invalid_tasks}"
        )
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
    _verify_runtime_file(path, str(digest), f"GT mask {sample_id}")
    with Image.open(path) as opened:
        pixels = np.asarray(opened)
    if pixels.ndim != 2 or pixels.shape != (height, width):
        raise ValueError(f"GT shape mismatch for {sample_id}: {pixels.shape}")
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"GT mask {sample_id} is not binary 0/255")
    positive = int(np.count_nonzero(pixels == 255))
    if positive <= 0 or positive != int(row.get("gt_positive_pixels", -1)):
        raise ValueError(f"GT positive-pixel count mismatch for {sample_id}")
    return np.asarray(pixels, dtype=np.uint8)


def validate_selected_inputs(
    selected: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    for row in selected:
        sample_id = str(row["sample_id"])
        path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            path,
            str(row["canonical_sha256"]),
            f"canonical input {sample_id}",
        )
        with Image.open(path) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"canonical dimensions changed for {sample_id}")
        _load_gt_mask(row, repo_root)


def compute_preprocess_geometry(width: int, height: int) -> dict[str, Any]:
    """Mirror torchvision Resize(int) and CenterCrop pixel geometry."""

    if width <= 0 or height <= 0:
        raise ValueError("native image dimensions must be positive")
    short, long = (width, height) if width <= height else (height, width)
    new_short = RESIZE_SHORT_SIDE
    new_long = int(new_short * long / short)
    resized_width, resized_height = (
        (new_short, new_long)
        if width <= height
        else (new_long, new_short)
    )
    crop_left = int(round((resized_width - MODEL_INPUT_SIZE) / 2.0))
    crop_top = int(round((resized_height - MODEL_INPUT_SIZE) / 2.0))
    crop_right = crop_left + MODEL_INPUT_SIZE
    crop_bottom = crop_top + MODEL_INPUT_SIZE
    native_crop = [
        crop_left * width / resized_width,
        crop_top * height / resized_height,
        crop_right * width / resized_width,
        crop_bottom * height / resized_height,
    ]
    return {
        "profile_id": PREPROCESS_PROFILE,
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
            "size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "end_xy": [crop_right, crop_bottom],
            "rounding": "int(round((dimension-crop)/2.0))",
        },
        "effective_native_crop_xyxy": native_crop,
        "pixel_center_mapping": (
            "d=(native_index+0.5)*resized_size/native_size-0.5"
        ),
        "normalize": {
            "to_tensor_scale": "uint8_div_255_to_float32",
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }


def _intersection_xyxy(
    first: list[float],
    second: list[float],
) -> list[float] | None:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return (
        [left, top, right, bottom]
        if right > left and bottom > top
        else None
    )


def _edit_box_visibility(
    edit_region: list[int],
    native_crop: list[float],
) -> dict[str, Any]:
    box = [float(value) for value in edit_region]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("edit_region_xyxy has non-positive area")
    intersection = _intersection_xyxy(box, native_crop)
    area = (box[2] - box[0]) * (box[3] - box[1])
    visible = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    fraction = min(1.0, max(0.0, visible / area))
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": native_crop,
        "intersection_xyxy": intersection,
        "edit_area": area,
        "visible_area": visible,
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


def _gt_visibility(
    forged_row: Mapping[str, Any],
    gt: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    positive_y, positive_x = np.nonzero(gt == 255)
    total = int(positive_x.size)
    if total <= 0:
        raise ValueError("forged exact-diff GT has no positive pixels")
    width, height = [int(value) for value in geometry["native_size"]]
    resized_width, resized_height = [
        int(value) for value in geometry["resize"]["destination_size"]
    ]
    crop_left, crop_top = [
        float(value) for value in geometry["center_crop"]["start_xy"]
    ]
    crop_right, crop_bottom = [
        float(value) for value in geometry["center_crop"]["end_xy"]
    ]
    destination_x = (
        (positive_x.astype(np.float64) + 0.5)
        * resized_width
        / width
        - 0.5
    )
    destination_y = (
        (positive_y.astype(np.float64) + 0.5)
        * resized_height
        / height
        - 0.5
    )
    visible_mask = (
        (destination_x >= crop_left)
        & (destination_x < crop_right)
        & (destination_y >= crop_top)
        & (destination_y < crop_bottom)
    )
    visible = int(np.count_nonzero(visible_mask))
    return {
        "category": (
            "none" if visible == 0 else "full" if visible == total else "partial"
        ),
        "visible_fraction": visible / total,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": str(forged_row["sample_id"]),
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_through_"
            "official_resize440_and_center_crop384"
        ),
        "profile_id": PREPROCESS_PROFILE,
        "formula": str(geometry["pixel_center_mapping"]),
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
        geometry = compute_preprocess_geometry(
            int(forged["width"]),
            int(forged["height"]),
        )
        evidence = _gt_visibility(forged, gt, geometry)
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
        result[task_id] = {
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": evidence,
                "edit_box": _edit_box_visibility(
                    edit_region,
                    list(geometry["effective_native_crop_xyxy"]),
                ),
            },
        }
    return result


def validate_frozen_visibility_census(
    visibility: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    categories = Counter(
        str(value["edit_visibility"]) for value in visibility.values()
    )
    expected = {"full": 162, "partial": 32, "none": 81}
    if len(visibility) != 275 or dict(categories) != expected:
        raise ValueError(
            "Community Forensics frozen visibility census changed: "
            f"pairs={len(visibility)}, census={dict(categories)}"
        )
    fractions = [
        float(value["edit_visible_gt_fraction"])
        for value in visibility.values()
    ]
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in fractions
    ):
        raise ValueError("visibility census contains invalid fractions")
    mean_fraction = float(np.mean(fractions))
    if round(mean_fraction, 6) != 0.646589:
        raise ValueError(
            "Community Forensics frozen mean visible fraction changed: "
            f"{mean_fraction}"
        )
    return {
        "pairs": len(visibility),
        "census": expected,
        "mean_edit_visible_gt_fraction": mean_fraction,
        "rounded_mean_6_decimals": round(mean_fraction, 6),
        "basis": (
            "exact_diff_positive_pixel_centers_after_official_resize_crop"
        ),
    }


def _verify_source_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"missing Community Forensics source-root: {source_root}"
        )
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            "Community Forensics source commit mismatch: "
            f"{commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(
            "Community Forensics tracked source tree is dirty: "
            f"{dirty[:1000]}"
        )
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"Community Forensics source {relative}",
        )

    eval_single: dict[str, Any] = {}
    for relative, digest in EVAL_SINGLE_FILES.items():
        blob = _git_bytes(
            source_root,
            f"{EVAL_SINGLE_COMMIT}:{relative}",
        )
        actual = hashlib.sha256(blob).hexdigest()
        if actual != digest:
            raise ValueError(
                f"eval_single {relative} blob changed: {actual} != {digest}"
            )
        eval_single[relative] = {
            "git_object": f"{EVAL_SINGLE_COMMIT}:{relative}",
            "bytes": len(blob),
            "sha256": actual,
        }

    main_models = (source_root / "models.py").read_text(encoding="utf-8")
    main_loader = (source_root / "dataloader.py").read_text(encoding="utf-8")
    historical_main = _git_bytes(
        source_root,
        f"{EVAL_SINGLE_COMMIT}:main.py",
    ).decode("utf-8")
    historical_models = _git_bytes(
        source_root,
        f"{EVAL_SINGLE_COMMIT}:models.py",
    ).decode("utf-8")
    main_evidence = (
        "vit_small_patch16_384.augreg_in21k_ft_in1k",
        "self.vit.head = nn.Linear(in_features=384, out_features=1",
    )
    loader_evidence = (
        "resize_size=440",
        "crop_size=384",
        "transforms.Resize(resize_size)",
        "transforms.CenterCrop(crop_size)",
        "transforms.Normalize(mean=norm_mean, std=norm_std)",
    )
    single_evidence = (
        "transforms.Resize(resize_size)",
        "transforms.CenterCrop(crop_size)",
        "transforms.ToTensor()",
        "norm_mean = [0.485, 0.456, 0.406]",
        "torch.nn.functional.sigmoid(x)",
    )
    missing = [
        text
        for text in main_evidence
        if text not in main_models
    ] + [
        text
        for text in loader_evidence
        if text not in main_loader
    ] + [
        text
        for text in single_evidence
        if text not in historical_models
    ]
    if "if prob > 0.5:" not in historical_main:
        missing.append("if prob > 0.5:")
    if missing:
        raise ValueError(
            f"official source semantic evidence changed: missing {missing}"
        )

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
        "eval_single": {
            "commit": EVAL_SINGLE_COMMIT,
            "branch_relationship": (
                "separate_official_eval_single_branch_not_main_ancestor"
            ),
            "files": eval_single,
            "role": (
                "corroborates single-image RGB/resize/crop/normalize/sigmoid/"
                "strict-threshold execution semantics"
            ),
        },
        "license_record": LICENSE_RECORD,
    }


def _verify_processor_contract(processor_root: Path) -> dict[str, Any]:
    if not processor_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned Community Forensics processor: {processor_root}"
        )
    commit = _git_value(processor_root, "rev-parse", "HEAD")
    if commit != PROCESSOR_HF_REVISION:
        raise ValueError(
            "Community Forensics processor revision mismatch: "
            f"{commit} != {PROCESSOR_HF_REVISION}"
        )
    dirty = _git_value(
        processor_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(
            "Community Forensics processor tracked tree is dirty: "
            f"{dirty[:1000]}"
        )
    for relative, digest in PROCESSOR_ASSET_FILES.items():
        _verify_runtime_file(
            processor_root / relative,
            digest,
            f"Community Forensics processor {relative}",
        )
    config = json.loads(
        (processor_root / "preprocessor_config.json").read_text(
            encoding="utf-8"
        )
    )
    if config != {
        "image_processor_type": "CommForImageProcessor",
        "size": 384,
        "auto_map": {
            "AutoImageProcessor": (
                "dataprocessor_hf.CommForImageProcessor"
            )
        },
    }:
        raise ValueError("Community Forensics processor config changed")
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
    missing = [text for text in required if text not in loader]
    if missing:
        raise ValueError(
            f"pinned processor transform changed: missing {missing}"
        )
    return {
        "repository": PROCESSOR_HF_REPO,
        "root": str(processor_root.resolve()),
        "revision": commit,
        "tracked_dirty": False,
        "files": {
            relative: {
                "path": str((processor_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in PROCESSOR_ASSET_FILES.items()
        },
        "config": config,
    }


def _state_schema(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
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
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"Community Forensics state {key} is not float32"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"Community Forensics state {key} is non-finite"
                )
            items.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                    "sha256": _tensor_sha256(tensor),
                }
            )
    if metadata not in (None, {}):
        raise ValueError("Community Forensics safetensors metadata changed")
    if (
        len(items) != int(CHECKPOINT["tensor_count"])
        or sum(item["numel"] for item in items)
        != int(CHECKPOINT["state_elements"])
    ):
        raise ValueError("Community Forensics safetensors schema changed")
    state = load_file(checkpoint_path, device="cpu")
    if list(state) != keys:
        raise ValueError("safetensors safe_open/load_file key order changed")
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
    return state, schema


def verify_assets(
    *,
    source_root: Path,
    model_root: Path,
    processor_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    source_audit = _verify_source_contract(source_root)
    processor_audit = _verify_processor_contract(processor_root)
    if not model_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned Community Forensics model root: {model_root}"
        )
    for relative, digest in MODEL_ASSET_FILES.items():
        _verify_runtime_file(
            model_root / relative,
            digest,
            f"Community Forensics model asset {relative}",
        )
    checkpoint_path = model_root / str(CHECKPOINT["filename"])
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("Community Forensics checkpoint byte size changed")
    config = json.loads(
        (model_root / "config.json").read_text(encoding="utf-8")
    )
    if config != {
        "device": "cuda",
        "freeze_backbone": False,
        "input_size": 384,
        "model_size": "small",
        "patch_size": 16,
    }:
        raise ValueError("Community Forensics model config changed")
    model_card = (model_root / "README.md").read_text(encoding="utf-8")
    if "license: mit" not in model_card:
        raise ValueError("Community Forensics model-card MIT metadata changed")
    state, schema = _state_schema(checkpoint_path)
    asset_audit = {
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path.resolve()),
            "actual_bytes": checkpoint_path.stat().st_size,
            "actual_sha256": sha256_file(checkpoint_path),
            "serialization_safety": {
                "format": "safetensors",
                "pickle_executed": False,
                "loader": "safetensors.torch.load_file",
            },
            "schema": schema,
        },
        "model_repository": {
            "repository": MODEL_HF_REPO,
            "revision": MODEL_HF_REVISION,
            "root": str(model_root.resolve()),
            "files": {
                relative: {
                    "path": str((model_root / relative).resolve()),
                    "sha256": digest,
                }
                for relative, digest in MODEL_ASSET_FILES.items()
            },
            "config": config,
        },
        "processor": processor_audit,
        "bundle_sha256": hashlib.sha256(
            stable_json(
                {
                    "model_revision": MODEL_HF_REVISION,
                    "model_sha256": CHECKPOINT["sha256"],
                    "processor_revision": PROCESSOR_HF_REVISION,
                    "processor_files": PROCESSOR_ASSET_FILES,
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
    return source_audit, asset_audit, state


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Community Forensics supports only cpu or cuda")
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.backends.cudnn.enabled = False
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
        "cudnn_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }
    if runtime_truth != expected_truth:
        raise RuntimeError(
            "Community Forensics deterministic runtime did not apply: "
            f"{runtime_truth}"
        )
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "safetensors": _package_version("safetensors"),
        "scikit_learn": _package_version("scikit-learn"),
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
        **runtime_truth,
        "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_allow_tf32_cublas_override_env": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
        "timm_fused_attention_policy": (
            "official_timm_1.0.15_default_recorded_at_model_load"
        ),
    }
    if runtime["timm"] != TIMM_VERSION:
        raise ValueError(
            f"timm version mismatch: {runtime['timm']} != {TIMM_VERSION}"
        )
    return device, runtime


def load_model(
    *,
    state: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import timm

    if timm.__version__ != TIMM_VERSION:
        raise ValueError(
            f"timm version mismatch: {timm.__version__} != {TIMM_VERSION}"
        )

    class CommunityForensicsModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vit = timm.create_model(
                MODEL_ARCH,
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

    network_attempts: dict[str, int] = {
        "urllib_urlopen": 0,
        "socket_create_connection": 0,
        "socket_connect": 0,
    }

    def reject(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            network_attempts[name] += 1
            raise RuntimeError(
                "network access is forbidden during model construction"
            )

        return blocked

    offline_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with ExitStack() as stack:
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
            f"model construction attempted network access: {network_attempts}"
        )

    model_keys = list(model.state_dict())
    state_keys = list(state)
    if set(model_keys) != set(state_keys):
        missing = sorted(set(model_keys) - set(state_keys))
        unexpected = sorted(set(state_keys) - set(model_keys))
        raise ValueError(
            "Community Forensics full-state schema mismatch before load: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            "strict Community Forensics load reported incompatibilities"
        )
    for key, loaded in model.state_dict().items():
        if not torch.equal(loaded.detach().cpu(), state[key]):
            raise ValueError(
                f"loaded Community Forensics tensor differs: {key}"
            )
    parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    if parameter_count != int(CHECKPOINT["trainable_parameters"]):
        raise ValueError("Community Forensics parameter count changed")
    if not isinstance(model.vit.head, torch.nn.Linear) or (
        model.vit.head.in_features,
        model.vit.head.out_features,
    ) != (FEATURE_DIMENSION, 1):
        raise ValueError("Community Forensics classifier shape changed")
    fused_flags = [
        bool(getattr(block.attn, "fused_attn", True))
        for block in model.vit.blocks
    ]
    if not fused_flags or len(set(fused_flags)) != 1:
        raise ValueError("timm attention implementation flags are inconsistent")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "construction": {
            "framework": "timm",
            "version": TIMM_VERSION,
            "architecture": MODEL_ARCH,
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
            "loaded_tensor_count": len(state_keys),
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


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        resized = transforms.Resize(
            RESIZE_SHORT_SIDE,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )(rgb)
        resized_rgb = np.ascontiguousarray(
            np.asarray(resized, dtype=np.uint8)
        )
        cropped = transforms.CenterCrop(MODEL_INPUT_SIZE)(resized)
        crop_rgb = np.ascontiguousarray(
            np.asarray(cropped, dtype=np.uint8)
        )
    tensor = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)(
        transforms.ToTensor()(cropped)
    )
    array = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    geometry = compute_preprocess_geometry(width, height)
    expected_resized = tuple(geometry["resize"]["destination_size"])
    if (
        decoded.shape != (height, width, 3)
        or resized.size != expected_resized
        or resized_rgb.shape
        != (expected_resized[1], expected_resized[0], 3)
        or crop_rgb.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
        or array.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise ValueError("Community Forensics preprocessing contract changed")
    return array, {
        "profile": PREPROCESS_PROFILE,
        "geometry": geometry,
        "decoded_rgb_sha256": _array_sha256(decoded),
        "resized_rgb_sha256": _array_sha256(resized_rgb),
        "resized_rgb_shape": list(resized_rgb.shape),
        "resized_rgb_dtype": str(resized_rgb.dtype),
        "crop_rgb_sha256": _array_sha256(crop_rgb),
        "crop_rgb_shape": list(crop_rgb.shape),
        "crop_rgb_dtype": str(crop_rgb.dtype),
        "tensor_shape": list(array.shape),
        "tensor_dtype": str(array.dtype),
        "tensor_sha256": _array_sha256(array),
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }


def replay_classifier(
    official_output: Any,
    feature: Any,
    classifier: Any,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as functional

    if (
        not isinstance(feature, torch.Tensor)
        or tuple(feature.shape) != (1, FEATURE_DIMENSION)
        or feature.dtype != torch.float32
        or not bool(torch.isfinite(feature).all())
    ):
        raise ValueError(
            "captured Community Forensics feature violates [1,384] float32"
        )
    if (
        not isinstance(official_output, torch.Tensor)
        or tuple(official_output.shape) != (1, 1)
        or official_output.dtype != torch.float32
        or not bool(torch.isfinite(official_output).all())
    ):
        raise ValueError(
            "official Community Forensics output violates [1,1] float32"
        )
    with torch.inference_mode():
        manual_output = functional.linear(
            feature,
            classifier.weight,
            classifier.bias,
        )
        probability_tensor = torch.sigmoid(official_output)
        manual_probability = torch.sigmoid(manual_output)
    if not torch.equal(manual_output, official_output):
        raise ValueError(
            "manual classifier replay differs from official output"
        )
    if not torch.equal(manual_probability, probability_tensor):
        raise ValueError("manual sigmoid replay differs from official sigmoid")
    raw_logit = float(official_output.reshape(()).item())
    probability = float(probability_tensor.reshape(()).item())
    if not math.isfinite(raw_logit) or not 0.0 <= probability <= 1.0:
        raise ValueError("Community Forensics produced an invalid score")
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
    return {
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
            "raw_logit": float(manual_output.reshape(()).item()),
            "probability": float(manual_probability.reshape(()).item()),
            "ai_score": float(manual_probability.reshape(()).item()),
            "classification_decision": bool(
                (manual_probability > CLASSIFICATION_THRESHOLD).item()
            ),
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, int | None, float]:
    import torch

    if (
        image.dtype != np.float32
        or image.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        or not image.flags.c_contiguous
    ):
        raise ValueError("Community Forensics input array contract changed")
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    captured: list[Any] = []

    def capture_classifier(
        _module: Any,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if len(inputs) != 1:
            raise RuntimeError(
                "Community Forensics head received unexpected inputs"
            )
        captured.append(inputs[0].detach().clone())

    hook = model.vit.head.register_forward_hook(capture_classifier)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            official_output = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "Community Forensics classifier hook did not fire exactly once"
        )
    scoring = replay_classifier(
        official_output,
        captured[0],
        model.vit.head,
    )
    feature = np.ascontiguousarray(
        captured[0]
        .reshape(FEATURE_DIMENSION)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if (
        feature.shape != (FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or not np.isfinite(feature).all()
    ):
        raise ValueError("Community Forensics feature artifact is invalid")
    return scoring, feature, peak_bytes, latency_ms


def validate_official_golden(
    *,
    model: Any,
    device: Any,
    source_root: Path,
) -> dict[str, Any]:
    """Gate a run on the official notebook's five DALL-E 2 examples."""

    import torch

    arrays: list[np.ndarray] = []
    cases: list[dict[str, Any]] = []
    for frozen in GOLDEN_CASES:
        path = source_root / "test_images" / str(frozen["filename"])
        _verify_runtime_file(
            path,
            str(frozen["sha256"]),
            f"official golden image {frozen['filename']}",
        )
        array, preprocess = preprocess_image(path)
        arrays.append(array)
        cases.append(
            {
                **frozen,
                "path": str(path.resolve()),
                "tensor_sha256": preprocess["tensor_sha256"],
            }
        )
    batch = torch.from_numpy(np.stack(arrays)).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        output = model(batch)
        probabilities = torch.sigmoid(output)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if (
        tuple(output.shape) != (len(cases), 1)
        or output.dtype != torch.float32
        or tuple(probabilities.shape) != (len(cases), 1)
        or probabilities.dtype != torch.float32
        or not bool(torch.isfinite(output).all())
        or not bool(torch.isfinite(probabilities).all())
    ):
        raise ValueError("Community Forensics golden output contract changed")
    actual_values = [
        float(value)
        for value in probabilities.detach().cpu().reshape(-1).tolist()
    ]
    for case, actual in zip(cases, actual_values, strict=True):
        expected = float(case["probability"])
        difference = abs(actual - expected)
        case.update(
            {
                "expected_probability": expected,
                "actual_probability": actual,
                "absolute_difference": difference,
                "passed": difference <= GOLDEN_ABS_TOLERANCE,
            }
        )
        if difference > GOLDEN_ABS_TOLERANCE:
            raise ValueError(
                "Community Forensics official golden mismatch for "
                f"{case['filename']}: {actual} != {expected}"
            )
    return {
        "status": "passed",
        "source": (
            "official eval_using_huggingface.ipynb five DALL-E 2 images; "
            "full-precision frozen values independently reproduced before "
            "Mouse scoring"
        ),
        "notebook_sha256": SOURCE_FILES[
            "eval_using_huggingface.ipynb"
        ],
        "batch_size": len(cases),
        "dtype": "float32",
        "score": "torch.sigmoid(raw_logit)",
        "absolute_tolerance": GOLDEN_ABS_TOLERANCE,
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


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
    config_fingerprint: str,
    classifier: Any,
    device: Any,
) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume identity field {key} changed")
    if row.get("status") != "ok":
        raise ValueError("only successful rows may be resume-skipped")
    if row.get("config_fingerprint") != config_fingerprint:
        raise ValueError("resume row config fingerprint changed")

    raw_logit = _finite_number(row.get("raw_logit"), "resume raw_logit")
    probability = _finite_number(
        row.get("probability"),
        "resume probability",
    )
    ai_score = _finite_number(row.get("ai_score"), "resume ai_score")
    if not 0.0 <= probability <= 1.0 or ai_score != probability:
        raise ValueError("resume probability/ai_score aliases changed")
    decision = probability > CLASSIFICATION_THRESHOLD
    scalar_aliases = {
        "score": probability,
        "score_semantics": SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
    }
    for key, value in scalar_aliases.items():
        if row.get(key) != value:
            raise ValueError(f"resume scoring alias {key} changed")
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
    if row.get("classification_decision") is not decision:
        raise ValueError("resume strict classification decision changed")
    replay = row.get("manual_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("resume manual classifier replay is missing")
    required_replay = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "classifier_hook_calls": 1,
    }
    for key, value in required_replay.items():
        if replay.get(key) != value:
            raise ValueError(f"resume manual replay field {key} changed")

    preprocess = row.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ValueError("resume preprocess audit is missing")
    geometry = compute_preprocess_geometry(
        int(expected["input_width"]),
        int(expected["input_height"]),
    )
    required_preprocess = {
        "profile": PREPROCESS_PROFILE,
        "geometry": geometry,
        "crop_rgb_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3],
        "crop_rgb_dtype": "uint8",
        "tensor_shape": [3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "tensor_dtype": "float32",
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }
    resized_width, resized_height = geometry["resize"]["destination_size"]
    required_preprocess["resized_rgb_shape"] = [
        resized_height,
        resized_width,
        3,
    ]
    required_preprocess["resized_rgb_dtype"] = "uint8"
    for key, value in required_preprocess.items():
        if preprocess.get(key) != value:
            raise ValueError(f"resume preprocess field {key} changed")
    for key in (
        "decoded_rgb_sha256",
        "resized_rgb_sha256",
        "crop_rgb_sha256",
        "tensor_sha256",
    ):
        if not _valid_sha256(preprocess.get(key)):
            raise ValueError(f"resume preprocess field {key} is invalid")
    input_path = _anchored(Path(str(expected["input_path"])), repo_root)
    _verify_runtime_file(
        input_path,
        str(expected["input_sha256"]),
        f"resume Community Forensics input {row.get('id')}",
    )
    replay_input, replay_preprocess = preprocess_image(input_path)
    del replay_input
    if dict(preprocess) != replay_preprocess:
        raise ValueError("resume preprocessing does not replay exactly")

    preprocess_latency = _finite_number(
        row.get("preprocess_latency_ms"),
        "resume preprocess_latency_ms",
    )
    inference_latency = _finite_number(
        row.get("latency_ms"),
        "resume latency_ms",
    )
    if preprocess_latency < 0.0 or inference_latency < 0.0:
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
        raise ValueError("resume peak CUDA memory is invalid")

    feature_value = row.get("commfor_feature_path")
    feature_digest = row.get("commfor_feature_sha256")
    feature_array_digest = row.get("commfor_feature_array_sha256")
    if (
        not isinstance(feature_value, str)
        or not _valid_sha256(feature_digest)
        or not _valid_sha256(feature_array_digest)
    ):
        raise ValueError("resume feature artifact metadata is invalid")
    if (
        row.get("commfor_feature_shape") != [FEATURE_DIMENSION]
        or row.get("commfor_feature_dtype") != "float32"
        or row.get("commfor_feature_semantics") != FEATURE_SEMANTICS
        or row.get("artifact_paths")
        != {"commfor_feature_npy": feature_value}
        or row.get("valid_for_metrics") is not True
        or row.get("feature_array_sha256") != feature_array_digest
    ):
        raise ValueError("resume feature contract changed")
    feature_path = _anchored(Path(feature_value), repo_root)
    if not feature_path.is_file():
        feature_path = (run_dir / feature_value).resolve()
    expected_feature_path = (
        run_dir / "features" / f"{expected['id']}.npy"
    ).resolve()
    if feature_path.resolve() != expected_feature_path:
        raise ValueError(
            "resume feature path is not the selected sample artifact"
        )
    _verify_runtime_file(
        feature_path,
        str(feature_digest),
        f"resume Community Forensics feature {row.get('id')}",
    )
    feature = np.load(feature_path, allow_pickle=False)
    if (
        feature.shape != (FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or not np.isfinite(feature).all()
        or _array_sha256(feature) != feature_array_digest
    ):
        raise ValueError("resume Community Forensics feature is invalid")
    import torch
    from torch.nn import functional as functional

    with torch.inference_mode():
        feature_tensor = torch.from_numpy(
            np.ascontiguousarray(feature)
        ).unsqueeze(0).to(device)
        replay_output = functional.linear(
            feature_tensor,
            classifier.weight,
            classifier.bias,
        )
        replay_probability = torch.sigmoid(replay_output)
    replay_logit = float(replay_output.reshape(()).item())
    replay_score = float(replay_probability.reshape(()).item())
    if replay_logit != raw_logit or replay_score != probability:
        raise ValueError(
            "resume persisted feature does not exactly replay head/sigmoid"
        )
    if (replay_score > CLASSIFICATION_THRESHOLD) is not decision:
        raise ValueError("resume feature replay decision changed")

    forbidden = {
        "t2",
        "localization",
        "localization_metrics",
        "attention_map",
        "attention_map_path",
        "score_map",
        "score_map_path",
        "predicted_mask",
        "predicted_mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "dice",
        "s_joint",
    }
    present = sorted(forbidden.intersection(row))
    if present:
        raise ValueError(
            f"resume Community Forensics row invents T2 fields: {present}"
        )


def _run_config(
    *,
    adapter: Mapping[str, Any],
    runtime: Mapping[str, Any],
    release: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source_audit: Mapping[str, Any],
    asset_audit: Mapping[str, Any],
    model_audit: Mapping[str, Any],
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
        "source_commit": source_audit["commit"],
        "eval_single_commit": source_audit["eval_single"]["commit"],
        "source_files": SOURCE_FILES,
        "checkpoint_id": CHECKPOINT["id"],
        "checkpoint_revision": CHECKPOINT["revision"],
        "checkpoint_sha256": asset_audit["checkpoint"]["actual_sha256"],
        "checkpoint_schema_sha256": asset_audit["checkpoint"]["schema"][
            "items_sha256"
        ],
        "processor_revision": asset_audit["processor"]["revision"],
        "processor_files": PROCESSOR_ASSET_FILES,
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "resize": {
                "short_side": RESIZE_SHORT_SIDE,
                "aspect_preserving": True,
                "interpolation": "PIL_BILINEAR",
                "antialias": True,
            },
            "crop": {
                "kind": "torchvision_CenterCrop",
                "size": MODEL_INPUT_SIZE,
            },
            "to_tensor": "uint8_div_255_to_float32",
            "normalization_mean": list(IMAGE_MEAN),
            "normalization_std": list(IMAGE_STD),
            "batch_size": 1,
        },
        "model_contract": {
            "construction": (
                "timm_1.0.15_create_model_pretrained_false_then_replace_head"
            ),
            "strict_full_safetensors_load": True,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_semantics": FEATURE_SEMANTICS,
            "output": "one_raw_logit",
            "model_mode": "eval",
            "score": SCORE_SEMANTICS,
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "t1_policy": T1_POLICY,
            "score_direction": "higher_means_fake",
            "valid_for_t2": False,
            "attention_or_features_are_localization": False,
        },
        "runtime_contract": {
            "device": device_text,
            "seed": MODEL_SEED,
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
            "dataset_id": release.get("dataset_id"),
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
        },
        "license": LICENSE_RECORD,
        "checkpoint_selection_frozen_before_scores": True,
        "primary_checkpoint_reason": (
            "paper identifies High res. 384 model as best-performing and "
            "uses it for subsequent experiments"
        ),
        "excluded_primary_alternatives": {
            "OwensLab/commfor-model-224": (
                "lower-resolution release excluded before Mouse scoring; "
                "never selected or added based on Mouse results"
            )
        },
        "frozen_full_dataset_visibility_census": {
            "pairs": 275,
            "full": 162,
            "partial": 32,
            "none": 81,
            "mean_edit_visible_gt_fraction": 0.646589,
            "role": "input_condition_stratum_not_model_localization",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--processor-root",
        type=Path,
        default=DEFAULT_PROCESSOR_ROOT,
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
    model_root = _anchored(args.model_root, repo_root)
    processor_root = _anchored(args.processor_root, repo_root)
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    _safe_component(args.run_id, label="run-id")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    results_root = results_root.resolve()
    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root:
        raise ValueError("run-id escapes the results directory")
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    feature_dir = run_dir / "features"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(all_rows, args.pair_limit, args.sample_id)
    validate_selected_inputs(selected, repo_root)
    visibility = build_pair_visibility(all_rows, repo_root)
    visibility_audit = validate_frozen_visibility_census(visibility)
    source_audit, asset_audit, state = verify_assets(
        source_root=source_root,
        model_root=model_root,
        processor_root=processor_root,
    )
    device, runtime = configure_runtime(args.device)
    model, model_audit = load_model(state=state, device=device)
    del state
    golden = validate_official_golden(
        model=model,
        device=device,
        source_root=source_root,
    )

    config = _run_config(
        adapter=adapter_contract(repo_root),
        runtime=runtime,
        release=release,
        selected=selected,
        source_audit=source_audit,
        asset_audit=asset_audit,
        model_audit=model_audit,
        golden=golden,
        device_text=str(device),
        pair_limit=args.pair_limit,
        sample_id=args.sample_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    config_fingerprint = _manifest_fingerprint(config)

    prior_manifest: dict[str, Any] | None = None
    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires existing manifest and expected-input snapshot"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            prior_manifest.get("schema_version")
            != "community_forensics_detection_run_manifest_v1"
            or prior_manifest.get("run_id") != args.run_id
            or prior_manifest.get("config") != config
            or _manifest_fingerprint(prior_manifest.get("config", {}))
            != config_fingerprint
        ):
            raise ValueError("resume manifest identity/config changed")
        if prior_manifest.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume manifest config fingerprint mismatch")
        if prior_manifest.get("runtime") != runtime:
            raise ValueError("resume runtime evidence changed")
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
                expected_digest = prior_outputs.get(key)
                if expected_digest is not None:
                    if not _valid_sha256(expected_digest):
                        raise ValueError(
                            f"resume manifest {key} is invalid"
                        )
                    _verify_runtime_file(
                        path,
                        str(expected_digest),
                        f"resume prior output {path.name}",
                    )
    else:
        atomic_write_jsonl(expected_path, selected)

    selected_tasks = sorted({str(row["task_id"]) for row in selected})
    visibility_census = dict(
        sorted(
            Counter(
                visibility[task_id]["edit_visibility"]
                for task_id in selected_tasks
            ).items()
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": (
            "community_forensics_detection_run_manifest_v1"
        ),
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
        "source": source_audit,
        "assets": asset_audit,
        "model_audit": model_audit,
        "official_golden": golden,
        "runtime": runtime,
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
        "full_dataset_visibility_audit": visibility_audit,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "feature_dir": repo_relative(feature_dir, repo_root),
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
                classifier=model.vit.head,
                device=device,
            )
            skipped += 1
            print(
                f"[{index}/{len(selected)}] resume {sample_id}",
                flush=True,
            )
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
            scoring, feature, peak_bytes, latency_ms = infer_one(
                model,
                device,
                image,
            )
            feature_path = feature_dir / f"{sample_id}.npy"
            if feature_path.resolve().parent != feature_dir.resolve():
                raise ValueError("feature artifact path escapes feature directory")
            _atomic_save_npy(feature_path, feature)
            feature_digest = sha256_file(feature_path)
            feature_array_digest = _array_sha256(feature)
            persisted_feature = np.load(feature_path, allow_pickle=False)
            if (
                persisted_feature.shape != (FEATURE_DIMENSION,)
                or persisted_feature.dtype != np.float32
                or not persisted_feature.flags.c_contiguous
                or not np.isfinite(persisted_feature).all()
                or not np.array_equal(persisted_feature, feature)
                or _array_sha256(persisted_feature)
                != feature_array_digest
            ):
                raise ValueError(
                    "persisted Community Forensics feature failed readback"
                )
            feature_relative = repo_relative(feature_path, repo_root)
            result = {
                **identity,
                "status": "ok",
                "valid_for_metrics": True,
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_latency_ms,
                "commfor_feature_path": feature_relative,
                "commfor_feature_sha256": feature_digest,
                "commfor_feature_array_sha256": feature_array_digest,
                "feature_array_sha256": feature_array_digest,
                "commfor_feature_shape": list(feature.shape),
                "commfor_feature_dtype": str(feature.dtype),
                "commfor_feature_semantics": FEATURE_SEMANTICS,
                "artifact_paths": {
                    "commfor_feature_npy": feature_relative,
                },
                "latency_ms": latency_ms,
                "peak_cuda_memory_bytes": peak_bytes,
                **scoring,
            }
            append_jsonl(results_path, result)
            latest[sample_id] = result
            completed += 1
            print(
                f"[{index}/{len(selected)}] ok {sample_id} "
                f"score={result['ai_score']:.9f}",
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

    physical_results = (
        read_jsonl(results_path) if results_path.is_file() else []
    )
    summary = summarize_community_forensics_results(
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
            "feature_files": sum(
                1 for path in feature_dir.glob("*.npy") if path.is_file()
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
