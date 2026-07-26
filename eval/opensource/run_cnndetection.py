#!/usr/bin/env python3
"""Run the official CNNDetection Blur+JPEG(0.1) whole-image detector.

The primary protocol freezes the post-release official recommendation:
native-resolution RGB, no resize, no crop, batch size one, and ImageNet
normalization.  The paper-era native center-crop-224 profile is available only
as a separately identified sensitivity run.  Blur and JPEG are *training*
augmentations and are never applied here.

CNNDetection is an image-level T1 detector.  It has no native localization,
pixel mask, bounding box, or T2 output.
"""

from __future__ import annotations

import argparse
import gc
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
import tempfile
import time
import traceback
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.cnndetection_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_cnndetection_raw_logits,
    summarize_cnndetection_results,
)
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


MODEL_NAME = "CNNDetection"
MODEL_SLUG = "cnndetection_blur_jpg_prob0_1"
MODEL_ARCH = "official vendored ResNet-50, one-logit head"
MODEL_REPO_URL = "https://github.com/PeterWang512/CNNDetection"
MODEL_SOURCE_COMMIT = "ea0b5622365e3a9cd31d1b54b6b5971131a839ab"
PAPER_ERA_STABLE_COMMIT = "f692c138482137c92280c01a45ae190379f16790"
PAPER_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2020/html/"
    "Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_"
    "CVPR_2020_paper.html"
)

MODEL_SEED = 20260725
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
RESUME_SIGMOID_ABS_TOLERANCE = 1e-7
RESUME_LOGIT_ABS_TOLERANCE = 1e-5
FEATURE_DIMENSION = 2048
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224

PRIMARY_PROFILE = "official_recommended_native_rgb_no_resize_no_crop"
PAPER_CROP_PROFILE = "paper_native_center_crop224_no_resize"
PREPROCESS_PROFILES = {
    PRIMARY_PROFILE: {
        "profile_id": PRIMARY_PROFILE,
        "role": "primary",
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "no_EXIF_transpose",
            "no_resize",
            "no_crop",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
        ],
        "resize": None,
        "center_crop": None,
        "batch_size": 1,
        "source_basis": "official_2020_06_uncropped_update_and_demo_default",
        "source_commit": MODEL_SOURCE_COMMIT,
        "selection_frozen_before_mouse_scores": True,
    },
    PAPER_CROP_PROFILE: {
        "profile_id": PAPER_CROP_PROFILE,
        "role": "preregistered_sensitivity_not_primary",
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "no_EXIF_transpose",
            "no_resize",
            "torchvision.transforms.functional.center_crop_224",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
        ],
        "resize": None,
        "center_crop": CROP_SIZE,
        "batch_size": 1,
        "source_basis": "CVPR_2020_paper_evaluation",
        "source_commit": PAPER_ERA_STABLE_COMMIT,
        "selection_frozen_before_mouse_scores": True,
    },
}

LICENSE_RECORD = {
    "repository": {
        "license": "CC-BY-NC-SA-4.0",
        "osi_open_source": False,
        "commercial_use_permitted": False,
    },
    "checkpoint": {
        "separate_terms_found": False,
        "conservative_terms": "CC-BY-NC-SA-4.0",
        "commercial_clearance_established": False,
    },
    "overall_commercial_clearance": "not_established",
}

CHECKPOINT = {
    "id": "CNNDetection-BlurJPEG0.1@official-dropbox",
    "filename": "blur_jpg_prob0.1.pth",
    "bytes": 282_442_597,
    "sha256": "a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea",
    "format": "torch_legacy_pickle_nested_training_checkpoint",
    "safe_load": "torch.load_weights_only_true",
    "outer_keys": ["model", "optimizer", "total_steps"],
    "state_entries": 320,
    "state_elements": 23_563_254,
    "trainable_parameters": 23_510_081,
    "optimizer_state_entries": 161,
    "optimizer_param_groups": 1,
    "total_steps": 270_048,
    "state_payload_sha256": (
        "8c62f887d5b97a0337f0ed598ac80cb9d86929613d3bc5c08fb0331b470c8931"
    ),
    "official_url": (
        "https://www.dropbox.com/s/h7tkpcgiwuftb6g/"
        "blur_jpg_prob0.1.pth?dl=1"
    ),
    "official_digest_published": False,
    "selection_basis": (
        "paper mean cross-generator AP 92.6; paper calls 0.1 a good balance "
        "and uses it for StyleGAN2, ranking, and calibration experiments"
    ),
}

SOURCE_FILES = {
    "LICENSE.txt": (
        "6079b20c4344bc679958d3709a8e2c09bafc5acb897531257288f4f6dcfb7471"
    ),
    "README.md": (
        "00a70018864cd01f873ce728a5b766cbc148aa7811e8e8e8b308dd4ff2b6d856"
    ),
    "demo.py": (
        "ae5aa6d4e6eb5490a31d9f2f50d5211c9be1f13392b264dc9189636b7565a1b4"
    ),
    "demo_dir.py": (
        "d4a30eee1477216f52393b9d2e9b37bf59ce12fede02de1220f3b6bd626bee0c"
    ),
    "eval.py": (
        "f71436d78631e074cfaf33b6ae86514144fd8894a3335ee8e60091a87b2c554b"
    ),
    "eval_config.py": (
        "2d6151e649f8ce5afb296381656107fa8223790d2942a31519dd2c19e01828f1"
    ),
    "validate.py": (
        "dca8aa2aed02d6630ba4d58feab69e2f80e4f76e803d5365bf35b2b9ca776be3"
    ),
    "data/datasets.py": (
        "689e113556829c2fc72e249f1ca36d8e65661e13fadf57da462c1c49e2cf8e32"
    ),
    "networks/resnet.py": (
        "987a2ea7f70f80bd60fbf46d5e610fb89df91cbaa35fc887e65243e205eb8e58"
    ),
    "options/base_options.py": (
        "ca8c38346ba9b4e1d8689c774b0a6f01053e31a8d1442228951c1e1e9b26521f"
    ),
    "options/test_options.py": (
        "89cdad6b02bf171f132bf19f4a0728925c6d76237568b2dc947674a3d8abc8f5"
    ),
    "requirements.txt": (
        "b627f074063b6212324f961af65ebfe0195330247afde041f1901d57d0d09d6b"
    ),
    "weights/download_weights.sh": (
        "3d9bdbb89c64e6ee1789b4bbd93bc4919bc099eaf9eef74dbf9de27b13f18f3c"
    ),
}

GOLDEN = {
    "examples": {
        "real.png": {
            "sha256": (
                "a2560a17f14006305580ba64d8bdc7d9128adbef5ce0f995396a267e23a587f9"
            ),
            "expected_fake": False,
        },
        "fake.png": {
            "sha256": (
                "7407a8a7bda0c1f8295347d8b5751ad8b9b3847f9243afdd362c2a923cdafe2e"
            ),
            "expected_fake": True,
        },
    },
    "profiles": {
        PRIMARY_PROFILE: {
            "real.png": {
                "tensor_sha256": (
                    "6e8f90314237016ba2ad3e9be22cbe2872e8a59c5ba717fb8d4313c0c5cdc7ef"
                ),
                "raw_logit": -23.74152374267578,
                "fake_score": 4.888630819599449e-11,
            },
            "fake.png": {
                "tensor_sha256": (
                    "5a314701ebf163d9082b939b6b8c3d786243cce7a8eb709e01b7b6e5250f1f7a"
                ),
                "raw_logit": 8.645806312561035,
                "fake_score": 0.9998242259025574,
            },
        },
        PAPER_CROP_PROFILE: {
            "real.png": {
                "tensor_sha256": (
                    "b7ea71e41313f29b4d6e80719cd5e60ca3307e78c5ea643b0660f7b14eba40da"
                ),
                "raw_logit": -31.707826614379883,
                "fake_score": 1.6961562024158348e-14,
            },
            "fake.png": {
                "tensor_sha256": (
                    "940d495730f1f7a9131d74148a2d4bd9b58e26a9dbd0dc059977cd7391262b6e"
                ),
                "raw_logit": 11.13868236541748,
                "fake_score": 0.9999854564666748,
            },
        },
    },
    "runtime": {
        "python": "3.12.3",
        "torch": "2.8.0.dev20250627+cu128",
        "torchvision": "0.23.0.dev20250627+cu128",
        "pillow": "11.1.0",
        "device": "cpu",
        "dtype": "float32",
    },
    "raw_logit_abs_tolerance": 1e-5,
    "fake_score_abs_tolerance": 1e-7,
}

DEFAULT_SOURCE_ROOT = Path("/root/.cache/claimforge/cnndetection_official")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/models/cnndetection/blur_jpg_prob0.1.pth"
)
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/cnndetection")
DEFAULT_RUN_ID = (
    "cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_"
    "full275_20260725"
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return _array_sha256(tensor.numpy())


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _verify_runtime_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if not _valid_sha256(expected):
        raise ValueError(f"invalid frozen SHA-256 for {label}")
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
    finally:
        temporary.unlink(missing_ok=True)


def adapter_contract(repo_root: Path) -> dict[str, Any]:
    relative_paths = (
        "eval/opensource/run_cnndetection.py",
        "eval/opensource/cnndetection_metrics.py",
        "eval/opensource/analyze_cnndetection_run.py",
        "eval/opensource/common.py",
        "eval/opensource/ufd_metrics.py",
    )
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing CNNDetection adapter component: {path}"
            )
        result[relative] = sha256_file(path)
    return result


def load_release(
    repo_root: Path,
    dataset_manifest_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    release = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != "claimforge_mouse_canonical_v1":
        raise ValueError("unsupported canonical release schema")
    inputs_value = release.get("inputs_path")
    if not isinstance(inputs_value, str):
        raise ValueError("canonical release has no inputs_path")
    inputs_path = _anchored(Path(inputs_value), repo_root)
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
    return release, inputs_path, rows


def select_inputs(
    rows: list[dict[str, Any]],
    pair_limit: int | None,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    if pair_limit is not None and sample_id is not None:
        raise ValueError("pair-limit and sample-id are mutually exclusive")
    if sample_id is not None:
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
    kinds: dict[int, set[str]] = {}
    for row in selected:
        kinds.setdefault(int(row["pair_rank"]), set()).add(str(row["kind"]))
    invalid = {
        rank: values
        for rank, values in kinds.items()
        if values != {"real", "forged"}
    }
    if invalid:
        raise ValueError(f"canonical selection contains incomplete pairs: {invalid}")
    return selected


def _load_gt_mask(
    row: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    sample_id = str(row.get("sample_id"))
    width, height = int(row["width"]), int(row["height"])
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


def compute_preprocess_geometry(
    width: int,
    height: int,
    profile_id: str,
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if profile_id not in PREPROCESS_PROFILES:
        raise ValueError(f"unknown preprocessing profile {profile_id}")
    if profile_id == PRIMARY_PROFILE:
        return {
            "decoded_size": [width, height],
            "resize": {"enabled": False},
            "center_crop": {"enabled": False},
            "output_size": [width, height],
            "effective_native_xyxy": [0, 0, width, height],
        }

    crop_width = crop_height = CROP_SIZE
    pad_left = (
        (crop_width - width) // 2 if crop_width > width else 0
    )
    pad_top = (
        (crop_height - height) // 2 if crop_height > height else 0
    )
    pad_right = (
        (crop_width - width + 1) // 2 if crop_width > width else 0
    )
    pad_bottom = (
        (crop_height - height + 1) // 2 if crop_height > height else 0
    )
    padded_width = width + pad_left + pad_right
    padded_height = height + pad_top + pad_bottom
    crop_left = int(round((padded_width - crop_width) / 2.0))
    crop_top = int(round((padded_height - crop_height) / 2.0))
    native_left = crop_left - pad_left
    native_top = crop_top - pad_top
    native_right = native_left + crop_width
    native_bottom = native_top + crop_height
    return {
        "decoded_size": [width, height],
        "resize": {"enabled": False},
        "center_crop": {
            "enabled": True,
            "size": [crop_width, crop_height],
            "padding_ltrb": [
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            "start_xy_in_padded_image": [crop_left, crop_top],
            "native_crop_xyxy": [
                native_left,
                native_top,
                native_right,
                native_bottom,
            ],
        },
        "output_size": [crop_width, crop_height],
        "effective_native_xyxy": [
            max(0, native_left),
            max(0, native_top),
            min(width, native_right),
            min(height, native_bottom),
        ],
    }


def preprocess_image(
    path: Path,
    profile_id: str = PRIMARY_PROFILE,
) -> tuple[Any, dict[str, Any]]:
    """Apply the exact official decode/crop/tensor/normalization contract."""

    import torch
    from torchvision.transforms import functional as vision_functional

    if profile_id not in PREPROCESS_PROFILES:
        raise ValueError(f"unknown preprocessing profile {profile_id}")
    with Image.open(path) as opened:
        rgb_image = opened.convert("RGB")
        width, height = rgb_image.size
        decoded_rgb = np.ascontiguousarray(
            np.asarray(rgb_image, dtype=np.uint8)
        )
        if profile_id == PAPER_CROP_PROFILE:
            transformed = vision_functional.center_crop(
                rgb_image,
                [CROP_SIZE, CROP_SIZE],
            )
        else:
            transformed = rgb_image
        output_rgb = np.ascontiguousarray(
            np.asarray(transformed, dtype=np.uint8)
        )
        tensor = vision_functional.to_tensor(transformed)
    tensor = vision_functional.normalize(tensor, IMAGE_MEAN, IMAGE_STD)
    tensor = tensor.contiguous()
    if tensor.dtype != torch.float32:
        raise ValueError("CNNDetection preprocessing did not produce float32")
    geometry = compute_preprocess_geometry(width, height, profile_id)
    if list(tensor.shape) != [
        3,
        int(geometry["output_size"][1]),
        int(geometry["output_size"][0]),
    ]:
        raise ValueError("CNNDetection tensor shape disagrees with geometry")
    audit = {
        "profile": profile_id,
        "steps": list(PREPROCESS_PROFILES[profile_id]["steps"]),
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(decoded_rgb.shape),
        "decoded_rgb_dtype": str(decoded_rgb.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded_rgb),
        "output_rgb_shape": list(output_rgb.shape),
        "output_rgb_dtype": str(output_rgb.dtype),
        "output_rgb_sha256": _array_sha256(output_rgb),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.numpy().dtype),
        "tensor_sha256": _tensor_sha256(tensor),
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "geometry": geometry,
    }
    return tensor, audit


def build_pair_visibility(
    rows: list[dict[str, Any]],
    repo_root: Path,
    profile_id: str,
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair_rows in by_task.items():
        forged = [row for row in pair_rows if row.get("kind") == "forged"]
        if len(forged) != 1:
            raise ValueError(f"task {task_id} does not have one forged row")
        row = forged[0]
        mask = _load_gt_mask(row, repo_root)
        assert mask is not None
        width, height = int(row["width"]), int(row["height"])
        geometry = compute_preprocess_geometry(width, height, profile_id)
        left, top, right, bottom = [
            int(value) for value in geometry["effective_native_xyxy"]
        ]
        total = int(np.count_nonzero(mask == 255))
        visible = int(np.count_nonzero(mask[top:bottom, left:right] == 255))
        fraction = visible / total
        category = (
            "none"
            if visible == 0
            else "full"
            if visible == total
            else "partial"
        )
        result[task_id] = {
            "edit_visibility": category,
            "edit_visible_gt_fraction": fraction,
            "edit_visible_gt_pixels": visible,
            "edit_total_gt_pixels": total,
            "effective_native_xyxy": [left, top, right, bottom],
            "evidence": (
                "exact_diff_mask_intersection_with_frozen_preprocess_geometry"
            ),
            "preprocess_profile": profile_id,
        }
    return result


def _verify_source_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"missing CNNDetection source-root: {source_root}"
        )
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"CNNDetection source commit mismatch: "
            f"{commit} != {MODEL_SOURCE_COMMIT}"
        )
    tracked_status = _git_value(
        source_root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if tracked_status is None:
        raise ValueError("cannot inspect CNNDetection source status")
    if tracked_status:
        raise ValueError("CNNDetection tracked source tree is dirty")
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"CNNDetection source {relative}",
        )
    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root.resolve()),
        "commit": commit,
        "paper_era_stable_commit": PAPER_ERA_STABLE_COMMIT,
        "tracked_dirty": False,
        "source_files": dict(SOURCE_FILES),
        "core_inference_byte_identical_to_paper_era_commit": True,
    }


def _import_official_resnet(source_root: Path) -> Any:
    module_name = f"_claimforge_cnndetection_resnet_{MODEL_SOURCE_COMMIT[:12]}"
    path = source_root / "networks/resnet.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official ResNet source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        if not isinstance(name, str) or not hasattr(tensor, "detach"):
            raise ValueError("checkpoint model state is not tensor-only")
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def verify_assets(
    *,
    source_root: Path,
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    OrderedDict[str, Any],
    Any,
]:
    """Verify immutable source/checkpoint and safely load the model state."""

    import torch

    source = _verify_source_contract(source_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing CNNDetection checkpoint: {checkpoint_path}"
        )
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("CNNDetection checkpoint byte-size mismatch")
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "CNNDetection checkpoint",
    )

    static_scan: dict[str, Any]
    try:
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(
            checkpoint_path
        )
        static_scan = {
            "supported": True,
            "unsafe_globals": list(unsafe),
        }
        if unsafe:
            raise ValueError(
                "CNNDetection checkpoint contains unsafe pickle globals"
            )
    except ValueError as exc:
        if "Expected input to be a checkpoint returned by torch.save" not in str(
            exc
        ):
            raise
        static_scan = {
            "supported": False,
            "unsafe_globals": None,
            "reason": (
                "PyTorch static scanner does not support this legacy "
                "torch.save stream"
            ),
        }

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("CNNDetection checkpoint outer payload is not a dict")
    if list(payload) != list(CHECKPOINT["outer_keys"]):
        raise ValueError("CNNDetection checkpoint outer keys changed")
    state = payload.get("model")
    if not isinstance(state, OrderedDict):
        raise ValueError("CNNDetection model payload is not an OrderedDict")
    if len(state) != int(CHECKPOINT["state_entries"]):
        raise ValueError("CNNDetection state entry count changed")
    elements = sum(int(tensor.numel()) for tensor in state.values())
    if elements != int(CHECKPOINT["state_elements"]):
        raise ValueError("CNNDetection state element count changed")
    fingerprint = _state_dict_fingerprint(state)
    if fingerprint != CHECKPOINT["state_payload_sha256"]:
        raise ValueError("CNNDetection state payload fingerprint changed")
    expected_shapes = {
        "conv1.weight": (64, 3, 7, 7),
        "fc.weight": (1, FEATURE_DIMENSION),
        "fc.bias": (1,),
    }
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"CNNDetection {name} shape changed")
        if state[name].dtype != torch.float32:
            raise ValueError(f"CNNDetection {name} dtype changed")
    if int(payload.get("total_steps", -1)) != int(CHECKPOINT["total_steps"]):
        raise ValueError("CNNDetection total_steps changed")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "state",
        "param_groups",
    }:
        raise ValueError("CNNDetection optimizer payload changed")
    if len(optimizer["state"]) != int(CHECKPOINT["optimizer_state_entries"]):
        raise ValueError("CNNDetection optimizer state count changed")
    if len(optimizer["param_groups"]) != int(
        CHECKPOINT["optimizer_param_groups"]
    ):
        raise ValueError("CNNDetection optimizer group count changed")

    module = _import_official_resnet(source_root)
    model = module.resnet50(num_classes=1)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("CNNDetection strict model load was not clean")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(CHECKPOINT["trainable_parameters"]):
        raise ValueError("CNNDetection model parameter count changed")
    del model
    del payload
    gc.collect()
    asset = {
        **CHECKPOINT,
        "path": str(checkpoint_path.resolve()),
        "serialization_safety": {
            "weights_only": True,
            "weights_only_load_succeeded": True,
            "static_unsafe_global_scan": static_scan,
            "unrestricted_pickle_used": False,
        },
        "schema": {
            "outer_type": "dict",
            "outer_keys": list(CHECKPOINT["outer_keys"]),
            "model_type": "OrderedDict",
            "state_entries": len(state),
            "state_elements": elements,
            "state_payload_sha256": fingerprint,
            "conv1_weight_shape": [64, 3, 7, 7],
            "fc_weight_shape": [1, FEATURE_DIMENSION],
            "fc_bias_shape": [1],
            "optimizer_state_entries": len(optimizer["state"]),
            "optimizer_param_groups": len(optimizer["param_groups"]),
            "total_steps": int(CHECKPOINT["total_steps"]),
            "strict_model_load": True,
        },
    }
    return source, asset, state, module


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
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

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    evidence: dict[str, Any] = {
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "pillow": _package_version("Pillow"),
        "numpy": np.__version__,
        "seed": MODEL_SEED,
        "dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": (
                bool(getattr(torch.backends.cudnn, "allow_tf32", False))
                if device.type == "cuda"
                else False
            ),
        },
        "matmul_allow_tf32": (
            bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
            if device.type == "cuda"
            else False
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        evidence["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    return device, evidence


def load_model(
    *,
    module: Any,
    state: OrderedDict[str, Any],
    device: Any,
) -> Any:
    import torch

    model = module.resnet50(num_classes=1)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def infer_one(
    *,
    model: Any,
    tensor: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch
    import torch.nn.functional as functional

    captured: list[Any] = []

    def capture_fc_input(_module: Any, arguments: tuple[Any, ...]) -> None:
        if len(arguments) != 1:
            raise RuntimeError(
                "CNNDetection fc hook received unexpected arguments"
            )
        captured.append(arguments[0].detach())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    hook = model.fc.register_forward_pre_hook(capture_fc_input)
    try:
        with torch.inference_mode():
            output = model(
                tensor.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
            )
    finally:
        hook.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if list(output.shape) != [1, 1]:
        raise ValueError(
            f"unexpected CNNDetection output shape {list(output.shape)}"
        )
    if len(captured) != 1 or list(captured[0].shape) != [
        1,
        FEATURE_DIMENSION,
    ]:
        raise ValueError("CNNDetection fc feature hook did not fire once")
    feature_device = captured[0]
    with torch.inference_mode():
        replay_output = functional.linear(
            feature_device,
            model.fc.weight,
            model.fc.bias,
        )
        score_tensor = torch.sigmoid(output)
        replay_score_tensor = torch.sigmoid(replay_output)
    if not torch.equal(output, replay_output):
        raise ValueError("CNNDetection manual fc replay changed the logit")
    if not torch.equal(score_tensor, replay_score_tensor):
        raise ValueError("CNNDetection manual sigmoid replay changed the score")
    raw_logit = float(output.reshape(()).item())
    score = float(score_tensor.reshape(()).item())
    if not math.isfinite(raw_logit):
        raise ValueError("CNNDetection emitted a non-finite raw logit")
    if not 0.0 <= score <= 1.0:
        raise ValueError("CNNDetection score falls outside [0, 1]")
    decision = score > CLASSIFICATION_THRESHOLD
    feature = np.ascontiguousarray(
        feature_device.squeeze(0).detach().cpu().numpy(),
        dtype=np.float32,
    )
    if feature.shape != (FEATURE_DIMENSION,) or not np.isfinite(feature).all():
        raise ValueError("CNNDetection pooled feature is invalid")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    aliases = {
        "raw_logit": raw_logit,
        "probability": score,
        "ai_score": score,
        "score": score,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "semantics": (
            "official_float32_sigmoid_uncalibrated_fake_score"
        ),
    }
    return (
        {
            "raw_logit": raw_logit,
            "probability": score,
            "ai_score": score,
            "score": score,
            "score_semantics": aliases["semantics"],
            "calibrated_probability": False,
            "classification_decision": decision,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "classification_threshold_operator": (
                CLASSIFICATION_THRESHOLD_OPERATOR
            ),
            "classification": dict(aliases),
            "t1": {
                **aliases,
                "policy": (
                    "official_CNNDetection_float32_sigmoid_strict_gt_0_5"
                ),
            },
            "manual_replay": {
                "raw_logit": raw_logit,
                "probability": score,
                "ai_score": score,
                "classification_decision": decision,
                "model_forward_calls": 1,
                "fc_hook_calls": 1,
                "official_logit_exact_match": True,
                "official_score_exact_match": True,
            },
            "latency_ms": latency_ms,
            "peak_cuda_memory_bytes": peak_memory,
        },
        feature,
    )


def _result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    visibility: Mapping[str, Any],
    config_fingerprint: str,
    profile_id: str,
) -> dict[str, Any]:
    path = _anchored(Path(str(row["canonical_path"])), repo_root)
    return {
        "schema_version": "cnndetection_result_v1",
        "id": str(row["sample_id"]),
        "sample_id": str(row["sample_id"]),
        "rank": int(row["rank"]),
        "pair_rank": int(row["pair_rank"]),
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "label": int(row["label"]),
        "domain": str(row["domain"]),
        "candidate": str(row.get("candidate", "mouse")),
        "dataset_id": str(row.get("dataset_id")),
        "input_path": repo_relative(path, repo_root),
        "input_sha256": str(row["canonical_sha256"]),
        "input_width": int(row["width"]),
        "input_height": int(row["height"]),
        "preprocess_profile": profile_id,
        "checkpoint_id": str(CHECKPOINT["id"]),
        "config_fingerprint": config_fingerprint,
        "edit_visibility": str(visibility["edit_visibility"]),
        "edit_visible_gt_fraction": float(
            visibility["edit_visible_gt_fraction"]
        ),
        "edit_visibility_evidence": dict(visibility),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
    }


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    config_fingerprint: str,
    model: Any,
    device: Any,
) -> None:
    import torch
    import torch.nn.functional as functional

    if row.get("status") != "ok":
        raise ValueError("resume row is not successful")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume row field {key} changed")
    if row.get("config_fingerprint") != config_fingerprint:
        raise ValueError("resume row config fingerprint changed")

    input_value = expected.get("input_path")
    profile_id = expected.get("preprocess_profile")
    if not isinstance(input_value, str) or profile_id not in PREPROCESS_PROFILES:
        raise ValueError("resume expected preprocessing contract is invalid")
    input_path = _anchored(Path(input_value), repo_root)
    tensor, expected_preprocess = preprocess_image(input_path, str(profile_id))
    del tensor
    if row.get("preprocess") != expected_preprocess:
        raise ValueError("resume row preprocessing audit changed")

    for key in ("preprocess_latency_ms", "latency_ms"):
        value = row.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"resume row {key} is invalid")
    peak_memory = row.get("peak_cuda_memory_bytes")
    if peak_memory is not None and (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < 0
    ):
        raise ValueError("resume row peak CUDA memory is invalid")

    score = row.get("ai_score")
    logit = row.get("raw_logit")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("resume row has invalid CNNDetection score")
    if (
        isinstance(logit, bool)
        or not isinstance(logit, (int, float))
        or not math.isfinite(float(logit))
    ):
        raise ValueError("resume row has invalid CNNDetection logit")
    expected_score = float(
        torch.sigmoid(
            torch.tensor(float(logit), dtype=torch.float32)
        ).item()
    )
    if not math.isclose(
        float(score),
        expected_score,
        rel_tol=0.0,
        abs_tol=RESUME_SIGMOID_ABS_TOLERANCE,
    ):
        raise ValueError("resume row sigmoid/logit relation changed")
    for key in ("probability", "score"):
        if row.get(key) != score:
            raise ValueError(f"resume row score alias {key} changed")
    decision = float(score) > CLASSIFICATION_THRESHOLD
    if row.get("classification_decision") is not decision:
        raise ValueError("resume row decision changed")
    semantics = "official_float32_sigmoid_uncalibrated_fake_score"
    if row.get("score_semantics") != semantics:
        raise ValueError("resume row score semantics changed")
    if row.get("calibrated_probability") is not False:
        raise ValueError("resume row incorrectly marks score calibrated")
    if row.get("classification_threshold") != CLASSIFICATION_THRESHOLD:
        raise ValueError("resume row classification threshold changed")
    if (
        row.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
    ):
        raise ValueError("resume row threshold operator changed")
    aliases = {
        "raw_logit": float(logit),
        "probability": float(score),
        "ai_score": float(score),
        "score": float(score),
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "semantics": semantics,
    }
    if row.get("classification") != aliases:
        raise ValueError("resume row nested classification changed")
    if row.get("t1") != {
        **aliases,
        "policy": "official_CNNDetection_float32_sigmoid_strict_gt_0_5",
    }:
        raise ValueError("resume row nested T1 scoring changed")
    if row.get("manual_replay") != {
        "raw_logit": float(logit),
        "probability": float(score),
        "ai_score": float(score),
        "classification_decision": decision,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "official_logit_exact_match": True,
        "official_score_exact_match": True,
    }:
        raise ValueError("resume row manual replay changed")

    feature_value = row.get("cnndetection_feature_path")
    feature_sha = row.get("cnndetection_feature_sha256")
    if not isinstance(feature_value, str) or not _valid_sha256(feature_sha):
        raise ValueError("resume row feature artifact contract changed")
    feature_path = _anchored(Path(feature_value), repo_root)
    _verify_runtime_file(
        feature_path,
        str(feature_sha),
        "CNNDetection resume feature",
    )
    feature = np.load(feature_path, allow_pickle=False)
    if feature.shape != (FEATURE_DIMENSION,):
        raise ValueError("resume row feature shape changed")
    if feature.dtype != np.float32:
        raise ValueError("resume row feature dtype changed")
    if not np.isfinite(feature).all():
        raise ValueError("resume row feature contains non-finite values")
    if row.get("cnndetection_feature_shape") != [FEATURE_DIMENSION]:
        raise ValueError("resume row feature shape metadata changed")
    if row.get("cnndetection_feature_dtype") != "float32":
        raise ValueError("resume row feature dtype metadata changed")
    if row.get("cnndetection_feature_semantics") != (
        "official_fc_input_after_adaptive_global_average_pool"
    ):
        raise ValueError("resume row feature semantics changed")

    with torch.inference_mode():
        feature_tensor = torch.from_numpy(feature).to(
            device=device,
            dtype=torch.float32,
        )
        replay_logit_tensor = functional.linear(
            feature_tensor.unsqueeze(0),
            model.fc.weight,
            model.fc.bias,
        )
        replay_logit = float(replay_logit_tensor.reshape(()).item())
        replay_score = float(
            torch.sigmoid(replay_logit_tensor).reshape(()).item()
        )
    if not math.isclose(
        replay_logit,
        float(logit),
        rel_tol=0.0,
        abs_tol=RESUME_LOGIT_ABS_TOLERANCE,
    ):
        raise ValueError("resume row saved feature/logit replay changed")
    if not math.isclose(
        replay_score,
        float(score),
        rel_tol=0.0,
        abs_tol=RESUME_SIGMOID_ABS_TOLERANCE,
    ):
        raise ValueError("resume row saved feature/score replay changed")


def _run_config(
    *,
    adapter: Mapping[str, Any],
    release: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source_audit: Mapping[str, Any],
    asset_audit: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    profile_id: str,
    pair_limit: int | None,
    sample_id: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "model_arch": MODEL_ARCH,
        "repo_url": MODEL_REPO_URL,
        "source_commit": MODEL_SOURCE_COMMIT,
        "paper_url": PAPER_URL,
        "checkpoint_id": CHECKPOINT["id"],
        "checkpoint_sha256": CHECKPOINT["sha256"],
        "checkpoint_bytes": CHECKPOINT["bytes"],
        "checkpoint_selection_frozen_before_mouse_scores": True,
        "checkpoint_variant": "Blur+JPEG(0.1)",
        "excluded_primary_variant": {
            "name": "Blur+JPEG(0.5)",
            "reason": (
                "README quick-start default only; not selected through "
                "post-hoc Mouse performance"
            ),
        },
        "preprocess_profile": profile_id,
        "preprocess_profile_contract": PREPROCESS_PROFILES[profile_id],
        "available_profiles": PREPROCESS_PROFILES,
        "profile_selection_frozen_before_mouse_scores": True,
        "no_test_time_blur_or_jpeg": True,
        "batch_size": 1,
        "classification": {
            "raw_output": "one_float32_logit",
            "score": (
                "torch_float32_sigmoid_uncalibrated_fake_score"
            ),
            "calibrated_probability": False,
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "score_direction": "higher_means_fake",
            "target_domain_calibration": "forbidden",
        },
        "task_scope": {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "joint_output": None,
        },
        "adapter": dict(adapter),
        "source": dict(source_audit),
        "asset": dict(asset_audit),
        "runtime": dict(runtime_evidence),
        "dataset": {
            "schema_version": release["schema_version"],
            "dataset_id": release.get("dataset_id"),
            "inputs_sha256": release["inputs_sha256"],
            "selected_ids": [str(row["sample_id"]) for row in selected],
            "selected_rows_sha256": hashlib.sha256(
                "".join(
                    f"{stable_json(row)}\n" for row in selected
                ).encode("utf-8")
            ).hexdigest(),
            "pair_limit": pair_limit,
            "sample_id": sample_id,
        },
        "metrics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "raw_logit_diagnostic": (
                "always_reported_never_replaces_released_sigmoid_rule"
            ),
        },
        "license": LICENSE_RECORD,
    }


def run_preflight(
    *,
    source_root: Path,
    checkpoint_path: Path,
    device_text: str = "cpu",
) -> dict[str, Any]:
    """Verify assets and execute both frozen official-example goldens."""

    if device_text != "cpu":
        raise ValueError("CNNDetection preflight is intentionally CPU-only")
    source, asset, state, module = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
    device, runtime = configure_runtime(device_text)
    model = load_model(module=module, state=state, device=device)
    cases: list[dict[str, Any]] = []
    try:
        for profile_id in (PRIMARY_PROFILE, PAPER_CROP_PROFILE):
            for filename in ("real.png", "fake.png"):
                path = source_root / "examples" / filename
                expected_image = GOLDEN["examples"][filename]
                _verify_runtime_file(
                    path,
                    str(expected_image["sha256"]),
                    f"CNNDetection golden {filename}",
                )
                tensor, preprocess = preprocess_image(path, profile_id)
                expected = GOLDEN["profiles"][profile_id][filename]
                if preprocess["tensor_sha256"] != expected["tensor_sha256"]:
                    raise ValueError(
                        f"CNNDetection {profile_id}/{filename} tensor golden "
                        "changed"
                    )
                scoring, feature = infer_one(
                    model=model,
                    tensor=tensor,
                    device=device,
                )
                if not math.isclose(
                    scoring["raw_logit"],
                    float(expected["raw_logit"]),
                    rel_tol=0.0,
                    abs_tol=float(GOLDEN["raw_logit_abs_tolerance"]),
                ):
                    raise ValueError(
                        f"CNNDetection {profile_id}/{filename} logit golden "
                        "changed"
                    )
                if not math.isclose(
                    scoring["ai_score"],
                    float(expected["fake_score"]),
                    rel_tol=0.0,
                    abs_tol=float(GOLDEN["fake_score_abs_tolerance"]),
                ):
                    raise ValueError(
                        f"CNNDetection {profile_id}/{filename} score golden "
                        "changed"
                    )
                if scoring["classification_decision"] is not bool(
                    expected_image["expected_fake"]
                ):
                    raise ValueError(
                        f"CNNDetection {profile_id}/{filename} decision golden "
                        "changed"
                    )
                cases.append(
                    {
                        "profile": profile_id,
                        "filename": filename,
                        "input_sha256": expected_image["sha256"],
                        "tensor_sha256": preprocess["tensor_sha256"],
                        "raw_logit": scoring["raw_logit"],
                        "fake_score": scoring["ai_score"],
                        "classification_decision": scoring[
                            "classification_decision"
                        ],
                        "feature_sha256": _array_sha256(feature),
                    }
                )
    finally:
        del model
        del state
        gc.collect()
    return {
        "schema_version": "cnndetection_preflight_v1",
        "status": "passed",
        "source": source,
        "asset": asset,
        "runtime": runtime,
        "golden_reference_runtime": GOLDEN["runtime"],
        "golden_cases": cases,
        "cuda_used": False,
        "mouse_inference_run": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--preprocess-profile",
        choices=tuple(PREPROCESS_PROFILES),
        default=PRIMARY_PROFILE,
    )
    parser.add_argument(
        "--device",
        help=(
            "explicit cpu or cuda:N; defaults to cpu for --preflight and "
            "cuda:0 for inference runs"
        ),
    )
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight", action="store_true")
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
    device_text = args.device or ("cpu" if args.preflight else "cuda:0")

    if args.preflight:
        if args.resume or args.pair_limit is not None or args.sample_id:
            raise ValueError(
                "preflight cannot be combined with resume or input selection"
            )
        report = run_preflight(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device_text=device_text,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one non-empty path component")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    run_dir = results_root / args.run_id
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    feature_dir = run_dir / "features"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(all_rows, args.pair_limit, args.sample_id)
    validate_selected_inputs(selected, repo_root)
    visibility = build_pair_visibility(
        all_rows,
        repo_root,
        args.preprocess_profile,
    )
    source_audit, asset_audit, state, module = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
    device, runtime = configure_runtime(device_text)
    config = _run_config(
        adapter=adapter_contract(repo_root),
        release=release,
        selected=selected,
        source_audit=source_audit,
        asset_audit=asset_audit,
        runtime_evidence=runtime,
        profile_id=args.preprocess_profile,
        pair_limit=args.pair_limit,
        sample_id=args.sample_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    config_fingerprint = _manifest_fingerprint(config)

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires existing manifest and expected-input snapshot"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if prior_manifest.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume manifest config fingerprint mismatch")
        if prior_manifest.get("runtime") != runtime:
            raise ValueError("resume runtime evidence changed")
        if read_jsonl(expected_path) != selected:
            raise ValueError("resume expected-input snapshot changed")
    else:
        atomic_write_jsonl(expected_path, selected)

    manifest: dict[str, Any] = {
        "schema_version": "cnndetection_run_manifest_v1",
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "repo_root": str(repo_root),
        "config_fingerprint": config_fingerprint,
        "config": config,
        "source": source_audit,
        "asset": asset_audit,
        "runtime": runtime,
        "dataset": {
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "inputs_path": repo_relative(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
            "selected_tasks": len(
                {str(row["task_id"]) for row in selected}
            ),
        },
        "visibility_census": dict(
            Counter(
                visibility[task_id]["edit_visibility"]
                for task_id in sorted(
                    {str(row["task_id"]) for row in selected}
                )
            )
        ),
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "feature_dir": repo_relative(feature_dir, repo_root),
        },
    }
    atomic_write_json(manifest_path, manifest)

    model = load_model(module=module, state=state, device=device)
    del state
    latest = read_latest_by_id(results_path)
    completed = 0
    skipped = 0
    errors = 0
    for index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        pair_visibility = visibility[str(row["task_id"])]
        identity = _result_identity(
            row,
            repo_root=repo_root,
            visibility=pair_visibility,
            config_fingerprint=config_fingerprint,
            profile_id=args.preprocess_profile,
        )
        prior = latest.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
            _validate_resume_row(
                prior,
                expected=identity,
                repo_root=repo_root,
                config_fingerprint=config_fingerprint,
                model=model,
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
            tensor, preprocess = preprocess_image(
                input_path,
                args.preprocess_profile,
            )
            preprocess_latency_ms = (
                time.perf_counter() - preprocess_started
            ) * 1000.0
            scoring, feature = infer_one(
                model=model,
                tensor=tensor,
                device=device,
            )
            feature_path = feature_dir / f"{sample_id}.npy"
            _atomic_save_npy(feature_path, feature)
            result = {
                **identity,
                "status": "ok",
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_latency_ms,
                "cnndetection_feature_path": repo_relative(
                    feature_path,
                    repo_root,
                ),
                "cnndetection_feature_sha256": sha256_file(feature_path),
                "cnndetection_feature_shape": list(feature.shape),
                "cnndetection_feature_dtype": str(feature.dtype),
                "cnndetection_feature_semantics": (
                    "official_fc_input_after_adaptive_global_average_pool"
                ),
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
                "completed_at": utc_now(),
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
    summary = summarize_cnndetection_results(
        physical_results,
        selected,
        threshold=CLASSIFICATION_THRESHOLD,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary["raw_logit_diagnostic"] = summarize_cnndetection_raw_logits(
        physical_results,
        selected,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_id": CHECKPOINT["id"],
            "preprocess_profile": args.preprocess_profile,
            "config_fingerprint": config_fingerprint,
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
    del model
    gc.collect()
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
