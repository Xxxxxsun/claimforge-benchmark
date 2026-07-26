#!/usr/bin/env python3
"""Run the pinned official B-Free DINOv2-register detector.

The release contract is frozen before any Mouse score is inspected.  B-Free
decodes RGB without resizing, embeds native pixels with a stride-14 patch
projection, optionally repeat-wraps a too-small patch grid, evaluates five
504-pixel crops, and averages their raw logits.  The official strict decision
is ``raw_logit > 0``.  B-Free is a whole-image classifier (T1) and does not
release a localization output (T2).
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import importlib.util
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
import zipfile
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
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


MODEL_NAME = "B-Free"
MODEL_SLUG = "bfree_dino2reg4"
MODEL_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
MODEL_REPO_URL = "https://github.com/grip-unina/B-Free"
MODEL_SOURCE_COMMIT = "c6a9f898782fb466b29af01f21960b67415afb0e"
PAPER_URL = "https://arxiv.org/abs/2412.17671"
WEIGHTS_URL = (
    "https://www.grip.unina.it/download/prog/B-Free/weights/"
    "BFREE_dino2reg4.zip"
)

PREPROCESS_PROFILE = "official_native_rgb_resnet_norm_dinov2_5crop504"
MODEL_SEED = 20260725
PATCH_STRIDE = 14
PATCH_SIZE = PATCH_STRIDE
CROP_SIZE = 504
CROP_PATCHES = 36
CROP_COUNT = 5
CROPS = CROP_COUNT
FEATURE_DIMENSION = 768
FEATURE_SEMANTICS = "five_dinov2_head_input_vectors_official_crop_order"
SCORE_SEMANTICS = "official_float32_mean_of_five_crop_raw_logits"
T1_POLICY = "released_mean_raw_logit_strictly_greater_than_0"
CLASSIFICATION_THRESHOLD = 0.0
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
TIMM_VERSION = "1.0.12"
TRANSFORMERS_VERSION = "4.43.4"

OFFICIAL_ZIP = {
    "filename": "BFREE_dino2reg4.zip",
    "url": WEIGHTS_URL,
    "bytes": 321_653_488,
    "sha256": "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f",
    "md5": "f3f53fa647848b16cf81c913f148a198",
    "members": {
        "BFREE_dino2reg4/": 0,
        "BFREE_dino2reg4/config.yaml": 153,
        "BFREE_dino2reg4/model_epoch_best.pth": 346_171_370,
    },
}
CHECKPOINT = {
    "id": "official_grip_unina_BFREE_dino2reg4_model_epoch_best",
    "filename": "model_epoch_best.pth",
    "bytes": 346_171_370,
    "sha256": "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947",
    "format": "torch_pth",
    "top_level_keys": ["model"],
    "state_container": "collections.OrderedDict",
    "tensor_count": 177,
    "state_elements": 86_526_721,
    "dtype": "float32",
    "schema_sha256": "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524",
}
CONFIG = {
    "filename": "config.yaml",
    "bytes": 153,
    "sha256": "1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40",
    "parsed": {
        "arch": "timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m",
        "model_name": "BFREE_dino2reg4",
        "norm_type": "resnet",
        "patch_size": None,
        "weights_file": "model_epoch_best.pth",
    },
}

SOURCE_FILES = {
    "LICENSE.txt": "cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27",
    "README.md": "6633483801daf6574c05afe8d2e892d4f84afb338cde467218899c346366c185",
    "code/LICENSE.txt": "cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27",
    "code/README.md": "81386f0127828890f1a8a4126470b9bc311592dbb9ba7f601eb1806adb8cde5f",
    "code/main_bfree.py": "eea848f57d1415c2c52804ff013a611435cdb38371c297c2de67fb459e4079ce",
    "code/main_bfree_single.py": "72f43b8999d2b36689dfa77728f2cf51000a1333a62aa2aad65973ef881c24f7",
    "code/networks/__init__.py": "048638ddd724ebbfbe995c3a735284a0551b8ca5fec74ca6a0ac2a5a4e6dd8cf",
    "code/networks/wrapper5crops.py": "1f4de65b82b33c3864ab368836bab009a18f9bf0f828335777272645f236d60b",
    "code/utils/normalization.py": "12a244d489f001ee7f25aae9bbe2b8fe1b1172f365b1ac0b8d632257f6c2354b",
    "code/utils/dmetrics.py": "fbb4a29d4f7d1d8492a28190f011dc4703a4894b8622d86a2f84ab920f836244",
    "code/requirements.txt": "8fd3131b5fbe8e16cdcbd0b022dee6f1213a216e989f52dd39005ca31393a168",
    "code/demo_images/metainfo.csv": "b939eee3b5c8a6a2bf41a687e1be0454aaee7a850a4581d06bbad50fd49496d4",
    "code/demo_images/results.csv": "81a3c434bfd3ec1aceb667fa82b86f3a55c84c390ab8b97029a4e5de0a1958a3",
    "code/demo_images/results_metrics.csv": "5f565b8d51606670a5044d5c9928ca6f7966bd336d4f56aca04d2b0fdcb916ea",
    "code/demo_images/img0000.png": "c7351aee67f37fe5acf1aa7781612b2760b90e0d56010038ec2e48ff9a79360e",
    "code/demo_images/img0001.png": "34f54d4ee77bea6640b87d2665ff5c3871ee848837776b53ab58fc1ec3cddead",
    "code/demo_images/img0002.png": "0d54da3b1a23b2a9aa235c7cbddfb5100d9a4d019fc6621e2c1126d910eb5f08",
    "code/demo_images/img0003.png": "a0947013ca31ac169878892fa6c0efa43e22beb1f069a26028d23604d5f931fa",
}

GOLDEN_CASES = (
    {
        "filename": "img0000.png",
        "sha256": SOURCE_FILES["code/demo_images/img0000.png"],
        "label": 0,
        "published_raw_logit": -5.9374785,
        "frozen_cpu_raw_logit": -5.93747091293335,
        "frozen_cuda_raw_logit": -5.937470436096191,
        "decoded_rgb_sha256": "13f331e6926c61747afb70325fa423408a2cff09405b36a5feb6a24b0723e216",
        "tensor_sha256": "aa250c75b0da43bc9eafacaef948a87094149579337c37249a432ea6bd70412b",
        "tensor_shape": [3, 1256, 835],
        "patch_grid_wh": [59, 89],
        "crop_starts_patch_xy": [
            [11, 26],
            [0, 0],
            [0, 53],
            [23, 53],
            [23, 0],
        ],
    },
    {
        "filename": "img0001.png",
        "sha256": SOURCE_FILES["code/demo_images/img0001.png"],
        "label": 0,
        "published_raw_logit": -4.441922,
        "frozen_cpu_raw_logit": -4.441921710968018,
        "frozen_cuda_raw_logit": -4.441922187805176,
        "decoded_rgb_sha256": "425f264c07dc97ec044b32a9a2bcb538e8750ec2b584feb178d5e6d29d09925d",
        "tensor_sha256": "e205b12364b070d29500ba6733e05b0735f2e37049c251ef9a76b890ef8ee91f",
        "tensor_shape": [3, 833, 1258],
        "patch_grid_wh": [89, 59],
        "crop_starts_patch_xy": [
            [26, 11],
            [0, 0],
            [0, 23],
            [53, 23],
            [53, 0],
        ],
    },
    {
        "filename": "img0002.png",
        "sha256": SOURCE_FILES["code/demo_images/img0002.png"],
        "label": 1,
        "published_raw_logit": 4.430519,
        "frozen_cpu_raw_logit": 4.430544853210449,
        "frozen_cuda_raw_logit": 4.430531978607178,
        "decoded_rgb_sha256": "32a4651c3efbe85d917a325de56cec5dee78151785a962301d7f29645e328491",
        "tensor_sha256": "b2d25ae0bebc69f0fa1580031dec8f7cd3a5e088bd8449a2c5276c4cc9c332db",
        "tensor_shape": [3, 1024, 1024],
        "patch_grid_wh": [73, 73],
        "crop_starts_patch_xy": [
            [18, 18],
            [0, 0],
            [0, 37],
            [37, 37],
            [37, 0],
        ],
    },
    {
        "filename": "img0003.png",
        "sha256": SOURCE_FILES["code/demo_images/img0003.png"],
        "label": 1,
        "published_raw_logit": 3.8499813,
        "frozen_cpu_raw_logit": 3.8499996662139893,
        "frozen_cuda_raw_logit": 3.8499915599823,
        "decoded_rgb_sha256": "b17c899e7f66a8e9cb383be47889ec153404bb449db25466c26be6b47ea281f1",
        "tensor_sha256": "223319e2821751263cc8819cd42f30825cf86b49f75a882be4ed4fbe1ebc7893",
        "tensor_shape": [3, 1024, 1024],
        "patch_grid_wh": [73, 73],
        "crop_starts_patch_xy": [
            [18, 18],
            [0, 0],
            [0, 37],
            [37, 37],
            [37, 0],
        ],
    },
)
GOLDEN_ABS_TOLERANCE = 5e-5
GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE = 1e-6

LICENSE_RECORD = {
    "code_and_weights": {
        "license": "GRIP_UNINA_nonprofit_research_only",
        "license_file_sha256": SOURCE_FILES["LICENSE.txt"],
        "commercial_use": False,
    },
    "benchmark_role": "research_evaluation_only",
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

DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/b-free-c6a9f898"
)
DEFAULT_WEIGHTS_DIR = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4"
)
DEFAULT_WEIGHTS_ZIP = Path(
    "/root/.cache/claimforge/third_party/BFREE_dino2reg4.zip"
)
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/bfree")
DEFAULT_RUN_ID = (
    "bfree_dino2reg4_mouse_canonical_v1_full275_20260725"
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


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_save_artifact(
    path: Path,
    features: np.ndarray,
    crop_logits: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                features=np.ascontiguousarray(features, dtype=np.float32),
                crop_logits=np.ascontiguousarray(
                    crop_logits,
                    dtype=np.float32,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"features", "crop_logits"}:
                raise ValueError("B-Free artifact keys changed")
            features = np.ascontiguousarray(payload["features"])
            crop_logits = np.ascontiguousarray(payload["crop_logits"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot safely load B-Free artifact: {path}") from exc
    if (
        features.shape != (CROP_COUNT, FEATURE_DIMENSION)
        or features.dtype != np.float32
        or crop_logits.shape != (CROP_COUNT,)
        or crop_logits.dtype != np.float32
        or not np.isfinite(features).all()
        or not np.isfinite(crop_logits).all()
    ):
        raise ValueError("B-Free artifact arrays violate frozen contract")
    return features, crop_logits


def _artifact_path(run_dir: Path, sample_id: str) -> Path:
    _safe_component(sample_id, label="sample-id")
    artifact_dir = (run_dir / "artifacts").resolve()
    path = (artifact_dir / f"{sample_id}.npz").resolve()
    if path.parent != artifact_dir:
        raise ValueError("artifact path escapes artifact directory")
    return path


def adapter_contract(repo_root: Path) -> dict[str, Any]:
    relatives = (
        "eval/opensource/run_bfree.py",
        "eval/opensource/bfree_metrics.py",
        "eval/opensource/whole_image_metrics.py",
        "eval/opensource/common.py",
    )
    result: dict[str, Any] = {}
    for relative in relatives:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing B-Free adapter file: {path}")
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
    if not isinstance(inputs_value, str):
        raise ValueError("canonical release has no inputs_path")
    inputs_path = _anchored(Path(inputs_value), repo_root)
    _verify_runtime_file(
        inputs_path,
        str(release["inputs_sha256"]),
        "canonical inputs.jsonl",
    )
    rows = read_jsonl(inputs_path)
    if len(rows) != int(release["images"]):
        raise ValueError("canonical input count changed")
    ranks = [int(row["rank"]) for row in rows]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("canonical ranks are not unique and sorted")
    ids = [
        _safe_component(row.get("sample_id"), label="canonical sample_id")
        for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical sample IDs are not unique")
    pairs_value = release.get("pairs_path")
    if not isinstance(pairs_value, str):
        raise ValueError("canonical release has no pairs_path")
    pairs_path = _anchored(Path(pairs_value), repo_root)
    _verify_runtime_file(
        pairs_path,
        str(release["pairs_sha256"]),
        "canonical pairs.jsonl",
    )
    if len(read_jsonl(pairs_path)) != int(release["pairs"]):
        raise ValueError("canonical pair count changed")
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
            raise ValueError("sample-id must select exactly one input")
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
    by_pair: dict[int, list[str]] = {}
    by_task: dict[int, set[str]] = {}
    for row in selected:
        rank = int(row["pair_rank"])
        by_pair.setdefault(rank, []).append(str(row["kind"]))
        by_task.setdefault(rank, set()).add(str(row["task_id"]))
    if any(sorted(kinds) != ["forged", "real"] for kinds in by_pair.values()):
        raise ValueError("canonical pair selection is incomplete")
    if any(len(tasks) != 1 for tasks in by_task.values()):
        raise ValueError("canonical pair selection has mismatched task IDs")
    return selected


def _load_gt_mask(
    row: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    width, height = int(row["width"]), int(row["height"])
    sample_id = str(row["sample_id"])
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
    path_value, digest = row.get("gt_mask_path"), row.get("gt_mask_sha256")
    if not isinstance(path_value, str) or not _valid_sha256(digest):
        raise ValueError(f"invalid forged GT metadata for {sample_id}")
    path = _anchored(Path(path_value), repo_root)
    _verify_runtime_file(path, str(digest), f"GT mask {sample_id}")
    with Image.open(path) as opened:
        pixels = np.asarray(opened)
    if pixels.shape != (height, width) or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"invalid GT pixels for {sample_id}")
    positive = int(np.count_nonzero(pixels == 255))
    if positive <= 0 or positive != int(row.get("gt_positive_pixels", -1)):
        raise ValueError(f"GT positive count changed for {sample_id}")
    return np.ascontiguousarray(pixels, dtype=np.uint8)


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
    """Reproduce the official stride-14 grid, wrap, and five-crop geometry."""

    if width < PATCH_STRIDE or height < PATCH_STRIDE:
        raise ValueError("B-Free requires at least one 14x14 input patch")
    grid_width, grid_height = width // PATCH_STRIDE, height // PATCH_STRIDE
    wrapped = grid_width < CROP_PATCHES or grid_height < CROP_PATCHES
    if wrapped:
        starts = [[0, 0] for _ in range(CROP_COUNT)]
        used_width = min(grid_width, CROP_PATCHES) * PATCH_STRIDE
        used_height = min(grid_height, CROP_PATCHES) * PATCH_STRIDE
        rectangles = [
            [0, 0, used_width, used_height] for _ in range(CROP_COUNT)
        ]
        post_grid = [CROP_PATCHES, CROP_PATCHES]
    else:
        center_x = (grid_width - CROP_PATCHES) // 2
        center_y = (grid_height - CROP_PATCHES) // 2
        max_x = grid_width - CROP_PATCHES
        max_y = grid_height - CROP_PATCHES
        starts = [
            [center_x, center_y],
            [0, 0],
            [0, max_y],
            [max_x, max_y],
            [max_x, 0],
        ]
        rectangles = [
            [
                x * PATCH_STRIDE,
                y * PATCH_STRIDE,
                (x + CROP_PATCHES) * PATCH_STRIDE,
                (y + CROP_PATCHES) * PATCH_STRIDE,
            ]
            for x, y in starts
        ]
        post_grid = [grid_width, grid_height]
    return {
        "profile_id": PREPROCESS_PROFILE,
        "decoder": "Pillow.Image.open.convert_RGB",
        "native_size": [width, height],
        "resize": {"enabled": False},
        "to_tensor": "torchvision.transforms.ToTensor_uint8_div_255_float32",
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "patch_projection": {
            "kernel_size": [PATCH_STRIDE, PATCH_STRIDE],
            "stride": [PATCH_STRIDE, PATCH_STRIDE],
            "right_bottom_remainders_dropped": [
                width % PATCH_STRIDE,
                height % PATCH_STRIDE,
            ],
        },
        "patch_grid_wh": [grid_width, grid_height],
        "replicate_wrap_applied": wrapped,
        "replicate_wrap_trigger": "either_patch_grid_dimension_below_36",
        "replicate_wrap_semantics": (
            "repeat_both_grid_dimensions_then_truncate_both_to_36"
            if wrapped
            else "not_applicable"
        ),
        "post_wrap_patch_grid_wh": post_grid,
        "crop_size_pixels": [CROP_SIZE, CROP_SIZE],
        "crop_size_patches": [CROP_PATCHES, CROP_PATCHES],
        "crop_order": ["center", "top_left", "bottom_left", "bottom_right", "top_right"],
        "crop_starts_patch_xy": starts,
        "distinct_crop_starts": len({tuple(value) for value in starts}),
        "used_native_rectangles_xyxy": rectangles,
        "used_native_pixel_rule": (
            "union_of_integer_half_open_patch_receptive_field_rectangles"
        ),
    }


def _gt_visibility(
    forged_row: Mapping[str, Any],
    gt: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    y, x = np.nonzero(gt == 255)
    total = int(x.size)
    visible_mask = np.zeros(total, dtype=bool)
    for left, top, right, bottom in geometry[
        "used_native_rectangles_xyxy"
    ]:
        visible_mask |= (
            (x >= int(left))
            & (x < int(right))
            & (y >= int(top))
            & (y < int(bottom))
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
            "exact_diff_positive_pixels_intersecting_union_of_official_"
            "five_patch_crop_receptive_fields"
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
            raise ValueError(f"duplicate canonical {kind}: {task_id}")
        pairs[task_id][kind] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in pairs.items():
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"incomplete canonical task: {task_id}")
        real, forged = pair["real"], pair["forged"]
        if (
            int(real["width"]) != int(forged["width"])
            or int(real["height"]) != int(forged["height"])
            or real.get("domain") != forged.get("domain")
        ):
            raise ValueError(f"canonical pair geometry changed: {task_id}")
        gt = _load_gt_mask(forged, repo_root)
        assert gt is not None
        geometry = compute_preprocess_geometry(
            int(forged["width"]),
            int(forged["height"]),
        )
        evidence = _gt_visibility(forged, gt, geometry)
        result[task_id] = {
            "edit_visibility": evidence["category"],
            "edit_visible_gt_fraction": evidence["visible_fraction"],
            "edit_visibility_evidence": evidence,
            "replicate_wrap_applied": geometry["replicate_wrap_applied"],
            "distinct_crop_starts": geometry["distinct_crop_starts"],
            "domain": str(forged["domain"]),
        }
    return result


def validate_frozen_visibility_census(
    visibility: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    census = Counter(
        str(value["edit_visibility"]) for value in visibility.values()
    )
    expected = dict(FROZEN_VISIBILITY_CENSUS["edit_visibility"])
    if len(visibility) != 275 or dict(census) != expected:
        raise ValueError(
            f"B-Free frozen visibility changed: {len(visibility)}, {dict(census)}"
        )
    fractions = [
        float(value["edit_visible_gt_fraction"])
        for value in visibility.values()
    ]
    mean_fraction = float(np.mean(fractions))
    if not math.isclose(
        mean_fraction,
        float(FROZEN_VISIBILITY_CENSUS["mean_edit_visible_gt_fraction"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"B-Free mean visibility changed: {mean_fraction}")
    wrapped = sum(
        bool(value["replicate_wrap_applied"]) for value in visibility.values()
    )
    distinct = Counter(
        int(value["distinct_crop_starts"]) for value in visibility.values()
    )
    if (
        wrapped != int(FROZEN_VISIBILITY_CENSUS["wrap_pairs"])
        or dict(distinct) != FROZEN_VISIBILITY_CENSUS[
            "distinct_crop_starts"
        ]
    ):
        raise ValueError("B-Free crop-mode census changed")
    domain: dict[str, dict[str, int]] = {}
    for value in visibility.values():
        counts = domain.setdefault(
            str(value["domain"]),
            {"full": 0, "partial": 0, "none": 0},
        )
        counts[str(value["edit_visibility"])] += 1
    expected_domain = FROZEN_VISIBILITY_CENSUS["by_domain"]
    if domain != expected_domain:
        raise ValueError(f"B-Free domain visibility changed: {domain}")
    wrapped_visibility = Counter(
        str(value["edit_visibility"])
        for value in visibility.values()
        if bool(value["replicate_wrap_applied"])
    )
    if (
        dict(wrapped_visibility)
        != FROZEN_VISIBILITY_CENSUS["wrap_edit_visibility"]
    ):
        raise ValueError("B-Free wrapped visibility census changed")
    return {
        "pairs": len(visibility),
        "census": expected,
        "mean_edit_visible_gt_fraction": mean_fraction,
        "wrapped_pairs": wrapped,
        "wrapped_visibility_census": dict(wrapped_visibility),
        "distinct_crop_starts_census": dict(distinct),
        "domain_census": domain,
        "basis": (
            "frozen_before_Mouse_scores_exact_diff_pixels_intersecting_"
            "official_five_crop_receptive_field_union"
        ),
    }


def _verify_source(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing B-Free source: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(f"B-Free source commit changed: {commit}")
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"B-Free tracked source is dirty: {dirty[:1000]}")
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"B-Free source {relative}",
        )
    main = (source_root / "code/main_bfree.py").read_text(encoding="utf-8")
    wrapper = (
        source_root / "code/networks/wrapper5crops.py"
    ).read_text(encoding="utf-8")
    normalization = (
        source_root / "code/utils/normalization.py"
    ).read_text(encoding="utf-8")
    required = {
        "main_bfree.py": (
            "Image.open(filename).convert('RGB')",
            "out_tens[:, 0]",
            "score>0",
        ),
        "wrapper5crops.py": (
            "self.model.set_input_size(img_size=patch_size)",
            "embeddings = self.patch_embed.proj(x)",
            "embeddings = replicate_wrap(embeddings, patch_size)",
            "torch.mean(torch.stack(torch.split(y, y.shape[0]//5, 0), 0), 0)",
        ),
        "normalization.py": (
            "transforms.ToTensor()",
            "mean=[0.485, 0.456, 0.406]",
            "std=[0.229, 0.224, 0.225]",
        ),
    }
    texts = {
        "main_bfree.py": main,
        "wrapper5crops.py": wrapper,
        "normalization.py": normalization,
    }
    missing = [
        f"{name}:{needle}"
        for name, needles in required.items()
        for needle in needles
        if needle not in texts[name]
    ]
    if missing:
        raise ValueError(f"B-Free source evidence changed: {missing}")
    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "files": {
            relative: {
                "path": str((source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in SOURCE_FILES.items()
        },
        "license": LICENSE_RECORD,
    }


def _checkpoint_state(
    checkpoint_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    import torch

    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(
        checkpoint_path
    )
    if unsafe != []:
        raise ValueError(f"B-Free checkpoint has unsafe globals: {unsafe}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or list(payload) != ["model"]:
        raise ValueError("B-Free checkpoint top-level schema changed")
    state = payload["model"]
    if not isinstance(state, collections.OrderedDict):
        raise ValueError("B-Free model state is not an OrderedDict")
    items: list[dict[str, Any]] = []
    for key, value in state.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"B-Free state tensor changed: {key!r}")
        items.append(
            {
                "k": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "n": int(value.numel()),
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            items,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        len(items) != int(CHECKPOINT["tensor_count"])
        or sum(item["n"] for item in items)
        != int(CHECKPOINT["state_elements"])
        or digest != CHECKPOINT["schema_sha256"]
    ):
        raise ValueError("B-Free checkpoint tensor schema changed")
    return state, {
        "top_level_keys": ["model"],
        "state_container": "collections.OrderedDict",
        "tensor_count": len(items),
        "state_elements": sum(item["n"] for item in items),
        "all_dtype": "torch.float32",
        "all_finite": True,
        "schema_sha256": digest,
        "keys": list(state),
        "items": items,
    }


def verify_assets(
    *,
    source_root: Path,
    weights_dir: Path,
    weights_zip: Path,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    source = _verify_source(source_root)
    _verify_runtime_file(
        weights_zip,
        str(OFFICIAL_ZIP["sha256"]),
        "official B-Free weights ZIP",
    )
    if weights_zip.stat().st_size != int(OFFICIAL_ZIP["bytes"]):
        raise ValueError("official B-Free ZIP byte size changed")
    md5 = _md5_file(weights_zip)
    if md5 != OFFICIAL_ZIP["md5"]:
        raise ValueError("official B-Free ZIP MD5 changed")
    with zipfile.ZipFile(weights_zip) as archive:
        members = {item.filename: item.file_size for item in archive.infolist()}
        if members != OFFICIAL_ZIP["members"]:
            raise ValueError(f"official B-Free ZIP members changed: {members}")
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in members
        ):
            raise ValueError("official B-Free ZIP has unsafe member paths")
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"missing B-Free weights dir: {weights_dir}")
    config_path = weights_dir / str(CONFIG["filename"])
    checkpoint_path = weights_dir / str(CHECKPOINT["filename"])
    _verify_runtime_file(config_path, str(CONFIG["sha256"]), "B-Free config")
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "B-Free checkpoint",
    )
    if config_path.stat().st_size != int(CONFIG["bytes"]):
        raise ValueError("B-Free config byte size changed")
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("B-Free checkpoint byte size changed")
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config != CONFIG["parsed"]:
        raise ValueError(f"B-Free parsed config changed: {config}")
    state, schema = _checkpoint_state(checkpoint_path)
    return source, {
        "zip": {
            **OFFICIAL_ZIP,
            "path": str(weights_zip.resolve()),
            "verified_sha256": sha256_file(weights_zip),
            "verified_md5": md5,
        },
        "config": {
            **CONFIG,
            "path": str(config_path.resolve()),
            "parsed_actual": config,
        },
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path.resolve()),
            "schema": schema,
            "safe_weights_only_load": True,
            "unsafe_globals": [],
        },
    }, state


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    device = torch.device(device_text)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("B-Free supports only cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    timm_version = _package_version("timm")
    transformers_version = _package_version("transformers")
    if timm_version != TIMM_VERSION:
        raise ValueError(f"timm version changed: {timm_version}")
    if transformers_version != TRANSFORMERS_VERSION:
        raise ValueError(
            f"transformers version changed: {transformers_version}"
        )
    return device, {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": timm_version,
        "transformers": transformers_version,
        "pyyaml": _package_version("PyYAML"),
        "scikit_learn": _package_version("scikit-learn"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "seed": MODEL_SEED,
        "dtype": "float32",
        "autocast": False,
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
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "network_allowed": False,
    }


def _official_network_module(source_root: Path) -> Any:
    package_name = "_claimforge_bfree_official_networks"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    init_path = source_root / "code/networks/__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned B-Free network package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_model(
    *,
    state: Mapping[str, Any],
    source_root: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import timm

    if timm.__version__ != TIMM_VERSION:
        raise ValueError("B-Free timm runtime changed")
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

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
        )
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
        official = _official_network_module(source_root)
        model = official.get_network(str(CONFIG["parsed"]["arch"]))
    if any(attempts.values()):
        raise RuntimeError(f"B-Free construction attempted network: {attempts}")
    if (
        type(model).__name__ != "Wrapper5crops"
        or tuple(model.patch_embed.grid_size) != (CROP_PATCHES, CROP_PATCHES)
        or tuple(model.patch_embed.patch_size) != (PATCH_STRIDE, PATCH_STRIDE)
        or model.model.num_features != FEATURE_DIMENSION
        or model.model.num_reg_tokens != 4
        or model.model.num_prefix_tokens != 5
        or model.model.global_pool != "token"
        or not isinstance(model.model.head, torch.nn.Linear)
        or (model.model.head.in_features, model.model.head.out_features)
        != (FEATURE_DIMENSION, 1)
    ):
        raise ValueError("B-Free official architecture changed")
    expected_keys = list(model.patch_embed.state_dict())
    expected_keys = [f"patch_embed.{key}" for key in expected_keys]
    expected_keys += [
        key
        for key in model.model.state_dict()
        if not key.startswith("patch_embed.")
    ]
    if set(expected_keys) != set(state):
        raise ValueError("B-Free model/checkpoint state keys differ")
    model.load_state_dict(state)
    loaded: dict[str, Any] = {
        f"patch_embed.{key}": value
        for key, value in model.patch_embed.state_dict().items()
    }
    loaded.update(
        {
            key: value
            for key, value in model.model.state_dict().items()
            if not key.startswith("patch_embed.")
        }
    )
    if set(loaded) != set(state) or any(
        not torch.equal(loaded[key].detach().cpu(), state[key])
        for key in state
    ):
        raise ValueError("B-Free strict load did not preserve exact tensors")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fused = [
        bool(getattr(block.attn, "fused_attn", True))
        for block in model.model.blocks
    ]
    return model, {
        "construction": {
            "official_source_module": "code/networks",
            "architecture": str(CONFIG["parsed"]["arch"]),
            "timm_architecture": MODEL_ARCH,
            "pretrained": False,
            "wrapper": "Wrapper5crops",
            "patch_size": PATCH_STRIDE,
            "crop_size": CROP_SIZE,
            "crop_count": CROP_COUNT,
            "feature_dimension": FEATURE_DIMENSION,
            "register_tokens": 4,
            "global_pool": "token",
            "fused_attention": sorted(set(fused)),
        },
        "load": {
            "strict_full_state_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "loaded_tensor_count": len(state),
            "loaded_state_elements": sum(
                int(value.numel()) for value in state.values()
            ),
        },
        "network": {
            "allowed": False,
            "attempts": attempts,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "model_mode": "eval",
        "requires_grad": False,
    }


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from torchvision import transforms

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        tensor = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)(
            transforms.ToTensor()(rgb)
        )
    array = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if (
        decoded.shape != (height, width, 3)
        or array.shape != (3, height, width)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise ValueError("B-Free preprocessing contract changed")
    geometry = compute_preprocess_geometry(width, height)
    return array, {
        "profile": PREPROCESS_PROFILE,
        "geometry": geometry,
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "tensor_shape": list(array.shape),
        "tensor_dtype": str(array.dtype),
        "tensor_sha256": _array_sha256(array),
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "resize_applied": False,
        "crop_applied_before_patch_projection": False,
    }


def replay_head(
    official_mean: Any,
    features: Any,
    head: Any,
    *,
    official_crop_logits: Any | None = None,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (CROP_COUNT, FEATURE_DIMENSION)
        or features.dtype != torch.float32
    ):
        raise ValueError("B-Free pre-head features must be [5,768] float32")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("B-Free pre-head features must be finite")
    if (
        not isinstance(official_mean, torch.Tensor)
        or tuple(official_mean.shape) not in ((1,), (1, 1))
        or official_mean.dtype != torch.float32
        or not bool(torch.isfinite(official_mean).all())
    ):
        raise ValueError("B-Free official mean must be finite float32 scalar")
    with torch.inference_mode():
        manual_crop = functional.linear(
            features,
            head.weight,
            head.bias,
        )
        manual_mean = manual_crop.mean(dim=0)
    if official_crop_logits is not None and not torch.equal(
        manual_crop,
        official_crop_logits,
    ):
        raise ValueError("B-Free crop-logit replay differs from official head")
    if not torch.equal(manual_mean.reshape(-1), official_mean.reshape(-1)):
        raise ValueError("B-Free mean-logit replay differs from official output")
    crop = np.ascontiguousarray(
        manual_crop.detach().cpu().reshape(CROP_COUNT).numpy(),
        dtype=np.float32,
    )
    raw_logit = float(official_mean.detach().reshape(()).item())
    if not math.isfinite(raw_logit):
        raise ValueError("B-Free raw logit is not finite")
    probability = bfree_fake_probability_float32(raw_logit)
    decision = raw_logit > CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw_logit,
        "ai_score": raw_logit,
        "fake_probability": probability,
        "decision": decision,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": SCORE_SEMANTICS,
    }
    return {
        "raw_logit": raw_logit,
        "ai_score": raw_logit,
        "score": raw_logit,
        "fake_probability": probability,
        "crop_logits": [float(value) for value in crop.tolist()],
        "score_semantics": SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": {
            **{
                key: value
                for key, value in classification.items()
                if key != "semantics"
            },
            "policy": T1_POLICY,
        },
        "manual_replay": {
            "crop_logits": [float(value) for value in crop.tolist()],
            "raw_logit": float(manual_mean.reshape(()).item()),
            "ai_score": float(manual_mean.reshape(()).item()),
            "official_crop_logits_exact_match": (
                official_crop_logits is not None
            ),
            "official_mean_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int | None, float]:
    import torch

    if (
        image.ndim != 3
        or image.shape[0] != 3
        or image.dtype != np.float32
        or not image.flags.c_contiguous
        or image.shape[1] < PATCH_STRIDE
        or image.shape[2] < PATCH_STRIDE
    ):
        raise ValueError("B-Free input tensor violates native float32 contract")
    value = torch.from_numpy(image).unsqueeze(0).to(device)
    captured_features: list[Any] = []
    captured_crop_logits: list[Any] = []

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("B-Free head received unexpected inputs")
        captured_features.append(inputs[0].detach().clone())

    def post_hook(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        captured_crop_logits.append(output.detach().clone())

    pre = model.model.head.register_forward_pre_hook(pre_hook)
    post = model.model.head.register_forward_hook(post_hook)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            official_mean = model(value)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    finally:
        pre.remove()
        post.remove()
    if len(captured_features) != 1 or len(captured_crop_logits) != 1:
        raise RuntimeError("B-Free classifier hooks did not fire once")
    scoring = replay_head(
        official_mean,
        captured_features[0],
        model.model.head,
        official_crop_logits=captured_crop_logits[0],
    )
    features = np.ascontiguousarray(
        captured_features[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    crop_logits = np.ascontiguousarray(
        captured_crop_logits[0].detach().cpu().reshape(CROP_COUNT).numpy(),
        dtype=np.float32,
    )
    if (
        features.shape != (CROP_COUNT, FEATURE_DIMENSION)
        or crop_logits.shape != (CROP_COUNT,)
        or not np.isfinite(features).all()
        or not np.isfinite(crop_logits).all()
    ):
        raise ValueError("B-Free captured artifact violates contract")
    return scoring, features, crop_logits, peak, latency_ms


def validate_official_golden(
    *,
    model: Any,
    device: Any,
    source_root: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for frozen in GOLDEN_CASES:
        path = source_root / "code/demo_images" / str(frozen["filename"])
        _verify_runtime_file(
            path,
            str(frozen["sha256"]),
            f"B-Free golden {frozen['filename']}",
        )
        image, preprocess = preprocess_image(path)
        for key in (
            "decoded_rgb_sha256",
            "tensor_sha256",
            "tensor_shape",
        ):
            if preprocess[key] != frozen[key]:
                raise ValueError(
                    f"B-Free golden preprocess {key} changed for {path.name}"
                )
        geometry = preprocess["geometry"]
        for key in ("patch_grid_wh", "crop_starts_patch_xy"):
            if geometry[key] != frozen[key]:
                raise ValueError(
                    f"B-Free golden geometry {key} changed for {path.name}"
                )
        first = infer_one(model, device, image)
        second = infer_one(model, device, image)
        scoring, features, crop_logits, _, _ = first
        scoring2, features2, crop_logits2, _, _ = second
        if (
            scoring["raw_logit"] != scoring2["raw_logit"]
            or not np.array_equal(features, features2)
            or not np.array_equal(crop_logits, crop_logits2)
        ):
            raise ValueError(f"B-Free golden is not repeatable: {path.name}")
        actual = float(scoring["raw_logit"])
        published = float(frozen["published_raw_logit"])
        difference = abs(actual - published)
        runtime_reference_key = (
            "frozen_cuda_raw_logit"
            if device.type == "cuda"
            else "frozen_cpu_raw_logit"
        )
        runtime_reference = float(frozen[runtime_reference_key])
        runtime_difference = abs(actual - runtime_reference)
        case = {
            **frozen,
            "path": str(path.resolve()),
            "actual_raw_logit": actual,
            "absolute_difference_from_published": difference,
            "runtime_reference_kind": runtime_reference_key,
            "runtime_reference_raw_logit": runtime_reference,
            "absolute_difference_from_runtime_reference": (
                runtime_difference
            ),
            "crop_logits": scoring["crop_logits"],
            "feature_array_sha256": _array_sha256(features),
            "crop_logits_array_sha256": _array_sha256(crop_logits),
            "repeat_bit_identical": True,
            "passed": (
                difference <= GOLDEN_ABS_TOLERANCE
                and runtime_difference
                <= GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE
            ),
        }
        cases.append(case)
        if difference > GOLDEN_ABS_TOLERANCE:
            raise ValueError(
                f"B-Free golden mismatch {path.name}: {actual} != {published}"
            )
        if runtime_difference > GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE:
            raise ValueError(
                "B-Free runtime regression mismatch "
                f"{path.name}: {actual} != {runtime_reference}"
            )
    return {
        "status": "passed",
        "source": "official_code_demo_images_results.csv",
        "score": SCORE_SEMANTICS,
        "absolute_tolerance": GOLDEN_ABS_TOLERANCE,
        "runtime_regression_absolute_tolerance": (
            GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE
        ),
        "cases": cases,
        "mouse_model_scores_computed": 0,
    }


def _result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    visibility: Mapping[str, Any],
    config_fingerprint: str,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    path = _anchored(Path(str(row["canonical_path"])), repo_root)
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": str(row["task_id"]),
        "pair_rank": int(row["pair_rank"]),
        "rank": int(row["rank"]),
        "kind": str(row["kind"]),
        "label": int(row["label"]),
        "domain": str(row["domain"]),
        "input_path": repo_relative(path, repo_root),
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
        "valid_for_t1": True,
        "valid_for_t2": False,
    }


_FORBIDDEN_T2_KEYS = {
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


def _reject_t2_payload(value: Any, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        present = sorted(_FORBIDDEN_T2_KEYS.intersection(value))
        if present:
            raise ValueError(f"{path} invents B-Free T2 fields: {present}")
        for key, child in value.items():
            _reject_t2_payload(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_t2_payload(child, path=f"{path}[{index}]")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not real")
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
    head: Any,
    device: Any,
) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume identity field {key} changed")
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("only valid successful rows may be resume-skipped")
    raw = _finite_number(row.get("raw_logit"), "resume raw_logit")
    if row.get("ai_score") != raw or row.get("score") != raw:
        raise ValueError("resume raw-logit aliases changed")
    crop_values = row.get("crop_logits")
    if (
        not isinstance(crop_values, list)
        or len(crop_values) != CROP_COUNT
        or any(not math.isfinite(float(value)) for value in crop_values)
    ):
        raise ValueError("resume crop logits changed")
    probability = bfree_fake_probability_float32(raw)
    if row.get("fake_probability") != probability:
        raise ValueError("resume diagnostic probability changed")
    if row.get("classification_decision") is not (raw > 0.0):
        raise ValueError("resume strict decision changed")
    preprocess = row.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ValueError("resume preprocessing audit missing")
    input_path = _anchored(Path(str(expected["input_path"])), repo_root)
    _verify_runtime_file(
        input_path,
        str(expected["input_sha256"]),
        "resume B-Free input",
    )
    _, replay_preprocess = preprocess_image(input_path)
    if dict(preprocess) != replay_preprocess:
        raise ValueError("resume B-Free preprocessing does not replay")
    artifact_value = row.get("artifact_path")
    if not isinstance(artifact_value, str):
        raise ValueError("resume B-Free artifact path missing")
    artifact_path = _anchored(Path(artifact_value), repo_root)
    expected_path = _artifact_path(run_dir, str(row["id"]))
    if artifact_path != expected_path:
        raise ValueError("resume B-Free artifact path changed or escapes")
    _verify_runtime_file(
        artifact_path,
        str(row.get("artifact_sha256")),
        "resume B-Free artifact",
    )
    features, crop_logits = _load_artifact(artifact_path)
    if (
        _array_sha256(features) != row.get("feature_array_sha256")
        or _array_sha256(crop_logits)
        != row.get("crop_logits_array_sha256")
        or [float(value) for value in crop_logits.tolist()] != crop_values
    ):
        raise ValueError("resume B-Free artifact content changed")
    import torch
    from torch.nn import functional

    with torch.inference_mode():
        feature_tensor = torch.from_numpy(features).to(device)
        replay_crop = functional.linear(
            feature_tensor,
            head.weight,
            head.bias,
        ).reshape(CROP_COUNT)
        replay_mean = replay_crop.mean()
    if (
        not torch.equal(replay_crop.detach().cpu(), torch.from_numpy(crop_logits))
        or float(replay_mean.item()) != raw
    ):
        raise ValueError("resume B-Free head replay changed")
    _reject_t2_payload(row)


def _run_config(
    *,
    adapter: Mapping[str, Any],
    runtime: Mapping[str, Any],
    release: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    golden: Mapping[str, Any],
    visibility_audit: Mapping[str, Any],
    device_text: str,
    pair_limit: int | None,
    sample_id: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected_bytes = "".join(
        f"{stable_json(row)}\n" for row in selected
    ).encode("utf-8")
    return {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "model_arch": MODEL_ARCH,
        "source_commit": source["commit"],
        "source_files": SOURCE_FILES,
        "adapter_contract": dict(adapter),
        "official_zip_sha256": assets["zip"]["sha256"],
        "config_sha256": assets["config"]["sha256"],
        "checkpoint_sha256": assets["checkpoint"]["sha256"],
        "checkpoint_schema_sha256": assets["checkpoint"]["schema_sha256"],
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "resize": False,
            "to_tensor": "torchvision_ToTensor",
            "normalization_mean": list(IMAGE_MEAN),
            "normalization_std": list(IMAGE_STD),
            "patch_projection_stride": PATCH_STRIDE,
            "wrap": (
                "if either grid dimension below36 repeat both then truncate"
            ),
            "five_crop_size": CROP_SIZE,
            "five_crop_order": [
                "center",
                "top_left",
                "bottom_left",
                "bottom_right",
                "top_right",
            ],
            "batch_size": 1,
        },
        "model_contract": {
            "official_wrapper": True,
            "strict_full_checkpoint_load": True,
            "feature_shape": [CROP_COUNT, FEATURE_DIMENSION],
            "feature_semantics": FEATURE_SEMANTICS,
            "crop_logits_shape": [CROP_COUNT],
            "primary_score": SCORE_SEMANTICS,
            "score_direction": "higher_means_fake",
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "valid_for_t1": True,
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
            "network_allowed": False,
        },
        "runtime_evidence": dict(runtime),
        "model_audit": dict(model_audit),
        "official_golden": dict(golden),
        "dataset": {
            "schema_version": release["schema_version"],
            "dataset_id": release["dataset_id"],
            "inputs_sha256": release["inputs_sha256"],
            "selected_ids": [str(row["sample_id"]) for row in selected],
            "selected_rows_sha256": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "pair_limit": pair_limit,
            "sample_id": sample_id,
        },
        "metrics": {
            "primary": "raw_logit",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "official_calibration": ["balanced_nll", "balanced_ece_15_bins"],
        },
        "license": LICENSE_RECORD,
        "frozen_full_dataset_visibility": dict(visibility_audit),
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
    )
    parser.add_argument(
        "--weights-zip",
        type=Path,
        default=DEFAULT_WEIGHTS_ZIP,
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
    parser.add_argument("--retry-errors", action="store_true")
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
    weights_dir = _anchored(args.weights_dir, repo_root)
    weights_zip = _anchored(args.weights_zip, repo_root)
    dataset_manifest = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    _safe_component(args.run_id, label="run-id")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.retry_errors and not args.resume:
        raise ValueError("--retry-errors requires --resume")

    release, inputs_path, all_rows = load_release(repo_root, dataset_manifest)
    selected = select_inputs(all_rows, args.pair_limit, args.sample_id)
    validate_selected_inputs(selected, repo_root)
    visibility = build_pair_visibility(all_rows, repo_root)
    visibility_audit = validate_frozen_visibility_census(visibility)
    source, assets, state = verify_assets(
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
    )
    device, runtime = configure_runtime(args.device)
    model, model_audit = load_model(
        state=state,
        source_root=source_root,
        device=device,
    )
    del state
    golden = validate_official_golden(
        model=model,
        device=device,
        source_root=source_root,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model": MODEL_NAME,
                    "source_commit": source["commit"],
                    "checkpoint_sha256": assets["checkpoint"]["sha256"],
                    "official_golden": golden,
                    "visibility_audit": visibility_audit,
                    "mouse_model_scores_computed": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root.resolve():
        raise ValueError("run-id escapes results directory")
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    artifact_dir = run_dir / "artifacts"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _run_config(
        adapter=adapter_contract(repo_root),
        runtime=runtime,
        release=release,
        selected=selected,
        source=source,
        assets=assets,
        model_audit=model_audit,
        golden=golden,
        visibility_audit=visibility_audit,
        device_text=str(device),
        pair_limit=args.pair_limit,
        sample_id=args.sample_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    # Compare resume contracts in their persisted JSON representation.  This
    # also canonicalizes the integer keys in crop-start census dictionaries.
    config = json.loads(stable_json(config))
    fingerprint = _manifest_fingerprint(config)
    prior_manifest: dict[str, Any] | None = None
    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires run_manifest and expected_inputs"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            prior_manifest.get("schema_version")
            != "bfree_detection_run_manifest_v1"
            or prior_manifest.get("run_id") != args.run_id
            or prior_manifest.get("config") != config
            or prior_manifest.get("config_fingerprint") != fingerprint
            or read_jsonl(expected_path) != selected
        ):
            raise ValueError("B-Free resume identity/config changed")
        prior_outputs = prior_manifest.get("outputs")
        if isinstance(prior_outputs, Mapping):
            for key, path in (
                ("results_sha256", results_path),
                ("summary_sha256", summary_path),
            ):
                digest = prior_outputs.get(key)
                if digest is not None:
                    _verify_runtime_file(path, str(digest), f"prior {key}")
    else:
        atomic_write_jsonl(expected_path, selected)
    selected_tasks = sorted({str(row["task_id"]) for row in selected})
    manifest: dict[str, Any] = {
        "schema_version": "bfree_detection_run_manifest_v1",
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
        "config_fingerprint": fingerprint,
        "config": config,
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "official_golden": golden,
        "runtime": runtime,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "primary_score": "ai_score",
            "primary_score_semantics": SCORE_SEMANTICS,
        },
        "dataset": {
            "manifest_path": repo_relative(dataset_manifest, repo_root),
            "manifest_sha256": sha256_file(dataset_manifest),
            "inputs_path": repo_relative(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
            "selected_tasks": len(selected_tasks),
        },
        "visibility_census": dict(
            Counter(
                visibility[task]["edit_visibility"]
                for task in selected_tasks
            )
        ),
        "full_dataset_visibility_audit": visibility_audit,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_dir": repo_relative(artifact_dir, repo_root),
        },
    }
    atomic_write_json(manifest_path, manifest)

    latest = read_latest_by_id(results_path)
    completed = skipped = errors = prior_errors_skipped = 0
    for index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        identity = _result_identity(
            row,
            repo_root=repo_root,
            visibility=visibility[str(row["task_id"])],
            config_fingerprint=fingerprint,
        )
        prior = latest.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
            _validate_resume_row(
                prior,
                expected=identity,
                repo_root=repo_root,
                run_dir=run_dir,
                head=model.model.head,
                device=device,
            )
            skipped += 1
            print(f"[{index}/{len(selected)}] resume {sample_id}", flush=True)
            continue
        if (
            prior is not None
            and prior.get("status") == "error"
            and not args.retry_errors
        ):
            prior_errors_skipped += 1
            print(
                f"[{index}/{len(selected)}] retain-error {sample_id}",
                flush=True,
            )
            continue
        input_path = _anchored(Path(str(row["canonical_path"])), repo_root)
        try:
            preprocess_started = time.perf_counter()
            image, preprocess = preprocess_image(input_path)
            preprocess_ms = (
                time.perf_counter() - preprocess_started
            ) * 1000.0
            scoring, features, crop_logits, peak, latency = infer_one(
                model,
                device,
                image,
            )
            artifact_path = _artifact_path(run_dir, sample_id)
            _atomic_save_artifact(artifact_path, features, crop_logits)
            persisted_features, persisted_logits = _load_artifact(
                artifact_path
            )
            if not np.array_equal(features, persisted_features) or not np.array_equal(
                crop_logits,
                persisted_logits,
            ):
                raise ValueError("B-Free artifact readback differs")
            relative_artifact = repo_relative(artifact_path, repo_root)
            result = {
                **identity,
                "status": "ok",
                "valid_for_metrics": True,
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_ms,
                "artifact_path": relative_artifact,
                "artifact_sha256": sha256_file(artifact_path),
                "artifact_keys": ["features", "crop_logits"],
                "artifact_paths": {"bfree_npz": relative_artifact},
                "feature_shape": [CROP_COUNT, FEATURE_DIMENSION],
                "feature_dtype": "float32",
                "feature_semantics": FEATURE_SEMANTICS,
                "feature_array_sha256": _array_sha256(features),
                "crop_logits_shape": [CROP_COUNT],
                "crop_logits_dtype": "float32",
                "crop_logits_array_sha256": _array_sha256(crop_logits),
                "latency_ms": latency,
                "peak_cuda_memory_bytes": peak,
                **scoring,
            }
            _reject_t2_payload(result)
            append_jsonl(results_path, result)
            latest[sample_id] = result
            completed += 1
            print(
                f"[{index}/{len(selected)}] ok {sample_id} "
                f"raw_logit={result['ai_score']:.9f}",
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
                "ai_score": None,
                "score": None,
                "fake_probability": None,
                "crop_logits": None,
                "classification_decision": None,
                "latency_ms": 0.0,
                "peak_cuda_memory_bytes": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _reject_t2_payload(error_row)
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

    physical = read_jsonl(results_path) if results_path.is_file() else []
    summary = summarize_bfree_results(
        physical,
        selected,
        threshold=FIXED_THRESHOLD,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_id": CHECKPOINT["id"],
            "checkpoint_sha256": CHECKPOINT["sha256"],
            "preprocess_profile": PREPROCESS_PROFILE,
            "config_fingerprint": fingerprint,
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
        "prior_errors_retained": prior_errors_skipped,
        "new_errors": errors,
        "physical_result_rows": len(physical),
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": sum(
                path.is_file() for path in artifact_dir.glob("*.npz")
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
                "official_calibration": summary["official_calibration"],
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
