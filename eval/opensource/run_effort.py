#!/usr/bin/env python3
"""Run the official Effort CLIP ViT-L/14 GenImage detector.

The released natural-image checkpoint is evaluated through the inference
contract exposed by ``DeepfakeBench/training/demo.py``: OpenCV BGR decode,
RGB conversion, a direct 224x224 ``INTER_LINEAR`` resize, CLIP
normalization, and the two-class Effort head.  The official demo loads the
checkpoint with ``strict=False`` even though that line is marked ``FIXME``.
This adapter instead constructs the exact forward-relevant graph represented
by the released checkpoint and requires a strict 681/681 tensor match.

Effort is an image-level T1 detector.  It does not produce a manipulation
mask, so T2 and joint localization metrics are deliberately unavailable.
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


MODEL_NAME = "Effort"
MODEL_SLUG = "effort_clip_l14_genimage_sdv14"
MODEL_ARCH = "HF CLIP ViT-L/14 vision + rank-1 SVD residual attention + 2-class head"
MODEL_REPO_URL = "https://github.com/YZY-stack/Effort-AIGI-Detection"
MODEL_SOURCE_COMMIT = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
PAPER_URL = "https://proceedings.mlr.press/v267/yan25b.html"
ARXIV_URL = "https://arxiv.org/abs/2411.15633"

PREPROCESS_PROFILE = "official_deepfakebench_demo_natural_image_linear224_v1"
MODEL_SEED = 20260724
MODEL_INPUT_SIZE = 224
FEATURE_DIMENSION = 1024
CLASS_COUNT = 2
SVD_MODULE_COUNT = 96
SVD_FROZEN_RANK = 1023
SVD_RESIDUAL_RANK = 1
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
SCORE_SEMANTICS = "official_float32_softmax_class1_probability_higher_is_fake"
FEATURE_SEMANTICS = "official_clip_vision_pooler_output_before_effort_head"
T1_POLICY = "official_fake_probability_strictly_greater_than_0_5"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CPU_THREADS = 16

TORCH_VERSION = "2.8.0.dev20250627+cu128"
TORCHVISION_VERSION = "0.23.0.dev20250627+cu128"
TRANSFORMERS_VERSION = "4.53.2"
NUMPY_VERSION = "1.26.4"
OPENCV_VERSION = "4.10.0"

SOURCE_FILES = {
    "README.md": "f5c0f66ed8566c65818162722c9935485721ad401bc8494d2742b32b75fd5721",
    "install.sh": "b37e791f514b28f09f64ad84b689d5a9deaff2c3abb688350aae7cbb0711c7fd",
    "DeepfakeBench/training/demo.py": (
        "009db8f76d3983e22d0e241ef602b11908f652f84d0fd5f5857f1973fdd12f9c"
    ),
    "DeepfakeBench/training/detectors/effort_detector.py": (
        "366b1cde008f537e4b9c8c8e4c65ee20b430c4bca1ccee1b1b86c20a9831fac9"
    ),
    "DeepfakeBench/training/config/detector/effort.yaml": (
        "1fd1398cf245b3a5c13cb130d7c6e209057ae6b56b561eca5f3c032283c5527b"
    ),
    "figs/effort_pipeline.png": (
        "f84fad60b6152b915874cfbd58ee7e21646fd4a36a642683dbb425e6f6bc879b"
    ),
    "figs/deepfake_tab1.png": (
        "f8494b571f9d663639193344fa8e0e18f1d41f42089f01f06133dc881ab39fc7"
    ),
}

CHECKPOINT = {
    "id": "official_effort_genimage_sdv14_clip_l14",
    "training_release": "GenImage_SDv1.4",
    "filename": "effort_clip_L14_trainOn_sdv14.pth",
    "google_drive_id": "1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d",
    "official_url": (
        "https://drive.google.com/file/d/"
        "1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d/view"
    ),
    "bytes": 1_213_769_519,
    "sha256": "7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813",
    "format": "torch_zip_direct_ordered_state_dict",
    "tensor_count": 681,
    "state_elements": 303_378_530,
    "dtype": "float32",
    "ordered_key_sha256": (
        "1782f72f07007cebae76a0f315845f1c60456d9223d47c8ce2f35a8f43816da7"
    ),
    "schema_sha256": (
        "bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489"
    ),
}

HF_CONFIG = {
    "model_id": "openai/clip-vit-large-patch14",
    "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "filename": "config.json",
    "bytes": 4_519,
    "sha256": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
    "official_url": (
        "https://huggingface.co/openai/clip-vit-large-patch14/resolve/"
        "32bd64288804d66eefd0ccbe215aa642df71cc41/config.json"
    ),
}

LICENSE_RECORD = {
    "repository_readme_badge": "CC-BY-NC-4.0",
    "tracked_license_file_present": False,
    "code_license_text_verified": False,
    "checkpoint_license_text_verified": False,
    "commercial_use_cleared": False,
    "benchmark_role": "research_evaluation_only",
    "note": (
        "The official README displays a CC BY-NC 4.0 badge, but the pinned "
        "repository contains no tracked LICENSE/COPYING/NOTICE text."
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

GOLDEN_CASES = (
    {
        "path": "figs/effort_pipeline.png",
        "sha256": SOURCE_FILES["figs/effort_pipeline.png"],
        "decoded_rgb_sha256": (
            "4342b8fcd8fd3f3fd9149b5d8a90f54c6484a3cb564763ed4255c618121397cd"
        ),
        "resized_rgb_sha256": (
            "72a7fc8ecc1e1eef640db7b9f18961bd8a94a6cf70243399272544d45b11ee2b"
        ),
        "tensor_sha256": (
            "afa0e21082a140a5996e1e9d0d0fad288f567f2f0f7dd7015f7b57653f920911"
        ),
        "cpu_logits": [0.22981402277946472, -1.3408616781234741],
        "cpu_probability": 0.1721200793981552,
        "cuda_logits": [0.2298070341348648, -1.3408538103103638],
        "cuda_probability": 0.17212221026420593,
    },
    {
        "path": "figs/deepfake_tab1.png",
        "sha256": SOURCE_FILES["figs/deepfake_tab1.png"],
        "decoded_rgb_sha256": (
            "620ccd5c30d7499a37412f8d647e8e04d28621eb269121017111eebfdf4e4eb9"
        ),
        "resized_rgb_sha256": (
            "e7a64c277aeede3cfa3d5daff7a2540ba6f8839e5426e4bab621a720a5b50104"
        ),
        "tensor_sha256": (
            "e9fd00a326a8b68bd8d28c984a64cfc6529bf10e7a0d3773fe1975a2449b7429"
        ),
        "cpu_logits": [-0.16196167469024658, -0.9527153372764587],
        "cpu_probability": 0.3120068609714508,
        "cuda_logits": [-0.16196896135807037, -0.9527121186256409],
        "cuda_probability": 0.3120090961456299,
    },
)
GOLDEN_RUNTIME_ABS_TOLERANCE = 1e-6
GOLDEN_CPU_CUDA_ABS_TOLERANCE = 5e-5

DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/effort-aigi-96f5dea2"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/models/effort/"
    "effort_clip_L14_trainOn_sdv14.pth"
)
DEFAULT_HF_CONFIG = Path(
    "/root/.cache/claimforge/models/effort/"
    "clip-vit-large-patch14-config.json"
)
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/effort")
DEFAULT_RUN_ID = (
    "effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725"
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


def _safe_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} is not a safe path component")
    return value


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


def _state_schema_sha256(state: Mapping[str, Any]) -> str:
    canonical = "\n".join(
        f"{key}\t{tuple(value.shape)}\t{value.dtype}\t{value.numel()}"
        for key, value in state.items()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adapter_contract(repo_root: Path) -> dict[str, Any]:
    relatives = (
        "eval/opensource/run_effort.py",
        "eval/opensource/effort_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
    )
    result: dict[str, Any] = {}
    for relative in relatives:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Effort adapter file: {path}")
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


def build_pair_visibility(
    all_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    """Freeze whole-canvas geometric visibility before model scoring."""

    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        by_task.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in sorted(by_task.items()):
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"incomplete canonical pair {task_id}")
        forged = pair["forged"]
        mask = _load_gt_mask(forged, repo_root)
        assert mask is not None
        positive = int(np.count_nonzero(mask == 255))
        if positive <= 0:
            raise ValueError(f"empty forged mask for {task_id}")
        result[task_id] = {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "definition": (
                    "whole_canvas_direct_resize_to_224_all_native_pixels_"
                    "within_geometric_input_domain"
                ),
                "native_width": int(forged["width"]),
                "native_height": int(forged["height"]),
                "gt_positive_pixels": positive,
                "model_input_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "resize": "cv2_INTER_LINEAR_direct_aspect_ratio_distortion",
                "crop": None,
            },
        }
    census = Counter(value["edit_visibility"] for value in result.values())
    if len(result) != 275 or dict(census) != {"full": 275}:
        raise ValueError("Effort full-dataset visibility census changed")
    return result


def verify_source(source_root: Path) -> dict[str, Any]:
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"Effort source commit changed: {commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError("Effort tracked source tree is dirty")
    file_records: dict[str, Any] = {}
    for relative, expected in SOURCE_FILES.items():
        path = source_root / relative
        _verify_runtime_file(path, expected, f"Effort source {relative}")
        file_records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": expected,
        }
    tracked = _git_value(source_root, "ls-files") or ""
    tracked_names = set(tracked.splitlines())
    license_names = sorted(
        name
        for name in tracked_names
        if Path(name).name.lower()
        in {"license", "license.txt", "copying", "notice", "notice.txt"}
    )
    if license_names:
        raise ValueError("Effort pinned source unexpectedly gained license text")
    return {
        "repository": MODEL_REPO_URL,
        "path": str(source_root),
        "commit": commit,
        "tracked_dirty": False,
        "tracked_license_files": license_names,
        "files": file_records,
    }


def verify_assets(
    checkpoint_path: Path,
    hf_config_path: Path,
) -> tuple[dict[str, Any], OrderedDict[str, Any], dict[str, Any]]:
    import torch

    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("Effort checkpoint byte size changed")
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "Effort GenImage checkpoint",
    )
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(
        checkpoint_path
    )
    if unsafe:
        raise ValueError(f"Effort checkpoint has unsafe globals: {unsafe}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, OrderedDict):
        raise ValueError("Effort checkpoint is not a direct OrderedDict")
    if len(payload) != int(CHECKPOINT["tensor_count"]):
        raise ValueError("Effort checkpoint tensor count changed")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("Effort checkpoint contains non-tensor values")
    if any(value.dtype != torch.float32 for value in payload.values()):
        raise ValueError("Effort checkpoint contains non-FP32 tensors")
    if any(not torch.isfinite(value).all().item() for value in payload.values()):
        raise ValueError("Effort checkpoint contains non-finite tensors")
    elements = sum(value.numel() for value in payload.values())
    if elements != int(CHECKPOINT["state_elements"]):
        raise ValueError("Effort checkpoint element count changed")
    key_digest = hashlib.sha256(
        "\n".join(payload.keys()).encode("utf-8")
    ).hexdigest()
    if key_digest != CHECKPOINT["ordered_key_sha256"]:
        raise ValueError("Effort checkpoint ordered keys changed")
    schema_digest = _state_schema_sha256(payload)
    if schema_digest != CHECKPOINT["schema_sha256"]:
        raise ValueError("Effort checkpoint schema changed")
    stripped: OrderedDict[str, Any] = OrderedDict()
    for key, value in payload.items():
        if not key.startswith("module."):
            raise ValueError("Effort checkpoint key lacks module. prefix")
        new_key = key[len("module.") :]
        if new_key in stripped:
            raise ValueError("Effort checkpoint prefix stripping collides")
        stripped[new_key] = value
    if tuple(stripped["head.weight"].shape) != (2, FEATURE_DIMENSION):
        raise ValueError("Effort head weight shape changed")
    if tuple(stripped["head.bias"].shape) != (2,):
        raise ValueError("Effort head bias shape changed")

    if hf_config_path.stat().st_size != int(HF_CONFIG["bytes"]):
        raise ValueError("pinned CLIP config byte size changed")
    _verify_runtime_file(
        hf_config_path,
        str(HF_CONFIG["sha256"]),
        "pinned CLIP config",
    )
    config_payload = json.loads(hf_config_path.read_text(encoding="utf-8"))
    vision = config_payload.get("vision_config")
    expected_vision = {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "projection_dim": 768,
    }
    if not isinstance(vision, Mapping):
        raise ValueError("pinned CLIP config has no vision_config")
    for key, expected in expected_vision.items():
        if vision.get(key) != expected:
            raise ValueError(f"pinned CLIP vision config field changed: {key}")

    assets = {
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path),
            "serialization_safety": {
                "weights_only": True,
                "unsafe_globals": [],
            },
            "top_level_type": "collections.OrderedDict",
            "schema_verified": True,
        },
        "clip_config": {
            **HF_CONFIG,
            "path": str(hf_config_path),
            "vision_config_contract": expected_vision,
        },
    }
    return assets, stripped, config_payload


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import cv2
    import torch
    import torchvision
    import transformers

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    expected = {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "numpy": NUMPY_VERSION,
        "opencv": OPENCV_VERSION,
    }
    for key, value in expected.items():
        if versions[key] != value:
            raise ValueError(
                f"Effort runtime {key} changed: {versions[key]} != {value}"
            )
    device = torch.device(device_text)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Effort supports only cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device, {
        **versions,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "autocast": False,
        "model_dtype": "float32",
        "batch_size": 1,
        "cpu_threads": CPU_THREADS,
        "seed": MODEL_SEED,
    }


def _build_model(
    state: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import CLIPVisionConfig, CLIPVisionModel

    class SVDResidualLinear(nn.Module):
        def __init__(self, dimension: int = FEATURE_DIMENSION) -> None:
            super().__init__()
            self.in_features = dimension
            self.out_features = dimension
            self.r = SVD_FROZEN_RANK
            self.weight_main = nn.Parameter(
                torch.empty(dimension, dimension),
                requires_grad=False,
            )
            self.bias = nn.Parameter(torch.empty(dimension))
            self.S_residual = nn.Parameter(torch.empty(SVD_RESIDUAL_RANK))
            self.U_residual = nn.Parameter(
                torch.empty(dimension, SVD_RESIDUAL_RANK)
            )
            self.V_residual = nn.Parameter(
                torch.empty(SVD_RESIDUAL_RANK, dimension)
            )

        def forward(self, value: Any) -> Any:
            residual = (
                self.U_residual
                @ torch.diag(self.S_residual)
                @ self.V_residual
            )
            return functional.linear(
                value,
                self.weight_main + residual,
                self.bias,
            )

    class EffortInferenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vision_config = CLIPVisionConfig(
                **dict(config_payload["vision_config"])
            )
            self.backbone = CLIPVisionModel(vision_config).vision_model
            for layer in self.backbone.encoder.layers:
                attention = layer.self_attn
                for name in ("k_proj", "v_proj", "q_proj", "out_proj"):
                    setattr(attention, name, SVDResidualLinear())
            self.head = nn.Linear(FEATURE_DIMENSION, CLASS_COUNT)

        def forward(self, image: Any) -> tuple[Any, Any]:
            feature = self.backbone(image).pooler_output
            logits = self.head(feature)
            return logits, feature

    with torch.device("meta"):
        model = EffortInferenceModel()
    model_state = model.state_dict()
    if list(model_state) != list(state):
        missing = sorted(set(model_state) - set(state))
        unexpected = sorted(set(state) - set(model_state))
        raise ValueError(
            "Effort strict model schema mismatch before load: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    load_result = model.load_state_dict(
        state,
        strict=True,
        assign=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError("Effort strict load unexpectedly reported drift")
    # ``position_ids`` is a non-persistent Hugging Face buffer.  A meta-device
    # construction therefore needs the same arange materialization that the
    # ordinary constructor performs before the model can be moved.
    model.backbone.embeddings.position_ids = torch.arange(
        257,
        dtype=torch.long,
    ).expand((1, -1))
    model = model.to(device)
    model.eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Effort model retains meta parameters")
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError("Effort model retains meta buffers")
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SVDResidualLinear)
    ]
    if len(modules) != SVD_MODULE_COUNT:
        raise ValueError("Effort SVD module count changed")
    expected_names = [
        f"backbone.encoder.layers.{layer}.self_attn.{projection}"
        for layer in range(24)
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj")
    ]
    if [name for name, _ in modules] != expected_names:
        raise ValueError("Effort SVD module ordering changed")
    for _, module in modules:
        if (
            tuple(module.weight_main.shape) != (1024, 1024)
            or tuple(module.U_residual.shape) != (1024, 1)
            or tuple(module.S_residual.shape) != (1,)
            or tuple(module.V_residual.shape) != (1, 1024)
        ):
            raise ValueError("Effort SVD residual shape changed")
    audit = {
        "constructor": "shape_only_exact_checkpoint_forward_graph",
        "official_forward_formula": (
            "weight_main + U_residual @ diag(S_residual) @ V_residual"
        ),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "state_entries": len(model_state),
        "state_elements": sum(value.numel() for value in model_state.values()),
        "svd_modules": len(modules),
        "svd_module_names": [name for name, _ in modules],
        "frozen_rank": SVD_FROZEN_RANK,
        "residual_rank": SVD_RESIDUAL_RANK,
        "feature_dimension": FEATURE_DIMENSION,
        "head_weight_shape": list(model.head.weight.shape),
        "head_bias_shape": list(model.head.bias.shape),
        "nonpersistent_position_ids_materialized": True,
        "position_ids_shape": list(
            model.backbone.embeddings.position_ids.shape
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "eval_mode": not model.training,
    }
    return model, audit


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV failed to decode image: {path}")
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Effort OpenCV decode is not uint8 BGR")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (
        resized.astype(np.float32) / np.float32(255.0) - mean
    ) / std
    tensor = np.ascontiguousarray(
        normalized.transpose(2, 0, 1),
        dtype=np.float32,
    )
    if tensor.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("Effort normalized tensor shape changed")
    if not np.isfinite(tensor).all():
        raise ValueError("Effort normalized tensor is non-finite")
    audit = {
        "decode": "cv2.imread_IMREAD_COLOR",
        "native_shape_hwc": [int(value) for value in bgr.shape],
        "native_width": int(bgr.shape[1]),
        "native_height": int(bgr.shape[0]),
        "decoded_bgr_sha256": _array_sha256(bgr),
        "color_conversion": "cv2_COLOR_BGR2RGB",
        "decoded_rgb_sha256": _array_sha256(rgb),
        "resize": {
            "output_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "interpolation": "cv2_INTER_LINEAR",
            "preserve_aspect_ratio": False,
            "crop": None,
            "face_alignment": False,
        },
        "resized_rgb_sha256": _array_sha256(resized),
        "to_tensor": "uint8_to_float32_divide_255_CHW",
        "normalization_mean": list(CLIP_MEAN),
        "normalization_std": list(CLIP_STD),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_sha256": _array_sha256(tensor),
    }
    return tensor, audit


def _float32_probability(logits: Any) -> Any:
    import torch

    if logits.dtype != torch.float32:
        raise ValueError("Effort logits are not float32")
    return torch.softmax(logits, dim=1)[:, 1]


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int | None, float]:
    import torch
    from torch.nn import functional

    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    peak: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits, feature = model(tensor)
        replay = functional.linear(
            feature,
            model.head.weight,
            model.head.bias,
        )
        probability = _float32_probability(logits)
        margin = logits[:, 1] - logits[:, 0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not torch.equal(logits, replay):
        raise ValueError("Effort head replay differs from model logits")
    feature_array = np.ascontiguousarray(
        feature[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    logits_array = np.ascontiguousarray(
        logits[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    if (
        feature_array.shape != (FEATURE_DIMENSION,)
        or logits_array.shape != (CLASS_COUNT,)
        or not np.isfinite(feature_array).all()
        or not np.isfinite(logits_array).all()
    ):
        raise ValueError("Effort output arrays violate frozen contract")
    score = float(probability[0].item())
    raw_margin = float(margin[0].item())
    if not 0.0 <= score <= 1.0 or not math.isfinite(raw_margin):
        raise ValueError("Effort score is invalid")
    scoring = {
        "class_logits": [float(value) for value in logits_array.tolist()],
        "raw_logit_margin": raw_margin,
        "fake_probability": score,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": SCORE_SEMANTICS,
        "classification_decision": score > CLASSIFICATION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": {
            "decision": score > CLASSIFICATION_THRESHOLD,
            "threshold": CLASSIFICATION_THRESHOLD,
            "operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        },
        "t1": {
            "valid": True,
            "score": score,
            "decision": score > CLASSIFICATION_THRESHOLD,
        },
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
        },
    }
    return scoring, feature_array, logits_array, peak, latency_ms


def validate_runtime_golden(
    model: Any,
    device: Any,
    source_root: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    field = "cuda" if device.type == "cuda" else "cpu"
    for frozen in GOLDEN_CASES:
        path = source_root / str(frozen["path"])
        _verify_runtime_file(path, str(frozen["sha256"]), "Effort fixture")
        image, preprocess = preprocess_image(path)
        if (
            preprocess["decoded_rgb_sha256"]
            != frozen["decoded_rgb_sha256"]
            or preprocess["resized_rgb_sha256"]
            != frozen["resized_rgb_sha256"]
            or preprocess["tensor_sha256"] != frozen["tensor_sha256"]
        ):
            raise ValueError("Effort fixture preprocessing drifted")
        first, feature_a, logits_a, _, _ = infer_one(model, device, image)
        second, feature_b, logits_b, _, _ = infer_one(model, device, image)
        if not np.array_equal(feature_a, feature_b):
            raise ValueError("Effort fixture features are not repeatable")
        if not np.array_equal(logits_a, logits_b):
            raise ValueError("Effort fixture logits are not repeatable")
        if first != second:
            raise ValueError("Effort fixture scores are not repeatable")
        expected_logits = np.asarray(
            frozen[f"{field}_logits"],
            dtype=np.float64,
        )
        logit_diff = float(
            np.max(np.abs(logits_a.astype(np.float64) - expected_logits))
        )
        probability_diff = abs(
            float(first["ai_score"])
            - float(frozen[f"{field}_probability"])
        )
        if (
            logit_diff > GOLDEN_RUNTIME_ABS_TOLERANCE
            or probability_diff > GOLDEN_RUNTIME_ABS_TOLERANCE
        ):
            raise ValueError("Effort runtime fixture regression changed")
        cross_runtime = float(
            np.max(
                np.abs(
                    np.asarray(frozen["cpu_logits"], dtype=np.float64)
                    - np.asarray(frozen["cuda_logits"], dtype=np.float64)
                )
            )
        )
        if cross_runtime > GOLDEN_CPU_CUDA_ABS_TOLERANCE:
            raise ValueError("Effort frozen CPU/CUDA fixture gap is too large")
        cases.append(
            {
                "path": str(frozen["path"]),
                "input_sha256": str(frozen["sha256"]),
                "preprocess": preprocess,
                "logits": [float(value) for value in logits_a.tolist()],
                "fake_probability": float(first["ai_score"]),
                "feature_sha256": _array_sha256(feature_a),
                "repeat_feature_max_abs_diff": 0.0,
                "repeat_logit_max_abs_diff": 0.0,
                "frozen_runtime_logit_max_abs_diff": logit_diff,
                "frozen_runtime_probability_abs_diff": probability_diff,
                "frozen_cpu_cuda_logit_max_abs_diff": cross_runtime,
            }
        )
    return {
        "status": "passed",
        "kind": "repository_fixture_runtime_regression_not_author_published_golden",
        "device_family": field,
        "runtime_abs_tolerance": GOLDEN_RUNTIME_ABS_TOLERANCE,
        "cpu_cuda_abs_tolerance": GOLDEN_CPU_CUDA_ABS_TOLERANCE,
        "cases": cases,
        "mouse_model_scores_computed": 0,
    }


def _atomic_save_artifact(
    path: Path,
    feature: np.ndarray,
    logits: np.ndarray,
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
                pooler_output=np.ascontiguousarray(
                    feature,
                    dtype=np.float32,
                ),
                class_logits=np.ascontiguousarray(
                    logits,
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
            if set(payload.files) != {"pooler_output", "class_logits"}:
                raise ValueError("Effort artifact keys changed")
            feature = np.ascontiguousarray(payload["pooler_output"])
            logits = np.ascontiguousarray(payload["class_logits"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot safely load Effort artifact: {path}") from exc
    if (
        feature.shape != (FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or logits.shape != (CLASS_COUNT,)
        or logits.dtype != np.float32
        or not np.isfinite(feature).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Effort artifact arrays violate frozen contract")
    return feature, logits


def _artifact_path(run_dir: Path, sample_id: str) -> Path:
    _safe_component(sample_id, label="sample-id")
    artifact_dir = (run_dir / "artifacts").resolve()
    path = (artifact_dir / f"{sample_id}.npz").resolve()
    if path.parent != artifact_dir:
        raise ValueError("Effort artifact path escapes artifact directory")
    return path


_FORBIDDEN_T2_KEYS = {
    "t2",
    "localization",
    "localisation",
    "localization_metrics",
    "localisation_metrics",
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
    "miou",
    "dice",
    "pixel_f1",
    "s_joint",
    "joint_score",
    "joint_metrics",
}


def _reject_t2_payload(value: Any, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        present = sorted(_FORBIDDEN_T2_KEYS.intersection(value))
        if present:
            raise ValueError(f"{path} invents Effort T2 fields: {present}")
        for key, child in value.items():
            _reject_t2_payload(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_t2_payload(child, path=f"{path}[{index}]")


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
        "checkpoint_id": str(CHECKPOINT["id"]),
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
    model: Any,
    device: Any,
) -> None:
    import torch
    from torch.nn import functional

    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Effort resume identity field {key} changed")
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("only valid successful Effort rows may be skipped")
    score = _finite_number(row.get("ai_score"), "resume ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("resume Effort score falls outside [0,1]")
    if row.get("score") != score or row.get("probability") != score:
        raise ValueError("resume Effort score aliases changed")
    if row.get("fake_probability") != score:
        raise ValueError("resume Effort fake probability changed")
    if row.get("classification_decision") is not (score > 0.5):
        raise ValueError("resume Effort strict decision changed")
    input_path = _anchored(Path(str(expected["input_path"])), repo_root)
    _verify_runtime_file(
        input_path,
        str(expected["input_sha256"]),
        "resume Effort input",
    )
    _, replay_preprocess = preprocess_image(input_path)
    if row.get("preprocess") != replay_preprocess:
        raise ValueError("resume Effort preprocessing does not replay")
    artifact_value = row.get("artifact_path")
    if not isinstance(artifact_value, str):
        raise ValueError("resume Effort artifact path missing")
    artifact_path = _anchored(Path(artifact_value), repo_root)
    expected_path = _artifact_path(run_dir, str(row["id"]))
    if artifact_path != expected_path:
        raise ValueError("resume Effort artifact path changed or escapes")
    _verify_runtime_file(
        artifact_path,
        str(row.get("artifact_sha256")),
        "resume Effort artifact",
    )
    feature, logits = _load_artifact(artifact_path)
    if (
        _array_sha256(feature) != row.get("feature_array_sha256")
        or _array_sha256(logits) != row.get("class_logits_array_sha256")
        or [float(value) for value in logits.tolist()]
        != row.get("class_logits")
    ):
        raise ValueError("resume Effort artifact content changed")
    with torch.inference_mode():
        replay = functional.linear(
            torch.from_numpy(feature).to(device).unsqueeze(0),
            model.head.weight,
            model.head.bias,
        )[0]
        replay_probability = _float32_probability(
            replay.unsqueeze(0)
        )[0]
    if not torch.equal(replay.detach().cpu(), torch.from_numpy(logits)):
        raise ValueError("resume Effort head replay changed")
    if float(replay_probability.item()) != score:
        raise ValueError("resume Effort softmax replay changed")
    _reject_t2_payload(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-config", type=Path, default=DEFAULT_HF_CONFIG)
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
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source_root = _anchored(args.source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    hf_config_path = _anchored(args.hf_config, repo_root)
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
    source = verify_source(source_root)
    device, runtime = configure_runtime(args.device)
    assets, state, config_payload = verify_assets(
        checkpoint_path,
        hf_config_path,
    )
    model, model_audit = _build_model(state, config_payload, device)
    del state
    golden = validate_runtime_golden(model, device, source_root)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model": MODEL_NAME,
                    "source_commit": source["commit"],
                    "checkpoint_sha256": assets["checkpoint"]["sha256"],
                    "checkpoint_schema_sha256": assets["checkpoint"][
                        "schema_sha256"
                    ],
                    "strict_model_load": model_audit["strict_load"],
                    "svd_modules": model_audit["svd_modules"],
                    "runtime_golden": golden,
                    "visibility_census": {"full": 275},
                    "mouse_model_scores_computed": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    from eval.opensource.effort_metrics import summarize_effort_results

    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root.resolve():
        raise ValueError("run-id escapes Effort results directory")
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

    selected_ids = [str(row["sample_id"]) for row in selected]
    selected_tasks = sorted({str(row["task_id"]) for row in selected})
    config = {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "architecture": MODEL_ARCH,
        "source_commit": MODEL_SOURCE_COMMIT,
        "checkpoint": CHECKPOINT,
        "clip_config": HF_CONFIG,
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "cv2.imread_IMREAD_COLOR",
            "color": "cv2_BGR2RGB",
            "resize": "cv2_resize_224x224_INTER_LINEAR_no_aspect_preservation",
            "to_tensor": "official_torchvision_equivalent_float32_divide_255",
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
            "face_alignment": False,
            "known_official_drift": (
                "DeepfakeBench dataset test path uses INTER_CUBIC; the "
                "released natural-image README demo uses INTER_LINEAR and "
                "is frozen here"
            ),
        },
        "score_semantics": SCORE_SEMANTICS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "runtime": runtime,
        "adapter": adapter_contract(repo_root),
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "runtime_golden": golden,
        "release": release,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "selected_sample_ids": selected_ids,
        "selected_tasks": selected_tasks,
        "pair_limit": args.pair_limit,
        "sample_id": args.sample_id,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "device": str(device),
        "visibility_census": {"full": 275},
        "task_scope": {
            "primary": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "license": LICENSE_RECORD,
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
    }
    config = json.loads(stable_json(config))
    fingerprint = _manifest_fingerprint(config)
    prior_manifest: dict[str, Any] | None = None
    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "Effort resume requires run_manifest and expected_inputs"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            prior_manifest.get("schema_version")
            != "effort_detection_run_manifest_v1"
            or prior_manifest.get("run_id") != args.run_id
            or prior_manifest.get("config") != config
            or prior_manifest.get("config_fingerprint") != fingerprint
            or read_jsonl(expected_path) != selected
        ):
            raise ValueError("Effort resume identity/config changed")
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

    manifest: dict[str, Any] = {
        "schema_version": "effort_detection_run_manifest_v1",
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
        "runtime_golden": golden,
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
        "visibility_census": {"full": len(selected_tasks)},
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
                model=model,
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
            scoring, feature, logits, peak, latency = infer_one(
                model,
                device,
                image,
            )
            artifact_path = _artifact_path(run_dir, sample_id)
            _atomic_save_artifact(artifact_path, feature, logits)
            persisted_feature, persisted_logits = _load_artifact(
                artifact_path
            )
            if not np.array_equal(feature, persisted_feature):
                raise ValueError("Effort persisted feature differs")
            if not np.array_equal(logits, persisted_logits):
                raise ValueError("Effort persisted logits differ")
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
                "artifact_keys": ["pooler_output", "class_logits"],
                "artifact_paths": {"effort_npz": relative_artifact},
                "feature_shape": [FEATURE_DIMENSION],
                "feature_dtype": "float32",
                "feature_semantics": FEATURE_SEMANTICS,
                "feature_array_sha256": _array_sha256(feature),
                "class_logits_shape": [CLASS_COUNT],
                "class_logits_dtype": "float32",
                "class_logits_array_sha256": _array_sha256(logits),
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
                f"p_fake={result['ai_score']:.9f}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            error_row = {
                **identity,
                "status": "error",
                "valid_for_metrics": False,
                "completed_at": utc_now(),
                "class_logits": None,
                "raw_logit_margin": None,
                "fake_probability": None,
                "probability": None,
                "ai_score": None,
                "score": None,
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

    physical = read_jsonl(results_path) if results_path.is_file() else []
    summary = summarize_effort_results(
        physical,
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
            "checkpoint_sha256": CHECKPOINT["sha256"],
            "preprocess_profile": PREPROCESS_PROFILE,
            "config_fingerprint": fingerprint,
            "runtime_golden_status": golden["status"],
            "runtime_golden_fingerprint": _manifest_fingerprint(golden),
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
