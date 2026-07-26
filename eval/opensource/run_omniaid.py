#!/usr/bin/env python3
"""Run the official OmniAID-DINO v2 whole-image AIGC detector.

The current official Space defaults to OmniAID-DINO v2 in automatic-router
mode: PIL RGB decode, direct bilinear resize to 448x448, ImageNet
normalization, a DINOv3 ViT-L/16 routing backbone, and a six-expert hybrid
MoE detector.  Two of five semantic experts are selected per image and the
universal artifact expert is always active.

The DINOv3 base repository is gated, but the released OmniAID checkpoint
contains the complete state of both DINO graphs.  This adapter therefore
constructs the exact graph on the meta device from the pinned official
configuration, requires a strict 2852/2852 tensor match, and materializes
only the two non-persistent RoPE buffers from their official formula.

OmniAID is an image-level T1 detector.  It does not produce a manipulation
mask, so T2 and joint localization metrics are deliberately unavailable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
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
import types
from collections import Counter
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


MODEL_NAME = "OmniAID"
MODEL_SLUG = "omniaid_dino_v2_mirage_auto_router"
MODEL_ARCH = (
    "DINOv3 ViT-L/16 feature extractor + rank-1 six-expert SVD-MoE "
    "DINOv3 ViT-L/16 + 2-class head"
)
MODEL_REPO_URL = "https://github.com/yunncheng/OmniAID"
MODEL_SOURCE_COMMIT = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
MODEL_SPACE_URL = "https://huggingface.co/spaces/Yunncheng/OmniAID-Demo"
MODEL_SPACE_COMMIT = "cf99ed518af8b7256854d01994d6e41165553bb3"
MODEL_HF_URL = "https://huggingface.co/Yunncheng/OmniAID"
MODEL_HF_REVISION = "279cae7398ac6636f46fc4668f755f11210b36bf"
PAPER_URL = "https://arxiv.org/html/2511.08423"
ARXIV_URL = "https://arxiv.org/abs/2511.08423"

PREPROCESS_PROFILE = "official_omniaid_space_dino_v2_auto_router_448_v1"
MODEL_SEED = 20260725
MODEL_INPUT_SIZE = 448
FEATURE_DIMENSION = 1024
CLASS_COUNT = 2
SVD_MODULE_COUNT = 96
SVD_FROZEN_RANK = 1023
SVD_RESIDUAL_RANK = 1
EXPERT_COUNT = 6
SEMANTIC_EXPERT_COUNT = 5
SEMANTIC_TOP_K = 2
ARTIFACT_EXPERT_INDEX = 5
ROUTER_HIDDEN_DIMENSION = 256
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
SCORE_SEMANTICS = "official_float32_softmax_class1_probability_higher_is_fake"
FEATURE_SEMANTICS = "official_moe_cls_pooler_output_before_omniaid_head"
ROUTING_FEATURE_SEMANTICS = (
    "official_frozen_dinov3_feature_extractor_pooler_output_for_router"
)
T1_POLICY = "official_fake_probability_strictly_greater_than_0_5"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CPU_THREADS = 16

TORCH_VERSION = "2.8.0.dev20250627+cu128"
TORCHVISION_VERSION = "0.23.0.dev20250627+cu128"
TRANSFORMERS_VERSION = "4.57.3"
NUMPY_VERSION = "2.2.6"
PILLOW_VERSION = "12.0.0"
HUGGINGFACE_HUB_VERSION = "0.36.0"

SOURCE_FILES = {
    "README.md": "cbf0aee17e5da907019703361c444becffaf76e7ecae96f20cec6b4ccba4e6b7",
    "requirements.txt": "f531378e8f4868c71e673e9ec6d83186e6b04ae62f7f89207c6706c46da7ddb8",
    "config/config_omniaid_dino.json": (
        "b5423615e3a8e35794bb6ee6f3603e333b366d5729a58a6d87a2f689e2b72504"
    ),
    "models/OmniAID_DINO.py": (
        "e6ac608693f11ba51a833bedaa9f9e503cd03c0ffc64198e71496b951632ead1"
    ),
    "reward/omniaid-dino.py": (
        "000ff8d6e794f3a9242c4565d094b569e3e671566ce76e4cf010bcd1a3efa7f6"
    ),
    "reward/clean_test.py": (
        "e7ca2bc02485ce15c680bd8af296c654b9bd4f5147669f95bf960dbb61fffa24"
    ),
}

SPACE_SOURCE_FILES = {
    "README.md": "344d460d746ac6531ba2ac80b69edca66f94d75f916e92f6db4b8dc0c1fb324e",
    "app.py": "4c1ac7b4eb1850beb97cd4db02105b4034052e3e8a5a96e8c278305561b5f8f2",
    "requirements.txt": "f531378e8f4868c71e673e9ec6d83186e6b04ae62f7f89207c6706c46da7ddb8",
    "config/config_omniaid_dino_v2.json": (
        "d97ded19543ca9459a86eddd4c0f08a8476dcd013a50f3bf81c4649f67536719"
    ),
    "model/omniaid-dino.py": (
        "046d5cf55238a20ec969a4b471aaefbfcf57eeac9f599932dd4e46ef7c63b8fe"
    ),
    "examples/fake_0.jpg": (
        "c480d5f88a967b78aca6f696725b410d40bcb3cd3305ae2461e0c7da2416e2fb"
    ),
    "examples/fake_1.jpg": (
        "40ab2f2e08fc6a79ffbbb5cf95a5b3184ac5786034bd362e898796d10e24aa8e"
    ),
    "examples/real_0.jpg": (
        "7c0037c10f0abce429eb87c05b8e788629c6ae9660f0a8372527dcfec324546c"
    ),
    "examples/real_1.jpg": (
        "0e758c21eefd95dc138502a62d3392d00d3991ebdd36030c3098f9af8f36ab89"
    ),
}

CHECKPOINT = {
    "id": "official_omniaid_dino_v2_mirage_train",
    "training_release": "Mirage-Train_plus_DDA-COCO",
    "filename": "checkpoint_omniaid_dino_v2.pth",
    "hf_revision": MODEL_HF_REVISION,
    "official_url": (
        "https://huggingface.co/Yunncheng/OmniAID/blob/"
        f"{MODEL_HF_REVISION}/ckpt/checkpoint_omniaid_dino_v2.pth"
    ),
    "bytes": 3_238_483_725,
    "sha256": "8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9",
    "format": "torch_zip_training_checkpoint",
    "top_level_keys": ["model", "optimizer", "epoch", "scaler", "args"],
    "epoch": 0,
    "unsafe_globals_allowlisted": ["argparse.Namespace"],
    "tensor_count": 2_852,
    "state_elements": 808_835_239,
    "dtype": "float32",
    "ordered_key_sha256": (
        "d894e539c44bfd3b036413db0e5c91d7de75552df5af2153ca4a83bc40e7d788"
    ),
    "schema_sha256": (
        "1b5a03a08369fa7dc5034b1b9aa8a4757295386afd3c91f093ef41b6e2c9b67d"
    ),
}

OMNIAID_CONFIG = {
    "filename": "config_omniaid_dino_v2.json",
    "bytes": 696,
    "sha256": "d97ded19543ca9459a86eddd4c0f08a8476dcd013a50f3bf81c4649f67536719",
    "official_url": (
        "https://huggingface.co/Yunncheng/OmniAID/blob/"
        f"{MODEL_HF_REVISION}/config/config_omniaid_dino_v2.json"
    ),
}

DINO_BASE = {
    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "config_bytes": 745,
    "weights_bytes": 1_212_559_808,
    "weights_sha256": (
        "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
    ),
    "access": "gated_not_downloaded_complete_weights_embedded_in_omniaid_checkpoint",
    "license": "custom_dinov3_license",
    "architecture_contract": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 16,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "num_register_tokens": 4,
        "layerscale_value": 1e-5,
        "use_gated_mlp": False,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "rope_theta": 100.0,
        "rope_rescale_coords": 2.0,
        "rope_normalize_coords": "separate",
    },
}

LICENSE_RECORD = {
    "repository_readme_badge": "MIT",
    "hf_model_card_metadata": "mit",
    "hf_space_metadata": "mit",
    "tracked_license_file_present": False,
    "code_license_text_verified": False,
    "checkpoint_card_license_metadata": "mit",
    "dinov3_base_license": "custom_dinov3_license",
    "commercial_use_cleared": False,
    "benchmark_role": "research_evaluation_only",
    "note": (
        "The repositories/cards display MIT metadata but the pinned GitHub "
        "tree has no tracked license text, and the embedded DINOv3-derived "
        "weights remain subject to Meta's custom DINOv3 license. This audit "
        "does not establish commercial-use clearance."
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
        "path": "examples/real_0.jpg",
        "sha256": SPACE_SOURCE_FILES["examples/real_0.jpg"],
        "decoded_rgb_sha256": (
            "221962ba61644669c7980b79f8241aaf29241aa3e95e731bf366f72a0ea640e4"
        ),
        "resized_rgb_sha256": (
            "fbcc74ff94685616c29363ad2d47b249365f66a2ec57921c0798312d1ca79bff"
        ),
        "tensor_sha256": (
            "f1c62f1ebc1e1e5c9808a3bec62b7f5151f3ddb701039f7f7d53a9d59d0c335a"
        ),
        "official_service_probability": 0.23996350169181824,
        "cuda_probability": 0.23996131122112274,
        "cuda_logits": [0.8654401302337646, -0.28745144605636597],
        "cuda_final_gates": [
            0.404275506734848,
            0.0,
            0.5957244634628296,
            0.0,
            0.0,
            1.0,
        ],
    },
    {
        "path": "examples/real_1.jpg",
        "sha256": SPACE_SOURCE_FILES["examples/real_1.jpg"],
        "decoded_rgb_sha256": (
            "374ce4cd0946c68454ae7b35b7516650f5623e4d4fd8924662c47b960af6ee94"
        ),
        "resized_rgb_sha256": (
            "ef5c696c56f284ae0e78633eac38d65d37433191cfecf9c0800d1b6fb6e284f1"
        ),
        "tensor_sha256": (
            "59615b03c329825bb351dda4ba34e413b723c10d68d4ba8ddcc74b4b92b6719f"
        ),
        "official_service_probability": 0.07805733382701874,
        "cuda_probability": 0.0780571699142456,
        "cuda_logits": [1.2692718505859375, -1.1997699737548828],
        "cuda_final_gates": [
            0.8894949555397034,
            0.0,
            0.0,
            0.11050505936145782,
            0.0,
            1.0,
        ],
    },
    {
        "path": "examples/fake_0.jpg",
        "sha256": SPACE_SOURCE_FILES["examples/fake_0.jpg"],
        "decoded_rgb_sha256": (
            "936649362cd3d1bccfb66c09f384aa0d5db33ad15a2689f3e168ae581bda60e4"
        ),
        "resized_rgb_sha256": (
            "918c09de4f3a06ee8ad83cb3b0118c07edeeb08809b02b4ceba8901b4422331c"
        ),
        "tensor_sha256": (
            "3b03a20b37597a902a0101966289e7f47ea239a6df7b54b0fba954b7cbcdcd77"
        ),
        "official_service_probability": 0.8572260141372681,
        "cuda_probability": 0.8572245836257935,
        "cuda_logits": [-1.02321195602417, 0.7692152261734009],
        "cuda_final_gates": [
            0.9100778698921204,
            0.0,
            0.08992215991020203,
            0.0,
            0.0,
            1.0,
        ],
    },
    {
        "path": "examples/fake_1.jpg",
        "sha256": SPACE_SOURCE_FILES["examples/fake_1.jpg"],
        "decoded_rgb_sha256": (
            "2e954d4a6345bc2349d486afbb98fdda407c943c7d24ba2fd4e4a30f473f5844"
        ),
        "resized_rgb_sha256": (
            "073b8c9b15e7c4b679ba18f98fd86be0fc3f06cd47dd1035d293e99a537cec8b"
        ),
        "tensor_sha256": (
            "ecf965977e557699fc7416ee96dbd52c753013c1c4cd1548036f8820d9fa4e18"
        ),
        "official_service_probability": 0.6094988584518433,
        "cuda_probability": 0.609500527381897,
        "cuda_logits": [-0.39048659801483154, 0.05472652614116669],
        "cuda_final_gates": [
            0.0,
            0.0,
            0.06175694987177849,
            0.9382430911064148,
            0.0,
            1.0,
        ],
    },
)
GOLDEN_RUNTIME_ABS_TOLERANCE = 1e-5
GOLDEN_SERVICE_ABS_TOLERANCE = 5e-5

DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/omniaid-40749406"
)
DEFAULT_SPACE_ROOT = Path(
    "/root/.cache/claimforge/third_party/omniaid-space-cf99ed51"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/models/omniaid/checkpoint_omniaid_dino_v2.pth"
)
DEFAULT_OMNIAID_CONFIG = Path(
    "/root/.cache/claimforge/models/omniaid/config_omniaid_dino_v2.json"
)
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/omniaid")
DEFAULT_RUN_ID = (
    "omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725"
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
        "eval/opensource/run_omniaid.py",
        "eval/opensource/omniaid_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
    )
    result: dict[str, Any] = {}
    for relative in relatives:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing OmniAID adapter file: {path}")
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
                    "whole_canvas_direct_resize_to_448_all_native_pixels_"
                    "within_geometric_input_domain"
                ),
                "native_width": int(forged["width"]),
                "native_height": int(forged["height"]),
                "gt_positive_pixels": positive,
                "model_input_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "resize": (
                    "torchvision_PIL_bilinear_direct_aspect_ratio_distortion"
                ),
                "crop": None,
            },
        }
    census = Counter(value["edit_visibility"] for value in result.values())
    if len(result) != 275 or dict(census) != {"full": 275}:
        raise ValueError("OmniAID full-dataset visibility census changed")
    return result


def _verify_git_source(
    root: Path,
    *,
    repository: str,
    commit: str,
    expected_files: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    actual_commit = _git_value(root, "rev-parse", "HEAD")
    if actual_commit != commit:
        raise ValueError(
            f"{label} source commit changed: {actual_commit} != {commit}"
        )
    dirty = _git_value(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"{label} tracked source tree is dirty")
    file_records: dict[str, Any] = {}
    for relative, expected in expected_files.items():
        path = root / relative
        _verify_runtime_file(path, expected, f"{label} source {relative}")
        file_records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": expected,
        }
    tracked = _git_value(root, "ls-files") or ""
    license_names = sorted(
        name
        for name in set(tracked.splitlines())
        if Path(name).name.lower()
        in {"license", "license.txt", "copying", "notice", "notice.txt"}
    )
    if license_names:
        raise ValueError(f"{label} pinned source unexpectedly gained license text")
    return {
        "repository": repository,
        "path": str(root),
        "commit": actual_commit,
        "tracked_dirty": False,
        "tracked_license_files": license_names,
        "files": file_records,
    }


def verify_source(
    source_root: Path,
    space_root: Path,
) -> dict[str, Any]:
    return {
        "github": _verify_git_source(
            source_root,
            repository=MODEL_REPO_URL,
            commit=MODEL_SOURCE_COMMIT,
            expected_files=SOURCE_FILES,
            label="OmniAID GitHub",
        ),
        "space": _verify_git_source(
            space_root,
            repository=MODEL_SPACE_URL,
            commit=MODEL_SPACE_COMMIT,
            expected_files=SPACE_SOURCE_FILES,
            label="OmniAID Space",
        ),
        "inference_source": (
            "space/model/omniaid-dino.py automatic-router forward"
        ),
    }


def verify_assets(
    checkpoint_path: Path,
    omniaid_config_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    import torch

    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("OmniAID checkpoint byte size changed")
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "OmniAID-DINO v2 checkpoint",
    )
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(
        checkpoint_path
    )
    if unsafe != CHECKPOINT["unsafe_globals_allowlisted"]:
        raise ValueError(
            "OmniAID checkpoint unsafe-global census changed: "
            f"{unsafe!r}"
        )
    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("OmniAID checkpoint top level is not a mapping")
    if list(payload) != CHECKPOINT["top_level_keys"]:
        raise ValueError("OmniAID checkpoint top-level keys changed")
    if payload.get("epoch") != CHECKPOINT["epoch"]:
        raise ValueError("OmniAID checkpoint epoch changed")
    if not isinstance(payload.get("args"), argparse.Namespace):
        raise ValueError("OmniAID checkpoint args type changed")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("OmniAID checkpoint model state is not a mapping")
    if len(state) != int(CHECKPOINT["tensor_count"]):
        raise ValueError("OmniAID checkpoint tensor count changed")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("OmniAID checkpoint contains non-tensor values")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("OmniAID checkpoint contains non-FP32 tensors")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("OmniAID checkpoint contains non-finite tensors")
    elements = sum(value.numel() for value in state.values())
    if elements != int(CHECKPOINT["state_elements"]):
        raise ValueError("OmniAID checkpoint element count changed")
    key_digest = hashlib.sha256(
        "\n".join(state.keys()).encode("utf-8")
    ).hexdigest()
    if key_digest != CHECKPOINT["ordered_key_sha256"]:
        raise ValueError("OmniAID checkpoint ordered keys changed")
    schema_digest = _state_schema_sha256(state)
    if schema_digest != CHECKPOINT["schema_sha256"]:
        raise ValueError("OmniAID checkpoint schema changed")
    if tuple(state["head.weight"].shape) != (2, FEATURE_DIMENSION):
        raise ValueError("OmniAID head weight shape changed")
    if tuple(state["head.bias"].shape) != (2,):
        raise ValueError("OmniAID head bias shape changed")
    if tuple(state["gating_network.network.0.weight"].shape) != (
        ROUTER_HIDDEN_DIMENSION,
        FEATURE_DIMENSION,
    ):
        raise ValueError("OmniAID router input layer shape changed")
    if tuple(state["gating_network.network.2.weight"].shape) != (
        SEMANTIC_EXPERT_COUNT,
        ROUTER_HIDDEN_DIMENSION,
    ):
        raise ValueError("OmniAID router output layer shape changed")

    if omniaid_config_path.stat().st_size != int(OMNIAID_CONFIG["bytes"]):
        raise ValueError("pinned OmniAID config byte size changed")
    _verify_runtime_file(
        omniaid_config_path,
        str(OMNIAID_CONFIG["sha256"]),
        "pinned OmniAID-DINO v2 config",
    )
    config_payload = json.loads(
        omniaid_config_path.read_text(encoding="utf-8")
    )
    expected_moe = {
        "DINOV3_path": DINO_BASE["model_id"],
        "num_experts": EXPERT_COUNT,
        "rank_per_expert": SVD_RESIDUAL_RANK,
        "moe_router_hidden_dim": ROUTER_HIDDEN_DIMENSION,
        "moe_top_k": SEMANTIC_TOP_K,
        "gradient_checkpointing_enable": False,
    }
    for key, expected in expected_moe.items():
        if config_payload.get(key) != expected:
            raise ValueError(f"pinned OmniAID config field changed: {key}")

    assets = {
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path),
            "serialization_safety": {
                "weights_only": True,
                "mmap": True,
                "unsafe_globals_observed": unsafe,
                "allowlist": ["argparse.Namespace"],
                "arbitrary_code_execution_enabled": False,
            },
            "top_level_type": type(payload).__name__,
            "schema_verified": True,
            "training_args_contract": {
                "img_size": getattr(payload["args"], "img_size", None),
                "model": getattr(payload["args"], "model", None),
                "is_hybrid": getattr(payload["args"], "is_hybrid", None),
                "training_mode": getattr(
                    payload["args"], "training_mode", None
                ),
                "data_path": getattr(payload["args"], "data_path", None),
            },
        },
        "omniaid_config": {
            **OMNIAID_CONFIG,
            "path": str(omniaid_config_path),
            "moe_contract": expected_moe,
        },
        "dinov3_base": DINO_BASE,
    }
    return assets, state, config_payload


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import huggingface_hub
    import torch
    import torchvision
    import transformers
    import PIL

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    expected = {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "numpy": NUMPY_VERSION,
        "pillow": PILLOW_VERSION,
        "huggingface_hub": HUGGINGFACE_HUB_VERSION,
    }
    for key, value in expected.items():
        if versions[key] != value:
            raise ValueError(
                f"OmniAID runtime {key} changed: {versions[key]} != {value}"
            )
    device = torch.device(device_text)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("OmniAID supports only cpu or cuda")
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
    space_root: Path,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from unittest.mock import patch

    vision_config = DINOv3ViTConfig(
        **dict(DINO_BASE["architecture_contract"])
    )
    model_config = types.SimpleNamespace(**dict(config_payload))
    model_config.is_hybrid = True
    model_config.gradient_checkpointing_enable = False
    model_config.image_resolution = MODEL_INPUT_SIZE
    source_path = space_root / "model/omniaid-dino.py"
    spec = importlib.util.spec_from_file_location(
        "claimforge_official_omniaid_dino",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official OmniAID source: {source_path}")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)

    def shape_only_base(
        _class: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        return DINOv3ViTModel(vision_config)

    with torch.device("meta"), patch.object(
        DINOv3ViTModel,
        "from_pretrained",
        new=classmethod(shape_only_base),
    ):
        model = official.OmniAID_DINO(model_config)
    model_state = model.state_dict()
    if list(model_state) != list(state):
        missing = sorted(set(model_state) - set(state))
        unexpected = sorted(set(state) - set(model_state))
        raise ValueError(
            "OmniAID strict model schema mismatch before load: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    load_result = model.load_state_dict(
        state,
        strict=True,
        assign=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError("OmniAID strict load unexpectedly reported drift")
    head_dimension = FEATURE_DIMENSION // 16
    inv_freq = (
        1.0
        / float(DINO_BASE["architecture_contract"]["rope_theta"])
        ** torch.arange(
            0,
            1,
            4 / head_dimension,
            dtype=torch.float32,
        )
    )
    model.feature_extractor.rope_embeddings.inv_freq = inv_freq.clone()
    model.rope_embeddings.inv_freq = inv_freq.clone()
    model = model.to(device)
    model.eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("OmniAID model retains meta parameters")
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError("OmniAID model retains meta buffers")
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, official.SVDMoeLinear)
    ]
    if len(modules) != SVD_MODULE_COUNT:
        raise ValueError("OmniAID SVD module count changed")
    expected_names = [
        f"layer.{layer}.attention.{projection}"
        for layer in range(24)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    if [name for name, _ in modules] != expected_names:
        raise ValueError("OmniAID SVD module ordering changed")
    for _, module in modules:
        if (
            tuple(module.weight_main.shape) != (1024, 1024)
            or len(module.U_experts) != EXPERT_COUNT
            or len(module.S_experts) != EXPERT_COUNT
            or len(module.V_experts) != EXPERT_COUNT
            or any(tuple(value.shape) != (1024, 1) for value in module.U_experts)
            or any(tuple(value.shape) != (1,) for value in module.S_experts)
            or any(tuple(value.shape) != (1, 1024) for value in module.V_experts)
        ):
            raise ValueError("OmniAID SVD-MoE expert shape changed")
    audit = {
        "constructor": (
            "official_space_inference_class_shape_only_dinov3_base_on_meta"
        ),
        "official_forward_formula": (
            "weight_main plus top2 semantic rank1 experts plus always-on "
            "artifact rank1 expert"
        ),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "state_entries": len(model_state),
        "state_elements": sum(value.numel() for value in model_state.values()),
        "svd_modules": len(modules),
        "svd_module_names": [name for name, _ in modules],
        "main_rank": SVD_FROZEN_RANK,
        "rank_per_expert": SVD_RESIDUAL_RANK,
        "experts": EXPERT_COUNT,
        "semantic_experts": SEMANTIC_EXPERT_COUNT,
        "semantic_top_k": SEMANTIC_TOP_K,
        "artifact_expert_index": ARTIFACT_EXPERT_INDEX,
        "artifact_expert_always_on": True,
        "feature_dimension": FEATURE_DIMENSION,
        "head_weight_shape": list(model.head.weight.shape),
        "head_bias_shape": list(model.head.bias.shape),
        "nonpersistent_rope_buffers_materialized": [
            "feature_extractor.rope_embeddings.inv_freq",
            "rope_embeddings.inv_freq",
        ],
        "rope_inv_freq_shape": list(inv_freq.shape),
        "base_weights_downloaded": False,
        "checkpoint_contains_complete_feature_extractor_and_moe_state": True,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "eval_mode": not model.training,
    }
    return model, audit


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from torchvision import transforms

    try:
        with Image.open(path) as opened:
            decoded = opened.convert("RGB")
            decoded.load()
    except Exception as exc:
        raise ValueError(f"Pillow failed to decode OmniAID input: {path}") from exc
    rgb = np.ascontiguousarray(np.asarray(decoded), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("OmniAID Pillow decode is not uint8 RGB")
    resize = transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
    resized_image = resize(decoded)
    resized = np.ascontiguousarray(
        np.asarray(resized_image),
        dtype=np.uint8,
    )
    tensor_value = transforms.Normalize(
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
    )(transforms.ToTensor()(resized_image))
    tensor = np.ascontiguousarray(
        tensor_value.numpy(),
        dtype=np.float32,
    )
    if tensor.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("OmniAID normalized tensor shape changed")
    if not np.isfinite(tensor).all():
        raise ValueError("OmniAID normalized tensor is non-finite")
    audit = {
        "decode": "PIL.Image.open_then_convert_RGB",
        "exif_transpose": False,
        "native_shape_hwc": [int(value) for value in rgb.shape],
        "native_width": int(rgb.shape[1]),
        "native_height": int(rgb.shape[0]),
        "decoded_rgb_sha256": _array_sha256(rgb),
        "resize": {
            "output_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "implementation": "torchvision.transforms.Resize_list_hw",
            "interpolation": "PIL_BILINEAR_default",
            "antialias": True,
            "preserve_aspect_ratio": False,
            "crop": None,
            "face_alignment": False,
        },
        "resized_rgb_sha256": _array_sha256(resized),
        "to_tensor": "torchvision.transforms.ToTensor_uint8_divide_255_CHW",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_sha256": _array_sha256(tensor),
    }
    return tensor, audit


def _float32_probability(logits: Any) -> Any:
    import torch

    if logits.dtype != torch.float32:
        raise ValueError("OmniAID logits are not float32")
    return torch.softmax(logits, dim=1)[:, 1]


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], int | None, float]:
    import torch
    from torch.nn import functional

    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    captured: dict[str, Any] = {}

    def capture_head(_module: Any, inputs: Any, output: Any) -> None:
        captured["pooler_output"] = inputs[0].detach()
        captured["class_logits"] = output.detach()

    def capture_router_feature(_module: Any, _inputs: Any, output: Any) -> None:
        captured["routing_feature"] = output.pooler_output.detach()

    def capture_router(_module: Any, _inputs: Any, output: Any) -> None:
        captured["semantic_top_k_indices"] = output[
            "top_k_indices"
        ].detach()
        captured["semantic_top_k_gates"] = output["top_k_gates"].detach()

    hooks = (
        model.head.register_forward_hook(capture_head),
        model.feature_extractor.register_forward_hook(
            capture_router_feature
        ),
        model.gating_network.register_forward_hook(capture_router),
    )
    peak: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(tensor, manual_weights=None)
    finally:
        for hook in hooks:
            hook.remove()
    logits = captured.get("class_logits")
    feature = captured.get("pooler_output")
    routing_feature = captured.get("routing_feature")
    top_k_indices = captured.get("semantic_top_k_indices")
    top_k_gates = captured.get("semantic_top_k_gates")
    if any(
        value is None
        for value in (
            logits,
            feature,
            routing_feature,
            top_k_indices,
            top_k_gates,
        )
    ):
        raise ValueError("OmniAID official forward hooks did not all fire")
    final_gates = output.get("final_gates")
    probability = output.get("prob")
    if final_gates is None or probability is None:
        raise ValueError("OmniAID official forward output keys changed")
    with torch.inference_mode():
        replay = functional.linear(
            feature,
            model.head.weight,
            model.head.bias,
        )
        probability_replay = _float32_probability(logits)
        margin = logits[:, 1] - logits[:, 0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not torch.equal(logits, replay):
        raise ValueError("OmniAID head replay differs from model logits")
    if not torch.equal(probability, probability_replay):
        raise ValueError("OmniAID official probability differs from replay")
    feature_array = np.ascontiguousarray(
        feature[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    logits_array = np.ascontiguousarray(
        logits[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    routing_array = np.ascontiguousarray(
        routing_feature[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    top_k_indices_array = np.ascontiguousarray(
        top_k_indices[0].detach().cpu().numpy(),
        dtype=np.int64,
    )
    top_k_gates_array = np.ascontiguousarray(
        top_k_gates[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    final_gates_array = np.ascontiguousarray(
        final_gates[0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    if (
        feature_array.shape != (FEATURE_DIMENSION,)
        or logits_array.shape != (CLASS_COUNT,)
        or routing_array.shape != (FEATURE_DIMENSION,)
        or top_k_indices_array.shape != (SEMANTIC_TOP_K,)
        or top_k_gates_array.shape != (SEMANTIC_TOP_K,)
        or final_gates_array.shape != (EXPERT_COUNT,)
        or not np.isfinite(feature_array).all()
        or not np.isfinite(logits_array).all()
        or not np.isfinite(routing_array).all()
        or not np.isfinite(top_k_gates_array).all()
        or not np.isfinite(final_gates_array).all()
    ):
        raise ValueError("OmniAID output arrays violate frozen contract")
    if (
        len(set(int(value) for value in top_k_indices_array)) != SEMANTIC_TOP_K
        or np.any(top_k_indices_array < 0)
        or np.any(top_k_indices_array >= SEMANTIC_EXPERT_COUNT)
        or np.any(top_k_gates_array <= 0.0)
        or not np.isclose(top_k_gates_array.sum(), 1.0, atol=1e-6, rtol=0.0)
        or final_gates_array[ARTIFACT_EXPERT_INDEX] != 1.0
        or np.count_nonzero(final_gates_array[:SEMANTIC_EXPERT_COUNT])
        != SEMANTIC_TOP_K
        or not np.isclose(
            final_gates_array[:SEMANTIC_EXPERT_COUNT].sum(),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        or not np.isclose(final_gates_array.sum(), 2.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("OmniAID automatic-router gate invariant changed")
    scattered = np.zeros(EXPERT_COUNT, dtype=np.float32)
    scattered[top_k_indices_array] = top_k_gates_array
    scattered[ARTIFACT_EXPERT_INDEX] = np.float32(1.0)
    if not np.array_equal(scattered, final_gates_array):
        raise ValueError("OmniAID final expert gates do not replay exactly")
    score = float(probability[0].item())
    raw_margin = float(margin[0].item())
    if not 0.0 <= score <= 1.0 or not math.isfinite(raw_margin):
        raise ValueError("OmniAID score is invalid")
    scoring = {
        "class_logits": [float(value) for value in logits_array.tolist()],
        "raw_logit_margin": raw_margin,
        "fake_probability": score,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": SCORE_SEMANTICS,
        "routing_mode": "Auto (Router)",
        "semantic_expert_names": [
            "Human",
            "Animal",
            "Object",
            "Scene",
            "Anime",
        ],
        "artifact_expert_name": "Artifact",
        "semantic_top_k_indices": [
            int(value) for value in top_k_indices_array.tolist()
        ],
        "semantic_top_k_gates": [
            float(value) for value in top_k_gates_array.tolist()
        ],
        "final_expert_gates": [
            float(value) for value in final_gates_array.tolist()
        ],
        "semantic_gate_sum": float(
            final_gates_array[:SEMANTIC_EXPERT_COUNT].sum()
        ),
        "final_gate_sum": float(final_gates_array.sum()),
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
            "router_scatter_exact": True,
        },
    }
    artifacts = {
        "pooler_output": feature_array,
        "class_logits": logits_array,
        "routing_feature": routing_array,
        "semantic_top_k_indices": top_k_indices_array,
        "semantic_top_k_gates": top_k_gates_array,
        "final_gates": final_gates_array,
    }
    return scoring, artifacts, peak, latency_ms


def validate_runtime_golden(
    model: Any,
    device: Any,
    space_root: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for frozen in GOLDEN_CASES:
        path = space_root / str(frozen["path"])
        _verify_runtime_file(path, str(frozen["sha256"]), "OmniAID fixture")
        image, preprocess = preprocess_image(path)
        for key in (
            "decoded_rgb_sha256",
            "resized_rgb_sha256",
            "tensor_sha256",
        ):
            if preprocess.get(key) != frozen[key]:
                raise ValueError(
                    f"OmniAID fixture preprocessing drifted: {key}"
                )
        first, arrays_a, _, _ = infer_one(model, device, image)
        second, arrays_b, _, _ = infer_one(model, device, image)
        for key in arrays_a:
            if not np.array_equal(arrays_a[key], arrays_b[key]):
                raise ValueError(
                    f"OmniAID fixture {key} is not repeatable"
                )
        if first != second:
            raise ValueError("OmniAID fixture scores are not repeatable")
        service_diff = abs(
            float(first["ai_score"])
            - float(frozen["official_service_probability"])
        )
        runtime_logit_diff: float | None = None
        runtime_probability_diff: float | None = None
        runtime_gate_diff: float | None = None
        if device.type == "cuda":
            runtime_logit_diff = float(
                np.max(
                    np.abs(
                        arrays_a["class_logits"].astype(np.float64)
                        - np.asarray(frozen["cuda_logits"], dtype=np.float64)
                    )
                )
            )
            runtime_probability_diff = abs(
                float(first["ai_score"]) - float(frozen["cuda_probability"])
            )
            runtime_gate_diff = float(
                np.max(
                    np.abs(
                        arrays_a["final_gates"].astype(np.float64)
                        - np.asarray(
                            frozen["cuda_final_gates"],
                            dtype=np.float64,
                        )
                    )
                )
            )
            if max(
                runtime_logit_diff,
                runtime_probability_diff,
                runtime_gate_diff,
            ) > GOLDEN_RUNTIME_ABS_TOLERANCE:
                raise ValueError("OmniAID CUDA runtime fixture changed")
        if service_diff > GOLDEN_SERVICE_ABS_TOLERANCE:
            raise ValueError(
                "OmniAID output differs from observed official Space service"
            )
        cases.append(
            {
                "path": str(frozen["path"]),
                "input_sha256": str(frozen["sha256"]),
                "preprocess": preprocess,
                "logits": [
                    float(value)
                    for value in arrays_a["class_logits"].tolist()
                ],
                "fake_probability": float(first["ai_score"]),
                "final_expert_gates": first["final_expert_gates"],
                "array_sha256": {
                    key: _array_sha256(value)
                    for key, value in arrays_a.items()
                },
                "repeat_all_arrays_exact": True,
                "observed_official_service_probability": frozen[
                    "official_service_probability"
                ],
                "official_service_probability_abs_diff": service_diff,
                "frozen_runtime_logit_max_abs_diff": runtime_logit_diff,
                "frozen_runtime_probability_abs_diff": (
                    runtime_probability_diff
                ),
                "frozen_runtime_gate_max_abs_diff": runtime_gate_diff,
            }
        )
    return {
        "status": "passed",
        "kind": (
            "official_space_examples_plus_observed_official_service_oracle_"
            "not_author_published_numeric_golden"
        ),
        "device_family": device.type,
        "runtime_abs_tolerance": GOLDEN_RUNTIME_ABS_TOLERANCE,
        "official_service_abs_tolerance": GOLDEN_SERVICE_ABS_TOLERANCE,
        "official_service_observed_at": "2026-07-25",
        "cases": cases,
        "mouse_model_scores_computed": 0,
    }


ARTIFACT_SCHEMA = {
    "pooler_output": ((FEATURE_DIMENSION,), np.float32),
    "class_logits": ((CLASS_COUNT,), np.float32),
    "routing_feature": ((FEATURE_DIMENSION,), np.float32),
    "semantic_top_k_indices": ((SEMANTIC_TOP_K,), np.int64),
    "semantic_top_k_gates": ((SEMANTIC_TOP_K,), np.float32),
    "final_gates": ((EXPERT_COUNT,), np.float32),
}


def _atomic_save_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(arrays) != set(ARTIFACT_SCHEMA):
        raise ValueError("OmniAID artifact arrays have unexpected keys")
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
                **{
                    key: np.ascontiguousarray(value)
                    for key, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(ARTIFACT_SCHEMA):
                raise ValueError("OmniAID artifact keys changed")
            arrays = {
                key: np.ascontiguousarray(payload[key])
                for key in ARTIFACT_SCHEMA
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot safely load OmniAID artifact: {path}") from exc
    for key, (shape, dtype) in ARTIFACT_SCHEMA.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"OmniAID artifact {key} violates shape/dtype contract"
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"OmniAID artifact {key} is non-finite")
    return arrays


def _artifact_path(run_dir: Path, sample_id: str) -> Path:
    _safe_component(sample_id, label="sample-id")
    artifact_dir = (run_dir / "artifacts").resolve()
    path = (artifact_dir / f"{sample_id}.npz").resolve()
    if path.parent != artifact_dir:
        raise ValueError("OmniAID artifact path escapes artifact directory")
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
            raise ValueError(f"{path} invents OmniAID T2 fields: {present}")
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
            raise ValueError(f"OmniAID resume identity field {key} changed")
    if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
        raise ValueError("only valid successful OmniAID rows may be skipped")
    score = _finite_number(row.get("ai_score"), "resume ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("resume OmniAID score falls outside [0,1]")
    if row.get("score") != score or row.get("probability") != score:
        raise ValueError("resume OmniAID score aliases changed")
    if row.get("fake_probability") != score:
        raise ValueError("resume OmniAID fake probability changed")
    if row.get("classification_decision") is not (score > 0.5):
        raise ValueError("resume OmniAID strict decision changed")
    input_path = _anchored(Path(str(expected["input_path"])), repo_root)
    _verify_runtime_file(
        input_path,
        str(expected["input_sha256"]),
        "resume OmniAID input",
    )
    _, replay_preprocess = preprocess_image(input_path)
    if row.get("preprocess") != replay_preprocess:
        raise ValueError("resume OmniAID preprocessing does not replay")
    artifact_value = row.get("artifact_path")
    if not isinstance(artifact_value, str):
        raise ValueError("resume OmniAID artifact path missing")
    artifact_path = _anchored(Path(artifact_value), repo_root)
    expected_path = _artifact_path(run_dir, str(row["id"]))
    if artifact_path != expected_path:
        raise ValueError("resume OmniAID artifact path changed or escapes")
    _verify_runtime_file(
        artifact_path,
        str(row.get("artifact_sha256")),
        "resume OmniAID artifact",
    )
    arrays = _load_artifact(artifact_path)
    observed_hashes = {
        key: _array_sha256(value) for key, value in arrays.items()
    }
    if (
        observed_hashes != row.get("artifact_array_sha256")
        or observed_hashes["pooler_output"]
        != row.get("feature_array_sha256")
        or observed_hashes["class_logits"]
        != row.get("class_logits_array_sha256")
        or [float(value) for value in arrays["class_logits"].tolist()]
        != row.get("class_logits")
        or [float(value) for value in arrays["final_gates"].tolist()]
        != row.get("final_expert_gates")
    ):
        raise ValueError("resume OmniAID artifact content changed")
    with torch.inference_mode():
        replay = functional.linear(
            torch.from_numpy(arrays["pooler_output"]).to(device).unsqueeze(0),
            model.head.weight,
            model.head.bias,
        )[0]
        replay_probability = _float32_probability(
            replay.unsqueeze(0)
        )[0]
    if not torch.equal(
        replay.detach().cpu(),
        torch.from_numpy(arrays["class_logits"]),
    ):
        raise ValueError("resume OmniAID head replay changed")
    if float(replay_probability.item()) != score:
        raise ValueError("resume OmniAID softmax replay changed")
    _reject_t2_payload(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--space-root", type=Path, default=DEFAULT_SPACE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--omniaid-config",
        type=Path,
        default=DEFAULT_OMNIAID_CONFIG,
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
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source_root = _anchored(args.source_root, repo_root)
    space_root = _anchored(args.space_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    omniaid_config_path = _anchored(args.omniaid_config, repo_root)
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
    source = verify_source(source_root, space_root)
    device, runtime = configure_runtime(args.device)
    assets, state, config_payload = verify_assets(
        checkpoint_path,
        omniaid_config_path,
    )
    model, model_audit = _build_model(
        state,
        config_payload,
        device,
        space_root,
    )
    del state
    golden = validate_runtime_golden(model, device, space_root)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model": MODEL_NAME,
                    "source_commit": source["github"]["commit"],
                    "space_commit": source["space"]["commit"],
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

    from eval.opensource.omniaid_metrics import summarize_omniaid_results

    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root.resolve():
        raise ValueError("run-id escapes OmniAID results directory")
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
        "omniaid_config": OMNIAID_CONFIG,
        "dinov3_base": DINO_BASE,
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "PIL.Image.open_convert_RGB_no_EXIF_transpose",
            "resize": (
                "torchvision_Resize_list_448x448_PIL_BILINEAR_"
                "no_aspect_preservation"
            ),
            "to_tensor": "torchvision_ToTensor_float32_divide_255",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "face_alignment": False,
            "router_mode": "Auto (Router)",
            "manual_weights": None,
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
                "OmniAID resume requires run_manifest and expected_inputs"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            prior_manifest.get("schema_version")
            != "omniaid_detection_run_manifest_v1"
            or prior_manifest.get("run_id") != args.run_id
            or prior_manifest.get("config") != config
            or prior_manifest.get("config_fingerprint") != fingerprint
            or read_jsonl(expected_path) != selected
        ):
            raise ValueError("OmniAID resume identity/config changed")
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
        "schema_version": "omniaid_detection_run_manifest_v1",
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
            scoring, arrays, peak, latency = infer_one(
                model,
                device,
                image,
            )
            artifact_path = _artifact_path(run_dir, sample_id)
            _atomic_save_artifact(artifact_path, arrays)
            persisted = _load_artifact(artifact_path)
            for key, value in arrays.items():
                if not np.array_equal(value, persisted[key]):
                    raise ValueError(
                        f"OmniAID persisted {key} differs"
                    )
            relative_artifact = repo_relative(artifact_path, repo_root)
            array_hashes = {
                key: _array_sha256(value)
                for key, value in arrays.items()
            }
            result = {
                **identity,
                "status": "ok",
                "valid_for_metrics": True,
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_ms,
                "artifact_path": relative_artifact,
                "artifact_sha256": sha256_file(artifact_path),
                "artifact_keys": list(ARTIFACT_SCHEMA),
                "artifact_paths": {"omniaid_npz": relative_artifact},
                "artifact_array_sha256": array_hashes,
                "feature_shape": [FEATURE_DIMENSION],
                "feature_dtype": "float32",
                "feature_semantics": FEATURE_SEMANTICS,
                "feature_array_sha256": array_hashes["pooler_output"],
                "class_logits_shape": [CLASS_COUNT],
                "class_logits_dtype": "float32",
                "class_logits_array_sha256": array_hashes["class_logits"],
                "routing_feature_shape": [FEATURE_DIMENSION],
                "routing_feature_dtype": "float32",
                "routing_feature_semantics": ROUTING_FEATURE_SEMANTICS,
                "semantic_top_k_indices_shape": [SEMANTIC_TOP_K],
                "semantic_top_k_gates_shape": [SEMANTIC_TOP_K],
                "final_gates_shape": [EXPERT_COUNT],
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
    summary = summarize_omniaid_results(
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
