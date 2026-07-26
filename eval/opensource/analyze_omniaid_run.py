#!/usr/bin/env python3
"""Independently audit a frozen OmniAID whole-image detection run.

The analyzer treats the runner's JSON, NPZ artifacts, scores, and summary as
untrusted. It independently verifies both pinned source trees and the safely
loaded checkpoint, reconstructs the official Space DINOv3/MoE graph on meta,
decodes and preprocesses every selected image again, performs fresh full-model
forwards, verifies all six NPZ arrays, replays the head, softmax, automatic
router and final gate scatter, and recomputes all paired metrics.
"""

from __future__ import annotations

import argparse
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
import types
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.omniaid_metrics import summarize_omniaid_results


FROZEN_SOURCE_COMMIT = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
FROZEN_SPACE_COMMIT = "cf99ed518af8b7256854d01994d6e41165553bb3"
FROZEN_SOURCE_FILES = {
    "README.md": "cbf0aee17e5da907019703361c444becffaf76e7ecae96f20cec6b4ccba4e6b7",
    "requirements.txt": (
        "f531378e8f4868c71e673e9ec6d83186e6b04ae62f7f89207c6706c46da7ddb8"
    ),
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
FROZEN_SPACE_FILES = {
    "README.md": "344d460d746ac6531ba2ac80b69edca66f94d75f916e92f6db4b8dc0c1fb324e",
    "app.py": "4c1ac7b4eb1850beb97cd4db02105b4034052e3e8a5a96e8c278305561b5f8f2",
    "requirements.txt": (
        "f531378e8f4868c71e673e9ec6d83186e6b04ae62f7f89207c6706c46da7ddb8"
    ),
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
FROZEN_CHECKPOINT_BYTES = 3_238_483_725
FROZEN_CHECKPOINT_SHA256 = (
    "8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9"
)
FROZEN_CHECKPOINT_TENSORS = 2_852
FROZEN_CHECKPOINT_ELEMENTS = 808_835_239
FROZEN_ORDERED_KEY_SHA256 = (
    "d894e539c44bfd3b036413db0e5c91d7de75552df5af2153ca4a83bc40e7d788"
)
FROZEN_SCHEMA_SHA256 = (
    "1b5a03a08369fa7dc5034b1b9aa8a4757295386afd3c91f093ef41b6e2c9b67d"
)
FROZEN_CONFIG_BYTES = 696
FROZEN_CONFIG_SHA256 = (
    "d97ded19543ca9459a86eddd4c0f08a8476dcd013a50f3bf81c4649f67536719"
)
FROZEN_PREPROCESS_PROFILE = "official_omniaid_space_dino_v2_auto_router_448_v1"
FROZEN_SCORE_SEMANTICS = "official_float32_softmax_class1_probability_higher_is_fake"
FROZEN_CHECKPOINT_ID = "official_omniaid_dino_v2_mirage_train"
FROZEN_MODEL_SLUG = "omniaid_dino_v2_mirage_auto_router"
FEATURE_DIMENSION = 1024
CLASS_COUNT = 2
INPUT_SIZE = 448
SVD_MODULES = 96
SVD_FROZEN_RANK = 1023
SVD_RESIDUAL_RANK = 1
EXPERT_COUNT = 6
SEMANTIC_EXPERT_COUNT = 5
SEMANTIC_TOP_K = 2
ARTIFACT_EXPERT_INDEX = 5
ROUTER_HIDDEN_DIMENSION = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLD = 0.5
MODEL_SEED = 20260725
CPU_THREADS = 16

EXPECTED_RUNTIME = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "transformers": "4.57.3",
    "numpy": "2.2.6",
    "pillow": "12.0.0",
    "huggingface_hub": "0.36.0",
}

DEFAULT_RESULTS_DIR = Path("results/opensource/omniaid")
DEFAULT_RUN_ID = "omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725"
DEFAULT_SOURCE_ROOT = Path("/root/.cache/claimforge/third_party/omniaid-40749406")
DEFAULT_SPACE_ROOT = Path("/root/.cache/claimforge/third_party/omniaid-space-cf99ed51")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/models/omniaid/checkpoint_omniaid_dino_v2.pth"
)
DEFAULT_CONFIG = Path(
    "/root/.cache/claimforge/models/omniaid/config_omniaid_dino_v2.json"
)

DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DINO_ARCHITECTURE = {
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
}

FEATURE_SEMANTICS = "official_moe_cls_pooler_output_before_omniaid_head"
ROUTING_FEATURE_SEMANTICS = (
    "official_frozen_dinov3_feature_extractor_pooler_output_for_router"
)
ARTIFACT_SCHEMA = {
    "pooler_output": ((FEATURE_DIMENSION,), np.float32),
    "class_logits": ((CLASS_COUNT,), np.float32),
    "routing_feature": ((FEATURE_DIMENSION,), np.float32),
    "semantic_top_k_indices": ((SEMANTIC_TOP_K,), np.int64),
    "semantic_top_k_gates": ((SEMANTIC_TOP_K,), np.float32),
    "final_gates": ((EXPERT_COUNT,), np.float32),
}
CANONICAL_RELEASE = {
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


def _verify_canonical_release(value: Any) -> dict[str, Any]:
    """Verify the frozen release identity while allowing audited metadata.

    The runner embeds the complete canonical release manifest in its config.
    ``CANONICAL_RELEASE`` intentionally contains only the immutable identity
    fields that this analyzer freezes, so equality against the whole manifest
    would reject valid runs merely because they preserve additional provenance.
    """

    record = _require_mapping(value, "config release")
    for key, expected in CANONICAL_RELEASE.items():
        if record.get(key) != expected:
            raise ValueError(
                f"OmniAID manifest canonical release {key} mismatch"
            )
    return record


FORBIDDEN_T2_KEYS = frozenset(
    {
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _verify_file(path: Path, digest: Any, label: str) -> None:
    expected = _require_sha256(digest, f"{label} digest")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _reject_t2(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        present = sorted(FORBIDDEN_T2_KEYS.intersection(value))
        if present:
            raise ValueError(f"{path} invents OmniAID T2 fields: {present}")
        for key, child in value.items():
            _reject_t2(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_t2(child, path=f"{path}[{index}]")


def _verify_git_source(
    root: Path,
    *,
    expected_commit: str,
    expected_files: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"independent {label} source commit mismatch")
    dirty = _git_value(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"independent {label} source tree is dirty")
    records: dict[str, Any] = {}
    for relative, digest in expected_files.items():
        path = root / relative
        _verify_file(path, digest, f"{label} source {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    tracked = set((_git_value(root, "ls-files") or "").splitlines())
    licenses = sorted(
        name
        for name in tracked
        if Path(name).name.lower()
        in {"license", "license.txt", "copying", "notice", "notice.txt"}
    )
    if licenses:
        raise ValueError(f"independent {label} license census changed")
    return {
        "commit": commit,
        "tracked_dirty": False,
        "tracked_license_files": [],
        "files": records,
    }


def _verify_source(source_root: Path, space_root: Path) -> dict[str, Any]:
    return {
        "github": _verify_git_source(
            source_root,
            expected_commit=FROZEN_SOURCE_COMMIT,
            expected_files=FROZEN_SOURCE_FILES,
            label="OmniAID GitHub",
        ),
        "space": _verify_git_source(
            space_root,
            expected_commit=FROZEN_SPACE_COMMIT,
            expected_files=FROZEN_SPACE_FILES,
            label="OmniAID Space",
        ),
        "inference_source": "space/model/omniaid-dino.py automatic-router forward",
    }


def _schema_sha256(state: Mapping[str, Any]) -> str:
    canonical = "\n".join(
        f"{key}\t{tuple(value.shape)}\t{value.dtype}\t{value.numel()}"
        for key, value in state.items()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_adapter_contract(
    config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    recorded = _require_mapping(config.get("adapter"), "config adapter")
    expected_relatives = {
        "eval/opensource/run_omniaid.py",
        "eval/opensource/omniaid_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
    }
    if set(recorded) != expected_relatives:
        raise ValueError("OmniAID adapter file census mismatch")
    evidence: dict[str, Any] = {}
    for relative in sorted(expected_relatives):
        row = _require_mapping(recorded.get(relative), f"adapter {relative}")
        path = (repo_root / relative).resolve()
        if path != Path(str(row.get("path"))).resolve():
            raise ValueError(f"OmniAID adapter path mismatch: {relative}")
        _verify_file(path, row.get("sha256"), f"OmniAID adapter {relative}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"OmniAID adapter byte size mismatch: {relative}")
        evidence[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return evidence


def _load_assets(
    checkpoint_path: Path,
    config_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    if checkpoint_path.stat().st_size != FROZEN_CHECKPOINT_BYTES:
        raise ValueError("independent OmniAID checkpoint byte size mismatch")
    _verify_file(
        checkpoint_path,
        FROZEN_CHECKPOINT_SHA256,
        "OmniAID checkpoint",
    )
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    if unsafe != ["argparse.Namespace"]:
        raise ValueError(
            "independent OmniAID unsafe-global census mismatch: " f"{unsafe!r}"
        )
    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("independent OmniAID checkpoint is not a mapping")
    if list(payload) != ["model", "optimizer", "epoch", "scaler", "args"]:
        raise ValueError("independent OmniAID checkpoint top-level keys mismatch")
    if payload.get("epoch") != 0:
        raise ValueError("independent OmniAID checkpoint epoch mismatch")
    if not isinstance(payload.get("args"), argparse.Namespace):
        raise ValueError("independent OmniAID checkpoint args type mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("independent OmniAID model state is not a mapping")
    if len(state) != FROZEN_CHECKPOINT_TENSORS:
        raise ValueError("independent OmniAID tensor count mismatch")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("independent OmniAID state has non-tensor values")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("independent OmniAID state has non-FP32 values")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("independent OmniAID state has non-finite values")
    if sum(value.numel() for value in state.values()) != FROZEN_CHECKPOINT_ELEMENTS:
        raise ValueError("independent OmniAID element count mismatch")
    key_digest = hashlib.sha256("\n".join(state.keys()).encode("utf-8")).hexdigest()
    if key_digest != FROZEN_ORDERED_KEY_SHA256:
        raise ValueError("independent OmniAID ordered-key digest mismatch")
    if _schema_sha256(state) != FROZEN_SCHEMA_SHA256:
        raise ValueError("independent OmniAID state schema mismatch")
    if tuple(state["head.weight"].shape) != (CLASS_COUNT, FEATURE_DIMENSION):
        raise ValueError("independent OmniAID head weight shape mismatch")
    if tuple(state["head.bias"].shape) != (CLASS_COUNT,):
        raise ValueError("independent OmniAID head bias shape mismatch")
    if tuple(state["gating_network.network.0.weight"].shape) != (
        ROUTER_HIDDEN_DIMENSION,
        FEATURE_DIMENSION,
    ):
        raise ValueError("independent OmniAID router input shape mismatch")
    if tuple(state["gating_network.network.2.weight"].shape) != (
        SEMANTIC_EXPERT_COUNT,
        ROUTER_HIDDEN_DIMENSION,
    ):
        raise ValueError("independent OmniAID router output shape mismatch")

    if config_path.stat().st_size != FROZEN_CONFIG_BYTES:
        raise ValueError("independent OmniAID config byte size mismatch")
    _verify_file(
        config_path,
        FROZEN_CONFIG_SHA256,
        "OmniAID-DINO v2 config",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "DINOV3_path": DINO_MODEL_ID,
        "num_experts": EXPERT_COUNT,
        "rank_per_expert": SVD_RESIDUAL_RANK,
        "moe_router_hidden_dim": ROUTER_HIDDEN_DIMENSION,
        "moe_top_k": SEMANTIC_TOP_K,
        "gradient_checkpointing_enable": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"independent OmniAID config mismatch: {key}")
    return (
        state,
        config,
        {
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "checkpoint_bytes": FROZEN_CHECKPOINT_BYTES,
            "checkpoint_tensor_count": FROZEN_CHECKPOINT_TENSORS,
            "checkpoint_state_elements": FROZEN_CHECKPOINT_ELEMENTS,
            "checkpoint_schema_sha256": FROZEN_SCHEMA_SHA256,
            "checkpoint_ordered_key_sha256": FROZEN_ORDERED_KEY_SHA256,
            "unsafe_globals": unsafe,
            "safe_globals_allowlist": ["argparse.Namespace"],
            "weights_only": True,
            "mmap": True,
            "arbitrary_code_execution_enabled": False,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "config_bytes": FROZEN_CONFIG_BYTES,
            "config_contract": expected,
        },
    )


def _configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import huggingface_hub
    import PIL
    import torch
    import torchvision
    import transformers

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    for key, expected in EXPECTED_RUNTIME.items():
        if actual[key] != expected:
            raise ValueError(
                f"independent OmniAID runtime {key} mismatch: "
                f"{actual[key]} != {expected}"
            )
    device = torch.device(device_text)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("independent OmniAID supports cpu/cuda only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("independent OmniAID CUDA is unavailable")
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
        **actual,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "cpu_threads": CPU_THREADS,
        "autocast": False,
        "model_dtype": "float32",
        "batch_size": 1,
        "seed": MODEL_SEED,
    }


def _build_model(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
    space_root: Path,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from unittest.mock import patch

    vision_config = DINOv3ViTConfig(**dict(DINO_ARCHITECTURE))
    model_config = types.SimpleNamespace(**dict(config))
    model_config.is_hybrid = True
    model_config.gradient_checkpointing_enable = False
    model_config.image_resolution = INPUT_SIZE

    source_path = space_root / "model" / "omniaid-dino.py"
    spec = importlib.util.spec_from_file_location(
        "claimforge_independent_omniaid_dino",
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
            "independent OmniAID model/state key mismatch: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    loaded = model.load_state_dict(state, strict=True, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("independent OmniAID strict load failed")

    head_dimension = FEATURE_DIMENSION // 16
    inv_freq = 1.0 / float(DINO_ARCHITECTURE["rope_theta"]) ** torch.arange(
        0,
        1,
        4 / head_dimension,
        dtype=torch.float32,
    )
    model.feature_extractor.rope_embeddings.inv_freq = inv_freq.clone()
    model.rope_embeddings.inv_freq = inv_freq.clone()
    model = model.to(device).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("independent OmniAID model retains meta parameters")
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError("independent OmniAID model retains meta buffers")
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, official.SVDMoeLinear)
    ]
    if len(modules) != SVD_MODULES:
        raise ValueError("independent OmniAID SVD module count mismatch")
    expected_names = [
        f"layer.{layer}.attention.{projection}"
        for layer in range(24)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    if [name for name, _ in modules] != expected_names:
        raise ValueError("independent OmniAID SVD module ordering mismatch")
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
            raise ValueError("independent OmniAID SVD expert shape mismatch")
    return model, {
        "constructor": (
            "official_space_inference_class_shape_only_dinov3_base_on_meta"
        ),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "svd_modules": len(modules),
        "svd_module_names": [name for name, _ in modules],
        "state_entries": len(model_state),
        "state_elements": sum(value.numel() for value in model_state.values()),
        "main_rank": SVD_FROZEN_RANK,
        "rank_per_expert": SVD_RESIDUAL_RANK,
        "experts": EXPERT_COUNT,
        "semantic_experts": SEMANTIC_EXPERT_COUNT,
        "semantic_top_k": SEMANTIC_TOP_K,
        "artifact_expert_index": ARTIFACT_EXPERT_INDEX,
        "artifact_expert_always_on": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "nonpersistent_rope_buffers_materialized": [
            "feature_extractor.rope_embeddings.inv_freq",
            "rope_embeddings.inv_freq",
        ],
        "rope_inv_freq_shape": list(inv_freq.shape),
        "base_weights_downloaded": False,
        "checkpoint_contains_complete_feature_extractor_and_moe_state": True,
        "eval_mode": not model.training,
    }


def _preprocess(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from torchvision import transforms

    try:
        with Image.open(path) as opened:
            decoded = opened.convert("RGB")
            decoded.load()
    except Exception as exc:
        raise ValueError(f"independent Pillow decode failed: {path}") from exc
    rgb = np.ascontiguousarray(np.asarray(decoded), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("independent OmniAID decode contract changed")
    resize = transforms.Resize([INPUT_SIZE, INPUT_SIZE])
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
    if tensor.shape != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError("independent OmniAID tensor shape changed")
    if not np.isfinite(tensor).all():
        raise ValueError("independent OmniAID tensor is non-finite")
    return tensor, {
        "decode": "PIL.Image.open_then_convert_RGB",
        "exif_transpose": False,
        "native_shape_hwc": [int(value) for value in rgb.shape],
        "native_width": int(rgb.shape[1]),
        "native_height": int(rgb.shape[0]),
        "decoded_rgb_sha256": _array_sha256(rgb),
        "resize": {
            "output_wh": [INPUT_SIZE, INPUT_SIZE],
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


def _validate_gate_arrays(
    top_k_indices: np.ndarray,
    top_k_gates: np.ndarray,
    final_gates: np.ndarray,
    *,
    label: str,
) -> None:
    if (
        len(set(int(value) for value in top_k_indices)) != SEMANTIC_TOP_K
        or np.any(top_k_indices < 0)
        or np.any(top_k_indices >= SEMANTIC_EXPERT_COUNT)
        or np.any(top_k_gates <= 0.0)
        or not np.isclose(
            top_k_gates.sum(),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        or final_gates[ARTIFACT_EXPERT_INDEX] != 1.0
        or np.count_nonzero(final_gates[:SEMANTIC_EXPERT_COUNT]) != SEMANTIC_TOP_K
        or not np.isclose(
            final_gates[:SEMANTIC_EXPERT_COUNT].sum(),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        or not np.isclose(final_gates.sum(), 2.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(f"{label} gate invariant mismatch")
    scattered = np.zeros(EXPERT_COUNT, dtype=np.float32)
    scattered[top_k_indices] = top_k_gates
    scattered[ARTIFACT_EXPERT_INDEX] = np.float32(1.0)
    if not np.array_equal(scattered, final_gates):
        raise ValueError(f"{label} gate scatter mismatch")


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(ARTIFACT_SCHEMA):
                raise ValueError("independent OmniAID artifact keys mismatch")
            arrays = {
                key: np.ascontiguousarray(payload[key]) for key in ARTIFACT_SCHEMA
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot safely load OmniAID artifact {path}") from exc
    for key, (shape, dtype) in ARTIFACT_SCHEMA.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"independent OmniAID artifact {key} shape/dtype mismatch")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"independent OmniAID artifact {key} is non-finite")
    _validate_gate_arrays(
        arrays["semantic_top_k_indices"],
        arrays["semantic_top_k_gates"],
        arrays["final_gates"],
        label="independent OmniAID artifact",
    )
    return arrays


def _fresh_forward(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from torch.nn import functional

    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    captured: dict[str, Any] = {}

    def capture_head(_module: Any, inputs: Any, output: Any) -> None:
        captured["pooler_output"] = inputs[0].detach()
        captured["class_logits"] = output.detach()

    def capture_routing_feature(
        _module: Any,
        _inputs: Any,
        output: Any,
    ) -> None:
        captured["routing_feature"] = output.pooler_output.detach()

    def capture_router(_module: Any, _inputs: Any, output: Any) -> None:
        captured["semantic_top_k_indices"] = output["top_k_indices"].detach()
        captured["semantic_top_k_gates"] = output["top_k_gates"].detach()

    hooks = (
        model.head.register_forward_hook(capture_head),
        model.feature_extractor.register_forward_hook(capture_routing_feature),
        model.gating_network.register_forward_hook(capture_router),
    )
    try:
        with torch.inference_mode():
            output = model(tensor, manual_weights=None)
    finally:
        for hook in hooks:
            hook.remove()
    if not isinstance(output, Mapping):
        raise ValueError("independent OmniAID forward output is not a mapping")
    required = (
        "pooler_output",
        "class_logits",
        "routing_feature",
        "semantic_top_k_indices",
        "semantic_top_k_gates",
    )
    if any(captured.get(key) is None for key in required):
        raise ValueError("independent OmniAID forward hooks did not all fire")
    probability = output.get("prob")
    final_gates = output.get("final_gates")
    if probability is None or final_gates is None:
        raise ValueError("independent OmniAID forward output keys mismatch")

    feature = captured["pooler_output"]
    logits = captured["class_logits"]
    routing_feature = captured["routing_feature"]
    with torch.inference_mode():
        head_replay = functional.linear(
            feature,
            model.head.weight,
            model.head.bias,
        )
        probability_replay = torch.softmax(logits, dim=1)[:, 1]
        margin = logits[:, 1] - logits[:, 0]
        router_replay = model.gating_network(routing_feature)
    if not torch.equal(head_replay, logits):
        raise ValueError("independent OmniAID fresh head replay mismatch")
    if not torch.equal(probability, probability_replay):
        raise ValueError("independent OmniAID fresh softmax replay mismatch")

    arrays = {
        "pooler_output": np.ascontiguousarray(
            feature[0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "class_logits": np.ascontiguousarray(
            logits[0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "routing_feature": np.ascontiguousarray(
            routing_feature[0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "semantic_top_k_indices": np.ascontiguousarray(
            captured["semantic_top_k_indices"][0].detach().cpu().numpy(),
            dtype=np.int64,
        ),
        "semantic_top_k_gates": np.ascontiguousarray(
            captured["semantic_top_k_gates"][0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "final_gates": np.ascontiguousarray(
            final_gates[0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
    }
    for key, (shape, dtype) in ARTIFACT_SCHEMA.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"independent OmniAID fresh {key} shape/dtype mismatch")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"independent OmniAID fresh {key} is non-finite")
    _validate_gate_arrays(
        arrays["semantic_top_k_indices"],
        arrays["semantic_top_k_gates"],
        arrays["final_gates"],
        label="independent OmniAID fresh forward",
    )

    replay_indices = np.ascontiguousarray(
        router_replay["top_k_indices"][0].detach().cpu().numpy(),
        dtype=np.int64,
    )
    replay_gates = np.ascontiguousarray(
        router_replay["top_k_gates"][0].detach().cpu().numpy(),
        dtype=np.float32,
    )
    if not np.array_equal(replay_indices, arrays["semantic_top_k_indices"]):
        raise ValueError("independent OmniAID fresh router index replay mismatch")
    if not np.array_equal(replay_gates, arrays["semantic_top_k_gates"]):
        raise ValueError("independent OmniAID fresh router gate replay mismatch")
    score = float(probability_replay[0].item())
    margin_value = float(margin[0].item())
    if not 0.0 <= score <= 1.0 or not math.isfinite(margin_value):
        raise ValueError("independent OmniAID fresh score is invalid")
    return arrays, {
        "score": score,
        "raw_logit_margin": margin_value,
        "head_replay": np.ascontiguousarray(
            head_replay[0].detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "router_replay_indices": replay_indices,
        "router_replay_gates": replay_gates,
    }


def _latest_rows(
    physical: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(physical):
        if not isinstance(row, dict):
            raise ValueError(f"physical result row {index} is not an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {index} has invalid id")
        _reject_t2(row, path=f"result[{index}]")
        latest[row_id] = row
    return latest


def _visibility_by_task(
    expected_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in expected_rows:
        task_id = str(row.get("task_id"))
        kind = str(row.get("kind"))
        if kind not in {"real", "forged"} or kind in by_task.setdefault(
            task_id,
            {},
        ):
            raise ValueError("OmniAID expected pair identity mismatch")
        by_task[task_id][kind] = row
    visibility: dict[str, dict[str, Any]] = {}
    for task_id, pair in by_task.items():
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"OmniAID expected pair is incomplete: {task_id}")
        forged = pair["forged"]
        positive = int(forged.get("gt_positive_pixels", -1))
        if positive <= 0:
            raise ValueError(f"OmniAID forged GT count is invalid: {task_id}")
        visibility[task_id] = {
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
                "model_input_wh": [INPUT_SIZE, INPUT_SIZE],
                "resize": ("torchvision_PIL_bilinear_direct_aspect_ratio_distortion"),
                "crop": None,
            },
        }
    return visibility


def _expected_identity(
    expected: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: Path,
    visibility: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(expected["sample_id"])
    path = _anchored(Path(str(expected["canonical_path"])), repo_root)
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": str(expected["task_id"]),
        "pair_rank": int(expected["pair_rank"]),
        "rank": int(expected["rank"]),
        "kind": str(expected["kind"]),
        "label": int(expected["label"]),
        "domain": str(expected["domain"]),
        "input_path": str(path.relative_to(repo_root)),
        "input_sha256": str(expected["canonical_sha256"]),
        "input_width": int(expected["width"]),
        "input_height": int(expected["height"]),
        "model": "OmniAID",
        "model_slug": FROZEN_MODEL_SLUG,
        "checkpoint_id": FROZEN_CHECKPOINT_ID,
        "preprocess_profile": FROZEN_PREPROCESS_PROFILE,
        "score_semantics": FROZEN_SCORE_SEMANTICS,
        "classification_threshold": THRESHOLD,
        "classification_threshold_operator": ">",
        "config_fingerprint": manifest["config_fingerprint"],
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": visibility["edit_visibility_evidence"],
        "valid_for_t1": True,
        "valid_for_t2": False,
    }


def _audit_run(
    *,
    repo_root: Path,
    run_dir: Path,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    device_text: str,
    output_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for path in (manifest_path, results_path, expected_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing OmniAID run file: {path}")
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "OmniAID manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "OmniAID summary",
    )
    if manifest.get("schema_version") != "omniaid_detection_run_manifest_v1":
        raise ValueError("OmniAID manifest schema mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("OmniAID manifest is not complete")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("OmniAID manifest run_id mismatch")
    config_record = _require_mapping(manifest.get("config"), "config")
    if _fingerprint(config_record) != manifest.get("config_fingerprint"):
        raise ValueError("OmniAID manifest config fingerprint mismatch")
    if config_record.get("source_commit") != FROZEN_SOURCE_COMMIT:
        raise ValueError("OmniAID manifest source pin mismatch")
    if config_record.get("model") != "OmniAID":
        raise ValueError("OmniAID manifest model mismatch")
    if config_record.get("model_slug") != FROZEN_MODEL_SLUG:
        raise ValueError("OmniAID manifest model slug mismatch")
    if config_record.get("preprocess_profile") != FROZEN_PREPROCESS_PROFILE:
        raise ValueError("OmniAID manifest preprocess profile mismatch")
    if config_record.get("score_semantics") != FROZEN_SCORE_SEMANTICS:
        raise ValueError("OmniAID manifest score semantics mismatch")
    if config_record.get("classification_threshold") != THRESHOLD:
        raise ValueError("OmniAID manifest threshold mismatch")
    if config_record.get("classification_threshold_operator") != ">":
        raise ValueError("OmniAID manifest threshold operator mismatch")
    if (
        config_record.get("checkpoint_and_protocol_frozen_before_mouse_scores")
        is not True
    ):
        raise ValueError("OmniAID manifest lacks pre-score freeze")
    _verify_canonical_release(config_record.get("release"))
    checkpoint_record = _require_mapping(
        config_record.get("checkpoint"),
        "config checkpoint",
    )
    for key, expected_value in {
        "id": FROZEN_CHECKPOINT_ID,
        "bytes": FROZEN_CHECKPOINT_BYTES,
        "sha256": FROZEN_CHECKPOINT_SHA256,
        "tensor_count": FROZEN_CHECKPOINT_TENSORS,
        "state_elements": FROZEN_CHECKPOINT_ELEMENTS,
        "ordered_key_sha256": FROZEN_ORDERED_KEY_SHA256,
        "schema_sha256": FROZEN_SCHEMA_SHA256,
        "unsafe_globals_allowlisted": ["argparse.Namespace"],
    }.items():
        if checkpoint_record.get(key) != expected_value:
            raise ValueError(f"OmniAID manifest checkpoint {key} mismatch")
    config_asset_record = _require_mapping(
        config_record.get("omniaid_config"),
        "config OmniAID config asset",
    )
    if (
        config_asset_record.get("bytes") != FROZEN_CONFIG_BYTES
        or config_asset_record.get("sha256") != FROZEN_CONFIG_SHA256
    ):
        raise ValueError("OmniAID manifest config asset mismatch")
    dino_record = _require_mapping(
        config_record.get("dinov3_base"),
        "config DINOv3 base",
    )
    if (
        dino_record.get("model_id") != DINO_MODEL_ID
        or dino_record.get("revision") != DINO_REVISION
        or dino_record.get("architecture_contract") != DINO_ARCHITECTURE
    ):
        raise ValueError("OmniAID manifest DINOv3 architecture mismatch")
    source_record = _require_mapping(
        config_record.get("source"),
        "config source",
    )
    if (
        _require_mapping(source_record.get("github"), "config GitHub source").get(
            "commit"
        )
        != FROZEN_SOURCE_COMMIT
        or _require_mapping(source_record.get("space"), "config Space source").get(
            "commit"
        )
        != FROZEN_SPACE_COMMIT
    ):
        raise ValueError("OmniAID manifest source records mismatch")
    preprocess_contract = _require_mapping(
        config_record.get("preprocess_contract"),
        "config preprocess contract",
    )
    if preprocess_contract != {
        "decode": "PIL.Image.open_convert_RGB_no_EXIF_transpose",
        "resize": (
            "torchvision_Resize_list_448x448_PIL_BILINEAR_" "no_aspect_preservation"
        ),
        "to_tensor": "torchvision_ToTensor_float32_divide_255",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "face_alignment": False,
        "router_mode": "Auto (Router)",
        "manual_weights": None,
    }:
        raise ValueError("OmniAID manifest preprocess contract mismatch")
    if manifest.get("source") != config_record.get("source"):
        raise ValueError("OmniAID manifest source copies disagree")
    if manifest.get("assets") != config_record.get("assets"):
        raise ValueError("OmniAID manifest asset copies disagree")
    if manifest.get("model_audit") != config_record.get("model_audit"):
        raise ValueError("OmniAID manifest model-audit copies disagree")
    if manifest.get("runtime_golden") != config_record.get("runtime_golden"):
        raise ValueError("OmniAID manifest golden copies disagree")
    if manifest.get("runtime") != config_record.get("runtime"):
        raise ValueError("OmniAID manifest runtime copies disagree")
    adapter_evidence = _verify_adapter_contract(config_record, repo_root)
    _reject_t2(manifest, path="manifest")
    _reject_t2(summary, path="summary")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    for key, path in {
        "results_path": results_path,
        "summary_path": summary_path,
        "artifact_dir": run_dir / "artifacts",
    }.items():
        try:
            relative = str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError as exc:
            raise ValueError(f"OmniAID {key} escapes repository") from exc
        if outputs.get(key) != relative:
            raise ValueError(f"OmniAID manifest {key} mismatch")
    _verify_file(
        results_path,
        outputs.get("results_sha256"),
        "OmniAID results",
    )
    _verify_file(
        summary_path,
        outputs.get("summary_sha256"),
        "OmniAID summary",
    )
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    if dataset.get("inputs_sha256") != CANONICAL_RELEASE["inputs_sha256"]:
        raise ValueError("OmniAID canonical inputs hash mismatch")
    try:
        relative_expected = str(
            expected_path.resolve().relative_to(repo_root.resolve())
        )
    except ValueError as exc:
        raise ValueError("OmniAID expected inputs escape repository") from exc
    if dataset.get("expected_inputs_path") != relative_expected:
        raise ValueError("OmniAID expected-input path mismatch")
    _verify_file(
        expected_path,
        dataset.get("expected_inputs_sha256"),
        "OmniAID expected inputs",
    )
    expected = read_jsonl(expected_path)
    physical = read_jsonl(results_path)
    if len(expected) != int(dataset.get("selected_images", -1)):
        raise ValueError("OmniAID expected image count mismatch")
    expected_ids = [str(row["sample_id"]) for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("OmniAID expected IDs are duplicated")
    visibility = _visibility_by_task(expected)
    expected_tasks = sorted(visibility)
    if int(dataset.get("selected_tasks", -1)) != len(expected_tasks):
        raise ValueError("OmniAID expected task count mismatch")
    if config_record.get("selected_sample_ids") != expected_ids:
        raise ValueError("OmniAID configured sample selection mismatch")
    if config_record.get("selected_tasks") != expected_tasks:
        raise ValueError("OmniAID configured task selection mismatch")
    if manifest.get("visibility_census") != {"full": len(expected_tasks)}:
        raise ValueError("OmniAID manifest visibility census mismatch")
    if manifest["run_id"] == DEFAULT_RUN_ID and (
        len(expected) != 550 or len(expected_tasks) != 275
    ):
        raise ValueError("OmniAID default full run coverage mismatch")
    latest = _latest_rows(physical)
    if set(latest) != set(expected_ids):
        raise ValueError("OmniAID latest result coverage mismatch")

    source_evidence = _verify_source(source_root, space_root)
    state, model_config, asset_evidence = _load_assets(
        checkpoint_path,
        config_path,
    )
    device, runtime_evidence = _configure_runtime(device_text)
    recorded_runtime = _require_mapping(
        manifest.get("runtime"),
        "recorded runtime",
    )
    if recorded_runtime.get("device") != str(device):
        raise ValueError(
            "OmniAID audit device differs from run: "
            f"{device} != {recorded_runtime.get('device')}"
        )
    if runtime_evidence != recorded_runtime:
        raise ValueError("OmniAID independent runtime evidence mismatch")
    model, model_evidence = _build_model(
        state,
        model_config,
        device,
        space_root,
    )
    del state
    if source_evidence["github"]["commit"] != FROZEN_SOURCE_COMMIT:
        raise ValueError("OmniAID independently verified GitHub pin mismatch")
    if source_evidence["space"]["commit"] != FROZEN_SPACE_COMMIT:
        raise ValueError("OmniAID independently verified Space pin mismatch")
    if asset_evidence["checkpoint_sha256"] != checkpoint_record["sha256"]:
        raise ValueError("OmniAID independent checkpoint evidence mismatch")
    if model_evidence["state_entries"] != FROZEN_CHECKPOINT_TENSORS:
        raise ValueError("OmniAID independent model state count mismatch")

    replay_rows: list[dict[str, Any]] = []
    artifact_paths: set[Path] = set()
    max_array_diffs = {key: 0.0 for key in ARTIFACT_SCHEMA}
    max_head_replay_diff = 0.0
    max_router_index_replay_diff = 0.0
    max_router_gate_replay_diff = 0.0
    max_probability_diff = 0.0
    max_margin_diff = 0.0
    fresh_forwards = 0
    for expected_row in expected:
        sample_id = str(expected_row["sample_id"])
        row = latest[sample_id]
        if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
            raise ValueError(f"OmniAID latest row is not valid: {sample_id}")
        identity = _expected_identity(
            expected_row,
            manifest,
            repo_root,
            visibility[str(expected_row["task_id"])],
        )
        for key, value in identity.items():
            if row.get(key) != value:
                raise ValueError(
                    f"OmniAID result identity {key} mismatch for {sample_id}"
                )
        input_path = _anchored(Path(str(row["input_path"])), repo_root)
        _verify_file(
            input_path,
            row.get("input_sha256"),
            f"OmniAID input {sample_id}",
        )
        image, preprocess = _preprocess(input_path)
        if row.get("preprocess") != preprocess:
            raise ValueError(f"OmniAID preprocess mismatch for {sample_id}")

        artifact_value = row.get("artifact_path")
        if not isinstance(artifact_value, str):
            raise ValueError(f"OmniAID artifact path missing for {sample_id}")
        artifact = _anchored(Path(artifact_value), repo_root)
        artifact_root = (run_dir / "artifacts").resolve()
        if artifact.parent != artifact_root:
            raise ValueError(f"OmniAID artifact escapes run for {sample_id}")
        expected_artifact = (artifact_root / f"{sample_id}.npz").resolve()
        if artifact != expected_artifact:
            raise ValueError(f"OmniAID artifact name mismatch for {sample_id}")
        if artifact in artifact_paths:
            raise ValueError("OmniAID artifact reused by multiple rows")
        artifact_paths.add(artifact)
        _verify_file(
            artifact,
            row.get("artifact_sha256"),
            f"OmniAID artifact {sample_id}",
        )
        persisted = _load_artifact(artifact)
        array_hashes = {key: _array_sha256(value) for key, value in persisted.items()}
        if row.get("artifact_keys") != list(ARTIFACT_SCHEMA):
            raise ValueError(f"OmniAID artifact key record mismatch for {sample_id}")
        if row.get("artifact_paths") != {
            "omniaid_npz": str(artifact.relative_to(repo_root))
        }:
            raise ValueError(f"OmniAID artifact path record mismatch for {sample_id}")
        if row.get("artifact_array_sha256") != array_hashes:
            raise ValueError(f"OmniAID artifact array hashes mismatch for {sample_id}")
        if array_hashes["pooler_output"] != row.get("feature_array_sha256"):
            raise ValueError(f"OmniAID feature hash mismatch for {sample_id}")
        if array_hashes["class_logits"] != row.get("class_logits_array_sha256"):
            raise ValueError(f"OmniAID logits hash mismatch for {sample_id}")
        metadata = {
            "feature_shape": [FEATURE_DIMENSION],
            "feature_dtype": "float32",
            "feature_semantics": FEATURE_SEMANTICS,
            "class_logits_shape": [CLASS_COUNT],
            "class_logits_dtype": "float32",
            "routing_feature_shape": [FEATURE_DIMENSION],
            "routing_feature_dtype": "float32",
            "routing_feature_semantics": ROUTING_FEATURE_SEMANTICS,
            "semantic_top_k_indices_shape": [SEMANTIC_TOP_K],
            "semantic_top_k_gates_shape": [SEMANTIC_TOP_K],
            "final_gates_shape": [EXPERT_COUNT],
        }
        for key, value in metadata.items():
            if row.get(key) != value:
                raise ValueError(
                    f"OmniAID result metadata {key} mismatch for {sample_id}"
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fresh, fresh_replay = _fresh_forward(model, device, image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fresh_forwards += 1
        for key in ARTIFACT_SCHEMA:
            difference = float(
                np.max(
                    np.abs(
                        fresh[key].astype(np.float64)
                        - persisted[key].astype(np.float64)
                    )
                )
            )
            max_array_diffs[key] = max(max_array_diffs[key], difference)
            if not np.array_equal(fresh[key], persisted[key]):
                raise ValueError(f"OmniAID fresh {key} replay differs for {sample_id}")

        persisted_feature = (
            torch.from_numpy(persisted["pooler_output"]).to(device).unsqueeze(0)
        )
        persisted_routing = (
            torch.from_numpy(persisted["routing_feature"]).to(device).unsqueeze(0)
        )
        with torch.inference_mode():
            head_replay = functional.linear(
                persisted_feature,
                model.head.weight,
                model.head.bias,
            )
            probability = torch.softmax(head_replay, dim=1)[:, 1]
            margin = head_replay[:, 1] - head_replay[:, 0]
            router_replay = model.gating_network(persisted_routing)
        replay_logits = np.ascontiguousarray(
            head_replay[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        replay_indices = np.ascontiguousarray(
            router_replay["top_k_indices"][0].detach().cpu().numpy(),
            dtype=np.int64,
        )
        replay_gates = np.ascontiguousarray(
            router_replay["top_k_gates"][0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        head_diff = float(
            np.max(
                np.abs(
                    replay_logits.astype(np.float64)
                    - persisted["class_logits"].astype(np.float64)
                )
            )
        )
        router_index_diff = float(
            np.max(
                np.abs(
                    replay_indices.astype(np.float64)
                    - persisted["semantic_top_k_indices"].astype(np.float64)
                )
            )
        )
        router_gate_diff = float(
            np.max(
                np.abs(
                    replay_gates.astype(np.float64)
                    - persisted["semantic_top_k_gates"].astype(np.float64)
                )
            )
        )
        max_head_replay_diff = max(max_head_replay_diff, head_diff)
        max_router_index_replay_diff = max(
            max_router_index_replay_diff,
            router_index_diff,
        )
        max_router_gate_replay_diff = max(
            max_router_gate_replay_diff,
            router_gate_diff,
        )
        if head_diff != 0.0 or router_index_diff != 0.0 or router_gate_diff != 0.0:
            raise ValueError(f"OmniAID artifact replay differs for {sample_id}")

        class_logits = row.get("class_logits")
        if (
            not isinstance(class_logits, list)
            or len(class_logits) != CLASS_COUNT
            or [float(value) for value in persisted["class_logits"].tolist()]
            != class_logits
        ):
            raise ValueError(f"OmniAID embedded logits mismatch for {sample_id}")
        embedded_arrays = {
            "semantic_top_k_indices": [
                int(value) for value in persisted["semantic_top_k_indices"].tolist()
            ],
            "semantic_top_k_gates": [
                float(value) for value in persisted["semantic_top_k_gates"].tolist()
            ],
            "final_expert_gates": [
                float(value) for value in persisted["final_gates"].tolist()
            ],
        }
        for key, value in embedded_arrays.items():
            if row.get(key) != value:
                raise ValueError(f"OmniAID embedded {key} mismatch for {sample_id}")
        score = float(probability[0].item())
        stored_score = _require_finite(
            row.get("ai_score"),
            f"OmniAID score {sample_id}",
        )
        probability_diff = abs(score - stored_score)
        max_probability_diff = max(max_probability_diff, probability_diff)
        if probability_diff != 0.0:
            raise ValueError(f"OmniAID probability replay differs for {sample_id}")
        for alias in ("score", "probability", "fake_probability"):
            if row.get(alias) != stored_score:
                raise ValueError(f"OmniAID score alias {alias} drifted")
        margin_value = float(margin[0].item())
        stored_margin = _require_finite(
            row.get("raw_logit_margin"),
            f"OmniAID margin {sample_id}",
        )
        margin_diff = abs(margin_value - stored_margin)
        max_margin_diff = max(max_margin_diff, margin_diff)
        if margin_diff != 0.0:
            raise ValueError(f"OmniAID margin replay differs for {sample_id}")
        if row.get("classification_decision") is not (score > THRESHOLD):
            raise ValueError(f"OmniAID decision mismatch for {sample_id}")
        expected_scoring = {
            "routing_mode": "Auto (Router)",
            "semantic_expert_names": [
                "Human",
                "Animal",
                "Object",
                "Scene",
                "Anime",
            ],
            "artifact_expert_name": "Artifact",
            "semantic_gate_sum": float(
                persisted["final_gates"][:SEMANTIC_EXPERT_COUNT].sum()
            ),
            "final_gate_sum": float(persisted["final_gates"].sum()),
            "classification": {
                "decision": score > THRESHOLD,
                "threshold": THRESHOLD,
                "operator": ">",
            },
            "t1": {
                "valid": True,
                "score": score,
                "decision": score > THRESHOLD,
            },
            "manual_replay": {
                "head_logits_exact": True,
                "softmax_dtype": "float32",
                "fake_class_index": 1,
                "router_scatter_exact": True,
            },
        }
        for key, value in expected_scoring.items():
            if row.get(key) != value:
                raise ValueError(
                    f"OmniAID scoring field {key} mismatch for {sample_id}"
                )
        if fresh_replay["score"] != score:
            raise ValueError(f"OmniAID fresh score differs for {sample_id}")
        if fresh_replay["raw_logit_margin"] != margin_value:
            raise ValueError(f"OmniAID fresh margin differs for {sample_id}")
        replay_row = dict(row)
        replay_row.update(
            {
                "ai_score": fresh_replay["score"],
                "fake_probability": fresh_replay["score"],
                "probability": fresh_replay["score"],
                "score": fresh_replay["score"],
                "raw_logit_margin": fresh_replay["raw_logit_margin"],
                "classification_decision": fresh_replay["score"] > THRESHOLD,
            }
        )
        replay_rows.append(replay_row)

    disk_artifacts = {
        path.resolve()
        for path in (run_dir / "artifacts").glob("*.npz")
        if path.is_file()
    }
    if disk_artifacts != artifact_paths:
        raise ValueError("OmniAID artifact directory has missing/extra files")
    if int(outputs.get("artifact_files", -1)) != len(artifact_paths):
        raise ValueError("OmniAID manifest artifact count mismatch")

    bootstrap_samples = int(config_record["bootstrap_samples"])
    bootstrap_seed = int(config_record["bootstrap_seed"])
    recomputed = summarize_omniaid_results(
        physical,
        expected,
        threshold=THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    recomputed.update(
        {
            "run_id": manifest["run_id"],
            "model": "OmniAID",
            "model_slug": FROZEN_MODEL_SLUG,
            "checkpoint_id": FROZEN_CHECKPOINT_ID,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "preprocess_profile": FROZEN_PREPROCESS_PROFILE,
            "config_fingerprint": manifest["config_fingerprint"],
            "runtime_golden_status": _require_mapping(
                manifest.get("runtime_golden"),
                "manifest runtime golden",
            ).get("status"),
            "runtime_golden_fingerprint": _fingerprint(
                _require_mapping(
                    manifest.get("runtime_golden"),
                    "manifest runtime golden",
                )
            ),
            "generated_at": summary.get("generated_at"),
        }
    )
    if recomputed != summary:
        raise ValueError("OmniAID summary does not exactly recompute")
    replay_by_id = {str(row["id"]): row for row in replay_rows}
    fresh_physical = [
        (replay_by_id[str(row["id"])] if latest[str(row["id"])] is row else row)
        for row in physical
    ]
    fresh_recomputed = summarize_omniaid_results(
        fresh_physical,
        expected,
        threshold=THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    fresh_recomputed.update(
        {
            "run_id": manifest["run_id"],
            "model": "OmniAID",
            "model_slug": FROZEN_MODEL_SLUG,
            "checkpoint_id": FROZEN_CHECKPOINT_ID,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "preprocess_profile": FROZEN_PREPROCESS_PROFILE,
            "config_fingerprint": manifest["config_fingerprint"],
            "runtime_golden_status": _require_mapping(
                manifest.get("runtime_golden"),
                "manifest runtime golden",
            ).get("status"),
            "runtime_golden_fingerprint": _fingerprint(
                _require_mapping(
                    manifest.get("runtime_golden"),
                    "manifest runtime golden",
                )
            ),
            "generated_at": summary.get("generated_at"),
        }
    )
    if fresh_recomputed != summary:
        raise ValueError("OmniAID fresh-forward summary does not exactly recompute")

    result_fingerprints = {
        sample_id: _fingerprint(latest[sample_id]) for sample_id in expected_ids
    }
    audit = {
        "schema_version": "omniaid_independent_audit_v1",
        "status": "ok",
        "run_id": manifest["run_id"],
        "audited_at": utc_now(),
        "run_dir": str(run_dir),
        "source": source_evidence,
        "adapter": adapter_evidence,
        "assets": asset_evidence,
        "runtime": runtime_evidence,
        "model": model_evidence,
        "coverage": {
            "expected_images": len(expected),
            "physical_result_rows": len(physical),
            "latest_images": len(latest),
            "fresh_full_model_forwards": fresh_forwards,
            "artifact_replays": len(artifact_paths),
            "complete_pairs": int(
                recomputed["paired_coverage"]["complete_valid_pairs"]
            ),
        },
        "replay": {
            "max_abs_array_diff": max_array_diffs,
            "max_abs_head_replay_diff": max_head_replay_diff,
            "max_abs_router_index_replay_diff": (max_router_index_replay_diff),
            "max_abs_router_gate_replay_diff": max_router_gate_replay_diff,
            "max_abs_probability_diff": max_probability_diff,
            "max_abs_margin_diff": max_margin_diff,
            "all_six_artifact_arrays_exact": True,
            "all_router_gates_exact": True,
            "all_decisions_exact": True,
            "all_preprocess_records_exact": True,
            "summary_exact_recompute": True,
            "fresh_forward_summary_exact_recompute": True,
        },
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "results_sha256": sha256_file(results_path),
            "expected_inputs_sha256": sha256_file(expected_path),
            "summary_sha256": sha256_file(summary_path),
            "config_fingerprint": manifest["config_fingerprint"],
            "result_fingerprints": result_fingerprints,
        },
        "recomputed_summary": recomputed,
        "fresh_forward_recomputed_summary": fresh_recomputed,
    }
    _reject_t2(audit, path="audit")
    atomic_write_json(output_path, audit)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--space-root", type=Path, default=DEFAULT_SPACE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--omniaid-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    results_root = _anchored(args.results_dir, repo_root)
    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root.resolve():
        raise ValueError("OmniAID audit run-id escapes results directory")
    source_root = _anchored(args.source_root, repo_root)
    space_root = _anchored(args.space_root, repo_root)
    checkpoint = _anchored(args.checkpoint, repo_root)
    config = _anchored(args.omniaid_config, repo_root)
    output = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else run_dir / "independent_audit.json"
    )
    protected_evidence = {
        (run_dir / name).resolve()
        for name in (
            "run_manifest.json",
            "results.jsonl",
            "expected_inputs.jsonl",
            "summary.json",
        )
    }
    if output.resolve() in protected_evidence:
        raise ValueError("OmniAID audit output cannot overwrite run evidence")
    if output.suffix != ".json":
        raise ValueError("OmniAID audit output must be a JSON file")
    audit = _audit_run(
        repo_root=repo_root,
        run_dir=run_dir,
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint,
        config_path=config,
        device_text=args.device,
        output_path=output,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "run_id": audit["run_id"],
                "coverage": audit["coverage"],
                "replay": audit["replay"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
