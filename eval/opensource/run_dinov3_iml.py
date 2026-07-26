#!/usr/bin/env python3
"""Run the official DINOv3-IML CAT ViT-L LoRA-r32 checkpoint.

The released checkpoint contains the complete DINOv3 backbone, LoRA adapters,
and segmentation head.  The Meta repository is therefore used only to
construct the frozen architecture with ``pretrained=False``; no separate,
gated backbone weights are loaded.  DINOv3-IML has no image-level prediction
head, so this adapter reports T2 localization only and never derives T1 from
the probability map.
"""

from __future__ import annotations

import os

# PyTorch requires this to be present before CUDA/cuBLAS initialization when
# deterministic algorithms are enabled.  Do not overwrite a conflicting user
# value; configure_determinism rejects anything outside the frozen contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    read_latest_by_id,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.dinov3_iml_metrics import (
    binary_pixel_metrics_strict,
    summarize_dinov3_iml_results,
)


MODEL_NAME = "DINOv3-IML"
MODEL_SLUG = "dinov3_iml_cat_vitl_lora_r32_checkpoint48_official"
MODEL_REPO_URL = "https://github.com/Irennnne/DINOv3-IML"
MODEL_SOURCE_COMMIT = "ba45b0a203c698b36fe2b0e658bb49ebbb1163cc"
DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3"
DINOV3_SOURCE_COMMIT = "31703e4cbf1ccb7c4a72daa1350405f86754b6d1"
MODEL_INPUT_SIZE = 512
INTERNAL_LOGIT_SIZE = 32
MASK_THRESHOLD = 0.5
PREPROCESS_PROTOCOL = "official_standalone_pillow_rgb_bilinear_stretch_512_imagenet"
LOGIT_CAPTURE = "one_forward_hook_on_author_model_seg_head"
MODEL_LOGIT_RESIZE = "bilinear_seg_head_logits_32_to_512_align_corners_false"
NATIVE_RESTORE = "bilinear_official_model_512_probability_to_native_align_corners_false"
ARCHITECTURE_ONLY_WEIGHTS_SENTINEL = (
    "CLAIMFORGE_FULL_STATE_CHECKPOINT_NO_SEPARATE_BACKBONE_WEIGHTS"
)
AUTHOR_MODULE_ALIAS = "_claimforge_dinov3_iml_author_lora_model"

SOURCE_FILES = {
    "README.md": "afab44826398bd7fb532747a3358a0fb0412a24970392d91b55932f32171b8f1",
    "LICENSE": "f73896baf1965c31ca84a67f1e48f4e926faefd34ab96c78cd17566cb9d2f431",
    "inference.py": (
        "c3c05dc74983e7c5037944ee89f975056d3a036035dfad7bacc7d0271683d5b9"
    ),
    "requirements.txt": (
        "4a48b83d9fec3391f63918d1787e4c1970c144661285ae3ed555550e5e93a0df"
    ),
    "configs/lora_vitl_r32.yaml": (
        "5e1340c7086f268a0cb67ae5d8e019e15d2361a5be91957ac85077b7f7425155"
    ),
    "configs/_base/lora_common.yaml": (
        "2f74d897e666a83f81164bc4c138c4ace03d618f0f5d67421f4c522f1c17d75e"
    ),
    "configs/_base/paths_template.yaml": (
        "b1a4a65e1027e65534a4f0209691548b521c312025112f546e5ff25bf370f353"
    ),
    "models/__init__.py": (
        "1fe7c4f5de662f98a1b097573a12610132c12c7b99bbae347d41f3113a7d32fa"
    ),
    "models/dinov3_forensics_lora.py": (
        "2ef507af139925ee3b99f791ec38b5f279f46727f251244bf55a331cdd367de6"
    ),
    "test.py": "4d992e86871cbd002e75a2883e2f6754f181f525cfca6fb47e18f8823c8a0bc9",
}

DINOV3_SOURCE_FILES = {
    "hubconf.py": ("b36bf0b0502127fb938b6f3424c6e20b48206ee361422aae777e87e20006ba24"),
    "LICENSE.md": ("25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"),
    "dinov3/hub/backbones.py": (
        "1b4a111e028e7a414c40f8b4bb84df6aac9c509bcc62aa6979ead92e8a81daae"
    ),
    "dinov3/models/vision_transformer.py": (
        "1af05405e19bc42188f80bfcb204b87ed1ec72af998e0bbd4658a85d7fae93f2"
    ),
    "dinov3/layers/attention.py": (
        "dfc21def6ba17d00cba18ea23a8726e7f1dd5a0dc8e43510852c999a173d37a8"
    ),
    "dinov3/layers/block.py": (
        "809a3615ec93019042ab440f31c21d5242186d9e503c67ac0f42a324ceae1950"
    ),
    "dinov3/layers/patch_embed.py": (
        "c592c7262779c8e1789ed502aee81bdd55eba4607065f8d04ae95eea683d0ef1"
    ),
}

CHECKPOINT = {
    "provider": "official_author_google_drive",
    "folder_id": "125leLub_M-lICa1ILTOL-FCz4ZY6eutj",
    "file_id": "1xqZDqhSQUl_1vs3SD4EfjHHmeu2pwLh9",
    "original_filename": "checkpoint-48.pth",
    "last_modified_utc": "2026-04-07T14:28:38Z",
    "bytes": 1_321_705_819,
    "sha256": "01f23401e048f706ea0e63fb0429ddef80db3197ac0f5707bd584a8b056177fa",
    "container": "mapping_with_model_optimizer_epoch_scaler_args",
    "top_level_keys": ["model", "optimizer", "epoch", "scaler", "args"],
    "epoch": 48,
    "args_type": "argparse.Namespace",
    "state_container": "collections.OrderedDict",
    "state_keys": 432,
    "state_elements": 312_275_987,
    "tensor_bytes": 1_249_103_956,
    "state_dtypes": {"torch.float32": 430, "torch.int64": 2},
    "state_prefix_counts": {"backbone.": 416, "seg_head.": 16},
    "lora_state_keys": 48,
    "parameters": 312_200_705,
    "buffers": 75_282,
    "trainable_parameters": 9_046_529,
    "optimizer_contract": {
        "top_level_keys": ["state", "param_groups"],
        "state_entries": 58,
        "param_groups": 2,
        "group_parameter_counts": [7, 51],
        "group_weight_decay": [0.0, 0.05],
        "group_betas": [[0.9, 0.999], [0.9, 0.999]],
        "group_decoupled_weight_decay": [True, True],
    },
    "scaler_contract": {
        "top_level_keys": [
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        ],
        "values": {
            "scale": 131072.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 1105,
        },
    },
    "expected_args": {
        "model": "DINOv3ForensicsLoRA",
        "if_predict_label": False,
        "image_size": 512,
        "if_padding": False,
        "if_resizing": True,
        "edge_mask_width": 7,
        "batch_size": 24,
        "test_batch_size": 48,
        "epochs": 100,
        "accum_iter": 10,
        "weight_decay": 0.05,
        "lr": 0.0003,
        "device": "cuda",
        "seed": 42,
        "start_epoch": 41,
        "opt": "AdamW",
    },
    "critical_state_shapes": {
        "backbone.cls_token": [1, 1, 1024],
        "backbone.storage_tokens": [1, 4, 1024],
        "backbone.patch_embed.proj.weight": [1024, 3, 16, 16],
        "backbone.blocks.0.attn.qkv.base_layer.weight": [3072, 1024],
        "backbone.blocks.0.attn.qkv.lora_A.default.weight": [32, 1024],
        "backbone.blocks.0.attn.qkv.lora_B.default.weight": [3072, 32],
        "backbone.blocks.23.attn.qkv.lora_A.default.weight": [32, 1024],
        "backbone.blocks.23.attn.qkv.lora_B.default.weight": [3072, 32],
        "seg_head.0.weight": [512, 1024, 3, 3],
        "seg_head.3.weight": [256, 512, 3, 3],
        "seg_head.6.weight": [1, 256, 1, 1],
    },
}

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/mouse_canonical_v1/manifest.json")
DEFAULT_DINOV3_IML_ROOT = Path("/root/.cache/claimforge/third_party/DINOv3-IML")
DEFAULT_DINOV3_ROOT = Path("/root/.cache/claimforge/third_party/dinov3")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/" "dinov3iml-ba45b0a/checkpoint-48.pth"
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


def _installed_module_contract(
    module_name: str,
    distribution_names: tuple[str, ...],
) -> dict[str, Any]:
    """Bind a runtime module to files owned by an allowed distribution."""

    module = importlib.import_module(module_name)
    source_value = getattr(module, "__file__", None)
    if not isinstance(source_value, str):
        raise ValueError(f"{module_name} has no verifiable module source")
    source = Path(source_value).resolve()
    matches: list[dict[str, str]] = []
    for distribution_name in distribution_names:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        files = distribution.files
        if files is None:
            continue
        if any(
            Path(distribution.locate_file(item)).resolve() == source for item in files
        ):
            matches.append(
                {
                    "name": distribution.metadata["Name"],
                    "version": distribution.version,
                }
            )
    if not matches:
        raise ValueError(
            f"{module_name} source is not owned by an allowed installed "
            f"distribution: {source}"
        )
    module_version = getattr(module, "__version__", None)
    return {
        "module": module_name,
        "module_version": (str(module_version) if module_version is not None else None),
        "source": str(source),
        "distributions": sorted(
            matches,
            key=lambda item: (item["name"].lower(), item["version"]),
        ),
    }


def _runtime_contract(device_name: str) -> dict[str, Any]:
    """Capture numerical packages, accelerator, and attention math flags."""

    import torch

    packages = {
        "torch": _installed_module_contract("torch", ("torch",)),
        "peft": _installed_module_contract("peft", ("peft",)),
        "transformers": _installed_module_contract(
            "transformers",
            ("transformers",),
        ),
        "accelerate": _installed_module_contract(
            "accelerate",
            ("accelerate",),
        ),
        "huggingface-hub": _installed_module_contract(
            "huggingface_hub",
            ("huggingface-hub",),
        ),
        "safetensors": _installed_module_contract(
            "safetensors",
            ("safetensors",),
        ),
        "numpy": _installed_module_contract("numpy", ("numpy",)),
        "Pillow": _installed_module_contract("PIL", ("Pillow",)),
        "scikit-learn": _installed_module_contract(
            "sklearn",
            ("scikit-learn",),
        ),
    }
    pillow_versions = {item["version"] for item in packages["Pillow"]["distributions"]}
    if pillow_versions != {"11.1.0"}:
        raise ValueError(
            "DINOv3-IML frozen preprocessing requires Pillow 11.1.0, got "
            f"{sorted(pillow_versions)}"
        )

    requested_device = torch.device(device_name)
    cuda_active = requested_device.type == "cuda" and torch.cuda.is_available()
    accelerator = {
        "requested_device": str(requested_device),
        "torch_cuda": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(requested_device) if cuda_active else None
        ),
        "gpu_capability": (
            list(torch.cuda.get_device_capability(requested_device))
            if cuda_active
            else None
        ),
    }

    def _cuda_flag(name: str) -> bool | None:
        function = getattr(torch.backends.cuda, name, None)
        return bool(function()) if callable(function) else None

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": packages,
        "optional_imdlbenco_present": (
            importlib.util.find_spec("IMDLBenCo") is not None
        ),
        "accelerator": accelerator,
        "numerical_flags": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": (torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": (torch.get_float32_matmul_precision()),
            "flash_sdp_enabled": _cuda_flag("flash_sdp_enabled"),
            "mem_efficient_sdp_enabled": _cuda_flag("mem_efficient_sdp_enabled"),
            "math_sdp_enabled": _cuda_flag("math_sdp_enabled"),
        },
    }


def configure_determinism(seed: int) -> None:
    """Freeze the one-pass float32 inference math before fingerprinting."""

    import torch

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "frozen DINOv3-IML inference requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def _manifest_fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _verify_runtime_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _verify_file_contract(
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    _verify_runtime_file(path, str(contract["sha256"]), label)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(contract["bytes"]):
        raise ValueError(
            f"{label} byte-size mismatch: " f"{actual_bytes} != {contract['bytes']}"
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


def _atomic_save_mask(path: Path, mask: np.ndarray) -> None:
    pixels = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    if pixels.ndim != 2:
        raise ValueError("binary mask artifact must be two-dimensional")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".png",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(pixels, mode="L").save(
            temporary,
            format="PNG",
            optimize=False,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    if sha256_file(inputs_path) != release.get("inputs_sha256"):
        raise ValueError("canonical inputs.jsonl hash does not match release manifest")
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
    """Select complete leading pairs, or one explicit preflight sample."""

    if sample_id is not None:
        if pair_limit is not None:
            raise ValueError("sample_id and pair_limit are mutually exclusive")
        matches = [row for row in rows if row.get("sample_id") == sample_id]
        if len(matches) != 1:
            raise ValueError(
                f"sample_id must identify exactly one canonical input: {sample_id}"
            )
        return matches

    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [row for row in rows if int(row["pair_rank"]) in selected_ranks]
    by_pair: dict[int, list[str]] = {}
    for row in selected:
        by_pair.setdefault(int(row["pair_rank"]), []).append(str(row["kind"]))
    invalid = {
        rank: kinds
        for rank, kinds in by_pair.items()
        if len(kinds) != 2 or set(kinds) != {"real", "forged"}
    }
    if invalid:
        raise ValueError(f"canonical selection contains incomplete pairs: {invalid}")
    return selected


def preprocess_image(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    """Replay the author's standalone Pillow preprocessing byte-for-byte."""

    with path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoder_format = opened.format
            image = opened.convert("RGB")
    native_width, native_height = image.size
    if native_width <= 0 or native_height <= 0:
        raise ValueError("DINOv3-IML input has invalid native dimensions")
    resized = image.resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    rgb = np.array(resized, dtype=np.float32)
    if rgb.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
        raise ValueError(f"unexpected DINOv3-IML resized image: {rgb.shape}")
    normalized = rgb / 255.0
    normalized = (
        normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = normalized.astype(np.float32)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected DINOv3-IML input shape: {chw.shape}")
    if chw.dtype != np.float32 or not np.isfinite(chw).all():
        raise ValueError("DINOv3-IML normalized input is not finite float32")
    metadata = {
        "protocol": PREPROCESS_PROTOCOL,
        "reference": "upstream_inference._load_and_preprocess",
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "native_size_wh": [native_width, native_height],
        "model_size_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize": "Pillow.Image.resize",
        "resize_interpolation": "Pillow.Image.Resampling.BILINEAR",
        "resize_box": None,
        "resize_reducing_gap": None,
        "input_crop": None,
        "input_reencode": False,
        "normalization": {
            "scale": "float32_divide_255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "tensor_shape": list(chw.shape),
        "tensor_dtype": str(chw.dtype),
        "tensor_sha256": _sha256_array(chw),
    }
    return chw, (native_width, native_height), metadata


def _float32_map(tensor: Any, label: str) -> np.ndarray:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{label} is not a tensor")
    array = np.ascontiguousarray(
        tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def postprocess_outputs(
    probability: Any,
    raw_logits: Any,
    *,
    native_width: int,
    native_height: int,
) -> dict[str, np.ndarray]:
    """Verify the author's probability chain and create native probability."""

    import torch
    from torch.nn import functional as F

    if not isinstance(raw_logits, torch.Tensor) or tuple(raw_logits.shape) != (
        1,
        1,
        INTERNAL_LOGIT_SIZE,
        INTERNAL_LOGIT_SIZE,
    ):
        raise ValueError(
            "unexpected DINOv3-IML seg_head logit shape: "
            f"{getattr(raw_logits, 'shape', None)}"
        )
    if not isinstance(probability, torch.Tensor) or tuple(probability.shape) != (
        1,
        1,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            "unexpected DINOv3-IML probability shape: "
            f"{getattr(probability, 'shape', None)}"
        )
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native dimensions must be positive")

    logits_512 = F.interpolate(
        raw_logits.float(),
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    expected_probability = torch.sigmoid(logits_512)
    if not torch.allclose(
        probability.float(),
        expected_probability,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "DINOv3-IML output does not match sigmoid of the captured "
            "32-to-512 align_corners=False logits"
        )
    native_probability = F.interpolate(
        probability.float(),
        size=(native_height, native_width),
        mode="bilinear",
        align_corners=False,
    )
    result = {
        "raw_logits_model_32": _float32_map(
            raw_logits[0, 0],
            "DINOv3-IML seg_head logits",
        ),
        "raw_logits_model_512": _float32_map(
            logits_512[0, 0],
            "DINOv3-IML resized logits",
        ),
        "probability_model_512": _float32_map(
            probability[0, 0],
            "DINOv3-IML model probability",
        ),
        "probability_native": _float32_map(
            native_probability[0, 0],
            "DINOv3-IML native probability",
        ),
    }
    for label in ("probability_model_512", "probability_native"):
        array = result[label]
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"DINOv3-IML {label} falls outside [0, 1]")
    return result


def _load_checkpoint_payload(
    *,
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    """Safely deserialize and validate the complete released checkpoint."""

    import torch

    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} checkpoint is not a mapping")
    if list(payload) != list(contract["top_level_keys"]):
        raise ValueError(
            f"{label} checkpoint top-level schema mismatch: {list(payload)}"
        )
    if payload.get("epoch") != int(contract["epoch"]):
        raise ValueError(f"{label} checkpoint epoch mismatch")
    if not isinstance(payload.get("args"), argparse.Namespace):
        raise ValueError(f"{label} checkpoint args is not argparse.Namespace")
    optimizer_contract = contract.get("optimizer_contract")
    if optimizer_contract is not None:
        if not isinstance(optimizer_contract, Mapping):
            raise ValueError(f"{label} optimizer contract is invalid")
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError(f"{label} checkpoint optimizer is not a mapping")
        if list(optimizer) != list(optimizer_contract["top_level_keys"]):
            raise ValueError(f"{label} checkpoint optimizer schema mismatch")
        optimizer_state = optimizer.get("state")
        parameter_groups = optimizer.get("param_groups")
        if (
            not isinstance(optimizer_state, Mapping)
            or len(optimizer_state) != int(optimizer_contract["state_entries"])
            or not isinstance(parameter_groups, list)
            or len(parameter_groups) != int(optimizer_contract["param_groups"])
        ):
            raise ValueError(f"{label} checkpoint optimizer size mismatch")
        all_parameter_ids: list[int] = []
        for index, group in enumerate(parameter_groups):
            if not isinstance(group, Mapping):
                raise ValueError(
                    f"{label} checkpoint optimizer group {index} is invalid"
                )
            parameter_ids = group.get("params")
            if (
                not isinstance(parameter_ids, list)
                or len(parameter_ids)
                != int(optimizer_contract["group_parameter_counts"][index])
                or float(group.get("weight_decay"))
                != float(optimizer_contract["group_weight_decay"][index])
                or list(group.get("betas", ()))
                != list(optimizer_contract["group_betas"][index])
                or group.get("decoupled_weight_decay")
                is not optimizer_contract["group_decoupled_weight_decay"][index]
            ):
                raise ValueError(f"{label} checkpoint optimizer group {index} mismatch")
            all_parameter_ids.extend(int(value) for value in parameter_ids)
        if len(all_parameter_ids) != len(set(all_parameter_ids)) or set(
            all_parameter_ids
        ) != set(optimizer_state):
            raise ValueError(f"{label} checkpoint optimizer parameter IDs mismatch")
    scaler_contract = contract.get("scaler_contract")
    if scaler_contract is not None:
        if not isinstance(scaler_contract, Mapping):
            raise ValueError(f"{label} scaler contract is invalid")
        scaler = payload.get("scaler")
        if not isinstance(scaler, Mapping):
            raise ValueError(f"{label} checkpoint scaler is not a mapping")
        if list(scaler) != list(scaler_contract["top_level_keys"]):
            raise ValueError(f"{label} checkpoint scaler schema mismatch")
        if dict(scaler) != dict(scaler_contract["values"]):
            raise ValueError(f"{label} checkpoint scaler values mismatch")
    expected_args = contract.get("expected_args", {})
    if not isinstance(expected_args, Mapping):
        raise ValueError(f"{label} expected args contract is invalid")
    for name, expected in expected_args.items():
        actual = getattr(payload["args"], name, None)
        if isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"{label} checkpoint args.{name} mismatch: "
                f"{actual!r} != {expected!r}"
            )

    state = payload.get("model")
    if type(state).__name__ != "OrderedDict" or not isinstance(state, Mapping):
        raise ValueError(f"{label} model state is not the registered OrderedDict")
    if len(state) != int(contract["state_keys"]):
        raise ValueError(f"{label} checkpoint state-key count mismatch")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{label} checkpoint has non-tensor model state")
    elements = sum(int(value.numel()) for value in state.values())
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in state.values()
    )
    dtype_counts = Counter(str(value.dtype) for value in state.values())
    if elements != int(contract["state_elements"]):
        raise ValueError(f"{label} checkpoint state-element count mismatch")
    if tensor_bytes != int(contract["tensor_bytes"]):
        raise ValueError(f"{label} checkpoint tensor-byte count mismatch")
    if dict(dtype_counts) != dict(contract["state_dtypes"]):
        raise ValueError(f"{label} checkpoint dtype schema mismatch")

    prefix_counts = contract.get("state_prefix_counts", {})
    for prefix, expected_count in prefix_counts.items():
        actual_count = sum(str(key).startswith(str(prefix)) for key in state)
        if actual_count != int(expected_count):
            raise ValueError(f"{label} checkpoint {prefix!r} key-count mismatch")
    if "lora_state_keys" in contract:
        lora_keys = [key for key in state if ".lora_" in str(key)]
        if len(lora_keys) != int(contract["lora_state_keys"]):
            raise ValueError(f"{label} checkpoint LoRA key-count mismatch")
        expected_lora = {
            f"backbone.blocks.{block}.attn.qkv." f"lora_{matrix}.default.weight"
            for block in range(24)
            for matrix in ("A", "B")
        }
        if set(lora_keys) != expected_lora:
            raise ValueError(f"{label} checkpoint LoRA key layout mismatch")
    critical_shapes = contract.get("critical_state_shapes", {})
    for key, expected_shape in critical_shapes.items():
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or list(value.shape) != list(
            expected_shape
        ):
            raise ValueError(f"{label} checkpoint state shape mismatch for {key}")
    return payload


def _strict_load_checkpoint_state(
    *,
    module: Any,
    payload: Mapping[str, Any],
    label: str,
) -> Any:
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} checkpoint has no validated model state")
    incompatible = module.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", ()))
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise ValueError(
            f"{label} strict load returned incompatible keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return module


def _verify_repository_contract(
    *,
    root: Path,
    expected_commit: str,
    files: Mapping[str, str],
    label: str,
) -> None:
    source_commit = _git_value(root, "rev-parse", "HEAD")
    if source_commit != expected_commit:
        raise ValueError(
            f"{label} source commit mismatch: " f"{source_commit} != {expected_commit}"
        )
    if _git_value(root, "status", "--short", "--untracked-files=no"):
        raise ValueError(f"{label} tracked source files have local modifications")
    for relative, expected_sha in files.items():
        _verify_runtime_file(
            root / relative,
            expected_sha,
            f"{label} source file {relative}",
        )


def _verify_static_contract(
    *,
    dinov3_iml_root: Path,
    dinov3_root: Path,
    checkpoint_path: Path,
) -> None:
    if dinov3_iml_root.resolve() == dinov3_root.resolve():
        raise ValueError("author and Meta source roots must be distinct")
    _verify_repository_contract(
        root=dinov3_iml_root,
        expected_commit=MODEL_SOURCE_COMMIT,
        files=SOURCE_FILES,
        label="DINOv3-IML",
    )
    _verify_repository_contract(
        root=dinov3_root,
        expected_commit=DINOV3_SOURCE_COMMIT,
        files=DINOV3_SOURCE_FILES,
        label="Meta DINOv3 architecture",
    )
    _verify_file_contract(
        checkpoint_path,
        CHECKPOINT,
        "DINOv3-IML official CAT checkpoint 48",
    )


def _require_cached_module_origin(
    module_name: str,
    expected_path: Path,
) -> None:
    cached = sys.modules.get(module_name)
    if cached is None:
        return
    source_value = getattr(cached, "__file__", None)
    if not isinstance(source_value, str):
        raise ValueError(f"cached {module_name} module has no verifiable source file")
    actual = Path(source_value).resolve()
    expected = expected_path.resolve()
    if actual != expected:
        raise ValueError(
            f"cached {module_name} module source mismatch: " f"{actual} != {expected}"
        )


def _require_module_tree_origin(prefix: str, expected_root: Path) -> None:
    root = expected_root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != prefix and not name.startswith(f"{prefix}."):
            continue
        source_value = getattr(module, "__file__", None)
        if source_value is None:
            continue
        if not isinstance(source_value, str):
            raise ValueError(f"cached {name} module has invalid source metadata")
        source = Path(source_value).resolve()
        if source != root and root not in source.parents:
            raise ValueError(
                f"cached {name} module source escapes pinned Meta tree: " f"{source}"
            )


@contextlib.contextmanager
def _temporary_sys_path(path: Path):
    """Prepend one pinned source root and restore the exact prior path."""

    previous = list(sys.path)
    text = str(path.resolve())
    sys.path[:] = [text, *[item for item in previous if item != text]]
    try:
        yield
    finally:
        sys.path[:] = previous


def _construct_pinned_dinov3_vitl16(dinov3_root: Path) -> Any:
    """Import only the official backbone module, avoiding broad hubconf imports."""

    expected_source = (dinov3_root / "dinov3" / "hub" / "backbones.py").resolve()
    _require_module_tree_origin("dinov3", dinov3_root)
    with _temporary_sys_path(dinov3_root):
        module = importlib.import_module("dinov3.hub.backbones")
    _require_module_tree_origin("dinov3", dinov3_root)
    source_value = getattr(module, "__file__", None)
    if (
        not isinstance(source_value, str)
        or Path(source_value).resolve() != expected_source
    ):
        raise ValueError("Meta DINOv3 backbone factory source is not pinned")
    factory = getattr(module, "dinov3_vitl16", None)
    if factory is None:
        raise ValueError("pinned Meta source has no dinov3_vitl16 factory")
    factory_source = inspect.getsourcefile(factory)
    if factory_source is None or Path(factory_source).resolve() != expected_source:
        raise ValueError("Meta dinov3_vitl16 factory source is not pinned")
    return factory(pretrained=False)


def _load_author_model_class(dinov3_iml_root: Path) -> Any:
    source = (dinov3_iml_root / "models" / "dinov3_forensics_lora.py").resolve()
    _require_cached_module_origin(AUTHOR_MODULE_ALIAS, source)
    cached = sys.modules.get(AUTHOR_MODULE_ALIAS)
    if cached is None:
        spec = importlib.util.spec_from_file_location(
            AUTHOR_MODULE_ALIAS,
            source,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned author module: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[AUTHOR_MODULE_ALIAS] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(AUTHOR_MODULE_ALIAS, None)
            raise
    else:
        module = cached
    model_class = getattr(module, "DINOv3ForensicsLoRA", None)
    if model_class is None:
        raise ValueError("pinned author module has no DINOv3ForensicsLoRA")
    class_source = inspect.getsourcefile(model_class)
    if class_source is None or Path(class_source).resolve() != source:
        raise ValueError("DINOv3ForensicsLoRA class source is not pinned")
    return model_class


def _construct_author_model(
    *,
    model_class: Any,
    dinov3_root: Path,
    backbone_factory: Any | None = None,
) -> Any:
    """Construct once while replacing the gated base-weight request.

    The author constructor always supplies a weights path to ``torch.hub``.
    Because checkpoint-48 is a complete state dict, the intercepted call
    builds the exact pinned Meta architecture with ``pretrained=False``.
    """

    import torch

    expected_root = dinov3_root.resolve()
    author_calls: list[dict[str, Any]] = []

    def architecture_only_load(
        repo_or_dir: Any,
        model_name: Any,
        *positional: Any,
        **kwargs: Any,
    ) -> Any:
        source = kwargs.pop("source", None)
        weights = kwargs.pop("weights", None)
        if (
            Path(repo_or_dir).resolve() != expected_root
            or model_name != "dinov3_vitl16"
            or positional
            or source != "local"
            or weights != ARCHITECTURE_ONLY_WEIGHTS_SENTINEL
            or kwargs
        ):
            raise ValueError("DINOv3-IML constructor torch.hub.load signature changed")
        author_calls.append(
            {
                "repo": str(expected_root),
                "model": model_name,
                "source": source,
                "weights": weights,
            }
        )
        if backbone_factory is not None:
            return backbone_factory(pretrained=False)
        return _construct_pinned_dinov3_vitl16(expected_root)

    def reject_weight_download(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "architecture-only DINOv3 construction attempted a weight download"
        )

    with (
        mock.patch.object(
            torch.hub,
            "load",
            side_effect=architecture_only_load,
        ) as patched_hub,
        mock.patch.object(
            torch.hub,
            "load_state_dict_from_url",
            side_effect=reject_weight_download,
        ) as blocked_download,
    ):
        model = model_class(
            dinov3_repo_path=str(expected_root),
            dinov3_weights_path=ARCHITECTURE_ONLY_WEIGHTS_SENTINEL,
            dinov3_model_type="dinov3_vitl16",
            image_size=MODEL_INPUT_SIZE,
            edge_lambda=20.0,
            lora_rank=32,
            lora_alpha=64.0,
        )
    if patched_hub.call_count != 1 or len(author_calls) != 1:
        raise ValueError(
            "DINOv3-IML constructor did not perform exactly one "
            "registered architecture load"
        )
    if blocked_download.call_count:
        raise ValueError("DINOv3 architecture construction requested weights")
    return model


def _validate_model_architecture(model: Any) -> None:
    """Validate the ViT-L/16, LoRA-r32, and segmentation-head topology."""

    import torch
    from torch import nn

    parameters = sum(int(value.numel()) for value in model.parameters())
    buffers = sum(int(value.numel()) for value in model.buffers())
    trainable = sum(
        int(value.numel()) for value in model.parameters() if value.requires_grad
    )
    if parameters != CHECKPOINT["parameters"]:
        raise ValueError(f"DINOv3-IML parameter-count mismatch: {parameters}")
    if buffers != CHECKPOINT["buffers"]:
        raise ValueError(f"DINOv3-IML buffer-count mismatch: {buffers}")
    if trainable != CHECKPOINT["trainable_parameters"]:
        raise ValueError(f"DINOv3-IML trainable-parameter mismatch: {trainable}")
    if (
        getattr(model, "image_size", None) != MODEL_INPUT_SIZE
        or getattr(model, "feat_dim", None) != 1024
    ):
        raise ValueError("DINOv3-IML image size or feature dimension changed")
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 24:
        raise ValueError("DINOv3-IML backbone is not a 24-block ViT-L")
    patch_projection = getattr(
        getattr(backbone, "patch_embed", None),
        "proj",
        None,
    )
    if not isinstance(patch_projection, nn.Conv2d) or tuple(
        patch_projection.weight.shape
    ) != (1024, 3, 16, 16):
        raise ValueError("DINOv3-IML patch projection is not ViT-L/16")

    seg_head = getattr(model, "seg_head", None)
    if not isinstance(seg_head, nn.Sequential) or len(seg_head) != 7:
        raise ValueError("DINOv3-IML segmentation head structure changed")
    expected_head = (
        (0, nn.Conv2d, (512, 1024, 3, 3)),
        (1, nn.BatchNorm2d, (512,)),
        (2, nn.ReLU, None),
        (3, nn.Conv2d, (256, 512, 3, 3)),
        (4, nn.BatchNorm2d, (256,)),
        (5, nn.ReLU, None),
        (6, nn.Conv2d, (1, 256, 1, 1)),
    )
    for index, expected_class, expected_shape in expected_head:
        layer = seg_head[index]
        if not isinstance(layer, expected_class):
            raise ValueError(f"DINOv3-IML seg_head layer {index} type changed")
        if expected_shape is not None:
            weight = getattr(layer, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise ValueError(f"DINOv3-IML seg_head layer {index} has no weight")
            if isinstance(layer, nn.BatchNorm2d):
                actual_shape = tuple(weight.shape)
            else:
                actual_shape = tuple(weight.shape)
            if actual_shape != expected_shape:
                raise ValueError(f"DINOv3-IML seg_head layer {index} shape changed")

    lora_modules = []
    for block_index, block in enumerate(blocks):
        qkv = getattr(getattr(block, "attn", None), "qkv", None)
        lora_a = getattr(qkv, "lora_A", None)
        lora_b = getattr(qkv, "lora_B", None)
        if (
            lora_a is None
            or lora_b is None
            or "default" not in lora_a
            or "default" not in lora_b
            or tuple(lora_a["default"].weight.shape) != (32, 1024)
            or tuple(lora_b["default"].weight.shape) != (3072, 32)
        ):
            raise ValueError(f"DINOv3-IML block {block_index} LoRA-r32 layout changed")
        if (
            int(qkv.r["default"]) != 32
            or float(qkv.lora_alpha["default"]) != 64.0
            or float(qkv.scaling["default"]) != 2.0
        ):
            raise ValueError(f"DINOv3-IML block {block_index} LoRA config changed")
        dropout = qkv.lora_dropout["default"]
        if not (
            isinstance(dropout, nn.Identity)
            or (isinstance(dropout, nn.Dropout) and float(dropout.p) == 0.0)
        ):
            raise ValueError(f"DINOv3-IML block {block_index} LoRA dropout changed")
        lora_modules.append(qkv)
    if len(lora_modules) != 24:
        raise ValueError("DINOv3-IML does not contain 24 QKV LoRA adapters")

    for name, parameter in model.named_parameters():
        should_train = ".lora_" in name or name.startswith("seg_head.")
        if parameter.requires_grad != should_train:
            raise ValueError(f"DINOv3-IML frozen/trainable partition changed at {name}")


def load_model(
    *,
    dinov3_iml_root: Path,
    dinov3_root: Path,
    checkpoint_path: Path,
    device_name: str,
) -> tuple[Any, Any]:
    """Construct pinned architecture, strict-load full state, and move once."""

    import torch

    _runtime_contract(device_name)
    _verify_static_contract(
        dinov3_iml_root=dinov3_iml_root,
        dinov3_root=dinov3_root,
        checkpoint_path=checkpoint_path,
    )
    payload = _load_checkpoint_payload(
        path=checkpoint_path,
        contract=CHECKPOINT,
        label="DINOv3-IML",
    )
    _require_module_tree_origin("dinov3", dinov3_root)
    model_class = _load_author_model_class(dinov3_iml_root)
    model = _construct_author_model(
        model_class=model_class,
        dinov3_root=dinov3_root,
    )
    _require_module_tree_origin("dinov3", dinov3_root)
    backbone_source = inspect.getsourcefile(type(model.backbone))
    expected_backbone_source = (
        dinov3_root / "dinov3" / "models" / "vision_transformer.py"
    ).resolve()
    if (
        backbone_source is None
        or Path(backbone_source).resolve() != expected_backbone_source
    ):
        raise ValueError("constructed DINOv3 backbone source is not pinned")
    _validate_model_architecture(model)
    model = _strict_load_checkpoint_state(
        module=model,
        payload=payload,
        label="DINOv3-IML",
    )
    del payload
    gc.collect()
    device = torch.device(device_name)
    model.to(device).eval()
    return model, device


def infer_one(
    model: Any,
    device: Any,
    image_array: np.ndarray,
    *,
    native_width: int,
    native_height: int,
) -> tuple[dict[str, np.ndarray], int, float]:
    """Call the author's ``predict`` once and hook its seg_head once."""

    import torch

    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("DINOv3-IML input tensor shape changed unexpectedly")
    captured: list[Any] = []

    def capture_seg_head(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError("DINOv3-IML seg_head returned a non-tensor")
        captured.append(output.detach().clone())

    handle = model.seg_head.register_forward_hook(capture_seg_head)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            probability = model.predict(image)
    finally:
        handle.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0
    if len(captured) != 1:
        raise ValueError(
            f"DINOv3-IML seg_head was called {len(captured)} times, expected 1"
        )
    processed = postprocess_outputs(
        probability,
        captured[0],
        native_width=native_width,
        native_height=native_height,
    )
    return processed, peak_bytes, latency_ms


def _load_target(
    row: dict[str, Any],
    repo_root: Path,
    width: int,
    height: int,
) -> np.ndarray:
    if row.get("gt_mask_kind") == "all_zero":
        return np.zeros((height, width), dtype=bool)
    mask_value = row.get("gt_mask_path")
    expected_sha = row.get("gt_mask_sha256")
    if not isinstance(mask_value, str) or not isinstance(expected_sha, str):
        raise ValueError("forged input has no valid GT mask metadata")
    mask_path = _anchored(Path(mask_value), repo_root)
    _verify_runtime_file(mask_path, expected_sha, "ground-truth mask")
    with Image.open(mask_path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    if target.shape != (height, width):
        raise ValueError(f"GT mask shape mismatch: {target.shape} != {(height, width)}")
    return target


def model_space_target(target: np.ndarray) -> np.ndarray:
    """Resize canonical native GT to 512 with lossless class semantics."""

    truth = np.asarray(target, dtype=np.uint8)
    if truth.ndim != 2:
        raise ValueError("DINOv3-IML GT target must be two-dimensional")
    resized = np.asarray(
        Image.fromarray(truth, mode="L").resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    return np.ascontiguousarray(resized > 0)


def _selection_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


def _complete_pair_count(rows: list[dict[str, Any]]) -> int:
    kinds_by_task: dict[str, set[str]] = {}
    for row in rows:
        kinds_by_task.setdefault(str(row["task_id"]), set()).add(str(row["kind"]))
    return sum(kinds == {"real", "forged"} for kinds in kinds_by_task.values())


def build_run_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    dataset_manifest_path: Path,
    release: dict[str, Any],
    inputs_path: Path,
    selected: list[dict[str, Any]],
    dinov3_iml_root: Path,
    dinov3_root: Path,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    ordered_inputs = _selection_contract(selected)
    runtime_contract = _runtime_contract(args.device)
    immutable = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "input": {
            "dataset_id": release["dataset_id"],
            "dataset_manifest": repo_relative(
                dataset_manifest_path,
                repo_root,
            ),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "dataset_contract_sha256": release["contract_sha256"],
            "inputs_manifest": repo_relative(inputs_path, repo_root),
            "inputs_sha256": release["inputs_sha256"],
            "selection_mode": (
                "explicit_single_sample_preflight"
                if getattr(args, "sample_id", None) is not None
                else "complete_leading_pairs"
            ),
            "selection_sha256": hashlib.sha256(
                stable_json(ordered_inputs).encode("utf-8")
            ).hexdigest(),
            "encoding": release["jpeg"],
        },
        "ordered_inputs": ordered_inputs,
        "runtime_contract": runtime_contract,
        "model": {
            "name": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "repo_url": MODEL_REPO_URL,
            "source_root": str(dinov3_iml_root),
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_pin_role": (
                "operational_reproduction_pin_not_claimed_training_commit"
            ),
            "source_tracked_clean": not bool(
                _git_value(
                    dinov3_iml_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "source_files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in SOURCE_FILES.items()
            ],
            "dinov3_architecture_source": {
                "name": "Meta DINOv3",
                "repo_url": DINOV3_REPO_URL,
                "source_root": str(dinov3_root),
                "source_commit": DINOV3_SOURCE_COMMIT,
                "source_pin_rule": (
                    "latest_observed_commit_before_checkpoint_last_modified"
                ),
                "source_pin_role": (
                    "architecture_reproduction_pin_not_claimed_training_commit"
                ),
                "source_tracked_clean": not bool(
                    _git_value(
                        dinov3_root,
                        "status",
                        "--short",
                        "--untracked-files=no",
                    )
                ),
                "source_files": [
                    {"path": path, "sha256": sha256}
                    for path, sha256 in DINOV3_SOURCE_FILES.items()
                ],
                "pretrained": False,
                "separate_backbone_weights_loaded": False,
                "role": "architecture_only",
                "license": {
                    "path": "LICENSE.md",
                    "sha256": DINOV3_SOURCE_FILES["LICENSE.md"],
                    "spdx": None,
                    "name": "DINOv3 License Agreement",
                },
            },
            "variant": "official_CAT_ViT-L16_LoRA-r32_checkpoint-48",
            "license": {
                "path": "LICENSE",
                "sha256": SOURCE_FILES["LICENSE"],
                "spdx": "MIT",
                "scope": "DINOv3-IML_repository_code_only",
                "checkpoint_license": "not_separately_stated_by_release",
            },
            "checkpoint": {
                **CHECKPOINT,
                "path": str(checkpoint_path),
                "strict_load": True,
                "safe_weights_only_load": True,
                "safe_globals": ["argparse.Namespace"],
                "container_selection": "top_level_model_only",
                "schema_fallbacks": False,
                "prefix_rewrites": False,
                "full_state_includes_backbone_lora_and_seg_head": True,
                "separate_backbone_weights_required": False,
            },
            "checkpoint_selection": {
                "selected_before_claimforge_evaluation": True,
                "claimforge_used_for_selection": False,
                "paper_rule": ("maximum_mean_F1_across_four_external_test_sets"),
                "disclosure": "external_test_set_selected_checkpoint",
            },
            "constructor": {
                "class": ("models.dinov3_forensics_lora." "DINOv3ForensicsLoRA"),
                "dinov3_model_type": "dinov3_vitl16",
                "image_size": MODEL_INPUT_SIZE,
                "edge_lambda": 20.0,
                "lora_rank": 32,
                "lora_alpha": 64.0,
                "lora_target_modules": ["qkv"],
                "lora_dropout": 0.0,
                "lora_bias": "none",
                "torch_hub_author_calls": 1,
                "torch_hub_author_call_weights": (ARCHITECTURE_ONLY_WEIGHTS_SENTINEL),
                "torch_hub_substitution": (
                    "direct_pinned_Meta_backbones_dinov3_vitl16_" "pretrained_false"
                ),
                "weight_downloads_blocked": True,
                "author_from_pretrained_used": False,
                "lora_merged": False,
            },
            "parameter_count": CHECKPOINT["parameters"],
            "buffer_elements": CHECKPOINT["buffers"],
            "trainable_parameter_count": CHECKPOINT["trainable_parameters"],
            "supports_image_level_t1": False,
            "image_score_source": None,
            "supports_pixel_level_t2": True,
            "primary_localization_output": (
                "author_predict_float32_sigmoid_probability"
            ),
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "Pillow.Image.open.convert_RGB",
            "channel_order": "RGB",
            "input_geometry": (
                "direct_stretch_to_512x512_without_aspect_ratio_preservation"
            ),
            "preprocess_protocol": PREPROCESS_PROTOCOL,
            "resize": "Pillow.Image.resize",
            "resize_interpolation": "Pillow.Image.Resampling.BILINEAR",
            "resize_box": None,
            "resize_reducing_gap": None,
            "input_crop": None,
            "input_reencode": False,
            "normalization": {
                "scale": "float32_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "official_model_output": {
                "seg_head_logits_shape": [
                    1,
                    1,
                    INTERNAL_LOGIT_SIZE,
                    INTERNAL_LOGIT_SIZE,
                ],
                "logit_resize": MODEL_LOGIT_RESIZE,
                "resized_logits_shape": [
                    1,
                    1,
                    MODEL_INPUT_SIZE,
                    MODEL_INPUT_SIZE,
                ],
                "probability": "single_sigmoid_after_logit_resize",
                "captured_by": LOGIT_CAPTURE,
                "author_predict_calls_per_image": 1,
            },
            "native_compatibility_adapter": {
                "purpose": ("CLAIMFORGE cross-method native-resolution comparison"),
                "source": "official_model_512_probability_not_logits",
                "operation": NATIVE_RESTORE,
                "mode": "bilinear",
                "align_corners": False,
                "threshold_after_restore": True,
                "official_model_space_retained_as_auxiliary": True,
            },
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "strict_greater_than",
            "test_time_augmentation": False,
            "ensemble": False,
            "forward_passes_per_image": 1,
        },
        "metrics": {
            "task": "T2_pixel_localization_only",
            "positive_class": "manipulated_pixel",
            "t1_policy": "unsupported_no_derived_image_score",
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_512",
            "mask_threshold": args.mask_threshold,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_gt_resize": ("Pillow.Image.Resampling.NEAREST_to_512x512"),
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "raw_logits_model_32": {
                "format": "npy",
                "dtype": "float32",
                "shape": [INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE],
                "semantics": "official_seg_head_pre_resize_logits",
                "captured_from": LOGIT_CAPTURE,
            },
            "raw_logits_model_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "semantics": "official_bilinear_resized_pre_sigmoid_logits",
                "derivation": MODEL_LOGIT_RESIZE,
            },
            "score_maps_model_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "semantics": "official_author_predict_sigmoid_probability",
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": ("model_512_probability_restored_to_native"),
                "restore": NATIVE_RESTORE,
            },
            "masks_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "score_map_native > 0.5",
            },
        },
        "expected_complete_pairs": _complete_pair_count(selected),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": [
            {
                "path": repo_relative(Path(__file__), repo_root),
                "sha256": sha256_file(Path(__file__)),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("dinov3_iml_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("dinov3_iml_metrics.py")
                ),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("mesorch_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(Path(__file__).with_name("mesorch_metrics.py")),
                "role": "shared_512_native_t2_numerical_kernel",
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("common.py"),
                    repo_root,
                ),
                "sha256": sha256_file(Path(__file__).with_name("common.py")),
            },
        ],
    }
    return {
        **immutable,
        "fingerprint": _manifest_fingerprint(immutable),
        "created_at": utc_now(),
        "adapter": {
            "path": repo_relative(Path(__file__), repo_root),
            "sha256": sha256_file(Path(__file__)),
            "repo_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        },
        "environment": runtime_contract,
    }


def _write_or_validate_run_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest.get("fingerprint"):
            raise ValueError(
                f"existing run manifest is incompatible with this run: {path}"
            )
        return
    atomic_write_json(path, manifest)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_npy_artifact(
    *,
    sample_id: str,
    label: str,
    path: Path,
    row: Mapping[str, Any],
    path_field: str,
    sha_field: str,
    shape_field: str,
    dtype_field: str,
    expected_shape: list[int],
    repo_root: Path,
) -> None:
    if row.get(path_field) != repo_relative(path, repo_root):
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} artifact path"
        )
    if not path.is_file():
        raise ValueError(
            f"existing result {sample_id} is missing {label} artifact: {path}"
        )
    recorded_sha = row.get(sha_field)
    if not _valid_sha256(recorded_sha):
        raise ValueError(
            f"existing result {sample_id} has invalid {label} SHA-256 metadata"
        )
    if sha256_file(path) != recorded_sha:
        raise ValueError(f"existing result {sample_id} has modified {label} artifact")
    if row.get(shape_field) != expected_shape:
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} shape metadata"
        )
    if row.get(dtype_field) != "float32":
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} dtype metadata"
        )
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        actual_shape = list(array.shape)
        actual_dtype = str(array.dtype)
    except Exception as exc:
        raise ValueError(
            f"existing result {sample_id} has unreadable {label} artifact"
        ) from exc
    if actual_shape != expected_shape or actual_dtype != "float32":
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} artifact schema"
        )


def _validate_mask_artifact(
    *,
    sample_id: str,
    path: Path,
    row: Mapping[str, Any],
    expected_shape: list[int],
    repo_root: Path,
) -> None:
    label = "native mask"
    if row.get("mask_path") != repo_relative(path, repo_root):
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} artifact path"
        )
    if not path.is_file():
        raise ValueError(
            f"existing result {sample_id} is missing {label} artifact: {path}"
        )
    recorded_sha = row.get("mask_sha256")
    if not _valid_sha256(recorded_sha):
        raise ValueError(
            f"existing result {sample_id} has invalid {label} SHA-256 metadata"
        )
    if sha256_file(path) != recorded_sha:
        raise ValueError(f"existing result {sample_id} has modified {label} artifact")
    if row.get("mask_shape") != expected_shape or row.get("mask_dtype") != "uint8":
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} metadata"
        )
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG":
                raise ValueError("mask artifact is not PNG")
            array = np.asarray(opened)
        if list(array.shape) != expected_shape or array.dtype != np.uint8:
            raise ValueError("mask artifact schema changed")
        if not np.isin(array, (0, 255)).all():
            raise ValueError("mask artifact is not binary")
    except Exception as exc:
        raise ValueError(
            f"existing result {sample_id} has unreadable {label} artifact"
        ) from exc


FORBIDDEN_T1_FIELDS = frozenset(
    {
        "score",
        "decision",
        "image_score",
        "image_decision",
        "classification",
        "classification_metrics",
        "detection",
        "detection_metrics",
        "image_auroc",
        "image_average_precision",
        "paired_score_delta",
        "paired_ranking_accuracy",
    }
)


def _find_forbidden_t1_fields(
    value: Any,
    *,
    path: str = "$",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_T1_FIELDS:
                found.append(child_path)
            found.extend(_find_forbidden_t1_fields(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(
                _find_forbidden_t1_fields(
                    child,
                    path=f"{path}[{index}]",
                )
            )
    return found


def _validate_resume_rows(
    latest: Mapping[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
    *,
    repo_root: Path,
    artifact_dir: Path,
) -> None:
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    for sample_id, row in latest.items():
        if sample_id not in selected_by_id:
            continue
        if row.get("run_manifest_fingerprint") != manifest_fingerprint:
            raise ValueError(
                f"existing result {sample_id} has incompatible run fingerprint"
            )
        forbidden = _find_forbidden_t1_fields(row)
        if forbidden:
            raise ValueError(
                f"existing result {sample_id} contains forbidden T1 fields: "
                f"{forbidden}"
            )
        if (
            row.get("valid_for_t1") is not False
            or row.get("valid_for_t2") is not True
            or row.get("t1_policy") != "unsupported_no_derived_image_score"
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible task policy"
            )
        expected = selected_by_id[sample_id]
        if (
            row.get("image_sha256") != expected.get("canonical_sha256")
            or row.get("task_id") != expected.get("task_id")
            or row.get("kind") != expected.get("kind")
            or row.get("checkpoint_sha256") != CHECKPOINT["sha256"]
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible input identity"
            )
        if row.get("status") != "ok":
            continue
        if (
            row.get("mask_threshold") != MASK_THRESHOLD
            or row.get("mask_threshold_operator") != ">"
            or row.get("raw_logits_capture") != LOGIT_CAPTURE
            or row.get("resized_logits_derivation") != MODEL_LOGIT_RESIZE
            or row.get("score_map_native_restore") != NATIVE_RESTORE
            or row.get("score_map_native_source") != "official_model_512_probability"
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible probability chain"
            )
        width = int(expected["width"])
        height = int(expected["height"])
        native_shape = [height, width]
        contracts = (
            (
                "raw 32 logits",
                artifact_dir / "raw_logits_model_32" / f"{sample_id}.npy",
                "raw_logits_model_path",
                "raw_logits_model_sha256",
                "raw_logits_model_shape",
                "raw_logits_model_dtype",
                [INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE],
            ),
            (
                "resized 512 logits",
                artifact_dir / "raw_logits_model_512" / f"{sample_id}.npy",
                "resized_logits_model_path",
                "resized_logits_model_sha256",
                "resized_logits_model_shape",
                "resized_logits_model_dtype",
                [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            ),
            (
                "model probability",
                artifact_dir / "score_maps_model_512" / f"{sample_id}.npy",
                "score_map_model_path",
                "score_map_model_sha256",
                "score_map_model_shape",
                "score_map_model_dtype",
                [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            ),
            (
                "native probability",
                artifact_dir / "score_maps_native" / f"{sample_id}.npy",
                "score_map_path",
                "score_map_sha256",
                "score_map_shape",
                "score_map_dtype",
                native_shape,
            ),
        )
        for (
            label,
            path,
            path_field,
            sha_field,
            shape_field,
            dtype_field,
            shape,
        ) in contracts:
            _validate_npy_artifact(
                sample_id=sample_id,
                label=label,
                path=path,
                row=row,
                path_field=path_field,
                sha_field=sha_field,
                shape_field=shape_field,
                dtype_field=dtype_field,
                expected_shape=shape,
                repo_root=repo_root,
            )
        _validate_mask_artifact(
            sample_id=sample_id,
            path=artifact_dir / "masks_native" / f"{sample_id}.png",
            row=row,
            expected_shape=native_shape,
            repo_root=repo_root,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("official DINOv3-IML mask threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    dinov3_iml_root = args.dinov3_iml_root.resolve()
    dinov3_root = args.dinov3_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        (
            args.artifact_dir
            if args.artifact_dir is not None
            else Path(f"outputs/opensource/dinov3_iml/{args.run_id}")
        ),
        repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.jsonl"
    run_manifest_path = output_dir / f"{args.run_id}.run_manifest.json"
    summary_path = output_dir / f"{args.run_id}.summary.json"

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(
        all_rows,
        args.pair_limit,
        getattr(args, "sample_id", None),
    )
    for row in selected:
        image_path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            image_path,
            str(row["canonical_sha256"]),
            f"canonical input {row['sample_id']}",
        )
    _verify_static_contract(
        dinov3_iml_root=dinov3_iml_root,
        dinov3_root=dinov3_root,
        checkpoint_path=checkpoint_path,
    )

    configure_determinism(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    run_manifest = build_run_manifest(
        args=args,
        repo_root=repo_root,
        dataset_manifest_path=dataset_manifest_path,
        release=release,
        inputs_path=inputs_path,
        selected=selected,
        dinov3_iml_root=dinov3_iml_root,
        dinov3_root=dinov3_root,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(run_manifest_path, run_manifest)
    existing = read_latest_by_id(output_path)
    _validate_resume_rows(
        existing,
        selected,
        run_manifest["fingerprint"],
        repo_root=repo_root,
        artifact_dir=artifact_dir,
    )
    pending = [
        row
        for row in selected
        if existing.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"DINOv3-IML run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            dinov3_iml_root=dinov3_iml_root,
            dinov3_root=dinov3_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(
            "loaded official DINOv3-IML CAT checkpoint epoch 48 "
            f"{CHECKPOINT['sha256'][:12]} on {device}",
            flush=True,
        )
        try:
            for index, input_row in enumerate(pending, start=1):
                sample_id = str(input_row["sample_id"])
                edit_region = input_row.get("edit_region_xyxy")
                identity = {
                    "schema_version": "opensource_result_v1",
                    "run_id": args.run_id,
                    "run_manifest_fingerprint": run_manifest["fingerprint"],
                    "input_manifest_sha256": release["inputs_sha256"],
                    "id": sample_id,
                    "rank": int(input_row["rank"]),
                    "task_id": str(input_row["task_id"]),
                    "pair_rank": int(input_row["pair_rank"]),
                    "domain": str(input_row["domain"]),
                    "kind": str(input_row["kind"]),
                    "label": int(input_row["label"]),
                    "image_path": str(input_row["canonical_path"]),
                    "image_sha256": str(input_row["canonical_sha256"]),
                    "image_size": [
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ],
                    "gt_mask_kind": str(input_row["gt_mask_kind"]),
                    "gt_mask_sha256": input_row.get("gt_mask_sha256"),
                    "edit_region_xyxy": (
                        [int(value) for value in edit_region]
                        if isinstance(edit_region, list)
                        else None
                    ),
                    "model": MODEL_NAME,
                    "model_slug": MODEL_SLUG,
                    "model_source_commit": MODEL_SOURCE_COMMIT,
                    "dinov3_source_commit": DINOV3_SOURCE_COMMIT,
                    "checkpoint_sha256": CHECKPOINT["sha256"],
                    "checkpoint_epoch": CHECKPOINT["epoch"],
                    "valid_for_t1": False,
                    "valid_for_t2": True,
                    "t1_policy": "unsupported_no_derived_image_score",
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    image_array, native_size, preprocess = preprocess_image(image_path)
                    width, height = native_size
                    if (width, height) != (
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ):
                        raise ValueError("canonical image dimensions changed")
                    processed, peak_bytes, latency_ms = infer_one(
                        model,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                    )
                    logits_32 = processed["raw_logits_model_32"]
                    logits_512 = processed["raw_logits_model_512"]
                    probability_model = processed["probability_model_512"]
                    probability_native = processed["probability_native"]
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = model_space_target(target_native)
                    include_ap = input_row["kind"] == "forged"
                    localization = {
                        "model_512": binary_pixel_metrics_strict(
                            probability_model,
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_pixel_metrics_strict(
                            probability_native,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    logits_32_path = (
                        artifact_dir / "raw_logits_model_32" / f"{sample_id}.npy"
                    )
                    logits_512_path = (
                        artifact_dir / "raw_logits_model_512" / f"{sample_id}.npy"
                    )
                    model_score_path = (
                        artifact_dir / "score_maps_model_512" / f"{sample_id}.npy"
                    )
                    native_score_path = (
                        artifact_dir / "score_maps_native" / f"{sample_id}.npy"
                    )
                    native_mask_path = (
                        artifact_dir / "masks_native" / f"{sample_id}.png"
                    )
                    _atomic_save_npy(logits_32_path, logits_32)
                    _atomic_save_npy(logits_512_path, logits_512)
                    _atomic_save_npy(model_score_path, probability_model)
                    _atomic_save_npy(native_score_path, probability_native)
                    _atomic_save_mask(
                        native_mask_path,
                        probability_native > args.mask_threshold,
                    )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "raw_logits_model_path": repo_relative(
                            logits_32_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(logits_32_path),
                        "raw_logits_model_shape": list(logits_32.shape),
                        "raw_logits_model_dtype": str(logits_32.dtype),
                        "raw_logits_model_semantics": (
                            "official_seg_head_pre_resize_logits"
                        ),
                        "raw_logits_capture": LOGIT_CAPTURE,
                        "resized_logits_model_path": repo_relative(
                            logits_512_path,
                            repo_root,
                        ),
                        "resized_logits_model_sha256": sha256_file(logits_512_path),
                        "resized_logits_model_shape": list(logits_512.shape),
                        "resized_logits_model_dtype": str(logits_512.dtype),
                        "resized_logits_model_semantics": (
                            "official_bilinear_resized_pre_sigmoid_logits"
                        ),
                        "resized_logits_derivation": MODEL_LOGIT_RESIZE,
                        "score_map_model_path": repo_relative(
                            model_score_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(model_score_path),
                        "score_map_model_shape": list(probability_model.shape),
                        "score_map_model_dtype": str(probability_model.dtype),
                        "score_map_model_semantics": (
                            "official_author_predict_sigmoid_probability"
                        ),
                        "score_map_path": repo_relative(
                            native_score_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(native_score_path),
                        "score_map_shape": list(probability_native.shape),
                        "score_map_dtype": str(probability_native.dtype),
                        "score_map_semantics": (
                            "model_512_probability_restored_to_native"
                        ),
                        "score_map_native_source": ("official_model_512_probability"),
                        "score_map_native_restore": NATIVE_RESTORE,
                        "mask_path": repo_relative(
                            native_mask_path,
                            repo_root,
                        ),
                        "mask_sha256": sha256_file(native_mask_path),
                        "mask_shape": list(probability_native.shape),
                        "mask_dtype": "uint8",
                        "mask_threshold": args.mask_threshold,
                        "mask_threshold_operator": ">",
                        "localization": localization,
                        "preprocess": preprocess,
                        "latency_ms": round(latency_ms, 3),
                        "peak_cuda_memory_bytes": peak_bytes,
                        "completed_at": utc_now(),
                    }
                except Exception as exc:
                    row = {
                        **identity,
                        "status": "error",
                        "valid_for_metrics": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(limit=8)[-8000:],
                        "completed_at": utc_now(),
                    }
                append_jsonl(output_path, row)
                if row["status"] == "ok":
                    native_metrics = row["localization"]["native"]
                    detail = (
                        f" f1={native_metrics.get('f1')} "
                        f"positive={native_metrics['predicted_positive_fraction']:.6f} "
                        f"latency={row['latency_ms']:.1f}ms"
                    )
                else:
                    detail = f" {row['error_type']}: {row['error_message']}"
                print(
                    f"[{index}/{len(pending)}] "
                    f"{input_row['task_id']} {input_row['kind']}: "
                    f"{row['status']}{detail}",
                    flush=True,
                )
                if row["status"] != "ok" and args.fail_fast:
                    raise RuntimeError(
                        f"DINOv3-IML failed for {sample_id}: " f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_dinov3_iml_results(
        result_rows,
        selected,
        mask_threshold=args.mask_threshold,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "model_source_commit": MODEL_SOURCE_COMMIT,
            "dinov3_source_commit": DINOV3_SOURCE_COMMIT,
            "checkpoint_sha256": CHECKPOINT["sha256"],
            "checkpoint_epoch": CHECKPOINT["epoch"],
            "input_manifest_sha256": release["inputs_sha256"],
            "run_manifest_fingerprint": run_manifest["fingerprint"],
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(summary_path, summary)
    coverage = summary["coverage"]
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.allow_errors and (
        coverage["valid_images"] != coverage["expected_images"]
        or coverage["error_images"]
        or coverage["missing_images"]
    ):
        raise RuntimeError(f"incomplete DINOv3-IML run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--dinov3-iml-root",
        type=Path,
        default=DEFAULT_DINOV3_IML_ROOT,
    )
    parser.add_argument(
        "--dinov3-root",
        type=Path,
        default=DEFAULT_DINOV3_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/dinov3_iml"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pair-limit", type=int)
    selection.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
