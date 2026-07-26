#!/usr/bin/env python3
"""Run the official RelayFormer image-only checkpoint on CLAIMFORGE inputs.

RelayFormer is a native pixel-localization model.  It does not expose an
image-level classification head, so this adapter reports T2 only and never
derives a synthetic T1 score or decision from its localization map.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Iterator, Mapping
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
from eval.opensource.relayformer_metrics import (
    binary_pixel_metrics_strict,
    summarize_relayformer_results,
)


MODEL_NAME = "RelayFormer"
MODEL_SLUG = "relayformer_checkpoint164_paper_v3_official"
MODEL_REPO_URL = "https://github.com/WenOOI/RelayFormer"
MODEL_SOURCE_COMMIT = "3fc863c7691d93fb5b11ca8e12e3a214d771e384"
MODEL_INPUT_SIZE = 1024
MASK_THRESHOLD = 0.5
PREPROCESS_PROTOCOL = "paper_v3_long_edge_1024_top_left_zero_pad"
LOGIT_CAPTURE = (
    "temporary_instance_method_wrapper_output_from_official_"
    "assemble_and_decode"
)
NATIVE_RESTORE = (
    "crop_right_bottom_padding_then_bilinear_align_corners_false_if_downscaled"
)

SOURCE_FILES = {
    "README.md": "0e99a828e2f2fba15e5c0cf844d1ec740ba4261ef61efec6c4eab9b0e4ae4119",
    "LICENSE": "49b8601bc931cb5cff60edc3204b342519dc6436d7f3d1aeb13da594006aa97a",
    "models/RelayFormer.py": (
        "c6dabe3c2ca692b8bce85090a668d2e2c8a75b154c02500b333965f51d418abc"
    ),
    "models/GLoRA_vit.py": (
        "f70a4a7880e092c68cebc472b55633ebb0c7f3e2b79df8f940f8e17b6e7ccd51"
    ),
    "datasets/inference_dataset.py": (
        "cefe47a02548032879eac1baa6a5af8af94de0994962ebaf7ebfb7ef35648985"
    ),
    "datasets/hybrid_dataset.py": (
        "10db75a7767ded419477593d84150a8b15395a3df176246ee123e07889daf164"
    ),
    "infer.py": "4c13b383c0c3dda2e1314e014d86a5021ef4ea31a008cc7bc151b03363d180e5",
    "test.py": "87a472f2240b03bd5d0544dc4c76109bcdc24f6738a2b2afb32aedec2bc6366c",
    "requirements.txt": (
        "d3c4bd72301ad08fe621eecc4171e1660247ee03c2e4d76e1ad92f283e630bfb"
    ),
    "scripts/test.sh": (
        "fb9ad4f96b4be0ec8a7b5754ff16978f039ce04c09712833505d8e9c61c07b5c"
    ),
}

CHECKPOINT = {
    "provider": "official_author_hugging_face",
    "model_repo": "Wenn11/RelayFormer",
    "revision": "9ef11f4ac16ac50e2684d4af522e442cb290e2c1",
    "original_filename": "checkpoint-164.pth",
    "download_url": (
        "https://huggingface.co/Wenn11/RelayFormer/resolve/"
        "9ef11f4ac16ac50e2684d4af522e442cb290e2c1/checkpoint-164.pth"
    ),
    "bytes": 1_102_625_388,
    "sha256": "00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090",
    "container": "mapping_with_model_optimizer_epoch_scaler_args",
    "top_level_keys": ["model", "optimizer", "epoch", "scaler", "args"],
    "epoch": 164,
    "args_type": "argparse.Namespace",
    "state_container": "collections.OrderedDict",
    "state_keys": 410,
    "state_elements": 91_909_179,
    "tensor_bytes": 367_636_716,
    "state_dtypes": {"torch.float32": 410},
    "parameters": 91_909_179,
    "buffers": 786_435,
    "checkpoint_state_buffer_elements": 0,
}

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RELAYFORMER_ROOT = Path(
    "/root/.cache/claimforge/third_party/RelayFormer"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/relayformer/checkpoint-164.pth"
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


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _installed_module_contract(
    module_name: str,
    distribution_names: tuple[str, ...],
) -> dict[str, Any]:
    """Bind a numerical module to files owned by an allowed distribution."""

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
            Path(distribution.locate_file(item)).resolve() == source
            for item in files
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
        "module_version": (
            str(module_version) if module_version is not None else None
        ),
        "source": str(source),
        "distributions": sorted(
            matches,
            key=lambda item: (item["name"].lower(), item["version"]),
        ),
    }


def _runtime_contract(device_name: str) -> dict[str, Any]:
    """Capture all numerical dependencies in the immutable run fingerprint."""

    import torch

    packages = {
        "torch": _installed_module_contract("torch", ("torch",)),
        "torchvision": _installed_module_contract(
            "torchvision",
            ("torchvision",),
        ),
        "timm": _installed_module_contract("timm", ("timm",)),
        "IMDLBenCo": _installed_module_contract(
            "IMDLBenCo",
            ("IMDLBenCo",),
        ),
        "rotary-embedding-torch": _installed_module_contract(
            "rotary_embedding_torch",
            ("rotary-embedding-torch",),
        ),
        "numpy": _installed_module_contract("numpy", ("numpy",)),
        "Pillow": _installed_module_contract("PIL", ("Pillow",)),
        "albumentations": _installed_module_contract(
            "albumentations",
            ("albumentations",),
        ),
        "cv2": _installed_module_contract(
            "cv2",
            (
                "opencv-python-headless",
                "opencv-python",
                "opencv-contrib-python-headless",
                "opencv-contrib-python",
            ),
        ),
        "scikit-learn": _installed_module_contract(
            "sklearn",
            ("scikit-learn",),
        ),
    }
    critical_submodules = {
        "IMDLBenCo.registry": _installed_module_contract(
            "IMDLBenCo.registry",
            ("IMDLBenCo",),
        ),
        "timm.layers": _installed_module_contract(
            "timm.layers",
            ("timm",),
        ),
    }
    pillow_versions = {
        item["version"]
        for item in packages["Pillow"]["distributions"]
    }
    if pillow_versions != {"11.1.0"}:
        raise ValueError(
            "RelayFormer frozen preprocessing requires Pillow 11.1.0, got "
            f"{sorted(pillow_versions)}"
        )
    requested_device = torch.device(device_name)
    cuda_active = (
        requested_device.type == "cuda" and torch.cuda.is_available()
    )
    accelerator: dict[str, Any] = {
        "requested_device": str(requested_device),
        "torch_cuda": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(requested_device)
            if cuda_active
            else None
        ),
        "gpu_capability": (
            list(torch.cuda.get_device_capability(requested_device))
            if cuda_active
            else None
        ),
    }
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": packages,
        "critical_submodules": critical_submodules,
        "accelerator": accelerator,
    }


def _manifest_fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


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
            f"{label} byte-size mismatch: "
            f"{actual_bytes} != {contract['bytes']}"
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
) -> list[dict[str, Any]]:
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [row for row in rows if int(row["pair_rank"]) in selected_ranks]
    by_pair: dict[int, set[str]] = {}
    for row in selected:
        by_pair.setdefault(int(row["pair_rank"]), set()).add(str(row["kind"]))
    invalid = {
        rank: kinds
        for rank, kinds in by_pair.items()
        if kinds != {"real", "forged"}
    }
    if invalid:
        raise ValueError(f"canonical selection contains incomplete pairs: {invalid}")
    return selected


def preprocess_image(
    path: Path,
) -> tuple[
    np.ndarray,
    tuple[int, int],
    tuple[int, int],
    dict[str, Any],
]:
    """Replay the frozen paper-v3 long-edge resize and top-left padding."""

    with path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoder_format = opened.format
            image = opened.convert("RGB")
    native_width, native_height = image.size
    if native_width <= 0 or native_height <= 0:
        raise ValueError("RelayFormer input has invalid native dimensions")
    resized = image.copy()
    resized.thumbnail(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.BILINEAR,
        reducing_gap=None,
    )
    resized_width, resized_height = resized.size
    resized_array = np.asarray(resized, dtype=np.uint8)
    if (
        resized_array.shape != (resized_height, resized_width, 3)
        or resized_array.dtype != np.uint8
    ):
        raise ValueError(
            f"unexpected RelayFormer resized image: {resized_array.shape}"
        )
    canvas = np.zeros(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
        dtype=np.uint8,
    )
    canvas[:resized_height, :resized_width] = resized_array
    normalized = canvas.astype(np.float32) / np.float32(255.0)
    normalized = (
        normalized
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected RelayFormer input shape: {chw.shape}")
    if chw.dtype != np.float32 or not np.isfinite(chw).all():
        raise ValueError("RelayFormer normalized input is not finite float32")
    metadata = {
        "protocol": PREPROCESS_PROTOCOL,
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size_wh": [native_width, native_height],
        "scale": min(
            1.0,
            MODEL_INPUT_SIZE / max(native_width, native_height),
        ),
        "resized_size_wh": [resized_width, resized_height],
        "resize_rounding": "pillow_thumbnail_floor_ceil_min_aspect_error",
        "canvas_size_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "padding_ltrb": [
            0,
            0,
            MODEL_INPUT_SIZE - resized_width,
            MODEL_INPUT_SIZE - resized_height,
        ],
        "resize_interpolation": (
            "Pillow.Image.Resampling.BILINEAR_reducing_gap_None"
        ),
        "padding_raw_rgb_value": 0,
        "padding_before_normalization": True,
        "input_crop": None,
        "input_reencode": False,
        "normalization": {
            "scale": "uint8_divide_255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "tensor_shape": list(chw.shape),
        "tensor_dtype": str(chw.dtype),
        "tensor_sha256": _sha256_array(chw),
    }
    return (
        chw,
        (native_width, native_height),
        (resized_width, resized_height),
        metadata,
    )


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
    resized_width: int,
    resized_height: int,
) -> dict[str, np.ndarray]:
    """Verify RelayFormer sigmoid output and restore valid maps to native size."""

    import torch
    from torch.nn import functional as F

    expected_shape = (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    if not isinstance(probability, torch.Tensor) or tuple(probability.shape) != (
        expected_shape
    ):
        raise ValueError(
            "unexpected RelayFormer probability shape: "
            f"{getattr(probability, 'shape', None)}"
        )
    if not isinstance(raw_logits, torch.Tensor) or tuple(raw_logits.shape) != (
        expected_shape
    ):
        raise ValueError(
            "unexpected RelayFormer raw-logit shape: "
            f"{getattr(raw_logits, 'shape', None)}"
        )
    if min(
        native_width,
        native_height,
        resized_width,
        resized_height,
    ) <= 0:
        raise ValueError("RelayFormer dimensions must be positive")
    if (
        resized_width > MODEL_INPUT_SIZE
        or resized_height > MODEL_INPUT_SIZE
        or resized_width > native_width
        or resized_height > native_height
    ):
        raise ValueError("RelayFormer resized dimensions violate protocol")

    expected_probability = torch.sigmoid(raw_logits.float())
    if not torch.allclose(
        probability.float(),
        expected_probability,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "RelayFormer output does not match sigmoid of captured logits"
        )
    valid_logits = raw_logits.float()[
        :,
        :,
        :resized_height,
        :resized_width,
    ]
    valid_probability = probability.float()[
        :,
        :,
        :resized_height,
        :resized_width,
    ]
    if (resized_width, resized_height) == (native_width, native_height):
        native_logits = valid_logits
        native_probability = valid_probability
    else:
        native_logits = F.interpolate(
            valid_logits,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
        native_probability = F.interpolate(
            valid_probability,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )

    result = {
        "raw_logits_model_1024": _float32_map(
            raw_logits[0, 0],
            "RelayFormer model logits",
        ),
        "raw_logits_native": _float32_map(
            native_logits[0, 0],
            "RelayFormer native logits",
        ),
        "probability_model_1024": _float32_map(
            probability[0, 0],
            "RelayFormer model probability",
        ),
        "probability_valid": _float32_map(
            valid_probability[0, 0],
            "RelayFormer valid-content probability",
        ),
        "probability_native": _float32_map(
            native_probability[0, 0],
            "RelayFormer native probability",
        ),
    }
    for label in (
        "probability_model_1024",
        "probability_valid",
        "probability_native",
    ):
        array = result[label]
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"RelayFormer {label} falls outside [0, 1]")
    return result


def _validate_module_counts(
    module: Any,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    parameters = sum(int(value.numel()) for value in module.parameters())
    buffers = sum(int(value.numel()) for value in module.buffers())
    if parameters != int(contract["parameters"]):
        raise ValueError(
            f"{label} parameter-count mismatch: "
            f"{parameters} != {contract['parameters']}"
        )
    if buffers != int(contract["buffers"]):
        raise ValueError(
            f"{label} buffer-count mismatch: {buffers} != {contract['buffers']}"
        )


def _load_checkpoint_payload(
    *,
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    """Safely deserialize and validate the complete released checkpoint."""

    import torch

    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
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
    checkpoint_args = payload["args"]
    expected_args = {
        "model": "RelayFormer",
        "image_size": MODEL_INPUT_SIZE,
        "if_padding": True,
        "if_resizing": False,
    }
    for name, expected in expected_args.items():
        actual = getattr(checkpoint_args, name, None)
        if actual != expected:
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
        int(value.numel()) * int(value.element_size())
        for value in state.values()
    )
    dtype_counts = Counter(str(value.dtype) for value in state.values())
    if elements != int(contract["state_elements"]):
        raise ValueError(f"{label} checkpoint state-element count mismatch")
    if tensor_bytes != int(contract["tensor_bytes"]):
        raise ValueError(f"{label} checkpoint tensor-byte count mismatch")
    if dict(dtype_counts) != dict(contract["state_dtypes"]):
        raise ValueError(f"{label} checkpoint dtype schema mismatch")
    reg_token = state.get("vit.reg_token")
    if (
        not isinstance(reg_token, torch.Tensor)
        or list(reg_token.shape) != [1, 2, 768]
    ):
        raise ValueError(
            f"{label} checkpoint vit.reg_token shape mismatch"
        )
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
    module.load_state_dict(state, strict=True)
    return module


def _verify_static_contract(
    *,
    relayformer_root: Path,
    checkpoint_path: Path,
) -> None:
    source_commit = _git_value(relayformer_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            "RelayFormer source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        relayformer_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("RelayFormer tracked source files have local modifications")
    for relative, expected in SOURCE_FILES.items():
        _verify_runtime_file(
            relayformer_root / relative,
            expected,
            f"RelayFormer source file {relative}",
        )
    _verify_file_contract(
        checkpoint_path,
        CHECKPOINT,
        "RelayFormer official image-only checkpoint 164",
    )


def _require_cached_module_origin(
    module_name: str,
    expected_path: Path,
) -> None:
    """Reject cached modules that bypass the pinned source checkout."""

    cached = sys.modules.get(module_name)
    if cached is None:
        return
    source_value = getattr(cached, "__file__", None)
    if not isinstance(source_value, str):
        raise ValueError(
            f"cached {module_name} module has no verifiable source file"
        )
    actual = Path(source_value).resolve()
    expected = expected_path.resolve()
    if actual != expected:
        raise ValueError(
            f"cached {module_name} module source mismatch: "
            f"{actual} != {expected}"
        )


def load_model(
    *,
    relayformer_root: Path,
    checkpoint_path: Path,
    device_name: str,
) -> tuple[Any, Any]:
    """Construct from pinned source without timm network access, then load strictly."""

    import torch

    # Validate installed packages before placing the upstream repository on
    # sys.path, preventing untracked shadow modules from satisfying imports.
    _runtime_contract(device_name)
    _verify_static_contract(
        relayformer_root=relayformer_root,
        checkpoint_path=checkpoint_path,
    )
    payload = _load_checkpoint_payload(
        path=checkpoint_path,
        contract=CHECKPOINT,
        label="RelayFormer",
    )

    root_text = str(relayformer_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    pinned_modules = {
        "models": relayformer_root / "models" / "__init__.py",
        "models.RelayFormer": (
            relayformer_root / "models" / "RelayFormer.py"
        ),
        "models.GLoRA_vit": (
            relayformer_root / "models" / "GLoRA_vit.py"
        ),
    }
    for module_name, expected_path in pinned_modules.items():
        _require_cached_module_origin(module_name, expected_path)

    with _working_directory(relayformer_root):
        relay_module = importlib.import_module("models.RelayFormer")
        relay_class = getattr(relay_module, "RelayFormer")

        def safe_constructor_load(
            requested_path: Any,
            *,
            map_location: Any = None,
            weights_only: Any = None,
            **kwargs: Any,
        ) -> Mapping[str, Any]:
            if Path(requested_path).resolve() != checkpoint_path.resolve():
                raise ValueError(
                    "RelayFormer constructor attempted an unregistered "
                    f"checkpoint load: {requested_path}"
                )
            if str(map_location) != "cpu" or weights_only is not False or kwargs:
                raise ValueError(
                    "RelayFormer constructor checkpoint-load signature changed"
                )
            return payload

        # Passing a non-empty ViT path makes timm use pretrained=False.  The
        # official constructor still calls torch.load internally, so replace
        # only that call with the already safely loaded payload.
        with mock.patch.object(
            relay_module.torch,
            "load",
            side_effect=safe_constructor_load,
        ) as constructor_load:
            model = relay_class(
                input_size=MODEL_INPUT_SIZE,
                grid_size=2,
                patch_size=528,
                overlap=16,
                feature_patch_size=33,
                feature_overlap=1,
                tokens_per_patch=3,
                vit_pretrain_path=str(checkpoint_path),
            )
        if constructor_load.call_count != 1:
            raise ValueError(
                "RelayFormer constructor did not perform exactly one "
                "registered checkpoint load"
            )

    for module_name, expected_path in pinned_modules.items():
        _require_cached_module_origin(module_name, expected_path)
    class_source = inspect.getsourcefile(relay_class)
    if class_source is None or Path(class_source).resolve() != (
        relayformer_root / "models" / "RelayFormer.py"
    ).resolve():
        raise ValueError("RelayFormer class source does not match pinned source")
    decoder_source = inspect.getsourcefile(type(model.mask_decoder))
    if decoder_source is None or Path(decoder_source).resolve() != (
        relayformer_root / "models" / "RelayFormer.py"
    ).resolve():
        raise ValueError("RelayFormer decoder source does not match pinned source")
    block_source = inspect.getsourcefile(type(model.vit.blocks[0]))
    if block_source is None or Path(block_source).resolve() != (
        relayformer_root / "models" / "GLoRA_vit.py"
    ).resolve():
        raise ValueError("RelayFormer block source does not match pinned source")

    _validate_module_counts(model, CHECKPOINT, "RelayFormer")
    model = _strict_load_checkpoint_state(
        module=model,
        payload=payload,
        label="RelayFormer",
    )
    model.merge_lora()
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
    resized_width: int,
    resized_height: int,
) -> tuple[dict[str, np.ndarray], int, float]:
    """Run one official pure forward and capture assemble-and-decode logits."""

    import torch

    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("RelayFormer input tensor shape changed unexpectedly")
    origin_shape = torch.tensor(
        [[resized_height, resized_width]],
        dtype=torch.int64,
        device="cpu",
    )
    clip_len = torch.tensor([1], dtype=torch.int64, device="cpu")
    captured: list[Any] = []
    original_assemble = model.assemble_and_decode

    def capture_assemble_output(*args: Any, **kwargs: Any) -> Any:
        logits = original_assemble(*args, **kwargs)
        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "RelayFormer assemble_and_decode returned a non-tensor"
            )
        captured.append(logits.detach().clone())
        return logits

    model.assemble_and_decode = capture_assemble_output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            probability = model(
                image,
                origin_shape=origin_shape,
                clip_len=clip_len,
            )
    finally:
        model.assemble_and_decode = original_assemble
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0
    if len(captured) != 1:
        raise ValueError(
            "RelayFormer assemble_and_decode was called "
            f"{len(captured)} times, expected 1"
        )
    processed = postprocess_outputs(
        probability,
        captured[0],
        native_width=native_width,
        native_height=native_height,
        resized_width=resized_width,
        resized_height=resized_height,
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
        raise ValueError(
            f"GT mask shape mismatch: {target.shape} != {(height, width)}"
        )
    return target


def model_space_target(
    target: np.ndarray,
    *,
    resized_width: int,
    resized_height: int,
) -> np.ndarray:
    """Resize native GT to valid model content only, without canvas padding."""

    truth = np.asarray(target, dtype=np.uint8)
    if truth.ndim != 2:
        raise ValueError("RelayFormer GT target must be two-dimensional")
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError("RelayFormer GT resized dimensions must be positive")
    if truth.shape == (resized_height, resized_width):
        return np.ascontiguousarray(truth > 0)
    resized = np.asarray(
        Image.fromarray(truth, mode="L").resize(
            (resized_width, resized_height),
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


def build_run_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    dataset_manifest_path: Path,
    release: dict[str, Any],
    inputs_path: Path,
    selected: list[dict[str, Any]],
    relayformer_root: Path,
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
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_tracked_clean": not bool(
                _git_value(
                    relayformer_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "variant": "official_relay_vit_image_only_checkpoint_164",
            "source_files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in SOURCE_FILES.items()
            ],
            "license": {
                "path": "LICENSE",
                "sha256": SOURCE_FILES["LICENSE"],
                "spdx": "MIT",
                "scope": "project_repository_code_only",
                "checkpoint_license": (
                    "Apache-2.0_on_official_Hugging_Face_model_card"
                ),
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
                "pretraining_weights_reloaded": False,
                "hidden_timm_download": False,
                "constructor_load_replayed_from_safe_payload": True,
            },
            "constructor": {
                "class": "models.RelayFormer.RelayFormer",
                "input_size": MODEL_INPUT_SIZE,
                "grid_size": 2,
                "patch_size": 528,
                "overlap": 16,
                "feature_patch_size": 33,
                "feature_overlap": 1,
                "tokens_per_patch": 3,
                "vit_pretrain_path": "official_checkpoint_safe_payload",
                "merge_lora": True,
            },
            "parameter_count": CHECKPOINT["parameters"],
            "nonpersistent_buffer_elements_before_merge": CHECKPOINT["buffers"],
            "buffer_elements": CHECKPOINT["buffers"],
            "supports_image_level_t1": False,
            "image_score_source": None,
            "supports_pixel_level_t2": True,
            "primary_localization_output": (
                "official_sigmoid_probability_tensor"
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
            "input_geometry": PREPROCESS_PROTOCOL,
            "resize_condition": (
                "downscale_only_when_native_long_edge_exceeds_1024"
            ),
            "resize": "Pillow.Image.thumbnail",
            "resize_rounding": (
                "pillow_thumbnail_floor_ceil_min_aspect_error"
            ),
            "resize_interpolation": (
                "Pillow.Image.Resampling.BILINEAR_reducing_gap_None"
            ),
            "model_canvas_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "padding": {
                "placement": "top_left",
                "raw_rgb_value": 0,
                "applied_before_normalization": True,
            },
            "input_crop": None,
            "input_reencode": False,
            "normalization": {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "official_forward_arguments": {
                "image_shape": [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "clip": None,
                "origin_shape": "resized_valid_content_H_W_tensor",
                "clip_len": [1],
                "mask": None,
            },
            "official_model_output": {
                "raw_logits_shape": [
                    1,
                    1,
                    MODEL_INPUT_SIZE,
                    MODEL_INPUT_SIZE,
                ],
                "probability": (
                    "single_sigmoid_of_assemble_and_decode_logits"
                ),
                "captured_by": LOGIT_CAPTURE,
            },
            "native_compatibility_adapter": {
                "purpose": (
                    "CLAIMFORGE cross-method native-resolution comparison"
                ),
                "operation": NATIVE_RESTORE,
                "mode": "bilinear",
                "align_corners": False,
                "threshold_after_restore": True,
                "official_model_space_retained_as_auxiliary": True,
                "probability_restore_is_independent_of_logit_restore": True,
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
            "auxiliary_localization_space": "model_1024",
            "auxiliary_localization_extent": (
                "resized_valid_content_excluding_right_bottom_padding"
            ),
            "model_space_scope": (
                "valid_resized_content_only_excluding_right_bottom_padding"
            ),
            "mask_threshold": args.mask_threshold,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_gt_resize": (
                "Pillow.Image.Resampling.NEAREST_to_resized_valid_content"
            ),
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "raw_logits_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "semantics": "official_pre_sigmoid_logits",
                "captured_from": LOGIT_CAPTURE,
            },
            "raw_logits_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": "valid_content_logits_restored_to_native",
                "restore": NATIVE_RESTORE,
            },
            "score_maps_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "semantics": "official_sigmoid_probability",
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": (
                    "valid_content_probability_restored_to_native"
                ),
                "restore": NATIVE_RESTORE,
            },
            "masks_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "score_map_native > 0.5",
            },
        },
        "expected_pairs": len({int(row["pair_rank"]) for row in selected}),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": [
            {
                "path": repo_relative(Path(__file__), repo_root),
                "sha256": sha256_file(Path(__file__)),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("relayformer_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("relayformer_metrics.py")
                ),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("common.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("common.py")
                ),
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
        raise ValueError(
            f"existing result {sample_id} has modified {label} artifact"
        )
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
        raise ValueError(
            f"existing result {sample_id} has modified {label} artifact"
        )
    if row.get("mask_shape") != expected_shape:
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} shape metadata"
        )
    if row.get("mask_dtype") != "uint8":
        raise ValueError(
            f"existing result {sample_id} has incompatible {label} dtype metadata"
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


def _validate_resume_rows(
    latest: Mapping[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
    *,
    repo_root: Path,
    artifact_dir: Path,
) -> None:
    selected_by_id = {
        str(row["sample_id"]): row
        for row in selected
    }
    for sample_id, row in latest.items():
        if sample_id not in selected_by_id:
            continue
        if row.get("run_manifest_fingerprint") != manifest_fingerprint:
            raise ValueError(
                f"existing result {sample_id} has incompatible run fingerprint"
            )
        forbidden = {
            "score",
            "decision",
            "image_score",
            "image_decision",
            "classification",
            "detection",
        }.intersection(row)
        if forbidden:
            raise ValueError(
                f"existing result {sample_id} contains forbidden T1 fields: "
                f"{sorted(forbidden)}"
            )
        if (
            row.get("valid_for_t1") is not False
            or row.get("t1_policy")
            != "unsupported_no_derived_image_score"
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible T1 policy"
            )
        expected = selected_by_id[sample_id]
        if (
            row.get("image_sha256") != expected.get("canonical_sha256")
            or row.get("task_id") != expected.get("task_id")
            or row.get("kind") != expected.get("kind")
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible input identity"
            )
        if row.get("status") != "ok":
            continue
        width = int(expected["width"])
        height = int(expected["height"])
        full_shape = [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
        native_shape = [height, width]
        npy_contracts = (
            (
                "raw model logits",
                artifact_dir / "raw_logits_model_1024" / f"{sample_id}.npy",
                "raw_logits_model_path",
                "raw_logits_model_sha256",
                "raw_logits_model_shape",
                "raw_logits_model_dtype",
                full_shape,
            ),
            (
                "native logits",
                artifact_dir / "raw_logits_native" / f"{sample_id}.npy",
                "raw_logits_native_path",
                "raw_logits_native_sha256",
                "raw_logits_native_shape",
                "raw_logits_native_dtype",
                native_shape,
            ),
            (
                "model probability",
                artifact_dir / "score_maps_model_1024" / f"{sample_id}.npy",
                "score_map_model_path",
                "score_map_model_sha256",
                "score_map_model_shape",
                "score_map_model_dtype",
                full_shape,
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
        ) in npy_contracts:
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
        raise ValueError("official RelayFormer mask threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    relayformer_root = args.relayformer_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/relayformer/{args.run_id}"),
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
    selected = select_inputs(all_rows, args.pair_limit)
    for row in selected:
        image_path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            image_path,
            str(row["canonical_sha256"]),
            f"canonical input {row['sample_id']}",
        )
    _verify_static_contract(
        relayformer_root=relayformer_root,
        checkpoint_path=checkpoint_path,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        relayformer_root=relayformer_root,
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
        f"RelayFormer run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            relayformer_root=relayformer_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(
            "loaded official RelayFormer image-only checkpoint epoch 164 "
            f"{CHECKPOINT['sha256'][:12]} on {device}",
            flush=True,
        )
        try:
            for index, input_row in enumerate(pending, start=1):
                sample_id = str(input_row["sample_id"])
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
                    "edit_region_xyxy": [
                        int(value)
                        for value in input_row["edit_region_xyxy"]
                    ],
                    "model": MODEL_NAME,
                    "model_slug": MODEL_SLUG,
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
                    (
                        image_array,
                        native_size,
                        resized_size,
                        preprocess,
                    ) = preprocess_image(image_path)
                    width, height = native_size
                    resized_width, resized_height = resized_size
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
                        resized_width=resized_width,
                        resized_height=resized_height,
                    )
                    logits_model = processed["raw_logits_model_1024"]
                    logits_native = processed["raw_logits_native"]
                    probability_model = processed["probability_model_1024"]
                    probability_valid = processed["probability_valid"]
                    probability_native = processed["probability_native"]
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = model_space_target(
                        target_native,
                        resized_width=resized_width,
                        resized_height=resized_height,
                    )
                    include_ap = input_row["kind"] == "forged"
                    localization = {
                        "model_1024": binary_pixel_metrics_strict(
                            probability_valid,
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

                    logits_model_path = (
                        artifact_dir
                        / "raw_logits_model_1024"
                        / f"{sample_id}.npy"
                    )
                    logits_native_path = (
                        artifact_dir
                        / "raw_logits_native"
                        / f"{sample_id}.npy"
                    )
                    model_score_path = (
                        artifact_dir
                        / "score_maps_model_1024"
                        / f"{sample_id}.npy"
                    )
                    native_score_path = (
                        artifact_dir
                        / "score_maps_native"
                        / f"{sample_id}.npy"
                    )
                    native_mask_path = (
                        artifact_dir
                        / "masks_native"
                        / f"{sample_id}.png"
                    )
                    _atomic_save_npy(logits_model_path, logits_model)
                    _atomic_save_npy(logits_native_path, logits_native)
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
                            logits_model_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(
                            logits_model_path
                        ),
                        "raw_logits_model_shape": list(logits_model.shape),
                        "raw_logits_model_dtype": str(logits_model.dtype),
                        "raw_logits_capture": LOGIT_CAPTURE,
                        "raw_logits_native_path": repo_relative(
                            logits_native_path,
                            repo_root,
                        ),
                        "raw_logits_native_sha256": sha256_file(
                            logits_native_path
                        ),
                        "raw_logits_native_shape": list(logits_native.shape),
                        "raw_logits_native_dtype": str(logits_native.dtype),
                        "raw_logits_native_semantics": (
                            "valid_content_logits_restored_to_native"
                        ),
                        "raw_logits_native_restore": NATIVE_RESTORE,
                        "score_map_model_path": repo_relative(
                            model_score_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(
                            model_score_path
                        ),
                        "score_map_model_shape": list(
                            probability_model.shape
                        ),
                        "score_map_model_dtype": str(
                            probability_model.dtype
                        ),
                        "score_map_model_semantics": (
                            "official_sigmoid_probability"
                        ),
                        "score_map_path": repo_relative(
                            native_score_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(native_score_path),
                        "score_map_shape": list(probability_native.shape),
                        "score_map_dtype": str(probability_native.dtype),
                        "score_map_semantics": (
                            "valid_content_probability_restored_to_native"
                        ),
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
                        "model_valid_content_size_wh": [
                            resized_width,
                            resized_height,
                        ],
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
                        f"RelayFormer failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_relayformer_results(
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
    if (
        not args.allow_errors
        and (
            coverage["valid_images"] != coverage["expected_images"]
            or coverage["error_images"]
            or coverage["missing_images"]
        )
    ):
        raise RuntimeError(f"incomplete RelayFormer run: {coverage}")
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
        "--relayformer-root",
        type=Path,
        default=DEFAULT_RELAYFORMER_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/relayformer"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--pair-limit", type=int)
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
