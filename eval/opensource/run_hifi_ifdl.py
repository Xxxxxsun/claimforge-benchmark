#!/usr/bin/env python3
"""Run the official HiFi-IFDL general checkpoint 750001 on CLAIMFORGE.

The adapter exposes both native HiFi-Net tasks.  T1 uses the fine 14-class
head and stores the official argmax decision separately from CLAIMFORGE's
fixed score threshold.  T2 uses the raw hypersphere distance, never a
sigmoid-normalized surrogate.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
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
from eval.opensource.hifi_ifdl_metrics import (
    binary_distance_metrics_strict,
    summarize_hifi_ifdl_results,
)


MODEL_NAME = "HiFi-Net"
MODEL_SLUG = "hifi_ifdl_general_750001_official"
MODEL_REPO_URL = "https://github.com/CHELSEA234/HiFi_IFDL"
MODEL_SOURCE_COMMIT = "0ca70d651087bb09959dec583947031c47d30209"
MODEL_INPUT_SIZE = 256
EMBEDDING_CHANNELS = 18
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 2.3
PAIRWISE_P = 2.0
PAIRWISE_EPS = 1e-6

FINE_CLASS_NAMES = (
    "authentic",
    "splice",
    "inpainting",
    "copy_move",
    "faceshifter",
    "stgan",
    "star2",
    "hisd",
    "stylegan2",
    "stylegan3",
    "ddpm",
    "ddim",
    "d_latent",
    "glide",
)

SOURCE_FILES = {
    "README.md": "62fa6fa400f72797cb93c9104fe3781064f04c95fc3b72711e79690d536b6135",
    "LICENSE": "b01d7140e1f323024b0db35e0db18ba7cd3fd3380abbec57aec55e6141864e2f",
    "HiFi_Net.py": "353212467d2658284c91bd7ffb036599fea62bfe16f3c9855c082c42a0f0c088",
    "models/NLCDetection_api.py": "9af51775587b6d48e8d6c4e49844ab2b679856440fcadf51702683252c2ae441",
    "models/seg_hrnet.py": "f6cf64c1febbb7e9cb383332987928b6eb4acaa3f81068411d5de90bc86c1d9a",
    "models/seg_hrnet_config.py": "5ad6992a5ecc56a612245aa1bdfd5a4f8336be907c48cc6f86979f04235e4013",
    "models/LaPlacianMs.py": "32d37e3a98919dcbe5e80d433f254b58ed6818e092dad4ef07417caba7d185cc",
    "models/GaussianSmoothing.py": "bb6dedd7c955ddedea72d8e2f14258eb073e785520a6886516365f8b42312cef",
    "utils/custom_loss.py": "8973bc2930ff17e833f0d288f369e1acdec5d2720548f1a9a98627b3f322882b",
    "utils/utils.py": "0c46a3bcf008f6cb6157cd9511237a0efef2e463ab8463edd6c0e09021c63ff2",
    "utils/load_data.py": "be1a704ffd3b81fc5b832c7cae09cc82e08e908268f8df739b03851f18a263f9",
}

INITIALIZATION_WEIGHT = {
    "path": "models/hrnet_w18_small_v2.pth",
    "bytes": 16_012_341,
    "sha256": "06924c741ea8c076a569d5e164aa628910a72020800e4a4945e8b40b241ce5cb",
    "loaded": False,
    "reason": (
        "the released general HRNet model state strictly covers every "
        "parameter and buffer; loading initialization first is unnecessary"
    ),
}

CHECKPOINT_RELEASE = {
    "provider": "official_author_google_drive",
    "folder_id": "1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm",
    "released_identifier": "750001",
    "identifier_note": (
        "official released identifier only; this adapter does not label it "
        "as an epoch because the paper, release, and optimizer-step wording "
        "do not establish that equivalence"
    ),
}

CHECKPOINTS = {
    "feature_extractor": {
        "original_path": "HRNet/750001.pth",
        "file_id": "1BSg39gzlUM_odiiuV5NIJoc8f90JCCqf",
        "bytes": 81_112_652,
        "sha256": "be21278afb4e657bdafdf581d8d8bc6bc09f3b4507b10502ce98f1ae7ef1c5c1",
        "top_level_keys": ["model", "optimizer"],
        "model_container": "collections.OrderedDict",
        "state_keys": 699,
        "state_elements": 6_379_824,
        "tensor_bytes": 25_519_760,
        "state_dtypes": {"torch.float32": 583, "torch.int64": 116},
        "parameters": 6_361_208,
        "buffers": 18_616,
    },
    "hierarchical_localizer_classifier": {
        "original_path": "NLCDetection/750001.pth",
        "file_id": "1hELPa0bIyLrnr0a06vmr6GWNNinWB0nf",
        "bytes": 57_487_769,
        "sha256": "7615fcb054e7cbd0b25d647d72655a690424232668abbc911551648e84b5f8fc",
        "top_level_keys": ["model", "optimizer"],
        "model_container": "collections.OrderedDict",
        "state_keys": 66,
        "state_elements": 529_260,
        "tensor_bytes": 2_117_056,
        "state_dtypes": {"torch.float32": 62, "torch.int64": 4},
        "parameters": 529_112,
        "buffers": 148,
    },
}

CENTER_RADIUS = {
    "path": "center/radius_center.pth",
    "bytes": 543,
    "sha256": "e41e09256e65bcff9ba43e72f08701bf4d3904ccdb749f2d32a008af92c2483b",
    "top_level_keys": ["center", "radius"],
    "center_shape": [EMBEDDING_CHANNELS],
    "center_dtype": "torch.float32",
    "radius_shape": [],
    "radius_dtype": "torch.float32",
    "radius_value": 1.2404824495315552,
}

CHECKPOINT_BUNDLE_SHA256 = hashlib.sha256(
    stable_json(
        {
            **{
                role: contract["sha256"]
                for role, contract in CHECKPOINTS.items()
            },
            "center_radius": CENTER_RADIUS["sha256"],
        }
    ).encode("utf-8")
).hexdigest()

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_HIFI_ROOT = Path("/root/.cache/claimforge/third_party/HiFi_IFDL")
DEFAULT_HRNET_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/hifi-ifdl-general/"
    "HRNet/750001.pth"
)
DEFAULT_NLC_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/hifi-ifdl-general/"
    "NLCDetection/750001.pth"
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


@contextlib.contextmanager
def _numpy_int_compatibility() -> Iterator[None]:
    """Temporarily restore the alias used by the pinned HRNet constructor."""

    had_alias = "int" in np.__dict__
    previous = np.__dict__.get("int")
    if not had_alias:
        setattr(np, "int", int)
    try:
        yield
    finally:
        if had_alias:
            setattr(np, "int", previous)
        else:
            delattr(np, "int")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    """Reproduce HiFi-Net's ImageIO/Pillow 256-square preprocessing."""

    import imageio.v2 as imageio

    decoded = np.asarray(imageio.imread(path))
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(
            f"HiFi-IFDL expects a decoded RGB image, got {decoded.shape}"
        )
    if decoded.dtype != np.uint8:
        raise ValueError(
            f"HiFi-IFDL expects decoded uint8, got {decoded.dtype}"
        )
    native_height, native_width = decoded.shape[:2]
    resized = np.asarray(
        Image.fromarray(decoded).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            resample=Image.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    chw = np.ascontiguousarray(
        resized.astype(np.float32).transpose(2, 0, 1)
        / np.float32(255.0)
    )
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected HiFi-IFDL input shape: {chw.shape}")
    if not np.isfinite(chw).all():
        raise ValueError("HiFi-IFDL input tensor contains non-finite values")
    metadata = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [native_width, native_height],
        "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "tensor_shape": list(chw.shape),
        "tensor_dtype": str(chw.dtype),
        "tensor_sha256": _sha256_array(chw),
    }
    return chw, (native_width, native_height), metadata


def _tensor_to_float32_array(tensor: Any, label: str) -> np.ndarray:
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
    outputs: Any,
    center: Any,
    *,
    native_width: int,
    native_height: int,
) -> dict[str, Any]:
    """Validate one forward pass and derive the frozen T1/T2 outputs."""

    import torch
    from torch.nn import functional as F

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        length = len(outputs) if isinstance(outputs, (tuple, list)) else None
        raise ValueError(f"HiFi-IFDL returned {length} outputs, expected 6")
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native dimensions must be positive")
    embedding, auxiliary_mask, out0, out1, out2, out3 = outputs
    if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != (
        1,
        EMBEDDING_CHANNELS,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            "unexpected HiFi-IFDL localization embedding shape: "
            f"{getattr(embedding, 'shape', None)}"
        )
    if not isinstance(auxiliary_mask, torch.Tensor) or tuple(
        auxiliary_mask.shape
    ) != (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(
            "unexpected HiFi-IFDL auxiliary mask shape: "
            f"{getattr(auxiliary_mask, 'shape', None)}"
        )
    hierarchy = (
        ("out0_coarse_3class", out0, 3),
        ("out1_5class", out1, 5),
        ("out2_7class", out2, 7),
        ("out3_fine_14class", out3, 14),
    )
    hierarchy_logits: dict[str, list[float]] = {}
    for name, tensor, classes in hierarchy:
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (
            1,
            classes,
        ):
            raise ValueError(
                f"unexpected HiFi-IFDL {name} logits shape: "
                f"{getattr(tensor, 'shape', None)}"
            )
        array = _tensor_to_float32_array(tensor[0], f"{name} logits")
        hierarchy_logits[name] = [float(value) for value in array]

    if not isinstance(center, torch.Tensor) or tuple(center.shape) != (
        EMBEDDING_CHANNELS,
    ):
        raise ValueError(
            f"unexpected HiFi-IFDL center shape: {getattr(center, 'shape', None)}"
        )
    if center.dtype != torch.float32:
        raise ValueError(f"unexpected HiFi-IFDL center dtype: {center.dtype}")
    if not torch.isfinite(center).all():
        raise ValueError("HiFi-IFDL center contains non-finite values")

    flattened = embedding.permute(0, 2, 3, 1).reshape(
        -1,
        EMBEDDING_CHANNELS,
    )
    distance = torch.nn.PairwiseDistance(
        p=PAIRWISE_P,
        eps=PAIRWISE_EPS,
    )(flattened, center)
    distance_model_tensor = distance.reshape(
        1,
        1,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    )
    distance_native_tensor = F.interpolate(
        distance_model_tensor,
        size=(native_height, native_width),
        mode="bilinear",
        align_corners=False,
    )

    embedding_array = _tensor_to_float32_array(
        embedding[0],
        "localization embedding",
    )
    distance_model = _tensor_to_float32_array(
        distance_model_tensor[0, 0],
        "model-space distance map",
    )
    distance_native = _tensor_to_float32_array(
        distance_native_tensor[0, 0],
        "native distance map",
    )
    for label, array in (
        ("model-space distance map", distance_model),
        ("native distance map", distance_native),
    ):
        if float(array.min()) < 0.0:
            raise ValueError(f"{label} contains negative values")

    auxiliary = _tensor_to_float32_array(
        auxiliary_mask[0],
        "auxiliary learned mask",
    )
    if float(auxiliary.min()) < 0.0 or float(auxiliary.max()) > 1.0:
        raise ValueError("HiFi-IFDL auxiliary learned mask falls outside [0, 1]")

    probabilities_tensor = torch.softmax(out3, dim=1)
    probabilities = _tensor_to_float32_array(
        probabilities_tensor[0],
        "fine-class probabilities",
    )
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise ValueError("HiFi-IFDL fine probabilities fall outside [0, 1]")
    if not math.isclose(
        float(probabilities.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError("HiFi-IFDL fine probabilities do not sum to one")
    fine_index = int(torch.argmax(probabilities_tensor, dim=1).item())
    score = float(
        (torch.ones_like(probabilities_tensor[0, 0])
         - probabilities_tensor[0, 0]).item()
    )
    return {
        "embedding": embedding_array,
        "distance_model_256": distance_model,
        "distance_native": distance_native,
        "hierarchy_logits": hierarchy_logits,
        "fine_probabilities": probabilities,
        "score": score,
        "benchmark_binary_decision": score > CLASSIFICATION_THRESHOLD,
        "official_fine_class_index": fine_index,
        "official_fine_class_name": FINE_CLASS_NAMES[fine_index],
        "official_binary_decision": fine_index != 0,
        "auxiliary_learned_mask_stats": {
            "shape": list(auxiliary.shape),
            "dtype": str(auxiliary.dtype),
            "minimum": float(auxiliary.min()),
            "maximum": float(auxiliary.max()),
            "mean": float(auxiliary.mean()),
            "primary_output": False,
            "reason": (
                "the official public localize API ignores this sigmoid mask "
                "and thresholds hypersphere distance instead"
            ),
        },
    }


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


def _load_checkpoint_state(
    *,
    module: Any,
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> Any:
    """Strictly load the released DataParallel-prefixed model sub-dictionary."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} checkpoint is not a mapping")
    if set(payload) != set(contract["top_level_keys"]):
        raise ValueError(
            f"{label} checkpoint top-level schema mismatch: {list(payload)}"
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

    wrapped = torch.nn.DataParallel(module)
    wrapped.load_state_dict(state, strict=True)
    return wrapped.module


def _load_center_radius(path: Path) -> tuple[Any, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("HiFi-IFDL center/radius file is not a mapping")
    if set(payload) != set(CENTER_RADIUS["top_level_keys"]):
        raise ValueError("HiFi-IFDL center/radius top-level schema mismatch")
    center = payload.get("center")
    radius = payload.get("radius")
    if (
        not isinstance(center, torch.Tensor)
        or list(center.shape) != CENTER_RADIUS["center_shape"]
        or str(center.dtype) != CENTER_RADIUS["center_dtype"]
    ):
        raise ValueError("HiFi-IFDL center tensor schema mismatch")
    if (
        not isinstance(radius, torch.Tensor)
        or list(radius.shape) != CENTER_RADIUS["radius_shape"]
        or str(radius.dtype) != CENTER_RADIUS["radius_dtype"]
    ):
        raise ValueError("HiFi-IFDL radius tensor schema mismatch")
    if not torch.isfinite(center).all() or not torch.isfinite(radius).all():
        raise ValueError("HiFi-IFDL center/radius contains non-finite values")
    if float(radius.item()) != float(CENTER_RADIUS["radius_value"]):
        raise ValueError("HiFi-IFDL radius value mismatch")
    return center.detach().clone(), radius.detach().clone()


def _verify_static_contract(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
) -> None:
    source_commit = _git_value(hifi_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"HiFi-IFDL source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        hifi_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("HiFi-IFDL tracked source files have local modifications")
    for relative, expected in SOURCE_FILES.items():
        _verify_runtime_file(
            hifi_root / relative,
            expected,
            f"HiFi-IFDL source file {relative}",
        )
    _verify_file_contract(
        hifi_root / str(INITIALIZATION_WEIGHT["path"]),
        INITIALIZATION_WEIGHT,
        "HiFi-IFDL HRNet initialization weight",
    )
    _verify_file_contract(
        hrnet_checkpoint,
        CHECKPOINTS["feature_extractor"],
        "HiFi-IFDL general HRNet checkpoint 750001",
    )
    _verify_file_contract(
        nlc_checkpoint,
        CHECKPOINTS["hierarchical_localizer_classifier"],
        "HiFi-IFDL general NLCDetection checkpoint 750001",
    )
    _verify_file_contract(
        hifi_root / str(CENTER_RADIUS["path"]),
        CENTER_RADIUS,
        "HiFi-IFDL general center/radius",
    )


def load_model(
    *,
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
    device_name: str,
) -> tuple[tuple[Any, Any], Any, Any, Any]:
    import torch

    _verify_static_contract(
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
    )
    root_text = str(hifi_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    with _working_directory(hifi_root):
        from models.NLCDetection_api import NLCDetection
        from models.seg_hrnet import HighResolutionNet
        from models.seg_hrnet_config import get_cfg_defaults

        with _numpy_int_compatibility():
            feature_extractor = HighResolutionNet(get_cfg_defaults())
        # The pinned constructor calls .cuda() for two unused, unregistered
        # split tensors.  Prevent that construction-only side effect; all
        # registered model state is subsequently moved normally.
        with mock.patch.object(
            torch.Tensor,
            "cuda",
            new=lambda tensor, *args, **kwargs: tensor,
        ):
            hierarchical_head = NLCDetection()

    _validate_module_counts(
        feature_extractor,
        CHECKPOINTS["feature_extractor"],
        "HiFi-IFDL feature extractor",
    )
    _validate_module_counts(
        hierarchical_head,
        CHECKPOINTS["hierarchical_localizer_classifier"],
        "HiFi-IFDL hierarchical localizer/classifier",
    )
    feature_extractor = _load_checkpoint_state(
        module=feature_extractor,
        path=hrnet_checkpoint,
        contract=CHECKPOINTS["feature_extractor"],
        label="HiFi-IFDL feature extractor",
    )
    hierarchical_head = _load_checkpoint_state(
        module=hierarchical_head,
        path=nlc_checkpoint,
        contract=CHECKPOINTS["hierarchical_localizer_classifier"],
        label="HiFi-IFDL hierarchical localizer/classifier",
    )
    center, radius = _load_center_radius(
        hifi_root / str(CENTER_RADIUS["path"])
    )

    device = torch.device(device_name)
    feature_extractor.to(device).eval()
    hierarchical_head.to(device).eval()
    center = center.to(device)
    radius = radius.to(device)
    return (feature_extractor, hierarchical_head), center, radius, device


def infer_one(
    models: tuple[Any, Any],
    center: Any,
    device: Any,
    image_array: np.ndarray,
    *,
    native_width: int,
    native_height: int,
) -> tuple[dict[str, Any], int, float]:
    import torch

    feature_extractor, hierarchical_head = models
    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("HiFi-IFDL input tensor shape changed unexpectedly")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        features = feature_extractor(image)
        outputs = hierarchical_head(features, image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0
    processed = postprocess_outputs(
        outputs,
        center,
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
        raise ValueError(
            f"GT mask shape mismatch: {target.shape} != {(height, width)}"
        )
    return target


def model_space_target(target: np.ndarray) -> np.ndarray:
    image = Image.fromarray(
        np.where(np.asarray(target, dtype=bool), 255, 0).astype(np.uint8),
        mode="L",
    )
    resized = image.resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


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
    hifi_root: Path,
    hrnet_checkpoint: Path,
    nlc_checkpoint: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    import torch

    ordered_inputs = _selection_contract(selected)
    center_path = hifi_root / str(CENTER_RADIUS["path"])
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
        "model": {
            "name": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "repo_url": MODEL_REPO_URL,
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_tracked_clean": not bool(
                _git_value(
                    hifi_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "variant": "official_HiFi_IFDL_general_checkpoint_750001",
            "source_files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in SOURCE_FILES.items()
            ],
            "license": {
                "path": "LICENSE",
                "sha256": SOURCE_FILES["LICENSE"],
                "spdx": "MIT",
                "scope": "project_repository_code_only",
                "checkpoint_license": "not_separately_stated_by_release",
            },
            "initialization_weight": INITIALIZATION_WEIGHT,
            "checkpoint": {
                **CHECKPOINT_RELEASE,
                "bundle_sha256": CHECKPOINT_BUNDLE_SHA256,
                "components": [
                    {
                        "role": "feature_extractor",
                        **CHECKPOINTS["feature_extractor"],
                        "path": str(hrnet_checkpoint),
                    },
                    {
                        "role": "hierarchical_localizer_classifier",
                        **CHECKPOINTS[
                            "hierarchical_localizer_classifier"
                        ],
                        "path": str(nlc_checkpoint),
                    },
                ],
                "center_radius": {
                    **CENTER_RADIUS,
                    "runtime_path": str(center_path),
                    "loaded_center": True,
                    "loaded_radius_for_provenance_validation": True,
                },
                "strict_load": True,
                "safe_weights_only_load": True,
                "container_selection": "top_level_model_only",
                "prefix_rewrites": False,
                "schema_fallbacks": False,
            },
            "parameter_count": sum(
                int(value["parameters"]) for value in CHECKPOINTS.values()
            ),
            "buffer_elements": sum(
                int(value["buffers"]) for value in CHECKPOINTS.values()
            ),
            "fine_class_names": list(FINE_CLASS_NAMES),
            "fine_authentic_class_index": 0,
            "hierarchy_output_class_counts": [3, 5, 7, 14],
            "supports_image_level_t1": True,
            "image_score_source": (
                "native_out3_fine_14class_head"
            ),
            "supports_pixel_level_t2": True,
            "primary_localization_output": (
                "euclidean_distance_from_18d_pixel_embedding_to_"
                "released_authentic_center"
            ),
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "compatibility_shims": [
                {
                    "shim": "temporary_numpy.int=builtin_int",
                    "scope": "HRNet_constructor_only",
                    "numerical_effect": "none",
                },
                {
                    "shim": "temporary_torch.Tensor.cuda_identity",
                    "scope": (
                        "NLCDetection_constructor_two_unused_unregistered_"
                        "split_tensors_only"
                    ),
                    "numerical_effect": "none_in_forward",
                },
            ],
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "imageio.v2.imread",
            "channel_order": "RGB",
            "input_geometry": (
                "direct_stretch_to_256x256_without_aspect_ratio_preservation"
            ),
            "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
            "input_crop": None,
            "input_reencode": False,
            "normalization": "uint8_rgb_divide_255_float32",
            "feature_output_shapes": [
                [18, 256, 256],
                [36, 128, 128],
                [72, 64, 64],
                [144, 32, 32],
            ],
            "classification": {
                "continuous_score": (
                    "1_minus_softmax_fine_14class_probability_index_0"
                ),
                "score_source": (
                    "native_out3_fine_14class_head"
                ),
                "benchmark_threshold": args.classification_threshold,
                "benchmark_threshold_comparison": "strict_greater_than",
                "official_decision": (
                    "argmax_fine_14class_index_not_equal_to_0"
                ),
                "both_decisions_stored_separately": True,
            },
            "localization": {
                "embedding_shape": [18, 256, 256],
                "distance": "torch.nn.PairwiseDistance",
                "p": PAIRWISE_P,
                "eps": PAIRWISE_EPS,
                "center_source": "center/radius_center.pth:center",
                "score_semantics": (
                    "raw_unbounded_nonnegative_euclidean_distance"
                ),
                "sigmoid_or_probability_normalization": False,
                "public_api_threshold": args.mask_threshold,
                "threshold_comparison": "greater_than_or_equal",
                "internal_loss_threshold_1_85_times_radius": (
                    1.85 * float(CENTER_RADIUS["radius_value"])
                ),
                "internal_loss_threshold_used": False,
                "auxiliary_learned_sigmoid_mask_used": False,
            },
            "native_compatibility_adapter": {
                "purpose": (
                    "CLAIMFORGE cross-method native-resolution comparison"
                ),
                "operation": (
                    "bilinear_restore_raw_256_distance_to_native_before_"
                    "thresholding"
                ),
                "mode": "bilinear",
                "align_corners": False,
                "threshold_after_restore": True,
                "official_model_space_retained_as_auxiliary": True,
            },
            "test_time_augmentation": False,
            "ensemble": False,
            "forward_passes_per_image": 1,
        },
        "metrics": {
            "task": "T1_image_detection_and_T2_pixel_localization",
            "positive_class": "forged_or_manipulated",
            "classification_threshold": args.classification_threshold,
            "classification_threshold_comparison": "strict_greater_than",
            "official_argmax_reported_separately": True,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_256",
            "localization_score": "raw_hypersphere_euclidean_distance",
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "greater_than_or_equal",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_gt_resize": "Pillow_nearest_neighbor_to_256x256",
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "mask_features_model_256": {
                "format": "npy",
                "dtype": "float32",
                "shape": [EMBEDDING_CHANNELS, 256, 256],
            },
            "distance_maps_model_256": {
                "format": "npy",
                "dtype": "float32",
                "shape": [256, 256],
                "range": "nonnegative_unbounded",
            },
            "distance_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "range": "nonnegative_unbounded",
            },
            "masks_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "distance_map_native >= 2.3",
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
                    Path(__file__).with_name("hifi_ifdl_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("hifi_ifdl_metrics.py")
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
    requested_device = torch.device(args.device)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "imageio": _package_version("imageio"),
        "yacs": _package_version("yacs"),
        "device": args.device,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(requested_device)
            if requested_device.type == "cuda"
            and torch.cuda.is_available()
            else None
        ),
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
        "environment": environment,
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


def _validate_resume_rows(
    latest: Mapping[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
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
        expected = selected_by_id[sample_id]
        if (
            row.get("image_sha256") != expected.get("canonical_sha256")
            or row.get("task_id") != expected.get("task_id")
            or row.get("kind") != expected.get("kind")
        ):
            raise ValueError(
                f"existing result {sample_id} has incompatible input identity"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.classification_threshold) != CLASSIFICATION_THRESHOLD:
        raise ValueError(
            "HiFi-IFDL benchmark classification threshold must be 0.5"
        )
    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("HiFi-IFDL public localization threshold must be 2.3")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    hifi_root = args.hifi_root.resolve()
    hrnet_checkpoint = args.hrnet_checkpoint.resolve()
    nlc_checkpoint = args.nlc_checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/hifi_ifdl/{args.run_id}"),
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
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
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
        hifi_root=hifi_root,
        hrnet_checkpoint=hrnet_checkpoint,
        nlc_checkpoint=nlc_checkpoint,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(run_manifest_path, run_manifest)
    existing = read_latest_by_id(output_path)
    _validate_resume_rows(existing, selected, run_manifest["fingerprint"])
    pending = [
        row
        for row in selected
        if existing.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"HiFi-IFDL run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    models = None
    center = None
    radius = None
    if pending:
        models, center, radius, device = load_model(
            hifi_root=hifi_root,
            hrnet_checkpoint=hrnet_checkpoint,
            nlc_checkpoint=nlc_checkpoint,
            device_name=args.device,
        )
        print(
            "loaded official HiFi-IFDL general checkpoint 750001 "
            f"{CHECKPOINT_BUNDLE_SHA256[:12]} on {device}",
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
                    "checkpoint_sha256": CHECKPOINT_BUNDLE_SHA256,
                    "checkpoint_released_identifier": "750001",
                    "valid_for_t1": True,
                    "valid_for_t2": True,
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    image_array, native_size, preprocess = preprocess_image(
                        image_path
                    )
                    width, height = native_size
                    if (width, height) != (
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ):
                        raise ValueError("canonical image dimensions changed")
                    processed, peak_bytes, latency_ms = infer_one(
                        models,
                        center,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                    )
                    distance_model = processed["distance_model_256"]
                    distance_native = processed["distance_native"]
                    embedding = processed["embedding"]
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = model_space_target(target_native)
                    include_ap = input_row["kind"] == "forged"
                    localization = {
                        "model_256": binary_distance_metrics_strict(
                            distance_model,
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_distance_metrics_strict(
                            distance_native,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    embedding_path = (
                        artifact_dir
                        / "mask_features_model_256"
                        / f"{sample_id}.npy"
                    )
                    model_distance_path = (
                        artifact_dir
                        / "distance_maps_model_256"
                        / f"{sample_id}.npy"
                    )
                    native_distance_path = (
                        artifact_dir
                        / "distance_maps_native"
                        / f"{sample_id}.npy"
                    )
                    native_mask_path = (
                        artifact_dir
                        / "masks_native"
                        / f"{sample_id}.png"
                    )
                    _atomic_save_npy(embedding_path, embedding)
                    _atomic_save_npy(model_distance_path, distance_model)
                    _atomic_save_npy(native_distance_path, distance_native)
                    _atomic_save_mask(
                        native_mask_path,
                        distance_native >= args.mask_threshold,
                    )

                    score = float(processed["score"])
                    benchmark_binary = bool(
                        processed["benchmark_binary_decision"]
                    )
                    official_binary = bool(
                        processed["official_binary_decision"]
                    )
                    probabilities = processed["fine_probabilities"]
                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": score,
                        "score_source": (
                            "native_out3_fine_14class_head"
                        ),
                        "score_semantics": (
                            "one_minus_softmax_probability_fine_class_0_"
                            "authentic"
                        ),
                        "classification_threshold": (
                            args.classification_threshold
                        ),
                        "classification_threshold_operator": ">",
                        "decision": (
                            "forged" if benchmark_binary else "authentic"
                        ),
                        "benchmark_binary_decision": benchmark_binary,
                        "official_fine_class_index": int(
                            processed["official_fine_class_index"]
                        ),
                        "official_fine_class_name": str(
                            processed["official_fine_class_name"]
                        ),
                        "official_binary_decision": official_binary,
                        "official_decision": (
                            "forged" if official_binary else "authentic"
                        ),
                        "official_decision_rule": (
                            "argmax_fine_14class_index_not_equal_to_0"
                        ),
                        "classification_probabilities": [
                            float(value) for value in probabilities
                        ],
                        "classification_hierarchy_logits": processed[
                            "hierarchy_logits"
                        ],
                        "auxiliary_learned_mask": processed[
                            "auxiliary_learned_mask_stats"
                        ],
                        "mask_feature_model_path": repo_relative(
                            embedding_path,
                            repo_root,
                        ),
                        "mask_feature_model_sha256": sha256_file(
                            embedding_path
                        ),
                        "mask_feature_model_shape": list(embedding.shape),
                        "mask_feature_model_dtype": str(embedding.dtype),
                        "distance_map_model_path": repo_relative(
                            model_distance_path,
                            repo_root,
                        ),
                        "distance_map_model_sha256": sha256_file(
                            model_distance_path
                        ),
                        "distance_map_model_shape": list(
                            distance_model.shape
                        ),
                        "distance_map_model_dtype": str(
                            distance_model.dtype
                        ),
                        "score_map_path": repo_relative(
                            native_distance_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(
                            native_distance_path
                        ),
                        "score_map_shape": list(distance_native.shape),
                        "score_map_dtype": str(distance_native.dtype),
                        "score_map_semantics": (
                            "raw_hifi_hypersphere_euclidean_distance"
                        ),
                        "score_map_native_restore": (
                            "bilinear_align_corners_false_from_256_raw_"
                            "distance_compatibility_adapter"
                        ),
                        "mask_path": repo_relative(
                            native_mask_path,
                            repo_root,
                        ),
                        "mask_sha256": sha256_file(native_mask_path),
                        "mask_shape": list(distance_native.shape),
                        "mask_threshold": args.mask_threshold,
                        "mask_threshold_operator": ">=",
                        "pairwise_distance": {
                            "p": PAIRWISE_P,
                            "eps": PAIRWISE_EPS,
                        },
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
                        f" score={row['score']:.6f} "
                        f"benchmark={row['decision']} "
                        f"official={row['official_decision']} "
                        f"f1={native_metrics.get('f1')} "
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
                        f"HiFi-IFDL failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del models
            del center
            del radius
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_hifi_ifdl_results(
        result_rows,
        selected,
        classification_threshold=args.classification_threshold,
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
            "checkpoint_sha256": CHECKPOINT_BUNDLE_SHA256,
            "checkpoint_released_identifier": "750001",
            "input_manifest_sha256": release["inputs_sha256"],
            "run_manifest_fingerprint": run_manifest["fingerprint"],
            "valid_for_t1": True,
            "valid_for_t2": True,
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
        raise RuntimeError(f"incomplete HiFi-IFDL run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--hifi-root", type=Path, default=DEFAULT_HIFI_ROOT)
    parser.add_argument(
        "--hrnet-checkpoint",
        type=Path,
        default=DEFAULT_HRNET_CHECKPOINT,
    )
    parser.add_argument(
        "--nlc-checkpoint",
        type=Path,
        default=DEFAULT_NLC_CHECKPOINT,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/hifi_ifdl"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=MASK_THRESHOLD,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
