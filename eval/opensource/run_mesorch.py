#!/usr/bin/env python3
"""Run the official Mesorch checkpoint 98 on CLAIMFORGE inputs.

Mesorch is a native pixel-localization model and does not expose an
image-level classification head.  This adapter therefore reports T2 only and
never turns a localization map into a synthetic T1 score.
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
from eval.opensource.mesorch_metrics import (
    binary_pixel_metrics_strict,
    summarize_mesorch_results,
)


MODEL_NAME = "Mesorch"
MODEL_SLUG = "mesorch_full_epoch98_official"
MODEL_REPO_URL = "https://github.com/scu-zjz/Mesorch"
MODEL_SOURCE_COMMIT = "ea82b0274b92244115d09b81663c88f57c7b78ee"
MODEL_INPUT_SIZE = 512
INTERNAL_LOGIT_SIZE = 128
MASK_THRESHOLD = 0.5

SOURCE_FILES = {
    "README.md": "632ea74d61596bf6a276246ea8ffdf7e0065a4fa401b32bd2a0561dcd2021d18",
    "LICENSE": "e63a366fa3a228da2a22b4b5f9c4e1aa98ed37f2ee93487170413ea6675387a2",
    "mesorch.py": "b301b28faf0cc44ced9845a99400cafe00b33d3e07109d18ebbf0087ef0b915a",
    "extractor/__init__.py": (
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    ),
    "extractor/high_frequency_feature_extraction.py": (
        "ce9a9b7f0515a650631a71303cdf113d915d45a1a1b0614c311f1703eca25e5b"
    ),
    "extractor/low_frequency_feature_extraction.py": (
        "b2aaf318af4ce6bbb1caef24256bd3ae86949e449557bbd6c9990e989df34a04"
    ),
    "test.py": "5b0551bd18641b4179e4b3ee96097c99f4df1433048d4dbe744c0c80ec59f244",
    "test_mesorch_f1.sh": (
        "78fa9e52a2017766463604d7ee776ce6844b0fa3dbadcdce5e3b15694ed8a620"
    ),
}

CHECKPOINT = {
    "provider": "official_author_google_drive",
    "file_id": "1PJxKteinMyaAYokKy0JhuzBnBc6bGsau",
    "original_filename": "mesorch-98.pth",
    "last_modified_utc": "2024-12-19T08:31:40+00:00",
    "bytes": 1_023_886_070,
    "sha256": "6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a",
    "container": "mapping_with_model_optimizer_epoch_scaler_args",
    "top_level_keys": ["model", "optimizer", "epoch", "scaler", "args"],
    "epoch": 98,
    "args_type": "argparse.Namespace",
    "state_container": "collections.OrderedDict",
    "state_keys": 804,
    "state_elements": 85_753_944,
    "tensor_bytes": 343_015_776,
    "state_dtypes": {"torch.float32": 804},
    "parameters": 85_753_944,
    "buffers": 0,
}

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_MESORCH_ROOT = Path("/root/.cache/claimforge/third_party/Mesorch")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/mesorch/mesorch-98.pth"
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
    """Bind an imported numerical module to installed distribution files."""

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
    """Capture dependency and accelerator state inside the run fingerprint."""

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
        "numpy": _installed_module_contract("numpy", ("numpy",)),
        "Pillow": _installed_module_contract("PIL", ("Pillow",)),
        "albumentations": _installed_module_contract(
            "albumentations",
            ("albumentations",),
        ),
        "scikit-learn": _installed_module_contract(
            "sklearn",
            ("scikit-learn",),
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
    }
    critical_submodules = {
        "IMDLBenCo.registry": _installed_module_contract(
            "IMDLBenCo.registry",
            ("IMDLBenCo",),
        ),
    }
    requested_device = torch.device(device_name)
    cuda_active = (
        requested_device.type == "cuda"
        and torch.cuda.is_available()
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
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    """Reproduce IMDLBenCo's official resizing test transform."""

    import albumentations as albu
    import torch
    from albumentations.pytorch import ToTensorV2

    with path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            decoder_format = opened.format
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"unexpected Mesorch decoded image shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected Mesorch decoded image dtype: {decoded.dtype}")
    native_height, native_width = decoded.shape[:2]
    transform = albu.Compose(
        [
            albu.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            albu.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            albu.Crop(
                0,
                0,
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )
    transformed = transform(
        image=decoded,
        mask=np.zeros((native_height, native_width), dtype=np.uint8),
    )
    image_tensor = transformed["image"]
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError("Mesorch transform did not produce a tensor")
    chw = np.ascontiguousarray(
        image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected Mesorch input shape: {chw.shape}")
    if not np.isfinite(chw).all():
        raise ValueError("Mesorch normalized input contains non-finite values")
    metadata = {
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [native_width, native_height],
        "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "transform": (
            "albumentations.Compose_Resize_Normalize_Crop_ToTensorV2"
        ),
        "resize": "albumentations.Resize",
        "resize_interpolation": "cv2.INTER_LINEAR_default",
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
    outputs: Any,
    internal_logits: Any,
    *,
    native_width: int,
    native_height: int,
) -> dict[str, np.ndarray]:
    """Validate the official sigmoid output and restore it to native size."""

    import torch
    from torch.nn import functional as F

    if not isinstance(outputs, Mapping):
        raise ValueError("Mesorch forward output is not a mapping")
    if "pred_mask" not in outputs:
        raise ValueError("Mesorch forward output has no pred_mask")
    probability = outputs["pred_mask"]
    if (
        not isinstance(probability, torch.Tensor)
        or tuple(probability.shape)
        != (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ):
        raise ValueError(
            "unexpected Mesorch pred_mask shape: "
            f"{getattr(probability, 'shape', None)}"
        )
    if (
        not isinstance(internal_logits, torch.Tensor)
        or tuple(internal_logits.shape)
        != (1, 1, INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE)
    ):
        raise ValueError(
            "unexpected Mesorch internal-logit shape: "
            f"{getattr(internal_logits, 'shape', None)}"
        )
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native dimensions must be positive")

    expected_probability = torch.sigmoid(
        F.interpolate(
            internal_logits,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=True,
        ).float()
    )
    if not torch.allclose(
        probability.float(),
        expected_probability,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Mesorch pred_mask does not match sigmoid of the captured "
            "align_corners=True internal logits"
        )
    native_probability = F.interpolate(
        probability.float(),
        size=(native_height, native_width),
        mode="bilinear",
        align_corners=False,
    )

    logits_array = _float32_map(
        internal_logits[0, 0],
        "Mesorch internal logits",
    )
    model_probability = _float32_map(
        probability[0, 0],
        "Mesorch model probability",
    )
    native_array = _float32_map(
        native_probability[0, 0],
        "Mesorch native probability",
    )
    for label, array in (
        ("model probability", model_probability),
        ("native probability", native_array),
    ):
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"Mesorch {label} falls outside [0, 1]")
    return {
        "internal_logits_model_128": logits_array,
        "probability_model_512": model_probability,
        "probability_native": native_array,
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
    """Safely validate and strictly load the released model sub-dictionary."""

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
    module.load_state_dict(state, strict=True)
    return module


def _verify_static_contract(
    *,
    mesorch_root: Path,
    checkpoint_path: Path,
) -> None:
    source_commit = _git_value(mesorch_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"Mesorch source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        mesorch_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("Mesorch tracked source files have local modifications")
    for relative, expected in SOURCE_FILES.items():
        _verify_runtime_file(
            mesorch_root / relative,
            expected,
            f"Mesorch source file {relative}",
        )
    _verify_file_contract(
        checkpoint_path,
        CHECKPOINT,
        "Mesorch official checkpoint 98",
    )


def _require_cached_module_origin(
    module_name: str,
    expected_path: Path,
) -> None:
    """Reject an already-imported module that bypasses the pinned source."""

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
    mesorch_root: Path,
    checkpoint_path: Path,
    device_name: str,
) -> tuple[Any, Any]:
    import torch

    # Import and validate numerical dependencies before adding the upstream
    # source root to sys.path.  This prevents untracked timm.py, IMDLBenCo.py,
    # or similarly named files in that checkout from shadowing distributions.
    _runtime_contract(device_name)
    _verify_static_contract(
        mesorch_root=mesorch_root,
        checkpoint_path=checkpoint_path,
    )
    root_text = str(mesorch_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    pinned_modules = {
        "mesorch": mesorch_root / "mesorch.py",
        "extractor": mesorch_root / "extractor" / "__init__.py",
        "extractor.high_frequency_feature_extraction": (
            mesorch_root
            / "extractor"
            / "high_frequency_feature_extraction.py"
        ),
        "extractor.low_frequency_feature_extraction": (
            mesorch_root
            / "extractor"
            / "low_frequency_feature_extraction.py"
        ),
    }
    for module_name, expected_path in pinned_modules.items():
        _require_cached_module_origin(module_name, expected_path)
    with _working_directory(mesorch_root):
        from mesorch import MesorchFull

        model = MesorchFull(
            seg_pretrain_path=None,
            conv_pretrain=False,
            image_size=MODEL_INPUT_SIZE,
        )
    for module_name, expected_path in pinned_modules.items():
        _require_cached_module_origin(module_name, expected_path)
    class_source = inspect.getsourcefile(MesorchFull)
    if class_source is None or Path(class_source).resolve() != (
        mesorch_root / "mesorch.py"
    ).resolve():
        raise ValueError("MesorchFull class source does not match pinned mesorch.py")
    extractor_sources = (
        (
            type(model.high_dct),
            mesorch_root
            / "extractor"
            / "high_frequency_feature_extraction.py",
        ),
        (
            type(model.low_dct),
            mesorch_root
            / "extractor"
            / "low_frequency_feature_extraction.py",
        ),
    )
    for extractor_class, expected_path in extractor_sources:
        class_source = inspect.getsourcefile(extractor_class)
        if class_source is None or Path(class_source).resolve() != (
            expected_path.resolve()
        ):
            raise ValueError(
                f"{extractor_class.__name__} class source does not match "
                f"{expected_path}"
            )
    _validate_module_counts(model, CHECKPOINT, "MesorchFull")
    model = _load_checkpoint_state(
        module=model,
        path=checkpoint_path,
        contract=CHECKPOINT,
        label="Mesorch",
    )
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
    """Run one official forward and audit its internal pre-resize logits."""

    import torch

    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("Mesorch input tensor shape changed unexpectedly")
    dummy_mask = torch.zeros(
        (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        dtype=image.dtype,
        device=device,
    )
    captured: list[Any] = []

    def capture_resize_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1:
            raise ValueError("Mesorch resize received an unexpected input count")
        captured.append(inputs[0].detach().clone())

    handle = model.resize.register_forward_pre_hook(capture_resize_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            outputs = model(image, dummy_mask)
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
            f"Mesorch internal resize was called {len(captured)} times, expected 1"
        )
    processed = postprocess_outputs(
        outputs,
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
        raise ValueError(
            f"GT mask shape mismatch: {target.shape} != {(height, width)}"
        )
    return target


def model_space_target(target: np.ndarray) -> np.ndarray:
    import cv2

    truth = np.asarray(target, dtype=np.uint8)
    resized = cv2.resize(
        truth,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
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
    mesorch_root: Path,
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
                    mesorch_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "variant": "official_MesorchFull_checkpoint_epoch_98",
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
            },
            "parameter_count": CHECKPOINT["parameters"],
            "buffer_elements": CHECKPOINT["buffers"],
            "supports_image_level_t1": False,
            "image_score_source": None,
            "supports_pixel_level_t2": True,
            "primary_localization_output": (
                "official_pred_mask_sigmoid_probability"
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
            "transform": (
                "albumentations.Compose([Resize(512,512),"
                "Normalize(ImageNet),Crop(0,0,512,512),"
                "ToTensorV2(transpose_mask=True)])"
            ),
            "resize": "albumentations.Resize",
            "resize_interpolation": "cv2.INTER_LINEAR_default",
            "input_crop": None,
            "input_reencode": False,
            "normalization": {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "dummy_mask": {
                "shape": [1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "dtype": "same_as_image_float32",
                "value": 0,
                "purpose": (
                    "required only because the official forward computes "
                    "BCEWithLogitsLoss; it does not affect pred_mask"
                ),
            },
            "official_model_output": {
                "internal_logits_shape": [
                    1,
                    1,
                    INTERNAL_LOGIT_SIZE,
                    INTERNAL_LOGIT_SIZE,
                ],
                "internal_resize": (
                    "torch_nn_Upsample_bilinear_to_512_align_corners_true"
                ),
                "probability": "single_sigmoid_after_internal_resize",
                "captured_by": (
                    "one_forward_pre_hook_on_model.resize_for_audit"
                ),
            },
            "native_compatibility_adapter": {
                "purpose": (
                    "CLAIMFORGE cross-method native-resolution comparison"
                ),
                "operation": (
                    "bilinear_restore_official_512_probability_to_native"
                ),
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
            "model_space_gt_resize": "cv2_INTER_NEAREST_to_512x512",
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "raw_logits_model_128": {
                "format": "npy",
                "dtype": "float32",
                "shape": [INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE],
                "captured_before_official_resize": True,
            },
            "score_maps_model_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "semantics": "official_sigmoid_probability",
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": "bilinearly_restored_probability",
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
                    Path(__file__).with_name("mesorch_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("mesorch_metrics.py")
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
        artifact_contracts = (
            {
                "label": "raw internal logits",
                "path_field": "raw_logits_model_path",
                "sha_field": "raw_logits_model_sha256",
                "shape_field": "raw_logits_model_shape",
                "dtype_field": "raw_logits_model_dtype",
                "path": (
                    artifact_dir
                    / "raw_logits_model_128"
                    / f"{sample_id}.npy"
                ),
                "shape": [INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE],
                "dtype": "float32",
                "kind": "npy",
            },
            {
                "label": "model probability",
                "path_field": "score_map_model_path",
                "sha_field": "score_map_model_sha256",
                "shape_field": "score_map_model_shape",
                "dtype_field": "score_map_model_dtype",
                "path": (
                    artifact_dir
                    / "score_maps_model_512"
                    / f"{sample_id}.npy"
                ),
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "dtype": "float32",
                "kind": "npy",
            },
            {
                "label": "native probability",
                "path_field": "score_map_path",
                "sha_field": "score_map_sha256",
                "shape_field": "score_map_shape",
                "dtype_field": "score_map_dtype",
                "path": (
                    artifact_dir
                    / "score_maps_native"
                    / f"{sample_id}.npy"
                ),
                "shape": [height, width],
                "dtype": "float32",
                "kind": "npy",
            },
            {
                "label": "native mask",
                "path_field": "mask_path",
                "sha_field": "mask_sha256",
                "shape_field": "mask_shape",
                "dtype_field": "mask_dtype",
                "path": (
                    artifact_dir
                    / "masks_native"
                    / f"{sample_id}.png"
                ),
                "shape": [height, width],
                "dtype": "uint8",
                "kind": "mask",
            },
        )
        for contract in artifact_contracts:
            expected_path = Path(contract["path"])
            expected_record_path = repo_relative(expected_path, repo_root)
            record_path = row.get(str(contract["path_field"]))
            if record_path != expected_record_path:
                raise ValueError(
                    f"existing result {sample_id} has incompatible "
                    f"{contract['label']} artifact path"
                )
            if not expected_path.is_file():
                raise ValueError(
                    f"existing result {sample_id} is missing "
                    f"{contract['label']} artifact: {expected_path}"
                )
            recorded_sha = row.get(str(contract["sha_field"]))
            if (
                not isinstance(recorded_sha, str)
                or len(recorded_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in recorded_sha
                )
            ):
                raise ValueError(
                    f"existing result {sample_id} has invalid "
                    f"{contract['label']} SHA-256 metadata"
                )
            actual_sha = sha256_file(expected_path)
            if actual_sha != recorded_sha:
                raise ValueError(
                    f"existing result {sample_id} has modified "
                    f"{contract['label']} artifact"
                )
            if row.get(str(contract["shape_field"])) != contract["shape"]:
                raise ValueError(
                    f"existing result {sample_id} has incompatible "
                    f"{contract['label']} shape metadata"
                )
            if row.get(str(contract["dtype_field"])) != contract["dtype"]:
                raise ValueError(
                    f"existing result {sample_id} has incompatible "
                    f"{contract['label']} dtype metadata"
                )
            if contract["kind"] == "npy":
                try:
                    array = np.load(
                        expected_path,
                        allow_pickle=False,
                        mmap_mode="r",
                    )
                    actual_shape = list(array.shape)
                    actual_dtype = str(array.dtype)
                except Exception as exc:
                    raise ValueError(
                        f"existing result {sample_id} has unreadable "
                        f"{contract['label']} artifact"
                    ) from exc
            else:
                try:
                    with Image.open(expected_path) as opened:
                        if opened.format != "PNG":
                            raise ValueError("mask artifact is not PNG")
                        array = np.asarray(opened)
                    actual_shape = list(array.shape)
                    actual_dtype = str(array.dtype)
                    if not np.isin(array, (0, 255)).all():
                        raise ValueError("mask artifact is not binary")
                except Exception as exc:
                    raise ValueError(
                        f"existing result {sample_id} has unreadable "
                        f"{contract['label']} artifact"
                    ) from exc
            if (
                actual_shape != contract["shape"]
                or actual_dtype != contract["dtype"]
            ):
                raise ValueError(
                    f"existing result {sample_id} has incompatible "
                    f"{contract['label']} artifact schema"
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("official Mesorch mask threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    mesorch_root = args.mesorch_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/mesorch/{args.run_id}"),
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
        mesorch_root=mesorch_root,
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
        mesorch_root=mesorch_root,
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
        f"Mesorch run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            mesorch_root=mesorch_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(
            "loaded official MesorchFull checkpoint epoch 98 "
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
                        model,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                    )
                    internal_logits = processed[
                        "internal_logits_model_128"
                    ]
                    probability_model = processed[
                        "probability_model_512"
                    ]
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

                    logits_path = (
                        artifact_dir
                        / "raw_logits_model_128"
                        / f"{sample_id}.npy"
                    )
                    model_score_path = (
                        artifact_dir
                        / "score_maps_model_512"
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
                    _atomic_save_npy(logits_path, internal_logits)
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
                            logits_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(logits_path),
                        "raw_logits_model_shape": list(internal_logits.shape),
                        "raw_logits_model_dtype": str(internal_logits.dtype),
                        "raw_logits_capture": (
                            "pre_hook_input_to_official_model.resize"
                        ),
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
                            "official_pred_mask_sigmoid_probability"
                        ),
                        "score_map_path": repo_relative(
                            native_score_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(native_score_path),
                        "score_map_shape": list(probability_native.shape),
                        "score_map_dtype": str(probability_native.dtype),
                        "score_map_semantics": (
                            "official_probability_restored_to_native"
                        ),
                        "score_map_native_restore": (
                            "bilinear_align_corners_false_from_512_"
                            "probability"
                        ),
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
                        f"Mesorch failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_mesorch_results(
        result_rows,
        selected,
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
        raise RuntimeError(f"incomplete Mesorch run: {coverage}")
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
        "--mesorch-root",
        type=Path,
        default=DEFAULT_MESORCH_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/mesorch"),
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
