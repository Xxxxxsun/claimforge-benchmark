#!/usr/bin/env python3
"""Run the official IML-ViT CAT-protocol checkpoint on CLAIMFORGE inputs.

IML-ViT is a native pixel localizer without an image-level classification
head.  This adapter therefore emits T2 results only and never derives a T1
score from the predicted map.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
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
from eval.opensource.imlvit_metrics import (
    binary_pixel_metrics_strict,
    summarize_imlvit_results,
)


MODEL_NAME = "IML-ViT"
MODEL_SLUG = "imlvit_cat_protocol_2023_official"
MODEL_REPO_URL = "https://github.com/SunnyHaze/IML-ViT"
MODEL_SOURCE_COMMIT = "07dd2be0f4ea27a5c97c9fa5ffbe236733833eac"
MODEL_INPUT_SIZE = 1024
MASK_THRESHOLD = 0.5

SOURCE_FILES = {
    "README.md": "a5356aa04266719cb248723e07aca6db6fcc9d122ab2a08c20ce012d53793764",
    "Demo.ipynb": "98eff570ed0b98ac6f4db99d01ae2b07fb55684efce030f7df334f80cfbac4ba",
    "iml_vit_model.py": "6d3b4ba1749bbf6b188e68f9657fff961f8a42bce818042fc1ba064fd64b7048",
    "modules/decoderhead.py": "a3a1ac5c16a16a17ae65567e6670eb97b92f7cb6c56ef665f8ed80316fa3bd69",
    "modules/window_attention_ViT.py": "382eabf15045420c827c5ad4bd1e44fa32b254b3c1139d6040f32df6a6bcdbde",
    "utils/iml_transforms.py": "3abfda4ed3777f93db55dc05922c7f745a3fcb72d952409a195f43e1b33b36f2",
    "utils/evaluation.py": "ac8ada749ffc4acfaf47d67550be642c96291640a5b130971f25a6857a3962aa",
    "LICENSE": "7bce5d24d372c0abbf618951988ee2dc072e60027c55615f0229bdab0dad73c3",
}

CHECKPOINT = {
    "provider": "official_author_google_drive",
    "announcement_commit": "5ad22146b1223eac841fa3e0e28c1c4e8948cc95",
    "release_folder_id": "1Ztyiy2cKJVmyusYMUlwuyPecBefTJCPT",
    "file_id": "1jlXw97GkyBbY4u5-e_liuhahKSQWCAFu",
    "original_filename": "iml-vit_checkpoint_trufor_20231104.pth",
    "release_file_mtime_utc": "2024-03-24T06:52:08+00:00",
    "bytes": 367_195_954,
    "sha256": "9fa9ae88cafeb6eab28c2afd5bef74679416cf0a790b2370fa6a6fb4c122c58c",
    "container": "collections.OrderedDict_raw_state_dict",
    "state_keys": 212,
    "tensor_values": 212,
    "state_elements": 91_778_242,
    "tensor_bytes": 367_112_972,
    "state_dtypes": {"torch.float32": 211, "torch.int64": 1},
    "parameters": 91_777_729,
    "buffers": 513,
}

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_IMLVIT_ROOT = Path("/root/.cache/claimforge/third_party/IML-ViT")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/imlvit-official/"
    "iml-vit_checkpoint_trufor_20231104.pth"
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
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], dict[str, Any]]:
    """Apply the paper protocol: shrink long side if needed, then zero-pad.

    IML-ViT section 4.1 states that images larger than 1024 are resized on
    their longer side while preserving aspect ratio; all images are then
    top-left placed in a 1024 square.  Padding is raw RGB zero before ImageNet
    normalization.
    """

    import albumentations as albu
    import cv2

    with path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            decoder_format = opened.format
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"unexpected decoded image shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image dtype: {decoded.dtype}")
    native_height, native_width = decoded.shape[:2]

    if max(native_height, native_width) > MODEL_INPUT_SIZE:
        resize = albu.LongestMaxSize(
            max_size=MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
            always_apply=True,
        )
        resized = np.asarray(resize(image=decoded)["image"], dtype=np.uint8)
        resize_policy = "albumentations_longest_max_size_downscale_only"
    else:
        resized = decoded
        resize_policy = "none_image_within_1024_limit"
    resized_height, resized_width = resized.shape[:2]
    if (
        resized_height > MODEL_INPUT_SIZE
        or resized_width > MODEL_INPUT_SIZE
        or max(resized_height, resized_width) <= 0
    ):
        raise ValueError("IML-ViT resized content dimensions are invalid")

    canvas = np.zeros(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
        dtype=np.uint8,
    )
    canvas[:resized_height, :resized_width] = resized
    normalize = albu.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=True,
    )
    normalized = np.asarray(normalize(image=canvas)["image"], dtype=np.float32)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected IML-ViT input shape: {chw.shape}")
    if not np.isfinite(chw).all():
        raise ValueError("IML-ViT normalized input contains non-finite values")

    metadata = {
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "native_size": [native_width, native_height],
        "resized_content_size": [resized_width, resized_height],
        "model_canvas_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "resize_policy": resize_policy,
        "resize_interpolation": "cv2.INTER_LINEAR_via_albumentations",
        "resize_scale_x": resized_width / native_width,
        "resize_scale_y": resized_height / native_height,
        "aspect_ratio_preserved_with_rounding": True,
        "padding": {
            "placement": "top_left",
            "right_pixels": MODEL_INPUT_SIZE - resized_width,
            "bottom_pixels": MODEL_INPUT_SIZE - resized_height,
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


def postprocess_head_logits(
    head_logits: Any,
    *,
    native_width: int,
    native_height: int,
    resized_width: int,
    resized_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample logits, sigmoid once, crop padding, and restore native size."""

    import torch
    from torch.nn import functional as F

    if (
        not isinstance(head_logits, torch.Tensor)
        or head_logits.ndim != 4
        or tuple(head_logits.shape[:2]) != (1, 1)
    ):
        shape = getattr(head_logits, "shape", None)
        raise ValueError(f"unexpected IML-ViT head-logit shape: {shape}")
    if not (0 < resized_width <= MODEL_INPUT_SIZE):
        raise ValueError("invalid resized width")
    if not (0 < resized_height <= MODEL_INPUT_SIZE):
        raise ValueError("invalid resized height")
    if native_width <= 0 or native_height <= 0:
        raise ValueError("invalid native size")

    model_logits_tensor = F.interpolate(
        head_logits,
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    model_score_tensor = torch.sigmoid(model_logits_tensor)
    valid_score_tensor = model_score_tensor[
        :,
        :,
        :resized_height,
        :resized_width,
    ]
    if (resized_width, resized_height) == (native_width, native_height):
        native_score_tensor = valid_score_tensor
    else:
        native_score_tensor = F.interpolate(
            valid_score_tensor,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )

    model_logits = np.ascontiguousarray(
        model_logits_tensor[0, 0].float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
    )
    model_score = np.ascontiguousarray(
        model_score_tensor[0, 0].float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
    )
    native_score = np.ascontiguousarray(
        native_score_tensor[0, 0].float().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
    )
    if model_logits.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("IML-ViT model-logit map shape changed")
    if model_score.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("IML-ViT model score-map shape changed")
    if native_score.shape != (native_height, native_width):
        raise ValueError("IML-ViT native score-map shape changed")
    if not np.isfinite(model_logits).all():
        raise ValueError("IML-ViT model logits contain non-finite values")
    for label, array in (
        ("model score map", model_score),
        ("native score map", native_score),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"IML-ViT {label} contains non-finite values")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"IML-ViT {label} falls outside [0, 1]")
    return model_logits, model_score, native_score


def load_model(
    *,
    imlvit_root: Path,
    checkpoint_path: Path,
    device_name: str,
) -> tuple[Any, Any]:
    import torch

    source_commit = _git_value(imlvit_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"IML-ViT source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        imlvit_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("IML-ViT tracked source files have local modifications")
    for relative, expected in SOURCE_FILES.items():
        _verify_runtime_file(
            imlvit_root / relative,
            expected,
            f"IML-ViT source file {relative}",
        )
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "IML-ViT CAT-protocol checkpoint",
    )
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("IML-ViT checkpoint byte-size mismatch")

    if str(imlvit_root) not in sys.path:
        sys.path.insert(0, str(imlvit_root))
    with _working_directory(imlvit_root):
        from iml_vit_model import iml_vit_model

        model = iml_vit_model(vit_pretrain_path=None)
    parameters = sum(int(value.numel()) for value in model.parameters())
    buffers = sum(int(value.numel()) for value in model.buffers())
    if parameters != int(CHECKPOINT["parameters"]):
        raise ValueError("IML-ViT parameter-count mismatch")
    if buffers != int(CHECKPOINT["buffers"]):
        raise ValueError("IML-ViT buffer-count mismatch")

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if type(state).__name__ != "OrderedDict" or not isinstance(state, Mapping):
        raise ValueError("IML-ViT checkpoint is not the registered raw OrderedDict")
    if len(state) != int(CHECKPOINT["state_keys"]):
        raise ValueError("IML-ViT checkpoint state-key count mismatch")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("IML-ViT checkpoint contains a non-tensor state value")
    state_elements = sum(int(value.numel()) for value in state.values())
    state_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in state.values()
    )
    dtype_counts = Counter(str(value.dtype) for value in state.values())
    if state_elements != int(CHECKPOINT["state_elements"]):
        raise ValueError("IML-ViT checkpoint state-element count mismatch")
    if state_bytes != int(CHECKPOINT["tensor_bytes"]):
        raise ValueError("IML-ViT checkpoint tensor-byte count mismatch")
    if dict(dtype_counts) != CHECKPOINT["state_dtypes"]:
        raise ValueError("IML-ViT checkpoint dtype schema mismatch")
    model.load_state_dict(state, strict=True)

    device = torch.device(device_name)
    model.to(device)
    model.eval()
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    import torch

    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("IML-ViT input tensor shape changed unexpectedly")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        features = model.encoder_net(image)
        pyramid = model.featurePyramid_net(features)
        feature_list = list(pyramid.values())
        if len(feature_list) != 5:
            raise ValueError(
                f"IML-ViT feature pyramid returned {len(feature_list)} levels"
            )
        head_logits = model.predict_head(feature_list)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0
    model_logits, model_score, native_score = postprocess_head_logits(
        head_logits,
        native_width=native_width,
        native_height=native_height,
        resized_width=resized_width,
        resized_height=resized_height,
    )
    return model_logits, model_score, native_score, peak_bytes, latency_ms


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
    import cv2

    truth = np.asarray(target, dtype=np.uint8)
    resized = cv2.resize(
        truth,
        (resized_width, resized_height),
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
    imlvit_root: Path,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    import torch

    ordered_inputs = _selection_contract(selected)
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
                    imlvit_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "variant": "official_CAT_TruFor_protocol_checkpoint_20231104",
            "checkpoint_filename_note": (
                "the official release filename says trufor; README describes "
                "this weight as the CAT-Net protocol checkpoint and notes "
                "that TruFor follows that protocol"
            ),
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
                "schema_fallbacks": False,
                "prefix_rewrites": False,
                "mae_initialization_reloaded": False,
            },
            "parameter_count": CHECKPOINT["parameters"],
            "buffer_elements": CHECKPOINT["buffers"],
            "supports_image_level_t1": False,
            "image_score_source": None,
            "supports_pixel_level_t2": True,
            "primary_localization_output": (
                "sigmoid_of_bilinearly_upsampled_predict_head_logits"
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
            "input_geometry": {
                "protocol_reference": {
                    "paper": "https://arxiv.org/abs/2307.14863",
                    "version": "v4",
                    "section": "4.1",
                },
                "paper_protocol": (
                    "if max(H,W)>1024, resize longer side to 1024 while "
                    "preserving aspect ratio; otherwise keep native size; "
                    "top-left place and raw-zero pad right/bottom to 1024"
                ),
                "large_image_resize": (
                    "albumentations.LongestMaxSize_max_size_1024_"
                    "cv2_INTER_LINEAR_downscale_only_py3round"
                ),
                "small_image_resize": "none",
                "canvas": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "placement": "top_left",
                "padding_value_before_normalization": 0,
                "crop": None,
                "reason": (
                    "paper section 4.1; uses the intended conditional "
                    "LongestMaxSize semantics without copying the hosted "
                    "Colab PIL width-height bug, and avoids the README "
                    "demo's destructive top-left crop for large images"
                ),
            },
            "input_reencode": False,
            "normalization": {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "raw_head_output": "one_channel_logits_at_256x256",
            "model_logit_restore": (
                "bilinear_to_1024x1024_align_corners_false"
            ),
            "model_probability": "single_sigmoid_after_logit_restore",
            "native_restore": (
                "crop_right_bottom_padding_then_bilinear_probability_to_"
                "native_align_corners_false"
            ),
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "strict_greater_than",
            "test_time_augmentation": False,
            "ensemble": False,
        },
        "metrics": {
            "task": "T2_pixel_localization_only",
            "positive_class": "manipulated_pixel",
            "t1_policy": "unsupported_no_derived_image_score",
            "mask_threshold": args.mask_threshold,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "localization_spaces": ["model_1024", "native"],
            "model_space_policy": (
                "metrics use only the valid resized-content rectangle; "
                "right/bottom padding is excluded"
            ),
            "model_space_gt_resize": "cv2_INTER_NEAREST",
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "raw_logits_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            },
            "score_maps_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
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
                    Path(__file__).with_name("imlvit_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("imlvit_metrics.py")
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
        "timm": _package_version("timm"),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "albumentations": _package_version("albumentations"),
        "opencv": _package_version("opencv-python-headless"),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("official IML-ViT mask threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    imlvit_root = args.imlvit_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/imlvit/{args.run_id}"),
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
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(run_manifest_path, run_manifest)
    existing = read_latest_by_id(output_path)
    pending = [
        row
        for row in selected
        if existing.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"IML-ViT run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(
            f"loaded official IML-ViT CAT-protocol checkpoint "
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
                    (
                        model_logits,
                        model_score,
                        native_score,
                        peak_bytes,
                        latency_ms,
                    ) = infer_one(
                        model,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                        resized_width=resized_width,
                        resized_height=resized_height,
                    )
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
                            model_score[:resized_height, :resized_width],
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_pixel_metrics_strict(
                            native_score,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    raw_logits_path = (
                        artifact_dir
                        / "raw_logits_model_1024"
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
                    _atomic_save_npy(raw_logits_path, model_logits)
                    _atomic_save_npy(model_score_path, model_score)
                    _atomic_save_npy(native_score_path, native_score)
                    _atomic_save_mask(
                        native_mask_path,
                        native_score > args.mask_threshold,
                    )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "raw_logits_model_path": repo_relative(
                            raw_logits_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(raw_logits_path),
                        "raw_logits_model_shape": list(model_logits.shape),
                        "raw_logits_model_dtype": str(model_logits.dtype),
                        "score_map_model_path": repo_relative(
                            model_score_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(model_score_path),
                        "score_map_model_shape": list(model_score.shape),
                        "score_map_model_dtype": str(model_score.dtype),
                        "model_valid_content_size": [
                            resized_width,
                            resized_height,
                        ],
                        "score_map_path": repo_relative(
                            native_score_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(native_score_path),
                        "score_map_shape": list(native_score.shape),
                        "score_map_dtype": str(native_score.dtype),
                        "mask_path": repo_relative(
                            native_mask_path,
                            repo_root,
                        ),
                        "mask_sha256": sha256_file(native_mask_path),
                        "mask_shape": list(native_score.shape),
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
                        f"IML-ViT failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_imlvit_results(
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
        raise RuntimeError(f"incomplete IML-ViT run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--imlvit-root", type=Path, default=DEFAULT_IMLVIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/imlvit"),
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
