#!/usr/bin/env python3
"""Run the official PSCC-Net checkpoint on canonical CLAIMFORGE inputs."""

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
from pathlib import Path
from typing import Any, Iterator

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
from eval.opensource.psccnet_metrics import (
    binary_pixel_metrics_strict,
    summarize_psccnet_results,
)


MODEL_NAME = "PSCC-Net"
MODEL_SLUG = "psccnet_tcsvt2022_official"
MODEL_REPO_URL = "https://github.com/proteus1991/PSCC-Net"
MODEL_SOURCE_COMMIT = "53e5ff77d8dc5feddda060cd085f9b765761f816"
MODEL_CROP_SIZE = (256, 256)
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

SOURCE_FILES = {
    "test.py": "07d9911da874fb8dfe86658b0fb22f8736f23ae60997532c05b3415a3ab07458",
    "models/NLCDetection.py": "325d4a9402e4fe640ecfb0abd24a9bc4b490bc567fc80d3e93ad64936b9757a6",
    "models/detection_head.py": "f42116d0dc055f7e7333f57a5e2abee6d3091307fbfc16904a0152d826a07368",
    "models/seg_hrnet.py": "ccb71be379e591eb84329c34500ecc015a157c92ee73327e392b6187672d78e9",
    "models/seg_hrnet_config.py": "a933009348e8394a96c0c89d930d18ffb0ec10f693e58f4787a2ed3cb939a7d5",
    "utils/load_vdata.py": "eecc6bec8920e30367ddcf63f72e396c492750fd65ce6ccd833ada301af893d7",
    "utils/config.py": "24a52c41f3f227256be74fd6dec17bec6907fb4007321dbeae0ed926ee1423ed",
    "LICENSE": "6644ee3539b171a7ebd39193786e0654456c57df6ba0b6f389fdb9a7b09685c9",
}

INITIALIZATION_WEIGHT = {
    "path": "models/hrnet_w18_small_v2.pth",
    "bytes": 16_012_341,
    "sha256": "06924c741ea8c076a569d5e164aa628910a72020800e4a4945e8b40b241ce5cb",
}

CHECKPOINTS = {
    "feature_extractor": {
        "path": "checkpoint/HRNet_checkpoint/HRNet.pth",
        "bytes": 8_305_545,
        "sha256": "d3b21edc4930187a6801cc818bd7b999fb5d8078d8f2e2193572e91ea5160096",
        "state_keys": 444,
        "state_elements": 2_037_538,
        "parameters": 2_028_572,
        "buffers": 8_966,
    },
    "localization_head": {
        "path": "checkpoint/NLCDetection_checkpoint/NLCDetection.pth",
        "bytes": 2_900_709,
        "sha256": "11ea3461253cf059b299ad4b6b89008485f94a2d2b2da83ec28c2282a095b00b",
        "state_keys": 64,
        "state_elements": 719_824,
        "parameters": 719_824,
        "buffers": 0,
    },
    "classification_head": {
        "path": "checkpoint/DetectionHead_checkpoint/DetectionHead.pth",
        "bytes": 3_739_969,
        "sha256": "a17581e8a3489a360257a266ca9b2db1b7c9b43337fbcc4aeb8d751f593f66f5",
        "state_keys": 128,
        "state_elements": 924_070,
        "parameters": 919_546,
        "buffers": 4_524,
    },
}
CHECKPOINT_BUNDLE_SHA256 = hashlib.sha256(
    stable_json(
        {
            name: value["sha256"]
            for name, value in CHECKPOINTS.items()
        }
    ).encode("utf-8")
).hexdigest()

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_PSCCNET_ROOT = Path(
    "/root/.cache/claimforge/third_party/PSCC-Net"
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
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    """Reproduce the official ImageIO RGB float-[0,1] test preprocessing."""

    import imageio.v2 as imageio

    decoded = np.asarray(imageio.imread(path))
    if decoded.ndim == 2:
        decoded = np.repeat(decoded[..., None], 3, axis=2)
        alpha_policy = "grayscale_repeated_to_rgb"
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        rgba = decoded.astype(np.float32)
        alpha = rgba[..., 3:4] / np.float32(255.0)
        decoded = (
            rgba[..., :3] * alpha
            + np.float32(255.0) * (np.float32(1.0) - alpha)
        ).astype(np.uint8)
        alpha_policy = "official_white_background_rgba_composite"
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        alpha_policy = "not_applicable"
    else:
        raise ValueError(f"unexpected decoded image shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image dtype: {decoded.dtype}")
    height, width = decoded.shape[:2]
    chw = np.ascontiguousarray(
        decoded.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0),
        dtype=np.float32,
    )
    metadata = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "native_size": [width, height],
        "input_resize": "none",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255",
        "alpha_policy": alpha_policy,
        "tensor_shape": list(chw.shape),
        "tensor_sha256": _sha256_array(chw),
    }
    return chw, (width, height), metadata


def postprocess_outputs(
    progressive_masks: Any,
    logits: Any,
    *,
    native_width: int,
    native_height: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact official primary-head, softmax, and native restore."""

    import torch
    from torch.nn import functional as F

    if not isinstance(progressive_masks, (tuple, list)):
        raise ValueError("PSCC-Net localization output is not a sequence")
    if len(progressive_masks) != 4:
        raise ValueError(
            f"PSCC-Net returned {len(progressive_masks)} masks, expected 4"
        )
    expected_shapes = ((256, 256), (128, 128), (64, 64), (32, 32))
    model_masks: list[np.ndarray] = []
    for index, (mask, expected) in enumerate(
        zip(progressive_masks, expected_shapes, strict=True),
        start=1,
    ):
        if tuple(mask.shape) != (1, 1, *expected):
            raise ValueError(
                f"unexpected PSCC-Net mask{index} shape: {tuple(mask.shape)}"
            )
        array = (
            mask[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        )
        if not np.isfinite(array).all():
            raise ValueError(f"PSCC-Net mask{index} contains non-finite values")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"PSCC-Net mask{index} falls outside [0, 1]")
        model_masks.append(np.ascontiguousarray(array))

    if tuple(logits.shape) != (1, 2):
        raise ValueError(f"unexpected PSCC-Net logits shape: {tuple(logits.shape)}")
    probabilities = torch.softmax(logits, dim=1)
    logits_array = (
        logits[0].float().cpu().numpy().astype(np.float32, copy=False)
    )
    probabilities_array = (
        probabilities[0].float().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(logits_array).all():
        raise ValueError("PSCC-Net logits contain non-finite values")
    if (
        not np.isfinite(probabilities_array).all()
        or float(probabilities_array.min()) < 0.0
        or float(probabilities_array.max()) > 1.0
    ):
        raise ValueError("PSCC-Net probabilities are invalid")

    # Official test.py fixes mask1 before inference and restores the sigmoid
    # probability (not a logit) with bilinear align_corners=True.
    native = F.interpolate(
        progressive_masks[0],
        size=(native_height, native_width),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    native_array = (
        native.float().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(native_array).all():
        raise ValueError("PSCC-Net native map contains non-finite values")
    if float(native_array.min()) < 0.0 or float(native_array.max()) > 1.0:
        raise ValueError("PSCC-Net native map falls outside [0, 1]")
    return (
        model_masks,
        np.ascontiguousarray(native_array),
        np.ascontiguousarray(logits_array),
        np.ascontiguousarray(probabilities_array),
    )


def _load_state(
    *,
    module: Any,
    path: Path,
    contract: dict[str, Any],
) -> None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"PSCC-Net checkpoint is not a mapping: {path}")
    if len(state) != int(contract["state_keys"]):
        raise ValueError(
            f"PSCC-Net state-key mismatch for {path.name}: "
            f"{len(state)} != {contract['state_keys']}"
        )
    elements = sum(
        int(value.numel())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    if elements != int(contract["state_elements"]):
        raise ValueError(
            f"PSCC-Net state-element mismatch for {path.name}: "
            f"{elements} != {contract['state_elements']}"
        )
    module.load_state_dict(state, strict=True)


def _wrap_for_official_state(module: Any, device: Any) -> Any:
    import torch

    module.to(device)
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        return torch.nn.DataParallel(
            module,
            device_ids=[index],
            output_device=index,
        )
    # Official checkpoints were saved from DataParallel.  On CPU, use a
    # lightweight wrapper so strict loading retains the exact "module." keys.
    class _CPUWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.module = wrapped

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.module(*args, **kwargs)

    return _CPUWrapper(module)


def load_model(
    *,
    psccnet_root: Path,
    device_name: str,
) -> tuple[tuple[Any, Any, Any], Any]:
    import torch

    source_commit = _git_value(psccnet_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"PSCC-Net source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        psccnet_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("PSCC-Net tracked source files have local modifications")
    for relative, expected in SOURCE_FILES.items():
        _verify_runtime_file(
            psccnet_root / relative,
            expected,
            f"PSCC-Net source file {relative}",
        )
    initialization_path = psccnet_root / str(INITIALIZATION_WEIGHT["path"])
    _verify_runtime_file(
        initialization_path,
        str(INITIALIZATION_WEIGHT["sha256"]),
        "PSCC-Net HRNet initialization weight",
    )
    if initialization_path.stat().st_size != int(
        INITIALIZATION_WEIGHT["bytes"]
    ):
        raise ValueError("PSCC-Net HRNet initialization byte-size mismatch")
    for role, contract in CHECKPOINTS.items():
        path = psccnet_root / str(contract["path"])
        _verify_runtime_file(
            path,
            str(contract["sha256"]),
            f"PSCC-Net {role} checkpoint",
        )
        if path.stat().st_size != int(contract["bytes"]):
            raise ValueError(
                f"PSCC-Net {role} checkpoint byte-size mismatch"
            )

    if str(psccnet_root) not in sys.path:
        sys.path.insert(0, str(psccnet_root))
    # Official code uses the removed np.int alias once when constructing
    # HRNet.  This compatibility alias changes no numerical behavior.
    if not hasattr(np, "int"):
        setattr(np, "int", int)
    with _working_directory(psccnet_root):
        from models.NLCDetection import NLCDetection
        from models.detection_head import DetectionHead
        from models.seg_hrnet import get_seg_model
        from models.seg_hrnet_config import get_hrnet_cfg

        config = {"crop_size": list(MODEL_CROP_SIZE)}
        feature_extractor = get_seg_model(get_hrnet_cfg())
        localization_head = NLCDetection(config)
        classification_head = DetectionHead(config)

    expected_counts = (
        ("feature_extractor", feature_extractor),
        ("localization_head", localization_head),
        ("classification_head", classification_head),
    )
    for role, module in expected_counts:
        contract = CHECKPOINTS[role]
        parameters = sum(int(value.numel()) for value in module.parameters())
        buffers = sum(int(value.numel()) for value in module.buffers())
        if parameters != int(contract["parameters"]):
            raise ValueError(f"PSCC-Net {role} parameter-count mismatch")
        if buffers != int(contract["buffers"]):
            raise ValueError(f"PSCC-Net {role} buffer-count mismatch")

    device = torch.device(device_name)
    feature_extractor = _wrap_for_official_state(feature_extractor, device)
    localization_head = _wrap_for_official_state(localization_head, device)
    classification_head = _wrap_for_official_state(
        classification_head,
        device,
    )
    for role, module in (
        ("feature_extractor", feature_extractor),
        ("localization_head", localization_head),
        ("classification_head", classification_head),
    ):
        _load_state(
            module=module,
            path=psccnet_root / str(CHECKPOINTS[role]["path"]),
            contract=CHECKPOINTS[role],
        )
        module.eval()
    return (
        feature_extractor,
        localization_head,
        classification_head,
    ), device


def infer_one(
    models: tuple[Any, Any, Any],
    device: Any,
    image_array: np.ndarray,
    *,
    native_width: int,
    native_height: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, float]:
    import torch

    feature_extractor, localization_head, classification_head = models
    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, native_height, native_width):
        raise ValueError("PSCC-Net input tensor shape changed unexpectedly")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        features = feature_extractor(image)
        progressive_masks = localization_head(features)
        logits = classification_head(features)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0
    (
        model_masks,
        native_map,
        logits_array,
        probabilities_array,
    ) = postprocess_outputs(
        progressive_masks,
        logits,
        native_width=native_width,
        native_height=native_height,
    )
    return (
        model_masks,
        native_map,
        logits_array,
        probabilities_array,
        peak_bytes,
        latency_ms,
    )


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


def _model_target(target: np.ndarray) -> np.ndarray:
    image = Image.fromarray(
        np.where(target, 255, 0).astype(np.uint8),
        mode="L",
    )
    resized = image.resize(MODEL_CROP_SIZE, resample=Image.Resampling.NEAREST)
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
    psccnet_root: Path,
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
                    psccnet_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "variant": "official_committed_pretrained_checkpoint",
            "training_manipulations": [
                "authentic",
                "splicing",
                "copy_move",
                "RFR_Net_object_removal_inpainting",
            ],
            "source_files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in SOURCE_FILES.items()
            ],
            "license": {
                "path": "LICENSE",
                "sha256": SOURCE_FILES["LICENSE"],
                "spdx": "MIT",
                "scope": "project_repository",
            },
            "initialization_weight": INITIALIZATION_WEIGHT,
            "checkpoint": {
                "provider": "official_author_git_repository",
                "source_commit": MODEL_SOURCE_COMMIT,
                "bundle_sha256": CHECKPOINT_BUNDLE_SHA256,
                "components": [
                    {"role": role, **contract}
                    for role, contract in CHECKPOINTS.items()
                ],
                "strict_load": True,
                "safe_weights_only_load": True,
            },
            "parameter_count": sum(
                int(value["parameters"])
                for value in CHECKPOINTS.values()
            ),
            "buffer_elements": sum(
                int(value["buffers"])
                for value in CHECKPOINTS.values()
            ),
            "class_names": ["authentic", "forged"],
            "positive_class_index": 1,
            "supports_image_level_t1": True,
            "image_score_source": "native_independent_classification_head",
            "supports_pixel_level_t2": True,
            "primary_localization_output": "progressive_mask1",
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "compatibility_shim": "numpy.int=builtin_int_for_hrnet_constructor",
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "imageio.v2.imread",
            "channel_order": "RGB",
            "input_resize": "none",
            "input_crop": None,
            "input_reencode": False,
            "normalization": "uint8_rgb_divide_255",
            "feature_extractor": "HRNet-W18-small-v2",
            "internal_crop_size": list(MODEL_CROP_SIZE),
            "progressive_output_shapes": [
                [256, 256],
                [128, 128],
                [64, 64],
                [32, 32],
            ],
            "primary_map": "progressive_mask1_sigmoid_probability",
            "primary_map_selection": "fixed_by_official_test_py_index_0",
            "native_restore": (
                "bilinear_probability_align_corners_true_to_input_size"
            ),
            "classification_output": (
                "softmax_two_class_logits_positive_index_1"
            ),
            "classification_threshold": args.classification_threshold,
            "classification_threshold_comparison": "strict_greater_than",
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "strict_greater_than",
            "test_time_augmentation": False,
            "ensemble": False,
        },
        "metrics": {
            "task": "T1_image_detection_and_T2_pixel_localization",
            "positive_class": "forged_or_manipulated",
            "classification_threshold": args.classification_threshold,
            "mask_threshold": args.mask_threshold,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "model_space_gt_resize": "nearest_neighbor_to_256x256",
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
                    Path(__file__).with_name("psccnet_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("psccnet_metrics.py")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.classification_threshold) != CLASSIFICATION_THRESHOLD:
        raise ValueError(
            "official PSCC-Net classification threshold must be 0.5"
        )
    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("official PSCC-Net mask threshold must be 0.5")
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    psccnet_root = args.psccnet_root.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/psccnet/{args.run_id}"),
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
        psccnet_root=psccnet_root,
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
        f"PSCC-Net run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    models = None
    if pending:
        models, device = load_model(
            psccnet_root=psccnet_root,
            device_name=args.device,
        )
        print(
            f"loaded official PSCC-Net bundle "
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
                    (
                        model_masks,
                        native_map,
                        logits,
                        probabilities,
                        peak_bytes,
                        latency_ms,
                    ) = infer_one(
                        models,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                    )
                    score = float(probabilities[1])
                    decision = "forged" if score > 0.5 else "authentic"
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = _model_target(target_native)
                    localization = {
                        "model_256": binary_pixel_metrics_strict(
                            model_masks[0],
                            target_model,
                            args.mask_threshold,
                            include_ap=input_row["kind"] == "forged",
                        ),
                        "native": binary_pixel_metrics_strict(
                            native_map,
                            target_native,
                            args.mask_threshold,
                            include_ap=input_row["kind"] == "forged",
                        ),
                    }

                    primary_model_path = (
                        artifact_dir
                        / "score_maps_model_256"
                        / f"{sample_id}.npy"
                    )
                    native_map_path = (
                        artifact_dir
                        / "score_maps_native"
                        / f"{sample_id}.npy"
                    )
                    native_mask_path = (
                        artifact_dir
                        / "masks_native"
                        / f"{sample_id}.png"
                    )
                    _atomic_save_npy(primary_model_path, model_masks[0])
                    _atomic_save_npy(native_map_path, native_map)
                    _atomic_save_mask(
                        native_mask_path,
                        native_map > args.mask_threshold,
                    )
                    progressive_artifacts = [
                        {
                            "stage": 1,
                            "path": repo_relative(
                                primary_model_path,
                                repo_root,
                            ),
                            "sha256": sha256_file(primary_model_path),
                            "shape": list(model_masks[0].shape),
                            "primary": True,
                        }
                    ]
                    for stage, model_mask in enumerate(
                        model_masks[1:],
                        start=2,
                    ):
                        stage_path = (
                            artifact_dir
                            / f"progressive_mask{stage}"
                            / f"{sample_id}.npy"
                        )
                        _atomic_save_npy(stage_path, model_mask)
                        progressive_artifacts.append(
                            {
                                "stage": stage,
                                "path": repo_relative(
                                    stage_path,
                                    repo_root,
                                ),
                                "sha256": sha256_file(stage_path),
                                "shape": list(model_mask.shape),
                                "primary": False,
                            }
                        )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": score,
                        "score_source": "native_classification_head",
                        "score_semantics": (
                            "softmax_probability_class_1_forged"
                        ),
                        "classification_logits": [
                            float(value) for value in logits
                        ],
                        "classification_probabilities": [
                            float(value) for value in probabilities
                        ],
                        "classification_threshold": (
                            args.classification_threshold
                        ),
                        "classification_threshold_operator": ">",
                        "decision": decision,
                        "progressive_maps": progressive_artifacts,
                        "primary_model_score_map_path": repo_relative(
                            primary_model_path,
                            repo_root,
                        ),
                        "primary_model_score_map_sha256": sha256_file(
                            primary_model_path
                        ),
                        "primary_model_score_map_shape": list(
                            model_masks[0].shape
                        ),
                        "score_map_path": repo_relative(
                            native_map_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(native_map_path),
                        "score_map_shape": list(native_map.shape),
                        "mask_path": repo_relative(
                            native_mask_path,
                            repo_root,
                        ),
                        "mask_sha256": sha256_file(native_mask_path),
                        "mask_shape": list(native_map.shape),
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
                        f" score={row['score']:.6f} "
                        f"decision={row['decision']} "
                        f"f1={native_metrics.get('f1')} "
                        f"positive={native_metrics['predicted_positive_fraction']:.6f} "
                        f"latency={row['latency_ms']:.1f}ms"
                    )
                else:
                    detail = (
                        f" {row['error_type']}: {row['error_message']}"
                    )
                print(
                    f"[{index}/{len(pending)}] "
                    f"{input_row['task_id']} {input_row['kind']}: "
                    f"{row['status']}{detail}",
                    flush=True,
                )
                if row["status"] != "ok" and args.fail_fast:
                    raise RuntimeError(
                        f"PSCC-Net failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_psccnet_results(
        result_rows,
        selected,
        classification_threshold=args.classification_threshold,
        mask_threshold=args.mask_threshold,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": CHECKPOINT_BUNDLE_SHA256,
            "input_manifest_sha256": release["inputs_sha256"],
            "run_manifest_fingerprint": run_manifest["fingerprint"],
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
        raise RuntimeError(f"incomplete PSCC-Net run: {coverage}")
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
        "--psccnet-root",
        type=Path,
        default=DEFAULT_PSCCNET_ROOT,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/psccnet"),
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
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
