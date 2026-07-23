#!/usr/bin/env python3
"""Run the official CASIAv2-trained MVSS-Net on canonical CLAIMFORGE inputs."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
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
from eval.opensource.mvssnet_metrics import (
    binary_pixel_metrics_strict,
    summarize_mvssnet_results,
)


MODEL_NAME = "MVSS-Net (CASIAv2)"
MODEL_SLUG = "mvssnet_casiav2_iccv2021"
MODEL_REPO_URL = "https://github.com/dong03/MVSS-Net"
MODEL_SOURCE_COMMIT = "cc2aed77a823723015f95e4a6a3e344f3ddb7ccc"
MODEL_NETWORK_SHA256 = (
    "c05ef94536b27c46f138e0c1241b850df519fb605e5530307061b32e5f063350"
)
MODEL_INFERENCE_SHA256 = (
    "9561dfb72a42ce1fa8e6b08635de6ee0b858f3577f392093bb6a7af43b708602"
)
MODEL_EVALUATE_SHA256 = (
    "8d1aa0e1f4f48d6ceaa68238f50d1698a07c9c118744a3d70638d859eae43bf3"
)
MODEL_TOOLS_SHA256 = (
    "e18e93dd34aa67c5929bebc332bcd7d490a056e675c37756754a734f2e942c26"
)
MODEL_TRANSFORMS_SHA256 = (
    "f85427943560a3debd0dff29cb9f2407f26f04e8f3e7036325671d79f4243bf8"
)

CHECKPOINT_DRIVE_ID = "1MHoe91a24GiBMG2JYoghPRDd4Ro6RIVq"
CHECKPOINT_BYTES = 588_270_735
CHECKPOINT_SHA256 = (
    "080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50"
)
CHECKPOINT_STATE_KEYS = 800
CHECKPOINT_STATE_ELEMENTS = 146_994_922
MODEL_PARAMETER_COUNT = 146_880_335
MODEL_BUFFER_ELEMENTS = 114_587

MODEL_INPUT_SIZE = 512
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
NORMALIZE_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)
NORMALIZE_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_MVSSNET_ROOT = Path(
    "/root/.cache/claimforge/third_party/MVSS-Net"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/mvssnet-official/mvssnet_casia.pt"
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


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


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


def _atomic_save_gray_png(path: Path, array: np.ndarray) -> None:
    pixels = np.asarray(array)
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("PNG artifact must be a two-dimensional uint8 array")
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


def _verify_runtime_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _manifest_fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


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
    """Reproduce official cv2 BGR resize and old Albumentations normalization."""

    import cv2

    decoded_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    if decoded_bgr.dtype != np.uint8 or decoded_bgr.ndim != 3:
        raise ValueError("unexpected OpenCV image representation")
    height, width = decoded_bgr.shape[:2]
    resized_bgr = cv2.resize(
        decoded_bgr,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized_hwc = resized_bgr.astype(np.float32) / np.float32(255.0)
    normalized_hwc = (
        normalized_hwc - NORMALIZE_MEAN
    ) / NORMALIZE_STD
    normalized_chw = np.ascontiguousarray(
        normalized_hwc.transpose(2, 0, 1),
        dtype=np.float32,
    )
    metadata = {
        "native_size": [width, height],
        "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "decoder": "opencv_imread_color",
        "channel_order": "BGR",
        "resize": "opencv_inter_linear_stretch",
        "normalization": {
            "scale": "uint8_divide_255",
            "mean_in_bgr_order": NORMALIZE_MEAN.tolist(),
            "std_in_bgr_order": NORMALIZE_STD.tolist(),
        },
        "decoded_bgr_dtype": str(decoded_bgr.dtype),
        "decoded_bgr_shape": list(decoded_bgr.shape),
        "decoded_bgr_sha256": _sha256_array(decoded_bgr),
        "resized_bgr_dtype": str(resized_bgr.dtype),
        "resized_bgr_shape": list(resized_bgr.shape),
        "resized_bgr_sha256": _sha256_array(resized_bgr),
        "normalized_chw_dtype": str(normalized_chw.dtype),
        "normalized_chw_shape": list(normalized_chw.shape),
        "normalized_chw_sha256": _sha256_array(normalized_chw),
    }
    return normalized_chw, (width, height), metadata


def official_postprocess(
    model_score_map: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Reproduce ToPILImage(float)->uint8 then native uint8 OpenCV resize."""

    import cv2

    score_map = np.asarray(model_score_map, dtype=np.float32)
    if score_map.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected model map shape: {score_map.shape}")
    if not np.isfinite(score_map).all():
        raise ValueError("model map contains non-finite values")
    if float(score_map.min()) < 0.0 or float(score_map.max()) > 1.0:
        raise ValueError("model map falls outside [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError("native dimensions must be positive")
    # torchvision 0.6.1 ToPILImage multiplied a float tensor by 255 and cast
    # directly to byte. For a sigmoid map this is truncation in [0, 255].
    model_uint8 = (
        score_map * np.float32(255.0)
    ).astype(np.uint8)
    native_uint8 = cv2.resize(
        model_uint8,
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(native_uint8, dtype=np.uint8)


def resize_target(
    target: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        np.asarray(target, dtype=np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _load_official_module(mvssnet_root: Path):
    module_path = mvssnet_root / "models/mvssnet.py"
    spec = importlib.util.spec_from_file_location(
        "_claimforge_official_mvssnet",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official MVSS-Net module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(
    *,
    mvssnet_root: Path,
    checkpoint_path: Path,
    device_name: str,
):
    import torch

    source_commit = _git_value(mvssnet_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"MVSS-Net source commit mismatch: "
            f"{source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(
        mvssnet_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("MVSS-Net tracked source files have local modifications")
    for relative_path, expected_hash in (
        ("models/mvssnet.py", MODEL_NETWORK_SHA256),
        ("inference.py", MODEL_INFERENCE_SHA256),
        ("evaluate.py", MODEL_EVALUATE_SHA256),
        ("common/tools.py", MODEL_TOOLS_SHA256),
        ("common/transforms.py", MODEL_TRANSFORMS_SHA256),
    ):
        _verify_runtime_file(
            mvssnet_root / relative_path,
            expected_hash,
            f"official MVSS-Net source {relative_path}",
        )
    _verify_runtime_file(
        checkpoint_path,
        CHECKPOINT_SHA256,
        "official MVSS-Net CASIAv2 checkpoint",
    )
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("MVSS-Net checkpoint byte size mismatch")

    official = _load_official_module(mvssnet_root)
    original_load_url = official.model_zoo.load_url
    official.model_zoo.load_url = lambda *args, **kwargs: {}
    try:
        with _working_directory(mvssnet_root):
            model = official.get_mvss(
                backbone="resnet50",
                pretrained_base=True,
                nclass=1,
                sobel=True,
                constrain=True,
                n_input=3,
            )
    finally:
        official.model_zoo.load_url = original_load_url

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("MVSS-Net checkpoint is not a state dictionary")
    if len(checkpoint) != CHECKPOINT_STATE_KEYS:
        raise ValueError("MVSS-Net checkpoint state-key count mismatch")
    if not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError("MVSS-Net checkpoint contains a non-tensor value")
    state_elements = sum(int(value.numel()) for value in checkpoint.values())
    if state_elements != CHECKPOINT_STATE_ELEMENTS:
        raise ValueError("MVSS-Net checkpoint element count mismatch")
    model_state = model.state_dict()
    if len(model_state) != CHECKPOINT_STATE_KEYS:
        raise ValueError("official MVSS-Net constructor state-key count mismatch")
    result = model.load_state_dict(checkpoint, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("MVSS-Net strict checkpoint load was incomplete")
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    buffer_count = sum(int(value.numel()) for value in model.buffers())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise ValueError("MVSS-Net parameter count mismatch")
    if buffer_count != MODEL_BUFFER_ELEMENTS:
        raise ValueError("MVSS-Net buffer count mismatch")
    del checkpoint, model_state
    gc.collect()

    device = torch.device(device_name)
    model.float().to(device)
    model.eval()
    return model, device


def infer_one(
    model: Any,
    device: Any,
    tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    import torch

    image = torch.from_numpy(
        np.ascontiguousarray(tensor, dtype=np.float32)
    ).unsqueeze(0).to(device)
    if image.shape != (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected MVSS-Net input shape: {tuple(image.shape)}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        edge_logits, segmentation_logits = model(image)
        model_score_map = torch.sigmoid(segmentation_logits)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000

    if tuple(segmentation_logits.shape) != (
        1,
        1,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            "unexpected MVSS-Net segmentation output shape: "
            f"{tuple(segmentation_logits.shape)}"
        )
    if edge_logits.ndim != 4 or tuple(edge_logits.shape[:2]) != (1, 1):
        raise ValueError(
            f"unexpected MVSS-Net edge output shape: {tuple(edge_logits.shape)}"
        )
    raw_logits = (
        segmentation_logits[0, 0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    score_map = (
        model_score_map[0, 0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if not np.isfinite(raw_logits).all() or not np.isfinite(score_map).all():
        raise ValueError("MVSS-Net produced non-finite output")
    if float(score_map.min()) < 0.0 or float(score_map.max()) > 1.0:
        raise ValueError("MVSS-Net score map falls outside [0, 1]")
    return raw_logits, score_map, peak_bytes, latency_ms


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
    mvssnet_root: Path,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    import cv2
    import torch

    ordered_inputs = _selection_contract(selected)
    immutable = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "input": {
            "dataset_id": release["dataset_id"],
            "dataset_manifest": repo_relative(dataset_manifest_path, repo_root),
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
                    mvssnet_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "implementation": {
                "network_path": "models/mvssnet.py",
                "network_sha256": MODEL_NETWORK_SHA256,
                "inference_path": "inference.py",
                "inference_sha256": MODEL_INFERENCE_SHA256,
                "evaluation_path": "evaluate.py",
                "evaluation_sha256": MODEL_EVALUATE_SHA256,
                "tools_sha256": MODEL_TOOLS_SHA256,
                "transforms_sha256": MODEL_TRANSFORMS_SHA256,
            },
            "variant": "original_mvssnet_not_mvssnet_plus",
            "training_dataset": "CASIAv2",
            "constructor": {
                "backbone": "resnet50",
                "nclass": 1,
                "sobel": True,
                "constrain": True,
                "n_input": 3,
                "construction_pretrained_files_used": False,
                "reason": (
                    "the complete checkpoint strictly covers all 800 state "
                    "entries; unpinned ImageNet initialization is bypassed"
                ),
            },
            "license": {
                "license_file_present": False,
                "project_wide_status": "no_project_license_found",
                "weight_terms_status": "no_separate_weight_terms_found",
                "classification": "source_available_research_release",
                "commercial_or_redistribution_permission": "not_established",
            },
            "checkpoint": {
                "provider": "official_author_google_drive",
                "drive_file_id": CHECKPOINT_DRIVE_ID,
                "filename": checkpoint_path.name,
                "bytes": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
                "format": "raw_ordered_state_dict",
                "state_keys": CHECKPOINT_STATE_KEYS,
                "state_elements": CHECKPOINT_STATE_ELEMENTS,
                "parameter_count": MODEL_PARAMETER_COUNT,
                "buffer_elements": MODEL_BUFFER_ELEMENTS,
                "strict_load": True,
                "safe_weights_only_load": True,
            },
            "supports_image_level_t1": True,
            "image_level_head": "none_map_global_max_pooling",
            "supports_pixel_level_t2": True,
        },
        "inference": {
            "precision": "float32",
            "apex": "disabled",
            "apex_note": (
                "official legacy script used Apex O1; FP32 preserves source "
                "semantics and is the deterministic audited protocol"
            ),
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "input_source": "canonical_jpeg",
            "decoder": "opencv_imread_color",
            "channel_order": "BGR",
            "model_input_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "input_resize": "opencv_inter_linear_stretch",
            "normalization_mean_in_bgr_order": NORMALIZE_MEAN.tolist(),
            "normalization_std_in_bgr_order": NORMALIZE_STD.tolist(),
            "raw_output": "one_channel_segmentation_logits_512",
            "model_map": "sigmoid_of_segmentation_logits",
            "native_map_restore": (
                "multiply_model_probability_by_255_then_uint8_truncate_"
                "then_opencv_inter_linear_resize_at_uint8_dtype"
            ),
            "primary_t1_score": (
                "continuous_global_max_of_model_512_sigmoid_probability"
            ),
            "official_evaluate_t1_score": (
                "maximum_of_saved_native_uint8_map_divided_by_255"
            ),
            "image_level_head": "none_map_derived_gmp",
            "classification_threshold": args.classification_threshold,
            "classification_threshold_comparison": "strict_greater_than",
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "strict_greater_than",
            "edge_output_policy": "official_inference_discards_auxiliary_edge_logits",
            "bayar_constraint_state": {
                "behavior": (
                    "official source normalizes the constrained kernel in "
                    "place on every forward"
                ),
                "reproducibility_contract": (
                    "fresh checkpoint plus fixed ordered inputs; a partial "
                    "resume replays the completed prefix before new outputs"
                ),
            },
        },
        "metrics": {
            "tasks": ["T1_image_detection", "T2_pixel_localization"],
            "positive_class": "manipulated",
            "prediction_inversion": False,
            "primary_t1": "continuous_model_space_gmp",
            "secondary_t1": "official_saved_png_gmp",
            "primary_t2": "official_native_uint8_map_divided_by_255",
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
                    Path(__file__).with_name("mvssnet_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("mvssnet_metrics.py")
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
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": _package_version("Pillow"),
        "device": args.device,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
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


def _write_or_validate_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
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
            "official MVSS-Net protocol requires classification threshold 0.5"
        )
    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError(
            "official MVSS-Net protocol requires mask threshold 0.5"
        )
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    mvssnet_root = args.mvssnet_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/mvssnet/{args.run_id}"),
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
        mvssnet_root=mvssnet_root,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(run_manifest_path, run_manifest)
    existing = read_latest_by_id(output_path)
    completed_flags = [
        existing.get(str(row["sample_id"]), {}).get("status") == "ok"
        for row in selected
    ]
    completed_prefix = 0
    while (
        completed_prefix < len(completed_flags)
        and completed_flags[completed_prefix]
    ):
        completed_prefix += 1
    if any(completed_flags[completed_prefix:]):
        raise ValueError(
            "MVSS-Net resume requires successful results to form an ordered "
            "prefix because the official Bayar constraint mutates per forward"
        )
    if output_path.is_file() and completed_prefix < len(selected):
        history = read_jsonl(output_path)
        if any(row.get("status") == "error" for row in history):
            raise ValueError(
                "cannot deterministically resume MVSS-Net after an error row; "
                "use a new run ID"
            )
    pending = selected[completed_prefix:]
    print(
        f"{MODEL_NAME} run {args.run_id}: "
        f"{len(selected)} selected, {len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            mvssnet_root=mvssnet_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(f"loaded {MODEL_NAME} on {device}", flush=True)
        try:
            if completed_prefix:
                print(
                    "replaying "
                    f"{completed_prefix} completed inputs to restore the "
                    "official stateful Bayar constraint",
                    flush=True,
                )
                for completed_row in selected[:completed_prefix]:
                    completed_image = _anchored(
                        Path(str(completed_row["canonical_path"])),
                        repo_root,
                    )
                    replay_tensor, _, _ = preprocess_image(completed_image)
                    infer_one(model, device, replay_tensor)
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
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "valid_for_t1": True,
                    "valid_for_t2": True,
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    tensor, (width, height), preprocess = preprocess_image(
                        image_path
                    )
                    if (width, height) != (
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ):
                        raise ValueError("canonical image dimensions changed")
                    (
                        raw_logits,
                        model_score_map,
                        peak_bytes,
                        latency_ms,
                    ) = infer_one(model, device, tensor)
                    native_uint8 = official_postprocess(
                        model_score_map,
                        width,
                        height,
                    )
                    native_score_map = (
                        native_uint8.astype(np.float32)
                        / np.float32(255.0)
                    )
                    native_mask = native_score_map > args.mask_threshold
                    raw_score = float(np.max(model_score_map))
                    official_png_score = float(np.max(native_uint8)) / 255.0
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = resize_target(
                        target_native,
                        MODEL_INPUT_SIZE,
                        MODEL_INPUT_SIZE,
                    )
                    include_ap = input_row["kind"] == "forged"
                    localization = {
                        "model_512": binary_pixel_metrics_strict(
                            model_score_map,
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_pixel_metrics_strict(
                            native_score_map,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    raw_logits_path = (
                        artifact_dir
                        / "raw_logits_model_512"
                        / f"{sample_id}.npy"
                    )
                    model_map_path = (
                        artifact_dir
                        / "score_maps_model_512"
                        / f"{sample_id}.npy"
                    )
                    native_map_path = (
                        artifact_dir
                        / "score_maps_native_official"
                        / f"{sample_id}.png"
                    )
                    mask_path = (
                        artifact_dir
                        / "masks_native"
                        / f"{sample_id}.png"
                    )
                    _atomic_save_npy(
                        raw_logits_path,
                        raw_logits.astype(np.float32, copy=False),
                    )
                    _atomic_save_npy(
                        model_map_path,
                        model_score_map.astype(np.float32, copy=False),
                    )
                    _atomic_save_gray_png(native_map_path, native_uint8)
                    _atomic_save_gray_png(
                        mask_path,
                        np.where(native_mask, 255, 0).astype(np.uint8),
                    )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": raw_score,
                        "raw_score_semantics": (
                            "continuous_global_max_of_model_512_"
                            "sigmoid_probability"
                        ),
                        "decision": bool(
                            raw_score > args.classification_threshold
                        ),
                        "official_png_score": official_png_score,
                        "official_png_score_semantics": (
                            "maximum_of_saved_native_uint8_map_divided_by_255"
                        ),
                        "official_png_decision": bool(
                            official_png_score
                            > args.classification_threshold
                        ),
                        "score_source": "map_derived_no_separate_image_head",
                        "classification_threshold": (
                            args.classification_threshold
                        ),
                        "classification_threshold_operator": ">",
                        "raw_logits_model_path": repo_relative(
                            raw_logits_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(
                            raw_logits_path
                        ),
                        "raw_logits_model_shape": list(raw_logits.shape),
                        "score_map_model_path": repo_relative(
                            model_map_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(
                            model_map_path
                        ),
                        "score_map_model_shape": list(model_score_map.shape),
                        "score_map_native_path": repo_relative(
                            native_map_path,
                            repo_root,
                        ),
                        "score_map_native_sha256": sha256_file(
                            native_map_path
                        ),
                        "score_map_native_shape": list(native_uint8.shape),
                        "mask_path": repo_relative(mask_path, repo_root),
                        "mask_sha256": sha256_file(mask_path),
                        "mask_shape": list(native_mask.shape),
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
                        f"png_score={row['official_png_score']:.6f} "
                        f"f1={native_metrics.get('f1')} "
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
                        f"MVSS-Net failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_mvssnet_results(
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
            "checkpoint_sha256": CHECKPOINT_SHA256,
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
        raise RuntimeError(f"incomplete MVSS-Net run: {coverage}")
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
        "--mvssnet-root",
        type=Path,
        default=DEFAULT_MVSSNET_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/mvssnet"),
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
