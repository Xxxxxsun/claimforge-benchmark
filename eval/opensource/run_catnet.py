#!/usr/bin/env python3
"""Run the official CAT-Net v2 checkpoint on canonical CLAIMFORGE JPEGs."""

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
from eval.opensource.catnet_metrics import summarize_catnet_results
from eval.opensource.maskclip_metrics import binary_pixel_metrics


MODEL_REPO_URL = "https://github.com/mjkwon2021/CAT-Net"
MODEL_SOURCE_COMMIT = "b50d391ffc423d3631fd7947714468788c791805"
MODEL_CONFIG_SHA256 = (
    "81210f5f283504549d4e15a162bda453caea8b1692022d816a46bcc5427ed95d"
)
MODEL_NETWORK_SHA256 = (
    "7177d18a814d6eb55f43eee77aa652f5c9983ecbfc1d392f55a7261339ffc00f"
)
MODEL_LICENSE_SHA256 = (
    "f1f33c3bec144f048d1cbff4dcae8d47a28faf263930ce779c61a7f4913bf055"
)
CHECKPOINT_DRIVE_ID = "1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE"
CHECKPOINT_FILENAME = "CAT_full_v2.pth.tar"
# Filled from the author's public file and enforced before any inference.
CHECKPOINT_SHA256 = (
    "f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989"
)
CHECKPOINT_BYTES = 915_503_873
CHECKPOINT_EPOCH = 196
CHECKPOINT_STATE_KEYS = 2_926
MASK_THRESHOLD = 0.5
DCT_BINS = 21

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_CATNET_ROOT = Path("/root/.cache/claimforge/third_party/CAT-Net")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/catnet-v2/CAT_full_v2.pth.tar"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".png",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(
            np.where(mask, 255, 0).astype(np.uint8),
            mode="L",
        ).save(
            temporary,
            format="PNG",
            optimize=False,
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_sha256_int32(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.int32)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _manifest_fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _verify_runtime_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


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


def dct_volume_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Create the exact 21-bin absolute-DCT volume used by CAT-Net."""
    coefficient_array = np.asarray(coefficients)
    if coefficient_array.ndim != 2:
        raise ValueError("CAT-Net expects a two-dimensional luminance DCT array")
    absolute = np.abs(coefficient_array.astype(np.int64, copy=False))
    volume = np.empty(
        (DCT_BINS, coefficient_array.shape[0], coefficient_array.shape[1]),
        dtype=np.float32,
    )
    volume[0] = absolute == 0
    for index in range(1, DCT_BINS - 1):
        volume[index] = absolute == index
    volume[DCT_BINS - 1] = absolute >= DCT_BINS - 1
    if not np.all(volume.sum(axis=0) == 1):
        raise ValueError("DCT binning did not produce exactly one active bin")
    return volume


def preprocess_jpeg(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read RGB and luminance JPEG artifacts without resizing or re-encoding."""
    import jpegio

    with Image.open(path) as opened:
        if opened.format != "JPEG":
            raise ValueError(f"CAT-Net input is not JPEG: {path}")
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8

    jpeg = jpegio.read(str(path))
    if not jpeg.coef_arrays or not jpeg.comp_info:
        raise ValueError("jpegio returned no JPEG components")
    sampling = [
        [int(component.h_samp_factor), int(component.v_samp_factor)]
        for component in jpeg.comp_info
    ]
    if any(value != [1, 1] for value in sampling[:3]):
        raise ValueError(
            "canonical CAT-Net protocol requires 4:4:4 JPEG sampling"
        )
    coefficients = np.asarray(jpeg.coef_arrays[0], dtype=np.int32)
    if coefficients.shape != (padded_height, padded_width):
        raise ValueError(
            "luminance DCT shape does not match the ceil-8 image grid: "
            f"{coefficients.shape} != {(padded_height, padded_width)}"
        )
    qtable_index = int(jpeg.comp_info[0].quant_tbl_no)
    qtable = np.asarray(jpeg.quant_tables[qtable_index], dtype=np.int32)
    if qtable.shape != (8, 8):
        raise ValueError(f"unexpected luminance qtable shape: {qtable.shape}")

    padded_rgb = np.full(
        (padded_height, padded_width, 3),
        127.5,
        dtype=np.float32,
    )
    padded_rgb[:height, :width] = rgb
    normalized_rgb = (padded_rgb - 127.5) / 127.5
    dct_volume = dct_volume_from_coefficients(coefficients)
    image = np.empty(
        (3 + DCT_BINS, padded_height, padded_width),
        dtype=np.float32,
    )
    image[:3] = normalized_rgb.transpose(2, 0, 1)
    image[3:] = dct_volume
    qtable_float = np.ascontiguousarray(qtable[None], dtype=np.float32)
    metadata = {
        "native_size": [width, height],
        "padded_size": [padded_width, padded_height],
        "padding": {
            "left": 0,
            "top": 0,
            "right": padded_width - width,
            "bottom": padded_height - height,
        },
        "jpeg_sampling_factors": sampling,
        "luminance_qtable_index": qtable_index,
        "qtable_sha256": _array_sha256_int32(qtable),
        "dct_y_sha256": _array_sha256_int32(coefficients),
    }
    return image, qtable_float, metadata


def postprocess_logits(
    logits: Any,
    *,
    padded_width: int,
    padded_height: int,
    native_width: int,
    native_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply official bilinear-logits restore, then softmax and native crop."""
    import torch
    from torch.nn import functional as F

    if (
        logits.ndim != 4
        or int(logits.shape[0]) != 1
        or int(logits.shape[1]) != 2
    ):
        raise ValueError(f"unexpected CAT-Net output shape: {tuple(logits.shape)}")
    expected_shape = (padded_height // 4, padded_width // 4)
    if tuple(int(value) for value in logits.shape[-2:]) != expected_shape:
        raise ValueError(
            "unexpected CAT-Net quarter-resolution shape: "
            f"{tuple(logits.shape[-2:])} != {expected_shape}"
        )
    restored_logits = F.interpolate(
        logits,
        size=(padded_height, padded_width),
        mode="bilinear",
        align_corners=False,
    )
    score_map = torch.softmax(restored_logits, dim=1)[
        0, 1, :native_height, :native_width
    ]
    raw_logits = logits[0].float().cpu().numpy().astype(np.float32, copy=False)
    native_map = (
        score_map.float().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(raw_logits).all() or not np.isfinite(native_map).all():
        raise ValueError("CAT-Net produced non-finite output")
    if float(native_map.min()) < 0.0 or float(native_map.max()) > 1.0:
        raise ValueError("CAT-Net score map falls outside [0, 1]")
    return raw_logits, native_map


def load_model(
    *,
    catnet_root: Path,
    checkpoint_path: Path,
    device_name: str,
):
    import torch

    source_commit = _git_value(catnet_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"CAT-Net source commit mismatch: {source_commit} "
            f"!= {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(catnet_root, "status", "--short", "--untracked-files=no"):
        raise ValueError("CAT-Net tracked source files have local modifications")
    _verify_runtime_file(
        catnet_root / "experiments/CAT_full.yaml",
        MODEL_CONFIG_SHA256,
        "CAT-Net full-model configuration",
    )
    _verify_runtime_file(
        catnet_root / "lib/models/network_CAT.py",
        MODEL_NETWORK_SHA256,
        "CAT-Net network implementation",
    )
    _verify_runtime_file(
        catnet_root / "LICENSE of HRNet",
        MODEL_LICENSE_SHA256,
        "CAT-Net HRNet component license notice",
    )
    _verify_runtime_file(
        checkpoint_path,
        CHECKPOINT_SHA256,
        "CAT-Net v2 checkpoint",
    )
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(
            f"CAT-Net checkpoint size mismatch: "
            f"{checkpoint_path.stat().st_size} != {CHECKPOINT_BYTES}"
        )

    if str(catnet_root) not in sys.path:
        sys.path.insert(0, str(catnet_root))
    with _working_directory(catnet_root):
        from lib.config import config as base_config
        from lib.models.network_CAT import CAT_Net

        config = base_config.clone()
        config.defrost()
        config.merge_from_file(str(catnet_root / "experiments/CAT_full.yaml"))
        config.MODEL.PRETRAINED_RGB = ""
        config.MODEL.PRETRAINED_DCT = ""
        config.freeze()
        model = CAT_Net(config)
        checkpoint_safe_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(checkpoint_safe_globals):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
    if not isinstance(checkpoint, dict):
        raise ValueError("CAT-Net checkpoint is not a mapping")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("CAT-Net checkpoint has no state_dict mapping")
    if len(state) != CHECKPOINT_STATE_KEYS:
        raise ValueError(
            f"CAT-Net state-key count mismatch: "
            f"{len(state)} != {CHECKPOINT_STATE_KEYS}"
        )
    epoch = checkpoint.get("epoch")
    if int(epoch) != CHECKPOINT_EPOCH:
        raise ValueError(
            f"CAT-Net checkpoint epoch mismatch: {epoch} != {CHECKPOINT_EPOCH}"
        )
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()

    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return model, device


def infer_one(
    model: Any,
    device: Any,
    image_array: np.ndarray,
    qtable_array: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    import torch

    image = torch.from_numpy(image_array).unsqueeze(0).to(device)
    qtable = torch.from_numpy(qtable_array).unsqueeze(0).to(device)
    if tuple(image.shape[1:]) != (
        3 + DCT_BINS,
        int(metadata["padded_size"][1]),
        int(metadata["padded_size"][0]),
    ):
        raise ValueError("CAT-Net input tensor does not match preprocessing metadata")
    if tuple(qtable.shape) != (1, 1, 8, 8):
        raise ValueError(f"unexpected CAT-Net qtable shape: {tuple(qtable.shape)}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        logits = model(image, qtable)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000.0

    raw_logits, native_map = postprocess_logits(
        logits,
        padded_width=int(metadata["padded_size"][0]),
        padded_height=int(metadata["padded_size"][1]),
        native_width=int(metadata["native_size"][0]),
        native_height=int(metadata["native_size"][1]),
    )
    return raw_logits, native_map, peak_bytes, latency_ms


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
    catnet_root: Path,
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
            "name": "CAT-Net v2",
            "model_slug": "catnet_v2_ijcv2022",
            "repo_url": MODEL_REPO_URL,
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_tracked_clean": not bool(
                _git_value(
                    catnet_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "configuration": {
                "path": "experiments/CAT_full.yaml",
                "sha256": MODEL_CONFIG_SHA256,
                "network_sha256": MODEL_NETWORK_SHA256,
            },
            "network_implementation": {
                "path": "lib/models/network_CAT.py",
                "sha256": MODEL_NETWORK_SHA256,
            },
            "license": {
                "path": "LICENSE of HRNet",
                "sha256": MODEL_LICENSE_SHA256,
                "scope": "hrnet_component_only",
                "project_wide_status": (
                    "no_project_wide_license_found"
                ),
                "classification": "source_available_research_release",
            },
            "checkpoint": {
                "provider": "official_author_google_drive",
                "drive_file_id": CHECKPOINT_DRIVE_ID,
                "filename": checkpoint_path.name,
                "bytes": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
                "epoch": CHECKPOINT_EPOCH,
                "state_keys": CHECKPOINT_STATE_KEYS,
                "safe_load": (
                    "torch.load(weights_only=True) with three NumPy "
                    "dtype/scalar allowlisted globals"
                ),
                "strict_load": True,
                "safe_weights_only_load": True,
            },
            "class_names": ["authentic", "tampered"],
            "positive_class_index": 1,
            "supports_image_level_t1": False,
            "supports_pixel_level_t2": True,
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "input_source": "canonical_jpeg_original_bytes",
            "input_resize": "none",
            "input_crop": None,
            "input_reencode": False,
            "jpeg_reader": "jpegio",
            "jpeg_component": "luminance_y",
            "rgb_normalization": "(uint8 - 127.5) / 127.5",
            "padding": {
                "alignment": 8,
                "sides": ["bottom", "right"],
                "rgb_value": 127.5,
                "dct_value": 0,
            },
            "dct_volume": {
                "channels": DCT_BINS,
                "bins": "0, abs=1..19, abs>=20",
            },
            "qtable": "original_luminance_quantization_table",
            "raw_output": "two_channel_logits_at_quarter_resolution",
            "map_restore": (
                "bilinear_logits_to_padded_native_align_corners_false_"
                "then_softmax_channel_1_then_crop"
            ),
            "map_semantics": "probability_of_channel_1_tampered",
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "greater_than_or_equal",
            "t1_policy": "unsupported_no_derived_image_score",
        },
        "metrics": {
            "task": "T2_pixel_localization_only",
            "positive_class": "tampered",
            "fixed_threshold": args.mask_threshold,
            "prediction_inversion": False,
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
                    Path(__file__).with_name("catnet_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("catnet_metrics.py")
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
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "jpegio": _package_version("jpegio"),
        "yacs": _package_version("yacs"),
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

    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError(
            f"official CAT-Net protocol requires mask threshold {MASK_THRESHOLD}"
        )
    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    catnet_root = args.catnet_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/catnet/{args.run_id}"),
        repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.jsonl"
    run_manifest_path = output_dir / f"{args.run_id}.run_manifest.json"
    summary_path = output_dir / f"{args.run_id}.summary.json"

    release, inputs_path, all_rows = load_release(repo_root, dataset_manifest_path)
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
        catnet_root=catnet_root,
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
        f"CAT-Net v2 run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            catnet_root=catnet_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(f"loaded CAT-Net v2 epoch {CHECKPOINT_EPOCH} on {device}", flush=True)
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
                    "model": "CAT-Net v2",
                    "model_slug": "catnet_v2_ijcv2022",
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "valid_for_t1": False,
                    "valid_for_t2": True,
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    image_array, qtable_array, preprocess = preprocess_jpeg(
                        image_path
                    )
                    width, height = (
                        int(preprocess["native_size"][0]),
                        int(preprocess["native_size"][1]),
                    )
                    if (width, height) != (
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ):
                        raise ValueError("canonical image dimensions changed")
                    (
                        raw_logits,
                        native_map,
                        peak_bytes,
                        latency_ms,
                    ) = infer_one(
                        model,
                        device,
                        image_array,
                        qtable_array,
                        preprocess,
                    )
                    target = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    localization = {
                        "native": binary_pixel_metrics(
                            native_map,
                            target,
                            args.mask_threshold,
                            include_ap=input_row["kind"] == "forged",
                        )
                    }

                    raw_logits_path = artifact_dir / "raw_logits_quarter" / (
                        f"{sample_id}.npy"
                    )
                    score_map_path = artifact_dir / "score_maps_native" / (
                        f"{sample_id}.npy"
                    )
                    mask_path = artifact_dir / "masks_native" / (
                        f"{sample_id}.png"
                    )
                    _atomic_save_npy(
                        raw_logits_path,
                        raw_logits.astype(np.float32, copy=False),
                    )
                    _atomic_save_npy(
                        score_map_path,
                        native_map.astype(np.float32, copy=False),
                    )
                    _atomic_save_mask(
                        mask_path,
                        native_map >= args.mask_threshold,
                    )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "raw_logits_path": repo_relative(
                            raw_logits_path,
                            repo_root,
                        ),
                        "raw_logits_sha256": sha256_file(raw_logits_path),
                        "raw_logits_shape": list(raw_logits.shape),
                        "score_map_path": repo_relative(
                            score_map_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(score_map_path),
                        "score_map_shape": list(native_map.shape),
                        "mask_path": repo_relative(mask_path, repo_root),
                        "mask_sha256": sha256_file(mask_path),
                        "mask_shape": list(native_map.shape),
                        "mask_threshold": args.mask_threshold,
                        "qtable_sha256": preprocess["qtable_sha256"],
                        "dct_y_sha256": preprocess["dct_y_sha256"],
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
                        f"CAT-Net failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_catnet_results(
        result_rows,
        selected,
        mask_threshold=args.mask_threshold,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": "CAT-Net v2",
            "model_slug": "catnet_v2_ijcv2022",
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
        raise RuntimeError(f"incomplete CAT-Net run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--catnet-root", type=Path, default=DEFAULT_CATNET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/catnet"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
