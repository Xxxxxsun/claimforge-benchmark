#!/usr/bin/env python3
"""Run the official OpenSDI MaskCLIP checkpoint on canonical CLAIMFORGE inputs."""

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
from eval.opensource.maskclip_metrics import binary_pixel_metrics, summarize_results


MODEL_REPO_URL = "https://github.com/iamwangyabin/OpenSDI"
MODEL_SOURCE_COMMIT = "02c93d4891303637cb5d6852d3de63a099d69843"
CHECKPOINT_REPO = "nebula/MaskCLIP-weights"
CHECKPOINT_REVISION = "765f09adbce63ae201dfa451256fbbc419919450"
CHECKPOINT_FILENAME = "MaskCLIP_sd15_20241109_08_53_19.pth"
CHECKPOINT_SHA256 = "481c8bd16077f942efec2901f93c1bc7008f6992402a1ab69fda2652408ca90f"
MAE_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_GIT_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
MODEL_INPUT_SIZE = 512
CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/mouse_canonical_v1/manifest.json")
DEFAULT_OPENSDI_ROOT = Path("/root/.cache/claimforge/third_party/OpenSDI")
DEFAULT_CHECKPOINT = Path(
    f"/root/.cache/claimforge/checkpoints/opensdi/{CHECKPOINT_FILENAME}"
)
DEFAULT_CLIP_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/openai-clip/ViT-L-14.pt"
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
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
            temporary,
            format="PNG",
            optimize=False,
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def preprocess_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    resized = cv2.resize(
        rgb,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    normalized = (resized / 255.0 - CLIP_MEAN) / CLIP_STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    return tensor, (width, height)


def restore_score_map(
    model_map: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    restored = cv2.resize(
        np.asarray(model_map, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def resize_target(
    target: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        np.asarray(target, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _verify_runtime_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def load_model(
    *,
    opensdi_root: Path,
    checkpoint_path: Path,
    clip_checkpoint_path: Path,
    device_name: str,
):
    import torch

    source_commit = _git_value(opensdi_root, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"OpenSDI source commit mismatch: {source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(opensdi_root, "status", "--short", "--untracked-files=no"):
        raise ValueError("OpenSDI tracked source files have local modifications")

    expected_mae = opensdi_root / "weights/mae_pretrain_vit_base.pth"
    _verify_runtime_file(checkpoint_path, CHECKPOINT_SHA256, "MaskCLIP checkpoint")
    _verify_runtime_file(expected_mae, MAE_SHA256, "MAE initialization checkpoint")
    _verify_runtime_file(clip_checkpoint_path, CLIP_SHA256, "OpenAI CLIP checkpoint")
    if clip_checkpoint_path.name != "ViT-L-14.pt":
        raise ValueError("OpenAI CLIP checkpoint must retain the name ViT-L-14.pt")

    if str(opensdi_root) not in sys.path:
        sys.path.insert(0, str(opensdi_root))
    import clip
    from model.MaskCLIP import MaskCLIP

    original_clip_load = clip.load

    def pinned_clip_load(name: str, *args: Any, **kwargs: Any):
        kwargs["download_root"] = str(clip_checkpoint_path.parent)
        return original_clip_load(name, *args, **kwargs)

    with _working_directory(opensdi_root):
        clip.load = pinned_clip_load
        try:
            model = MaskCLIP("ViTL")
        finally:
            clip.load = original_clip_load
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("MaskCLIP checkpoint has no model state dictionary")
        model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()

    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return model, device


class LogitCapture:
    def __init__(self, model: Any):
        self.value: Any = None
        self.handle = model.ce_criterion.register_forward_pre_hook(self._capture)

    def _capture(self, _module: Any, args: tuple[Any, ...]) -> None:
        self.value = args[0].detach()

    def pop(self):
        value = self.value
        self.value = None
        if value is None:
            raise RuntimeError("MaskCLIP classification logits were not captured")
        return value

    def close(self) -> None:
        self.handle.remove()


def infer_one(
    model: Any,
    capture: LogitCapture,
    device: Any,
    tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
    import torch

    image = torch.from_numpy(tensor).unsqueeze(0).to(device)
    dummy_mask = torch.zeros(
        (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        dtype=image.dtype,
        device=device,
    )
    dummy_label = torch.zeros((1,), dtype=torch.long, device=device)
    edge_weight = torch.ones_like(dummy_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        output = model(image, dummy_mask, dummy_label, edge_weight)
        logits = capture.pop()
        probabilities = torch.softmax(logits, dim=1)
        score_map = output["pred_mask"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000

    logits_np = logits[0].float().cpu().numpy()
    probabilities_np = probabilities[0].float().cpu().numpy()
    score_map_np = score_map[0, 0].float().cpu().numpy()
    if logits_np.shape != (2,):
        raise ValueError(f"unexpected MaskCLIP logits shape: {logits_np.shape}")
    if score_map_np.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected MaskCLIP map shape: {score_map_np.shape}")
    if not np.isfinite(logits_np).all() or not np.isfinite(score_map_np).all():
        raise ValueError("MaskCLIP produced non-finite output")
    if float(score_map_np.min()) < 0.0 or float(score_map_np.max()) > 1.0:
        raise ValueError("MaskCLIP score map falls outside [0, 1]")
    return logits_np, probabilities_np, peak_bytes, latency_ms, score_map_np


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
    opensdi_root: Path,
    checkpoint_path: Path,
    clip_checkpoint_path: Path,
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
            "name": "MaskCLIP",
            "model_slug": "opensdi_maskclip_sd15",
            "repo_url": MODEL_REPO_URL,
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_tracked_clean": not bool(
                _git_value(
                    opensdi_root,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "model_setting_name": "ViTL",
            "checkpoint": {
                "repository": CHECKPOINT_REPO,
                "revision": CHECKPOINT_REVISION,
                "filename": checkpoint_path.name,
                "sha256": CHECKPOINT_SHA256,
                "epoch": 13,
            },
            "mae_initialization_sha256": MAE_SHA256,
            "clip": {
                "git_commit": CLIP_GIT_COMMIT,
                "filename": clip_checkpoint_path.name,
                "sha256": CLIP_SHA256,
            },
            "class_names": ["real", "forged"],
            "positive_class_index": 1,
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "model_input_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "input_resize": "opencv_inter_linear_stretch",
            "normalization_mean": CLIP_MEAN.tolist(),
            "normalization_std": CLIP_STD.tolist(),
            "score_semantics": "softmax_probability_of_class_1_forged",
            "score_margin_semantics": "logit_forged_minus_logit_real",
            "map_semantics": "sigmoid_forged_probability",
            "map_restore": "opencv_inter_linear_to_native_size",
            "classification_threshold": args.classification_threshold,
            "mask_threshold": args.mask_threshold,
        },
        "expected_pairs": len({int(row["pair_rank"]) for row in selected}),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": {
            "path": repo_relative(Path(__file__), repo_root),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": _package_version("Pillow"),
        "imdlbenco": _package_version("IMDLBenCo"),
        "timm": _package_version("timm"),
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

    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    opensdi_root = args.opensdi_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    clip_checkpoint_path = args.clip_checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/maskclip/{args.run_id}"),
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
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    run_manifest = build_run_manifest(
        args=args,
        repo_root=repo_root,
        dataset_manifest_path=dataset_manifest_path,
        release=release,
        inputs_path=inputs_path,
        selected=selected,
        opensdi_root=opensdi_root,
        checkpoint_path=checkpoint_path,
        clip_checkpoint_path=clip_checkpoint_path,
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
        f"MaskCLIP run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    capture = None
    if pending:
        model, device = load_model(
            opensdi_root=opensdi_root,
            checkpoint_path=checkpoint_path,
            clip_checkpoint_path=clip_checkpoint_path,
            device_name=args.device,
        )
        capture = LogitCapture(model)
        print(f"loaded MaskCLIP on {device}", flush=True)
        try:
            for index, input_row in enumerate(pending, start=1):
                sample_id = str(input_row["sample_id"])
                identity = {
                    "schema_version": "opensource_result_v1",
                    "run_id": args.run_id,
                    "input_manifest_sha256": release["inputs_sha256"],
                    "id": sample_id,
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
                    "model": "MaskCLIP",
                    "model_slug": "opensdi_maskclip_sd15",
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    tensor, (width, height) = preprocess_image(image_path)
                    if (width, height) != (
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ):
                        raise ValueError("canonical image dimensions changed")
                    (
                        logits,
                        probabilities,
                        peak_bytes,
                        latency_ms,
                        model_map,
                    ) = infer_one(model, capture, device, tensor)
                    native_map = restore_score_map(model_map, width, height)
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
                        "model_512": binary_pixel_metrics(
                            model_map,
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_pixel_metrics(
                            native_map,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    model_map_path = artifact_dir / "score_maps_model_512" / (
                        f"{sample_id}.npy"
                    )
                    native_map_path = artifact_dir / "score_maps_native" / (
                        f"{sample_id}.npy"
                    )
                    binary_mask_path = artifact_dir / "masks_native" / (
                        f"{sample_id}.png"
                    )
                    _atomic_save_npy(model_map_path, model_map.astype(np.float32))
                    _atomic_save_npy(native_map_path, native_map)
                    _atomic_save_mask(
                        binary_mask_path,
                        native_map >= args.mask_threshold,
                    )

                    score = float(probabilities[1])
                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": score,
                        "score_margin": float(logits[1] - logits[0]),
                        "class_logits": {
                            "real": float(logits[0]),
                            "forged": float(logits[1]),
                        },
                        "class_probabilities": {
                            "real": float(probabilities[0]),
                            "forged": score,
                        },
                        "decision": bool(
                            score >= args.classification_threshold
                        ),
                        "classification_threshold": args.classification_threshold,
                        "score_map_model_path": repo_relative(
                            model_map_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(model_map_path),
                        "score_map_native_path": repo_relative(
                            native_map_path,
                            repo_root,
                        ),
                        "score_map_native_sha256": sha256_file(native_map_path),
                        "mask_path": repo_relative(binary_mask_path, repo_root),
                        "mask_sha256": sha256_file(binary_mask_path),
                        "mask_threshold": args.mask_threshold,
                        "localization": localization,
                        "preprocess": {
                            "input_size": [
                                MODEL_INPUT_SIZE,
                                MODEL_INPUT_SIZE,
                            ],
                            "resize": "opencv_inter_linear_stretch",
                            "restore": "opencv_inter_linear",
                        },
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
                print(
                    f"[{index}/{len(pending)}] "
                    f"{input_row['task_id']} {input_row['kind']}: "
                    f"{row['status']}"
                    + (
                        f" score={row['score']:.6f} "
                        f"latency={row['latency_ms']:.1f}ms"
                        if row["status"] == "ok"
                        else f" {row['error_type']}: {row['error_message']}"
                    ),
                    flush=True,
                )
                if row["status"] != "ok" and args.fail_fast:
                    raise RuntimeError(
                        f"MaskCLIP failed for {sample_id}: {row['error_message']}"
                    )
        finally:
            if capture is not None:
                capture.close()
            del capture, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_results(
        result_rows,
        selected,
        classification_threshold=args.classification_threshold,
        mask_threshold=args.mask_threshold,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": "MaskCLIP",
            "model_slug": "opensdi_maskclip_sd15",
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
        raise RuntimeError(f"incomplete MaskCLIP run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--opensdi-root", type=Path, default=DEFAULT_OPENSDI_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=DEFAULT_CLIP_CHECKPOINT,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/maskclip"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
