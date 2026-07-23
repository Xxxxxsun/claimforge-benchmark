#!/usr/bin/env python3
"""Run the official TruFor checkpoint on canonical CLAIMFORGE inputs."""

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
from eval.opensource.maskclip_metrics import binary_pixel_metrics
from eval.opensource.trufor_metrics import summarize_trufor_results


MODEL_REPO_URL = "https://github.com/grip-unina/TruFor"
MODEL_SOURCE_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
MODEL_LICENSE = "informational_and_nonprofit_only"
MODEL_LICENSE_SHA256 = (
    "07201e07e3d2c1ac55480037a87734fcccacbb0cd0e25a31e3b89ac7ffadf8b4"
)
MODEL_CONFIG_SHA256 = (
    "a87108eb0df40d9bab6a303eb91419564b7c106d5105bbd5d8ecaec1567b5b8b"
)
CHECKPOINT_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
CHECKPOINT_ZIP_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
CHECKPOINT_ZIP_SHA256 = (
    "953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8"
)
CHECKPOINT_SHA256 = (
    "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
)
CHECKPOINT_EPOCH = 81

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/mouse_canonical_v1/manifest.json")
DEFAULT_TRUFOR_ROOT = Path(
    "/root/.cache/claimforge/third_party/TruFor/TruFor_train_test"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/trufor/weights/trufor.pth.tar"
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
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(
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
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1),
        dtype=np.float32,
    )
    tensor /= 256.0
    return tensor, (width, height)


def postprocess_outputs(
    pred: Any,
    confidence: Any,
    detection: Any,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    import torch

    if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 2:
        raise ValueError(f"unexpected TruFor localization shape: {tuple(pred.shape)}")
    if (
        confidence is None
        or confidence.ndim != 4
        or confidence.shape[0] != 1
        or confidence.shape[1] != 1
    ):
        raise ValueError("unexpected TruFor reliability output")
    if detection is None or detection.numel() != 1:
        raise ValueError("unexpected TruFor detection output")
    score_map = torch.softmax(pred, dim=1)[0, 1].float().cpu().numpy()
    reliability = torch.sigmoid(confidence)[0, 0].float().cpu().numpy()
    detection_logit = float(detection.reshape(-1)[0].float().cpu())
    score = float(torch.sigmoid(detection.reshape(-1)[0]).float().cpu())
    for name, value in (("score map", score_map), ("reliability", reliability)):
        if not np.isfinite(value).all():
            raise ValueError(f"TruFor {name} contains non-finite values")
        if float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise ValueError(f"TruFor {name} falls outside [0, 1]")
    if not np.isfinite(score) or not np.isfinite(detection_logit):
        raise ValueError("TruFor produced a non-finite image score")
    return score, detection_logit, score_map, reliability


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
    trufor_root: Path,
    checkpoint_path: Path,
    device_name: str,
):
    import torch

    source_repo = trufor_root.parent
    source_commit = _git_value(source_repo, "rev-parse", "HEAD")
    if source_commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"TruFor source commit mismatch: {source_commit} != {MODEL_SOURCE_COMMIT}"
        )
    if _git_value(source_repo, "status", "--short", "--untracked-files=no"):
        raise ValueError("TruFor tracked source files have local modifications")
    _verify_runtime_file(
        trufor_root / "LICENSE.txt",
        MODEL_LICENSE_SHA256,
        "TruFor license file",
    )
    _verify_runtime_file(
        trufor_root / "lib/config/trufor_ph3.yaml",
        MODEL_CONFIG_SHA256,
        "TruFor phase-3 configuration",
    )
    _verify_runtime_file(checkpoint_path, CHECKPOINT_SHA256, "TruFor checkpoint")

    if str(trufor_root) not in sys.path:
        sys.path.insert(0, str(trufor_root))
    with _working_directory(trufor_root):
        from lib.config import config as base_config
        from lib.utils import get_model

        config = base_config.clone()
        config.defrost()
        config.merge_from_file(str(trufor_root / "lib/config/trufor_ph3.yaml"))
        config.freeze()
        model = get_model(config)
        safe_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("TruFor checkpoint has no state_dict")
        if int(checkpoint.get("epoch", -1)) != CHECKPOINT_EPOCH:
            raise ValueError("TruFor checkpoint epoch does not match pinned release")
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
    tensor: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, int, float]:
    import torch

    image = torch.from_numpy(tensor).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        pred, confidence, detection, _ = model(image, save_np=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_bytes = 0
    latency_ms = (time.monotonic() - started) * 1000
    score, logit, score_map, reliability = postprocess_outputs(
        pred,
        confidence,
        detection,
    )
    return score, logit, score_map, reliability, peak_bytes, latency_ms


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
    trufor_root: Path,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    import torch

    ordered_inputs = _selection_contract(selected)
    dependency_paths = [
        Path(__file__),
        Path(__file__).with_name("trufor_metrics.py"),
        Path(__file__).with_name("maskclip_metrics.py"),
        Path(__file__).with_name("common.py"),
    ]
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
            "name": "TruFor",
            "model_slug": "trufor_cvpr2023",
            "repo_url": MODEL_REPO_URL,
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_tracked_clean": not bool(
                _git_value(
                    trufor_root.parent,
                    "status",
                    "--short",
                    "--untracked-files=no",
                )
            ),
            "license": {
                "summary": MODEL_LICENSE,
                "file": "TruFor_train_test/LICENSE.txt",
                "sha256": MODEL_LICENSE_SHA256,
            },
            "configuration": {
                "name": "trufor_ph3",
                "sha256": MODEL_CONFIG_SHA256,
            },
            "checkpoint": {
                "url": CHECKPOINT_URL,
                "zip_md5": CHECKPOINT_ZIP_MD5,
                "zip_sha256": CHECKPOINT_ZIP_SHA256,
                "filename": checkpoint_path.name,
                "sha256": CHECKPOINT_SHA256,
                "epoch": CHECKPOINT_EPOCH,
                "state_dict_keys": 952,
                "strict_load": True,
                "safe_weights_only_load": True,
            },
            "construction_pretrained_files_used": False,
            "positive_class_index": 1,
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "input_decode": "PIL_convert_RGB",
            "input_scale": "float32_divide_by_256",
            "input_resize": "none",
            "network_map_upsample": "bilinear_align_corners_false_to_input",
            "map_restore": "none_already_native",
            "score_semantics": "sigmoid_binary_logit_probability_of_forged",
            "score_margin_semantics": "binary_forged_log_odds",
            "map_semantics": "softmax_channel_1_forged_probability",
            "reliability_semantics": (
                "sigmoid_tcp_localization_reliability_not_forged_probability"
            ),
            "classification_threshold": args.classification_threshold,
            "mask_threshold": args.mask_threshold,
            "mask_threshold_comparison": "greater_than_or_equal",
            "cudnn": {
                "benchmark": False,
                "deterministic": False,
                "enabled": False,
                "source": "official_trufor_ph3_config",
            },
        },
        "expected_pairs": len({int(row["pair_rank"]) for row in selected}),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": [
            {
                "path": repo_relative(path, repo_root),
                "sha256": sha256_file(path),
            }
            for path in dependency_paths
        ],
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": np.__version__,
        "pillow": _package_version("Pillow"),
        "scikit_learn": _package_version("scikit-learn"),
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
    trufor_root = args.trufor_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(f"outputs/opensource/trufor/{args.run_id}"),
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = False
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    run_manifest = build_run_manifest(
        args=args,
        repo_root=repo_root,
        dataset_manifest_path=dataset_manifest_path,
        release=release,
        inputs_path=inputs_path,
        selected=selected,
        trufor_root=trufor_root,
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
        f"TruFor run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    if pending:
        model, device = load_model(
            trufor_root=trufor_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(f"loaded TruFor on {device}", flush=True)
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
                    "model": "TruFor",
                    "model_slug": "trufor_cvpr2023",
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
                        score,
                        detection_logit,
                        score_map,
                        reliability_map,
                        peak_bytes,
                        latency_ms,
                    ) = infer_one(model, device, tensor)
                    if score_map.shape != (height, width):
                        raise ValueError(
                            "TruFor score map is not in native image space"
                        )
                    if reliability_map.shape != (height, width):
                        raise ValueError(
                            "TruFor reliability map is not in native image space"
                        )
                    target = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    localization = {
                        "native": binary_pixel_metrics(
                            score_map,
                            target,
                            args.mask_threshold,
                            include_ap=input_row["kind"] == "forged",
                        )
                    }

                    score_map_path = artifact_dir / "score_maps_native" / (
                        f"{sample_id}.npy"
                    )
                    reliability_path = artifact_dir / "reliability_maps_native" / (
                        f"{sample_id}.npy"
                    )
                    binary_mask_path = artifact_dir / "masks_native" / (
                        f"{sample_id}.png"
                    )
                    _atomic_save_npy(score_map_path, score_map.astype(np.float32))
                    _atomic_save_npy(
                        reliability_path,
                        reliability_map.astype(np.float32),
                    )
                    _atomic_save_mask(
                        binary_mask_path,
                        score_map >= args.mask_threshold,
                    )
                    reliability_stats = {
                        "semantics": (
                            "localization_reliability_not_forged_probability"
                        ),
                        "used_for_primary_metrics": False,
                        "min": float(np.min(reliability_map)),
                        "mean": float(np.mean(reliability_map)),
                        "median": float(np.median(reliability_map)),
                        "p05": float(np.quantile(reliability_map, 0.05)),
                        "p95": float(np.quantile(reliability_map, 0.95)),
                        "max": float(np.max(reliability_map)),
                    }

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": score,
                        "score_margin": detection_logit,
                        "raw_outputs": {
                            "binary_forged_logit": detection_logit,
                        },
                        "class_probabilities": {
                            "real": 1.0 - score,
                            "forged": score,
                        },
                        "decision": bool(
                            score >= args.classification_threshold
                        ),
                        "classification_threshold": args.classification_threshold,
                        "score_map_native_path": repo_relative(
                            score_map_path,
                            repo_root,
                        ),
                        "score_map_native_sha256": sha256_file(score_map_path),
                        "reliability_map_native_path": repo_relative(
                            reliability_path,
                            repo_root,
                        ),
                        "reliability_map_native_sha256": sha256_file(
                            reliability_path
                        ),
                        "mask_path": repo_relative(binary_mask_path, repo_root),
                        "mask_sha256": sha256_file(binary_mask_path),
                        "mask_threshold": args.mask_threshold,
                        "localization": localization,
                        "reliability": reliability_stats,
                        "preprocess": {
                            "decode": "PIL_convert_RGB",
                            "input_scale": "float32_divide_by_256",
                            "resize": "none",
                            "network_upsample": (
                                "bilinear_align_corners_false_to_native"
                            ),
                            "restore": "none",
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
                        f"TruFor failed for {sample_id}: {row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_trufor_results(
        result_rows,
        selected,
        classification_threshold=args.classification_threshold,
        mask_threshold=args.mask_threshold,
    )
    summary.update(
        {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": "TruFor",
            "model_slug": "trufor_cvpr2023",
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
        raise RuntimeError(f"incomplete TruFor run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--trufor-root", type=Path, default=DEFAULT_TRUFOR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/trufor"),
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
