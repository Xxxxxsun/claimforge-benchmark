#!/usr/bin/env python3
"""Run the official NPR ProGAN-4class whole-image detector.

This adapter freezes the repository's AIGCDetectBenchmark preprocessing with
the official ``test.py`` evaluation mode before looking at full CLAIMFORGE
scores. The Hugging Face Space corroborates the checkpoint and native-size
preprocessing, but is not treated as an executable reference because it omits
``model.eval()``. NPR is an image-level classifier (T1) and has no native
localization output (T2).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    read_latest_by_id,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.npr_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    summarize_npr_raw_logit_diagnostic,
    summarize_npr_results,
)


MODEL_NAME = "NPR"
MODEL_SLUG = "npr_aigcdetect_progan4class"
MODEL_ARCH = "NPR truncated ResNet-50 (stem + layer1 + layer2)"
MODEL_REPO_URL = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"
MODEL_SOURCE_COMMIT = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"
PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2024/html/"
    "Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_"
    "Network_for_Generalizable_CVPR_2024_paper.html"
)
HF_SPACE_URL = (
    "https://huggingface.co/spaces/tancc/"
    "Generalizable_Deepfake_Detection-NPR-CVPR2024"
)
HF_SPACE_COMMIT = "522a9f1020f7454d486f28a0d5c148ec37919b32"
HF_SOURCE_FILES = {
    "app.py": "06da679323935bb7f6c8387f18d2d9ce58b488d33d0d1e67286c6ab8d8a7b35a",
    "README.md": (
        "d73e2354f53c45238a831ed18cecede4e9c3d4da13b9cfd57baf327a502430df"
    ),
    "requirements.txt": (
        "f32a2c183d5e1974b0447ff9262cdb289020dd3c1118854e8a3a95cb3f0ba66c"
    ),
}

PREPROCESS_PROFILE = "official_aigcdetect_native_even_trim"
MODEL_SEED = 100
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
FEATURE_DIMENSION = 512
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

LICENSE_RECORD = {
    "github_code": {
        "license": "not_stated",
        "osi_open_source_license_established": False,
    },
    "github_checkpoint": {
        "license": "not_stated",
        "commercial_clearance_established": False,
    },
    "hf_space_metadata": {
        "license": "Apache-2.0",
        "scope": "space_repository_only_not_upstream_github_checkpoint",
    },
    "overall_commercial_clearance": "not_established",
}

CHECKPOINT = {
    "id": "NPR-AIGC-ProGAN4class@68338a",
    "filename": "model_epoch_last_3090.pth",
    "repo_relative_path": "model_epoch_last_3090.pth",
    "introduced_commit": "68338a07847e891534f3d0b0a0e25bb137b684f7",
    "bytes": 5_842_385,
    "sha256": "b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a",
    "state_entries": 146,
    "state_elements": 1_447_897,
    "trainable_parameters": 1_437_761,
    "format": "torch_ordered_state_dict_weights_only",
    "official_role": (
        "AIGCDetectBenchmark ProGAN-4class checkpoint; also downloaded by "
        "the HF demo"
    ),
    "pinned_url": (
        "https://raw.githubusercontent.com/chuangchuangtan/"
        "NPR-DeepfakeDetection/68338a07847e891534f3d0b0a0e25bb137b684f7/"
        "model_epoch_last_3090.pth"
    ),
}

EXCLUDED_RELEASE_ASSETS = {
    "NPR.pth": {
        "sha256": (
            "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53"
        ),
        "bytes": 17_393_733,
        "reason": (
            "paper-era training snapshot with nested model/optimizer/step "
            "payload and module-prefixed keys; incompatible with current "
            "test.py and not the AIGCDetectBenchmark/HF checkpoint"
        ),
    },
    "NPR_GenImage_sdv4.pth": {
        "sha256": (
            "9bc961e7d643581aa0ea879cbd322dcc2e543877568a43d2f6cdb92906379015"
        ),
        "bytes": 5_842_385,
        "reason": "separately trained SDv1.4/GenImage checkpoint",
    },
}

SOURCE_FILES = {
    "README.md": "65d4d9806fea6ab49ecaa6d0bd20e32380348d4858c9a6305d09387b64644871",
    "test.py": "fbe86617998638b325be8d12eac15e903817ebc2a54784e40266cf7e1f788e79",
    "validate.py": "dca8aa2aed02d6630ba4d58feab69e2f80e4f76e803d5365bf35b2b9ca776be3",
    "data/datasets.py": (
        "6b82e0bfc5251ba12e94b10bd9a9b8bcb9405472dbd3b10773035dfd5907717f"
    ),
    "options/base_options.py": (
        "e967e19807ab44ecefaeddf7e75022dfe0f470a8bd3326fc806eca0099139e91"
    ),
    "options/test_options.py": (
        "d0a6e520c9d4a9b034ba237f97d594a0284872879042ba5a7325b601adced10e"
    ),
    "networks/resnet.py": (
        "c7663a02a322dc2a68625535b367e8800df42bacca3d0526cafef8c32168c67e"
    ),
}

DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/NPR-DeepfakeDetection-781ced3f"
)
DEFAULT_HF_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/NPR-HF-Space-522a9f10"
)
DEFAULT_CHECKPOINT = DEFAULT_SOURCE_ROOT / str(CHECKPOINT["filename"])
DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/npr")
DEFAULT_RUN_ID = (
    "npr_aigcdetect_progan4class_mouse_canonical_v1_full275_20260725"
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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return _array_sha256(tensor.numpy())


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def adapter_contract(repo_root: Path) -> dict[str, Any]:
    relative_paths = (
        "eval/opensource/run_npr.py",
        "eval/opensource/npr_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
    )
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing NPR adapter component: {path}")
        result[relative] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _verify_runtime_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
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
    _verify_runtime_file(
        inputs_path,
        str(release.get("inputs_sha256")),
        "canonical inputs.jsonl",
    )
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
    if pair_limit is not None and sample_id is not None:
        raise ValueError("pair-limit and sample-id are mutually exclusive")
    if sample_id is not None:
        selected = [
            row for row in rows if str(row.get("sample_id")) == sample_id
        ]
        if len(selected) != 1:
            raise ValueError(
                f"sample-id must select exactly one row: {sample_id}"
            )
        return selected
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair-limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [
        row for row in rows if int(row["pair_rank"]) in selected_ranks
    ]
    kinds: dict[int, set[str]] = {}
    for row in selected:
        kinds.setdefault(int(row["pair_rank"]), set()).add(str(row["kind"]))
    invalid = {
        rank: values
        for rank, values in kinds.items()
        if values != {"real", "forged"}
    }
    if invalid:
        raise ValueError(f"canonical selection contains incomplete pairs: {invalid}")
    return selected


def _load_gt_mask(row: Mapping[str, Any], repo_root: Path) -> np.ndarray | None:
    sample_id = str(row.get("sample_id"))
    width, height = int(row["width"]), int(row["height"])
    if row.get("kind") == "real":
        if (
            row.get("label") != 0
            or row.get("gt_mask_kind") != "all_zero"
            or row.get("gt_mask_path") is not None
            or row.get("gt_mask_sha256") is not None
            or int(row.get("gt_positive_pixels", -1)) != 0
        ):
            raise ValueError(f"invalid real GT contract for {sample_id}")
        return None
    if (
        row.get("kind") != "forged"
        or row.get("label") != 1
        or row.get("gt_mask_kind") != "exact_diff"
    ):
        raise ValueError(f"invalid forged GT contract for {sample_id}")
    path_value = row.get("gt_mask_path")
    digest = row.get("gt_mask_sha256")
    if not isinstance(path_value, str) or not _valid_sha256(digest):
        raise ValueError(f"invalid forged GT path/hash for {sample_id}")
    path = _anchored(Path(path_value), repo_root)
    _verify_runtime_file(path, str(digest), f"GT mask {sample_id}")
    with Image.open(path) as opened:
        pixels = np.asarray(opened)
    if pixels.ndim != 2 or pixels.shape != (height, width):
        raise ValueError(f"GT shape mismatch for {sample_id}: {pixels.shape}")
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"GT mask {sample_id} is not binary 0/255")
    positive = int(np.count_nonzero(pixels == 255))
    if positive <= 0 or positive != int(row.get("gt_positive_pixels", -1)):
        raise ValueError(f"GT positive-pixel count mismatch for {sample_id}")
    return np.asarray(pixels, dtype=np.uint8)


def validate_selected_inputs(
    selected: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    for row in selected:
        sample_id = str(row["sample_id"])
        path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            path,
            str(row["canonical_sha256"]),
            f"canonical input {sample_id}",
        )
        with Image.open(path) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"canonical dimensions changed for {sample_id}")
        _load_gt_mask(row, repo_root)


def effective_native_size(width: int, height: int) -> tuple[int, int]:
    if width <= 1 or height <= 1:
        raise ValueError("NPR input dimensions must exceed one pixel")
    return width - (width % 2), height - (height % 2)


def build_pair_visibility(
    rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair_rows in by_task.items():
        forged = [row for row in pair_rows if row.get("kind") == "forged"]
        if len(forged) != 1:
            raise ValueError(f"task {task_id} does not have one forged row")
        row = forged[0]
        mask = _load_gt_mask(row, repo_root)
        assert mask is not None
        width, height = int(row["width"]), int(row["height"])
        effective_width, effective_height = effective_native_size(width, height)
        total = int(np.count_nonzero(mask == 255))
        visible = int(
            np.count_nonzero(
                mask[:effective_height, :effective_width] == 255
            )
        )
        fraction = visible / total
        category = (
            "none"
            if visible == 0
            else "full"
            if visible == total
            else "partial"
        )
        result[task_id] = {
            "edit_visibility": category,
            "edit_visible_gt_fraction": fraction,
            "edit_visible_gt_pixels": visible,
            "edit_total_gt_pixels": total,
            "effective_native_xyxy": [
                0,
                0,
                effective_width,
                effective_height,
            ],
            "trim_bottom": height - effective_height,
            "trim_right": width - effective_width,
            "evidence": "exact_diff_mask_intersection_with_native_even_trim",
        }
    return result


def _verify_source_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing NPR source-root: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"NPR source commit mismatch: {commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"NPR tracked source tree is dirty: {dirty[:1000]}")
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"NPR source {relative}",
        )
    bundled = source_root / str(CHECKPOINT["repo_relative_path"])
    _verify_runtime_file(
        bundled,
        str(CHECKPOINT["sha256"]),
        "repository-bundled NPR AIGC checkpoint",
    )
    if bundled.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("repository-bundled NPR checkpoint size changed")
    history = _git_value(
        source_root,
        "log",
        "--format=%H",
        "--",
        str(CHECKPOINT["repo_relative_path"]),
    )
    if history != CHECKPOINT["introduced_commit"]:
        raise ValueError("NPR checkpoint introduction history changed")
    license_names = ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE")
    present_licenses = [
        name for name in license_names if (source_root / name).exists()
    ]
    if present_licenses:
        raise ValueError(
            "frozen NPR no-license finding changed: "
            f"found {present_licenses}"
        )
    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str((source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in SOURCE_FILES.items()
        },
        "checkpoint_history": {
            "path": str(bundled.resolve()),
            "introduced_commit": history,
        },
        "root_license_files": present_licenses,
        "license_record": LICENSE_RECORD,
    }


def _verify_hf_source_contract(hf_source_root: Path) -> dict[str, Any]:
    if not hf_source_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned NPR Hugging Face Space source: {hf_source_root}"
        )
    commit = _git_value(hf_source_root, "rev-parse", "HEAD")
    if commit != HF_SPACE_COMMIT:
        raise ValueError(
            "NPR Hugging Face Space commit mismatch: "
            f"{commit} != {HF_SPACE_COMMIT}"
        )
    dirty = _git_value(
        hf_source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(
            f"NPR Hugging Face tracked source tree is dirty: {dirty[:1000]}"
        )
    for relative, digest in HF_SOURCE_FILES.items():
        _verify_runtime_file(
            hf_source_root / relative,
            digest,
            f"NPR Hugging Face source {relative}",
        )
    app_text = (hf_source_root / "app.py").read_text(encoding="utf-8")
    required_evidence = (
        "model_epoch_last_3090.pth",
        "transforms.ToTensor()",
        "transforms.Normalize(mean=[0.485, 0.456, 0.406]",
        "if w%2 == 1: img = img[:, :, :-1,:  ]",
        "if h%2 == 1: img = img[:, :, :  ,:-1]",
        "NPR = img - interpolate(img, 0.5)",
        "x.sigmoid()",
    )
    missing = [
        evidence for evidence in required_evidence if evidence not in app_text
    ]
    if missing:
        raise ValueError(
            "pinned NPR Hugging Face app contract changed: "
            f"missing {missing}"
        )
    if "NPRmodel.eval()" in app_text:
        raise ValueError(
            "frozen finding changed: NPR Hugging Face app now calls eval()"
        )
    return {
        "space_url": HF_SPACE_URL,
        "root": str(hf_source_root.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str((hf_source_root / relative).resolve()),
                "sha256": digest,
            }
            for relative, digest in HF_SOURCE_FILES.items()
        },
        "role": (
            "corroborating checkpoint and native-size preprocessing only; "
            "official GitHub test.py eval-mode remains the inference contract"
        ),
        "deployment_mode_defect": {
            "calls_model_eval": False,
            "impact": (
                "BatchNorm remains in train mode with batch-one statistics; "
                "the output is batch/input-composition dependent, mutates "
                "running buffers, and materially differs from checkpoint "
                "eval semantics; it is not reproduced or used as a "
                "sensitivity condition"
            ),
        },
        "supply_chain_note": (
            "original app downloads a mutable GitHub-main pickle checkpoint; "
            "this adapter instead hashes and weights-only loads the pinned "
            "GitHub checkpoint"
        ),
    }


def _import_official_resnet(source_root: Path) -> Any:
    path = source_root / "networks" / "resnet.py"
    module_name = f"_claimforge_npr_resnet_{MODEL_SOURCE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official NPR resnet from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_schema(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    items: list[dict[str, Any]] = []
    elements = 0
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("NPR checkpoint must map string keys to tensors")
        if value.is_complex():
            raise ValueError(f"NPR checkpoint tensor {key} is complex")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"NPR checkpoint tensor {key} is not finite")
        elements += int(value.numel())
        items.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    return {
        "container": f"{type(state).__module__}.{type(state).__name__}",
        "entries": len(items),
        "elements": elements,
        "items_sha256": hashlib.sha256(
            stable_json(items).encode("utf-8")
        ).hexdigest(),
    }


def verify_assets(
    *,
    source_root: Path,
    checkpoint_path: Path,
    hf_source_root: Path = DEFAULT_HF_SOURCE_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], OrderedDict[str, Any], Any]:
    import torch

    source_audit = _verify_source_contract(source_root)
    hf_source_audit = _verify_hf_source_contract(hf_source_root)
    checkpoint_path = checkpoint_path.resolve()
    _verify_runtime_file(
        checkpoint_path,
        str(CHECKPOINT["sha256"]),
        "explicit NPR AIGC checkpoint",
    )
    if checkpoint_path.stat().st_size != int(CHECKPOINT["bytes"]):
        raise ValueError("explicit NPR checkpoint size changed")
    unsafe = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    )
    if unsafe:
        raise ValueError(f"NPR checkpoint contains unsafe globals: {unsafe}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, OrderedDict):
        raise ValueError(
            "NPR AIGC checkpoint is not the frozen flat OrderedDict schema"
        )
    state = OrderedDict(payload)
    schema = _checkpoint_schema(state)
    if schema["entries"] != int(CHECKPOINT["state_entries"]):
        raise ValueError("NPR checkpoint state-entry count changed")
    if schema["elements"] != int(CHECKPOINT["state_elements"]):
        raise ValueError("NPR checkpoint state-element count changed")
    module = _import_official_resnet(source_root)
    reference_model = module.resnet50(num_classes=1)
    reference_keys = list(reference_model.state_dict())
    if list(state) != reference_keys:
        raise ValueError("NPR checkpoint key order/schema does not match model")
    reference_model.load_state_dict(state, strict=True)
    parameters = sum(
        int(parameter.numel()) for parameter in reference_model.parameters()
    )
    if parameters != int(CHECKPOINT["trainable_parameters"]):
        raise ValueError("NPR official model parameter count changed")
    asset_audit = {
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path),
            "actual_bytes": checkpoint_path.stat().st_size,
            "actual_sha256": sha256_file(checkpoint_path),
            "serialization_safety": {
                "unsafe_globals": unsafe,
                "weights_only": True,
                "map_location": "cpu",
            },
            "schema": schema,
        },
        "excluded_release_assets": EXCLUDED_RELEASE_ASSETS,
        "bundle_sha256": hashlib.sha256(
            stable_json(
                {
                    "source_commit": MODEL_SOURCE_COMMIT,
                    "source_files": SOURCE_FILES,
                    "hf_space_commit": HF_SPACE_COMMIT,
                    "hf_source_files": HF_SOURCE_FILES,
                    "checkpoint_sha256": CHECKPOINT["sha256"],
                    "checkpoint_schema": schema,
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
    source_audit["hf_space"] = hf_source_audit
    return source_audit, asset_audit, state, module


def configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace_config != ":4096:8":
        raise ValueError(
            "NPR deterministic CUDA contract requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )

    import torch

    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        if int(device.index) >= torch.cuda.device_count():
            raise ValueError(f"invalid CUDA device {device}")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("NPR device must be cpu or cuda")

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    evidence: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "pillow": _package_version("Pillow"),
        "numpy": np.__version__,
        "scikit_learn": _package_version("scikit-learn"),
        "device": str(device),
        "seed": MODEL_SEED,
        "inference_dtype": "torch.float32",
        "autocast": False,
        "grad_enabled": False,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        evidence["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    return device, evidence


def load_model(
    *,
    module: Any,
    state: OrderedDict[str, Any],
    device: Any,
) -> Any:
    model = module.resnet50(num_classes=1)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=__import__("torch").float32)
    model.eval()
    return model


def preprocess_image(path: Path) -> tuple[Any, dict[str, Any]]:
    """Reproduce the official native-size AIGC/HF preprocessing."""

    import torch
    import torch.nn.functional as functional
    from torchvision.transforms import functional as vision_functional

    with Image.open(path) as opened:
        rgb_image = opened.convert("RGB")
        width, height = rgb_image.size
        decoded_rgb = np.ascontiguousarray(
            np.asarray(rgb_image, dtype=np.uint8)
        )
        tensor = vision_functional.to_tensor(rgb_image)
    tensor = vision_functional.normalize(tensor, IMAGE_MEAN, IMAGE_STD)
    if tensor.dtype != torch.float32:
        raise ValueError("official NPR preprocessing did not produce float32")
    effective_width, effective_height = effective_native_size(width, height)
    trimmed = tensor[:, :effective_height, :effective_width].contiguous()
    down = functional.interpolate(
        trimmed.unsqueeze(0),
        scale_factor=0.5,
        mode="nearest",
        recompute_scale_factor=True,
    )
    reconstructed = functional.interpolate(
        down,
        scale_factor=2.0,
        mode="nearest",
        recompute_scale_factor=True,
    )
    residual = (trimmed.unsqueeze(0) - reconstructed).squeeze(0).contiguous()
    residual_float = residual.to(dtype=torch.float64)
    residual_stats = {
        "minimum": float(residual.min().item()),
        "maximum": float(residual.max().item()),
        "mean": float(residual_float.mean().item()),
        "mean_absolute": float(residual_float.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(residual_float).item()),
        "nonzero_elements": int(torch.count_nonzero(residual).item()),
        "elements": int(residual.numel()),
    }
    audit = {
        "profile": PREPROCESS_PROFILE,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.functional.to_tensor",
            "torchvision.transforms.functional.normalize_ImageNet",
            "trim_last_row_if_height_odd",
            "trim_last_column_if_width_odd",
        ],
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(decoded_rgb.shape),
        "decoded_rgb_dtype": str(decoded_rgb.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded_rgb),
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "tensor_shape": list(trimmed.shape),
        "tensor_dtype": str(trimmed.numpy().dtype),
        "tensor_sha256": _tensor_sha256(trimmed),
        "npr_residual_shape": list(residual.shape),
        "npr_residual_dtype": str(residual.numpy().dtype),
        "npr_residual_sha256": _tensor_sha256(residual),
        "npr_residual_stats": residual_stats,
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }
    return trimmed, audit


def _infer_one(
    *,
    model: Any,
    tensor: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch
    import torch.nn.functional as functional

    captured: list[Any] = []

    def capture_fc_input(_module: Any, arguments: tuple[Any, ...]) -> None:
        if len(arguments) != 1:
            raise RuntimeError("NPR fc1 hook received unexpected arguments")
        captured.append(arguments[0].detach())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    hook = model.fc1.register_forward_pre_hook(capture_fc_input)
    try:
        with torch.inference_mode():
            output = model(
                tensor.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
            )
    finally:
        hook.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if list(output.shape) != [1, 1]:
        raise ValueError(f"unexpected NPR output shape {list(output.shape)}")
    if len(captured) != 1 or list(captured[0].shape) != [1, FEATURE_DIMENSION]:
        raise ValueError("NPR fc1 feature hook did not fire exactly once")
    feature_device = captured[0]
    with torch.inference_mode():
        replay_output = functional.linear(
            feature_device,
            model.fc1.weight,
            model.fc1.bias,
        )
        probability_tensor = torch.sigmoid(output)
        replay_probability = torch.sigmoid(replay_output)
    raw_logit = float(output.reshape(()).item())
    replay_logit = float(replay_output.reshape(()).item())
    probability = float(probability_tensor.reshape(()).item())
    replay_score = float(replay_probability.reshape(()).item())
    if not math.isfinite(raw_logit):
        raise ValueError("NPR emitted a non-finite raw logit")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("NPR sigmoid probability falls outside [0, 1]")
    if not torch.equal(output, replay_output):
        raise ValueError("NPR fc1 manual replay does not exactly match output")
    if not torch.equal(probability_tensor, replay_probability):
        raise ValueError("NPR sigmoid manual replay does not exactly match")
    decision = probability > CLASSIFICATION_THRESHOLD
    feature = np.ascontiguousarray(
        feature_device.squeeze(0).detach().cpu().numpy(),
        dtype=np.float32,
    )
    if feature.shape != (FEATURE_DIMENSION,) or not np.isfinite(feature).all():
        raise ValueError("NPR pooled feature has an invalid shape/value")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return (
        {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "score_semantics": (
                "official_float32_sigmoid_probability_higher_is_fake"
            ),
            "classification_decision": decision,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "classification_threshold_operator": (
                CLASSIFICATION_THRESHOLD_OPERATOR
            ),
            "classification": {
                "raw_logit": raw_logit,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "threshold": CLASSIFICATION_THRESHOLD,
                "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
                "decision": decision,
                "semantics": (
                    "official_float32_sigmoid_probability_higher_is_fake"
                ),
            },
            "t1": {
                "raw_logit": raw_logit,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "threshold": CLASSIFICATION_THRESHOLD,
                "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
                "decision": decision,
                "policy": "official_NPR_AIGC_float32_sigmoid",
            },
            "manual_replay": {
                "raw_logit": replay_logit,
                "probability": replay_score,
                "ai_score": replay_score,
                "classification_decision": (
                    replay_score > CLASSIFICATION_THRESHOLD
                ),
                "model_forward_calls": 1,
                "fc_hook_calls": len(captured),
                "official_logit_exact_match": True,
                "official_probability_exact_match": True,
            },
            "latency_ms": latency_ms,
            "peak_cuda_memory_bytes": peak_memory,
        },
        feature,
    )


def _result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    visibility: Mapping[str, Any],
    config_fingerprint: str,
) -> dict[str, Any]:
    path = _anchored(Path(str(row["canonical_path"])), repo_root)
    return {
        "schema_version": "npr_detection_result_v1",
        "id": str(row["sample_id"]),
        "sample_id": str(row["sample_id"]),
        "rank": int(row["rank"]),
        "pair_rank": int(row["pair_rank"]),
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "label": int(row["label"]),
        "domain": str(row["domain"]),
        "candidate": str(row.get("candidate", "mouse")),
        "dataset_id": str(row.get("dataset_id")),
        "input_path": repo_relative(path, repo_root),
        "input_sha256": str(row["canonical_sha256"]),
        "input_width": int(row["width"]),
        "input_height": int(row["height"]),
        "preprocess_profile": PREPROCESS_PROFILE,
        "checkpoint_id": str(CHECKPOINT["id"]),
        "config_fingerprint": config_fingerprint,
        "edit_visibility": str(visibility["edit_visibility"]),
        "edit_visible_gt_fraction": float(
            visibility["edit_visible_gt_fraction"]
        ),
        "edit_visibility_evidence": dict(visibility),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
    }


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    run_dir: Path,
    config_fingerprint: str,
) -> None:
    if row.get("status") != "ok":
        raise ValueError("resume row is not successful")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume row field {key} changed")
    if row.get("config_fingerprint") != config_fingerprint:
        raise ValueError("resume row config fingerprint changed")

    def finite_number(value: Any, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"resume row has an invalid {label}")
        return float(value)

    raw_logit = finite_number(row.get("raw_logit"), "raw_logit")
    score = finite_number(row.get("ai_score"), "ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("resume row ai_score falls outside [0, 1]")
    decision = score > CLASSIFICATION_THRESHOLD
    scalar_aliases = {
        "probability": score,
        "score": score,
        "classification_decision": decision,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "score_semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
    }
    for key, value in scalar_aliases.items():
        if row.get(key) != value:
            raise ValueError(f"resume row scoring alias {key} changed")
    expected_classification = {
        "raw_logit": raw_logit,
        "probability": score,
        "ai_score": score,
        "score": score,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
    }
    if row.get("classification") != expected_classification:
        raise ValueError("resume row classification aliases changed")
    expected_t1 = {
        "raw_logit": raw_logit,
        "probability": score,
        "ai_score": score,
        "score": score,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "decision": decision,
        "policy": "official_NPR_AIGC_float32_sigmoid",
    }
    if row.get("t1") != expected_t1:
        raise ValueError("resume row T1 aliases changed")
    manual = row.get("manual_replay")
    if not isinstance(manual, Mapping):
        raise ValueError("resume row manual_replay is missing")
    if (
        manual.get("raw_logit") != raw_logit
        or manual.get("probability") != score
        or manual.get("ai_score") != score
        or manual.get("classification_decision") != decision
        or manual.get("model_forward_calls") != 1
        or manual.get("fc_hook_calls") != 1
        or manual.get("official_logit_exact_match") is not True
        or manual.get("official_probability_exact_match") is not True
    ):
        raise ValueError("resume row manual replay contract changed")

    preprocess = row.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ValueError("resume row preprocess audit is missing")
    width = int(expected["input_width"])
    height = int(expected["input_height"])
    effective_width, effective_height = effective_native_size(width, height)
    required_preprocess = {
        "profile": PREPROCESS_PROFILE,
        "decoded_size": [width, height],
        "effective_size": [effective_width, effective_height],
        "trim_bottom": height - effective_height,
        "trim_right": width - effective_width,
        "tensor_shape": [3, effective_height, effective_width],
        "tensor_dtype": "float32",
        "npr_residual_shape": [3, effective_height, effective_width],
        "npr_residual_dtype": "float32",
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
    }
    for key, value in required_preprocess.items():
        if preprocess.get(key) != value:
            raise ValueError(f"resume preprocess field {key} changed")
    for key in (
        "decoded_rgb_sha256",
        "tensor_sha256",
        "npr_residual_sha256",
    ):
        if not _valid_sha256(preprocess.get(key)):
            raise ValueError(f"resume preprocess field {key} is invalid")
    residual_stats = preprocess.get("npr_residual_stats")
    if not isinstance(residual_stats, Mapping):
        raise ValueError("resume residual statistics are missing")
    if residual_stats.get("elements") != 3 * effective_width * effective_height:
        raise ValueError("resume residual element count changed")
    nonzero = residual_stats.get("nonzero_elements")
    if (
        isinstance(nonzero, bool)
        or not isinstance(nonzero, int)
        or not 0 <= nonzero <= int(residual_stats["elements"])
    ):
        raise ValueError("resume residual nonzero count is invalid")
    for key in ("minimum", "maximum", "mean", "mean_absolute", "l2"):
        finite_number(residual_stats.get(key), f"residual statistic {key}")

    finite_number(row.get("preprocess_latency_ms"), "preprocess_latency_ms")
    if float(row["preprocess_latency_ms"]) < 0.0:
        raise ValueError("resume preprocess latency is negative")
    finite_number(row.get("latency_ms"), "latency_ms")
    if float(row["latency_ms"]) < 0.0:
        raise ValueError("resume inference latency is negative")
    peak_memory = row.get("peak_cuda_memory_bytes")
    if (
        peak_memory is not None
        and (
            isinstance(peak_memory, bool)
            or not isinstance(peak_memory, int)
            or peak_memory < 0
        )
    ):
        raise ValueError("resume peak CUDA memory is invalid")

    feature_value = row.get("npr_feature_path")
    feature_digest = row.get("npr_feature_sha256")
    if not isinstance(feature_value, str) or not _valid_sha256(feature_digest):
        raise ValueError("resume row has an invalid feature artifact")
    if (
        row.get("npr_feature_shape") != [FEATURE_DIMENSION]
        or row.get("npr_feature_dtype") != "float32"
        or row.get("npr_feature_semantics")
        != "official_fc1_input_after_adaptive_global_average_pool"
    ):
        raise ValueError("resume NPR feature metadata changed")
    feature_path = _anchored(Path(feature_value), repo_root)
    if not feature_path.is_file():
        # New runs persist repo-relative paths, but keep absolute compatibility.
        feature_path = (run_dir / feature_value).resolve()
    _verify_runtime_file(
        feature_path,
        str(feature_digest),
        f"resume NPR feature {row.get('id')}",
    )
    feature = np.load(feature_path, allow_pickle=False)
    if feature.shape != (FEATURE_DIMENSION,) or feature.dtype != np.float32:
        raise ValueError("resume NPR feature schema changed")
    if not np.isfinite(feature).all():
        raise ValueError("resume NPR feature contains non-finite values")

    forbidden = {
        "t2",
        "localization",
        "localization_metrics",
        "score_map",
        "score_map_path",
        "predicted_mask",
        "predicted_mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "dice",
        "s_joint",
    }
    present = sorted(forbidden.intersection(row))
    if present:
        raise ValueError(f"resume NPR row invents T2 fields: {present}")


def _run_config(
    *,
    adapter: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    release: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source_audit: Mapping[str, Any],
    asset_audit: Mapping[str, Any],
    device_text: str,
    pair_limit: int | None,
    sample_id: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "model_arch": MODEL_ARCH,
        "adapter_contract": dict(adapter),
        "source_commit": source_audit["commit"],
        "source_files": SOURCE_FILES,
        "checkpoint_id": CHECKPOINT["id"],
        "checkpoint_sha256": asset_audit["checkpoint"]["actual_sha256"],
        "checkpoint_schema_sha256": asset_audit["checkpoint"]["schema"][
            "items_sha256"
        ],
        "preprocess_profile": PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "Pillow_RGB_no_EXIF_transpose",
            "resize": None,
            "crop": None,
            "batch_size": 1,
            "odd_dimension_policy": (
                "drop_last_bottom_row_and_or_right_column_before_NPR"
            ),
            "normalization_mean": list(IMAGE_MEAN),
            "normalization_std": list(IMAGE_STD),
        },
        "model_contract": {
            "npr": (
                "x - nearest_upsample_2x(nearest_downsample_0.5x(x))"
            ),
            "npr_scale": 2.0 / 3.0,
            "feature_dimension": FEATURE_DIMENSION,
            "output": "one_raw_logit",
            "model_mode": "eval",
            "model_mode_source": "official_GitHub_test.py",
            "score": "torch_float32_sigmoid",
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "score_direction": "higher_means_fake",
            "valid_for_t2": False,
        },
        "runtime_contract": {
            "device": device_text,
            "seed": MODEL_SEED,
            "dtype": "float32",
            "autocast": False,
            "cudnn_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "runtime_evidence": dict(runtime_evidence),
        "runtime_evidence_fingerprint": _manifest_fingerprint(
            runtime_evidence
        ),
        "dataset": {
            "schema_version": release["schema_version"],
            "dataset_id": release.get("dataset_id"),
            "inputs_sha256": release["inputs_sha256"],
            "selected_ids": [str(row["sample_id"]) for row in selected],
            "selected_rows_sha256": hashlib.sha256(
                "".join(
                    f"{stable_json(row)}\n" for row in selected
                ).encode("utf-8")
            ).hexdigest(),
            "pair_limit": pair_limit,
            "sample_id": sample_id,
        },
        "metrics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "raw_logit_numerical_diagnostic": {
                "preregistered_before_full_mouse_run": True,
                "preregistered_at": "2026-07-25",
                "trigger": (
                    "CUDA pair-1 smoke produced finite raw logits near -170 "
                    "whose official float32 sigmoid saturated to exact zero"
                ),
                "policy": (
                    "always report raw-logit AUROC/AP, real-only 5% FPR, "
                    "paired ranking/delta and pair bootstrap beside the "
                    "official probability metrics; never choose whichever "
                    "looks better and never replace the released >0.5 rule"
                ),
            },
        },
        "license": LICENSE_RECORD,
        "checkpoint_selection_frozen_before_scores": True,
        "excluded_release_assets": EXCLUDED_RELEASE_ASSETS,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--hf-source-root",
        type=Path,
        default=DEFAULT_HF_SOURCE_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    source_root = _anchored(args.source_root, repo_root)
    hf_source_root = _anchored(args.hf_source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one non-empty path component")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    run_dir = results_root / args.run_id
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    feature_dir = run_dir / "features"

    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"run directory is non-empty; pass --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(all_rows, args.pair_limit, args.sample_id)
    validate_selected_inputs(selected, repo_root)
    # Build visibility from complete canonical pairs even for a one-image
    # diagnostic selection, then consume only the selected task entries.
    visibility = build_pair_visibility(all_rows, repo_root)
    source_audit, asset_audit, state, module = verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_source_root=hf_source_root,
    )
    device, runtime = configure_runtime(args.device)

    config = _run_config(
        adapter=adapter_contract(repo_root),
        runtime_evidence=runtime,
        release=release,
        selected=selected,
        source_audit=source_audit,
        asset_audit=asset_audit,
        device_text=str(device),
        pair_limit=args.pair_limit,
        sample_id=args.sample_id,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    config_fingerprint = _manifest_fingerprint(config)

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires existing manifest and expected-input snapshot"
            )
        prior_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if prior_manifest.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume manifest config fingerprint mismatch")
        if prior_manifest.get("runtime") != runtime:
            raise ValueError("resume runtime evidence changed")
        prior_expected = read_jsonl(expected_path)
        if prior_expected != selected:
            raise ValueError("resume expected-input snapshot changed")
    else:
        atomic_write_jsonl(expected_path, selected)

    manifest: dict[str, Any] = {
        "schema_version": "npr_detection_run_manifest_v1",
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "repo_root": str(repo_root),
        "config_fingerprint": config_fingerprint,
        "config": config,
        "source": source_audit,
        "assets": asset_audit,
        "runtime": runtime,
        "dataset": {
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "inputs_path": repo_relative(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
            "selected_tasks": len(
                {str(row["task_id"]) for row in selected}
            ),
        },
        "visibility_census": dict(
            Counter(
                visibility[task_id]["edit_visibility"]
                for task_id in sorted(
                    {str(row["task_id"]) for row in selected}
                )
            )
        ),
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "summary_path": repo_relative(summary_path, repo_root),
            "feature_dir": repo_relative(feature_dir, repo_root),
        },
    }
    atomic_write_json(manifest_path, manifest)

    model = load_model(module=module, state=state, device=device)
    del state
    latest = read_latest_by_id(results_path)
    completed = 0
    skipped = 0
    errors = 0
    for index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        pair_visibility = visibility[str(row["task_id"])]
        identity = _result_identity(
            row,
            repo_root=repo_root,
            visibility=pair_visibility,
            config_fingerprint=config_fingerprint,
        )
        prior = latest.get(sample_id)
        if prior is not None and prior.get("status") == "ok":
            _validate_resume_row(
                prior,
                expected=identity,
                repo_root=repo_root,
                run_dir=run_dir,
                config_fingerprint=config_fingerprint,
            )
            skipped += 1
            print(
                f"[{index}/{len(selected)}] resume {sample_id}",
                flush=True,
            )
            continue

        input_path = _anchored(
            Path(str(row["canonical_path"])),
            repo_root,
        )
        try:
            preprocess_started = time.perf_counter()
            tensor, preprocess = preprocess_image(input_path)
            preprocess_latency_ms = (
                time.perf_counter() - preprocess_started
            ) * 1000.0
            scoring, feature = _infer_one(
                model=model,
                tensor=tensor,
                device=device,
            )
            feature_path = feature_dir / f"{sample_id}.npy"
            _atomic_save_npy(feature_path, feature)
            result = {
                **identity,
                "status": "ok",
                "completed_at": utc_now(),
                "preprocess": preprocess,
                "preprocess_latency_ms": preprocess_latency_ms,
                "npr_feature_path": repo_relative(feature_path, repo_root),
                "npr_feature_sha256": sha256_file(feature_path),
                "npr_feature_shape": list(feature.shape),
                "npr_feature_dtype": str(feature.dtype),
                "npr_feature_semantics": (
                    "official_fc1_input_after_adaptive_global_average_pool"
                ),
                **scoring,
            }
            append_jsonl(results_path, result)
            latest[sample_id] = result
            completed += 1
            print(
                f"[{index}/{len(selected)}] ok {sample_id} "
                f"score={result['ai_score']:.9f}",
                flush=True,
            )
        except BaseException as exc:
            errors += 1
            error_row = {
                **identity,
                "status": "error",
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(results_path, error_row)
            latest[sample_id] = error_row
            print(
                f"[{index}/{len(selected)}] error {sample_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                __import__("torch").cuda.empty_cache()

    physical_results = read_jsonl(results_path) if results_path.is_file() else []
    summary = summarize_npr_results(
        physical_results,
        selected,
        threshold=CLASSIFICATION_THRESHOLD,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary["raw_logit_numerical_diagnostic"] = (
        summarize_npr_raw_logit_diagnostic(
            physical_results,
            selected,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    )
    summary.update(
        {
            "run_id": args.run_id,
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_id": CHECKPOINT["id"],
            "preprocess_profile": PREPROCESS_PROFILE,
            "config_fingerprint": config_fingerprint,
            "generated_at": utc_now(),
        }
    )
    atomic_write_json(summary_path, summary)

    manifest["status"] = (
        "complete" if summary["coverage"]["is_complete"] else "incomplete"
    )
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": completed,
        "resume_skips": skipped,
        "new_errors": errors,
        "physical_result_rows": len(physical_results),
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "feature_files": sum(
                1 for path in feature_dir.glob("*.npy") if path.is_file()
            ),
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": manifest["status"],
                "coverage": summary["coverage"],
                "paired_coverage": summary["paired_coverage"],
                "detection": summary["detection"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
