#!/usr/bin/env python3
"""Run the official UniversalFakeDetect CLIP ViT-L/14 linear probe.

The official repository's current validation transform differs from the
transform present when ``fc_weights.pth`` was first committed.  This adapter
therefore requires one of two explicit preprocessing profiles and never mixes
them within a run.  Source, linear head, and OpenAI CLIP backbone paths are
all explicit, fully hashed, and network access is blocked during construction.
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
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
import urllib.request
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
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


MODEL_NAME = "UniversalFakeDetect"
MODEL_SLUG = "universalfakedetect_clip_vit_l14_ours_lc"
MODEL_ARCH = "CLIP:ViT-L/14"
MODEL_REPO_URL = (
    "https://github.com/WisconsinAIVision/UniversalFakeDetect"
)
MODEL_SOURCE_COMMIT = "76a0e3e60a8a06458707a625d269ba815a2e5919"
PAPER_URL = "https://arxiv.org/abs/2302.10174"
LICENSE_SPDX = "MIT"
LICENSE_RECORD = {
    "code": {"spdx": "MIT"},
    "linear_head_weights": {
        "terms": "no_separate_explicit_terms",
        "commercial_cleared": False,
    },
    "openai_clip": {
        "code_license": "MIT",
        "model_card_note": "deployment_out_of_scope_risk_note",
        "commercial_cleared": False,
    },
    "overall_commercial_clearance": "not_established",
}

FEATURE_DIMENSION = 768
MODEL_INPUT_SIZE = 224
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
# CUDA and CPU float32 sigmoid kernels can differ by one ULP.  Resume
# validation permits only this frozen absolute replay tolerance; every score
# persisted by the original inference remains subject to exact alias checks.
RESUME_CPU_SIGMOID_ABS_TOLERANCE = 1e-7
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CURRENT_PROFILE = "current_head_native_center_crop224"
CHECKPOINT_ERA_PROFILE = "checkpoint_era_resize256_center_crop224"
HEAD_INTRO_COMMIT = "763391eff3284f6950ffb323599c1a7a819f2ecd"
RESIZE_REMOVAL_COMMIT = "3bf72282088e47be7e784e104e577790a55d4e48"
HEAD_ERA_VALIDATE_SHA256 = (
    "8a68ba2af3e7586f294bdbc5643a1e46f418916406c3129f24647f275da8d77a"
)

PREPROCESS_PROFILES = {
    CURRENT_PROFILE: {
        "profile_id": CURRENT_PROFILE,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.CenterCrop_224",
            "torchvision.transforms.ToTensor",
            "CLIP_Normalize",
        ],
        "resize_short_side": None,
        "center_crop": 224,
        "source_basis": "current_HEAD_validate.py",
        "source_commit": MODEL_SOURCE_COMMIT,
        "drift": (
            "current HEAD removed Resize(256) after the released linear head "
            "was committed"
        ),
    },
    CHECKPOINT_ERA_PROFILE: {
        "profile_id": CHECKPOINT_ERA_PROFILE,
        "steps": [
            "Pillow.Image.open.convert_RGB",
            "torchvision.transforms.Resize_short_side_256_PIL_bilinear",
            "torchvision.transforms.CenterCrop_224",
            "torchvision.transforms.ToTensor",
            "CLIP_Normalize",
        ],
        "resize_short_side": 256,
        "center_crop": 224,
        "source_basis": "validate.py_at_head_introduction_commit",
        "source_commit": HEAD_INTRO_COMMIT,
        "validate_blob_sha256": HEAD_ERA_VALIDATE_SHA256,
        "resize_removed_by_commit": RESIZE_REMOVAL_COMMIT,
        "drift": (
            "fc_weights.pth was introduced while validate.py still applied "
            "Resize(256) before CenterCrop(224)"
        ),
    },
}

SOURCE_FILES = {
    "LICENSE": "923dbc0fe040cc826122491e98b20d6f401e510d50c8b666de908e59ebf2f2b2",
    "README.md": "9006d38199a13f64968f20e9a8ea6532cf54295d25efffb4354cc0d58a0a4951",
    "validate.py": "9ab4021cc6f85002a8b8cd0fc28baa9b4861b59bffcfbd98e10d02b08c42b2d6",
    "models/__init__.py": "7ea7f9931cfb773d552e30d2333a83c416798a2862119373356c7d03fb552299",
    "models/clip_models.py": "57ce5898bf0bc7ff52b5922a0aefae8bc34a9237e4d59e4a1615ff5d8c6ff7a6",
    "models/imagenet_models.py": "f291a0179027a8efc0e59b62edb9eecfe7a28add8312e5929a81dfb46e3096e0",
    "models/resnet.py": "c2e13589767e3da0af9f5037dfb6933bcd5bddffef4d50c81d80f8a22a49a263",
    "models/vision_transformer.py": "f1fd9e3023a0cf246e99bec02096a30228c72ae79e70e1346234468bc8f415f3",
    "models/vision_transformer_misc.py": "d25390fc38a08ec0b005d76eeeb122a89a9c3eb296d8fb2b1da3f73745e1e715",
    "models/vision_transformer_utils.py": "25be7a4a7a74a58e47ab4c32770b8334f1b000c9b4cd15fd0eb55e2faee68bcc",
    "models/clip/__init__.py": "d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8",
    "models/clip/clip.py": "c3f1c09abe0a0d9c429e0d47f2b8d4a4ec09dbd11eec3d7dd9846babd717e43b",
    "models/clip/model.py": "c071d011e92226f1ca0a6f7c5098d8d7d08eadc7d6db125a81b52fc234b1ec59",
    "models/clip/simple_tokenizer.py": "d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad",
    "models/clip/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
}

HEAD_CHECKPOINT = {
    "released_name": "Ours (L/14 + LC)",
    "filename": "fc_weights.pth",
    "repo_relative_path": "pretrained_weights/fc_weights.pth",
    "introduced_commit": HEAD_INTRO_COMMIT,
    "bytes": 4_083,
    "sha256": "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
    "format": "torch_state_dict_weights_only",
}

BACKBONE_CHECKPOINT = {
    "provider": "OpenAI",
    "architecture": "ViT-L/14",
    "filename": "ViT-L-14.pt",
    "bytes": 932_768_134,
    "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    "format": "torchscript_archive",
    "official_url": (
        "https://openaipublic.azureedge.net/clip/models/"
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
        "ViT-L-14.pt"
    ),
}

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
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


def _git_bytes(repo: Path, object_name: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", object_name],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot read pinned git object {object_name}") from exc


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_real(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
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
        raise ValueError("pair_limit and sample_id are mutually exclusive")
    if sample_id is not None:
        selected = [
            row for row in rows if str(row.get("sample_id")) == sample_id
        ]
        if len(selected) != 1:
            raise ValueError(f"sample-id must select exactly one row: {sample_id}")
        return selected
    pair_ranks = sorted({int(row["pair_rank"]) for row in rows})
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        pair_ranks = pair_ranks[:pair_limit]
    selected_ranks = set(pair_ranks)
    selected = [
        row for row in rows if int(row["pair_rank"]) in selected_ranks
    ]
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


def _validate_gt_row(
    row: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    sample_id = str(row.get("sample_id"))
    width, height = int(row["width"]), int(row["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions for {sample_id}")
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


def _validate_selected_inputs(
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
        _validate_gt_row(row, repo_root)


def _verify_source_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"missing explicit UniversalFakeDetect source-root: {source_root}"
        )
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            "UniversalFakeDetect source commit mismatch: "
            f"{commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(
            f"UniversalFakeDetect tracked source tree is dirty: {dirty[:1000]}"
        )
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"UniversalFakeDetect source {relative}",
        )
    bundled_head = source_root / str(HEAD_CHECKPOINT["repo_relative_path"])
    _verify_runtime_file(
        bundled_head,
        str(HEAD_CHECKPOINT["sha256"]),
        "repository-bundled Ours (L/14 + LC) head",
    )
    if bundled_head.stat().st_size != int(HEAD_CHECKPOINT["bytes"]):
        raise ValueError("repository-bundled UFD head size changed")
    historical_validate = _git_bytes(
        source_root,
        f"{HEAD_INTRO_COMMIT}:validate.py",
    )
    historical_sha = hashlib.sha256(historical_validate).hexdigest()
    if historical_sha != HEAD_ERA_VALIDATE_SHA256:
        raise ValueError("checkpoint-era validate.py blob changed")
    head_history = _git_value(
        source_root,
        "log",
        "--format=%H",
        "--",
        str(HEAD_CHECKPOINT["repo_relative_path"]),
    )
    if head_history != HEAD_INTRO_COMMIT:
        raise ValueError("released head introduction history changed")
    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            relative: {
                "path": str(source_root / relative),
                "sha256": digest,
            }
            for relative, digest in SOURCE_FILES.items()
        },
        "bundled_head": {
            "path": str(bundled_head),
            "bytes": bundled_head.stat().st_size,
            "sha256": sha256_file(bundled_head),
        },
        "preprocess_drift": {
            "head_introduction_commit": HEAD_INTRO_COMMIT,
            "head_era_validate_sha256": historical_sha,
            "resize_removal_commit": RESIZE_REMOVAL_COMMIT,
            "current_source_commit": MODEL_SOURCE_COMMIT,
            "interpretation": (
                "the repository does not identify whether the released head "
                "should use its checkpoint-era or current validation transform"
            ),
        },
    }


def _payload_schema(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
    if isinstance(value, Mapping):
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "items": {
                str(key): _payload_schema(item)
                for key, item in value.items()
            },
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    raise ValueError(
        "unexpected object in UFD weights-only head: "
        f"{type(value).__module__}.{type(value).__name__}"
    )


def _verify_asset_contract(
    *,
    source_root: Path,
    head_checkpoint: Path,
    backbone_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    source_audit = _verify_source_contract(source_root)
    head_checkpoint = head_checkpoint.resolve()
    backbone_checkpoint = backbone_checkpoint.resolve()
    _verify_runtime_file(
        head_checkpoint,
        str(HEAD_CHECKPOINT["sha256"]),
        "explicit UFD Ours (L/14 + LC) head",
    )
    if head_checkpoint.stat().st_size != int(HEAD_CHECKPOINT["bytes"]):
        raise ValueError("explicit UFD head size changed")
    unsafe = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(head_checkpoint)
    )
    if unsafe:
        raise ValueError(f"UFD linear head contains unsafe globals: {unsafe}")
    state = torch.load(
        head_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(state, (dict, OrderedDict))
        or list(state) != ["weight", "bias"]
        or not isinstance(state["weight"], torch.Tensor)
        or tuple(state["weight"].shape) != (1, FEATURE_DIMENSION)
        or state["weight"].dtype != torch.float32
        or not isinstance(state["bias"], torch.Tensor)
        or tuple(state["bias"].shape) != (1,)
        or state["bias"].dtype != torch.float32
        or not bool(torch.isfinite(state["weight"]).all())
        or not bool(torch.isfinite(state["bias"]).all())
    ):
        raise ValueError("UFD linear-head state schema changed")
    _verify_runtime_file(
        backbone_checkpoint,
        str(BACKBONE_CHECKPOINT["sha256"]),
        "explicit OpenAI CLIP ViT-L/14 backbone",
    )
    if backbone_checkpoint.stat().st_size != int(BACKBONE_CHECKPOINT["bytes"]):
        raise ValueError("OpenAI CLIP ViT-L/14 byte size changed")
    if not zipfile.is_zipfile(backbone_checkpoint):
        raise ValueError("OpenAI CLIP backbone is not a TorchScript archive")
    head_audit = {
        **HEAD_CHECKPOINT,
        "path": str(head_checkpoint),
        "serialization_safety": {
            "preflight": (
                "torch.serialization.get_unsafe_globals_in_checkpoint"
            ),
            "unsafe_globals": unsafe,
            "required_unsafe_globals": [],
            "weights_only": True,
        },
        "payload_schema": _payload_schema(state),
    }
    backbone_audit = {
        **BACKBONE_CHECKPOINT,
        "path": str(backbone_checkpoint),
        "archive_preflight": "zipfile.is_zipfile",
        "archive_preflight_passed": True,
        "load_api": "official_clip.load_via_torch.jit.load",
        "network_allowed": False,
    }
    bundle_value = [
        {
            "role": "linear_head",
            "bytes": int(HEAD_CHECKPOINT["bytes"]),
            "sha256": str(HEAD_CHECKPOINT["sha256"]),
        },
        {
            "role": "clip_backbone",
            "bytes": int(BACKBONE_CHECKPOINT["bytes"]),
            "sha256": str(BACKBONE_CHECKPOINT["sha256"]),
        },
    ]
    asset_audit = {
        "bundle_sha256": hashlib.sha256(
            stable_json(bundle_value).encode("utf-8")
        ).hexdigest(),
        "head": head_audit,
        "backbone": backbone_audit,
    }
    return source_audit, asset_audit, state


def _load_official_models_package(source_root: Path) -> types.ModuleType:
    package_name = "_claimforge_ufd_76a0e3e.models"
    existing = sys.modules.get(package_name)
    init_path = source_root / "models" / "__init__.py"
    if existing is not None:
        if Path(str(getattr(existing, "__file__", ""))).resolve() != init_path.resolve():
            raise RuntimeError("pinned UniversalFakeDetect module collision")
        return existing
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official models package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise
    return module


def configure_determinism(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_model(
    *,
    source_root: Path,
    head_checkpoint: Path,
    backbone_checkpoint: Path,
    device_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    source_audit, asset_audit, head_state = _verify_asset_contract(
        source_root=source_root,
        head_checkpoint=head_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    models_module = _load_official_models_package(source_root)
    package_name = str(models_module.__name__)
    clip_module = sys.modules.get(f"{package_name}.clip.clip")
    if clip_module is None:
        raise RuntimeError("official bundled CLIP module was not imported")
    official_url = getattr(clip_module, "_MODELS", {}).get("ViT-L/14")
    if official_url != BACKBONE_CHECKPOINT["official_url"]:
        raise ValueError("official bundled ViT-L/14 URL/hash contract changed")
    download_calls: list[dict[str, str]] = []

    def resolve_pinned_backbone(url: str, root: str) -> str:
        if url != BACKBONE_CHECKPOINT["official_url"]:
            raise RuntimeError(f"unexpected official CLIP asset URL: {url}")
        _verify_runtime_file(
            backbone_checkpoint,
            str(BACKBONE_CHECKPOINT["sha256"]),
            "pinned OpenAI CLIP backbone during constructor",
        )
        download_calls.append({"url": url, "requested_cache_root": str(root)})
        return str(backbone_checkpoint)

    def reject_torch_load(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "official CLIP TorchScript fallback attempted unsafe torch.load"
        )

    with (
        mock.patch.object(
            clip_module,
            "_download",
            side_effect=resolve_pinned_backbone,
        ),
        mock.patch.object(
            urllib.request,
            "urlopen",
            side_effect=RuntimeError("network access is forbidden"),
        ) as urlopen,
        mock.patch.object(torch, "load", side_effect=reject_torch_load),
    ):
        model = models_module.get_model(MODEL_ARCH)
    if len(download_calls) != 1 or urlopen.call_count != 0:
        raise RuntimeError("official model construction violated no-network contract")
    if not isinstance(model.fc, torch.nn.Linear) or (
        model.fc.in_features,
        model.fc.out_features,
    ) != (FEATURE_DIMENSION, 1):
        raise ValueError("official UFD classifier shape changed")
    incompatibility = model.fc.load_state_dict(head_state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("strict UFD linear-head load reported incompatibilities")
    if not torch.equal(model.fc.weight.detach().cpu(), head_state["weight"]):
        raise ValueError("loaded UFD head weight differs from audited state")
    if not torch.equal(model.fc.bias.detach().cpu(), head_state["bias"]):
        raise ValueError("loaded UFD head bias differs from audited state")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_resolution = getattr(model.model.visual, "input_resolution", None)
    if hasattr(visual_resolution, "item"):
        visual_resolution = int(visual_resolution.item())
    else:
        visual_resolution = int(visual_resolution)
    if visual_resolution != MODEL_INPUT_SIZE:
        raise ValueError("OpenAI CLIP ViT-L/14 input resolution changed")
    audit = {
        "source": source_audit,
        "assets": asset_audit,
        "class_module": model.__class__.__module__,
        "class_name": model.__class__.__name__,
        "construction_api": "official models.get_model('CLIP:ViT-L/14')",
        "official_download_patch_calls": download_calls,
        "urlopen_calls": urlopen.call_count,
        "network_blocked": True,
        "clip_torch_load_fallback_blocked": True,
        "head_load": {
            "api": "torch.load",
            "weights_only": True,
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "feature_dimension": FEATURE_DIMENSION,
        "visual_input_resolution": visual_resolution,
        "head_parameters": sum(
            int(parameter.numel()) for parameter in model.fc.parameters()
        ),
    }
    return model, device, audit


def compute_preprocess_geometry(
    width: int,
    height: int,
    profile_id: str,
) -> dict[str, Any]:
    if profile_id not in PREPROCESS_PROFILES:
        raise ValueError(f"unsupported UFD preprocess profile: {profile_id}")
    if width <= 0 or height <= 0:
        raise ValueError("native image dimensions must be positive")
    resize_short = PREPROCESS_PROFILES[profile_id]["resize_short_side"]
    if resize_short is None:
        resized_width, resized_height = width, height
        resize = {
            "enabled": False,
            "source_size": [width, height],
            "destination_size": [width, height],
            "short_side": None,
            "interpolation": None,
            "antialias": None,
            "rounding": None,
        }
    else:
        short, long = (
            (width, height) if width <= height else (height, width)
        )
        new_short = int(resize_short)
        new_long = int(new_short * long / short)
        resized_width, resized_height = (
            (new_short, new_long)
            if width <= height
            else (new_long, new_short)
        )
        resize = {
            "enabled": True,
            "source_size": [width, height],
            "destination_size": [resized_width, resized_height],
            "short_side": int(resize_short),
            "interpolation": "PIL_BILINEAR",
            "antialias": True,
            "rounding": "torchvision_int_truncation_for_long_side",
        }

    crop_size = MODEL_INPUT_SIZE
    pad_left = (
        (crop_size - resized_width) // 2
        if crop_size > resized_width
        else 0
    )
    pad_top = (
        (crop_size - resized_height) // 2
        if crop_size > resized_height
        else 0
    )
    pad_right = (
        (crop_size - resized_width + 1) // 2
        if crop_size > resized_width
        else 0
    )
    pad_bottom = (
        (crop_size - resized_height + 1) // 2
        if crop_size > resized_height
        else 0
    )
    padded_width = resized_width + pad_left + pad_right
    padded_height = resized_height + pad_top + pad_bottom
    crop_left = int(round((padded_width - crop_size) / 2.0))
    crop_top = int(round((padded_height - crop_size) / 2.0))
    crop_right, crop_bottom = (
        crop_left + crop_size,
        crop_top + crop_size,
    )

    visible_left = max(0.0, float(crop_left - pad_left))
    visible_top = max(0.0, float(crop_top - pad_top))
    visible_right = min(
        float(resized_width),
        float(crop_right - pad_left),
    )
    visible_bottom = min(
        float(resized_height),
        float(crop_bottom - pad_top),
    )
    native_crop = [
        visible_left * width / resized_width,
        visible_top * height / resized_height,
        visible_right * width / resized_width,
        visible_bottom * height / resized_height,
    ]
    return {
        "profile_id": profile_id,
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "native_size": [width, height],
        "resize": resize,
        "center_crop": {
            "input_size": [resized_width, resized_height],
            "padding_ltrb": [
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            "padding_fill": 0,
            "padded_size": [padded_width, padded_height],
            "start_xy": [crop_left, crop_top],
            "size": [crop_size, crop_size],
            "end_xy": [crop_right, crop_bottom],
            "rounding": "int(round((dimension-crop)/2.0))",
        },
        "effective_native_crop_xyxy": native_crop,
        "pixel_center_mapping": (
            "if resized: d=(native_index+0.5)*resized_size/native_size-0.5; "
            "else d=native_index; then d+=center_crop_padding_left_or_top"
        ),
        "normalize": {
            "to_tensor_scale": "uint8_div_255_to_float32",
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
        },
    }


def preprocess_image(
    path: Path,
    profile_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if profile_id == CHECKPOINT_ERA_PROFILE:
            resized = transforms.Resize(
                256,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )(rgb)
        elif profile_id == CURRENT_PROFILE:
            resized = rgb
        else:
            raise ValueError(
                f"unsupported UFD preprocess profile: {profile_id}"
            )
        cropped = transforms.CenterCrop(MODEL_INPUT_SIZE)(resized)
        crop_rgb = np.ascontiguousarray(
            np.asarray(cropped, dtype=np.uint8)
        )
    tensor = transforms.Normalize(CLIP_MEAN, CLIP_STD)(
        transforms.ToTensor()(cropped)
    )
    tensor_array = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if (
        decoded.shape != (height, width, 3)
        or crop_rgb.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
        or tensor_array.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        or tensor_array.dtype != np.float32
        or not np.isfinite(tensor_array).all()
    ):
        raise ValueError("UFD preprocessing output contract changed")
    geometry = compute_preprocess_geometry(width, height, profile_id)
    if cropped.size != tuple(geometry["center_crop"]["size"]):
        raise ValueError("UFD crop geometry differs from torchvision output")
    audit = {
        "geometry": geometry,
        "decoded_rgb_sha256": _sha256_array(decoded),
        "crop_rgb_sha256": _sha256_array(crop_rgb),
        "crop_rgb_shape": list(crop_rgb.shape),
        "crop_rgb_dtype": str(crop_rgb.dtype),
        "tensor_shape": list(tensor_array.shape),
        "tensor_dtype": str(tensor_array.dtype),
        "tensor_sha256": _sha256_array(tensor_array),
    }
    return tensor_array, audit


def _intersection_xyxy(
    first: list[float],
    second: list[float],
) -> list[float] | None:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return (
        [left, top, right, bottom]
        if right > left and bottom > top
        else None
    )


def _edit_box_visibility(
    edit_region: list[int],
    native_crop: list[float],
) -> dict[str, Any]:
    box = [float(value) for value in edit_region]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("edit_region_xyxy has non-positive area")
    intersection = _intersection_xyxy(box, native_crop)
    area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    fraction = min(1.0, max(0.0, visible_area / area))
    category = (
        "none"
        if fraction == 0.0
        else "full"
        if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else "partial"
    )
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": native_crop,
        "intersection_xyxy": intersection,
        "edit_area": area,
        "visible_area": visible_area,
        "visible_fraction": fraction,
        "category": category,
        "basis": (
            "continuous_edit_box_area_intersection_with_effective_native_crop"
        ),
    }


def _gt_visibility(
    forged_row: Mapping[str, Any],
    gt: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    positive_y, positive_x = np.nonzero(gt == 255)
    total = int(positive_x.size)
    if total <= 0:
        raise ValueError("forged exact-diff GT has no positive pixels")
    width, height = [int(value) for value in geometry["native_size"]]
    resized_width, resized_height = [
        int(value) for value in geometry["resize"]["destination_size"]
    ]
    pad_left, pad_top = [
        int(value)
        for value in geometry["center_crop"]["padding_ltrb"][:2]
    ]
    crop_left, crop_top = [
        float(value) for value in geometry["center_crop"]["start_xy"]
    ]
    crop_right, crop_bottom = [
        float(value) for value in geometry["center_crop"]["end_xy"]
    ]
    if geometry["resize"]["enabled"]:
        destination_x = (
            (positive_x.astype(np.float64) + 0.5)
            * resized_width
            / width
            - 0.5
        )
        destination_y = (
            (positive_y.astype(np.float64) + 0.5)
            * resized_height
            / height
            - 0.5
        )
    else:
        destination_x = positive_x.astype(np.float64)
        destination_y = positive_y.astype(np.float64)
    destination_x += pad_left
    destination_y += pad_top
    visible_mask = (
        (destination_x >= crop_left)
        & (destination_x < crop_right)
        & (destination_y >= crop_top)
        & (destination_y < crop_bottom)
    )
    visible = int(np.count_nonzero(visible_mask))
    category = (
        "none" if visible == 0 else "full" if visible == total else "partial"
    )
    return {
        "category": category,
        "visible_fraction": visible / total,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible,
        "forged_sample_id": str(forged_row["sample_id"]),
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_through_"
            "selected_UFD_resize_padding_and_center_crop"
        ),
        "profile_id": str(geometry["profile_id"]),
        "formula": str(geometry["pixel_center_mapping"]),
    }


def build_pair_visibility(
    all_rows: list[dict[str, Any]],
    repo_root: Path,
    profile_id: str,
) -> dict[str, dict[str, Any]]:
    if profile_id not in PREPROCESS_PROFILES:
        raise ValueError(f"unsupported UFD preprocess profile: {profile_id}")
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        pairs.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    result: dict[str, dict[str, Any]] = {}
    for task_id, pair in pairs.items():
        if set(pair) != {"real", "forged"}:
            raise ValueError(f"canonical task is incomplete: {task_id}")
        real, forged = pair["real"], pair["forged"]
        if (
            int(real["width"]) != int(forged["width"])
            or int(real["height"]) != int(forged["height"])
            or real.get("domain") != forged.get("domain")
            or real.get("edit_region_xyxy") != forged.get("edit_region_xyxy")
        ):
            raise ValueError(f"canonical pair geometry mismatch: {task_id}")
        gt = _validate_gt_row(forged, repo_root)
        assert gt is not None
        geometry = compute_preprocess_geometry(
            int(forged["width"]),
            int(forged["height"]),
            profile_id,
        )
        gt_evidence = _gt_visibility(forged, gt, geometry)
        edit_region = forged.get("edit_region_xyxy")
        if (
            not isinstance(edit_region, list)
            or len(edit_region) != 4
            or any(not isinstance(value, int) for value in edit_region)
        ):
            raise ValueError(f"invalid edit region for {task_id}")
        result[task_id] = {
            "edit_visibility": gt_evidence["category"],
            "edit_visible_gt_fraction": gt_evidence["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": gt_evidence,
                "edit_box": _edit_box_visibility(
                    edit_region,
                    list(geometry["effective_native_crop_xyxy"]),
                ),
            },
        }
    return result


def replay_classifier(
    official_output: Any,
    feature: Any,
    classifier: Any,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    if (
        not isinstance(feature, torch.Tensor)
        or tuple(feature.shape) != (1, FEATURE_DIMENSION)
        or feature.dtype != torch.float32
        or not bool(torch.isfinite(feature).all())
    ):
        raise ValueError("captured UFD CLIP feature violates [1,768] float32")
    if (
        not isinstance(official_output, torch.Tensor)
        or tuple(official_output.shape) != (1, 1)
        or official_output.dtype != torch.float32
        or not bool(torch.isfinite(official_output).all())
    ):
        raise ValueError("official UFD output violates [1,1] float32")
    with torch.inference_mode():
        manual_output = F.linear(
            feature,
            classifier.weight,
            classifier.bias,
        )
        official_probability = torch.sigmoid(official_output)
        manual_probability = torch.sigmoid(manual_output)
    if not torch.equal(manual_output, official_output):
        raise ValueError("manual F.linear replay differs from official fc output")
    if not torch.equal(manual_probability, official_probability):
        raise ValueError("manual sigmoid replay differs from official sigmoid")
    raw_logit = float(official_output.reshape(()).item())
    probability = float(official_probability.reshape(()).item())
    if not math.isfinite(raw_logit) or not math.isfinite(probability):
        raise ValueError("UFD classifier produced non-finite values")
    decision = probability > CLASSIFICATION_THRESHOLD
    return {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "manual_replay": {
            "raw_logit": float(manual_output.reshape(()).item()),
            "probability": float(manual_probability.reshape(()).item()),
            "ai_score": float(manual_probability.reshape(()).item()),
            "classification_decision": bool(
                (manual_probability > CLASSIFICATION_THRESHOLD).item()
            ),
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
        },
    }


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, int, float]:
    import torch

    if (
        image.dtype != np.float32
        or image.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        or not image.flags.c_contiguous
    ):
        raise ValueError("UFD input tensor array contract changed")
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    captured: list[Any] = []

    def capture_fc(
        _module: Any,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if len(inputs) != 1:
            raise RuntimeError("official UFD fc received unexpected inputs")
        captured.append(inputs[0].detach().clone())

    hook = model.fc.register_forward_hook(capture_fc)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            official_output = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"official UFD fc hook fired {len(captured)} times"
        )
    feature_tensor = captured[0]
    processed = replay_classifier(
        official_output,
        feature_tensor,
        model.fc,
    )
    feature = np.ascontiguousarray(
        feature_tensor[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return processed, feature, peak_bytes, latency_ms


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


def _complete_pair_count(rows: list[dict[str, Any]]) -> int:
    kinds_by_task: dict[str, set[str]] = {}
    for row in rows:
        kinds_by_task.setdefault(str(row["task_id"]), set()).add(str(row["kind"]))
    return sum(kinds == {"real", "forged"} for kinds in kinds_by_task.values())


def _runtime_contract(device_name: str) -> dict[str, Any]:
    import torch
    import torchvision

    requested = torch.device(device_name)
    cuda_active = requested.type == "cuda" and torch.cuda.is_available()
    device_index = (
        requested.index
        if requested.index is not None
        else torch.cuda.current_device()
        if cuda_active
        else None
    )
    resolved_cuda = (
        torch.device("cuda", device_index) if cuda_active else None
    )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "torch": {
                "version": str(torch.__version__),
                "full_version": str(torch.__version__),
                "distribution_version": _package_version("torch"),
            },
            "torchvision": {
                "version": str(torchvision.__version__),
                "full_version": str(torchvision.__version__),
                "distribution_version": _package_version("torchvision"),
            },
            "numpy": str(np.__version__),
            "Pillow": _package_version("Pillow"),
            "ftfy": _package_version("ftfy"),
            "regex": _package_version("regex"),
        },
        "accelerator": {
            "requested_device": str(requested),
            "device_type": requested.type,
            "device_index": device_index,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_version": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu_name": (
                torch.cuda.get_device_name(resolved_cuda)
                if resolved_cuda is not None
                else None
            ),
            "gpu_capability": (
                list(torch.cuda.get_device_capability(resolved_cuda))
                if resolved_cuda is not None
                else None
            ),
        },
        "numerical_flags": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def build_run_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    dataset_manifest_path: Path,
    release: dict[str, Any],
    inputs_path: Path,
    selected: list[dict[str, Any]],
    source_root: Path,
    head_checkpoint: Path,
    backbone_checkpoint: Path,
    source_audit: dict[str, Any],
    asset_audit: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    runtime = _runtime_contract(args.device)
    metrics_path = Path(__file__).with_name("ufd_metrics.py")
    adapter_paths = [Path(__file__), Path(__file__).with_name("common.py")]
    if metrics_path.is_file():
        adapter_paths.append(metrics_path)
    profile = PREPROCESS_PROFILES[args.preprocess_profile]
    immutable = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "preprocess_profile": args.preprocess_profile,
        "preprocess_profile_contract": profile,
        "source_checkpoint_drift": {
            "head_introduction_commit": HEAD_INTRO_COMMIT,
            "head_era_validate_sha256": HEAD_ERA_VALIDATE_SHA256,
            "checkpoint_era_profile": CHECKPOINT_ERA_PROFILE,
            "resize_removal_commit": RESIZE_REMOVAL_COMMIT,
            "current_profile": CURRENT_PROFILE,
            "current_source_commit": MODEL_SOURCE_COMMIT,
            "ambiguity": (
                "the release does not state whether the linear head should be "
                "evaluated with checkpoint-era Resize(256) or current native "
                "CenterCrop(224)"
            ),
            "resolution": (
                "profile is mandatory and immutable; the two profiles are "
                "reported as separate experimental conditions"
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "architecture": MODEL_ARCH,
            "repo_url": MODEL_REPO_URL,
            "paper_url": PAPER_URL,
            "source_root": str(source_root),
            "source_commit": MODEL_SOURCE_COMMIT,
            "license": LICENSE_RECORD,
            "task_support": {"t1": True, "t2": False},
            "source_audit": source_audit,
            "head_checkpoint": asset_audit["head"],
            "backbone_checkpoint": asset_audit["backbone"],
            "asset_bundle_sha256": asset_audit["bundle_sha256"],
            "explicit_paths": {
                "source_root": str(source_root),
                "head_checkpoint": str(head_checkpoint),
                "backbone_checkpoint": str(backbone_checkpoint),
            },
        },
        "dataset": {
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "dataset_id": release["dataset_id"],
            "inputs_path": repo_relative(inputs_path, repo_root),
            "inputs_sha256": release["inputs_sha256"],
        },
        "selection": {
            "pair_limit": args.pair_limit,
            "sample_id": getattr(args, "sample_id", None),
            "rows": _selection_contract(selected),
        },
        "protocol": {
            "official_model_api": "models.get_model('CLIP:ViT-L/14')",
            "official_forward_calls_per_image": 1,
            "feature_capture": (
                "forward hook on official model.fc input"
            ),
            "feature": {
                "shape": [FEATURE_DIMENSION],
                "dtype": "float32",
                "persisted_for_every_ok_image": True,
            },
            "preprocess_profile": args.preprocess_profile,
            "preprocess_profile_contract": profile,
            "backbone_resolution": (
                "official clip._download patched to exact explicit asset; "
                "urllib.request.urlopen blocked"
            ),
            "head_load": "torch.load(weights_only=True), strict fc state",
            "classification": {
                "raw": "official_linear_head_logit",
                "score": "torch.float32_sigmoid_probability",
                "threshold": CLASSIFICATION_THRESHOLD,
                "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
                "strict": True,
            },
            "manual_replay": (
                "captured feature through independent torch.nn.functional."
                "linear and torch.sigmoid; exact equality required"
            ),
            "edit_visibility": {
                "category_field": "edit_visibility",
                "fraction_field": "edit_visible_gt_fraction",
                "pair_level": True,
                "real_policy": (
                    "copy counterfactual forged exact-diff GT visibility"
                ),
                "profile_specific": True,
            },
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "runtime_contract": runtime,
        "environment": runtime,
        "expected_complete_pairs": _complete_pair_count(selected),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": [
            {
                "path": repo_relative(path, repo_root),
                "sha256": sha256_file(path),
            }
            for path in adapter_paths
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
    }


def _write_or_validate_run_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                "existing UFD run manifest fingerprint differs; "
                "use a new run-id"
            )
        return
    atomic_write_json(path, manifest)


def _float32_sigmoid(value: float) -> float:
    import torch

    tensor = torch.tensor(value, dtype=torch.float32)
    return float(torch.sigmoid(tensor).item())


def _validated_resume_decision(
    raw_logit: float,
    stored_probability: float,
    *,
    sample_id: Any,
) -> bool:
    replay_probability = _float32_sigmoid(raw_logit)
    if not math.isclose(
        stored_probability,
        replay_probability,
        rel_tol=0.0,
        abs_tol=RESUME_CPU_SIGMOID_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"resume CPU sigmoid replay changed for {sample_id}"
        )
    probability_decision = (
        stored_probability > CLASSIFICATION_THRESHOLD
    )
    logit_decision = raw_logit > 0.0
    if probability_decision != logit_decision:
        raise ValueError(
            f"resume decision boundary mismatch for {sample_id}"
        )
    return probability_decision


def _validate_ok_row(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    profile_id: str,
    repo_root: Path,
) -> None:
    feature_value = row.get("clip_feature_path")
    feature_hash = row.get("clip_feature_sha256")
    if not isinstance(feature_value, str) or not _valid_sha256(feature_hash):
        raise ValueError(f"resume row {row.get('id')} has invalid feature")
    feature_path = _anchored(Path(feature_value), repo_root)
    _verify_runtime_file(
        feature_path,
        str(feature_hash),
        f"resume CLIP feature {row.get('id')}",
    )
    feature = np.load(feature_path, allow_pickle=False)
    if (
        feature.dtype != np.float32
        or feature.shape != (FEATURE_DIMENSION,)
        or not feature.flags.c_contiguous
        or not np.isfinite(feature).all()
        or row.get("clip_feature_shape") != [FEATURE_DIMENSION]
        or row.get("clip_feature_dtype") != "float32"
    ):
        raise ValueError(f"resume feature contract changed for {row.get('id')}")
    raw_logit = _finite_real(row.get("raw_logit"))
    probability = _finite_real(row.get("probability"))
    ai_score = _finite_real(row.get("ai_score"))
    score = _finite_real(row.get("score"))
    if (
        raw_logit is None
        or probability is None
        or ai_score is None
        or score is None
    ):
        raise ValueError(f"resume scores are invalid for {row.get('id')}")
    decision = _validated_resume_decision(
        raw_logit,
        probability,
        sample_id=row.get("id"),
    )
    expected_classification = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": "official_sigmoid_probability_higher_is_fake",
    }
    expected_t1 = {
        **expected_classification,
        "policy": "official_UFD_CLIP_linear_probe_probability",
    }
    expected_t1.pop("semantics")
    expected_manual = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
    }
    if (
        ai_score != probability
        or score != probability
        or row.get("score_semantics")
        != "official_sigmoid_probability_higher_is_fake"
        or row.get("classification_decision") is not decision
        or row.get("classification_threshold") != CLASSIFICATION_THRESHOLD
        or row.get("classification_threshold_operator")
        != CLASSIFICATION_THRESHOLD_OPERATOR
        or row.get("classification") != expected_classification
        or row.get("t1") != expected_t1
        or row.get("manual_replay") != expected_manual
        or row.get("clip_feature_semantics")
        != "official_CLIP_encode_image_output_before_linear_head"
        or row.get("artifact_paths")
        != {"clip_feature_npy": feature_value}
        or row.get("valid_for_metrics") is not True
    ):
        raise ValueError(f"resume score aliases changed for {row.get('id')}")
    image_path = _anchored(Path(str(expected["canonical_path"])), repo_root)
    _tensor, expected_preprocess = preprocess_image(image_path, profile_id)
    if row.get("preprocess") != expected_preprocess:
        raise ValueError(f"resume preprocess changed for {row.get('id')}")
    latency = _finite_real(row.get("latency_ms"))
    peak = row.get("peak_cuda_memory_bytes")
    if latency is None or latency < 0.0:
        raise ValueError(f"resume latency is invalid for {row.get('id')}")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError(f"resume peak CUDA memory is invalid for {row.get('id')}")


def _validate_resume_rows(
    history: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
    asset_bundle_sha256: str,
    *,
    run_id: str,
    input_manifest_sha256: str,
    profile_id: str,
    pair_visibility: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> None:
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    for row in history:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id not in selected_by_id:
            raise ValueError(f"results contain ID outside selection: {sample_id}")
        expected = selected_by_id[sample_id]
        task_id = str(expected["task_id"])
        visibility = pair_visibility.get(task_id)
        if not isinstance(visibility, Mapping):
            raise ValueError(f"missing expected visibility for {task_id}")
        fraction = _finite_real(row.get("edit_visible_gt_fraction"))
        if (
            row.get("schema_version") != "opensource_result_v1"
            or row.get("run_id") != run_id
            or row.get("run_manifest_fingerprint") != manifest_fingerprint
            or row.get("input_manifest_sha256") != input_manifest_sha256
            or row.get("id") != str(expected["sample_id"])
            or row.get("rank") != int(expected["rank"])
            or row.get("task_id") != task_id
            or row.get("pair_rank") != int(expected["pair_rank"])
            or row.get("domain") != str(expected["domain"])
            or row.get("kind") != str(expected["kind"])
            or row.get("label") != int(expected["label"])
            or row.get("image_path") != str(expected["canonical_path"])
            or row.get("image_sha256") != str(expected["canonical_sha256"])
            or row.get("image_size")
            != [int(expected["width"]), int(expected["height"])]
            or row.get("gt_mask_kind") != str(expected["gt_mask_kind"])
            or row.get("gt_mask_sha256") != expected.get("gt_mask_sha256")
            or row.get("edit_region_xyxy") != expected.get("edit_region_xyxy")
            or row.get("edit_visibility") != visibility["edit_visibility"]
            or fraction is None
            or not math.isclose(
                fraction,
                float(visibility["edit_visible_gt_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or row.get("edit_visibility_evidence")
            != visibility["edit_visibility_evidence"]
            or row.get("model") != MODEL_NAME
            or row.get("model_slug") != MODEL_SLUG
            or row.get("model_arch") != MODEL_ARCH
            or row.get("model_source_commit") != MODEL_SOURCE_COMMIT
            or row.get("asset_bundle_sha256") != asset_bundle_sha256
            or row.get("preprocess_profile") != profile_id
            or row.get("valid_for_t1") is not True
            or row.get("valid_for_t2") is not False
            or row.get("t1_policy")
            != "official_UFD_CLIP_linear_probe_probability"
            or row.get("t2_policy") != "unsupported_whole_image_detector"
            or not isinstance(row.get("completed_at"), str)
            or not row["completed_at"]
        ):
            raise ValueError(f"resume provenance mismatch for {sample_id}")
        if row.get("status") == "ok":
            _validate_ok_row(
                row,
                expected,
                profile_id=profile_id,
                repo_root=repo_root,
            )
        elif row.get("status") == "error":
            forbidden = {
                "raw_logit",
                "probability",
                "ai_score",
                "score",
                "classification",
                "t1",
                "manual_replay",
                "clip_feature_path",
                "preprocess",
                "latency_ms",
                "peak_cuda_memory_bytes",
            }.intersection(row)
            if (
                row.get("valid_for_metrics") is not False
                or not isinstance(row.get("error_type"), str)
                or not row["error_type"]
                or not isinstance(row.get("error_message"), str)
                or not isinstance(row.get("traceback"), str)
                or forbidden
            ):
                raise ValueError(f"invalid resume error row for {sample_id}")
        else:
            raise ValueError(f"invalid resume status for {sample_id}")


def _load_completed_model_audit(
    summary_path: Path,
    *,
    run_id: str,
    manifest_fingerprint: str,
    asset_bundle_sha256: str,
    profile_id: str,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"completed UFD resume is missing summary: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field, expected in {
        "run_id": run_id,
        "run_manifest_fingerprint": manifest_fingerprint,
        "asset_bundle_sha256": asset_bundle_sha256,
        "preprocess_profile": profile_id,
    }.items():
        if summary.get(field) != expected:
            raise ValueError(f"completed UFD summary {field} mismatch")
    audit = summary.get("model_load_audit")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("completed UFD summary has no model_load_audit")
    head_safety = (
        audit.get("assets", {})
        .get("head", {})
        .get("serialization_safety")
    )
    if (
        audit.get("class_module")
        != "_claimforge_ufd_76a0e3e.models.clip_models"
        or audit.get("class_name") != "CLIPModel"
        or audit.get("construction_api")
        != "official models.get_model('CLIP:ViT-L/14')"
        or audit.get("network_blocked") is not True
        or audit.get("urlopen_calls") != 0
        or audit.get("clip_torch_load_fallback_blocked") is not True
        or audit.get("feature_dimension") != FEATURE_DIMENSION
        or audit.get("visual_input_resolution") != MODEL_INPUT_SIZE
        or audit.get("head_parameters") != FEATURE_DIMENSION + 1
        or audit.get("assets", {}).get("bundle_sha256")
        != asset_bundle_sha256
        or audit.get("source", {}).get("commit") != MODEL_SOURCE_COMMIT
        or not isinstance(head_safety, dict)
        or head_safety.get("unsafe_globals") != []
        or head_safety.get("required_unsafe_globals") != []
        or head_safety.get("weights_only") is not True
    ):
        raise ValueError("completed UFD model_load_audit is invalid")
    return audit


def _summarize(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    from eval.opensource.ufd_metrics import summarize_ufd_results

    return summarize_ufd_results(
        rows,
        selected,
        threshold=CLASSIFICATION_THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise ValueError("run-id contains unsafe characters")
    if args.preprocess_profile not in PREPROCESS_PROFILES:
        raise ValueError("unsupported explicit UFD preprocess profile")
    if float(args.classification_threshold) != CLASSIFICATION_THRESHOLD:
        raise ValueError("official UFD probability threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for field in ("source_root", "head_checkpoint", "backbone_checkpoint"):
        if not hasattr(args, field) or getattr(args, field) is None:
            raise ValueError(
                f"--{field.replace('_', '-')} is mandatory; "
                "implicit resolution/network access is forbidden"
            )

    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    source_root = args.source_root.resolve()
    head_checkpoint = args.head_checkpoint.resolve()
    backbone_checkpoint = args.backbone_checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        (
            args.artifact_dir
            if args.artifact_dir is not None
            else Path(
                f"outputs/opensource/universalfakedetect/{args.run_id}"
            )
        ),
        repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.jsonl"
    manifest_path = output_dir / f"{args.run_id}.run_manifest.json"
    summary_path = output_dir / f"{args.run_id}.summary.json"

    release, inputs_path, all_rows = load_release(
        repo_root,
        dataset_manifest_path,
    )
    selected = select_inputs(
        all_rows,
        args.pair_limit,
        getattr(args, "sample_id", None),
    )
    _validate_selected_inputs(selected, repo_root)
    pair_visibility = build_pair_visibility(
        all_rows,
        repo_root,
        args.preprocess_profile,
    )
    source_audit, asset_audit, _head_state = _verify_asset_contract(
        source_root=source_root,
        head_checkpoint=head_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
    )
    configure_determinism(args.seed)
    run_manifest = build_run_manifest(
        args=args,
        repo_root=repo_root,
        dataset_manifest_path=dataset_manifest_path,
        release=release,
        inputs_path=inputs_path,
        selected=selected,
        source_root=source_root,
        head_checkpoint=head_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
        source_audit=source_audit,
        asset_audit=asset_audit,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(manifest_path, run_manifest)
    history = read_jsonl(output_path) if output_path.is_file() else []
    _validate_resume_rows(
        history,
        selected,
        run_manifest["fingerprint"],
        asset_audit["bundle_sha256"],
        run_id=args.run_id,
        input_manifest_sha256=release["inputs_sha256"],
        profile_id=args.preprocess_profile,
        pair_visibility=pair_visibility,
        repo_root=repo_root,
    )
    existing = read_latest_by_id(output_path)
    pending = [
        row
        for row in selected
        if existing.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"UniversalFakeDetect run {args.run_id} "
        f"[{args.preprocess_profile}]: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    model_load_audit: dict[str, Any] | None = None
    if pending:
        model, device, model_load_audit = load_model(
            source_root=source_root,
            head_checkpoint=head_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
            device_name=args.device,
        )
        try:
            for index, input_row in enumerate(pending, start=1):
                sample_id = str(input_row["sample_id"])
                visibility = pair_visibility[str(input_row["task_id"])]
                edit_region = input_row.get("edit_region_xyxy")
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
                    "edit_region_xyxy": (
                        [int(value) for value in edit_region]
                        if isinstance(edit_region, list)
                        else None
                    ),
                    **visibility,
                    "model": MODEL_NAME,
                    "model_slug": MODEL_SLUG,
                    "model_arch": MODEL_ARCH,
                    "model_source_commit": MODEL_SOURCE_COMMIT,
                    "asset_bundle_sha256": asset_audit["bundle_sha256"],
                    "preprocess_profile": args.preprocess_profile,
                    "valid_for_t1": True,
                    "valid_for_t2": False,
                    "t1_policy": (
                        "official_UFD_CLIP_linear_probe_probability"
                    ),
                    "t2_policy": "unsupported_whole_image_detector",
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    image_array, preprocess = preprocess_image(
                        image_path,
                        args.preprocess_profile,
                    )
                    if preprocess["geometry"]["native_size"] != [
                        int(input_row["width"]),
                        int(input_row["height"]),
                    ]:
                        raise ValueError("canonical image dimensions changed")
                    processed, feature, peak_bytes, latency_ms = infer_one(
                        model,
                        device,
                        image_array,
                    )
                    feature_path = (
                        artifact_dir / "clip_features" / f"{sample_id}.npy"
                    )
                    _atomic_save_npy(feature_path, feature)
                    probability = processed["probability"]
                    decision = processed["classification_decision"]
                    classification = {
                        "raw_logit": processed["raw_logit"],
                        "probability": probability,
                        "ai_score": probability,
                        "score": probability,
                        "decision": decision,
                        "threshold": CLASSIFICATION_THRESHOLD,
                        "threshold_operator": (
                            CLASSIFICATION_THRESHOLD_OPERATOR
                        ),
                        "semantics": (
                            "official_sigmoid_probability_higher_is_fake"
                        ),
                    }
                    t1 = {
                        key: value
                        for key, value in classification.items()
                        if key != "semantics"
                    }
                    t1["policy"] = (
                        "official_UFD_CLIP_linear_probe_probability"
                    )
                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "raw_logit": processed["raw_logit"],
                        "probability": probability,
                        "ai_score": probability,
                        "score": probability,
                        "score_semantics": (
                            "official_sigmoid_probability_higher_is_fake"
                        ),
                        "classification_decision": decision,
                        "classification_threshold": (
                            CLASSIFICATION_THRESHOLD
                        ),
                        "classification_threshold_operator": (
                            CLASSIFICATION_THRESHOLD_OPERATOR
                        ),
                        "classification": classification,
                        "t1": t1,
                        "manual_replay": processed["manual_replay"],
                        "clip_feature_path": repo_relative(
                            feature_path,
                            repo_root,
                        ),
                        "clip_feature_sha256": sha256_file(feature_path),
                        "clip_feature_shape": list(feature.shape),
                        "clip_feature_dtype": str(feature.dtype),
                        "clip_feature_semantics": (
                            "official_CLIP_encode_image_output_"
                            "before_linear_head"
                        ),
                        "artifact_paths": {
                            "clip_feature_npy": repo_relative(
                                feature_path,
                                repo_root,
                            )
                        },
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
                detail = (
                    f" score={row['score']:.6f}"
                    f" logit={row['raw_logit']:.6f}"
                    f" visibility={row['edit_visibility']}"
                    f" latency={row['latency_ms']:.1f}ms"
                    if row["status"] == "ok"
                    else f" {row['error_type']}: {row['error_message']}"
                )
                print(
                    f"[{index}/{len(pending)}] "
                    f"{input_row['task_id']} {input_row['kind']}: "
                    f"{row['status']}{detail}",
                    flush=True,
                )
                if row["status"] != "ok" and args.fail_fast:
                    raise RuntimeError(
                        f"UFD failed for {sample_id}: {row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model_load_audit = _load_completed_model_audit(
            summary_path,
            run_id=args.run_id,
            manifest_fingerprint=run_manifest["fingerprint"],
            asset_bundle_sha256=asset_audit["bundle_sha256"],
            profile_id=args.preprocess_profile,
        )

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = _summarize(
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
            "model_arch": MODEL_ARCH,
            "model_source_commit": MODEL_SOURCE_COMMIT,
            "asset_bundle_sha256": asset_audit["bundle_sha256"],
            "head_checkpoint_sha256": HEAD_CHECKPOINT["sha256"],
            "backbone_checkpoint_sha256": BACKBONE_CHECKPOINT["sha256"],
            "preprocess_profile": args.preprocess_profile,
            "input_manifest_sha256": release["inputs_sha256"],
            "run_manifest_fingerprint": run_manifest["fingerprint"],
            "valid_for_t1": True,
            "valid_for_t2": False,
            "model_load_audit": model_load_audit,
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    coverage = summary["coverage"]
    if not args.allow_errors and (
        coverage.get("valid_images") != coverage.get("expected_images")
        or coverage.get("error_images", 0)
        or coverage.get("missing_images", 0)
    ):
        raise RuntimeError(f"incomplete UFD run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--preprocess-profile",
        choices=sorted(PREPROCESS_PROFILES),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/universalfakedetect"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pair-limit", type=int)
    selection.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
