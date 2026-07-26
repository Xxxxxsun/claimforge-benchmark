#!/usr/bin/env python3
"""Run the official FSD v1.2.0 detector on CLAIMFORGE Mouse.

This adapter evaluates the released whole-image detector only (T1).  It pins
the authors' source and four detection assets, never invokes the release's
automatic downloader, persists the raw 960-dimensional float64 FSD, and
independently replays descriptor -> 20 transforms -> GMM -> released z-score
for every image.
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


MODEL_NAME = "FSD"
MODEL_SLUG = "fsd_v1_2_0_official"
MODEL_REPO_URL = (
    "https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25"
)
MODEL_SOURCE_COMMIT = "50f2eae06efdac2e5a33f407ca9a27a2295133ac"
RELEASE_TAG = "v1.2.0"
RELEASE_TAG_COMMIT = "5b317a00251988b5ec5a47317f4d82e5bdfd009d"
PAPER_URL = "https://arxiv.org/abs/2503.21003"
LICENSE_SPDX = "CC-BY-NC-SA-4.0"

FSD_DIMENSION = 960
FRE_KERNEL_SIZE = 15
FRE_BORDER = FRE_KERNEL_SIZE // 2
FSD_NEIGHBORHOOD = 11
FSD_CHANNELS = 8
FSD_SCALES = 3
MAX_SIZE = 1024
PROJECTION_COUNT = 20
PROJECTION_PARAMETERS = 5_267_200
GMM_COMPONENTS = 5
RELEASED_Z_THRESHOLD = -2.0
AI_SCORE_THRESHOLD = 2.0
THRESHOLD_OPERATOR = ">"
RELEASED_THRESHOLD_OPERATOR = "<"
TRAIN_MEAN = -42.25325127289017
TRAIN_STD = 706.0556010649537

SOURCE_FILES = {
    "LICENSE": "30a334fa2dcf04640eda2341b4f5add02103fb81431c5e8ee2053114846b8d7d",
    "README.md": "f4decb89e649632c3f11eae26098891c05103da35fdc36459fee4a33f84c9d12",
    "CITATION.cff": "db7776258148c1cdad9a9ebfbdf9130b127d1159267df18f1dee62c05e666d3e",
    "pyproject.toml": "d3499085168fdae7b1205513e5e7ed2cfb6a2d2c33a6e256ad64cf5963ef14e9",
    "fsd/__init__.py": "cd55645d42b65b390780a99fcf1d1280ae7c5c111a15e6260978dd662a129449",
    "fsd/attribution.py": "a1904196cfb0c26deff4aefdad6aad6b5d1576a906a9f3da095a12c6c7f51861",
    "fsd/detector.py": "8d8ddbce2fc36b03ba5d13df64b442585b13f2dd8b25fa10c7266782150f2b0a",
    "fsd/fre.py": "351d4e41c6a46e55b156870336c192a08f5d4b5c72af169a978f1cb7a7073acb",
    "fsd/fsd_computation.py": "f60d4d6d7f26b31cae53e82b32e5747602cbc242089c793f964ca715af4eabc6",
    "fsd/gmm.py": "a6c476732fe783726d25cf0eb3ea2262bb3c23c69185a0b4d46b18148d9fa35d",
    "fsd/projection.py": "336dd6a59e41c84676846e5a23f77f8674268a739997a75ffcbcf01e80306bba",
    "fsd/weights.py": "6a1413c0a9fc50318729d941c1c4639d0b4e4b6950ae6cc596a5308178fb98ad",
}

WEIGHT_FILES = {
    "config.json": {
        "bytes": 634,
        "sha256": "7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4",
        "kind": "json",
    },
    "fre.pt": {
        "bytes": 9_861,
        "sha256": "d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb",
        "kind": "torch_weights_only",
    },
    "gmm.pt": {
        "bytes": 14_786_229,
        "sha256": "0f9fa030a3d5816266d0329fd0fb614b65e322d4bda6d083613c713bfe9bc829",
        "kind": "torch_weights_only",
    },
    "fsd_transforms.pt": {
        "bytes": 42_177_409,
        "sha256": "1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699",
        "kind": "torch_weights_only",
    },
}

EXPECTED_CONFIG = {
    "fre": {
        "in_channels": 1,
        "out_channels": FSD_CHANNELS,
        "kernel_size": FRE_KERNEL_SIZE,
        "weights_file": "fre.pt",
    },
    "fsd": {
        "kernel_size": FSD_NEIGHBORHOOD,
        "num_scales": FSD_SCALES,
        "max_size": MAX_SIZE,
        "resize_mode": "resize_and_crop",
    },
    "gmm": {
        "n_components": GMM_COMPONENTS,
        "covariance_type": "tied",
        "weights_file": "gmm.pt",
    },
    "transforms": {"weights_file": "fsd_transforms.pt"},
    "scoring": {
        "train_mean": TRAIN_MEAN,
        "train_std": TRAIN_STD,
        "default_threshold": RELEASED_Z_THRESHOLD,
    },
    "attribution": {
        "weights_file": "attribution_transforms.pt",
        "source_gmms_file": "source_gmms.pt",
    },
}

SOURCE_TAG_DRIFT = {
    "release_tag": RELEASE_TAG,
    "release_tag_commit": RELEASE_TAG_COMMIT,
    "source_commit": MODEL_SOURCE_COMMIT,
    "commits_ahead_of_tag": 1,
    "changed_files": [
        "fsd/__init__.py",
        "fsd/attribution.py",
        "fsd/weights.py",
    ],
    "detection_math_changed_files": [],
    "interpretation": (
        "the pinned main commit is one commit after the v1.2.0 tag; the "
        "detection math files used by score() are unchanged"
    ),
}

PAPER_RELEASE_DRIFT = {
    "paper": {
        "url": PAPER_URL,
        "venue": "CVPR 2025",
        "fre_neighborhood": "11x11",
        "per_image_solver": (
            "AdamW with plateau scheduling for at most 10000 iterations"
        ),
        "scoring_chain": "Phi_directly_to_GMM",
        "reported_gmm_parameters_approx": 2_000,
        "threshold_note": (
            "the paper and supplement discuss an older normalized threshold; "
            "that operating point is not reused by this adapter"
        ),
    },
    "release": {
        "tag": RELEASE_TAG,
        "public_inference_timeline": "post-paper_2026_release",
        "fre_weight_shape": [8, 1, 15, 15],
        "per_image_solver": "float64_KKT_constrained_least_squares",
        "pre_gmm_transforms": {
            "count": PROJECTION_COUNT,
            "architecture": "residual_MLP_960_128_128_960",
            "parameters": PROJECTION_PARAMETERS,
        },
        "threshold_source": "v1.2.0_config_json",
        "decision": "released_z_score < -2.0",
        "training_or_evaluation_scripts_in_repository": False,
    },
    "evaluation_claim": (
        "reproduces the pinned v1.2.0 inference release API only; it does not "
        "claim paper-protocol parity, independently verify real-only release "
        "training, or reproduce the paper's reported 0.960 AUC"
    ),
}

DEFAULT_DATASET_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "Forensic-Self-Descriptions-CVPR25-50f2eae"
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


def _manifest_fingerprint(value: dict[str, Any]) -> str:
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
    if sample_id is not None and pair_limit is not None:
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
    if row.get("kind") != "forged" or row.get("label") != 1:
        raise ValueError(f"invalid kind/label contract for {sample_id}")
    if row.get("gt_mask_kind") != "exact_diff":
        raise ValueError(f"invalid forged GT kind for {sample_id}")
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
        raise ValueError(f"GT mask {sample_id} contains non-binary pixels")
    positive = int(np.count_nonzero(pixels == 255))
    if positive != int(row.get("gt_positive_pixels", -1)) or positive <= 0:
        raise ValueError(f"GT positive-pixel count mismatch for {sample_id}")
    return np.asarray(pixels, dtype=np.uint8)


def _validate_selected_inputs(
    selected: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    for row in selected:
        sample_id = str(row["sample_id"])
        image_path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            image_path,
            str(row["canonical_sha256"]),
            f"canonical input {sample_id}",
        )
        with Image.open(image_path) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"canonical dimensions changed for {sample_id}")
        _validate_gt_row(row, repo_root)


def _verify_repository_contract(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing FSD source tree: {source_root}")
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(
            f"FSD source commit mismatch: {commit} != {MODEL_SOURCE_COMMIT}"
        )
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"FSD tracked source tree is dirty: {dirty[:1000]}")
    for relative, digest in SOURCE_FILES.items():
        _verify_runtime_file(
            source_root / relative,
            digest,
            f"FSD source {relative}",
        )
    tag_commit = _git_value(source_root, "rev-parse", f"{RELEASE_TAG}^{{commit}}")
    ahead = _git_value(
        source_root,
        "rev-list",
        "--count",
        f"{RELEASE_TAG}..HEAD",
    )
    changed_text = _git_value(
        source_root,
        "diff",
        "--name-only",
        f"{RELEASE_TAG}..HEAD",
    )
    changed = sorted(changed_text.splitlines()) if changed_text else []
    if (
        tag_commit != RELEASE_TAG_COMMIT
        or ahead != "1"
        or changed != sorted(SOURCE_TAG_DRIFT["changed_files"])
    ):
        raise ValueError("FSD source/release-tag drift contract changed")
    return {
        "repo_url": MODEL_REPO_URL,
        "root": str(source_root),
        "commit": commit,
        "tracked_dirty": False,
        "source_files": {
            name: {
                "path": str(source_root / name),
                "sha256": digest,
            }
            for name, digest in SOURCE_FILES.items()
        },
        "source_tag_drift": SOURCE_TAG_DRIFT,
    }


def _payload_schema(value: Any) -> Any:
    """Return a compact, deterministic description of a safe torch payload."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required for inference
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
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
    if isinstance(value, (list, tuple)):
        schemas = [_payload_schema(item) for item in value]
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "length": len(value),
            "items": schemas,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    raise ValueError(
        "unexpected object in weights-only payload: "
        f"{type(value).__module__}.{type(value).__name__}"
    )


def _validate_weight_payloads(payloads: Mapping[str, Any]) -> None:
    import torch

    fre = payloads["fre.pt"]
    if (
        not isinstance(fre, Mapping)
        or list(fre) != ["w", "one_middle"]
        or not isinstance(fre["w"], torch.Tensor)
        or tuple(fre["w"].shape) != (8, 1, 15, 15)
        or fre["w"].dtype != torch.float32
        or not isinstance(fre["one_middle"], torch.Tensor)
        or tuple(fre["one_middle"].shape) != (225,)
    ):
        raise ValueError("unexpected fre.pt schema")
    gmm = payloads["gmm.pt"]
    if (
        not isinstance(gmm, Mapping)
        or int(gmm.get("n_components", -1)) != GMM_COMPONENTS
        or gmm.get("covariance_type") != "tied"
        or tuple(gmm["means_"].shape) != (5, FSD_DIMENSION)
        or tuple(gmm["weights_"].shape) != (5,)
        or tuple(gmm["covariances_"].shape)
        != (FSD_DIMENSION, FSD_DIMENSION)
        or tuple(gmm["precisions_cholesky_"].shape)
        != (FSD_DIMENSION, FSD_DIMENSION)
        or any(
            gmm[key].dtype != torch.float64
            for key in (
                "means_",
                "weights_",
                "covariances_",
                "precisions_cholesky_",
            )
        )
    ):
        raise ValueError("unexpected gmm.pt schema")
    transforms = payloads["fsd_transforms.pt"]
    states = transforms.get("transforms") if isinstance(transforms, Mapping) else None
    if (
        not isinstance(transforms, Mapping)
        or transforms.get("config") != {"dim": 960, "hidden": 128}
        or not isinstance(states, list)
        or len(states) != PROJECTION_COUNT
        or any(not isinstance(state, (dict, OrderedDict)) for state in states)
    ):
        raise ValueError("unexpected fsd_transforms.pt schema")
    parameters = sum(
        int(tensor.numel())
        for state in states
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )
    if parameters != PROJECTION_PARAMETERS:
        raise ValueError(
            f"transform parameter count changed: {parameters}"
        )


def _verify_weights_contract(weights_dir: Path) -> dict[str, Any]:
    import torch

    if not weights_dir.is_dir():
        raise FileNotFoundError(f"missing explicit FSD weights-dir: {weights_dir}")
    files: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for filename, contract in WEIGHT_FILES.items():
        path = weights_dir / filename
        _verify_runtime_file(path, str(contract["sha256"]), f"FSD {filename}")
        if path.stat().st_size != int(contract["bytes"]):
            raise ValueError(
                f"FSD {filename} size mismatch: "
                f"{path.stat().st_size} != {contract['bytes']}"
            )
        audit: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": contract["kind"],
        }
        if filename.endswith(".pt"):
            unsafe = sorted(
                torch.serialization.get_unsafe_globals_in_checkpoint(path)
            )
            audit["serialization_safety"] = {
                "preflight": (
                    "torch.serialization.get_unsafe_globals_in_checkpoint"
                ),
                "unsafe_globals": unsafe,
                "required_unsafe_globals": [],
                "weights_only": True,
            }
            if unsafe:
                raise ValueError(
                    f"FSD {filename} contains unsafe globals: {unsafe}"
                )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payloads[filename] = payload
            audit["payload_schema"] = _payload_schema(payload)
        files[filename] = audit
    config = json.loads(
        (weights_dir / "config.json").read_text(encoding="utf-8")
    )
    if config != EXPECTED_CONFIG:
        raise ValueError("FSD v1.2.0 config.json contract changed")
    _validate_weight_payloads(payloads)
    bundle_value = [
        {
            "filename": filename,
            "bytes": int(WEIGHT_FILES[filename]["bytes"]),
            "sha256": str(WEIGHT_FILES[filename]["sha256"]),
        }
        for filename in sorted(WEIGHT_FILES)
    ]
    return {
        "provider": "official_github_release",
        "release_tag": RELEASE_TAG,
        "release_url": f"{MODEL_REPO_URL}/releases/tag/{RELEASE_TAG}",
        "weights_dir": str(weights_dir),
        "explicit_weights_dir_required": True,
        "automatic_download_used": False,
        "files": files,
        "config": config,
        "bundle_sha256": hashlib.sha256(
            stable_json(bundle_value).encode("utf-8")
        ).hexdigest(),
    }


def _verify_static_contract(
    *,
    source_root: Path,
    weights_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _verify_repository_contract(source_root)
    weights = _verify_weights_contract(weights_dir)
    return source, weights


def _load_official_package(source_root: Path) -> Any:
    package_name = "_claimforge_fsd_50f2eae"
    existing = sys.modules.get(package_name)
    if existing is not None:
        origin = Path(str(getattr(existing, "__file__", ""))).resolve()
        if origin != (source_root / "fsd" / "__init__.py").resolve():
            raise RuntimeError(f"pinned FSD module collision: {origin}")
        return existing
    init_path = source_root / "fsd" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned FSD package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise
    if getattr(module, "__version__", None) != "1.2.0":
        raise RuntimeError("pinned FSD package version changed")
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


def load_detector(
    *,
    source_root: Path,
    weights_dir: Path,
    device_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    source_audit, weights_audit = _verify_static_contract(
        source_root=source_root,
        weights_dir=weights_dir,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    package = _load_official_package(source_root)
    detector_class = getattr(package, "FSDDetector", None)
    if not isinstance(detector_class, type):
        raise RuntimeError("pinned source has no FSDDetector class")
    # Passing this path is mandatory: a None value invokes network download.
    detector = detector_class.load(
        weights_dir=weights_dir,
        device=str(device),
        threshold=RELEASED_Z_THRESHOLD,
        attribution=False,
    )
    if detector.config != EXPECTED_CONFIG:
        raise ValueError("loaded detector config differs from audited config")
    if (
        float(detector.threshold) != RELEASED_Z_THRESHOLD
        or float(detector.train_mean) != TRAIN_MEAN
        or float(detector.train_std) != TRAIN_STD
        or len(detector.projections) != PROJECTION_COUNT
        or tuple(detector.fre.conv.w.shape) != (8, 1, 15, 15)
    ):
        raise ValueError("loaded FSD detector architecture/calibration changed")
    audit = {
        "source": source_audit,
        "weights": weights_audit,
        "class_module": detector.__class__.__module__,
        "class_name": detector.__class__.__name__,
        "load_api": "FSDDetector.load",
        "weights_dir_argument": str(weights_dir),
        "weights_dir_was_explicit": True,
        "automatic_download_used": False,
        "attribution_loaded": False,
        "released_threshold": RELEASED_Z_THRESHOLD,
        "projection_count": len(detector.projections),
    }
    return detector, device, audit


def compute_preprocess_geometry(width: int, height: int) -> dict[str, Any]:
    if width <= 2 * FRE_BORDER or height <= 2 * FRE_BORDER:
        raise ValueError("image is too small for the released FRE border trim")
    residual_width = width - 2 * FRE_BORDER
    residual_height = height - 2 * FRE_BORDER
    scale = MAX_SIZE / min(residual_height, residual_width)
    resized_height = round(residual_height * scale)
    resized_width = round(residual_width * scale)
    crop_height = min(MAX_SIZE, resized_height)
    crop_width = min(MAX_SIZE, resized_width)
    start_y = (resized_height - crop_height) // 2
    start_x = (resized_width - crop_width) // 2
    end_x, end_y = start_x + crop_width, start_y + crop_height
    native_crop = [
        FRE_BORDER + start_x * residual_width / resized_width,
        FRE_BORDER + start_y * residual_height / resized_height,
        FRE_BORDER + end_x * residual_width / resized_width,
        FRE_BORDER + end_y * residual_height / resized_height,
    ]
    scale_sizes = [
        [
            math.floor(crop_width / (2**level)),
            math.floor(crop_height / (2**level)),
        ]
        for level in range(FSD_SCALES)
    ]
    return {
        "decoder": "Pillow.Image.open_then_convert_L",
        "exif_transpose": False,
        "icc_conversion": False,
        "native_size": [width, height],
        "grayscale_dtype": "uint8",
        "grayscale_range": [0, 255],
        "fre": {
            "kernel_size": FRE_KERNEL_SIZE,
            "output_channels": FSD_CHANNELS,
            "padding": FRE_BORDER,
            "border_each_side": FRE_BORDER,
            "post_trim_size": [residual_width, residual_height],
        },
        "resize": {
            "enabled": True,
            "mode": "torch.nn.functional.interpolate_bilinear",
            "align_corners": False,
            "antialias": False,
            "scale_factor_nominal": scale,
            "source_size": [residual_width, residual_height],
            "destination_size": [resized_width, resized_height],
            "rounding": "python_round",
            "rule": "short_side_to_1024",
        },
        "center_crop": {
            "enabled": True,
            "source_size": [resized_width, resized_height],
            "start_xy": [start_x, start_y],
            "size": [crop_width, crop_height],
            "end_xy": [end_x, end_y],
        },
        "effective_native_crop_xyxy": native_crop,
        "pixel_center_mapping": (
            "d=(native_index-border+0.5)*resized_size/"
            "post_trim_size-0.5"
        ),
        "scales": {
            "count": FSD_SCALES,
            "sizes": scale_sizes,
            "mode": "torch_bilinear",
            "align_corners": False,
            "antialias": False,
        },
        "descriptor": {
            "shape": [FSD_DIMENSION],
            "dtype": "float64",
            "neighborhood": [FSD_NEIGHBORHOOD, FSD_NEIGHBORHOOD],
            "solver": "float64_KKT_constrained_least_squares",
            "lambda_regularization": 1e-5,
        },
    }


def _intersection_xyxy(
    first: list[float],
    second: list[float],
) -> list[float] | None:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def _box_visibility(
    edit_region: list[int],
    native_crop: list[float],
) -> dict[str, Any]:
    box = [float(value) for value in edit_region]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("edit_region_xyxy has non-positive area")
    intersection = _intersection_xyxy(box, native_crop)
    edit_area = (box[2] - box[0]) * (box[3] - box[1])
    visible_area = (
        0.0
        if intersection is None
        else (intersection[2] - intersection[0])
        * (intersection[3] - intersection[1])
    )
    fraction = min(1.0, max(0.0, visible_area / edit_area))
    category = (
        "none"
        if fraction == 0.0
        else "full"
        if math.isclose(fraction, 1.0, abs_tol=1e-12, rel_tol=0.0)
        else "partial"
    )
    return {
        "edit_region_xyxy": edit_region,
        "effective_native_crop_xyxy": native_crop,
        "intersection_xyxy": intersection,
        "edit_area": edit_area,
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
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    positive_y, positive_x = np.nonzero(gt == 255)
    total = int(positive_x.size)
    if total <= 0:
        raise ValueError("forged exact-diff GT has no positive pixels")
    resize = preprocess["resize"]
    crop = preprocess["center_crop"]
    residual_width, residual_height = resize["source_size"]
    resized_width, resized_height = resize["destination_size"]
    start_x, start_y = crop["start_xy"]
    crop_width, crop_height = crop["size"]
    destination_x = (
        (positive_x.astype(np.float64) - FRE_BORDER + 0.5)
        * resized_width
        / residual_width
        - 0.5
    )
    destination_y = (
        (positive_y.astype(np.float64) - FRE_BORDER + 0.5)
        * resized_height
        / residual_height
        - 0.5
    )
    visible = (
        (destination_x >= start_x)
        & (destination_x < start_x + crop_width)
        & (destination_y >= start_y)
        & (destination_y < start_y + crop_height)
    )
    visible_count = int(np.count_nonzero(visible))
    fraction = visible_count / total
    category = (
        "none"
        if visible_count == 0
        else "full"
        if visible_count == total
        else "partial"
    )
    return {
        "category": category,
        "visible_fraction": fraction,
        "positive_pixels": total,
        "visible_positive_pixel_centers": visible_count,
        "forged_sample_id": str(forged_row["sample_id"]),
        "basis": (
            "forged_exact_diff_positive_pixel_centers_mapped_after_FRE_trim_"
            "with_align_corners_false_formula_into_official_center_crop"
        ),
        "formula": (
            "d=(native_index-7+0.5)*resized_size/(native_size-14)-0.5; "
            "visible iff crop_start <= d < crop_start+1024"
        ),
    }


def build_pair_visibility(
    all_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
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
            raise ValueError(f"pair geometry mismatch: {task_id}")
        gt = _validate_gt_row(forged, repo_root)
        assert gt is not None
        preprocess = compute_preprocess_geometry(
            int(forged["width"]),
            int(forged["height"]),
        )
        gt_diagnostic = _gt_visibility(forged, gt, preprocess)
        edit_region = forged.get("edit_region_xyxy")
        if (
            not isinstance(edit_region, list)
            or len(edit_region) != 4
            or any(not isinstance(value, int) for value in edit_region)
        ):
            raise ValueError(f"invalid edit region for {task_id}")
        box_diagnostic = _box_visibility(
            edit_region,
            list(preprocess["effective_native_crop_xyxy"]),
        )
        result[task_id] = {
            "edit_visibility": gt_diagnostic["category"],
            "edit_visible_gt_fraction": gt_diagnostic["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": gt_diagnostic,
                "edit_box": box_diagnostic,
            },
        }
    return result


def infer_one(
    detector: Any,
    device: Any,
    image_path: Path,
) -> tuple[dict[str, Any], np.ndarray, int | None, float]:
    """Run the official API once, capture its descriptor, and replay scoring."""

    import torch

    detector_module = sys.modules.get(detector.__class__.__module__)
    if detector_module is None:
        raise RuntimeError("official detector module is not loaded")
    original_compute = getattr(detector_module, "compute_fsd", None)
    if not callable(original_compute):
        raise RuntimeError("official detector compute_fsd binding is missing")
    captured: list[Any] = []

    def capture_descriptor(*args: Any, **kwargs: Any) -> Any:
        descriptor = original_compute(*args, **kwargs)
        captured.append(descriptor.detach().clone())
        return descriptor

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with mock.patch.object(
        detector_module,
        "compute_fsd",
        side_effect=capture_descriptor,
    ):
        official = detector.score(image_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    if len(captured) != 1:
        raise RuntimeError(
            f"official FSDDetector.score called compute_fsd {len(captured)} times"
        )
    descriptor_tensor = captured[0]
    if (
        not isinstance(descriptor_tensor, torch.Tensor)
        or descriptor_tensor.device.type != "cpu"
        or descriptor_tensor.dtype != torch.float64
        or tuple(descriptor_tensor.shape) != (FSD_DIMENSION,)
        or not bool(torch.isfinite(descriptor_tensor).all())
    ):
        raise ValueError(
            "official raw FSD descriptor violates float64 [960] CPU contract"
        )
    descriptor = np.ascontiguousarray(descriptor_tensor.numpy())

    projection_module = sys.modules.get(
        detector.__class__.__module__.rsplit(".", 1)[0] + ".projection"
    )
    apply_projections = getattr(projection_module, "apply_projections", None)
    if not callable(apply_projections):
        raise RuntimeError("official apply_projections binding is missing")
    with torch.inference_mode():
        projected = apply_projections(
            descriptor_tensor.to(device).unsqueeze(0),
            detector.projections,
        )
        manual_raw = float(detector.gmm.score_samples(projected).item())
    manual_z = (manual_raw - float(detector.train_mean)) / float(
        detector.train_std
    )
    manual_ai = -manual_z
    official_raw = float(official.raw_score)
    official_z = float(official.z_score)
    values = [manual_raw, manual_z, manual_ai, official_raw, official_z]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("FSD scoring produced non-finite values")
    # The same official transforms and GMM receive the same descriptor twice.
    # A mismatch is evidence of statefulness or a changed release contract.
    if manual_raw != official_raw or manual_z != official_z:
        raise ValueError(
            "manual descriptor->projection->GMM replay differs from "
            "FSDDetector.score"
        )
    released_decision = official_z < RELEASED_Z_THRESHOLD
    ai_decision = manual_ai > AI_SCORE_THRESHOLD
    if (
        bool(official.is_fake) != released_decision
        or released_decision != ai_decision
        or float(official.threshold) != RELEASED_Z_THRESHOLD
    ):
        raise ValueError("released and CLAIMFORGE strict decisions differ")
    return (
        {
            "raw_likelihood": official_raw,
            "released_z_score": official_z,
            "ai_score": manual_ai,
            "released_is_fake": released_decision,
            "classification_decision": ai_decision,
            "manual_replay": {
                "raw_likelihood": manual_raw,
                "released_z_score": manual_z,
                "ai_score": manual_ai,
                "released_is_fake": manual_z < RELEASED_Z_THRESHOLD,
                "classification_decision": ai_decision,
                "official_raw_exact_match": True,
                "official_z_exact_match": True,
                "compute_fsd_calls": 1,
            },
        },
        descriptor,
        peak_bytes,
        latency_ms,
    )


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

    requested = torch.device(device_name)
    cuda_active = requested.type == "cuda" and torch.cuda.is_available()
    device_index = (
        requested.index
        if requested.index is not None
        else torch.cuda.current_device()
        if cuda_active
        else None
    )
    resolved_cuda_device = (
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
            "numpy": str(np.__version__),
            "Pillow": _package_version("Pillow"),
        },
        "accelerator": {
            "requested_device": str(requested),
            "device_type": requested.type,
            "device_index": device_index,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_version": str(torch.__version__),
            "torch_distribution_version": _package_version("torch"),
            "torch_cuda": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu_name": (
                torch.cuda.get_device_name(resolved_cuda_device)
                if resolved_cuda_device is not None
                else None
            ),
            "gpu_capability": (
                list(torch.cuda.get_device_capability(resolved_cuda_device))
                if resolved_cuda_device is not None
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
    weights_dir: Path,
    source_audit: dict[str, Any],
    weights_audit: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    runtime = _runtime_contract(args.device)
    metrics_path = Path(__file__).with_name("whole_image_metrics.py")
    adapter_paths = [Path(__file__), Path(__file__).with_name("common.py")]
    if metrics_path.is_file():
        adapter_paths.append(metrics_path)
    immutable = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "model": {
            "name": MODEL_NAME,
            "slug": MODEL_SLUG,
            "task_support": {"t1": True, "t2": False},
            "repo_url": MODEL_REPO_URL,
            "source_root": str(source_root),
            "source_commit": MODEL_SOURCE_COMMIT,
            "release_tag": RELEASE_TAG,
            "license": {
                "spdx": LICENSE_SPDX,
                "commercial_use": False,
                "share_alike": True,
            },
            "source_audit": source_audit,
            "weights": weights_audit,
        },
        "source_tag_drift": SOURCE_TAG_DRIFT,
        "paper_release_drift": PAPER_RELEASE_DRIFT,
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
            "official_api": "FSDDetector.score",
            "weights_resolution": (
                "explicit_cli_weights_dir_no_automatic_download"
            ),
            "input_chain": [
                "Pillow path decode",
                "convert L",
                "FRE 15x15 then trim 7-pixel border",
                "bilinear resize short residual side to 1024",
                "center crop 1024x1024",
                "three bilinear scales",
                "float64 KKT constrained least-squares",
                "20 released residual MLP transforms",
                "released tied-covariance GMM",
            ],
            "descriptor": {
                "shape": [FSD_DIMENSION],
                "dtype": "float64",
                "persisted_for_every_ok_image": True,
            },
            "scores": {
                "raw_likelihood": "released_GMM_log_likelihood",
                "released_z_score": (
                    f"(raw_likelihood-{TRAIN_MEAN})/{TRAIN_STD}"
                ),
                "ai_score": "-released_z_score",
            },
            "classification": {
                "released_threshold": RELEASED_Z_THRESHOLD,
                "released_operator": RELEASED_THRESHOLD_OPERATOR,
                "ai_score_threshold": AI_SCORE_THRESHOLD,
                "ai_score_operator": THRESHOLD_OPERATOR,
                "strict": True,
            },
            "manual_replay": (
                "each descriptor is replayed through official "
                "apply_projections and GMM and must exactly equal "
                "FSDDetector.score"
            ),
            "edit_visibility": {
                "category_field": "edit_visibility",
                "fraction_field": "edit_visible_gt_fraction",
                "pair_level": True,
                "real_policy": (
                    "copy counterfactual forged exact-diff GT visibility"
                ),
                "formula": (
                    "d=(native_index-7+0.5)*resized_size/"
                    "(native_size-14)-0.5; crop_start<=d<crop_start+1024"
                ),
            },
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "runtime_contract": runtime,
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
        "environment": runtime,
    }


def _write_or_validate_run_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                "existing FSD run manifest fingerprint differs; use a new run-id"
            )
        return
    atomic_write_json(path, manifest)


def _validate_ok_row(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    descriptor_value = row.get("raw_descriptor_path")
    descriptor_hash = row.get("raw_descriptor_sha256")
    if not isinstance(descriptor_value, str) or not _valid_sha256(
        descriptor_hash
    ):
        raise ValueError(f"resume row {row.get('id')} has invalid descriptor")
    descriptor_path = _anchored(Path(descriptor_value), repo_root)
    _verify_runtime_file(
        descriptor_path,
        str(descriptor_hash),
        f"resume descriptor {row.get('id')}",
    )
    descriptor = np.load(descriptor_path, allow_pickle=False)
    if (
        descriptor.dtype != np.float64
        or tuple(descriptor.shape) != (FSD_DIMENSION,)
        or not np.isfinite(descriptor).all()
        or row.get("raw_descriptor_shape") != [FSD_DIMENSION]
        or row.get("raw_descriptor_dtype") != "float64"
    ):
        raise ValueError(f"resume descriptor contract changed for {row.get('id')}")
    raw = _finite_real(row.get("raw_likelihood"))
    z_score = _finite_real(row.get("released_z_score"))
    ai_score = _finite_real(row.get("ai_score"))
    if raw is None or z_score is None or ai_score is None:
        raise ValueError(f"resume scores are non-finite for {row.get('id')}")
    replay_z = (raw - TRAIN_MEAN) / TRAIN_STD
    if z_score != replay_z or ai_score != -z_score:
        raise ValueError(f"resume score calibration mismatch for {row.get('id')}")
    released_decision = z_score < RELEASED_Z_THRESHOLD
    ai_decision = ai_score > AI_SCORE_THRESHOLD
    expected_classification = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": ai_decision,
        "threshold": AI_SCORE_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "semantics": "higher_is_more_AI_negative_released_z",
    }
    expected_t1 = {
        "score": ai_score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": ai_decision,
        "threshold": AI_SCORE_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "policy": "released_FSD_whole_image_score_sign_inverted",
    }
    expected_manual = {
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "ai_score": ai_score,
        "released_is_fake": released_decision,
        "classification_decision": ai_decision,
        "official_raw_exact_match": True,
        "official_z_exact_match": True,
        "compute_fsd_calls": 1,
    }
    if (
        row.get("score") != ai_score
        or row.get("score_semantics") != "negative_released_FSD_z_score"
        or row.get("released_is_fake") is not released_decision
        or row.get("released_threshold") != RELEASED_Z_THRESHOLD
        or row.get("released_threshold_operator")
        != RELEASED_THRESHOLD_OPERATOR
        or row.get("classification_decision") is not ai_decision
        or row.get("classification_threshold") != AI_SCORE_THRESHOLD
        or row.get("classification_threshold_operator") != THRESHOLD_OPERATOR
        or row.get("classification") != expected_classification
        or row.get("t1") != expected_t1
        or row.get("manual_replay") != expected_manual
        or row.get("valid_for_t1") is not True
        or row.get("valid_for_t2") is not False
        or row.get("valid_for_metrics") is not True
    ):
        raise ValueError(f"resume decision contract changed for {row.get('id')}")
    if row.get("gt_mask_sha256") != expected.get("gt_mask_sha256"):
        raise ValueError(f"resume GT hash mismatch for {row.get('id')}")
    expected_preprocess = compute_preprocess_geometry(
        int(expected["width"]),
        int(expected["height"]),
    )
    if row.get("preprocess") != expected_preprocess:
        raise ValueError(f"resume preprocess changed for {row.get('id')}")
    if (
        row.get("raw_descriptor_semantics")
        != "official_compute_fsd_before_released_transforms"
        or row.get("artifact_paths")
        != {"raw_descriptor_npy": descriptor_value}
    ):
        raise ValueError(f"resume artifact aliases changed for {row.get('id')}")
    latency = row.get("latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
    ):
        raise ValueError(f"resume latency is invalid for {row.get('id')}")
    peak = row.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError(f"resume peak memory is invalid for {row.get('id')}")


def _validate_resume_rows(
    history: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
    weights_bundle_sha256: str,
    *,
    run_id: str,
    input_manifest_sha256: str,
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
        expected_edit_region = expected.get("edit_region_xyxy")
        recorded_visible_fraction = _finite_real(
            row.get("edit_visible_gt_fraction")
        )
        if (
            row.get("run_manifest_fingerprint") != manifest_fingerprint
            or row.get("run_id") != run_id
            or row.get("input_manifest_sha256") != input_manifest_sha256
            or row.get("schema_version") != "opensource_result_v1"
            or row.get("rank") != int(expected["rank"])
            or row.get("task_id") != task_id
            or row.get("pair_rank") != int(expected["pair_rank"])
            or row.get("domain") != str(expected["domain"])
            or row.get("kind") != str(expected["kind"])
            or row.get("label") != int(expected["label"])
            or row.get("image_path") != str(expected["canonical_path"])
            or row.get("image_sha256") != expected["canonical_sha256"]
            or row.get("image_size")
            != [int(expected["width"]), int(expected["height"])]
            or row.get("gt_mask_kind") != str(expected["gt_mask_kind"])
            or row.get("gt_mask_sha256") != expected.get("gt_mask_sha256")
            or row.get("edit_region_xyxy") != expected_edit_region
            or row.get("edit_visibility") != visibility["edit_visibility"]
            or recorded_visible_fraction is None
            or not math.isclose(
                recorded_visible_fraction,
                float(visibility["edit_visible_gt_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or row.get("edit_visibility_evidence")
            != visibility["edit_visibility_evidence"]
            or row.get("model") != MODEL_NAME
            or row.get("model_slug") != MODEL_SLUG
            or row.get("weights_bundle_sha256") != weights_bundle_sha256
            or row.get("model_source_commit") != MODEL_SOURCE_COMMIT
            or row.get("release_tag") != RELEASE_TAG
            or row.get("valid_for_t1") is not True
            or row.get("valid_for_t2") is not False
            or row.get("t1_policy")
            != "released_FSD_v1.2.0_whole_image_z_score"
            or row.get("t2_policy") != "unsupported_whole_image_detector"
            or not isinstance(row.get("completed_at"), str)
            or not row["completed_at"]
        ):
            raise ValueError(f"resume provenance mismatch for {sample_id}")
        if row.get("status") == "ok":
            _validate_ok_row(row, expected, repo_root=repo_root)
        elif row.get("status") == "error":
            if (
                row.get("valid_for_metrics") is not False
                or not isinstance(row.get("error_type"), str)
                or not row["error_type"]
                or not isinstance(row.get("error_message"), str)
                or not isinstance(row.get("traceback"), str)
                or any(
                    field in row
                    for field in (
                        "ai_score",
                        "released_z_score",
                        "raw_likelihood",
                        "raw_descriptor_path",
                    )
                )
            ):
                raise ValueError(f"invalid resume error row for {sample_id}")
        else:
            raise ValueError(f"invalid resume status for {sample_id}")


def _load_completed_model_audit(
    summary_path: Path,
    *,
    run_id: str,
    manifest_fingerprint: str,
    weights_bundle_sha256: str,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"completed FSD resume is missing summary: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field, expected in {
        "run_id": run_id,
        "run_manifest_fingerprint": manifest_fingerprint,
        "weights_bundle_sha256": weights_bundle_sha256,
    }.items():
        if summary.get(field) != expected:
            raise ValueError(f"completed resume summary {field} mismatch")
    audit = summary.get("model_load_audit")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("completed resume summary has no model_load_audit")
    if (
        audit.get("class_name") != "FSDDetector"
        or audit.get("class_module") != "_claimforge_fsd_50f2eae.detector"
        or audit.get("load_api") != "FSDDetector.load"
        or audit.get("weights_dir_was_explicit") is not True
        or audit.get("automatic_download_used") is not False
        or audit.get("attribution_loaded") is not False
        or audit.get("released_threshold") != RELEASED_Z_THRESHOLD
        or audit.get("projection_count") != PROJECTION_COUNT
        or audit.get("source", {}).get("commit") != MODEL_SOURCE_COMMIT
        or audit.get("weights", {}).get("bundle_sha256")
        != weights_bundle_sha256
        or audit.get("weights", {}).get("release_tag") != RELEASE_TAG
        or audit.get("weights", {}).get("explicit_weights_dir_required")
        is not True
        or audit.get("weights", {}).get("automatic_download_used") is not False
        or audit.get("weights_dir_argument")
        != audit.get("weights", {}).get("weights_dir")
    ):
        raise ValueError("completed resume model_load_audit identity is invalid")
    for asset in ("fre.pt", "gmm.pt", "fsd_transforms.pt"):
        safety = (
            audit.get("weights", {})
            .get("files", {})
            .get(asset, {})
            .get("serialization_safety")
        )
        if (
            not isinstance(safety, dict)
            or safety.get("preflight")
            != "torch.serialization.get_unsafe_globals_in_checkpoint"
            or safety.get("unsafe_globals") != []
            or safety.get("required_unsafe_globals") != []
            or safety.get("weights_only") is not True
        ):
            raise ValueError("completed resume has invalid weights safety audit")
    return audit


def _summarize(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    from eval.opensource.whole_image_metrics import summarize_whole_image_results

    return summarize_whole_image_results(
        rows,
        selected,
        threshold=AI_SCORE_THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise ValueError("run-id contains unsafe characters")
    if float(args.classification_threshold) != AI_SCORE_THRESHOLD:
        raise ValueError("official FSD ai_score threshold must be 2.0")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not hasattr(args, "weights_dir") or args.weights_dir is None:
        raise ValueError("--weights-dir is mandatory; auto-download is forbidden")

    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    source_root = args.source_root.resolve()
    weights_dir = args.weights_dir.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        (
            args.artifact_dir
            if args.artifact_dir is not None
            else Path(f"outputs/opensource/fsd/{args.run_id}")
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
    pair_visibility = build_pair_visibility(all_rows, repo_root)
    source_audit, weights_audit = _verify_static_contract(
        source_root=source_root,
        weights_dir=weights_dir,
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
        weights_dir=weights_dir,
        source_audit=source_audit,
        weights_audit=weights_audit,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(manifest_path, run_manifest)
    history = read_jsonl(output_path) if output_path.is_file() else []
    _validate_resume_rows(
        history,
        selected,
        run_manifest["fingerprint"],
        weights_audit["bundle_sha256"],
        run_id=args.run_id,
        input_manifest_sha256=release["inputs_sha256"],
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
        f"FSD run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    detector = None
    model_load_audit: dict[str, Any] | None = None
    if pending:
        detector, device, model_load_audit = load_detector(
            source_root=source_root,
            weights_dir=weights_dir,
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
                    "model_source_commit": MODEL_SOURCE_COMMIT,
                    "release_tag": RELEASE_TAG,
                    "weights_bundle_sha256": weights_audit["bundle_sha256"],
                    "valid_for_t1": True,
                    "valid_for_t2": False,
                    "t1_policy": "released_FSD_v1.2.0_whole_image_z_score",
                    "t2_policy": "unsupported_whole_image_detector",
                }
                try:
                    image_path = _anchored(
                        Path(str(input_row["canonical_path"])),
                        repo_root,
                    )
                    preprocess = compute_preprocess_geometry(
                        int(input_row["width"]),
                        int(input_row["height"]),
                    )
                    processed, descriptor, peak_bytes, latency_ms = infer_one(
                        detector,
                        device,
                        image_path,
                    )
                    descriptor_path = (
                        artifact_dir / "raw_descriptors" / f"{sample_id}.npy"
                    )
                    _atomic_save_npy(descriptor_path, descriptor)
                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": processed["ai_score"],
                        "score_semantics": "negative_released_FSD_z_score",
                        "raw_likelihood": processed["raw_likelihood"],
                        "released_z_score": processed["released_z_score"],
                        "ai_score": processed["ai_score"],
                        "released_is_fake": processed["released_is_fake"],
                        "released_threshold": RELEASED_Z_THRESHOLD,
                        "released_threshold_operator": (
                            RELEASED_THRESHOLD_OPERATOR
                        ),
                        "classification_decision": processed[
                            "classification_decision"
                        ],
                        "classification_threshold": AI_SCORE_THRESHOLD,
                        "classification_threshold_operator": (
                            THRESHOLD_OPERATOR
                        ),
                        "classification": {
                            "score": processed["ai_score"],
                            "raw_likelihood": processed["raw_likelihood"],
                            "released_z_score": processed["released_z_score"],
                            "decision": processed["classification_decision"],
                            "threshold": AI_SCORE_THRESHOLD,
                            "threshold_operator": THRESHOLD_OPERATOR,
                            "semantics": "higher_is_more_AI_negative_released_z",
                        },
                        "t1": {
                            "score": processed["ai_score"],
                            "raw_likelihood": processed["raw_likelihood"],
                            "released_z_score": processed["released_z_score"],
                            "decision": processed["classification_decision"],
                            "threshold": AI_SCORE_THRESHOLD,
                            "threshold_operator": THRESHOLD_OPERATOR,
                            "policy": (
                                "released_FSD_whole_image_score_sign_inverted"
                            ),
                        },
                        "manual_replay": processed["manual_replay"],
                        "raw_descriptor_path": repo_relative(
                            descriptor_path,
                            repo_root,
                        ),
                        "raw_descriptor_sha256": sha256_file(descriptor_path),
                        "raw_descriptor_shape": list(descriptor.shape),
                        "raw_descriptor_dtype": str(descriptor.dtype),
                        "raw_descriptor_semantics": (
                            "official_compute_fsd_before_released_transforms"
                        ),
                        "artifact_paths": {
                            "raw_descriptor_npy": repo_relative(
                                descriptor_path,
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
                    f" ai_score={row['ai_score']:.6f}"
                    f" z={row['released_z_score']:.6f}"
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
                        f"FSD failed for {sample_id}: {row['error_message']}"
                    )
        finally:
            del detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model_load_audit = _load_completed_model_audit(
            summary_path,
            run_id=args.run_id,
            manifest_fingerprint=run_manifest["fingerprint"],
            weights_bundle_sha256=weights_audit["bundle_sha256"],
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
            "model_source_commit": MODEL_SOURCE_COMMIT,
            "release_tag": RELEASE_TAG,
            "weights_bundle_sha256": weights_audit["bundle_sha256"],
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
        coverage["valid_images"] != coverage["expected_images"]
        or coverage["error_images"]
        or coverage["missing_images"]
    ):
        raise RuntimeError(f"incomplete FSD run: {coverage}")
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
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Explicit v1.2.0 asset directory; automatic download is forbidden.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/fsd"),
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
        default=AI_SCORE_THRESHOLD,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
