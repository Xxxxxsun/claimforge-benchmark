#!/usr/bin/env python3
"""Run the pinned official OmniAID-DINO v2 detector on Balanced250.

This v2 orchestration layer leaves the audited Mouse-v1 implementation
unchanged.  It executes the official Space DINOv3/MoE whole-image classifier
on the 1,775-image Balanced250 score cache (or a frozen smoke/single
selection), preserving all six auditable arrays in a local-only, gitignored
NPZ artifact.

OmniAID is T1-only.  Direct full-canvas resize visibility is an input
diagnostic, not a predicted mask, localization output, or joint T1/T2 score.
Scientific metrics belong to ``analyze_omniaid_balanced.py`` and the shared
Balanced250 metric implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import run_omniaid as legacy
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
    select_inputs,
)
from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


RUN_MANIFEST_SCHEMA = "omniaid_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "omniaid_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "omniaid_balanced_runtime_summary_v2"
CPU_PREFLIGHT_SCHEMA = "omniaid_balanced_cpu_preflight_v1"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = Path("results/opensource/omniaid")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/omniaid")
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_SPACE_ROOT = legacy.DEFAULT_SPACE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_OMNIAID_CONFIG = legacy.DEFAULT_OMNIAID_CONFIG
DEFAULT_FORMAL_RUN_ID = "omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727"
DEFAULT_SMOKE_RUN_ID_A = (
    "omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_a_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_b_20260727"
)
DEFAULT_SMOKE_LIMIT = 5
DEFAULT_SEED = legacy.MODEL_SEED
FROZEN_PYTHONHASHSEED = "0"
FROZEN_PROFILE = legacy.PREPROCESS_PROFILE
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MINIMUM_CUDA_FREE_BYTES = 6 * 1024**3

FROZEN_PYTHON_EXECUTABLE = Path("/root/.cache/claimforge/venvs/omniaid/bin/python")
FROZEN_VENV_PREFIX = Path("/root/.cache/claimforge/venvs/omniaid")
FROZEN_PYTHONPYCACHEPREFIX = Path(
    "/root/.cache/claimforge/pycache/omniaid-balanced-v2-empty"
)
FROZEN_PYVENV_CONFIG_SHA256 = (
    "5666a60cb8fef584ccb503bb9c5d6f6df51d86dfe759304236c29814d5107231"
)
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torch_distribution": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "torchvision_distribution": "0.23.0.dev20250627+cu128",
    "transformers": "4.57.3",
    "numpy": "2.2.6",
    "Pillow": "12.0.0",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.2",
    "safetensors": "0.5.2",
    "setuptools": "79.0.1",
}
FROZEN_RUNTIME_MODULE_FILES = {
    "torch": "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d",
    "torchvision": ("ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73"),
    "transformers": (
        "7eb0743ed843f24c1d4e8b8daf4b18249cb81403b4571a70c00be0a4c0a67bd4"
    ),
    "numpy": "6ae17b070c0f70a8e3cad89a510a256942e5a1f37ea5feb120cec167ed2a6236",
    "PIL": "43828e12947b4bf5ec8f7d1fbceb2f47de311295f8294b15794c1a54fd5f53cd",
    "huggingface_hub": (
        "017af8861f5bde565c6ce9f231457f63a2579f643588c09560f9ead4560f84e4"
    ),
}
FROZEN_DINOV3_MODULE_FILES = {
    "configuration_dinov3_vit.py": (
        "1ac7cb889e2314e8cadf9b0bed43d42cfb05dce6d730f719be4380c8a10a8a46"
    ),
    "modeling_dinov3_vit.py": (
        "79f9cc140c1eca19d992285cecfe57faf0f1c470e6f2b296bf01f7fd94473705"
    ),
    "modular_dinov3_vit.py": (
        "c2e7a9ed2faf064fe111c0f980af5a881144f65563d5ffe41b903bccba07dadb"
    ),
}

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
    threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
)

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}
FORMAL_SELECTED_ROWS_SHA256 = (
    "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
)
FORMAL_SELECTED_IDS_SHA256 = (
    "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
)
SMOKE5X7_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

PREPROCESS_CONTRACT = {
    "profile_id": FROZEN_PROFILE,
    "decoder": "Pillow.Image.open.convert_RGB",
    "exif_transpose": False,
    "icc_conversion": False,
    "resize": {
        "size": [legacy.MODEL_INPUT_SIZE, legacy.MODEL_INPUT_SIZE],
        "implementation": "torchvision.transforms.Resize_list_hw",
        "interpolation": "PIL_BILINEAR_default",
        "antialias": True,
        "preserve_aspect_ratio": False,
    },
    "crop": None,
    "tensor": "torchvision_ToTensor_uint8_div_255_float32",
    "normalization_mean": list(legacy.IMAGENET_MEAN),
    "normalization_std": list(legacy.IMAGENET_STD),
    "router_mode": "Auto (Router)",
    "manual_weights": None,
    "batch_size": 1,
    "autocast": False,
    "dtype": "float32",
}
FROZEN_PREPROCESS_CONTRACT = PREPROCESS_CONTRACT

ARTIFACT_FILE_BYTES = 9_848
ARTIFACT_CONTRACT = {
    "format": "NumPy NPZ, allow_pickle=False, ZIP_STORED",
    "keys": list(legacy.ARTIFACT_SCHEMA),
    "file_bytes": ARTIFACT_FILE_BYTES,
    "zip_members": {
        "pooler_output.npy": 4_224,
        "class_logits.npy": 136,
        "routing_feature.npy": 4_224,
        "semantic_top_k_indices.npy": 144,
        "semantic_top_k_gates.npy": 136,
        "final_gates.npy": 152,
    },
    "arrays": {
        key: {
            "shape": list(shape),
            "dtype": np.dtype(dtype).name,
            "nbytes": int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize,
        }
        for key, (shape, dtype) in legacy.ARTIFACT_SCHEMA.items()
    },
    "finite_float_arrays": True,
    "head_softmax_router_scatter_replay": True,
    "visibility": "local_only_gitignored_output",
}

TASK_SCOPE = {
    "primary_task": "T1_whole_image_AIGC_detection",
    "valid_for_t1": True,
    "valid_for_t2": False,
    "localization_output": None,
    "native_dense_output": False,
}

MODEL_CONTRACT = {
    "name": legacy.MODEL_NAME,
    "slug": legacy.MODEL_SLUG,
    "architecture": legacy.MODEL_ARCH,
    "repository": legacy.MODEL_REPO_URL,
    "source_commit": legacy.MODEL_SOURCE_COMMIT,
    "space": legacy.MODEL_SPACE_URL,
    "space_commit": legacy.MODEL_SPACE_COMMIT,
    "model_repository": legacy.MODEL_HF_URL,
    "model_revision": legacy.MODEL_HF_REVISION,
    "paper": legacy.PAPER_URL,
    "checkpoint": legacy.CHECKPOINT,
    "omniaid_config": legacy.OMNIAID_CONFIG,
    "dinov3_base": legacy.DINO_BASE,
    "license": {
        "release_record": legacy.LICENSE_RECORD,
        "commercial_clearance": False,
        "research_evaluation_only": True,
        "checkpoint_redistributed_by_benchmark": False,
    },
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_omniaid_balanced.py",
    "eval/opensource/analyze_omniaid_balanced.py",
    "eval/opensource/run_omniaid.py",
    "eval/opensource/analyze_omniaid_run.py",
    "eval/opensource/omniaid_metrics.py",
    "eval/opensource/ufd_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
)

IMMUTABLE_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "model",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "visibility_census",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "execution_model_load",
        "execution_official_golden",
        "artifact_contract",
        "artifact_policy",
        "outputs",
    }
)

CPU_GOLDEN_SAMPLE_ID = "2c80d38ac19c2d3b76950996"
CPU_GOLDEN_INPUT_PATH = (
    "outputs/opensource/balanced250_v1/images/" f"{CPU_GOLDEN_SAMPLE_ID}.jpg"
)
CPU_GOLDEN_IMAGE_SHA256 = (
    "12607f3cdada1480038f3d506146cdc1fa0c1c50034afda5e3a5f175433e716b"
)
CPU_GOLDEN_DECODED_RGB_SHA256 = (
    "5a4747a6e3a8313f8c9ec3dde2504bb53184666276d7e54dc5fab53ca0e7194b"
)
CPU_GOLDEN_RESIZED_RGB_SHA256 = (
    "32e2459e254b9181930ae93c2587959630807d99040bdde18f1273c0b43b0419"
)
CPU_GOLDEN_TENSOR_SHA256 = (
    "a8f204ec4f42f41c43c272ac5dab3799a0b5c80fece394955779de91bd6ced6e"
)
CPU_GOLDEN_ARRAY_SHA256 = {
    "pooler_output": (
        "fd014afddd6c62f524e4b8db2ec377e53661aa0699664ab2aa142f342ae02a54"
    ),
    "class_logits": (
        "e73c405b6284e9a2cacdb73bc7e129af37197f682cb9115b8518b6bb4c5b62dd"
    ),
    "routing_feature": (
        "e28ea53999015dcfb0f50af0c56432b2f07a9999b37d9591efd2cc26ddd18416"
    ),
    "semantic_top_k_indices": (
        "b38c612d3a2f2d9bd6f0e638bd328e6c24d1ae6d3867ffbd6c69a98d19b3b292"
    ),
    "semantic_top_k_gates": (
        "a0ac08baaf1dfa047a3e99c770f85c7df01621ee659bac2dc7572b9b2888c9b1"
    ),
    "final_gates": ("d232fa8f75b2dff1819ce374ff27301dbb33407e7e4e4faeabe257f347aa8a38"),
}
CPU_GOLDEN_ARTIFACT_SHA256 = (
    "43d43823ba66031847427d1a127a84bbb7afb5fa1a78c5ee07fdcdebccad443f"
)
CPU_GOLDEN_CLASS_LOGITS = [1.6302306652069092, -1.0535809993743896]
CPU_GOLDEN_RAW_LOGIT_MARGIN = -2.683811664581299
CPU_GOLDEN_PROBABILITY = 0.06393537670373917
CPU_GOLDEN_TOP_K_INDICES = [3, 2]
CPU_GOLDEN_TOP_K_GATES = [0.6834401488304138, 0.3165598511695862]
CPU_GOLDEN_FINAL_GATES = [
    0.0,
    0.0,
    0.3165598511695862,
    0.6834401488304138,
    0.0,
    1.0,
]

CPU_OFFICIAL_GOLDEN_CASES = (
    {
        "path": "examples/real_0.jpg",
        "probability": 0.23996387422084808,
        "logits": [0.8654325008392334, -0.2874450087547302],
        "final_gates": [
            0.4042757749557495,
            0.0,
            0.5957242846488953,
            0.0,
            0.0,
            1.0,
        ],
        "array_sha256": {
            "pooler_output": "8d28cc368e1c21880b45f0361acbb46d869e98095cdd1f8c9656f8a875f30287",
            "class_logits": "eee78c88cde70e714daff29dc181ef7a8379ca3a6b443eee9ee71400aeff27cd",
            "routing_feature": "788a4110e727299fb75fda470df9c94f8c33f35ebc9f1058f9a2b903cb7e6638",
            "semantic_top_k_indices": "b1535c7783ea8829b6b0cf67704539798b4d16c39bf0bfe09494c5d9f12eee30",
            "semantic_top_k_gates": "3ca5b742005ab9229059a447910488a44120f3e38dd6f268f685874671a79851",
            "final_gates": "7eba339327493dee0106b10c49c151f56e965c036fa8b890778186c22638dfed",
        },
    },
    {
        "path": "examples/real_1.jpg",
        "probability": 0.07805726677179337,
        "logits": [1.2692692279815674, -1.1997711658477783],
        "final_gates": [
            0.8894946575164795,
            0.0,
            0.0,
            0.1105053648352623,
            0.0,
            1.0,
        ],
        "array_sha256": {
            "pooler_output": "044bfa4837e254973fb5df4da5de28cd4f58dd5b103afbfd6c72e2229d384dbc",
            "class_logits": "cf336c2140362f8263d5a40ce1d002d9fde268cf3ec44fcabf442a27e4f10691",
            "routing_feature": "b5f70dc5c3df4296126d53008eb8363498cb876643bfe764d409e9bdc6f8f6c9",
            "semantic_top_k_indices": "96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a",
            "semantic_top_k_gates": "c0ce3d9ea75a0d446cbd340882ae467f1bfb4338e2254508de6761c312a7afaf",
            "final_gates": "307b066a3f5e14a131133d4b3cabeb5f9677127ae6a1dd75f01170080613c8ac",
        },
    },
    {
        "path": "examples/fake_0.jpg",
        "probability": 0.8572262525558472,
        "logits": [-1.0232189893722534, 0.7692219614982605],
        "final_gates": [
            0.910078227519989,
            0.0,
            0.089921735227108,
            0.0,
            0.0,
            1.0,
        ],
        "array_sha256": {
            "pooler_output": "acb487ff96400831cdc8b0aa00daef9de0bfb255d44aa62fc6da21607cfe2875",
            "class_logits": "28a72afea570336d50c13779f8d89e74a92ea2e8eb097b5ac3b690ff981947bf",
            "routing_feature": "dd24a22f696e9e4b8507892b8b1ce6b30618483098e9cca687705c1a9b8e3b81",
            "semantic_top_k_indices": "c571327cb01ac1de6972713cbf6cc1fc3c2cab8b581ee0bc3fe6d8b56963fd5b",
            "semantic_top_k_gates": "6a1dade3593d0f19934337db921ae6391279ae9944162f2ef9786357c62240bc",
            "final_gates": "9eab9136a2363c71d213a8d59a699f46a0ada8ccd8285cb43c25f855d1ecaafb",
        },
    },
    {
        "path": "examples/fake_1.jpg",
        "probability": 0.6094993352890015,
        "logits": [-0.39048415422439575, 0.05472393333911896],
        "final_gates": [
            0.0,
            0.0,
            0.061757173389196396,
            0.9382428526878357,
            0.0,
            1.0,
        ],
        "array_sha256": {
            "pooler_output": "429d05da1c9a98eedd3a0d6b060a00e91921bbeecf0dec151283af8e74ce9f8c",
            "class_logits": "0f136a95a6c1c2afa88f11ec1063a4861ba58e1990d8c9755784d59b2661ce64",
            "routing_feature": "2b8e6aded6c218ccfc2fdc4e07ad175f87f80dc9d3f07b5594ff03acd56086be",
            "semantic_top_k_indices": "b38c612d3a2f2d9bd6f0e638bd328e6c24d1ae6d3867ffbd6c69a98d19b3b292",
            "semantic_top_k_gates": "8ea995e3ecd17a44dd2cfa8a9f812ebc0263daaf590082ff8cbb041e76540d87",
            "final_gates": "fb1e593c44797b4e76a05e35640893cf1a9c4c955e2a20c5cf20e7e6f9cc8be4",
        },
    },
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if list(arrays) != list(legacy.ARTIFACT_SCHEMA):
        raise ValueError("OmniAID artifact key order changed")
    handle = io.BytesIO()
    np.savez(
        handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()}
    )
    return handle.getvalue()


def _same_json_type_and_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_same_json(left: Any, right: Any, label: str) -> None:
    if not _same_json_type_and_value(left, right):
        raise ValueError(f"{label} changed")


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value)
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run-id must be one safe ASCII path component (max 160 chars)")
    return value


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"missing or unsafe OmniAID Balanced source: {path}"
            )
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
        or any("pair_rank" in row for row in selected)
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("formal OmniAID Balanced250 selection drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        isinstance(per_condition_limit, bool)
        or not isinstance(per_condition_limit, int)
        or per_condition_limit != DEFAULT_SMOKE_LIMIT
    ):
        raise ValueError(
            "smoke per-condition-limit must be exactly " f"{DEFAULT_SMOKE_LIMIT}"
        )
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=per_condition_limit,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if (
            condition in Capability.WHOLE_IMAGE_T1.conditions
            and counts[condition] < per_condition_limit
        ):
            sample_id = str(panel_row["sample_id"])
            source = inputs_by_id.get(sample_id)
            if source is None or source.get("panel") is not True:
                raise ValueError("smoke panel has a dangling/non-panel input")
            selected.append(source)
            counts[condition] += 1
    expected = {condition: per_condition_limit for condition in BALANCED_CONDITIONS}
    selected.sort(key=lambda row: int(row["rank"]))
    if (
        dict(counts) != expected
        or any("pair_rank" in row for row in selected)
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != SMOKE5X7_SELECTED_IDS_SHA256
    ):
        raise ValueError("frozen OmniAID smoke selection drifted")
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if release.release_kind != "balanced250":
        raise ValueError("OmniAID v2 requires the Balanced250 release")
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode does not accept input selectors")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode does not accept sample-id")
        return _smoke_selection(
            release,
            DEFAULT_SMOKE_LIMIT if per_condition_limit is None else per_condition_limit,
        )
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode does not accept per-condition-limit")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            sample_id=sample_id,
        )
        selected = select_inputs(release, spec)
        if len(selected) != 1 or "pair_rank" in selected[0]:
            raise ValueError("single OmniAID selection drifted")
        return spec, selected
    raise ValueError(f"unsupported inference mode {mode!r}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_cublas_workspace() -> str:
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_CONFIG}, got {current!r}"
        )
    return CUBLAS_WORKSPACE_CONFIG


def _startup_isolation_contract() -> dict[str, Any]:
    expected_prefix = Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX))
    raw_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    actual_prefix = (
        Path(os.path.abspath(raw_prefix)) if isinstance(raw_prefix, str) else None
    )
    if (
        os.environ.get("PYTHONHASHSEED") != FROZEN_PYTHONHASHSEED
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("NO_ALBUMENTATIONS_UPDATE") != "1"
        or actual_prefix != expected_prefix
        or not expected_prefix.is_absolute()
        or expected_prefix.is_symlink()
        or (expected_prefix.exists() and not expected_prefix.is_dir())
        or (expected_prefix.is_dir() and any(expected_prefix.iterdir()))
        or sys.dont_write_bytecode is not True
        or Path(os.path.abspath(str(sys.pycache_prefix))) != expected_prefix
    ):
        raise RuntimeError(
            "OmniAID startup isolation requires PYTHONHASHSEED=0, "
            "PYTHONDONTWRITEBYTECODE=1, NO_ALBUMENTATIONS_UPDATE=1, "
            f"and an absolute empty PYTHONPYCACHEPREFIX={expected_prefix}"
        )
    return {
        "PYTHONHASHSEED": FROZEN_PYTHONHASHSEED,
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPYCACHEPREFIX": str(expected_prefix),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(expected_prefix),
        "pycache_prefix_initially_empty": True,
    }


def _venv_contract() -> dict[str, Any]:
    prefix = Path(os.path.abspath(sys.prefix))
    base_prefix = Path(os.path.abspath(sys.base_prefix))
    expected_prefix = Path(os.path.abspath(FROZEN_VENV_PREFIX))
    config_path = expected_prefix / "pyvenv.cfg"
    if (
        prefix != expected_prefix
        or base_prefix != Path("/usr")
        or config_path.is_symlink()
        or not config_path.is_file()
        or sha256_file(config_path) != FROZEN_PYVENV_CONFIG_SHA256
    ):
        raise RuntimeError("OmniAID virtual-environment contract drifted")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        normalized = key.strip()
        if separator != "=" or not normalized or normalized in values:
            raise RuntimeError("OmniAID pyvenv.cfg is malformed")
        values[normalized] = value.strip()
    expected_values = {
        "home": "/usr/bin",
        "include-system-site-packages": "true",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": (
            "/usr/bin/python -m venv --system-site-packages "
            "/root/.cache/claimforge/venvs/omniaid"
        ),
    }
    if values != expected_values:
        raise RuntimeError("OmniAID pyvenv.cfg values drifted")
    return {
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "pyvenv_cfg_path": str(config_path),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }


def _runtime_versions_and_modules() -> tuple[
    dict[str, str],
    dict[str, dict[str, str]],
]:
    import PIL
    import huggingface_hub
    import torch
    import torchvision
    import transformers

    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_distribution": str(_package_version("torch")),
        "torchvision": str(torchvision.__version__),
        "torchvision_distribution": str(_package_version("torchvision")),
        "transformers": str(transformers.__version__),
        "numpy": str(np.__version__),
        "Pillow": str(PIL.__version__),
        "huggingface-hub": str(huggingface_hub.__version__),
        "tokenizers": str(_package_version("tokenizers")),
        "safetensors": str(_package_version("safetensors")),
        "setuptools": str(_package_version("setuptools")),
    }
    if versions != FROZEN_RUNTIME_VERSIONS:
        raise RuntimeError(
            "OmniAID dedicated runtime version drifted: "
            f"expected {FROZEN_RUNTIME_VERSIONS}, got {versions}"
        )
    executable = Path(os.path.abspath(sys.executable))
    if executable != Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE)):
        raise RuntimeError(
            f"OmniAID must run in its frozen environment: "
            f"{FROZEN_PYTHON_EXECUTABLE}"
        )
    modules: dict[str, dict[str, str]] = {}
    for name, expected in FROZEN_RUNTIME_MODULE_FILES.items():
        module = importlib.import_module(name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise RuntimeError(f"OmniAID runtime module has no file: {name}")
        path = Path(raw_path).resolve()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"OmniAID runtime module changed: {name}")
        modules[name] = {"path": str(path), "sha256": expected}
    dino_root = Path(transformers.__file__).resolve().parent / "models" / "dinov3_vit"
    for filename, expected in FROZEN_DINOV3_MODULE_FILES.items():
        path = dino_root / filename
        key = f"transformers.models.dinov3_vit.{filename[:-3]}"
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"OmniAID DINOv3 runtime module changed: {filename}")
        modules[key] = {"path": str(path), "sha256": expected}
    return versions, modules


def configure_runtime(
    device_text: str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Freeze the runtime and resolve only CPU or an explicit CUDA device."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != DEFAULT_SEED:
        raise ValueError(f"OmniAID seed must be exactly {DEFAULT_SEED}")
    startup = _startup_isolation_contract()
    cublas_workspace = _configure_cublas_workspace()
    import torch

    versions, modules = _runtime_versions_and_modules()
    venv = _venv_contract()
    if device_text == "cpu":
        device = torch.device("cpu")
    elif re.fullmatch(r"cuda:[0-9]+", device_text):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_text)
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device_text}")
        torch.cuda.set_device(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        if int(free_bytes) < MINIMUM_CUDA_FREE_BYTES:
            raise RuntimeError(
                f"{device} has only {int(free_bytes)} free bytes; "
                f"OmniAID requires at least {MINIMUM_CUDA_FREE_BYTES}"
            )
    else:
        raise ValueError("device must be 'cpu' or an explicit 'cuda:N'")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(legacy.CPU_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    runtime: dict[str, Any] = {
        "device": str(device),
        "python": {
            "implementation": platform.python_implementation(),
            "version": versions["python"],
            "executable": str(Path(os.path.abspath(sys.executable))),
        },
        "venv": venv,
        "startup_isolation": startup,
        "platform": platform.platform(),
        "versions": versions,
        "module_files": modules,
        "seed": seed,
        "preprocess_profile": FROZEN_PROFILE,
        "dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "cpu_threads": legacy.CPU_THREADS,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": cublas_workspace,
        "cudnn": {
            "enabled": bool(torch.backends.cudnn.enabled),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        },
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "minimum_cuda_free_bytes": MINIMUM_CUDA_FREE_BYTES,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": int(device.index),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    validate_runtime_contract(runtime, label="configured runtime")
    return device, runtime


def validate_runtime_contract(
    value: Mapping[str, Any],
    *,
    label: str = "runtime",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    device = value.get("device")
    if not isinstance(device, str) or (
        device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None
    ):
        raise ValueError(f"{label}.device is unsupported")
    expected_keys = {
        "device",
        "python",
        "venv",
        "startup_isolation",
        "platform",
        "versions",
        "module_files",
        "seed",
        "preprocess_profile",
        "dtype",
        "batch_size",
        "autocast",
        "cpu_threads",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cublas_workspace_config",
        "cudnn",
        "matmul_allow_tf32",
        "float32_matmul_precision",
        "minimum_cuda_free_bytes",
    }
    if device.startswith("cuda:"):
        expected_keys.add("cuda")
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set changed")
    python_record = value.get("python")
    expected_python = {
        "implementation": "CPython",
        "version": FROZEN_RUNTIME_VERSIONS["python"],
        "executable": str(Path(os.path.abspath(FROZEN_PYTHON_EXECUTABLE))),
    }
    if not isinstance(python_record, Mapping) or dict(python_record) != expected_python:
        raise ValueError(f"{label}.python changed")
    if dict(value.get("venv", {})) != {
        "prefix": str(Path(os.path.abspath(FROZEN_VENV_PREFIX))),
        "base_prefix": "/usr",
        "pyvenv_cfg_path": str(
            Path(os.path.abspath(FROZEN_VENV_PREFIX)) / "pyvenv.cfg"
        ),
        "pyvenv_cfg_sha256": FROZEN_PYVENV_CONFIG_SHA256,
        "include_system_site_packages": True,
    }:
        raise ValueError(f"{label}.venv changed")
    if dict(value.get("versions", {})) != FROZEN_RUNTIME_VERSIONS:
        raise ValueError(f"{label}.versions changed")
    modules = value.get("module_files")
    expected_hashes = {
        **FROZEN_RUNTIME_MODULE_FILES,
        **{
            f"transformers.models.dinov3_vit.{name[:-3]}": digest
            for name, digest in FROZEN_DINOV3_MODULE_FILES.items()
        },
    }
    if (
        not isinstance(modules, Mapping)
        or set(modules) != set(expected_hashes)
        or any(
            not isinstance(modules[name], Mapping)
            or modules[name].get("sha256") != digest
            or not isinstance(modules[name].get("path"), str)
            for name, digest in expected_hashes.items()
        )
    ):
        raise ValueError(f"{label}.module_files changed")
    startup = value.get("startup_isolation")
    if (
        not isinstance(startup, Mapping)
        or startup.get("PYTHONHASHSEED") != FROZEN_PYTHONHASHSEED
        or startup.get("PYTHONDONTWRITEBYTECODE") != "1"
        or startup.get("NO_ALBUMENTATIONS_UPDATE") != "1"
        or startup.get("PYTHONPYCACHEPREFIX")
        != str(Path(os.path.abspath(FROZEN_PYTHONPYCACHEPREFIX)))
        or startup.get("python_dont_write_bytecode") is not True
        or startup.get("pycache_prefix_initially_empty") is not True
    ):
        raise ValueError(f"{label}.startup_isolation changed")
    cudnn = value.get("cudnn")
    if (
        not isinstance(cudnn, Mapping)
        or dict(cudnn)
        != {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        }
        or value.get("seed") != DEFAULT_SEED
        or value.get("preprocess_profile") != FROZEN_PROFILE
        or value.get("dtype") != "float32"
        or value.get("batch_size") != 1
        or value.get("autocast") is not False
        or value.get("cpu_threads") != legacy.CPU_THREADS
        or value.get("deterministic_algorithms_enabled") is not True
        or value.get("deterministic_algorithms_warn_only") is not False
        or value.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG
        or value.get("matmul_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("minimum_cuda_free_bytes") != MINIMUM_CUDA_FREE_BYTES
        or not isinstance(value.get("platform"), str)
        or not value["platform"]
    ):
        raise ValueError(f"{label} deterministic contract changed")
    if device.startswith("cuda:"):
        cuda = value.get("cuda")
        if (
            not isinstance(cuda, Mapping)
            or set(cuda)
            != {
                "runtime",
                "device_index",
                "device_name",
                "total_memory_bytes",
                "capability",
            }
            or cuda.get("runtime") != "12.8"
            or cuda.get("device_index") != int(device.split(":", 1)[1])
            or not isinstance(cuda.get("device_name"), str)
            or not cuda["device_name"]
            or isinstance(cuda.get("total_memory_bytes"), bool)
            or not isinstance(cuda.get("total_memory_bytes"), int)
            or cuda["total_memory_bytes"] <= 0
            or not isinstance(cuda.get("capability"), list)
            or len(cuda["capability"]) != 2
        ):
            raise ValueError(f"{label}.cuda changed")
    return value


def verify_assets(
    *,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    omniaid_config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Any],
    dict[str, Any],
]:
    for path, label in (
        (source_root, "source root"),
        (space_root, "Space root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(f"missing or unsafe OmniAID {label}: {path}")
    for path, label in (
        (checkpoint_path, "checkpoint"),
        (omniaid_config_path, "config"),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or unsafe OmniAID {label}: {path}")
    source = legacy.verify_source(source_root, space_root)
    assets, state, config = legacy.verify_assets(
        checkpoint_path,
        omniaid_config_path,
    )
    return source, assets, state, config


def _official_cpu_projection(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("OmniAID CPU official golden cases are missing")
    return [
        {
            "path": case.get("path"),
            "probability": case.get("fake_probability"),
            "logits": case.get("logits"),
            "final_gates": case.get("final_expert_gates"),
            "array_sha256": case.get("array_sha256"),
        }
        for case in cases
    ]


def _golden_artifact_record(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    payload = _npz_bytes(arrays)
    return {
        "artifact_bytes": len(payload),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "array_sha256": {key: _array_sha256(value) for key, value in arrays.items()},
    }


def _balanced_cpu_golden(
    *,
    model: Any,
    device: Any,
    repo_root: Path,
) -> dict[str, Any]:
    path = _anchored(Path(CPU_GOLDEN_INPUT_PATH), repo_root)
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != CPU_GOLDEN_IMAGE_SHA256
    ):
        raise ValueError("OmniAID Balanced CPU golden input changed")
    records: list[dict[str, Any]] = []
    for _index in range(2):
        image, preprocess = legacy.preprocess_image(path)
        scoring, arrays, peak, _latency = legacy.infer_one(
            model,
            device,
            image,
        )
        records.append(
            {
                "preprocess": preprocess,
                "artifact": _golden_artifact_record(arrays),
                "class_logits": scoring["class_logits"],
                "raw_logit_margin": scoring["raw_logit_margin"],
                "probability": scoring["probability"],
                "ai_score": scoring["ai_score"],
                "classification_decision": scoring["classification_decision"],
                "semantic_top_k_indices": scoring["semantic_top_k_indices"],
                "semantic_top_k_gates": scoring["semantic_top_k_gates"],
                "final_expert_gates": scoring["final_expert_gates"],
                "manual_replay": scoring["manual_replay"],
                "peak_cuda_memory_bytes": 0 if peak is None else int(peak),
            }
        )
    first, second = records
    if first != second:
        raise ValueError("OmniAID Balanced CPU golden is not byte-exact")
    preprocess = first["preprocess"]
    if (
        preprocess.get("decoded_rgb_sha256") != CPU_GOLDEN_DECODED_RGB_SHA256
        or preprocess.get("resized_rgb_sha256") != CPU_GOLDEN_RESIZED_RGB_SHA256
        or preprocess.get("tensor_sha256") != CPU_GOLDEN_TENSOR_SHA256
        or preprocess.get("native_width") != 1800
        or preprocess.get("native_height") != 1350
        or first["artifact"]
        != {
            "artifact_bytes": ARTIFACT_FILE_BYTES,
            "artifact_sha256": CPU_GOLDEN_ARTIFACT_SHA256,
            "array_sha256": CPU_GOLDEN_ARRAY_SHA256,
        }
        or first["class_logits"] != CPU_GOLDEN_CLASS_LOGITS
        or first["raw_logit_margin"] != CPU_GOLDEN_RAW_LOGIT_MARGIN
        or first["probability"] != CPU_GOLDEN_PROBABILITY
        or first["ai_score"] != CPU_GOLDEN_PROBABILITY
        or first["classification_decision"] is not False
        or first["semantic_top_k_indices"] != CPU_GOLDEN_TOP_K_INDICES
        or first["semantic_top_k_gates"] != CPU_GOLDEN_TOP_K_GATES
        or first["final_expert_gates"] != CPU_GOLDEN_FINAL_GATES
        or first["peak_cuda_memory_bytes"] != 0
    ):
        raise ValueError("OmniAID Balanced CPU golden output drifted")
    return {
        "sample_id": CPU_GOLDEN_SAMPLE_ID,
        "input_path": CPU_GOLDEN_INPUT_PATH,
        "image_sha256": CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        **first,
        "repeat_artifact_sha256": second["artifact"]["artifact_sha256"],
        "repeat_array_sha256": second["artifact"]["array_sha256"],
        "repeat_probability": second["probability"],
        "repeat_byte_exact": True,
    }


def run_cpu_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    omniaid_config_path: Path,
) -> dict[str, Any]:
    """Run restricted load, strict graph construction, and CPU goldens."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before OmniAID CPU preflight")
    device, runtime = configure_runtime("cpu", seed=DEFAULT_SEED)
    if device.type != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("OmniAID CPU preflight configured a non-CPU runtime")
    source, assets, state, config = verify_assets(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        omniaid_config_path=omniaid_config_path,
    )
    model = None
    try:
        model, model_load = legacy._build_model(
            state,
            config,
            device,
            space_root,
        )
        official = legacy.validate_runtime_golden(
            model,
            device,
            space_root,
        )
        expected_official = [dict(case) for case in CPU_OFFICIAL_GOLDEN_CASES]
        if (
            official.get("status") != "passed"
            or official.get("device_family") != "cpu"
            or _official_cpu_projection(official) != expected_official
        ):
            raise ValueError("OmniAID official CPU golden output drifted")
        balanced = _balanced_cpu_golden(
            model=model,
            device=device,
            repo_root=repo_root,
        )
        if torch.cuda.is_initialized():
            raise RuntimeError("OmniAID CPU preflight initialized CUDA")
        return {
            "schema_version": CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "assets": assets,
            "model_load": model_load,
            "runtime": runtime,
            "official_golden": official,
            "balanced_golden": balanced,
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "cuda_initialized_before_cpu_model_load": False,
            "cuda_initialized_after_cpu_forwards": False,
            "dataset_manifest_loaded": False,
        }
    finally:
        if model is not None:
            del model
        del state
        gc.collect()


def _validate_preflight_report(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("OmniAID CPU preflight report is not an object")
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "assets",
        "model_load",
        "runtime",
        "official_golden",
        "balanced_golden",
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
        "dataset_manifest_loaded",
    }
    if set(report) != expected_keys:
        raise ValueError("OmniAID CPU preflight key set changed")
    if (
        report.get("schema_version") != CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or report.get("source") != source
        or report.get("assets") != assets
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("OmniAID CPU preflight provenance changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device") != "cpu":
        raise ValueError("OmniAID CPU preflight runtime is not CPU")
    validate_runtime_contract(runtime, label="CPU preflight runtime")
    load = report.get("model_load")
    if (
        not isinstance(load, Mapping)
        or load.get("strict_load") is not True
        or load.get("state_entries") != legacy.CHECKPOINT["tensor_count"]
        or load.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or load.get("svd_modules") != legacy.SVD_MODULE_COUNT
        or load.get("parameter_count") != 507_041_863
        or load.get("base_weights_downloaded") is not False
        or load.get("eval_mode") is not True
    ):
        raise ValueError("OmniAID CPU model-load audit changed")
    official = report.get("official_golden")
    if (
        not isinstance(official, Mapping)
        or official.get("status") != "passed"
        or official.get("device_family") != "cpu"
        or _official_cpu_projection(official)
        != [dict(case) for case in CPU_OFFICIAL_GOLDEN_CASES]
    ):
        raise ValueError("OmniAID CPU official golden changed")
    balanced = report.get("balanced_golden")
    balanced_artifact = (
        balanced.get("artifact") if isinstance(balanced, Mapping) else None
    )
    if (
        not isinstance(balanced, Mapping)
        or balanced.get("sample_id") != CPU_GOLDEN_SAMPLE_ID
        or balanced.get("image_sha256") != CPU_GOLDEN_IMAGE_SHA256
        or not isinstance(balanced_artifact, Mapping)
        or balanced_artifact.get("artifact_sha256") != CPU_GOLDEN_ARTIFACT_SHA256
        or balanced_artifact.get("array_sha256") != CPU_GOLDEN_ARRAY_SHA256
        or balanced.get("probability") != CPU_GOLDEN_PROBABILITY
        or balanced.get("repeat_byte_exact") is not True
    ):
        raise ValueError("OmniAID Balanced CPU golden changed")


def visibility_diagnostic(row: Mapping[str, Any]) -> dict[str, Any]:
    """Describe geometric GT visibility without claiming localization."""

    gt_kind = row.get("gt_mask_kind")
    width = row.get("width")
    height = row.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("Balanced250 input dimensions are invalid")
    if gt_kind == "exact_diff":
        return {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "basis": "full_canvas_direct_resize_without_crop",
                "definition": (
                    "every_native_pixel_is_in_the_geometric_model_input_"
                    "domain_after_direct_resize"
                ),
                "native_width": width,
                "native_height": height,
                "model_input_wh": [
                    legacy.MODEL_INPUT_SIZE,
                    legacy.MODEL_INPUT_SIZE,
                ],
                "resize_preserves_aspect_ratio": False,
                "crop": None,
                "preprocess_profile": FROZEN_PROFILE,
            },
        }
    if gt_kind == "all_zero":
        basis = "authentic_input_has_all_zero_GT"
    elif gt_kind == "not_applicable":
        basis = "conditional_full_frame_edit_has_no_local_GT"
    else:
        raise ValueError(f"unsupported Balanced250 GT kind {gt_kind!r}")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": basis,
            "native_width": width,
            "native_height": height,
            "model_input_wh": [
                legacy.MODEL_INPUT_SIZE,
                legacy.MODEL_INPUT_SIZE,
            ],
            "crop": None,
            "preprocess_profile": FROZEN_PROFILE,
        },
    }


def visibility_census(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    census = Counter(visibility_diagnostic(row)["edit_visibility"] for row in selected)
    result = {
        key: int(census[key]) for key in ("full", "not_applicable") if census[key]
    }
    if sum(result.values()) != len(selected):
        raise ValueError("OmniAID visibility census is incomplete")
    if len(selected) == 1775 and result != {
        "full": 750,
        "not_applicable": 1025,
    }:
        raise ValueError("formal OmniAID visibility census changed")
    return result


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    """Extend the shared Balanced v2 identity with legacy OmniAID fields."""

    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    path = _anchored(Path(str(row["canonical_path"])), repo_root)
    identity = {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "input_path": repo_relative(path, repo_root),
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "checkpoint_id": str(legacy.CHECKPOINT["id"]),
        "preprocess_profile": FROZEN_PROFILE,
        "score_semantics": legacy.SCORE_SEMANTICS,
        "classification_threshold": legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
        "config_fingerprint": run_manifest_fingerprint,
        **visibility_diagnostic(row),
        "valid_for_t1": True,
        "valid_for_t2": False,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "native_dense_output": False,
        },
    }
    if "pair_rank" in identity:
        raise AssertionError("Balanced OmniAID identity invented pair_rank")
    return identity


_T2_EXACT_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "attention_map",
        "attention_map_path",
        "score_map",
        "score_map_path",
        "predicted_mask",
        "predicted_mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "miou",
        "dice",
        "pixel_f1",
        "mcc",
        "s_joint",
        "joint_score",
        "joint_metrics",
    }
)


def forbidden_t2_claims(
    value: Any,
    path: tuple[str, ...] = (),
) -> set[str]:
    """Return every invented T2/dense-output claim at any nesting depth."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            rendered = ".".join((*path, key))
            if normalized == "valid_for_t2":
                if child is not False:
                    found.add(rendered)
            elif normalized == "localization_output":
                if child is not None:
                    found.add(rendered)
            elif normalized == "native_dense_output":
                if child is not False:
                    found.add(rendered)
            elif normalized in _T2_EXACT_KEYS:
                found.add(rendered)
            found.update(forbidden_t2_claims(child, (*path, key)))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            found.update(forbidden_t2_claims(child, (*path, str(index))))
    return found


def _reject_t2_claims(value: Any, label: str) -> None:
    found = forbidden_t2_claims(value)
    if found:
        raise ValueError(f"{label} invents OmniAID T2 output {sorted(found)[0]!r}")


def _ensure_repo_child(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    require_directory: bool = False,
) -> Path:
    root = repo_root.resolve()
    raw = path if path.is_absolute() else root / path
    absolute = Path(os.path.abspath(raw))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if require_directory and not absolute.is_dir():
        raise FileNotFoundError(f"missing {label}: {absolute}")
    return absolute


def _safe_sample_id(sample_id: Any) -> str:
    if (
        not isinstance(sample_id, str)
        or re.fullmatch(r"[0-9a-f]{24}", sample_id) is None
    ):
        raise ValueError("OmniAID sample-id must be 24 lowercase hex digits")
    return sample_id


def artifact_path(artifact_root: Path, sample_id: Any) -> Path:
    safe_id = _safe_sample_id(sample_id)
    artifact_dir = (artifact_root / "artifacts").resolve()
    path = (artifact_dir / f"{safe_id}.npz").resolve()
    if path.parent != artifact_dir:
        raise ValueError("OmniAID artifact path escapes artifact directory")
    return path


def _validate_artifact_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if list(arrays) != list(legacy.ARTIFACT_SCHEMA):
        raise ValueError("OmniAID artifact key order changed")
    validated: dict[str, np.ndarray] = {}
    for key, (shape, dtype) in legacy.ARTIFACT_SCHEMA.items():
        value = arrays.get(key)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not value.flags.c_contiguous
            or (np.issubdtype(dtype, np.floating) and not np.isfinite(value).all())
        ):
            raise ValueError(f"OmniAID artifact array {key} violates its contract")
        validated[key] = value
    return validated


def persist_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Persist the six-array artifact as deterministic canonical NPZ bytes."""

    validated = _validate_artifact_arrays(arrays)
    payload = _npz_bytes(validated)
    if len(payload) != ARTIFACT_FILE_BYTES:
        raise ValueError("OmniAID canonical artifact byte size changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("OmniAID artifact path is symlinked")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_artifact(
    path: Path,
    *,
    repo_root: Path,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fail closed on NPZ layout, metadata, arrays, and canonical bytes."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe OmniAID artifact: {path}")
    payload = path.read_bytes()
    if len(payload) != ARTIFACT_FILE_BYTES:
        raise ValueError("OmniAID artifact file size changed")
    expected_members = list(ARTIFACT_CONTRACT["zip_members"])
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if [
                info.filename for info in infos
            ] != expected_members or archive.testzip() is not None:
                raise ValueError("OmniAID NPZ member inventory changed")
            for info in infos:
                pure = PurePosixPath(info.filename)
                if (
                    pure.is_absolute()
                    or len(pure.parts) != 1
                    or any(part in ("", ".", "..") for part in pure.parts)
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != ARTIFACT_CONTRACT["zip_members"][info.filename]
                    or info.compress_size != info.file_size
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                ):
                    raise ValueError(f"OmniAID NPZ member changed: {info.filename}")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("OmniAID artifact is not a safe NPZ") from error
    arrays = _validate_artifact_arrays(legacy._load_artifact(path))
    canonical = _npz_bytes(arrays)
    if payload != canonical:
        raise ValueError("OmniAID artifact is not canonical NPZ bytes")
    relative = repo_relative(path, repo_root)
    record = {
        "artifact_path": relative,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_bytes": len(payload),
        "artifact_keys": list(legacy.ARTIFACT_SCHEMA),
        "artifact_paths": {"omniaid_npz": relative},
        "artifact_array_sha256": {
            key: _array_sha256(value) for key, value in arrays.items()
        },
    }
    if expected is not None:
        for key, value in record.items():
            if expected.get(key) != value:
                raise ValueError(f"OmniAID artifact record {key} changed")
    return arrays, record


def artifact_policy_contract(
    *,
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Prove that per-image artifacts are local-only and Git-ignored."""

    root = repo_root.resolve()
    gitignore = root / ".gitignore"
    if gitignore.is_symlink() or not gitignore.is_file():
        raise FileNotFoundError("repository .gitignore is missing or unsafe")
    safe_root = _ensure_repo_child(
        artifact_root,
        repo_root=root,
        label="OmniAID artifact root",
    )
    relative_root = safe_root.relative_to(root).as_posix()
    probe = f"{relative_root}/artifacts/{'0' * 24}.npz"
    completed = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--no-index", probe],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "OmniAID artifact root is not covered by repository ignore policy"
        )
    return {
        "storage": "local_only",
        "gitignored": True,
        "artifact_root": relative_root,
        "artifact_subdirectory": "artifacts",
        "artifact_extension": ".npz",
        "gitignore_path": ".gitignore",
        "gitignore_sha256": sha256_file(gitignore),
        "checkpoint_redistributed": False,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_score_payload(
    value: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    score = _finite_number(value.get("ai_score"), "OmniAID ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("OmniAID ai_score is outside [0, 1]")
    for alias in ("score", "probability", "fake_probability"):
        if _finite_number(value.get(alias), f"OmniAID {alias}") != score:
            raise ValueError(f"OmniAID score alias {alias} changed")
    logits_raw = value.get("class_logits")
    if not isinstance(logits_raw, list) or len(logits_raw) != legacy.CLASS_COUNT:
        raise ValueError("OmniAID class logits changed")
    logits = [
        _finite_number(item, f"OmniAID class_logits[{index}]")
        for index, item in enumerate(logits_raw)
    ]
    expected_margin = float(np.float32(np.float32(logits[1]) - np.float32(logits[0])))
    if (
        _finite_number(
            value.get("raw_logit_margin"),
            "OmniAID raw_logit_margin",
        )
        != expected_margin
    ):
        raise ValueError("OmniAID raw logit margin changed")
    decision = score > legacy.CLASSIFICATION_THRESHOLD
    if (
        value.get("classification_decision") is not decision
        or value.get("classification_threshold") != legacy.CLASSIFICATION_THRESHOLD
        or value.get("classification_threshold_operator")
        != legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        or value.get("score_semantics") != legacy.SCORE_SEMANTICS
        or value.get("routing_mode") != "Auto (Router)"
        or value.get("semantic_expert_names")
        != ["Human", "Animal", "Object", "Scene", "Anime"]
        or value.get("artifact_expert_name") != "Artifact"
        or value.get("classification")
        != {
            "decision": decision,
            "threshold": legacy.CLASSIFICATION_THRESHOLD,
            "operator": legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        }
        or value.get("t1") != {"valid": True, "score": score, "decision": decision}
        or value.get("manual_replay")
        != {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
            "router_scatter_exact": True,
        }
    ):
        raise ValueError("OmniAID classification/scientific semantics changed")
    indices_raw = value.get("semantic_top_k_indices")
    gates_raw = value.get("semantic_top_k_gates")
    final_raw = value.get("final_expert_gates")
    if (
        not isinstance(indices_raw, list)
        or len(indices_raw) != legacy.SEMANTIC_TOP_K
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in indices_raw
        )
        or len(set(indices_raw)) != legacy.SEMANTIC_TOP_K
        or any(item < 0 or item >= legacy.SEMANTIC_EXPERT_COUNT for item in indices_raw)
        or not isinstance(gates_raw, list)
        or len(gates_raw) != legacy.SEMANTIC_TOP_K
        or not isinstance(final_raw, list)
        or len(final_raw) != legacy.EXPERT_COUNT
    ):
        raise ValueError("OmniAID router payload changed")
    gates = [
        _finite_number(item, f"OmniAID top gate {index}")
        for index, item in enumerate(gates_raw)
    ]
    final = [
        _finite_number(item, f"OmniAID final gate {index}")
        for index, item in enumerate(final_raw)
    ]
    scattered = np.zeros(legacy.EXPERT_COUNT, dtype=np.float32)
    scattered[np.asarray(indices_raw, dtype=np.int64)] = np.asarray(
        gates,
        dtype=np.float32,
    )
    scattered[legacy.ARTIFACT_EXPERT_INDEX] = np.float32(1.0)
    if (
        any(gate <= 0.0 for gate in gates)
        or not np.isclose(sum(gates), 1.0, atol=1e-6, rtol=0.0)
        or not np.array_equal(
            scattered,
            np.asarray(final, dtype=np.float32),
        )
        or _finite_number(
            value.get("semantic_gate_sum"),
            "OmniAID semantic_gate_sum",
        )
        != float(
            np.asarray(
                final[: legacy.SEMANTIC_EXPERT_COUNT],
                dtype=np.float32,
            ).sum(dtype=np.float32)
        )
        or _finite_number(
            value.get("final_gate_sum"),
            "OmniAID final_gate_sum",
        )
        != float(np.asarray(final, dtype=np.float32).sum(dtype=np.float32))
    ):
        raise ValueError("OmniAID router gate invariants changed")
    if arrays is not None:
        validated = _validate_artifact_arrays(arrays)
        expected_lists = {
            "class_logits": [
                float(item) for item in validated["class_logits"].tolist()
            ],
            "semantic_top_k_indices": [
                int(item) for item in validated["semantic_top_k_indices"].tolist()
            ],
            "semantic_top_k_gates": [
                float(item) for item in validated["semantic_top_k_gates"].tolist()
            ],
            "final_expert_gates": [
                float(item) for item in validated["final_gates"].tolist()
            ],
        }
        for key, expected_list in expected_lists.items():
            if value.get(key) != expected_list:
                raise ValueError(f"OmniAID {key} differs from artifact")
    _reject_t2_claims(value, "OmniAID score payload")


def _replay_head_softmax_router(
    *,
    arrays: Mapping[str, np.ndarray],
    score: float,
    model: Any,
    device: Any,
) -> None:
    import torch
    from torch.nn import functional

    with torch.inference_mode():
        feature = torch.from_numpy(arrays["pooler_output"]).to(device).unsqueeze(0)
        routing = torch.from_numpy(arrays["routing_feature"]).to(device).unsqueeze(0)
        logits = functional.linear(
            feature,
            model.head.weight,
            model.head.bias,
        )
        probability = legacy._float32_probability(logits)
        routed = model.gating_network(routing)
    expected_logits = torch.from_numpy(arrays["class_logits"])
    if not torch.equal(logits[0].detach().cpu(), expected_logits):
        raise ValueError("OmniAID persisted head replay changed")
    if float(probability[0].item()) != score:
        raise ValueError("OmniAID persisted softmax replay changed")
    top_indices = routed.get("top_k_indices")
    top_gates = routed.get("top_k_gates")
    if top_indices is None or top_gates is None:
        raise ValueError("OmniAID persisted router API changed")
    if not torch.equal(
        top_indices[0].detach().cpu(),
        torch.from_numpy(arrays["semantic_top_k_indices"]),
    ) or not torch.equal(
        top_gates[0].detach().cpu(),
        torch.from_numpy(arrays["semantic_top_k_gates"]),
    ):
        raise ValueError("OmniAID persisted router replay changed")
    final = np.zeros(legacy.EXPERT_COUNT, dtype=np.float32)
    final[arrays["semantic_top_k_indices"]] = arrays["semantic_top_k_gates"]
    final[legacy.ARTIFACT_EXPERT_INDEX] = np.float32(1.0)
    if not np.array_equal(final, arrays["final_gates"]):
        raise ValueError("OmniAID persisted final-gate replay changed")


_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
        "artifact_keys",
        "artifact_paths",
        "artifact_array_sha256",
        "feature_shape",
        "feature_dtype",
        "feature_semantics",
        "feature_array_sha256",
        "class_logits_shape",
        "class_logits_dtype",
        "class_logits_array_sha256",
        "routing_feature_shape",
        "routing_feature_dtype",
        "routing_feature_semantics",
        "semantic_top_k_indices_shape",
        "semantic_top_k_indices_dtype",
        "semantic_top_k_gates_shape",
        "semantic_top_k_gates_dtype",
        "final_gates_shape",
        "final_gates_dtype",
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
        "probability",
        "ai_score",
        "score",
        "routing_mode",
        "semantic_expert_names",
        "artifact_expert_name",
        "semantic_top_k_indices",
        "semantic_top_k_gates",
        "final_expert_gates",
        "semantic_gate_sum",
        "final_gate_sum",
        "classification_decision",
        "classification",
        "t1",
        "manual_replay",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
_ERROR_RESULT_FIELDS = frozenset(
    {
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
        "probability",
        "ai_score",
        "score",
        "classification_decision",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "error_type",
        "error",
        "traceback",
    }
)


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    model: Any | None = None,
    device: Any | None = None,
    replay_success: bool = False,
) -> None:
    if not isinstance(attempt, Mapping):
        raise ValueError("OmniAID result attempt is not an object")
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("OmniAID result attempt status changed")
    expected = result_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    expected_fields = set(expected) | {"status", "completed_at"}
    expected_fields.update(
        _OK_RESULT_FIELDS if status == "ok" else _ERROR_RESULT_FIELDS
    )
    if set(attempt) != expected_fields:
        raise ValueError(
            "OmniAID result key set changed: "
            f"missing={sorted(expected_fields - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_fields)[:1]}"
        )
    for key, expected_value in expected.items():
        if not _same_json_type_and_value(
            attempt.get(key),
            expected_value,
        ):
            raise ValueError(f"OmniAID result identity field {key} changed")
    if not isinstance(attempt.get("completed_at"), str) or not attempt["completed_at"]:
        raise ValueError("OmniAID result completed_at is invalid")
    _reject_t2_claims(attempt, "OmniAID result")
    if status == "error":
        nullable = (
            "class_logits",
            "raw_logit_margin",
            "fake_probability",
            "probability",
            "ai_score",
            "score",
            "classification_decision",
            "peak_cuda_memory_bytes",
        )
        if (
            any(attempt.get(key) is not None for key in nullable)
            or attempt.get("latency_ms") != 0.0
            or not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt["traceback"]
        ):
            raise ValueError("OmniAID error result payload changed")
        return

    sample_id = str(input_row["sample_id"])
    input_path = _anchored(
        Path(str(input_row["canonical_path"])),
        repo_root,
    )
    if (
        input_path.is_symlink()
        or not input_path.is_file()
        or sha256_file(input_path) != str(input_row["canonical_sha256"])
    ):
        raise ValueError(f"OmniAID input changed for {sample_id}")
    _image, replay_preprocess = legacy.preprocess_image(input_path)
    if attempt.get("preprocess") != replay_preprocess:
        raise ValueError(f"OmniAID preprocessing changed for {sample_id}")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _finite_number(attempt.get(field), field) < 0.0:
            raise ValueError(f"OmniAID {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError("OmniAID peak CUDA memory is invalid")
    expected_path = artifact_path(artifact_root, sample_id)
    recorded_path = attempt.get("artifact_path")
    if (
        not isinstance(recorded_path, str)
        or _anchored(Path(recorded_path), repo_root) != expected_path
    ):
        raise ValueError("OmniAID artifact path changed or escapes")
    arrays, _record = validate_artifact(
        expected_path,
        repo_root=repo_root,
        expected=attempt,
    )
    hashes = attempt.get("artifact_array_sha256")
    if (
        not isinstance(hashes, Mapping)
        or hashes.get("pooler_output") != attempt.get("feature_array_sha256")
        or hashes.get("class_logits") != attempt.get("class_logits_array_sha256")
        or attempt.get("feature_shape") != [legacy.FEATURE_DIMENSION]
        or attempt.get("feature_dtype") != "float32"
        or attempt.get("feature_semantics") != legacy.FEATURE_SEMANTICS
        or attempt.get("class_logits_shape") != [legacy.CLASS_COUNT]
        or attempt.get("class_logits_dtype") != "float32"
        or attempt.get("routing_feature_shape") != [legacy.FEATURE_DIMENSION]
        or attempt.get("routing_feature_dtype") != "float32"
        or attempt.get("routing_feature_semantics") != legacy.ROUTING_FEATURE_SEMANTICS
        or attempt.get("semantic_top_k_indices_shape") != [legacy.SEMANTIC_TOP_K]
        or attempt.get("semantic_top_k_indices_dtype") != "int64"
        or attempt.get("semantic_top_k_gates_shape") != [legacy.SEMANTIC_TOP_K]
        or attempt.get("semantic_top_k_gates_dtype") != "float32"
        or attempt.get("final_gates_shape") != [legacy.EXPERT_COUNT]
        or attempt.get("final_gates_dtype") != "float32"
    ):
        raise ValueError("OmniAID artifact metadata changed")
    _validate_score_payload(attempt, arrays=arrays)
    if replay_success:
        if model is None or device is None:
            raise ValueError("OmniAID success replay requires model and device")
        _replay_head_softmax_router(
            arrays=arrays,
            score=float(attempt["ai_score"]),
            model=model,
            device=device,
        )


def validate_attempt_history(
    *,
    selected: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    model: Any | None = None,
    device: Any | None = None,
    replay_successes: bool = False,
) -> Any:
    """Validate every physical row and reject mutations after a success."""

    latest = index_latest_attempts(
        selected,
        attempts,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    successful: set[str] = set()
    for attempt in attempts:
        sample_id = str(attempt["sample_id"])
        if sample_id in successful:
            raise ValueError(f"OmniAID attempt appears after success for {sample_id}")
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[sample_id],
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
            model=model,
            device=device,
            replay_success=replay_successes and attempt.get("status") == "ok",
        )
        if attempt.get("status") == "ok":
            successful.add(sample_id)
    return latest


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    if text != json.dumps(value, ensure_ascii=False, indent=2) + "\n":
        raise ValueError(f"{label} is not canonical repository JSON")
    return value


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    _require_regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row_label = f"{label}:{line_number}"
                if not line.endswith("\n"):
                    raise ValueError(f"{row_label} lacks final newline")
                if not line.strip():
                    raise ValueError(f"{row_label} is blank")
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(row, dict):
                    raise ValueError(f"{row_label} is not a JSON object")
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
    source: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    execution_model_load: Mapping[str, Any],
    execution_official_golden: Mapping[str, Any],
    artifact_policy: Mapping[str, Any],
    run_dir: Path,
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Build the exact immutable evidence bound by the run fingerprint."""

    _valid_run_id(run_id)
    validate_runtime_contract(runtime, label="immutable execution runtime")
    _validate_preflight_report(
        cpu_preflight,
        source=source,
        assets=assets,
    )
    if execution_model_load != cpu_preflight.get("model_load"):
        raise ValueError("OmniAID execution model load differs from preflight")
    if (
        execution_official_golden.get("status") != "passed"
        or execution_official_golden.get("device_family")
        != str(runtime["device"]).split(":", 1)[0]
    ):
        raise ValueError("OmniAID execution official golden changed")
    immutable = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": MODEL_CONTRACT,
        "preprocess": PREPROCESS_CONTRACT,
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": TASK_SCOPE,
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "visibility_census": visibility_census(selected),
        "source": dict(source),
        "assets": dict(assets),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_dataset_manifest_load": True,
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "execution_model_load": dict(execution_model_load),
        "execution_official_golden": dict(execution_official_golden),
        "artifact_contract": ARTIFACT_CONTRACT,
        "artifact_policy": dict(artifact_policy),
        "outputs": {
            "run_dir": repo_relative(run_dir, repo_root),
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_root": repo_relative(artifact_root, repo_root),
            "artifact_dir": repo_relative(
                artifact_root / "artifacts",
                repo_root,
            ),
        },
    }
    if set(immutable) != IMMUTABLE_CONFIG_KEYS:
        raise AssertionError("internal OmniAID immutable key set changed")
    _reject_t2_claims(immutable, "OmniAID immutable config")
    return immutable


def _prepare_output_directories(
    *,
    repo_root: Path,
    run_dir: Path,
    artifact_root: Path,
    resume: bool,
) -> Path:
    run_dir = _ensure_repo_child(
        run_dir,
        repo_root=repo_root,
        label="OmniAID run directory",
    )
    artifact_root = _ensure_repo_child(
        artifact_root,
        repo_root=repo_root,
        label="OmniAID artifact root",
    )
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("OmniAID result and artifact roots must be disjoint")
    artifact_dir = artifact_root / "artifacts"
    if resume:
        if not run_dir.is_dir() or not artifact_dir.is_dir():
            raise FileNotFoundError(
                "OmniAID resume requires result and artifact directories"
            )
        root_entries = list(artifact_root.iterdir())
        if (
            len(root_entries) != 1
            or root_entries[0].name != "artifacts"
            or root_entries[0].is_symlink()
            or not root_entries[0].is_dir()
        ):
            raise ValueError("OmniAID artifact-root inventory changed")
    else:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"run directory is non-empty; pass --resume: {run_dir}"
            )
        if artifact_root.exists() and (
            not artifact_root.is_dir() or any(artifact_root.iterdir())
        ):
            raise FileExistsError(
                "artifact root is non-empty; pass --resume: " f"{artifact_root}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_child(
        artifact_dir,
        repo_root=repo_root,
        label="OmniAID artifact directory",
        require_directory=True,
    )
    return artifact_dir


def validate_artifact_inventory(
    *,
    artifact_root: Path,
    latest_by_sample_id: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> int:
    artifact_dir = artifact_root / "artifacts"
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise FileNotFoundError("OmniAID artifact directory is missing/unsafe")
    root_entries = list(artifact_root.iterdir())
    if (
        len(root_entries) != 1
        or root_entries[0] != artifact_dir
        or root_entries[0].is_symlink()
    ):
        raise ValueError("OmniAID artifact root contains extra entries")
    entries = list(artifact_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("OmniAID artifact inventory has an unsafe entry")
    expected = {
        f"{sample_id}.npz"
        for sample_id, attempt in latest_by_sample_id.items()
        if attempt.get("status") == "ok"
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "OmniAID artifact inventory mismatch: "
            f"missing={sorted(expected - actual)[:1]}, "
            f"extra={sorted(actual - expected)[:1]}"
        )
    for sample_id, attempt in latest_by_sample_id.items():
        if attempt.get("status") == "ok":
            validate_artifact(
                artifact_path(artifact_root, sample_id),
                repo_root=repo_root,
                expected=attempt,
            )
    return len(actual)


def _build_ok_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    fingerprint: str,
    model: Any,
    device: Any,
    scoring: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    preprocess: Mapping[str, Any],
    preprocess_latency_ms: float,
    latency_ms: float,
    peak_cuda_memory_bytes: int | None,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    path = artifact_path(artifact_root, sample_id)
    persist_artifact(path, arrays)
    persisted, record = validate_artifact(path, repo_root=repo_root)
    if any(
        not np.array_equal(arrays[key], persisted[key])
        for key in legacy.ARTIFACT_SCHEMA
    ):
        raise ValueError("OmniAID persisted artifact differs from inference")
    hashes = record["artifact_array_sha256"]
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "completed_at": utc_now(),
        "preprocess": dict(preprocess),
        "preprocess_latency_ms": float(preprocess_latency_ms),
        **record,
        "feature_shape": [legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": hashes["pooler_output"],
        "class_logits_shape": [legacy.CLASS_COUNT],
        "class_logits_dtype": "float32",
        "class_logits_array_sha256": hashes["class_logits"],
        "routing_feature_shape": [legacy.FEATURE_DIMENSION],
        "routing_feature_dtype": "float32",
        "routing_feature_semantics": legacy.ROUTING_FEATURE_SEMANTICS,
        "semantic_top_k_indices_shape": [legacy.SEMANTIC_TOP_K],
        "semantic_top_k_indices_dtype": "int64",
        "semantic_top_k_gates_shape": [legacy.SEMANTIC_TOP_K],
        "semantic_top_k_gates_dtype": "float32",
        "final_gates_shape": [legacy.EXPERT_COUNT],
        "final_gates_dtype": "float32",
        "latency_ms": float(latency_ms),
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        **dict(scoring),
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        artifact_root=artifact_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        model=model,
        device=device,
        replay_success=True,
    )
    return result


def _build_error_result(
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    fingerprint: str,
    error: BaseException,
) -> dict[str, Any]:
    result = {
        **result_identity(
            input_row,
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            valid_for_metrics=False,
        ),
        "status": "error",
        "completed_at": utc_now(),
        "class_logits": None,
        "raw_logit_margin": None,
        "fake_probability": None,
        "probability": None,
        "ai_score": None,
        "score": None,
        "classification_decision": None,
        "latency_ms": 0.0,
        "peak_cuda_memory_bytes": None,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    _validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=repo_root,
        artifact_root=artifact_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    return result


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.mode == "formal":
        if args.smoke_replicate is not None:
            raise ValueError("formal mode does not accept --smoke-replicate")
        return _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if args.mode == "smoke":
        defaults = {
            "a": DEFAULT_SMOKE_RUN_ID_A,
            "b": DEFAULT_SMOKE_RUN_ID_B,
        }
        if args.smoke_replicate is not None:
            frozen = defaults[args.smoke_replicate]
            if args.run_id is not None and args.run_id != frozen:
                raise ValueError(
                    "smoke replicate A/B requires its frozen 20260727 run-id"
                )
            return frozen
        if args.run_id is None:
            raise ValueError("smoke mode requires --smoke-replicate a|b or --run-id")
        return _valid_run_id(args.run_id)
    if args.mode == "single":
        if args.smoke_replicate is not None:
            raise ValueError("single mode does not accept --smoke-replicate")
        if args.run_id is None:
            raise ValueError("single mode requires an explicit --run-id")
        return _valid_run_id(args.run_id)
    raise ValueError(f"mode {args.mode!r} has no inference run-id")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--space-root",
        type=Path,
        default=DEFAULT_SPACE_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--omniaid-config",
        type=Path,
        default=DEFAULT_OMNIAID_CONFIG,
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke", "single", "preflight"),
        default="formal",
    )
    parser.add_argument(
        "--smoke-replicate",
        choices=("a", "b"),
    )
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--device",
        help="explicit cpu or cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _validate_resume_evidence(
    *,
    manifest_path: Path,
    expected_path: Path,
    results_path: Path,
    summary_path: Path,
    immutable: Mapping[str, Any],
    fingerprint: str,
    run_id: str,
    selected: Sequence[Mapping[str, Any]],
    dataset_contract: Mapping[str, Any],
    release: CanonicalRelease,
    repo_root: Path,
    artifact_root: Path,
    model: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    for path, label in (
        (manifest_path, "prior OmniAID manifest"),
        (expected_path, "prior OmniAID expected inputs"),
        (results_path, "prior OmniAID results"),
        (summary_path, "prior OmniAID summary"),
    ):
        _require_regular_file(path, label)
    manifest = _load_json_strict(manifest_path, "prior OmniAID manifest")
    expected = _read_jsonl_strict(
        expected_path,
        "prior OmniAID expected inputs",
    )
    attempts = _read_jsonl_strict(results_path, "prior OmniAID results")
    summary = _load_json_strict(summary_path, "prior OmniAID summary")
    if (
        set(manifest)
        != {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "fingerprint",
            "immutable",
            "dataset",
            "outputs",
            "execution",
        }
        or manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") not in ("complete", "incomplete")
        or manifest.get("fingerprint") != fingerprint
        or not isinstance(manifest.get("completed_at"), str)
        or not manifest["completed_at"]
    ):
        raise ValueError("prior OmniAID manifest identity/status changed")
    _require_same_json(
        manifest.get("immutable"),
        dict(immutable),
        "prior OmniAID immutable config",
    )
    if expected != list(selected):
        raise ValueError("prior OmniAID expected-input snapshot changed")
    dataset = manifest.get("dataset")
    outputs = manifest.get("outputs")
    execution = manifest.get("execution")
    immutable_outputs = immutable.get("outputs")
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(immutable_outputs, Mapping)
        or set(dataset)
        != {
            "contract",
            "manifest_path",
            "manifest_sha256",
            "expected_inputs_path",
            "expected_inputs_sha256",
            "selected_images",
        }
        or set(outputs)
        != set(immutable_outputs)
        | {
            "expected_inputs_sha256",
            "results_sha256",
            "summary_sha256",
            "artifact_files",
        }
        or set(execution)
        != {
            "new_successes",
            "resume_skips",
            "new_errors",
            "prior_physical_result_rows",
            "physical_result_rows",
            "latest_result_rows",
            "superseded_attempts",
        }
        or any(outputs.get(key) != value for key, value in immutable_outputs.items())
        or dataset.get("contract") != dataset_contract
        or dataset.get("manifest_path")
        != repo_relative(release.manifest_path, repo_root)
        or dataset.get("manifest_sha256") != release.manifest_sha256
        or dataset.get("expected_inputs_path")
        != repo_relative(expected_path, repo_root)
        or dataset.get("expected_inputs_sha256") != sha256_file(expected_path)
        or dataset.get("selected_images") != len(selected)
        or outputs.get("results_sha256") != sha256_file(results_path)
        or outputs.get("summary_sha256") != sha256_file(summary_path)
        or outputs.get("expected_inputs_sha256") != sha256_file(expected_path)
    ):
        raise ValueError("prior OmniAID manifest output hashes changed")
    latest = validate_attempt_history(
        selected=selected,
        attempts=attempts,
        repo_root=repo_root,
        artifact_root=artifact_root,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        model=model,
        device=device,
        replay_successes=True,
    )
    coverage = summarize_coverage(latest)
    artifact_files = validate_artifact_inventory(
        artifact_root=artifact_root,
        latest_by_sample_id=latest.latest_by_sample_id,
        repo_root=repo_root,
    )
    if (
        set(summary)
        != {
            "schema_version",
            "summary_kind",
            "scientific_metrics",
            "scientific_metrics_owner",
            "run_id",
            "run_manifest_fingerprint",
            "status",
            "mode",
            "model",
            "model_slug",
            "checkpoint_id",
            "score_spec",
            "task_scope",
            "dataset_contract",
            "coverage",
            "artifact_files",
            "generated_at",
        }
        or summary.get("schema_version") != RUNTIME_SUMMARY_SCHEMA
        or summary.get("summary_kind") != "runtime_coverage_only"
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner") != "analyze_omniaid_balanced.py"
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != fingerprint
        or summary.get("mode") != immutable.get("mode")
        or summary.get("model") != legacy.MODEL_NAME
        or summary.get("model_slug") != legacy.MODEL_SLUG
        or summary.get("checkpoint_id") != legacy.CHECKPOINT["id"]
        or summary.get("score_spec") != SCORE_SPEC.as_dict()
        or summary.get("task_scope") != TASK_SCOPE
        or summary.get("dataset_contract") != dataset_contract
        or summary.get("coverage") != coverage.as_dict()
        or summary.get("artifact_files") != artifact_files
        or summary.get("status")
        != ("complete" if coverage.is_complete else "incomplete")
        or manifest.get("status") != summary.get("status")
        or outputs.get("artifact_files") != artifact_files
        or execution.get("physical_result_rows") != len(attempts)
        or execution.get("latest_result_rows") != len(latest.latest_by_sample_id)
        or execution.get("superseded_attempts") != latest.superseded_attempts
        or any(
            isinstance(execution.get(key), bool)
            or not isinstance(execution.get(key), int)
            or execution[key] < 0
            for key in execution
        )
        or not isinstance(summary.get("generated_at"), str)
        or not summary["generated_at"]
    ):
        raise ValueError("prior OmniAID runtime summary changed")
    return manifest, attempts, latest


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(args.source_root, repo_root)
    space_root = _anchored(args.space_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    omniaid_config_path = _anchored(args.omniaid_config, repo_root)
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"OmniAID seed must be exactly {DEFAULT_SEED}")

    if args.mode == "preflight":
        if (
            args.resume
            or args.fail_fast
            or args.run_id is not None
            or args.smoke_replicate is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError(
                "preflight accepts no run, selection, resume, or CUDA options"
            )
        report = run_cpu_preflight(
            repo_root=repo_root,
            source_root=source_root,
            space_root=space_root,
            checkpoint_path=checkpoint_path,
            omniaid_config_path=omniaid_config_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _resolve_run_id(args)
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_base = _anchored(args.results_dir, repo_root)
    artifacts_base = _anchored(args.artifacts_dir, repo_root)
    run_dir = results_base / run_id
    artifact_root = artifacts_base / run_id
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    # This full CPU gate intentionally runs before dataset-manifest loading and
    # before any accelerator selection or CUDA model operation.
    cpu_preflight = run_cpu_preflight(
        repo_root=repo_root,
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        omniaid_config_path=omniaid_config_path,
    )

    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    selection_spec, selected = select_mode_inputs(
        release,
        mode=args.mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract_value = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    ).as_dict()

    source, assets, state, config = verify_assets(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        omniaid_config_path=omniaid_config_path,
    )
    if source != cpu_preflight.get("source") or assets != cpu_preflight.get("assets"):
        raise ValueError("OmniAID execution assets differ from CPU preflight")

    model = None
    fatal_error: BaseException | None = None
    try:
        device, runtime = configure_runtime(
            device_text,
            seed=DEFAULT_SEED,
        )
        model, execution_model_load = legacy._build_model(
            state,
            config,
            device,
            space_root,
        )
        del state
        execution_official_golden = legacy.validate_runtime_golden(
            model,
            device,
            space_root,
        )
        adapter_sources = adapter_source_contract(repo_root)
        policy = artifact_policy_contract(
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
        immutable = build_immutable_run_config(
            repo_root=repo_root,
            run_id=run_id,
            mode=args.mode,
            dataset_contract=dataset_contract_value,
            selected=selected,
            adapter_sources=adapter_sources,
            source=source,
            assets=assets,
            runtime=runtime,
            cpu_preflight=cpu_preflight,
            execution_model_load=execution_model_load,
            execution_official_golden=execution_official_golden,
            artifact_policy=policy,
            run_dir=run_dir,
            results_path=results_path,
            expected_inputs_path=expected_path,
            summary_path=summary_path,
            artifact_root=artifact_root,
        )
        fingerprint = _fingerprint(immutable)

        prior_attempts: list[dict[str, Any]]
        if args.resume:
            _prepare_output_directories(
                repo_root=repo_root,
                run_dir=run_dir,
                artifact_root=artifact_root,
                resume=True,
            )
            prior_manifest, prior_attempts, latest_before = _validate_resume_evidence(
                manifest_path=manifest_path,
                expected_path=expected_path,
                results_path=results_path,
                summary_path=summary_path,
                immutable=immutable,
                fingerprint=fingerprint,
                run_id=run_id,
                selected=selected,
                dataset_contract=dataset_contract_value,
                release=release,
                repo_root=repo_root,
                artifact_root=artifact_root,
                model=model,
                device=device,
            )
            started_at = prior_manifest.get("started_at")
            if not isinstance(started_at, str) or not started_at:
                raise ValueError("prior OmniAID started_at changed")
        else:
            _prepare_output_directories(
                repo_root=repo_root,
                run_dir=run_dir,
                artifact_root=artifact_root,
                resume=False,
            )
            atomic_write_jsonl(expected_path, selected)
            prior_attempts = []
            latest_before = index_latest_attempts(
                selected,
                [],
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
                score_spec=SCORE_SPEC,
            )
            started_at = utc_now()

        manifest: dict[str, Any] = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": run_id,
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "fingerprint": fingerprint,
            "immutable": immutable,
            "dataset": {
                "contract": dataset_contract_value,
                "manifest_path": repo_relative(
                    dataset_manifest_path,
                    repo_root,
                ),
                "manifest_sha256": release.manifest_sha256,
                "expected_inputs_path": repo_relative(
                    expected_path,
                    repo_root,
                ),
                "expected_inputs_sha256": sha256_file(expected_path),
                "selected_images": len(selected),
            },
            "outputs": {
                **dict(immutable["outputs"]),
                "expected_inputs_sha256": sha256_file(expected_path),
            },
        }
        atomic_write_json(manifest_path, manifest)

        new_successes = 0
        resume_skips = 0
        new_errors = 0
        for index, row in enumerate(selected, start=1):
            sample_id = str(row["sample_id"])
            prior = latest_before.latest_by_sample_id.get(sample_id)
            if prior is not None and prior.get("status") == "ok":
                resume_skips += 1
                print(
                    f"[{index}/{len(selected)}] resume {sample_id}",
                    flush=True,
                )
                continue
            target = artifact_path(artifact_root, sample_id)
            try:
                preprocess_started = time.perf_counter()
                image, preprocess = legacy.preprocess_image(
                    _anchored(Path(str(row["canonical_path"])), repo_root)
                )
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if preprocess.get("native_width") != int(
                    row["width"]
                ) or preprocess.get("native_height") != int(row["height"]):
                    raise ValueError("OmniAID preprocessed native dimensions changed")
                scoring, arrays, peak, latency = legacy.infer_one(
                    model,
                    device,
                    image,
                )
                result = _build_ok_result(
                    input_row=row,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    model=model,
                    device=device,
                    scoring=scoring,
                    arrays=arrays,
                    preprocess=preprocess,
                    preprocess_latency_ms=preprocess_latency_ms,
                    latency_ms=latency,
                    peak_cuda_memory_bytes=peak,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"p_fake={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                target.unlink(missing_ok=True)
                new_errors += 1
                error_result = _build_error_result(
                    input_row=row,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    error=error,
                )
                append_jsonl(results_path, error_result)
                print(
                    f"[{index}/{len(selected)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
                    break
            finally:
                gc.collect()

        physical = _read_jsonl_strict(results_path, "OmniAID results")
        latest = validate_attempt_history(
            selected=selected,
            attempts=physical,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            model=model,
            device=device,
            replay_successes=True,
        )
        coverage = summarize_coverage(latest)
        artifact_files = validate_artifact_inventory(
            artifact_root=artifact_root,
            latest_by_sample_id=latest.latest_by_sample_id,
            repo_root=repo_root,
        )
        summary = {
            "schema_version": RUNTIME_SUMMARY_SCHEMA,
            "summary_kind": "runtime_coverage_only",
            "scientific_metrics": None,
            "scientific_metrics_owner": "analyze_omniaid_balanced.py",
            "run_id": run_id,
            "run_manifest_fingerprint": fingerprint,
            "status": "complete" if coverage.is_complete else "incomplete",
            "mode": args.mode,
            "model": legacy.MODEL_NAME,
            "model_slug": legacy.MODEL_SLUG,
            "checkpoint_id": legacy.CHECKPOINT["id"],
            "score_spec": SCORE_SPEC.as_dict(),
            "task_scope": TASK_SCOPE,
            "dataset_contract": dataset_contract_value,
            "coverage": coverage.as_dict(),
            "artifact_files": artifact_files,
            "generated_at": utc_now(),
        }
        _reject_t2_claims(summary, "OmniAID runtime summary")
        atomic_write_json(summary_path, summary)

        manifest["status"] = summary["status"]
        manifest["completed_at"] = utc_now()
        manifest["execution"] = {
            "new_successes": new_successes,
            "resume_skips": resume_skips,
            "new_errors": new_errors,
            "prior_physical_result_rows": len(prior_attempts),
            "physical_result_rows": len(physical),
            "latest_result_rows": len(latest.latest_by_sample_id),
            "superseded_attempts": latest.superseded_attempts,
        }
        manifest["outputs"].update(
            {
                "results_sha256": sha256_file(results_path),
                "summary_sha256": sha256_file(summary_path),
                "artifact_files": artifact_files,
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": manifest["status"],
                    "mode": args.mode,
                    "coverage": coverage.as_dict(),
                    "manifest": str(manifest_path),
                    "artifact_root": str(artifact_root),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        if model is not None:
            del model
        if "state" in locals():
            del state
        gc.collect()

    if fatal_error is not None:
        raise RuntimeError("OmniAID fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
