#!/usr/bin/env python3
"""Run the official BR-Gen NFA-ViT release on CLAIMFORGE Mouse.

The adapter preserves both native tasks.  T1 is the released classifier
sigmoid and T2 is the released segmentation sigmoid.  The public repository
currently has a broken package export (``NFA_ViT`` versus the only defined
class, ``NFA_ViT_modify1``), so this file imports that class from the pinned
source tree without modifying the upstream checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import types
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
from eval.opensource.nfa_vit_metrics import (
    binary_pixel_metrics_strict,
    summarize_nfa_vit_results,
)


MODEL_NAME = "NFA-ViT"
MODEL_SLUG = "nfa_vit_brgen_checkpoint_9999_official"
MODEL_REPO_URL = "https://github.com/clpbc/BR-Gen"
MODEL_SOURCE_COMMIT = "4ced0e0966e96b9bd637cb34aa4ab8ab8eade782"
IMDLBENCO_REPO_URL = "https://github.com/scu-zjz/IMDLBenCo"
IMDLBENCO_SOURCE_COMMIT = "4e55633c3e68ede63974f72ea9af1a803a7f5ae8"
MODEL_INPUT_SIZE = 512
DECODER_LOGIT_SIZE = 128
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
MASK_THRESHOLD_OPERATOR = ">"
UPSTREAM_CLASS_NAME = "NFA_ViT_modify1"
UPSTREAM_BROKEN_EXPORT_NAME = "NFA_ViT"

SOURCE_FILES = {
    "LICENSE": "e3cda76e3e7da38645c755e237be21aeb6dd67ba0a8bf9557ecea0570d024528",
    "requirements.txt": "c1ac5ed0e9d297ed60f274540bfba658a748599b8d7b0117cfcc7c032ad1c06a",
    "test.py": "9e73c6beda55919cb720c7f2743e53cea96d4faff4f6bd34fba17768cc2dab27",
    "model_zoo/__init__.py": "e9a2cd7451e9144922ce4593a2de961529e3212bdff57b08c3e06d96073aff42",
    "model_zoo/nfa_vit/DnCNN.py": "ff151c95a932062da348794efa4f4a5963eebcf5cb654b98b98e85c72ff95ad7",
    "model_zoo/nfa_vit/decoderhead.py": "366a224a72a810ba73b1173214fa1f2d0156b7203e3adb21d4a49f6f9eb087d7",
    "model_zoo/nfa_vit/imagebackbone_segformer_b2.py": "4f94d8eef25c2fd7284ca67a52151a22583e33fd8b82ca8e9c65a2ce45c3dbde",
    "model_zoo/nfa_vit/nfa_vit.py": "f6f4a82a69e900fc8245e0b0b9205fc5d9fc0848f4ad6b482995be6e3e4cb0a6",
    "model_zoo/nfa_vit/noisebackbone_segformer_b0.py": "549744003b83ee4abb779922afd56b37f376f645b4018ed920c5e927f6f63e17",
}

IMDLBENCO_SOURCE_FILES = {
    "IMDLBenCo/__init__.py": "f82e36e6d9a02d884348001c0f581e7a1540bac1340dd9f85f8dfb759d8917d4",
    "IMDLBenCo/registry.py": "bb8f2b1c5ee2fcb0fdaab5e9c544b043c720e302f9a32b0190eeaae458ebc2fb",
    "IMDLBenCo/datasets/abstract_dataset.py": "dbfee521ca87790c871dd9e72aa5836dde2651fb352ada6aed9ea0fdf6747de6",
    "IMDLBenCo/transforms/iml_transforms.py": "b1905a889acb4ea65d2d61b4138e978afe84c3e183d7ae3f3080f86fc047c849",
    "IMDLBenCo/training_scripts/utils/misc.py": "743545325d61b3f40ea2f2ffd30f77d6a941d4ebbe64532a961cf3fc98d422e5",
}

CHECKPOINT_SAFE_GLOBALS = {
    "argparse.Namespace": argparse.Namespace,
}

# The only trained model file in the authors' public NFA-ViT folder.  The
# release does not document what "9999" denotes, so it is not called an epoch.
# SHA-256 and state schema are filled only after locally auditing the bytes.
CHECKPOINT = {
    "provider": "official_author_baidu_netdisk",
    "share_url": "https://pan.baidu.com/s/1mqmMeoTzJf0TuIy17N6PFQ",
    "share_password": "cclp",
    "released_filename": "checkpoint-9999.pth",
    "released_identifier": "9999",
    "identifier_semantics": "opaque_released_filename_not_an_epoch_claim",
    "bytes": 312_770_914,
    "md5": "b7f0b0e3ff2be6d49b31ea57c2d09cff",
    "sha256": None,
}

INITIALIZATION_WEIGHTS = {
    "noiseprint": {
        "released_filename": "noiseprint.pth",
        "bytes": 2_264_619,
        "md5": "2d01a0db4a36edd258f853d8d3fce2ca",
    },
    "segformer_b0": {
        "released_filename": "segformer_b0_backbone_weights.pth",
        "bytes": 14_331_578,
        "md5": "0939b8acc9b288d3e55fdea680821728",
    },
    "segformer_b2": {
        "released_filename": "segformer_b2_backbone_weights.pth",
        "bytes": 98_887_027,
        "md5": "2a47effc730367feaa0ee1825449a689",
    },
}

CLASS_NAME_COMPATIBILITY = {
    "upstream_test_shell_requests": UPSTREAM_BROKEN_EXPORT_NAME,
    "upstream_package_exports": UPSTREAM_BROKEN_EXPORT_NAME,
    "upstream_source_defines": UPSTREAM_CLASS_NAME,
    "adapter_action": (
        "load the only class directly from pinned source under a synthetic "
        "package; do not edit or rename upstream files"
    ),
}

INITIALIZATION_BYPASS = {
    "enabled": True,
    "reason": (
        "the constructor requires three initialization-only files before the "
        "released full state is loaded; the full checkpoint is strictly "
        "required to cover every final parameter and buffer"
    ),
    "mechanism": (
        "return an architecture-compatible temporary DnCNN state and empty "
        "strict=False SegFormer states only during construction, then load "
        "the released full model state with strict=True"
    ),
}

LOGIT_CAPTURE = {
    "segmentation": (
        "forward hook on official model.seg_decoder before the official "
        "128-to-512 bilinear interpolation"
    ),
    "classification": (
        "forward hook on official model.cls_decoder before the official "
        "sigmoid"
    ),
}

MODEL_LOGIT_RESIZE = {
    "source": [DECODER_LOGIT_SIZE, DECODER_LOGIT_SIZE],
    "destination": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
    "mode": "torch.nn.functional.interpolate_bilinear",
    "align_corners": False,
}
NATIVE_RESTORE = {
    "source": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
    "mode": "torch.nn.functional.interpolate_bilinear_probability",
    "align_corners": False,
    "threshold_after_restore": True,
}

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/mouse_canonical_v1/manifest.json")
DEFAULT_BRGEN_ROOT = Path("/root/.cache/claimforge/third_party/BR-Gen")
DEFAULT_IMDLBENCO_ROOT = Path("/root/.cache/claimforge/third_party/IMDLBenCo-4e55633c")
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/checkpoints/nfa-vit-4ced0e0/checkpoint-9999.pth"
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


def _verify_file_contract(
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected_sha256 = contract.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(f"{label} has no frozen SHA-256 contract")
    _verify_runtime_file(path, expected_sha256, label)
    if path.stat().st_size != int(contract["bytes"]):
        raise ValueError(
            f"{label} byte-size mismatch: "
            f"{path.stat().st_size} != {contract['bytes']}"
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
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    if sample_id is not None and pair_limit is not None:
        raise ValueError("pair_limit and sample_id are mutually exclusive")
    if sample_id is not None:
        selected = [row for row in rows if str(row.get("sample_id")) == sample_id]
        if len(selected) != 1:
            raise ValueError(f"sample-id must select exactly one row: {sample_id}")
        return selected
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
    """Reproduce the pinned IMDLBenCo Albumentations resize pipeline."""

    import albumentations as albu

    with Image.open(path) as opened:
        decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"NFA-ViT expects decoded RGB, got {decoded.shape}")
    native_height, native_width = decoded.shape[:2]
    transform = albu.Compose(
        [
            albu.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    resized = transform(image=decoded)["image"]
    chw = np.ascontiguousarray(
        np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
    )
    if chw.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(f"unexpected NFA-ViT input shape: {chw.shape}")
    if not np.isfinite(chw).all():
        raise ValueError("NFA-ViT input contains non-finite values")
    metadata = {
        "decoder": "Pillow.Image.open.convert_RGB",
        "exif_transpose": False,
        "icc_conversion": False,
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [native_width, native_height],
        "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize": "albumentations_1_3_0.Resize_cv2_INTER_LINEAR",
        "normalization": {
            "kind": "albumentations.Normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "max_pixel_value": 255.0,
        },
        "noiseprint_input": (
            "same_ImageNet_normalized_RGB_tensor_no_separate_preprocessing"
        ),
        "tensor_shape": list(chw.shape),
        "tensor_dtype": str(chw.dtype),
        "tensor_sha256": _sha256_array(chw),
    }
    return chw, (native_width, native_height), metadata


def _float32_array(tensor: Any, label: str) -> np.ndarray:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{label} is not a tensor")
    value = np.ascontiguousarray(
        tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    )
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def postprocess_outputs(
    output: Any,
    captured_segmentation_logits: Any,
    captured_classification_logits: Any,
    *,
    native_width: int,
    native_height: int,
) -> dict[str, Any]:
    """Validate and expose the exact T1/T2 prediction chain."""

    import torch
    from torch.nn import functional as F

    if not isinstance(output, dict):
        raise ValueError("NFA-ViT forward output is not a dictionary")
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native dimensions must be positive")
    raw_segmentation = captured_segmentation_logits
    raw_classification = captured_classification_logits
    if not isinstance(raw_segmentation, torch.Tensor) or tuple(
        raw_segmentation.shape
    ) != (1, 1, DECODER_LOGIT_SIZE, DECODER_LOGIT_SIZE):
        raise ValueError(
            "unexpected NFA-ViT decoder logit shape: "
            f"{getattr(raw_segmentation, 'shape', None)}"
        )
    if not isinstance(raw_classification, torch.Tensor) or tuple(
        raw_classification.shape
    ) != (1, 1):
        raise ValueError(
            "unexpected NFA-ViT classifier logit shape: "
            f"{getattr(raw_classification, 'shape', None)}"
        )
    pred_mask = output.get("pred_mask")
    if not isinstance(pred_mask, torch.Tensor) or tuple(pred_mask.shape) != (
        1,
        1,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            f"unexpected NFA-ViT pred_mask shape: {getattr(pred_mask, 'shape', None)}"
        )
    pred_label = output.get("pred_label")
    if not isinstance(pred_label, torch.Tensor) or pred_label.numel() != 1:
        raise ValueError(
            f"unexpected NFA-ViT pred_label shape: {getattr(pred_label, 'shape', None)}"
        )

    resized_logits_tensor = F.interpolate(
        raw_segmentation.float(),
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    replay_probability = torch.sigmoid(resized_logits_tensor)
    if not torch.allclose(replay_probability, pred_mask.float(), atol=2e-6, rtol=0):
        difference = float(
            torch.max(torch.abs(replay_probability - pred_mask.float())).item()
        )
        raise ValueError(
            "official segmentation probability does not replay from captured "
            f"logits; max absolute difference {difference}"
        )
    replay_mask_decision = replay_probability > MASK_THRESHOLD
    output_mask_decision = pred_mask.float() > MASK_THRESHOLD
    if not torch.equal(replay_mask_decision, output_mask_decision):
        differing_pixels = int(
            torch.count_nonzero(
                torch.logical_xor(replay_mask_decision, output_mask_decision)
            ).item()
        )
        raise ValueError(
            "official segmentation strict >0.5 decision does not replay from "
            f"captured logits; {differing_pixels} pixels differ"
        )
    replay_label = torch.sigmoid(raw_classification.float()).reshape(())
    output_label = pred_label.float().reshape(())
    if not torch.allclose(replay_label, output_label, atol=2e-7, rtol=0):
        raise ValueError("official classifier probability does not replay from logit")
    replay_label_decision = bool(
        (replay_label > CLASSIFICATION_THRESHOLD).item()
    )
    output_label_decision = bool(
        (output_label > CLASSIFICATION_THRESHOLD).item()
    )
    if replay_label_decision != output_label_decision:
        raise ValueError(
            "official classifier strict >0.5 decision does not replay from logit"
        )

    native_probability_tensor = F.interpolate(
        pred_mask.float(),
        size=(native_height, native_width),
        mode="bilinear",
        align_corners=False,
    )
    decoder_logits = _float32_array(
        raw_segmentation[0, 0],
        "decoder logits",
    )
    resized_logits = _float32_array(
        resized_logits_tensor[0, 0],
        "resized logits",
    )
    probability_model = _float32_array(
        pred_mask[0, 0],
        "model probability",
    )
    probability_native = _float32_array(
        native_probability_tensor[0, 0],
        "native probability",
    )
    for label, value in (
        ("model probability", probability_model),
        ("native probability", probability_native),
    ):
        if float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise ValueError(f"{label} falls outside [0, 1]")
    raw_logit = float(raw_classification.float().reshape(()).item())
    score = float(output_label.item())
    if not math.isfinite(raw_logit) or not math.isfinite(score):
        raise ValueError("classification output is not finite")
    if score < 0.0 or score > 1.0:
        raise ValueError("classification score falls outside [0, 1]")
    return {
        "decoder_logits_128": decoder_logits,
        "resized_logits_512": resized_logits,
        "probability_512": probability_model,
        "probability_native": probability_native,
        "classification_raw_logit": raw_logit,
        "classification_score": score,
        "classification_decision": score > CLASSIFICATION_THRESHOLD,
    }


@contextlib.contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    previous = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = previous


def _module_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_source_module(
    qualified_name: str,
    path: Path,
    *,
    package: bool = False,
) -> types.ModuleType:
    if qualified_name in sys.modules:
        module = sys.modules[qualified_name]
        origin = getattr(module, "__file__", None)
        if package or (origin and Path(origin).resolve() == path.resolve()):
            return module
        raise RuntimeError(f"module collision for {qualified_name}: {origin}")
    kwargs: dict[str, Any] = {}
    if package:
        kwargs["submodule_search_locations"] = [str(path)]
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        path / "__init__.py" if package else path,
        **kwargs,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {qualified_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def _load_upstream_class(
    brgen_root: Path,
    imdlbenco_root: Path,
) -> type[Any]:
    """Import the only current class while bypassing the broken parent export."""

    package_name = "_claimforge_brgen_nfa_4ced0e0"
    if package_name in sys.modules:
        module = sys.modules[f"{package_name}.nfa_vit"]
        return getattr(module, UPSTREAM_CLASS_NAME)

    # Import the exact IMDLBenCo registry pinned by BR-Gen requirements.
    with _temporary_sys_path(imdlbenco_root):
        import IMDLBenCo

    imdl_origin = Path(str(IMDLBenCo.__file__))
    if not _module_under(imdl_origin, imdlbenco_root):
        raise RuntimeError(
            f"IMDLBenCo resolved outside pinned tree: {imdl_origin}"
        )

    nfa_root = brgen_root / "model_zoo" / "nfa_vit"
    package = types.ModuleType(package_name)
    package.__file__ = str(nfa_root / "__init__.py")
    package.__path__ = [str(nfa_root)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    previous_sys_path = list(sys.path)
    try:
        for stem in (
            "DnCNN",
            "noisebackbone_segformer_b0",
            "imagebackbone_segformer_b2",
            "decoderhead",
            "nfa_vit",
        ):
            _load_source_module(
                f"{package_name}.{stem}",
                nfa_root / f"{stem}.py",
            )
    finally:
        # nfa_vit.py appends "." as an import side effect.
        sys.path[:] = previous_sys_path
    module = sys.modules[f"{package_name}.nfa_vit"]
    if hasattr(module, UPSTREAM_BROKEN_EXPORT_NAME):
        raise RuntimeError(
            "upstream class-name contract changed; review compatibility shim"
        )
    model_class = getattr(module, UPSTREAM_CLASS_NAME, None)
    if not isinstance(model_class, type):
        raise RuntimeError(f"upstream class {UPSTREAM_CLASS_NAME} is missing")
    return model_class


def _safe_checkpoint_payload(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    import torch

    unsafe_globals = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    )
    unexpected_globals = sorted(
        set(unsafe_globals) - set(CHECKPOINT_SAFE_GLOBALS)
    )
    global_audit = {
        "preflight": (
            "torch.serialization.get_unsafe_globals_in_checkpoint"
        ),
        "unsafe_globals": unsafe_globals,
        "allowlisted_globals": sorted(CHECKPOINT_SAFE_GLOBALS),
        "unexpected_globals": unexpected_globals,
        "allowlist_scope": "torch.serialization.safe_globals_context",
        "weights_only": True,
    }
    if unexpected_globals:
        raise ValueError(
            "NFA-ViT checkpoint contains unexpected unsafe globals: "
            f"{unexpected_globals}; audit={stable_json(global_audit)}"
        )
    with torch.serialization.safe_globals(
        list(CHECKPOINT_SAFE_GLOBALS.values())
    ):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, dict):
        raise ValueError("NFA-ViT checkpoint is not a dictionary")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("NFA-ViT checkpoint has no non-empty model mapping")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise ValueError(
            f"NFA-ViT model state has non-tensor entries: {invalid[:5]}"
        )
    tensor_counts: dict[str, int] = {}
    state_elements = 0
    tensor_bytes = 0
    for value in state.values():
        dtype = str(value.dtype)
        tensor_counts[dtype] = tensor_counts.get(dtype, 0) + 1
        state_elements += int(value.numel())
        tensor_bytes += int(value.numel() * value.element_size())
    audit = {
        "top_level_keys": sorted(str(key) for key in payload),
        "model_container": f"{type(state).__module__}.{type(state).__name__}",
        "state_keys": len(state),
        "state_elements": state_elements,
        "tensor_bytes": tensor_bytes,
        "state_dtypes": dict(sorted(tensor_counts.items())),
        "first_state_keys": list(state)[:10],
        "last_state_keys": list(state)[-10:],
        "global_safety_audit": global_audit,
    }
    return payload, state, audit


def _construct_model_with_full_state(
    model_class: type[Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Construct without external init bytes, then strictly install full state."""

    import torch
    from unittest import mock

    module = sys.modules[model_class.__module__]
    dncnn_class = getattr(
        sys.modules[f"{module.__package__}.DnCNN"],
        "DnCNN",
    )
    temporary_noise = dncnn_class(
        nplanes_in=3,
        nplanes_out=1,
        features=64,
        kernel=3,
        depth=17,
        activation="relu",
        lastact="linear",
        residual=True,
        bn=True,
    ).layers.state_dict()
    calls: list[str] = []

    def constructor_load(path: Any, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        value = str(path)
        calls.append(value)
        if value == "__claimforge_noiseprint_init__":
            return temporary_noise
        if value in {
            "__claimforge_segformer_b0_init__",
            "__claimforge_segformer_b2_init__",
        }:
            return {}
        raise RuntimeError(f"unexpected torch.load during construction: {value}")

    with mock.patch.object(torch, "load", side_effect=constructor_load):
        model = model_class(
            np_pretrain_weights="__claimforge_noiseprint_init__",
            seg_b0_pretrain_weights="__claimforge_segformer_b0_init__",
            seg_b2_pretrain_weights="__claimforge_segformer_b2_init__",
        )
    expected_calls = [
        "__claimforge_noiseprint_init__",
        "__claimforge_segformer_b0_init__",
        "__claimforge_segformer_b2_init__",
    ]
    if calls != expected_calls:
        raise RuntimeError(
            f"unexpected NFA-ViT constructor load sequence: {calls}"
        )
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            "strict NFA-ViT state load reported incompatibilities: "
            f"{incompatibility}"
        )
    model_state = model.state_dict()
    if list(model_state) != list(state):
        raise ValueError("checkpoint/model state-key order or coverage mismatch")
    parameters = sum(int(value.numel()) for value in model.parameters())
    buffers = sum(int(value.numel()) for value in model.buffers())
    audit = {
        "constructor_torch_load_calls": calls,
        "strict_full_state_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "model_state_keys": len(model_state),
        "parameters": parameters,
        "buffers": buffers,
        "state_elements": sum(int(value.numel()) for value in model_state.values()),
        "noise_extractor_requires_grad": any(
            parameter.requires_grad
            for parameter in model.noise_extractor.parameters()
        ),
    }
    if audit["noise_extractor_requires_grad"]:
        raise ValueError("upstream frozen Noiseprint parameters require gradients")
    return model, audit


def _verify_repository_contract(
    root: Path,
    *,
    expected_commit: str,
    source_files: Mapping[str, str],
    label: str,
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"missing {label} source tree: {root}")
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(
            f"{label} commit mismatch: {commit} != {expected_commit}"
        )
    # Importing the pinned upstream modules creates untracked ``__pycache__``
    # entries unless bytecode writing is disabled. Treating those generated
    # files as source changes makes every resume fail after the first model
    # load. Tracked changes remain forbidden, and every imported release
    # source file used by the adapter is independently pinned by SHA-256 below.
    dirty = _git_value(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError(f"{label} source tree is dirty: {dirty[:1000]}")
    for relative, digest in source_files.items():
        _verify_runtime_file(root / relative, digest, f"{label} source {relative}")


def _verify_static_contract(
    *,
    brgen_root: Path,
    imdlbenco_root: Path,
    checkpoint_path: Path,
) -> None:
    _verify_repository_contract(
        brgen_root,
        expected_commit=MODEL_SOURCE_COMMIT,
        source_files=SOURCE_FILES,
        label="BR-Gen",
    )
    _verify_repository_contract(
        imdlbenco_root,
        expected_commit=IMDLBENCO_SOURCE_COMMIT,
        source_files=IMDLBENCO_SOURCE_FILES,
        label="IMDLBenCo",
    )
    _verify_file_contract(checkpoint_path, CHECKPOINT, "NFA-ViT checkpoint")


def configure_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(
    *,
    brgen_root: Path,
    imdlbenco_root: Path,
    checkpoint_path: Path,
    device_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    _verify_static_contract(
        brgen_root=brgen_root,
        imdlbenco_root=imdlbenco_root,
        checkpoint_path=checkpoint_path,
    )
    _, state, checkpoint_audit = _safe_checkpoint_payload(checkpoint_path)
    model_class = _load_upstream_class(brgen_root, imdlbenco_root)
    model, construction_audit = _construct_model_with_full_state(
        model_class,
        state,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.to(device)
    model.eval()
    audit = {
        "checkpoint": checkpoint_audit,
        "construction": construction_audit,
        "class_module": model.__class__.__module__,
        "class_name": model.__class__.__name__,
    }
    return model, device, audit


def infer_one(
    model: Any,
    device: Any,
    image: np.ndarray,
    *,
    native_width: int,
    native_height: int,
) -> tuple[dict[str, Any], int | None, float]:
    import torch

    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    dummy_mask = torch.zeros(
        (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        dtype=tensor.dtype,
        device=device,
    )
    dummy_label = torch.zeros((1,), dtype=torch.long, device=device)
    captured: dict[str, Any] = {}

    def capture_segmentation(_module: Any, _inputs: Any, output: Any) -> None:
        captured["segmentation"] = output

    def capture_classification(_module: Any, _inputs: Any, output: Any) -> None:
        captured["classification"] = output

    segmentation_hook = model.seg_decoder.register_forward_hook(
        capture_segmentation
    )
    classification_hook = model.cls_decoder.register_forward_hook(
        capture_classification
    )
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor, dummy_mask, dummy_label)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    finally:
        segmentation_hook.remove()
        classification_hook.remove()
    if set(captured) != {"segmentation", "classification"}:
        raise RuntimeError(f"required NFA-ViT hooks did not fire: {captured.keys()}")
    processed = postprocess_outputs(
        output,
        captured["segmentation"],
        captured["classification"],
        native_width=native_width,
        native_height=native_height,
    )
    return processed, peak_bytes, latency_ms


def _validate_gt_row(
    row: Mapping[str, Any],
    repo_root: Path,
    *,
    native_width: int | None = None,
    native_height: int | None = None,
) -> np.ndarray | None:
    sample_id = str(row.get("sample_id"))
    kind = row.get("kind")
    width = int(row["width"]) if native_width is None else native_width
    height = int(row["height"]) if native_height is None else native_height
    if width <= 0 or height <= 0:
        raise ValueError(f"GT dimensions must be positive for {sample_id}")
    if kind == "real":
        if row.get("label") != 0:
            raise ValueError(f"real row {sample_id} must have label 0")
        if row.get("gt_mask_kind") != "all_zero":
            raise ValueError(
                f"real row {sample_id} must declare gt_mask_kind=all_zero"
            )
        if row.get("gt_mask_path") is not None:
            raise ValueError(f"real row {sample_id} must have null GT mask path")
        if row.get("gt_mask_sha256") is not None:
            raise ValueError(f"real row {sample_id} must have null GT mask hash")
        return None
    if kind != "forged":
        raise ValueError(f"row {sample_id} has unsupported kind: {kind}")
    if row.get("label") != 1:
        raise ValueError(f"forged row {sample_id} must have label 1")
    if row.get("gt_mask_kind") != "exact_diff":
        raise ValueError(
            f"forged row {sample_id} must declare gt_mask_kind=exact_diff"
        )
    mask_value = row.get("gt_mask_path")
    if not isinstance(mask_value, str):
        raise ValueError(f"forged row {sample_id} has no GT mask path")
    path = _anchored(Path(mask_value), repo_root)
    expected = row.get("gt_mask_sha256")
    if not _valid_sha256(expected):
        raise ValueError(f"forged row {sample_id} has no valid GT mask hash")
    _verify_runtime_file(path, str(expected), f"GT mask {sample_id}")
    with Image.open(path) as opened:
        target = np.asarray(opened)
    if target.ndim != 2:
        raise ValueError(
            f"GT mask {sample_id} must contain native two-dimensional pixels"
        )
    if target.shape != (height, width):
        raise ValueError(
            f"GT shape mismatch for {sample_id}: {target.shape} != "
            f"{(height, width)}"
        )
    if not np.isin(target, (0, 255)).all():
        raise ValueError(f"GT mask {sample_id} contains pixels outside 0/255")
    return np.asarray(target, dtype=np.uint8)


def _validate_selected_gt_contract(
    selected: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    for row in selected:
        _validate_gt_row(row, repo_root)


def _load_target(
    row: Mapping[str, Any],
    repo_root: Path,
    native_width: int,
    native_height: int,
) -> np.ndarray:
    target = _validate_gt_row(
        row,
        repo_root,
        native_width=native_width,
        native_height=native_height,
    )
    if target is None:
        return np.zeros((native_height, native_width), dtype=bool)
    return target == 255


def model_space_target(target: np.ndarray) -> np.ndarray:
    import cv2

    raw = np.asarray(target)
    if raw.ndim != 2:
        raise ValueError(f"target must be two-dimensional, got {raw.shape}")
    resized = cv2.resize(
        raw.astype(np.uint8),
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
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


def _complete_pair_count(rows: list[dict[str, Any]]) -> int:
    kinds_by_task: dict[str, set[str]] = {}
    for row in rows:
        kinds_by_task.setdefault(str(row["task_id"]), set()).add(str(row["kind"]))
    return sum(kinds == {"real", "forged"} for kinds in kinds_by_task.values())


def _runtime_contract(device_name: str) -> dict[str, Any]:
    import cv2
    import torch

    requested = torch.device(device_name)
    cuda_active = requested.type == "cuda" and torch.cuda.is_available()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "albumentations",
                "opencv-python-headless",
                "timm",
                "scikit-learn",
            )
        },
        "opencv_runtime_version": cv2.__version__,
        "official_requirements_deviations": {
            "torch": {
                "official": "2.5.1+cu118",
                "runtime": torch.__version__,
            },
            "numpy": {
                "official": "1.26.3",
                "runtime": np.__version__,
            },
            "Pillow": {
                "official": "11.0.0",
                "runtime": _package_version("Pillow"),
            },
            "torchvision": {
                "official": "0.20.1+cu118",
                "runtime": _package_version("torchvision"),
            },
        },
        "accelerator": {
            "requested_device": str(requested),
            "torch_cuda": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu_name": (
                torch.cuda.get_device_name(requested) if cuda_active else None
            ),
            "gpu_capability": (
                list(torch.cuda.get_device_capability(requested))
                if cuda_active
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
    brgen_root: Path,
    imdlbenco_root: Path,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    ordered_inputs = _selection_contract(selected)
    runtime = _runtime_contract(args.device)
    immutable = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "input": {
            "dataset_id": release["dataset_id"],
            "dataset_manifest": repo_relative(dataset_manifest_path, repo_root),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "dataset_contract_sha256": release["contract_sha256"],
            "inputs_manifest": repo_relative(inputs_path, repo_root),
            "inputs_sha256": release["inputs_sha256"],
            "selection_mode": (
                "explicit_single_sample_preflight"
                if getattr(args, "sample_id", None) is not None
                else "complete_leading_pairs"
            ),
            "selection_sha256": hashlib.sha256(
                stable_json(ordered_inputs).encode("utf-8")
            ).hexdigest(),
            "encoding": release["jpeg"],
        },
        "ordered_inputs": ordered_inputs,
        "runtime_contract": runtime,
        "model": {
            "name": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "repo_url": MODEL_REPO_URL,
            "source_root": str(brgen_root),
            "source_commit": MODEL_SOURCE_COMMIT,
            "source_pin_role": (
                "latest_official_operational_pin_not_claimed_training_commit"
            ),
            "source_files": [
                {"path": path, "sha256": digest}
                for path, digest in SOURCE_FILES.items()
            ],
            "imdlbenco": {
                "repo_url": IMDLBENCO_REPO_URL,
                "source_root": str(imdlbenco_root),
                "source_commit": IMDLBENCO_SOURCE_COMMIT,
                "pin_source": "BR-Gen_requirements.txt",
                "source_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in IMDLBENCO_SOURCE_FILES.items()
                ],
            },
            "class_name_compatibility": CLASS_NAME_COMPATIBILITY,
            "checkpoint": {
                **CHECKPOINT,
                "path": str(checkpoint_path),
                "safe_weights_only_load": True,
                "container_selection": "top_level_model",
                "strict_full_state_load": True,
                "prefix_rewrites": False,
                "schema_fallbacks": False,
            },
            "initialization_weights": INITIALIZATION_WEIGHTS,
            "initialization_bypass": INITIALIZATION_BYPASS,
            "checkpoint_selection": {
                "only_trained_checkpoint_in_public_folder": True,
                "selected_before_claimforge_evaluation": True,
                "claimforge_used_for_selection": False,
                "exact_training_and_selection_protocol": (
                    "unpublished_and_indeterminate"
                ),
                "public_script_risk_not_checkpoint_provenance": (
                    "the public scripts contain test-set PixelF1 selection, "
                    "but the checkpoint predates the current architecture and "
                    "current script, so that rule is not attributed to it"
                ),
            },
            "license": {
                "repository_root": "CC-BY-4.0",
                "repository_license_sha256": SOURCE_FILES["LICENSE"],
                "root_license_added_after_checkpoint_release": True,
                "embedded_dncnn_header": (
                    "all_rights_reserved_nonprofit_only_GRIP_UNINA"
                ),
                "checkpoint_license": "not_separately_stated",
                "commercial_status": (
                    "rights_conflict_requires_clarification_not_unconditional"
                ),
            },
            "supports_image_level_t1": True,
            "image_score_source": "official_cls_decoder_sigmoid",
            "supports_pixel_level_t2": True,
            "primary_localization_output": "official_pred_mask_sigmoid",
        },
        "inference": {
            "precision": "float32",
            "batch_size": 1,
            "seed": args.seed,
            "deterministic": True,
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "Pillow.Image.open.convert_RGB",
            "exif_transpose": False,
            "channel_order": "RGB",
            "input_geometry": (
                "direct_stretch_to_512x512_without_aspect_ratio_preservation"
            ),
            "resize": "albumentations_1_3_0_cv2_INTER_LINEAR",
            "normalization": "ImageNet_mean_std_max_pixel_value_255",
            "dummy_training_targets": {
                "mask": "all_zero_1x1x512x512",
                "label": "all_zero_length_1",
                "prediction_dependency": False,
                "purpose": "satisfy_forward_loss_signature_only",
            },
            "official_outputs": {
                "segmentation": (
                    "sigmoid(bilinear(seg_decoder_logits_128_to_512,"
                    "align_corners=False))"
                ),
                "classification": (
                    "sigmoid(cls_decoder(global_average_pool(final_B2_feature)))"
                ),
                "captured_logits": LOGIT_CAPTURE,
            },
            "native_compatibility_adapter": {
                "source": "official_model_512_probability_not_logits",
                "operation": NATIVE_RESTORE,
            },
            "classification_threshold": args.classification_threshold,
            "classification_threshold_operator": CLASSIFICATION_THRESHOLD_OPERATOR,
            "mask_threshold": args.mask_threshold,
            "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
            "test_time_augmentation": False,
            "ensemble": False,
            "forward_passes_per_image": 1,
        },
        "metrics": {
            "tasks": ["T1_image_detection", "T2_pixel_localization"],
            "positive_class": "manipulated_or_locally_generated",
            "t1_score": "official_classifier_probability",
            "t1_threshold": args.classification_threshold,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_512",
            "mask_threshold": args.mask_threshold,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_gt_resize": "cv2_INTER_NEAREST_to_512x512",
            "forged_pixel_ap_only": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "task_id_pair",
        },
        "artifacts": {
            "decoder_logits_128": {
                "format": "npy",
                "dtype": "float32",
                "shape": [DECODER_LOGIT_SIZE, DECODER_LOGIT_SIZE],
            },
            "resized_logits_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            },
            "probability_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            },
            "probability_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
            },
            "mask_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "probability_native > 0.5",
            },
        },
        "expected_complete_pairs": _complete_pair_count(selected),
        "expected_images": len(selected),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
        "adapter_contract": [
            {
                "path": repo_relative(Path(__file__), repo_root),
                "sha256": sha256_file(Path(__file__)),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("nfa_vit_metrics.py"),
                    repo_root,
                ),
                "sha256": sha256_file(
                    Path(__file__).with_name("nfa_vit_metrics.py")
                ),
            },
            {
                "path": repo_relative(
                    Path(__file__).with_name("common.py"),
                    repo_root,
                ),
                "sha256": sha256_file(Path(__file__).with_name("common.py")),
            },
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
                "existing NFA-ViT run manifest fingerprint differs; use a new run-id"
            )
        return
    atomic_write_json(path, manifest)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_completed_resume_model_audit(
    summary_path: Path,
    *,
    run_id: str,
    manifest_fingerprint: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if not _valid_sha256(checkpoint_sha256):
        raise RuntimeError("completed resume has no frozen checkpoint SHA-256")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"completed resume is missing prior summary: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"completed resume summary is unreadable: {summary_path}"
        ) from exc
    if not isinstance(summary, dict):
        raise ValueError("completed resume summary is not a dictionary")
    expected_identity = {
        "run_id": run_id,
        "run_manifest_fingerprint": manifest_fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"completed resume summary {field} mismatch: "
                f"{summary.get(field)!r} != {expected!r}"
            )

    audit = summary.get("model_load_audit")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("completed resume summary has no valid model_load_audit")
    checkpoint_audit = audit.get("checkpoint")
    construction_audit = audit.get("construction")
    if not isinstance(checkpoint_audit, dict) or not isinstance(
        construction_audit,
        dict,
    ):
        raise ValueError("completed resume model_load_audit has invalid sections")
    global_audit = checkpoint_audit.get("global_safety_audit")
    if not isinstance(global_audit, dict):
        raise ValueError(
            "completed resume model_load_audit has no checkpoint global audit"
        )
    unsafe_globals = global_audit.get("unsafe_globals")
    if (
        global_audit.get("preflight")
        != "torch.serialization.get_unsafe_globals_in_checkpoint"
        or global_audit.get("allowlisted_globals")
        != sorted(CHECKPOINT_SAFE_GLOBALS)
        or global_audit.get("unexpected_globals") != []
        or global_audit.get("allowlist_scope")
        != "torch.serialization.safe_globals_context"
        or global_audit.get("weights_only") is not True
        or not isinstance(unsafe_globals, list)
        or any(
            not isinstance(value, str)
            or value not in CHECKPOINT_SAFE_GLOBALS
            for value in unsafe_globals
        )
    ):
        raise ValueError(
            "completed resume model_load_audit checkpoint safety is invalid"
        )
    if (
        construction_audit.get("strict_full_state_load") is not True
        or construction_audit.get("missing_keys") != []
        or construction_audit.get("unexpected_keys") != []
    ):
        raise ValueError(
            "completed resume model_load_audit strict state load is invalid"
        )
    if (
        not isinstance(audit.get("class_module"), str)
        or not audit["class_module"]
        or not isinstance(audit.get("class_name"), str)
        or not audit["class_name"]
    ):
        raise ValueError(
            "completed resume model_load_audit has invalid model identity"
        )
    return audit


def _validate_npy_artifact(
    row: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    shape_field: str,
    expected_shape: tuple[int, int],
    repo_root: Path,
) -> None:
    path_value = row.get(path_field)
    if not isinstance(path_value, str):
        raise ValueError(f"resume row {row.get('id')} has no {path_field}")
    path = _anchored(Path(path_value), repo_root)
    digest = row.get(hash_field)
    if not _valid_sha256(digest):
        raise ValueError(f"resume row {row.get('id')} has invalid {hash_field}")
    _verify_runtime_file(path, str(digest), f"resume artifact {path_field}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or tuple(array.shape) != expected_shape:
        raise ValueError(
            f"resume artifact {path_field} contract mismatch: "
            f"{array.dtype} {array.shape}"
        )
    if list(array.shape) != row.get(shape_field):
        raise ValueError(f"resume row {row.get('id')} {shape_field} mismatch")
    if not np.isfinite(array).all():
        raise ValueError(f"resume artifact {path_field} contains non-finite values")


def _validate_mask_artifact(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_shape: tuple[int, int],
) -> None:
    path_value = row.get("mask_path")
    if not isinstance(path_value, str):
        raise ValueError(f"resume row {row.get('id')} has no mask_path")
    path = _anchored(Path(path_value), repo_root)
    digest = row.get("mask_sha256")
    if not _valid_sha256(digest):
        raise ValueError(f"resume row {row.get('id')} has invalid mask hash")
    _verify_runtime_file(path, str(digest), "resume native mask")
    with Image.open(path) as opened:
        mask = np.asarray(opened.convert("L"), dtype=np.uint8)
    if mask.shape != expected_shape or not np.isin(mask, (0, 255)).all():
        raise ValueError("resume native mask contract mismatch")


def _validate_resume_rows(
    existing: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest_fingerprint: str,
    *,
    repo_root: Path,
) -> None:
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    unexpected = sorted(set(existing) - set(selected_by_id))
    if unexpected:
        raise ValueError(f"results contain IDs outside selection: {unexpected[:5]}")
    for sample_id, row in existing.items():
        expected = selected_by_id[sample_id]
        if row.get("run_manifest_fingerprint") != manifest_fingerprint:
            raise ValueError(f"resume row {sample_id} manifest fingerprint mismatch")
        if row.get("image_sha256") != expected["canonical_sha256"]:
            raise ValueError(f"resume row {sample_id} input hash mismatch")
        if row.get("status") != "ok":
            continue
        width = int(expected["width"])
        height = int(expected["height"])
        _validate_npy_artifact(
            row,
            path_field="raw_logits_model_path",
            hash_field="raw_logits_model_sha256",
            shape_field="raw_logits_model_shape",
            expected_shape=(DECODER_LOGIT_SIZE, DECODER_LOGIT_SIZE),
            repo_root=repo_root,
        )
        _validate_npy_artifact(
            row,
            path_field="resized_logits_model_path",
            hash_field="resized_logits_model_sha256",
            shape_field="resized_logits_model_shape",
            expected_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            repo_root=repo_root,
        )
        _validate_npy_artifact(
            row,
            path_field="score_map_model_path",
            hash_field="score_map_model_sha256",
            shape_field="score_map_model_shape",
            expected_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            repo_root=repo_root,
        )
        _validate_npy_artifact(
            row,
            path_field="score_map_path",
            hash_field="score_map_sha256",
            shape_field="score_map_shape",
            expected_shape=(height, width),
            repo_root=repo_root,
        )
        _validate_mask_artifact(
            row,
            repo_root=repo_root,
            expected_shape=(height, width),
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if float(args.mask_threshold) != MASK_THRESHOLD:
        raise ValueError("official NFA-ViT mask threshold must be 0.5")
    if float(args.classification_threshold) != CLASSIFICATION_THRESHOLD:
        raise ValueError("official NFA-ViT classification threshold must be 0.5")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")

    repo_root = args.repo_root.resolve()
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    brgen_root = args.brgen_root.resolve()
    imdlbenco_root = args.imdlbenco_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = _anchored(args.output_dir, repo_root)
    artifact_dir = _anchored(
        (
            args.artifact_dir
            if args.artifact_dir is not None
            else Path(f"outputs/opensource/nfa_vit/{args.run_id}")
        ),
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
    selected = select_inputs(
        all_rows,
        args.pair_limit,
        getattr(args, "sample_id", None),
    )
    for row in selected:
        image_path = _anchored(Path(str(row["canonical_path"])), repo_root)
        _verify_runtime_file(
            image_path,
            str(row["canonical_sha256"]),
            f"canonical input {row['sample_id']}",
        )
    _validate_selected_gt_contract(selected, repo_root)
    _verify_static_contract(
        brgen_root=brgen_root,
        imdlbenco_root=imdlbenco_root,
        checkpoint_path=checkpoint_path,
    )
    configure_determinism(args.seed)

    run_manifest = build_run_manifest(
        args=args,
        repo_root=repo_root,
        dataset_manifest_path=dataset_manifest_path,
        release=release,
        inputs_path=inputs_path,
        selected=selected,
        brgen_root=brgen_root,
        imdlbenco_root=imdlbenco_root,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )
    _write_or_validate_run_manifest(run_manifest_path, run_manifest)
    existing = read_latest_by_id(output_path)
    _validate_resume_rows(
        existing,
        selected,
        run_manifest["fingerprint"],
        repo_root=repo_root,
    )
    pending = [
        row
        for row in selected
        if existing.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"NFA-ViT run {args.run_id}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )

    model = None
    model_load_audit: dict[str, Any] | None = None
    if pending:
        model, device, model_load_audit = load_model(
            brgen_root=brgen_root,
            imdlbenco_root=imdlbenco_root,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
        )
        print(
            "loaded official NFA-ViT checkpoint-9999 "
            f"{str(CHECKPOINT['sha256'])[:12]} on {device}",
            flush=True,
        )
        try:
            for index, input_row in enumerate(pending, start=1):
                sample_id = str(input_row["sample_id"])
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
                    "model": MODEL_NAME,
                    "model_slug": MODEL_SLUG,
                    "model_source_commit": MODEL_SOURCE_COMMIT,
                    "imdlbenco_source_commit": IMDLBENCO_SOURCE_COMMIT,
                    "checkpoint_sha256": CHECKPOINT["sha256"],
                    "checkpoint_released_identifier": (
                        CHECKPOINT["released_identifier"]
                    ),
                    "valid_for_t1": True,
                    "valid_for_t2": True,
                    "t1_policy": "official_native_classifier_probability",
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
                    processed, peak_bytes, latency_ms = infer_one(
                        model,
                        device,
                        image_array,
                        native_width=width,
                        native_height=height,
                    )
                    decoder_logits = processed["decoder_logits_128"]
                    resized_logits = processed["resized_logits_512"]
                    probability_model = processed["probability_512"]
                    probability_native = processed["probability_native"]
                    raw_logit = processed["classification_raw_logit"]
                    score = processed["classification_score"]
                    classification_decision = processed[
                        "classification_decision"
                    ]
                    target_native = _load_target(
                        input_row,
                        repo_root,
                        width,
                        height,
                    )
                    target_model = model_space_target(target_native)
                    include_ap = input_row["kind"] == "forged"
                    localization = {
                        "model_512": binary_pixel_metrics_strict(
                            probability_model,
                            target_model,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                        "native": binary_pixel_metrics_strict(
                            probability_native,
                            target_native,
                            args.mask_threshold,
                            include_ap=include_ap,
                        ),
                    }

                    decoder_logits_path = (
                        artifact_dir
                        / "decoder_logits_128"
                        / f"{sample_id}.npy"
                    )
                    resized_logits_path = (
                        artifact_dir
                        / "resized_logits_512"
                        / f"{sample_id}.npy"
                    )
                    model_probability_path = (
                        artifact_dir / "probability_512" / f"{sample_id}.npy"
                    )
                    native_probability_path = (
                        artifact_dir / "probability_native" / f"{sample_id}.npy"
                    )
                    native_mask_path = (
                        artifact_dir / "mask_native" / f"{sample_id}.png"
                    )
                    _atomic_save_npy(decoder_logits_path, decoder_logits)
                    _atomic_save_npy(resized_logits_path, resized_logits)
                    _atomic_save_npy(model_probability_path, probability_model)
                    _atomic_save_npy(native_probability_path, probability_native)
                    _atomic_save_mask(
                        native_mask_path,
                        probability_native > args.mask_threshold,
                    )

                    row = {
                        **identity,
                        "status": "ok",
                        "valid_for_metrics": True,
                        "score": score,
                        "score_semantics": (
                            "official_cls_decoder_sigmoid_forgery_probability"
                        ),
                        "classification_raw_logit": raw_logit,
                        "classification_score": score,
                        "classification_decision": classification_decision,
                        "classification_threshold": (
                            args.classification_threshold
                        ),
                        "classification_threshold_operator": ">",
                        "classification": {
                            "raw_logit": raw_logit,
                            "probability": score,
                            "score": score,
                            "decision": classification_decision,
                            "threshold": args.classification_threshold,
                            "threshold_operator": ">",
                            "semantics": (
                                "official_cls_decoder_sigmoid_"
                                "forgery_probability"
                            ),
                        },
                        "raw_logits_model_path": repo_relative(
                            decoder_logits_path,
                            repo_root,
                        ),
                        "raw_logits_model_sha256": sha256_file(
                            decoder_logits_path
                        ),
                        "raw_logits_model_shape": list(decoder_logits.shape),
                        "raw_logits_model_dtype": str(decoder_logits.dtype),
                        "raw_logits_model_semantics": (
                            "official_seg_decoder_pre_resize_logits"
                        ),
                        "raw_logits_capture": LOGIT_CAPTURE,
                        "resized_logits_model_path": repo_relative(
                            resized_logits_path,
                            repo_root,
                        ),
                        "resized_logits_model_sha256": sha256_file(
                            resized_logits_path
                        ),
                        "resized_logits_model_shape": list(
                            resized_logits.shape
                        ),
                        "resized_logits_model_dtype": str(
                            resized_logits.dtype
                        ),
                        "resized_logits_model_semantics": (
                            "official_bilinear_resized_pre_sigmoid_logits"
                        ),
                        "resized_logits_derivation": MODEL_LOGIT_RESIZE,
                        "score_map_model_path": repo_relative(
                            model_probability_path,
                            repo_root,
                        ),
                        "score_map_model_sha256": sha256_file(
                            model_probability_path
                        ),
                        "score_map_model_shape": list(
                            probability_model.shape
                        ),
                        "score_map_model_dtype": str(
                            probability_model.dtype
                        ),
                        "score_map_model_semantics": (
                            "official_pred_mask_sigmoid_probability"
                        ),
                        "score_map_path": repo_relative(
                            native_probability_path,
                            repo_root,
                        ),
                        "score_map_sha256": sha256_file(
                            native_probability_path
                        ),
                        "score_map_shape": list(probability_native.shape),
                        "score_map_dtype": str(probability_native.dtype),
                        "score_map_semantics": (
                            "model_512_probability_restored_to_native"
                        ),
                        "score_map_native_source": (
                            "official_model_512_probability"
                        ),
                        "score_map_native_restore": NATIVE_RESTORE,
                        "mask_path": repo_relative(
                            native_mask_path,
                            repo_root,
                        ),
                        "mask_sha256": sha256_file(native_mask_path),
                        "mask_shape": list(probability_native.shape),
                        "mask_dtype": "uint8",
                        "mask_threshold": args.mask_threshold,
                        "mask_threshold_operator": ">",
                        "artifact_paths": {
                            "decoder_logits_128_npy": repo_relative(
                                decoder_logits_path,
                                repo_root,
                            ),
                            "resized_logits_512_npy": repo_relative(
                                resized_logits_path,
                                repo_root,
                            ),
                            "probability_512_npy": repo_relative(
                                model_probability_path,
                                repo_root,
                            ),
                            "probability_native_npy": repo_relative(
                                native_probability_path,
                                repo_root,
                            ),
                            "mask_native_png": repo_relative(
                                native_mask_path,
                                repo_root,
                            ),
                        },
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
                        f" score={row['score']:.6f}"
                        f" f1={native_metrics.get('f1')}"
                        f" positive={native_metrics['predicted_positive_fraction']:.6f}"
                        f" latency={row['latency_ms']:.1f}ms"
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
                        f"NFA-ViT failed for {sample_id}: "
                        f"{row['error_message']}"
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model_load_audit = _load_completed_resume_model_audit(
            summary_path,
            run_id=args.run_id,
            manifest_fingerprint=run_manifest["fingerprint"],
            checkpoint_sha256=str(CHECKPOINT["sha256"]),
        )

    result_rows = read_jsonl(output_path) if output_path.is_file() else []
    summary = summarize_nfa_vit_results(
        result_rows,
        selected,
        classification_threshold=args.classification_threshold,
        mask_threshold=args.mask_threshold,
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
            "imdlbenco_source_commit": IMDLBENCO_SOURCE_COMMIT,
            "checkpoint_sha256": CHECKPOINT["sha256"],
            "checkpoint_released_identifier": CHECKPOINT[
                "released_identifier"
            ],
            "input_manifest_sha256": release["inputs_sha256"],
            "run_manifest_fingerprint": run_manifest["fingerprint"],
            "valid_for_t1": True,
            "valid_for_t2": True,
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
        raise RuntimeError(f"incomplete NFA-ViT run: {coverage}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--brgen-root", type=Path, default=DEFAULT_BRGEN_ROOT)
    parser.add_argument(
        "--imdlbenco-root",
        type=Path,
        default=DEFAULT_IMDLBENCO_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", default="mouse_canonical_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/opensource/nfa_vit"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pair-limit", type=int)
    selection.add_argument("--sample-id")
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
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
