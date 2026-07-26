#!/usr/bin/env python3
"""Independently audit a frozen Effort whole-image detection run.

The analyzer treats the runner's JSON, NPZ artifacts, scores, and summary as
untrusted.  It independently verifies the pinned source and checkpoint,
reconstructs the exact forward-relevant Effort graph, decodes and preprocesses
every selected image again, performs fresh full-model forwards, replays the
two-class head and float32 softmax, and recomputes all paired metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.effort_metrics import summarize_effort_results


FROZEN_SOURCE_COMMIT = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
FROZEN_SOURCE_FILES = {
    "README.md": "f5c0f66ed8566c65818162722c9935485721ad401bc8494d2742b32b75fd5721",
    "install.sh": "b37e791f514b28f09f64ad84b689d5a9deaff2c3abb688350aae7cbb0711c7fd",
    "DeepfakeBench/training/demo.py": (
        "009db8f76d3983e22d0e241ef602b11908f652f84d0fd5f5857f1973fdd12f9c"
    ),
    "DeepfakeBench/training/detectors/effort_detector.py": (
        "366b1cde008f537e4b9c8c8e4c65ee20b430c4bca1ccee1b1b86c20a9831fac9"
    ),
    "DeepfakeBench/training/config/detector/effort.yaml": (
        "1fd1398cf245b3a5c13cb130d7c6e209057ae6b56b561eca5f3c032283c5527b"
    ),
    "figs/effort_pipeline.png": (
        "f84fad60b6152b915874cfbd58ee7e21646fd4a36a642683dbb425e6f6bc879b"
    ),
    "figs/deepfake_tab1.png": (
        "f8494b571f9d663639193344fa8e0e18f1d41f42089f01f06133dc881ab39fc7"
    ),
}
FROZEN_CHECKPOINT_BYTES = 1_213_769_519
FROZEN_CHECKPOINT_SHA256 = (
    "7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813"
)
FROZEN_CHECKPOINT_TENSORS = 681
FROZEN_CHECKPOINT_ELEMENTS = 303_378_530
FROZEN_ORDERED_KEY_SHA256 = (
    "1782f72f07007cebae76a0f315845f1c60456d9223d47c8ce2f35a8f43816da7"
)
FROZEN_SCHEMA_SHA256 = (
    "bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489"
)
FROZEN_HF_CONFIG_BYTES = 4_519
FROZEN_HF_CONFIG_SHA256 = (
    "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"
)
FROZEN_PREPROCESS_PROFILE = (
    "official_deepfakebench_demo_natural_image_linear224_v1"
)
FROZEN_SCORE_SEMANTICS = (
    "official_float32_softmax_class1_probability_higher_is_fake"
)
FROZEN_CHECKPOINT_ID = "official_effort_genimage_sdv14_clip_l14"
FROZEN_MODEL_SLUG = "effort_clip_l14_genimage_sdv14"
FEATURE_DIMENSION = 1024
CLASS_COUNT = 2
INPUT_SIZE = 224
SVD_MODULES = 96
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
THRESHOLD = 0.5
MODEL_SEED = 20260724
CPU_THREADS = 16

EXPECTED_RUNTIME = {
    "torch": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "transformers": "4.53.2",
    "numpy": "1.26.4",
    "opencv": "4.10.0",
}

DEFAULT_RESULTS_DIR = Path("results/opensource/effort")
DEFAULT_RUN_ID = (
    "effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725"
)
DEFAULT_SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/effort-aigi-96f5dea2"
)
DEFAULT_CHECKPOINT = Path(
    "/root/.cache/claimforge/models/effort/"
    "effort_clip_L14_trainOn_sdv14.pth"
)
DEFAULT_HF_CONFIG = Path(
    "/root/.cache/claimforge/models/effort/"
    "clip-vit-large-patch14-config.json"
)

FORBIDDEN_T2_KEYS = frozenset(
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
        "s_joint",
        "joint_score",
        "joint_metrics",
    }
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _verify_file(path: Path, digest: Any, label: str) -> None:
    expected = _require_sha256(digest, f"{label} digest")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _reject_t2(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        present = sorted(FORBIDDEN_T2_KEYS.intersection(value))
        if present:
            raise ValueError(f"{path} invents Effort T2 fields: {present}")
        for key, child in value.items():
            _reject_t2(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_t2(child, path=f"{path}[{index}]")


def _verify_source(source_root: Path) -> dict[str, Any]:
    commit = _git_value(source_root, "rev-parse", "HEAD")
    if commit != FROZEN_SOURCE_COMMIT:
        raise ValueError("independent Effort source commit mismatch")
    dirty = _git_value(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError("independent Effort source tree is dirty")
    records: dict[str, Any] = {}
    for relative, digest in FROZEN_SOURCE_FILES.items():
        path = source_root / relative
        _verify_file(path, digest, f"Effort source {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    tracked = set((_git_value(source_root, "ls-files") or "").splitlines())
    licenses = sorted(
        name
        for name in tracked
        if Path(name).name.lower()
        in {"license", "license.txt", "copying", "notice", "notice.txt"}
    )
    if licenses:
        raise ValueError("independent Effort license census changed")
    return {
        "commit": commit,
        "tracked_dirty": False,
        "tracked_license_files": [],
        "files": records,
    }


def _schema_sha256(state: Mapping[str, Any]) -> str:
    canonical = "\n".join(
        f"{key}\t{tuple(value.shape)}\t{value.dtype}\t{value.numel()}"
        for key, value in state.items()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_assets(
    checkpoint_path: Path,
    config_path: Path,
) -> tuple[OrderedDict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    if checkpoint_path.stat().st_size != FROZEN_CHECKPOINT_BYTES:
        raise ValueError("independent Effort checkpoint byte size mismatch")
    _verify_file(
        checkpoint_path,
        FROZEN_CHECKPOINT_SHA256,
        "Effort checkpoint",
    )
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(
        checkpoint_path
    )
    if unsafe:
        raise ValueError(f"independent Effort unsafe globals: {unsafe}")
    raw = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, OrderedDict):
        raise ValueError("independent Effort checkpoint is not OrderedDict")
    if len(raw) != FROZEN_CHECKPOINT_TENSORS:
        raise ValueError("independent Effort tensor count mismatch")
    if any(not isinstance(value, torch.Tensor) for value in raw.values()):
        raise ValueError("independent Effort state has non-tensor values")
    if any(value.dtype != torch.float32 for value in raw.values()):
        raise ValueError("independent Effort state has non-FP32 values")
    if any(not torch.isfinite(value).all().item() for value in raw.values()):
        raise ValueError("independent Effort state has non-finite values")
    if sum(value.numel() for value in raw.values()) != FROZEN_CHECKPOINT_ELEMENTS:
        raise ValueError("independent Effort element count mismatch")
    key_digest = hashlib.sha256(
        "\n".join(raw.keys()).encode("utf-8")
    ).hexdigest()
    if key_digest != FROZEN_ORDERED_KEY_SHA256:
        raise ValueError("independent Effort ordered-key digest mismatch")
    if _schema_sha256(raw) != FROZEN_SCHEMA_SHA256:
        raise ValueError("independent Effort state schema mismatch")
    state: OrderedDict[str, Any] = OrderedDict()
    for key, value in raw.items():
        if not key.startswith("module."):
            raise ValueError("independent Effort state prefix mismatch")
        stripped = key[len("module.") :]
        if stripped in state:
            raise ValueError("independent Effort prefix collision")
        state[stripped] = value

    if config_path.stat().st_size != FROZEN_HF_CONFIG_BYTES:
        raise ValueError("independent CLIP config byte size mismatch")
    _verify_file(
        config_path,
        FROZEN_HF_CONFIG_SHA256,
        "Effort CLIP config",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision = _require_mapping(config.get("vision_config"), "vision_config")
    expected = {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
    }
    for key, value in expected.items():
        if vision.get(key) != value:
            raise ValueError(f"independent CLIP config mismatch: {key}")
    return state, config, {
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "checkpoint_bytes": FROZEN_CHECKPOINT_BYTES,
        "checkpoint_tensor_count": FROZEN_CHECKPOINT_TENSORS,
        "checkpoint_state_elements": FROZEN_CHECKPOINT_ELEMENTS,
        "checkpoint_schema_sha256": FROZEN_SCHEMA_SHA256,
        "unsafe_globals": [],
        "weights_only": True,
        "config_sha256": FROZEN_HF_CONFIG_SHA256,
    }


def _configure_runtime(device_text: str) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import cv2
    import torch
    import torchvision
    import transformers

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    for key, expected in EXPECTED_RUNTIME.items():
        if actual[key] != expected:
            raise ValueError(
                f"independent Effort runtime {key} mismatch: "
                f"{actual[key]} != {expected}"
            )
    device = torch.device(device_text)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("independent Effort supports cpu/cuda only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("independent Effort CUDA is unavailable")
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device, {
        **actual,
        "device": str(device),
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "cpu_threads": CPU_THREADS,
        "autocast": False,
        "model_dtype": "float32",
    }


def _build_model(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import CLIPVisionConfig, CLIPVisionModel

    class IndependentSVDLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_main = nn.Parameter(
                torch.empty(FEATURE_DIMENSION, FEATURE_DIMENSION),
                requires_grad=False,
            )
            self.bias = nn.Parameter(torch.empty(FEATURE_DIMENSION))
            self.S_residual = nn.Parameter(torch.empty(1))
            self.U_residual = nn.Parameter(
                torch.empty(FEATURE_DIMENSION, 1)
            )
            self.V_residual = nn.Parameter(
                torch.empty(1, FEATURE_DIMENSION)
            )

        def forward(self, value: Any) -> Any:
            effective = self.weight_main + (
                self.U_residual
                @ torch.diag(self.S_residual)
                @ self.V_residual
            )
            return functional.linear(value, effective, self.bias)

    class IndependentEffort(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vision = CLIPVisionConfig(**dict(config["vision_config"]))
            self.backbone = CLIPVisionModel(vision).vision_model
            for layer in self.backbone.encoder.layers:
                for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
                    setattr(
                        layer.self_attn,
                        projection,
                        IndependentSVDLinear(),
                    )
            self.head = nn.Linear(FEATURE_DIMENSION, CLASS_COUNT)

        def forward(self, image: Any) -> tuple[Any, Any]:
            feature = self.backbone(image).pooler_output
            return self.head(feature), feature

    with torch.device("meta"):
        model = IndependentEffort()
    if list(model.state_dict()) != list(state):
        raise ValueError("independent Effort model/state key mismatch")
    loaded = model.load_state_dict(state, strict=True, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("independent Effort strict load failed")
    model.backbone.embeddings.position_ids = torch.arange(
        257,
        dtype=torch.long,
    ).expand((1, -1))
    model = model.to(device).eval()
    modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, IndependentSVDLinear)
    ]
    if len(modules) != SVD_MODULES:
        raise ValueError("independent Effort SVD module count mismatch")
    return model, {
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "svd_modules": len(modules),
        "svd_module_names": modules,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "position_ids_materialized": True,
    }


def _preprocess(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"independent OpenCV decode failed: {path}")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("independent Effort decode contract changed")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    tensor = np.ascontiguousarray(
        (
            (resized.astype(np.float32) / np.float32(255.0) - mean)
            / std
        ).transpose(2, 0, 1),
        dtype=np.float32,
    )
    if tensor.shape != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError("independent Effort tensor shape changed")
    return tensor, {
        "decode": "cv2.imread_IMREAD_COLOR",
        "native_shape_hwc": [int(value) for value in bgr.shape],
        "native_width": int(bgr.shape[1]),
        "native_height": int(bgr.shape[0]),
        "decoded_bgr_sha256": _array_sha256(bgr),
        "color_conversion": "cv2_COLOR_BGR2RGB",
        "decoded_rgb_sha256": _array_sha256(rgb),
        "resize": {
            "output_wh": [INPUT_SIZE, INPUT_SIZE],
            "interpolation": "cv2_INTER_LINEAR",
            "preserve_aspect_ratio": False,
            "crop": None,
            "face_alignment": False,
        },
        "resized_rgb_sha256": _array_sha256(resized),
        "to_tensor": "uint8_to_float32_divide_255_CHW",
        "normalization_mean": list(CLIP_MEAN),
        "normalization_std": list(CLIP_STD),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_sha256": _array_sha256(tensor),
    }


def _load_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"pooler_output", "class_logits"}:
                raise ValueError("independent Effort artifact keys mismatch")
            feature = np.ascontiguousarray(payload["pooler_output"])
            logits = np.ascontiguousarray(payload["class_logits"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot safely load Effort artifact {path}") from exc
    if (
        feature.shape != (FEATURE_DIMENSION,)
        or feature.dtype != np.float32
        or logits.shape != (CLASS_COUNT,)
        or logits.dtype != np.float32
        or not np.isfinite(feature).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("independent Effort artifact contract mismatch")
    return feature, logits


def _latest_rows(
    physical: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(physical):
        if not isinstance(row, dict):
            raise ValueError(f"physical result row {index} is not an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {index} has invalid id")
        _reject_t2(row, path=f"result[{index}]")
        latest[row_id] = row
    return latest


def _expected_identity(
    expected: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    sample_id = str(expected["sample_id"])
    path = _anchored(Path(str(expected["canonical_path"])), repo_root)
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": str(expected["task_id"]),
        "pair_rank": int(expected["pair_rank"]),
        "rank": int(expected["rank"]),
        "kind": str(expected["kind"]),
        "label": int(expected["label"]),
        "domain": str(expected["domain"]),
        "input_path": str(path.relative_to(repo_root)),
        "input_sha256": str(expected["canonical_sha256"]),
        "input_width": int(expected["width"]),
        "input_height": int(expected["height"]),
        "model": "Effort",
        "model_slug": FROZEN_MODEL_SLUG,
        "checkpoint_id": FROZEN_CHECKPOINT_ID,
        "preprocess_profile": FROZEN_PREPROCESS_PROFILE,
        "score_semantics": FROZEN_SCORE_SEMANTICS,
        "classification_threshold": THRESHOLD,
        "classification_threshold_operator": ">",
        "config_fingerprint": manifest["config_fingerprint"],
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "valid_for_t1": True,
        "valid_for_t2": False,
    }


def _audit_run(
    *,
    repo_root: Path,
    run_dir: Path,
    source_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    device_text: str,
    output_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for path in (manifest_path, results_path, expected_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Effort run file: {path}")
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "Effort manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "Effort summary",
    )
    if manifest.get("schema_version") != "effort_detection_run_manifest_v1":
        raise ValueError("Effort manifest schema mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("Effort manifest is not complete")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("Effort manifest run_id mismatch")
    config_record = _require_mapping(manifest.get("config"), "config")
    if _fingerprint(config_record) != manifest.get("config_fingerprint"):
        raise ValueError("Effort manifest config fingerprint mismatch")
    if config_record.get("source_commit") != FROZEN_SOURCE_COMMIT:
        raise ValueError("Effort manifest source pin mismatch")
    if config_record.get("preprocess_profile") != FROZEN_PREPROCESS_PROFILE:
        raise ValueError("Effort manifest preprocess profile mismatch")
    if config_record.get("score_semantics") != FROZEN_SCORE_SEMANTICS:
        raise ValueError("Effort manifest score semantics mismatch")
    if config_record.get("classification_threshold") != THRESHOLD:
        raise ValueError("Effort manifest threshold mismatch")
    if config_record.get("checkpoint_and_protocol_frozen_before_mouse_scores") is not True:
        raise ValueError("Effort manifest lacks pre-score freeze")
    _reject_t2(manifest, path="manifest")
    _reject_t2(summary, path="summary")

    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    _verify_file(
        results_path,
        outputs.get("results_sha256"),
        "Effort results",
    )
    _verify_file(
        summary_path,
        outputs.get("summary_sha256"),
        "Effort summary",
    )
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    _verify_file(
        expected_path,
        dataset.get("expected_inputs_sha256"),
        "Effort expected inputs",
    )
    expected = read_jsonl(expected_path)
    physical = read_jsonl(results_path)
    if len(expected) != int(dataset.get("selected_images", -1)):
        raise ValueError("Effort expected image count mismatch")
    expected_ids = [str(row["sample_id"]) for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Effort expected IDs are duplicated")
    latest = _latest_rows(physical)
    if set(latest) != set(expected_ids):
        raise ValueError("Effort latest result coverage mismatch")

    source_evidence = _verify_source(source_root)
    state, hf_config, asset_evidence = _load_assets(
        checkpoint_path,
        config_path,
    )
    device, runtime_evidence = _configure_runtime(device_text)
    recorded_device = _require_mapping(
        manifest.get("runtime"),
        "recorded runtime",
    ).get("device")
    if recorded_device != str(device):
        raise ValueError(
            f"Effort audit device differs from run: {device} != {recorded_device}"
        )
    model, model_evidence = _build_model(state, hf_config, device)
    del state

    replay_rows: list[dict[str, Any]] = []
    artifact_paths: set[Path] = set()
    max_feature_diff = 0.0
    max_logit_diff = 0.0
    max_head_replay_diff = 0.0
    max_probability_diff = 0.0
    max_margin_diff = 0.0
    fresh_forwards = 0
    for index, expected_row in enumerate(expected):
        sample_id = str(expected_row["sample_id"])
        row = latest[sample_id]
        if row.get("status") != "ok" or row.get("valid_for_metrics") is not True:
            raise ValueError(f"Effort latest row is not valid: {sample_id}")
        identity = _expected_identity(expected_row, manifest, repo_root)
        for key, value in identity.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Effort result identity {key} mismatch for {sample_id}"
                )
        evidence = _require_mapping(
            row.get("edit_visibility_evidence"),
            f"visibility evidence {sample_id}",
        )
        if evidence.get("crop") is not None:
            raise ValueError("Effort visibility evidence invents crop")
        input_path = _anchored(Path(str(row["input_path"])), repo_root)
        _verify_file(
            input_path,
            row.get("input_sha256"),
            f"Effort input {sample_id}",
        )
        image, preprocess = _preprocess(input_path)
        if row.get("preprocess") != preprocess:
            raise ValueError(f"Effort preprocess mismatch for {sample_id}")

        artifact_value = row.get("artifact_path")
        if not isinstance(artifact_value, str):
            raise ValueError(f"Effort artifact path missing for {sample_id}")
        artifact = _anchored(Path(artifact_value), repo_root)
        artifact_root = (run_dir / "artifacts").resolve()
        if artifact.parent != artifact_root:
            raise ValueError(f"Effort artifact escapes run for {sample_id}")
        expected_artifact = (artifact_root / f"{sample_id}.npz").resolve()
        if artifact != expected_artifact:
            raise ValueError(f"Effort artifact name mismatch for {sample_id}")
        if artifact in artifact_paths:
            raise ValueError("Effort artifact reused by multiple rows")
        artifact_paths.add(artifact)
        _verify_file(
            artifact,
            row.get("artifact_sha256"),
            f"Effort artifact {sample_id}",
        )
        persisted_feature, persisted_logits = _load_artifact(artifact)
        if _array_sha256(persisted_feature) != row.get(
            "feature_array_sha256"
        ):
            raise ValueError(f"Effort feature hash mismatch for {sample_id}")
        if _array_sha256(persisted_logits) != row.get(
            "class_logits_array_sha256"
        ):
            raise ValueError(f"Effort logits hash mismatch for {sample_id}")

        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            fresh_logits, fresh_feature = model(tensor)
            head_replay = functional.linear(
                fresh_feature,
                model.head.weight,
                model.head.bias,
            )
            probability = torch.softmax(fresh_logits, dim=1)[:, 1]
            margin = fresh_logits[:, 1] - fresh_logits[:, 0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fresh_forwards += 1
        feature_array = np.ascontiguousarray(
            fresh_feature[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        logits_array = np.ascontiguousarray(
            fresh_logits[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        replay_array = np.ascontiguousarray(
            head_replay[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        feature_diff = float(
            np.max(np.abs(feature_array - persisted_feature))
        )
        logit_diff = float(np.max(np.abs(logits_array - persisted_logits)))
        head_diff = float(np.max(np.abs(replay_array - logits_array)))
        max_feature_diff = max(max_feature_diff, feature_diff)
        max_logit_diff = max(max_logit_diff, logit_diff)
        max_head_replay_diff = max(max_head_replay_diff, head_diff)
        if feature_diff != 0.0 or logit_diff != 0.0 or head_diff != 0.0:
            raise ValueError(f"Effort fresh replay differs for {sample_id}")

        class_logits = row.get("class_logits")
        if (
            not isinstance(class_logits, list)
            or len(class_logits) != 2
            or [float(value) for value in logits_array.tolist()]
            != class_logits
        ):
            raise ValueError(f"Effort embedded logits mismatch for {sample_id}")
        score = float(probability[0].item())
        stored_score = _require_finite(
            row.get("ai_score"),
            f"Effort score {sample_id}",
        )
        probability_diff = abs(score - stored_score)
        max_probability_diff = max(max_probability_diff, probability_diff)
        if probability_diff != 0.0:
            raise ValueError(f"Effort probability replay differs for {sample_id}")
        for alias in ("score", "probability", "fake_probability"):
            if row.get(alias) != stored_score:
                raise ValueError(f"Effort score alias {alias} drifted")
        margin_value = float(margin[0].item())
        stored_margin = _require_finite(
            row.get("raw_logit_margin"),
            f"Effort margin {sample_id}",
        )
        margin_diff = abs(margin_value - stored_margin)
        max_margin_diff = max(max_margin_diff, margin_diff)
        if margin_diff != 0.0:
            raise ValueError(f"Effort margin replay differs for {sample_id}")
        if row.get("classification_decision") is not (score > THRESHOLD):
            raise ValueError(f"Effort decision mismatch for {sample_id}")
        replay_rows.append(dict(row))

    disk_artifacts = {
        path.resolve()
        for path in (run_dir / "artifacts").glob("*.npz")
        if path.is_file()
    }
    if disk_artifacts != artifact_paths:
        raise ValueError("Effort artifact directory has missing/extra files")
    if int(outputs.get("artifact_files", -1)) != len(artifact_paths):
        raise ValueError("Effort manifest artifact count mismatch")

    bootstrap_samples = int(config_record["bootstrap_samples"])
    bootstrap_seed = int(config_record["bootstrap_seed"])
    recomputed = summarize_effort_results(
        physical,
        expected,
        threshold=THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    recomputed.update(
        {
            "run_id": manifest["run_id"],
            "model": "Effort",
            "model_slug": FROZEN_MODEL_SLUG,
            "checkpoint_id": FROZEN_CHECKPOINT_ID,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "preprocess_profile": FROZEN_PREPROCESS_PROFILE,
            "config_fingerprint": manifest["config_fingerprint"],
            "runtime_golden_status": _require_mapping(
                manifest.get("runtime_golden"),
                "manifest runtime golden",
            ).get("status"),
            "runtime_golden_fingerprint": _fingerprint(
                _require_mapping(
                    manifest.get("runtime_golden"),
                    "manifest runtime golden",
                )
            ),
            "generated_at": summary.get("generated_at"),
        }
    )
    if recomputed != summary:
        raise ValueError("Effort summary does not exactly recompute")

    result_fingerprints = {
        sample_id: _fingerprint(latest[sample_id])
        for sample_id in expected_ids
    }
    audit = {
        "schema_version": "effort_independent_audit_v1",
        "status": "ok",
        "run_id": manifest["run_id"],
        "audited_at": utc_now(),
        "run_dir": str(run_dir),
        "source": source_evidence,
        "assets": asset_evidence,
        "runtime": runtime_evidence,
        "model": model_evidence,
        "coverage": {
            "expected_images": len(expected),
            "physical_result_rows": len(physical),
            "latest_images": len(latest),
            "fresh_full_model_forwards": fresh_forwards,
            "artifact_replays": len(artifact_paths),
            "complete_pairs": int(
                recomputed["paired_coverage"]["complete_valid_pairs"]
            ),
        },
        "replay": {
            "max_abs_feature_diff": max_feature_diff,
            "max_abs_class_logit_diff": max_logit_diff,
            "max_abs_head_replay_diff": max_head_replay_diff,
            "max_abs_probability_diff": max_probability_diff,
            "max_abs_margin_diff": max_margin_diff,
            "all_decisions_exact": True,
            "all_preprocess_records_exact": True,
            "summary_exact_recompute": True,
        },
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "results_sha256": sha256_file(results_path),
            "expected_inputs_sha256": sha256_file(expected_path),
            "summary_sha256": sha256_file(summary_path),
            "config_fingerprint": manifest["config_fingerprint"],
            "result_fingerprints": result_fingerprints,
        },
        "recomputed_summary": recomputed,
    }
    _reject_t2(audit, path="audit")
    atomic_write_json(output_path, audit)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-config", type=Path, default=DEFAULT_HF_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    results_root = _anchored(args.results_dir, repo_root)
    run_dir = (results_root / args.run_id).resolve()
    if run_dir.parent != results_root.resolve():
        raise ValueError("Effort audit run-id escapes results directory")
    source_root = _anchored(args.source_root, repo_root)
    checkpoint = _anchored(args.checkpoint, repo_root)
    config = _anchored(args.hf_config, repo_root)
    output = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else run_dir / "independent_audit.json"
    )
    if output.parent != run_dir:
        raise ValueError("Effort audit output must be inside run directory")
    audit = _audit_run(
        repo_root=repo_root,
        run_dir=run_dir,
        source_root=source_root,
        checkpoint_path=checkpoint,
        config_path=config,
        device_text=args.device,
        output_path=output,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "run_id": audit["run_id"],
                "coverage": audit["coverage"],
                "replay": audit["replay"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
