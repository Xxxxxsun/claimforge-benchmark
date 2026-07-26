import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource import analyze_dinov3_iml_run as dinov3_iml_analyzer
from eval.opensource import run_dinov3_iml as dinov3_iml_runner
from eval.opensource.analyze_dinov3_iml_run import (
    INTERNAL_LOGIT_SIZE,
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
    MODEL_SLUG,
    NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
    RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
    LocalizationPair,
    _bilinear_align_corners_false,
    _nearest_resize_mask,
    _preprocess_evidence,
    _quintiles,
    _reject_t1_contract,
    _sigmoid_float32,
    audit_artifacts,
    audit_prefix_reproducibility,
    summarize_result_history,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
)
from eval.opensource.dinov3_iml_metrics import (
    binary_pixel_metrics_strict,
    summarize_dinov3_iml_results,
)


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _save_l(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)


class ArtifactFixture:
    def __init__(self, root: Path):
        self.root = root
        self.image_dir = root / "images"
        self.artifact_dir = root / "artifacts"
        self.real_image = self.image_dir / "real.png"
        self.forged_image = self.image_dir / "forged.png"
        self.gt_path = self.image_dir / "forged_gt.png"

        real_pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        real_pixels[..., 1] = 90
        forged_pixels = real_pixels.copy()
        forged_pixels[:, :2, 0] = 210
        _save_rgb(self.real_image, real_pixels)
        _save_rgb(self.forged_image, forged_pixels)
        self.target = np.asarray(
            [[True, True, False], [True, True, False]],
            dtype=bool,
        )
        _save_l(
            self.gt_path,
            np.where(self.target, 255, 0).astype(np.uint8),
        )
        self.real = self._result("real", self.real_image)
        self.forged = self._result("forged", self.forged_image)
        self.input_row = {
            "sample_id": "forged",
            "task_id": "task",
            "kind": "forged",
            "gt_mask_kind": "exact_diff",
            "gt_mask_path": str(self.gt_path),
            "gt_mask_sha256": sha256_file(self.gt_path),
            "edit_region_xyxy": [0, 0, 2, 2],
        }
        self.pair = LocalizationPair(
            task_id="task",
            domain="lodging",
            real=self.real,
            forged=self.forged,
            input_row=self.input_row,
        )

    def _result(self, kind: str, image_path: Path) -> dict:
        logits = np.full(
            (INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE),
            np.float32(-2.0),
            dtype=np.float32,
        )
        if kind == "forged":
            logits[:, :21] = np.float32(2.0)
        resized_logits = _bilinear_align_corners_false(
            logits,
            width=MODEL_INPUT_SIZE,
            height=MODEL_INPUT_SIZE,
        )
        model_score = _sigmoid_float32(resized_logits)
        native_score = _bilinear_align_corners_false(
            model_score,
            width=3,
            height=2,
        )
        target = self.target if kind == "forged" else np.zeros((2, 3), dtype=bool)

        logits_path = self.artifact_dir / kind / "logits.npy"
        resized_logits_path = self.artifact_dir / kind / "resized_logits.npy"
        model_path = self.artifact_dir / kind / "model.npy"
        native_path = self.artifact_dir / kind / "native.npy"
        mask_path = self.artifact_dir / kind / "mask.png"
        _save_npy(logits_path, logits)
        _save_npy(resized_logits_path, resized_logits)
        _save_npy(model_path, model_score)
        _save_npy(native_path, native_score)
        _save_l(
            mask_path,
            np.where(
                native_score > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            ),
        )
        evidence, _, native_size = _preprocess_evidence(image_path)
        assert native_size == (3, 2)
        return {
            "id": kind,
            "status": "ok",
            "task_id": "task",
            "pair_rank": 0,
            "domain": "lodging",
            "kind": kind,
            "label": int(kind == "forged"),
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_size": [3, 2],
            "preprocess": evidence,
            "raw_logits_model_path": str(logits_path),
            "raw_logits_model_sha256": sha256_file(logits_path),
            "raw_logits_model_shape": [
                INTERNAL_LOGIT_SIZE,
                INTERNAL_LOGIT_SIZE,
            ],
            "raw_logits_model_dtype": "float32",
            "raw_logits_model_semantics": ("official_seg_head_pre_resize_logits"),
            "raw_logits_capture": "one_forward_hook_on_author_model_seg_head",
            "resized_logits_model_path": str(resized_logits_path),
            "resized_logits_model_sha256": sha256_file(resized_logits_path),
            "resized_logits_model_shape": [
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ],
            "resized_logits_model_dtype": "float32",
            "resized_logits_model_semantics": (
                "official_bilinear_resized_pre_sigmoid_logits"
            ),
            "resized_logits_derivation": (
                "bilinear_seg_head_logits_32_to_512_align_corners_false"
            ),
            "score_map_model_path": str(model_path),
            "score_map_model_sha256": sha256_file(model_path),
            "score_map_model_shape": [
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ],
            "score_map_model_dtype": "float32",
            "score_map_model_semantics": (
                "official_author_predict_sigmoid_probability"
            ),
            "score_map_path": str(native_path),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "score_map_semantics": ("model_512_probability_restored_to_native"),
            "score_map_native_source": "official_model_512_probability",
            "score_map_native_restore": (
                "bilinear_official_model_512_probability_to_native_"
                "align_corners_false"
            ),
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "localization": {
                "model_512": binary_pixel_metrics_strict(
                    model_score,
                    _nearest_resize_mask(
                        target,
                        width=MODEL_INPUT_SIZE,
                        height=MODEL_INPUT_SIZE,
                    ),
                    include_ap=kind == "forged",
                ),
                "native": binary_pixel_metrics_strict(
                    native_score,
                    target,
                    include_ap=kind == "forged",
                ),
            },
        }


def _repro_manifest(ids: list[str]) -> dict:
    ordered = [
        {
            "rank": index,
            "pair_rank": index // 2,
            "sample_id": sample_id,
            "task_id": f"task-{index // 2}",
            "kind": "real" if index % 2 == 0 else "forged",
            "label": index % 2,
            "canonical_path": f"images/{sample_id}.jpg",
            "canonical_sha256": f"{index:064x}",
            "gt_mask_sha256": (None if index % 2 == 0 else f"{index + 1:064x}"),
        }
        for index, sample_id in enumerate(ids)
    ]
    return {
        "ordered_inputs": ordered,
        "runtime_contract": {"python": "test"},
        "model": {"name": "DINOv3-IML"},
        "inference": {"seed": 1},
        "metrics": {"bootstrap_samples": 3, "task": "T2"},
        "artifacts": {"format": "npy"},
    }


def _repro_rows(ids: list[str]) -> list[dict]:
    rows = []
    for index, sample_id in enumerate(ids):
        rows.append(
            {
                "id": sample_id,
                "status": "ok",
                "raw_logits_model_sha256": f"{10 + index:064x}",
                "raw_logits_model_shape": [32, 32],
                "raw_logits_model_dtype": "float32",
                "raw_logits_model_semantics": "raw logits",
                "raw_logits_capture": "capture",
                "resized_logits_model_sha256": f"{15 + index:064x}",
                "resized_logits_model_shape": [512, 512],
                "resized_logits_model_dtype": "float32",
                "resized_logits_model_semantics": "resized logits",
                "resized_logits_derivation": "32 to 512 bilinear",
                "score_map_model_sha256": f"{20 + index:064x}",
                "score_map_model_shape": [512, 512],
                "score_map_model_dtype": "float32",
                "score_map_model_semantics": "probability",
                "score_map_sha256": f"{30 + index:064x}",
                "score_map_shape": [2, 3],
                "score_map_dtype": "float32",
                "score_map_semantics": "native",
                "score_map_native_source": "model probability",
                "score_map_native_restore": "valid crop then bilinear",
                "mask_sha256": f"{40 + index:064x}",
                "mask_shape": [2, 3],
                "mask_dtype": "uint8",
                "mask_threshold": 0.5,
                "mask_threshold_operator": ">",
                "localization": {"native": {"f1": index / 10}},
                "preprocess": {"tensor_sha256": f"{50 + index:064x}"},
            }
        )
    return rows


class ProvenanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "dinov3_iml-test"
        self.source_root = root / "upstream"
        self.source_root.mkdir()
        (self.source_root / "LICENSE").write_text(
            "MIT test license\n",
            encoding="utf-8",
        )
        (self.source_root / "models.py").write_text(
            "source\n",
            encoding="utf-8",
        )
        self.source_files = {
            name: sha256_file(self.source_root / name)
            for name in ("LICENSE", "models.py")
        }
        self.dinov3_root = root / "dinov3"
        self.dinov3_root.mkdir()
        (self.dinov3_root / "LICENSE.md").write_text(
            "DINOv3 test license\n",
            encoding="utf-8",
        )
        (self.dinov3_root / "vision_transformer.py").write_text(
            "architecture\n",
            encoding="utf-8",
        )
        self.dinov3_source_files = {
            name: sha256_file(self.dinov3_root / name)
            for name in ("LICENSE.md", "vision_transformer.py")
        }
        self.checkpoint_path = root / "checkpoint-48.pth"
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.checkpoint = {
            "provider": "official_author_google_drive",
            "google_drive_file_id": "test-file-id",
            "original_filename": "checkpoint-48.pth",
            "bytes": self.checkpoint_path.stat().st_size,
            "sha256": sha256_file(self.checkpoint_path),
            "container": ("mapping_with_model_optimizer_epoch_scaler_args"),
            "top_level_keys": [
                "model",
                "optimizer",
                "epoch",
                "scaler",
                "args",
            ],
            "epoch": 48,
            "parameters": 4,
            "buffers": 0,
            "trainable_parameters": 1,
        }
        self.commit = "3" * 40
        self.dinov3_commit = "4" * 40
        self.pins = SimpleNamespace(
            MODEL_REPO_URL="https://github.com/Irennnne/DINOv3-IML",
            MODEL_SOURCE_COMMIT=self.commit,
            SOURCE_FILES=self.source_files,
            DINOV3_REPO_URL="https://github.com/facebookresearch/dinov3",
            DINOV3_SOURCE_COMMIT=self.dinov3_commit,
            DINOV3_SOURCE_FILES=self.dinov3_source_files,
            CHECKPOINT=self.checkpoint,
        )

        image = root / "image.png"
        gt = root / "gt.png"
        _save_rgb(image, np.zeros((2, 2, 3), dtype=np.uint8))
        target = np.asarray([[True, False], [True, False]], dtype=bool)
        _save_l(gt, np.where(target, 255, 0).astype(np.uint8))
        self.input_rows = []
        for index, kind in enumerate(("real", "forged")):
            self.input_rows.append(
                {
                    "rank": index,
                    "pair_rank": 0,
                    "sample_id": kind,
                    "task_id": "task",
                    "domain": "lodging",
                    "kind": kind,
                    "label": index,
                    "canonical_path": str(image),
                    "canonical_sha256": sha256_file(image),
                    "gt_mask_path": None if kind == "real" else str(gt),
                    "gt_mask_kind": ("all_zero" if kind == "real" else "exact_diff"),
                    "gt_mask_sha256": (None if kind == "real" else sha256_file(gt)),
                    "edit_region_xyxy": [0, 0, 1, 2],
                    "width": 2,
                    "height": 2,
                }
            )
        self.inputs_path = root / "inputs.jsonl"
        atomic_write_jsonl(self.inputs_path, self.input_rows)
        self.inputs_sha256 = sha256_file(self.inputs_path)
        self.dataset_path = root / "dataset.json"
        atomic_write_json(
            self.dataset_path,
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "contract_sha256": "contract",
                "inputs_path": str(self.inputs_path),
                "inputs_sha256": self.inputs_sha256,
            },
        )
        self.adapter_path = root / "adapter.py"
        self.adapter_path.write_text("adapter\n", encoding="utf-8")
        self.artifact_dir = root / "artifacts"
        self.runtime_contract = {
            "python": {
                "implementation": "CPython",
                "version": "test",
                "executable": "/test/python",
            },
            "packages": {
                name: {"module": name}
                for name in (
                    "torch",
                    "peft",
                    "transformers",
                    "accelerate",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                    "Pillow",
                    "scikit-learn",
                )
            },
            "optional_imdlbenco_present": False,
            "accelerator": {"requested_device": "cuda:0"},
            "numerical_flags": {
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
        }
        self._install_manifest(self._manifest())

    def _install_manifest(self, manifest: dict) -> None:
        self.manifest = manifest
        self.fingerprint = self.manifest["fingerprint"]
        self.result_rows = [self._result(row) for row in self.input_rows]
        self.summary = summarize_dinov3_iml_results(
            self.result_rows,
            self.input_rows,
            bootstrap_samples=3,
            seed=42,
        )
        self.summary.update(
            {
                "run_id": self.run_id,
                "condition": "test",
                "model": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "checkpoint_sha256": self.checkpoint["sha256"],
                "checkpoint_epoch": 48,
                "input_manifest_sha256": self.inputs_sha256,
                "run_manifest_fingerprint": self.fingerprint,
                "valid_for_t1": False,
                "valid_for_t2": True,
                "t1_policy": "unsupported_no_derived_image_score",
            }
        )

    def _ordered_inputs(self) -> list[dict]:
        return [
            {
                "rank": row["rank"],
                "pair_rank": row["pair_rank"],
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "kind": row["kind"],
                "label": row["label"],
                "canonical_path": row["canonical_path"],
                "canonical_sha256": row["canonical_sha256"],
                "gt_mask_sha256": row["gt_mask_sha256"],
            }
            for row in self.input_rows
        ]

    @staticmethod
    def _inference() -> dict:
        return {
            "precision": "float32",
            "batch_size": 1,
            "deterministic": True,
            "input_source": "canonical_jpeg_original_bytes",
            "decoder": "Pillow.Image.open.convert_RGB",
            "channel_order": "RGB",
            "input_geometry": (
                "direct_stretch_to_512x512_without_aspect_ratio_preservation"
            ),
            "preprocess_protocol": (
                "official_standalone_pillow_rgb_bilinear_stretch_512_imagenet"
            ),
            "resize": "Pillow.Image.resize",
            "resize_interpolation": "Pillow.Image.Resampling.BILINEAR",
            "resize_box": None,
            "resize_reducing_gap": None,
            "input_crop": None,
            "input_reencode": False,
            "normalization": {
                "scale": "float32_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "official_model_output": {
                "seg_head_logits_shape": [1, 1, 32, 32],
                "logit_resize": (
                    "bilinear_seg_head_logits_32_to_512_align_corners_false"
                ),
                "resized_logits_shape": [1, 1, 512, 512],
                "probability": "single_sigmoid_after_logit_resize",
                "captured_by": "one_forward_hook_on_author_model_seg_head",
                "author_predict_calls_per_image": 1,
            },
            "native_compatibility_adapter": {
                "purpose": ("CLAIMFORGE cross-method native-resolution comparison"),
                "source": "official_model_512_probability_not_logits",
                "operation": (
                    "bilinear_official_model_512_probability_to_native_"
                    "align_corners_false"
                ),
                "mode": "bilinear",
                "align_corners": False,
                "threshold_after_restore": True,
                "official_model_space_retained_as_auxiliary": True,
            },
            "mask_threshold": 0.5,
            "mask_threshold_comparison": "strict_greater_than",
            "test_time_augmentation": False,
            "ensemble": False,
            "forward_passes_per_image": 1,
            "seed": 42,
        }

    @staticmethod
    def _metrics_contract() -> dict:
        return {
            "task": "T2_pixel_localization_only",
            "positive_class": "manipulated_pixel",
            "t1_policy": "unsupported_no_derived_image_score",
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_512",
            "mask_threshold": 0.5,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_gt_resize": ("Pillow.Image.Resampling.NEAREST_to_512x512"),
            "forged_pixel_ap_only": True,
            "bootstrap_unit": "task_id_pair",
            "bootstrap_samples": 3,
        }

    @staticmethod
    def _artifacts_contract() -> dict:
        return {
            "raw_logits_model_32": {
                "format": "npy",
                "dtype": "float32",
                "shape": [32, 32],
                "semantics": "official_seg_head_pre_resize_logits",
                "captured_from": "one_forward_hook_on_author_model_seg_head",
            },
            "raw_logits_model_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [512, 512],
                "semantics": "official_bilinear_resized_pre_sigmoid_logits",
                "derivation": (
                    "bilinear_seg_head_logits_32_to_512_align_corners_false"
                ),
            },
            "score_maps_model_512": {
                "format": "npy",
                "dtype": "float32",
                "shape": [512, 512],
                "semantics": "official_author_predict_sigmoid_probability",
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": "model_512_probability_restored_to_native",
                "restore": (
                    "bilinear_official_model_512_probability_to_native_"
                    "align_corners_false"
                ),
            },
            "masks_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "score_map_native > 0.5",
            },
        }

    def _manifest(self) -> dict:
        ordered = self._ordered_inputs()
        checkpoint = {
            **self.checkpoint,
            "path": str(self.checkpoint_path),
            "strict_load": True,
            "safe_weights_only_load": True,
            "safe_globals": ["argparse.Namespace"],
            "container_selection": "top_level_model_only",
            "schema_fallbacks": False,
            "prefix_rewrites": False,
            "full_state_includes_backbone_lora_and_seg_head": True,
            "separate_backbone_weights_required": False,
        }
        manifest = {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": self.run_id,
            "condition": "test",
            "runtime_contract": self.runtime_contract,
            "environment": self.runtime_contract,
            "input": {
                "inputs_manifest": str(self.inputs_path),
                "inputs_sha256": self.inputs_sha256,
                "dataset_manifest": str(self.dataset_path),
                "dataset_manifest_sha256": sha256_file(self.dataset_path),
                "dataset_id": "test-dataset",
                "dataset_contract_sha256": "contract",
                "selection_sha256": hashlib.sha256(
                    stable_json(ordered).encode("utf-8")
                ).hexdigest(),
            },
            "ordered_inputs": ordered,
            "expected_images": 2,
            "expected_complete_pairs": 1,
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "repo_url": self.pins.MODEL_REPO_URL,
                "source_root": str(self.source_root),
                "source_commit": self.commit,
                "source_tracked_clean": True,
                "source_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in self.source_files.items()
                ],
                "dinov3_architecture_source": {
                    "name": "Meta DINOv3",
                    "repo_url": self.pins.DINOV3_REPO_URL,
                    "source_root": str(self.dinov3_root),
                    "source_commit": self.dinov3_commit,
                    "source_tracked_clean": True,
                    "source_files": [
                        {"path": path, "sha256": digest}
                        for path, digest in self.dinov3_source_files.items()
                    ],
                    "pretrained": False,
                    "separate_backbone_weights_loaded": False,
                    "role": "architecture_only",
                    "license": {
                        "path": "LICENSE.md",
                        "sha256": self.dinov3_source_files["LICENSE.md"],
                        "spdx": None,
                        "name": "DINOv3 License Agreement",
                    },
                },
                "variant": "official_CAT_ViT-L16_LoRA-r32_checkpoint-48",
                "license": {
                    "path": "LICENSE",
                    "sha256": self.source_files["LICENSE"],
                    "spdx": "MIT",
                    "scope": "DINOv3-IML_repository_code_only",
                    "checkpoint_license": ("not_separately_stated_by_release"),
                },
                "checkpoint": checkpoint,
                "constructor": {
                    "dinov3_model_type": "dinov3_vitl16",
                    "image_size": 512,
                    "lora_rank": 32,
                    "lora_alpha": 64.0,
                    "lora_target_modules": ["qkv"],
                    "torch_hub_author_calls": 1,
                    "weight_downloads_blocked": True,
                    "author_from_pretrained_used": False,
                    "lora_merged": False,
                },
                "parameter_count": 4,
                "buffer_elements": 0,
                "trainable_parameter_count": 1,
                "supports_image_level_t1": False,
                "image_score_source": None,
                "supports_pixel_level_t2": True,
                "primary_localization_output": (
                    "author_predict_float32_sigmoid_probability"
                ),
            },
            "inference": self._inference(),
            "metrics": self._metrics_contract(),
            "artifacts": self._artifacts_contract(),
            "artifact_dir": str(self.artifact_dir),
            "adapter_contract": [
                {
                    "path": str(self.adapter_path),
                    "sha256": sha256_file(self.adapter_path),
                }
            ],
        }
        immutable = {
            key: value
            for key, value in manifest.items()
            if key not in {"fingerprint", "created_at", "adapter", "environment"}
        }
        manifest["fingerprint"] = hashlib.sha256(
            stable_json(immutable).encode("utf-8")
        ).hexdigest()
        return manifest

    def _result(self, source: dict) -> dict:
        kind = source["kind"]
        target = (
            np.asarray([[True, False], [True, False]], dtype=bool)
            if kind == "forged"
            else np.zeros((2, 2), dtype=bool)
        )
        score = np.asarray(
            [[0.8, 0.2], [0.8, 0.2]],
            dtype=np.float32,
        )
        metrics = binary_pixel_metrics_strict(
            score,
            target,
            include_ap=kind == "forged",
        )
        row_id = source["sample_id"]
        return {
            "schema_version": "opensource_result_v1",
            "run_id": self.run_id,
            "run_manifest_fingerprint": self.fingerprint,
            "input_manifest_sha256": self.inputs_sha256,
            "id": row_id,
            "rank": source["rank"],
            "task_id": source["task_id"],
            "pair_rank": source["pair_rank"],
            "domain": source["domain"],
            "kind": kind,
            "label": source["label"],
            "image_path": source["canonical_path"],
            "image_sha256": source["canonical_sha256"],
            "image_size": [2, 2],
            "gt_mask_kind": source["gt_mask_kind"],
            "gt_mask_sha256": source["gt_mask_sha256"],
            "edit_region_xyxy": source["edit_region_xyxy"],
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "model_source_commit": self.commit,
            "dinov3_source_commit": self.dinov3_commit,
            "checkpoint_sha256": self.checkpoint["sha256"],
            "checkpoint_epoch": 48,
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "status": "ok",
            "valid_for_metrics": True,
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "mask_dtype": "uint8",
            "raw_logits_model_path": str(
                self.artifact_dir / "raw_logits_model_32" / f"{row_id}.npy"
            ),
            "raw_logits_model_sha256": "1" * 64,
            "resized_logits_model_path": str(
                self.artifact_dir / "raw_logits_model_512" / f"{row_id}.npy"
            ),
            "resized_logits_model_sha256": "2" * 64,
            "score_map_model_path": str(
                self.artifact_dir / "score_maps_model_512" / f"{row_id}.npy"
            ),
            "score_map_model_sha256": "3" * 64,
            "score_map_path": str(
                self.artifact_dir / "score_maps_native" / f"{row_id}.npy"
            ),
            "score_map_sha256": "4" * 64,
            "mask_path": str(self.artifact_dir / "masks_native" / f"{row_id}.png"),
            "mask_sha256": "5" * 64,
            "localization": {
                "model_512": metrics,
                "native": metrics,
            },
            "preprocess": {"tensor_sha256": "6" * 64},
        }

    def validate(self) -> dict:
        def git_value(repo: Path, *args: str) -> str:
            if args != ("rev-parse", "HEAD"):
                return ""
            return self.dinov3_commit if repo == self.dinov3_root else self.commit

        with (
            mock.patch.object(
                dinov3_iml_analyzer,
                "_load_runner_pins",
                return_value=self.pins,
            ),
            mock.patch.object(
                dinov3_iml_analyzer,
                "_git_value",
                side_effect=git_value,
            ),
        ):
            return dinov3_iml_analyzer.validate_provenance(
                repo_root=self.root,
                dinov3_iml_root=self.source_root,
                dinov3_root=self.dinov3_root,
                run_id=self.run_id,
                input_path=self.inputs_path,
                input_rows=self.input_rows,
                result_rows=self.result_rows,
                manifest=self.manifest,
                summary=self.summary,
            )


class AnalyzeDINOv3IMLRunTests(unittest.TestCase):
    def test_history_uses_latest_physical_row_and_reports_recovery(self):
        history = summarize_result_history(
            [
                {"id": "a", "status": "error"},
                {"id": "b", "status": "ok"},
                {"id": "a", "status": "ok"},
            ]
        )
        self.assertEqual(history["physical_rows"], 3)
        self.assertEqual(history["unique_ids"], 2)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_ids"], ["a"])
        self.assertEqual(history["latest_status_counts"], {"ok": 2})

    def test_t1_fields_are_rejected_top_level_and_nested(self):
        manifest = {
            "inference": {},
            "metrics": {"t1_policy": "unsupported_no_derived_image_score"},
        }
        summary = {"valid_for_t1": False}
        with self.assertRaisesRegex(ValueError, "forbidden T1"):
            _reject_t1_contract(
                manifest=manifest,
                summary=summary,
                result_rows=[{"valid_for_t1": False, "score": 0.7}],
            )
        with self.assertRaisesRegex(ValueError, "semantic T1"):
            _reject_t1_contract(
                manifest=manifest,
                summary=summary,
                result_rows=[
                    {
                        "valid_for_t1": False,
                        "diagnostic": {"classification_logits": [0.0, 1.0]},
                    }
                ],
            )

    def test_preprocess_forces_square_and_preserves_far_edge(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide.png"
            pixels = np.zeros((4, 13, 3), dtype=np.uint8)
            pixels[..., 1] = 60
            pixels[:, -2:, 0] = 255
            _save_rgb(path, pixels)
            evidence, tensor, native_size = _preprocess_evidence(path)
        self.assertEqual(native_size, (13, 4))
        self.assertEqual(evidence["native_size_wh"], [13, 4])
        self.assertEqual(
            evidence["model_size_wh"],
            [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        )
        self.assertEqual(
            evidence["geometry"],
            "direct_stretch_without_aspect_ratio_preservation",
        )
        self.assertEqual(evidence["resize"], "Pillow.Image.resize")
        self.assertIsNone(evidence["resize_reducing_gap"])
        self.assertGreater(
            float(tensor[0, :, -1].mean()),
            float(tensor[0, :, 0].mean()),
        )

    def test_preprocess_replay_is_byte_exact_with_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "odd.png"
            pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
            _save_rgb(path, pixels)
            evidence, tensor, native_size = _preprocess_evidence(path)
            runner_tensor, runner_native_size, runner_evidence = (
                dinov3_iml_runner.preprocess_image(path)
            )
        self.assertEqual(native_size, runner_native_size)
        self.assertEqual(evidence, runner_evidence)
        self.assertTrue(np.array_equal(tensor, runner_tensor))

    def test_align_corners_false_geometry(self):
        source = np.asarray(
            [[0.0, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        )
        resized = _bilinear_align_corners_false(source, width=4, height=4)
        expected = np.asarray(
            [
                [0.0, 0.25, 0.75, 1.0],
                [0.5, 0.75, 1.25, 1.5],
                [1.5, 1.75, 2.25, 2.5],
                [2.0, 2.25, 2.75, 3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(resized, expected, rtol=0.0, atol=1e-7)

    def test_artifact_audit_accepts_probability_native_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            report = audit_artifacts(
                [fixture.pair],
                repo_root=fixture.root,
                histogram_bins=32,
            )
        self.assertEqual(report["artifact_integrity"]["status"], "ok")
        self.assertEqual(report["artifact_integrity"]["result_images"], 2)
        self.assertEqual(
            fixture.forged["localization"]["model_512"]["pixels"],
            MODEL_INPUT_SIZE * MODEL_INPUT_SIZE,
        )
        self.assertEqual(
            fixture.forged["localization"]["native"]["pixels"],
            6,
        )
        self.assertEqual(
            report["box_hit_at_native_mask_threshold_0_5"]["any_overlap"]["hits"],
            1,
        )
        self.assertFalse(
            report["box_hit_at_native_mask_threshold_0_5"][
                "eligible_for_primary_metrics"
            ]
        )
        self.assertFalse(
            report["localization_threshold_diagnostic"]["eligible_for_primary_metrics"]
        )

    def test_artifact_audit_accepts_bounded_cuda_logit_resize_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.forged["resized_logits_model_path"])
            value = np.load(path, allow_pickle=False)
            observed_cuda_difference = np.float32(4.76837158203125e-6)
            value[0, 0] += observed_cuda_difference
            _save_npy(path, value)
            fixture.forged["resized_logits_model_sha256"] = sha256_file(path)
            report = audit_artifacts(
                [fixture.pair],
                repo_root=fixture.root,
                histogram_bins=None,
            )
        self.assertEqual(report["artifact_integrity"]["status"], "ok")
        self.assertEqual(
            report["artifact_integrity"]["numeric_tolerances"][
                "resized_logits_absolute"
            ],
            RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
        )
        self.assertEqual(
            report["artifact_integrity"]["numeric_tolerances"][
                "model_probability_absolute"
            ],
            MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
        )
        self.assertAlmostEqual(
            report["artifact_integrity"]["observed_maximum_absolute_error"][
                "resized_logits_replay"
            ],
            float(observed_cuda_difference),
        )

    def test_artifact_audit_accepts_bounded_model_probability_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.forged["score_map_model_path"])
            value = np.load(path, allow_pickle=False)
            observed_cuda_difference = np.float32(5.9604644775390625e-7)
            value[0, 0] += observed_cuda_difference
            _save_npy(path, value)
            fixture.forged["score_map_model_sha256"] = sha256_file(path)
            fixture.forged["localization"]["model_512"] = (
                binary_pixel_metrics_strict(
                    value,
                    _nearest_resize_mask(
                        fixture.target,
                        width=MODEL_INPUT_SIZE,
                        height=MODEL_INPUT_SIZE,
                    ),
                    include_ap=True,
                )
            )
            report = audit_artifacts(
                [fixture.pair],
                repo_root=fixture.root,
                histogram_bins=None,
            )
        self.assertEqual(report["artifact_integrity"]["status"], "ok")
        self.assertAlmostEqual(
            report["artifact_integrity"]["observed_maximum_absolute_error"][
                "model_probability_replay"
            ],
            float(observed_cuda_difference),
        )

    def test_artifact_audit_rejects_logit_resize_beyond_rounding_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.forged["resized_logits_model_path"])
            value = np.load(path, allow_pickle=False)
            value[0, 0] += np.float32(1e-3)
            _save_npy(path, value)
            fixture.forged["resized_logits_model_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "bilinear resize of captured 32x32 logits",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_artifact_audit_rejects_tampered_model_probability(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.forged["score_map_model_path"])
            value = np.load(path, allow_pickle=False)
            value[0, 0] += np.float32(0.02)
            _save_npy(path, value)
            fixture.forged["score_map_model_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "sigmoid of independently resized 512x512 logits",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_artifact_audit_rejects_sigmoid_after_native_logit_resize(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            _, columns = np.indices((INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE))
            raw_logits = np.where(
                columns % 2,
                np.float32(12.0),
                np.float32(-4.0),
            ).astype(np.float32)
            resized_logits = _bilinear_align_corners_false(
                raw_logits,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
            )
            model_probability = _sigmoid_float32(resized_logits)
            correct_native = _bilinear_align_corners_false(
                model_probability,
                width=3,
                height=2,
            )
            wrong_native = _sigmoid_float32(
                _bilinear_align_corners_false(
                    resized_logits,
                    width=3,
                    height=2,
                )
            )
            self.assertFalse(
                np.allclose(
                    wrong_native,
                    correct_native,
                    rtol=0.0,
                    atol=NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
                )
            )
            artifacts = (
                (
                    "raw_logits_model_path",
                    "raw_logits_model_sha256",
                    raw_logits,
                ),
                (
                    "resized_logits_model_path",
                    "resized_logits_model_sha256",
                    resized_logits,
                ),
                (
                    "score_map_model_path",
                    "score_map_model_sha256",
                    model_probability,
                ),
                (
                    "score_map_path",
                    "score_map_sha256",
                    wrong_native,
                ),
            )
            for path_key, hash_key, value in artifacts:
                path = Path(fixture.forged[path_key])
                _save_npy(path, value)
                fixture.forged[hash_key] = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "native probability is not the bilinear",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_artifact_audit_rejects_coordinated_ulp_threshold_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            logits_path = Path(fixture.forged["raw_logits_model_path"])
            resized_logits_path = Path(fixture.forged["resized_logits_model_path"])
            model_path = Path(fixture.forged["score_map_model_path"])

            logits = np.load(logits_path, allow_pickle=False)
            logits[:, -1] = np.float32(0.0)
            resized_logits = _bilinear_align_corners_false(
                logits,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
            )
            model = _sigmoid_float32(resized_logits)
            model[0, -1] = np.nextafter(
                np.float32(0.5),
                np.float32(1.0),
            )
            _save_npy(logits_path, logits)
            _save_npy(resized_logits_path, resized_logits)
            _save_npy(model_path, model)
            for key, path in (
                ("raw_logits_model_sha256", logits_path),
                ("resized_logits_model_sha256", resized_logits_path),
                ("score_map_model_sha256", model_path),
            ):
                fixture.forged[key] = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "model probability threshold map",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_provenance_recomputes_summary_and_checks_contract_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            report = fixture.validate()
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["latest_result_rows_validated"], 2)
            self.assertEqual(report["adapter_contract_files_validated"], 1)
            fixture.summary["localization_forged"]["native"]["pixel_ap"]["mean"] = 0.0
            with self.assertRaisesRegex(
                ValueError,
                "recomputed summary field localization_forged",
            ):
                fixture.validate()

    def test_manifest_contract_literals_match_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ProvenanceFixture(root)
            args = SimpleNamespace(
                device="cpu",
                run_id=fixture.run_id,
                condition="test",
                seed=42,
                mask_threshold=0.5,
                bootstrap_samples=3,
                sample_id=None,
            )
            release = {
                "dataset_id": "test-dataset",
                "contract_sha256": "contract",
                "inputs_sha256": fixture.inputs_sha256,
                "jpeg": {"quality": 95},
            }
            with (
                mock.patch.object(
                    dinov3_iml_runner,
                    "_runtime_contract",
                    return_value=fixture.runtime_contract,
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "_git_value",
                    return_value="",
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "MODEL_SOURCE_COMMIT",
                    fixture.commit,
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "SOURCE_FILES",
                    fixture.source_files,
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "DINOV3_SOURCE_COMMIT",
                    fixture.dinov3_commit,
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "DINOV3_SOURCE_FILES",
                    fixture.dinov3_source_files,
                ),
                mock.patch.object(
                    dinov3_iml_runner,
                    "CHECKPOINT",
                    fixture.checkpoint,
                ),
            ):
                manifest = dinov3_iml_runner.build_run_manifest(
                    args=args,
                    repo_root=root,
                    dataset_manifest_path=fixture.dataset_path,
                    release=release,
                    inputs_path=fixture.inputs_path,
                    selected=fixture.input_rows,
                    dinov3_iml_root=fixture.source_root,
                    dinov3_root=fixture.dinov3_root,
                    checkpoint_path=fixture.checkpoint_path,
                    artifact_dir=fixture.artifact_dir,
                )
            self.assertEqual(manifest["inference"], fixture._inference())
            self.assertEqual(manifest["metrics"], fixture._metrics_contract())
            self.assertEqual(
                manifest["artifacts"],
                fixture._artifacts_contract(),
            )
            fixture._install_manifest(manifest)
            report = fixture.validate()
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["adapter_contract_files_validated"], 4)

    def test_provenance_requires_dinov3_iml_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            del fixture.manifest["runtime_contract"]["packages"]["transformers"]
            immutable = {
                key: value
                for key, value in fixture.manifest.items()
                if key
                not in {
                    "fingerprint",
                    "created_at",
                    "adapter",
                    "environment",
                }
            }
            fingerprint = hashlib.sha256(
                stable_json(immutable).encode("utf-8")
            ).hexdigest()
            fixture.manifest["fingerprint"] = fingerprint
            fixture.manifest["environment"] = fixture.manifest["runtime_contract"]
            fixture.summary["run_manifest_fingerprint"] = fingerprint
            for row in fixture.result_rows:
                row["run_manifest_fingerprint"] = fingerprint
            with self.assertRaisesRegex(
                ValueError,
                "missing packages.*transformers",
            ):
                fixture.validate()

    def test_provenance_rejects_dinov3_architecture_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            (fixture.dinov3_root / "vision_transformer.py").write_text(
                "drifted architecture\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "DINOv3 architecture source file.*SHA-256 mismatch",
            ):
                fixture.validate()

    def test_prefix_reproducibility_uses_latest_and_rejects_drift(self):
        ids = ["real-0", "forged-0", "real-1", "forged-1"]
        report = audit_prefix_reproducibility(
            full_manifest=_repro_manifest(ids),
            full_rows=_repro_rows(ids),
            prefix_manifest=_repro_manifest(ids[:2]),
            prefix_rows=[
                {"id": "real-0", "status": "error"},
                *_repro_rows(ids[:2]),
            ],
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["prefix_images"], 2)

        drifted = copy.deepcopy(_repro_rows(ids[:2]))
        drifted[1]["mask_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "mask_sha256"):
            audit_prefix_reproducibility(
                full_manifest=_repro_manifest(ids[:2]),
                full_rows=_repro_rows(ids[:2]),
                prefix_manifest=_repro_manifest(ids[:2]),
                prefix_rows=drifted,
            )

    def test_quintiles_are_pair_preserving_and_deterministic(self):
        pairs = []
        for index, positives in enumerate((5, 1, 3, 2, 4, 6)):
            pairs.append(
                LocalizationPair(
                    task_id=f"task-{index}",
                    domain="lodging",
                    real={},
                    forged={
                        "pair_rank": index,
                        "localization": {
                            "native": {
                                "target_positive_pixels": positives,
                                "pixels": 10,
                            }
                        },
                    },
                    input_row={},
                )
            )
        quintiles = _quintiles(pairs)
        flattened = [pair for _, chunk in quintiles for pair in chunk]
        self.assertEqual(len(quintiles), 5)
        self.assertEqual(len(flattened), len(pairs))
        self.assertEqual(flattened[0].edit_fraction, 0.1)
        self.assertEqual(flattened[-1].edit_fraction, 0.6)


if __name__ == "__main__":
    unittest.main()
