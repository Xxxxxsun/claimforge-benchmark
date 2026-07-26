import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource import analyze_relayformer_run as relayformer_analyzer
from eval.opensource.analyze_relayformer_run import (
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    MODEL_SLUG,
    LocalizationPair,
    _bilinear_align_corners_false,
    _pillow_thumbnail_size,
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
from eval.opensource.relayformer_metrics import (
    binary_pixel_metrics_strict,
    summarize_relayformer_results,
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
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            np.float32(-2.0),
            dtype=np.float32,
        )
        if kind == "forged":
            logits[:2, :2] = np.float32(2.0)
        # Deliberately make the padded canvas highly positive. The audit must
        # prove that model-space metrics and native restoration crop it away.
        logits[2:, :] = np.float32(4.0)
        logits[:, 3:] = np.float32(4.0)
        model_score = _sigmoid_float32(logits)
        native_logits = np.ascontiguousarray(logits[:2, :3])
        native_score = np.ascontiguousarray(model_score[:2, :3])
        target = (
            self.target
            if kind == "forged"
            else np.zeros((2, 3), dtype=bool)
        )

        logits_path = self.artifact_dir / kind / "logits.npy"
        native_logits_path = self.artifact_dir / kind / "native_logits.npy"
        model_path = self.artifact_dir / kind / "model.npy"
        native_path = self.artifact_dir / kind / "native.npy"
        mask_path = self.artifact_dir / kind / "mask.png"
        _save_npy(logits_path, logits)
        _save_npy(native_logits_path, native_logits)
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
        evidence, _, native_size, resized_size = _preprocess_evidence(
            image_path
        )
        assert native_size == resized_size == (3, 2)
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
            "model_valid_content_size_wh": [3, 2],
            "raw_logits_model_path": str(logits_path),
            "raw_logits_model_sha256": sha256_file(logits_path),
            "raw_logits_model_shape": [
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ],
            "raw_logits_model_dtype": "float32",
            "raw_logits_capture": (
                "temporary_instance_method_wrapper_output_from_"
                "official_assemble_and_decode"
            ),
            "raw_logits_native_path": str(native_logits_path),
            "raw_logits_native_sha256": sha256_file(native_logits_path),
            "raw_logits_native_shape": [2, 3],
            "raw_logits_native_dtype": "float32",
            "raw_logits_native_semantics": (
                "valid_content_logits_restored_to_native"
            ),
            "raw_logits_native_restore": (
                "crop_right_bottom_padding_then_bilinear_"
                "align_corners_false_if_downscaled"
            ),
            "score_map_model_path": str(model_path),
            "score_map_model_sha256": sha256_file(model_path),
            "score_map_model_shape": [
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ],
            "score_map_model_dtype": "float32",
            "score_map_model_semantics": "official_sigmoid_probability",
            "score_map_path": str(native_path),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "score_map_semantics": (
                "valid_content_probability_restored_to_native"
            ),
            "score_map_native_restore": (
                "crop_right_bottom_padding_then_bilinear_"
                "align_corners_false_if_downscaled"
            ),
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "localization": {
                "model_1024": binary_pixel_metrics_strict(
                    model_score[:2, :3],
                    target,
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
            "gt_mask_sha256": (
                None if index % 2 == 0 else f"{index + 1:064x}"
            ),
        }
        for index, sample_id in enumerate(ids)
    ]
    return {
        "ordered_inputs": ordered,
        "runtime_contract": {"python": "test"},
        "model": {"name": "RelayFormer"},
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
                "raw_logits_model_shape": [1024, 1024],
                "raw_logits_model_dtype": "float32",
                "raw_logits_capture": "capture",
                "raw_logits_native_sha256": f"{15 + index:064x}",
                "raw_logits_native_shape": [2, 3],
                "raw_logits_native_dtype": "float32",
                "raw_logits_native_semantics": "native logits",
                "raw_logits_native_restore": "valid crop then bilinear",
                "score_map_model_sha256": f"{20 + index:064x}",
                "score_map_model_shape": [1024, 1024],
                "score_map_model_dtype": "float32",
                "score_map_model_semantics": "probability",
                "score_map_sha256": f"{30 + index:064x}",
                "score_map_shape": [2, 3],
                "score_map_dtype": "float32",
                "score_map_semantics": "native",
                "score_map_native_restore": "valid crop then bilinear",
                "mask_sha256": f"{40 + index:064x}",
                "mask_shape": [2, 3],
                "mask_dtype": "uint8",
                "mask_threshold": 0.5,
                "mask_threshold_operator": ">",
                "localization": {"native": {"f1": index / 10}},
                "preprocess": {"tensor_sha256": f"{50 + index:064x}"},
                "model_valid_content_size_wh": [3, 2],
            }
        )
    return rows


class ProvenanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "relayformer-test"
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
        self.checkpoint_path = root / "checkpoint-164.pth"
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.checkpoint = {
            "provider": "official_author_huggingface",
            "model_revision": "9" * 40,
            "original_filename": "checkpoint-164.pth",
            "bytes": self.checkpoint_path.stat().st_size,
            "sha256": sha256_file(self.checkpoint_path),
            "container": (
                "mapping_with_model_optimizer_epoch_scaler_args"
            ),
            "top_level_keys": [
                "model",
                "optimizer",
                "epoch",
                "scaler",
                "args",
            ],
            "epoch": 164,
            "parameters": 4,
            "buffers": 0,
        }
        self.commit = "3" * 40
        self.pins = SimpleNamespace(
            MODEL_REPO_URL="https://github.com/WenOOI/RelayFormer",
            MODEL_SOURCE_COMMIT=self.commit,
            SOURCE_FILES=self.source_files,
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
                    "gt_mask_kind": (
                        "all_zero" if kind == "real" else "exact_diff"
                    ),
                    "gt_mask_sha256": (
                        None if kind == "real" else sha256_file(gt)
                    ),
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
                    "torchvision",
                    "timm",
                    "IMDLBenCo",
                    "numpy",
                    "Pillow",
                    "albumentations",
                    "cv2",
                    "scikit-learn",
                    "rotary-embedding-torch",
                )
            },
            "critical_submodules": {
                "rotary_embedding_torch": {
                    "module": "rotary_embedding_torch"
                }
            },
            "accelerator": {"requested_device": "cuda:0"},
        }
        self.manifest = self._manifest()
        self.fingerprint = self.manifest["fingerprint"]
        self.result_rows = [
            self._result(row) for row in self.input_rows
        ]
        self.summary = summarize_relayformer_results(
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
                "checkpoint_epoch": 164,
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
                "paper_v3_long_edge_1024_top_left_zero_pad"
            ),
            "resize_condition": (
                "downscale_only_when_native_long_edge_exceeds_1024"
            ),
            "resize_rounding": (
                "pillow_thumbnail_floor_ceil_min_aspect_error"
            ),
            "resize_interpolation": (
                "Pillow.Image.Resampling.BILINEAR_reducing_gap_None"
            ),
            "model_canvas_size": [1024, 1024],
            "padding": {
                "placement": "top_left",
                "raw_rgb_value": 0,
                "applied_before_normalization": True,
            },
            "input_crop": None,
            "input_reencode": False,
            "normalization": {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "official_model_output": {
                "raw_logits_shape": [1, 1, 1024, 1024],
                "probability": (
                    "single_sigmoid_of_assemble_and_decode_logits"
                ),
                "captured_by": (
                    "temporary_instance_method_wrapper_output_from_"
                    "official_assemble_and_decode"
                ),
            },
            "native_compatibility_adapter": {
                "purpose": (
                    "CLAIMFORGE cross-method native-resolution comparison"
                ),
                "operation": (
                    "crop_right_bottom_padding_then_bilinear_"
                    "align_corners_false_if_downscaled"
                ),
                "mode": "bilinear",
                "align_corners": False,
                "threshold_after_restore": True,
                "official_model_space_retained_as_auxiliary": True,
                "probability_restore_is_independent_of_logit_restore": True,
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
            "auxiliary_localization_space": "model_1024",
            "auxiliary_localization_extent": (
                "resized_valid_content_excluding_right_bottom_padding"
            ),
            "mask_threshold": 0.5,
            "threshold_comparison": "strict_greater_than",
            "prediction_inversion": False,
            "native_gt": "exact_canonical_mask",
            "model_space_scope": (
                "valid_resized_content_only_excluding_right_bottom_padding"
            ),
            "model_space_gt_resize": (
                "Pillow.Image.Resampling.NEAREST_to_resized_valid_content"
            ),
            "forged_pixel_ap_only": True,
            "bootstrap_unit": "task_id_pair",
            "bootstrap_samples": 3,
        }

    @staticmethod
    def _artifacts_contract() -> dict:
        restore = (
            "crop_right_bottom_padding_then_bilinear_"
            "align_corners_false_if_downscaled"
        )
        return {
            "raw_logits_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [1024, 1024],
                "semantics": "official_pre_sigmoid_logits",
                "captured_from": (
                    "temporary_instance_method_wrapper_output_from_"
                    "official_assemble_and_decode"
                ),
            },
            "raw_logits_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": (
                    "valid_content_logits_restored_to_native"
                ),
                "restore": restore,
            },
            "score_maps_model_1024": {
                "format": "npy",
                "dtype": "float32",
                "shape": [1024, 1024],
                "semantics": "official_sigmoid_probability",
            },
            "score_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "semantics": (
                    "valid_content_probability_restored_to_native"
                ),
                "restore": restore,
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
            "pretraining_weights_reloaded": False,
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
                "dataset_manifest_sha256": sha256_file(
                    self.dataset_path
                ),
                "dataset_id": "test-dataset",
                "dataset_contract_sha256": "contract",
                "selection_sha256": hashlib.sha256(
                    stable_json(ordered).encode("utf-8")
                ).hexdigest(),
            },
            "ordered_inputs": ordered,
            "expected_images": 2,
            "expected_pairs": 1,
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "repo_url": self.pins.MODEL_REPO_URL,
                "source_commit": self.commit,
                "source_tracked_clean": True,
                "source_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in self.source_files.items()
                ],
                "variant": (
                    "official_relay_vit_image_only_checkpoint_164"
                ),
                "license": {
                    "path": "LICENSE",
                    "sha256": self.source_files["LICENSE"],
                    "spdx": "MIT",
                    "scope": "project_repository_code_only",
                    "checkpoint_license": (
                        "Apache-2.0_on_official_Hugging_Face_model_card"
                    ),
                },
                "checkpoint": checkpoint,
                "parameter_count": 4,
                "buffer_elements": 0,
                "supports_image_level_t1": False,
                "image_score_source": None,
                "supports_pixel_level_t2": True,
                "primary_localization_output": (
                    "official_sigmoid_probability_tensor"
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
            if key
            not in {"fingerprint", "created_at", "adapter", "environment"}
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
            "checkpoint_sha256": self.checkpoint["sha256"],
            "checkpoint_epoch": 164,
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "status": "ok",
            "valid_for_metrics": True,
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "mask_dtype": "uint8",
            "raw_logits_model_path": str(
                self.artifact_dir
                / "raw_logits_model_1024"
                / f"{row_id}.npy"
            ),
            "raw_logits_model_sha256": "1" * 64,
            "raw_logits_native_path": str(
                self.artifact_dir
                / "raw_logits_native"
                / f"{row_id}.npy"
            ),
            "raw_logits_native_sha256": "2" * 64,
            "score_map_model_path": str(
                self.artifact_dir
                / "score_maps_model_1024"
                / f"{row_id}.npy"
            ),
            "score_map_model_sha256": "3" * 64,
            "score_map_path": str(
                self.artifact_dir
                / "score_maps_native"
                / f"{row_id}.npy"
            ),
            "score_map_sha256": "4" * 64,
            "mask_path": str(
                self.artifact_dir
                / "masks_native"
                / f"{row_id}.png"
            ),
            "mask_sha256": "5" * 64,
            "localization": {
                "model_1024": metrics,
                "native": metrics,
            },
            "preprocess": {"tensor_sha256": "6" * 64},
            "model_valid_content_size_wh": [2, 2],
        }

    def validate(self) -> dict:
        def git_value(repo: Path, *args: str) -> str:
            del repo
            return self.commit if args == ("rev-parse", "HEAD") else ""

        with (
            mock.patch.object(
                relayformer_analyzer,
                "_load_runner_pins",
                return_value=self.pins,
            ),
            mock.patch.object(
                relayformer_analyzer,
                "_git_value",
                side_effect=git_value,
            ),
        ):
            return relayformer_analyzer.validate_provenance(
                repo_root=self.root,
                relayformer_root=self.source_root,
                run_id=self.run_id,
                input_path=self.inputs_path,
                input_rows=self.input_rows,
                result_rows=self.result_rows,
                manifest=self.manifest,
                summary=self.summary,
            )


class AnalyzeRelayFormerRunTests(unittest.TestCase):
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

    def test_thumbnail_size_matches_pillow_for_odd_landscape_and_portrait(self):
        for width, height in ((1537, 777), (777, 1537), (1025, 1024)):
            image = Image.new("RGB", (width, height))
            image.thumbnail(
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                Image.Resampling.BILINEAR,
                reducing_gap=None,
            )
            self.assertEqual(
                _pillow_thumbnail_size(width, height),
                image.size,
            )

    def test_preprocess_preserves_far_edge_and_normalizes_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide.png"
            pixels = np.zeros((400, 1300, 3), dtype=np.uint8)
            pixels[..., 1] = 60
            pixels[:, -20:, 0] = 255
            _save_rgb(path, pixels)
            evidence, tensor, native_size, resized_size = (
                _preprocess_evidence(path)
            )
        self.assertEqual(native_size, (1300, 400))
        self.assertEqual(resized_size[0], MODEL_INPUT_SIZE)
        self.assertEqual(evidence["native_size_wh"], [1300, 400])
        self.assertEqual(
            evidence["padding_ltrb"],
            [0, 0, 0, MODEL_INPUT_SIZE - resized_size[1]],
        )
        # The source's far-right red band survives: this is a resize, not the
        # released code's destructive top-left crop.
        self.assertGreater(
            float(tensor[0, : resized_size[1], resized_size[0] - 1].mean()),
            float(tensor[0, : resized_size[1], 0].mean()),
        )
        expected_zero_red = np.float32(
            (np.float32(0.0) - np.float32(0.485)) / np.float32(0.229)
        )
        self.assertEqual(
            int(tensor[0, -1, -1].view(np.uint32)),
            int(expected_zero_red.view(np.uint32)),
        )

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

    def test_artifact_audit_accepts_chain_and_excludes_padding(self):
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
            fixture.forged["localization"]["model_1024"]["pixels"],
            6,
        )
        self.assertEqual(
            report["box_hit_at_native_mask_threshold_0_5"]["any_overlap"][
                "hits"
            ],
            1,
        )
        self.assertFalse(
            report["box_hit_at_native_mask_threshold_0_5"][
                "eligible_for_primary_metrics"
            ]
        )
        self.assertFalse(
            report["localization_threshold_diagnostic"][
                "eligible_for_primary_metrics"
            ]
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
                "sigmoid of captured 1024x1024 logits",
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
            native_logits_path = Path(
                fixture.forged["raw_logits_native_path"]
            )
            model_path = Path(fixture.forged["score_map_model_path"])
            native_path = Path(fixture.forged["score_map_path"])
            mask_path = Path(fixture.forged["mask_path"])

            logits = np.load(logits_path, allow_pickle=False)
            logits[0, 2] = np.float32(0.0)
            model = _sigmoid_float32(logits)
            model[0, 2] = np.nextafter(
                np.float32(0.5),
                np.float32(1.0),
            )
            native_logits = np.ascontiguousarray(logits[:2, :3])
            native = np.ascontiguousarray(model[:2, :3])
            _save_npy(logits_path, logits)
            _save_npy(native_logits_path, native_logits)
            _save_npy(model_path, model)
            _save_npy(native_path, native)
            _save_l(
                mask_path,
                np.where(native > 0.5, np.uint8(255), np.uint8(0)),
            )
            for key, path in (
                ("raw_logits_model_sha256", logits_path),
                ("raw_logits_native_sha256", native_logits_path),
                ("score_map_model_sha256", model_path),
                ("score_map_sha256", native_path),
                ("mask_sha256", mask_path),
            ):
                fixture.forged[key] = sha256_file(path)
            fixture.forged["localization"]["model_1024"] = (
                binary_pixel_metrics_strict(
                    model[:2, :3],
                    fixture.target,
                    include_ap=True,
                )
            )
            fixture.forged["localization"]["native"] = (
                binary_pixel_metrics_strict(
                    native,
                    fixture.target,
                    include_ap=True,
                )
            )
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
            fixture.summary["localization_forged"]["native"][
                "pixel_ap"
            ]["mean"] = 0.0
            with self.assertRaisesRegex(
                ValueError,
                "recomputed summary field localization_forged",
            ):
                fixture.validate()

    def test_provenance_requires_relayformer_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            del fixture.manifest["runtime_contract"]["packages"][
                "rotary-embedding-torch"
            ]
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
            fixture.manifest["environment"] = fixture.manifest[
                "runtime_contract"
            ]
            fixture.summary["run_manifest_fingerprint"] = fingerprint
            for row in fixture.result_rows:
                row["run_manifest_fingerprint"] = fingerprint
            with self.assertRaisesRegex(
                ValueError,
                "missing packages.*rotary-embedding-torch",
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
