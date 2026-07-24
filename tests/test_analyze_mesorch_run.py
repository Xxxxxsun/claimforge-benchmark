import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.analyze_mesorch_run import (
    INTERNAL_LOGIT_SIZE,
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    MODEL_SLUG,
    LocalizationPair,
    _bilinear_align_corners_false,
    _bilinear_align_corners_true,
    _preprocess_evidence,
    _quintiles,
    _reject_t1_contract,
    _sigmoid_float32,
    audit_artifacts,
    audit_prefix_reproducibility,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
)
from eval.opensource.mesorch_metrics import (
    binary_pixel_metrics_strict,
    summarize_mesorch_results,
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
            -2.0,
            dtype=np.float32,
        )
        if kind == "forged":
            logits[:, :85] = np.float32(2.0)
        model_score = _sigmoid_float32(
            _bilinear_align_corners_true(
                logits,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
            )
        )
        native_score = _bilinear_align_corners_false(
            model_score,
            width=3,
            height=2,
        )
        target = (
            self.target
            if kind == "forged"
            else np.zeros((2, 3), dtype=bool)
        )
        model_target = np.repeat(
            np.repeat(target, MODEL_INPUT_SIZE // 2, axis=0),
            171,
            axis=1,
        )[:, :MODEL_INPUT_SIZE]
        # Match cv2 INTER_NEAREST exactly for the non-divisible width.
        import cv2

        model_target = (
            cv2.resize(
                target.astype(np.uint8),
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

        logits_path = self.artifact_dir / kind / "logits.npy"
        model_path = self.artifact_dir / kind / "model.npy"
        native_path = self.artifact_dir / kind / "native.npy"
        mask_path = self.artifact_dir / kind / "mask.png"
        _save_npy(logits_path, logits)
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
        evidence, _, _ = _preprocess_evidence(image_path)
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
            "raw_logits_capture": "pre_hook_input_to_official_model.resize",
            "score_map_model_path": str(model_path),
            "score_map_model_sha256": sha256_file(model_path),
            "score_map_model_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "score_map_model_dtype": "float32",
            "score_map_model_semantics": (
                "official_pred_mask_sigmoid_probability"
            ),
            "score_map_path": str(native_path),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "score_map_semantics": (
                "official_probability_restored_to_native"
            ),
            "score_map_native_restore": (
                "bilinear_align_corners_false_from_512_probability"
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
                    model_target,
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
            "gt_mask_sha256": None if index % 2 == 0 else f"{index + 1:064x}",
        }
        for index, sample_id in enumerate(ids)
    ]
    return {
        "ordered_inputs": ordered,
        "runtime_contract": {"python": "test"},
        "model": {"name": "Mesorch"},
        "inference": {"seed": 1},
        "metrics": {"bootstrap_samples": 3, "task": "T2"},
        "artifacts": {"format": "npy"},
    }


def _repro_rows(ids: list[str]) -> list[dict]:
    rows = []
    for index, sample_id in enumerate(ids):
        common = {
            "id": sample_id,
            "status": "ok",
            "raw_logits_model_sha256": f"{10 + index:064x}",
            "raw_logits_model_shape": [128, 128],
            "raw_logits_model_dtype": "float32",
            "raw_logits_capture": "capture",
            "score_map_model_sha256": f"{20 + index:064x}",
            "score_map_model_shape": [512, 512],
            "score_map_model_dtype": "float32",
            "score_map_model_semantics": "probability",
            "score_map_sha256": f"{30 + index:064x}",
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "score_map_semantics": "native",
            "score_map_native_restore": "bilinear",
            "mask_sha256": f"{40 + index:064x}",
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "localization": {"native": {"f1": index / 10}},
            "preprocess": {"tensor_sha256": f"{50 + index:064x}"},
        }
        rows.append(common)
    return rows


class ProvenanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "mesorch-test"
        self.inputs_path = root / "inputs.jsonl"
        self.dataset_path = root / "dataset.json"
        self.source_root = root / "upstream"
        self.source_root.mkdir()
        (self.source_root / "LICENSE").write_text("license\n", encoding="utf-8")
        (self.source_root / "mesorch.py").write_text(
            "source\n",
            encoding="utf-8",
        )
        self.source_files = {
            name: sha256_file(self.source_root / name)
            for name in ("LICENSE", "mesorch.py")
        }
        self.checkpoint_path = root / "mesorch-98.pth"
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.checkpoint = {
            "provider": "official_author_google_drive",
            "file_id": "file",
            "original_filename": "mesorch-98.pth",
            "last_modified_utc": "2024-12-19T08:31:40+00:00",
            "bytes": self.checkpoint_path.stat().st_size,
            "sha256": sha256_file(self.checkpoint_path),
            "container": "mapping_with_model_optimizer_epoch_scaler_args",
            "top_level_keys": ["model", "optimizer", "epoch", "scaler", "args"],
            "epoch": 98,
            "args_type": "argparse.Namespace",
            "state_container": "collections.OrderedDict",
            "state_keys": 2,
            "state_elements": 4,
            "tensor_bytes": 16,
            "state_dtypes": {"torch.float32": 2},
            "parameters": 4,
            "buffers": 0,
        }
        self.commit = "b" * 40
        self.pins = SimpleNamespace(
            MODEL_REPO_URL="https://github.com/scu-zjz/Mesorch",
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
        atomic_write_jsonl(self.inputs_path, self.input_rows)
        self.inputs_sha256 = sha256_file(self.inputs_path)
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
                )
            },
            "critical_submodules": {
                "IMDLBenCo.registry": {"module": "IMDLBenCo.registry"}
            },
            "accelerator": {"requested_device": "cuda:0"},
        }
        self.manifest = self._manifest()
        self.fingerprint = self.manifest["fingerprint"]
        self.result_rows = [self._result(row) for row in self.input_rows]
        self.summary = summarize_mesorch_results(
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
                "checkpoint_epoch": 98,
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

    def _manifest(self) -> dict:
        ordered = self._ordered_inputs()
        immutable = {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": self.run_id,
            "condition": "test",
            "input": {
                "dataset_id": "test-dataset",
                "dataset_manifest": str(self.dataset_path),
                "dataset_manifest_sha256": sha256_file(self.dataset_path),
                "dataset_contract_sha256": "contract",
                "inputs_manifest": str(self.inputs_path),
                "inputs_sha256": self.inputs_sha256,
                "selection_sha256": hashlib.sha256(
                    stable_json(ordered).encode("utf-8")
                ).hexdigest(),
                "encoding": {},
            },
            "ordered_inputs": ordered,
            "runtime_contract": self.runtime_contract,
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "repo_url": self.pins.MODEL_REPO_URL,
                "source_commit": self.commit,
                "source_tracked_clean": True,
                "variant": "official_MesorchFull_checkpoint_epoch_98",
                "source_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in self.source_files.items()
                ],
                "license": {
                    "path": "LICENSE",
                    "sha256": self.source_files["LICENSE"],
                    "spdx": "MIT",
                    "scope": "project_repository_code_only",
                    "checkpoint_license": "not_separately_stated_by_release",
                },
                "checkpoint": {
                    **self.checkpoint,
                    "path": str(self.checkpoint_path),
                    "strict_load": True,
                    "safe_weights_only_load": True,
                    "safe_globals": ["argparse.Namespace"],
                    "container_selection": "top_level_model_only",
                    "schema_fallbacks": False,
                    "prefix_rewrites": False,
                    "pretraining_weights_reloaded": False,
                },
                "parameter_count": self.checkpoint["parameters"],
                "buffer_elements": self.checkpoint["buffers"],
                "supports_image_level_t1": False,
                "image_score_source": None,
                "supports_pixel_level_t2": True,
                "primary_localization_output": (
                    "official_pred_mask_sigmoid_probability"
                ),
            },
            "inference": {
                "precision": "float32",
                "batch_size": 1,
                "seed": 42,
                "deterministic": True,
                "input_source": "canonical_jpeg_original_bytes",
                "decoder": "Pillow.Image.open.convert_RGB",
                "channel_order": "RGB",
                "input_geometry": (
                    "direct_stretch_to_512x512_without_aspect_ratio_"
                    "preservation"
                ),
                "transform": (
                    "albumentations.Compose([Resize(512,512),"
                    "Normalize(ImageNet),Crop(0,0,512,512),"
                    "ToTensorV2(transpose_mask=True)])"
                ),
                "resize": "albumentations.Resize",
                "resize_interpolation": "cv2.INTER_LINEAR_default",
                "input_crop": None,
                "input_reencode": False,
                "normalization": {
                    "scale": "uint8_divide_255",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "dummy_mask": {
                    "shape": [1, 1, 512, 512],
                    "dtype": "same_as_image_float32",
                    "value": 0,
                    "purpose": (
                        "required only because the official forward computes "
                        "BCEWithLogitsLoss; it does not affect pred_mask"
                    ),
                },
                "official_model_output": {
                    "internal_logits_shape": [1, 1, 128, 128],
                    "internal_resize": (
                        "torch_nn_Upsample_bilinear_to_512_"
                        "align_corners_true"
                    ),
                    "probability": "single_sigmoid_after_internal_resize",
                    "captured_by": (
                        "one_forward_pre_hook_on_model.resize_for_audit"
                    ),
                },
                "native_compatibility_adapter": {
                    "purpose": (
                        "CLAIMFORGE cross-method native-resolution comparison"
                    ),
                    "operation": (
                        "bilinear_restore_official_512_probability_to_native"
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
            },
            "metrics": {
                "task": "T2_pixel_localization_only",
                "positive_class": "manipulated_pixel",
                "t1_policy": "unsupported_no_derived_image_score",
                "primary_localization_space": "native",
                "auxiliary_localization_space": "model_512",
                "mask_threshold": 0.5,
                "threshold_comparison": "strict_greater_than",
                "prediction_inversion": False,
                "native_gt": "exact_canonical_mask",
                "model_space_gt_resize": "cv2_INTER_NEAREST_to_512x512",
                "forged_pixel_ap_only": True,
                "bootstrap_samples": 3,
                "bootstrap_unit": "task_id_pair",
            },
            "artifacts": {
                "raw_logits_model_128": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": [128, 128],
                    "captured_before_official_resize": True,
                },
                "score_maps_model_512": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": [512, 512],
                    "semantics": "official_sigmoid_probability",
                },
                "score_maps_native": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": "native_HxW",
                    "semantics": "bilinearly_restored_probability",
                },
                "masks_native": {
                    "format": "lossless_png",
                    "dtype": "uint8",
                    "values": [0, 255],
                    "relation": "score_map_native > 0.5",
                },
            },
            "expected_pairs": 1,
            "expected_images": 2,
            "artifact_dir": "artifacts",
            "adapter_contract": [
                {
                    "path": str(self.adapter_path),
                    "sha256": sha256_file(self.adapter_path),
                }
            ],
        }
        return {
            **immutable,
            "fingerprint": hashlib.sha256(
                stable_json(immutable).encode("utf-8")
            ).hexdigest(),
            "created_at": "test",
            "adapter": {},
            "environment": self.runtime_contract,
        }

    def _result(self, row: dict) -> dict:
        target = np.asarray(
            [[True, False], [True, False]],
            dtype=bool,
        )
        if row["kind"] == "real":
            target = np.zeros((2, 2), dtype=bool)
            scores = np.full((2, 2), 0.1, dtype=np.float32)
        else:
            scores = np.asarray(
                [[0.9, 0.1], [0.9, 0.1]],
                dtype=np.float32,
            )
        metrics = binary_pixel_metrics_strict(
            scores,
            target,
            include_ap=row["kind"] == "forged",
        )
        sample_id = row["sample_id"]
        artifact_root = self.root / "artifacts"
        return {
            "schema_version": "opensource_result_v1",
            "run_id": self.run_id,
            "run_manifest_fingerprint": self.manifest["fingerprint"],
            "input_manifest_sha256": self.inputs_sha256,
            "id": sample_id,
            "rank": row["rank"],
            "task_id": row["task_id"],
            "pair_rank": row["pair_rank"],
            "domain": row["domain"],
            "kind": row["kind"],
            "label": row["label"],
            "image_path": row["canonical_path"],
            "image_sha256": row["canonical_sha256"],
            "image_size": [row["width"], row["height"]],
            "gt_mask_kind": row["gt_mask_kind"],
            "gt_mask_sha256": row["gt_mask_sha256"],
            "edit_region_xyxy": row["edit_region_xyxy"],
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": self.checkpoint["sha256"],
            "checkpoint_epoch": 98,
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "status": "ok",
            "valid_for_metrics": True,
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "mask_dtype": "uint8",
            "raw_logits_model_path": str(
                artifact_root / "raw_logits_model_128" / f"{sample_id}.npy"
            ),
            "raw_logits_model_sha256": "1" * 64,
            "score_map_model_path": str(
                artifact_root / "score_maps_model_512" / f"{sample_id}.npy"
            ),
            "score_map_model_sha256": "2" * 64,
            "score_map_path": str(
                artifact_root / "score_maps_native" / f"{sample_id}.npy"
            ),
            "score_map_sha256": "3" * 64,
            "mask_path": str(
                artifact_root / "masks_native" / f"{sample_id}.png"
            ),
            "mask_sha256": "4" * 64,
            "localization": {
                "model_512": copy.deepcopy(metrics),
                "native": copy.deepcopy(metrics),
            },
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": 0,
        }

    def validate(self) -> dict:
        def fake_git(_repo, *args):
            if args == ("rev-parse", "HEAD"):
                return self.commit
            if args == ("status", "--short", "--untracked-files=no"):
                return ""
            raise AssertionError(args)

        with (
            mock.patch(
                "eval.opensource.analyze_mesorch_run._load_runner_pins",
                return_value=self.pins,
            ),
            mock.patch(
                "eval.opensource.analyze_mesorch_run._git_value",
                side_effect=fake_git,
            ),
        ):
            return validate_provenance(
                repo_root=self.root,
                mesorch_root=self.source_root,
                run_id=self.run_id,
                input_path=self.inputs_path,
                input_rows=self.input_rows,
                result_rows=self.result_rows,
                manifest=self.manifest,
                summary=self.summary,
            )


class AnalyzeMesorchRunTests(unittest.TestCase):
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

    def test_align_corners_true_geometry(self):
        source = np.asarray(
            [[0.0, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        )
        resized = _bilinear_align_corners_true(source, width=3, height=3)
        expected = np.asarray(
            [
                [0.0, 0.5, 1.0],
                [1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(resized, expected, rtol=0.0, atol=1e-7)

    def test_align_corners_true_matches_cuda_fma_operation_order(self):
        source = np.asarray(
            [
                [0.01904541254043579, 0.018736183643341064],
                [0.15158653259277344, 0.035765767097473145],
            ],
            dtype=np.float32,
        )
        resized = _bilinear_align_corners_true(source, width=5, height=4)

        # Golden bit pattern from PyTorch CUDA bilinear interpolation. The
        # former non-fused NumPy operation order produced 0x3d5b3a6e.
        self.assertEqual(
            int(resized.view(np.uint32)[1, 1]),
            0x3D5B3A6D,
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

    def test_align_corners_false_high_resolution_cuda_fma_golden(self):
        y, x = np.indices((512, 512), dtype=np.int32)
        source = (
            (
                x * 37
                + y * 101
                + (x * y) % 251
            )
            % 1009
        ).astype(np.float32) / np.float32(1009)

        resized = _bilinear_align_corners_false(
            source,
            width=1800,
            height=1200,
        )

        # This full-map golden was reproduced bit-for-bit by PyTorch CUDA. The
        # former non-fused implementation had a different digest and differed
        # at 610,086 pixels for this realistic native-resolution restore.
        self.assertEqual(
            hashlib.sha256(
                np.ascontiguousarray(resized).tobytes(order="C")
            ).hexdigest(),
            "1cccfa97b61fee4b7a8276bce91313cbc187950e937f07e7bfa64442719978ab",
        )

    def test_preprocess_matches_official_runner_transform(self):
        from eval.opensource.run_mesorch import preprocess_image

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
            _save_rgb(path, pixels)
            evidence, tensor, native_size = _preprocess_evidence(path)
            official, official_size, official_evidence = preprocess_image(path)
        self.assertEqual(native_size, official_size)
        self.assertEqual(evidence, official_evidence)
        np.testing.assert_array_equal(tensor, official)

    def test_artifact_audit_accepts_complete_chain_and_box_statistics(self):
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
            report["box_hit_at_native_mask_threshold_0_5"]["any_overlap"][
                "hits"
            ],
            1,
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
            value[100, 100] += np.float32(0.02)
            _save_npy(path, value)
            fixture.forged["score_map_model_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "sigmoid of captured 128x128 logits",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_artifact_audit_rejects_non_float32_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.real["raw_logits_model_path"])
            value = np.load(path, allow_pickle=False).astype(np.float64)
            _save_npy(path, value)
            fixture.real["raw_logits_model_sha256"] = sha256_file(path)
            fixture.real["raw_logits_model_dtype"] = "float64"
            with self.assertRaisesRegex(ValueError, "invalid raw internal logits"):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_provenance_recomputes_summary_and_checks_adapter(self):
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

    def test_provenance_rejects_environment_runtime_contradiction(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            fixture.manifest["environment"] = {"different": True}
            with self.assertRaisesRegex(
                ValueError,
                "environment/runtime contract",
            ):
                fixture.validate()

    def test_provenance_requires_scikit_learn_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            del fixture.manifest["runtime_contract"]["packages"][
                "scikit-learn"
            ]
            immutable = {
                key: value
                for key, value in fixture.manifest.items()
                if key
                not in {"fingerprint", "created_at", "adapter", "environment"}
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
                "missing packages.*scikit-learn",
            ):
                fixture.validate()

    def test_prefix_reproducibility_accepts_latest_recovery(self):
        ids = ["real-0", "forged-0", "real-1", "forged-1"]
        full_manifest = _repro_manifest(ids)
        prefix_manifest = _repro_manifest(ids[:2])
        full_rows = _repro_rows(ids)
        prefix_rows = [
            {"id": "real-0", "status": "error"},
            *_repro_rows(ids[:2]),
        ]
        report = audit_prefix_reproducibility(
            full_manifest=full_manifest,
            full_rows=full_rows,
            prefix_manifest=prefix_manifest,
            prefix_rows=prefix_rows,
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["prefix_images"], 2)

    def test_prefix_reproducibility_rejects_artifact_drift(self):
        ids = ["real-0", "forged-0"]
        full_rows = _repro_rows(ids)
        prefix_rows = copy.deepcopy(full_rows)
        prefix_rows[1]["mask_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "mask_sha256"):
            audit_prefix_reproducibility(
                full_manifest=_repro_manifest(ids),
                full_rows=full_rows,
                prefix_manifest=_repro_manifest(ids),
                prefix_rows=prefix_rows,
            )

    def test_quintiles_are_pair_preserving_and_deterministic(self):
        pairs = []
        for index, positives in enumerate((5, 1, 3, 2, 4, 6)):
            forged = {
                "pair_rank": index,
                "localization": {
                    "native": {
                        "target_positive_pixels": positives,
                        "pixels": 10,
                    }
                },
            }
            pairs.append(
                LocalizationPair(
                    task_id=f"task-{index}",
                    domain="lodging",
                    real={},
                    forged=forged,
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
