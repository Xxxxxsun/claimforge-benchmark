import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.analyze_imlvit_run import (
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    MODEL_SLUG,
    LocalizationPair,
    _bilinear_align_corners_false,
    _binary_pixel_metrics_strict,
    _preprocess_evidence,
    _quintiles,
    _reject_t1_contract,
    _sigmoid_float32,
    audit_artifacts,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
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

        real_pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        real_pixels[..., 1] = 90
        forged_pixels = real_pixels.copy()
        forged_pixels[:, 0, 0] = 210
        _save_rgb(self.real_image, real_pixels)
        _save_rgb(self.forged_image, forged_pixels)
        self.target = np.asarray(
            [[True, False], [True, False]],
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
            "edit_region_xyxy": [0, 0, 1, 2],
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
            -2.0,
            dtype=np.float32,
        )
        if kind == "forged":
            logits[:2, :2] = np.asarray(
                [[2.0, -2.0], [2.0, -2.0]],
                dtype=np.float32,
            )
        model_score = _sigmoid_float32(logits)
        native_score = np.ascontiguousarray(model_score[:2, :2])
        target = (
            self.target
            if kind == "forged"
            else np.zeros((2, 2), dtype=bool)
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
        evidence, _, native_size, resized_size = _preprocess_evidence(image_path)
        self.asserted_sizes = (native_size, resized_size)
        return {
            "id": kind,
            "task_id": "task",
            "pair_rank": 0,
            "domain": "lodging",
            "kind": kind,
            "label": int(kind == "forged"),
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_size": [2, 2],
            "preprocess": evidence,
            "model_valid_content_size": [2, 2],
            "raw_logits_model_path": str(logits_path),
            "raw_logits_model_sha256": sha256_file(logits_path),
            "raw_logits_model_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "raw_logits_model_dtype": "float32",
            "score_map_model_path": str(model_path),
            "score_map_model_sha256": sha256_file(model_path),
            "score_map_model_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "score_map_model_dtype": "float32",
            "score_map_path": str(native_path),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": [2, 2],
            "score_map_dtype": "float32",
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": [2, 2],
            "localization": {
                "model_1024": _binary_pixel_metrics_strict(
                    model_score[:2, :2],
                    target,
                    include_ap=kind == "forged",
                ),
                "native": _binary_pixel_metrics_strict(
                    native_score,
                    target,
                    include_ap=kind == "forged",
                ),
            },
        }


class ProvenanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "imlvit_test"
        self.inputs_path = root / "inputs.jsonl"
        self.dataset_path = root / "dataset.json"
        self.source_root = root / "upstream"
        self.source_root.mkdir()
        (self.source_root / "README.md").write_text(
            "source\n",
            encoding="utf-8",
        )
        (self.source_root / "LICENSE").write_text(
            "license\n",
            encoding="utf-8",
        )
        self.source_files = {
            "README.md": sha256_file(self.source_root / "README.md"),
            "LICENSE": sha256_file(self.source_root / "LICENSE"),
        }
        self.checkpoint_path = root / "checkpoint.pth"
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.checkpoint = {
            "provider": "official_author_google_drive",
            "announcement_commit": "a" * 40,
            "release_folder_id": "folder",
            "file_id": "file",
            "original_filename": "iml-vit_checkpoint_trufor_20231104.pth",
            "release_file_mtime_utc": "2024-03-24T06:52:08+00:00",
            "bytes": self.checkpoint_path.stat().st_size,
            "sha256": sha256_file(self.checkpoint_path),
            "container": "collections.OrderedDict_raw_state_dict",
            "state_keys": 2,
            "tensor_values": 2,
            "state_elements": 4,
            "tensor_bytes": 16,
            "state_dtypes": {"torch.float32": 2},
            "parameters": 3,
            "buffers": 1,
        }
        self.commit = "b" * 40
        self.pins = SimpleNamespace(
            MODEL_REPO_URL="https://github.com/SunnyHaze/IML-ViT",
            MODEL_SOURCE_COMMIT=self.commit,
            SOURCE_FILES=self.source_files,
            CHECKPOINT=self.checkpoint,
        )

        image = root / "image.png"
        gt = root / "gt.png"
        _save_rgb(image, np.zeros((2, 2, 3), dtype=np.uint8))
        _save_l(gt, np.asarray([[255, 0], [255, 0]], dtype=np.uint8))
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
                    "gt_mask_kind": "all_zero" if kind == "real" else "exact_diff",
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
        self.manifest = self._manifest()
        self.fingerprint = self.manifest["fingerprint"]
        self.result_rows = [
            self._result(row)
            for row in self.input_rows
        ]
        self.summary = {
            "schema_version": "opensource_summary_v1",
            "task_scope": {
                "primary_task": "T2_localization",
                "valid_for_t1": False,
                "valid_for_t2": True,
                "primary_localization_space": "native",
                "auxiliary_localization_space": "model_1024",
                "localization_semantics": (
                    "imlvit_sigmoid_manipulation_probability_float32"
                ),
                "probability_dtype": "float32",
                "mask_threshold": 0.5,
                "threshold_operator": ">",
            },
            "coverage": {
                "expected_images": 2,
                "result_images": 2,
                "valid_images": 2,
                "error_images": 0,
                "missing_images": 0,
            },
            "paired_coverage": {
                "complete_pairs": 1,
                "paired_images": 2,
                "unpaired_valid_images": 0,
            },
            "pair_bootstrap": {
                "bootstrap_samples": 3,
                "seed": 42,
            },
            "run_id": self.run_id,
            "condition": "test",
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": self.checkpoint["sha256"],
            "input_manifest_sha256": self.inputs_sha256,
            "run_manifest_fingerprint": self.fingerprint,
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
        }

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
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "repo_url": self.pins.MODEL_REPO_URL,
                "source_commit": self.commit,
                "source_tracked_clean": True,
                "variant": "official_CAT_TruFor_protocol_checkpoint_20231104",
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
                    "schema_fallbacks": False,
                    "prefix_rewrites": False,
                    "mae_initialization_reloaded": False,
                },
                "parameter_count": self.checkpoint["parameters"],
                "buffer_elements": self.checkpoint["buffers"],
                "supports_image_level_t1": False,
                "image_score_source": None,
                "supports_pixel_level_t2": True,
                "primary_localization_output": (
                    "sigmoid_of_bilinearly_upsampled_predict_head_logits"
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
                "input_geometry": {
                    "protocol_reference": {
                        "paper": "https://arxiv.org/abs/2307.14863",
                        "version": "v4",
                        "section": "4.1",
                    },
                    "paper_protocol": (
                        "if max(H,W)>1024, resize longer side to 1024 while "
                        "preserving aspect ratio; otherwise keep native size; "
                        "top-left place and raw-zero pad right/bottom to 1024"
                    ),
                    "large_image_resize": (
                        "albumentations.LongestMaxSize_max_size_1024_"
                        "cv2_INTER_LINEAR_downscale_only_py3round"
                    ),
                    "small_image_resize": "none",
                    "canvas": [1024, 1024],
                    "placement": "top_left",
                    "padding_value_before_normalization": 0,
                    "crop": None,
                    "reason": (
                        "paper section 4.1; uses the intended conditional "
                        "LongestMaxSize semantics without copying the hosted "
                        "Colab PIL width-height bug, and avoids the README "
                        "demo's destructive top-left crop for large images"
                    ),
                },
                "input_reencode": False,
                "normalization": {
                    "scale": "uint8_divide_255",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "raw_head_output": "one_channel_logits_at_256x256",
                "model_logit_restore": (
                    "bilinear_to_1024x1024_align_corners_false"
                ),
                "model_probability": "single_sigmoid_after_logit_restore",
                "native_restore": (
                    "crop_right_bottom_padding_then_bilinear_probability_to_"
                    "native_align_corners_false"
                ),
                "mask_threshold": 0.5,
                "mask_threshold_comparison": "strict_greater_than",
                "test_time_augmentation": False,
                "ensemble": False,
            },
            "metrics": {
                "task": "T2_pixel_localization_only",
                "positive_class": "manipulated_pixel",
                "t1_policy": "unsupported_no_derived_image_score",
                "mask_threshold": 0.5,
                "threshold_comparison": "strict_greater_than",
                "prediction_inversion": False,
                "localization_spaces": ["model_1024", "native"],
                "model_space_policy": (
                    "metrics use only the valid resized-content rectangle; "
                    "right/bottom padding is excluded"
                ),
                "model_space_gt_resize": "cv2_INTER_NEAREST",
                "forged_pixel_ap_only": True,
                "bootstrap_samples": 3,
                "bootstrap_unit": "task_id_pair",
            },
            "artifacts": {
                "raw_logits_model_1024": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": [1024, 1024],
                },
                "score_maps_model_1024": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": [1024, 1024],
                },
                "score_maps_native": {
                    "format": "npy",
                    "dtype": "float32",
                    "shape": "native_HxW",
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
            "environment": {},
        }

    def _result(self, row: dict) -> dict:
        return {
            "schema_version": "opensource_result_v1",
            "run_id": self.run_id,
            "run_manifest_fingerprint": self.fingerprint,
            "input_manifest_sha256": self.inputs_sha256,
            "id": row["sample_id"],
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
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "status": "ok",
            "valid_for_metrics": True,
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
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
                "eval.opensource.analyze_imlvit_run._load_runner_pins",
                return_value=self.pins,
            ),
            mock.patch(
                "eval.opensource.analyze_imlvit_run._git_value",
                side_effect=fake_git,
            ),
        ):
            return validate_provenance(
                repo_root=self.root,
                imlvit_root=self.source_root,
                run_id=self.run_id,
                input_path=self.inputs_path,
                input_rows=self.input_rows,
                result_rows=self.result_rows,
                manifest=self.manifest,
                summary=self.summary,
            )


class AnalyzeIMLViTRunTests(unittest.TestCase):
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

    def test_t1_fields_are_rejected_top_level_and_semantically_nested(self):
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
                        "diagnostics": {"classification_logits": [0.0, 1.0]},
                    }
                ],
            )

    def test_strict_threshold_does_not_mark_exactly_half_positive(self):
        scores = np.asarray([[0.5, 0.500001]], dtype=np.float32)
        truth = np.asarray([[True, True]], dtype=bool)
        metrics = _binary_pixel_metrics_strict(
            scores,
            truth,
            include_ap=False,
        )
        self.assertEqual(metrics["predicted_positive_pixels"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["threshold_operator"], ">")

    def test_numpy_bilinear_uses_align_corners_false_half_pixel_geometry(self):
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
        diagnostic = report["localization_threshold_diagnostic"]
        self.assertFalse(diagnostic["eligible_for_primary_metrics"])
        self.assertTrue(diagnostic["uses_test_set_labels"])

    def test_artifact_audit_rejects_tampered_sigmoid_relation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.forged["score_map_model_path"])
            score = np.load(path, allow_pickle=False)
            score[100, 100] += np.float32(0.01)
            _save_npy(path, score)
            fixture.forged["score_map_model_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, r"sigmoid\(raw model logits\)"):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_artifact_audit_rejects_non_float32_raw_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            path = Path(fixture.real["raw_logits_model_path"])
            logits = np.load(path, allow_pickle=False).astype(np.float64)
            _save_npy(path, logits)
            fixture.real["raw_logits_model_sha256"] = sha256_file(path)
            fixture.real["raw_logits_model_dtype"] = "float64"
            with self.assertRaisesRegex(
                ValueError,
                "invalid raw model logits for real dtype",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=fixture.root,
                    histogram_bins=None,
                )

    def test_provenance_validates_pins_and_adapter_then_catches_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProvenanceFixture(Path(temporary))
            report = fixture.validate()
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["adapter_contract_files_validated"], 1)
            fixture.adapter_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fixture.validate()

    def test_quintiles_are_pair_preserving_and_deterministic(self):
        pairs = []
        for index, positives in enumerate((5, 1, 3, 2, 4, 6)):
            localization = {
                "native": {
                    "target_positive_pixels": positives,
                    "pixels": 10,
                }
            }
            forged = {
                "pair_rank": index,
                "localization": localization,
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
