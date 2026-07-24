import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.analyze_psccnet_run import (
    CLASSIFICATION_THRESHOLD,
    MASK_THRESHOLD,
    MODEL_NAME,
    MODEL_REPO_URL,
    MODEL_SLUG,
    Pair,
    _bilinear_align_corners_true,
    _binary_pixel_metrics_strict,
    _detection_point,
    _oracle_histograms,
    _preprocess_evidence,
    _quintiles,
    _recomputed_t1_score,
    _select_manifest_inputs,
    audit_and_best_threshold,
    summarize_psccnet_pair_slice,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
)


def _fake_pins():
    checkpoints = {
        "feature_extractor": {
            "path": "feature.pth",
            "bytes": 11,
            "sha256": "3" * 64,
            "state_keys": 4,
            "state_elements": 5,
            "parameters": 6,
            "buffers": 7,
        },
        "localization_head": {
            "path": "localization.pth",
            "bytes": 12,
            "sha256": "4" * 64,
            "state_keys": 8,
            "state_elements": 9,
            "parameters": 10,
            "buffers": 11,
        },
        "classification_head": {
            "path": "classification.pth",
            "bytes": 13,
            "sha256": "5" * 64,
            "state_keys": 12,
            "state_elements": 13,
            "parameters": 14,
            "buffers": 15,
        },
    }
    source_files = {
        "test.py": "1" * 64,
        "LICENSE": "2" * 64,
    }
    initialization = {
        "path": "initialization.pth",
        "bytes": 17,
        "sha256": "6" * 64,
    }
    bundle = hashlib.sha256(
        stable_json(
            {
                role: contract["sha256"]
                for role, contract in checkpoints.items()
            }
        ).encode("utf-8")
    ).hexdigest()
    return SimpleNamespace(
        MODEL_REPO_URL=MODEL_REPO_URL,
        MODEL_SOURCE_COMMIT="7" * 40,
        SOURCE_FILES=source_files,
        INITIALIZATION_WEIGHT=initialization,
        CHECKPOINTS=checkpoints,
        CHECKPOINT_BUNDLE_SHA256=bundle,
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


def _logits_for_probability(probability: float) -> list[float]:
    return [0.0, math.log(probability / (1.0 - probability))]


def _t1_fields(probability: float) -> dict:
    logits = _logits_for_probability(probability)
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= np.max(shifted)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    score = float(probabilities[1])
    return {
        "score": score,
        "score_source": "native_classification_head",
        "score_semantics": "softmax_probability_class_1_forged",
        "classification_logits": logits,
        "classification_probabilities": probabilities.tolist(),
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "decision": "forged" if score > 0.5 else "authentic",
    }


class PSCCNetFixture:
    def __init__(self, root: Path):
        self.root = root
        self.pins = _fake_pins()
        self.run_id = "psccnet_test"
        self.image_dir = root / "images"
        self.artifact_dir = root / "artifacts"
        self.real_image = self.image_dir / "real.png"
        self.forged_image = self.image_dir / "forged.png"
        self.gt_path = self.image_dir / "forged_gt.png"

        real_pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        real_pixels[..., 1] = 80
        forged_pixels = real_pixels.copy()
        forged_pixels[:, 0, 0] = 180
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

        self.input_rows = [
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "rank": index,
                "pair_rank": 0,
                "sample_id": kind,
                "task_id": "task",
                "domain": "lodging",
                "kind": kind,
                "label": index,
                "canonical_path": f"images/{kind}.png",
                "canonical_sha256": sha256_file(
                    self.real_image if kind == "real" else self.forged_image
                ),
                "gt_mask_path": (
                    None if kind == "real" else "images/forged_gt.png"
                ),
                "gt_mask_kind": (
                    "all_zero" if kind == "real" else "exact_diff"
                ),
                "gt_mask_sha256": (
                    None if kind == "real" else sha256_file(self.gt_path)
                ),
                "edit_region_xyxy": [0, 0, 1, 2],
                "width": 2,
                "height": 2,
            }
            for index, kind in enumerate(("real", "forged"))
        ]
        self.input_path = root / "inputs.jsonl"
        atomic_write_jsonl(self.input_path, self.input_rows)
        self.inputs_sha256 = sha256_file(self.input_path)
        self.dataset_path = root / "dataset.json"
        atomic_write_json(
            self.dataset_path,
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "contract_sha256": "contract",
                "inputs_path": "inputs.jsonl",
                "inputs_sha256": self.inputs_sha256,
            },
        )
        self.dataset_sha256 = sha256_file(self.dataset_path)
        self.adapter_path = root / "adapter.py"
        self.adapter_path.write_text("adapter\n", encoding="utf-8")

        self.ok_rows = {
            "real": self._result("real", probability=0.2),
            "forged": self._result("forged", probability=0.8),
        }
        self.ordered_inputs = [
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
            for row in self.input_rows
        ]
        immutable = self._immutable_manifest()
        self.fingerprint = hashlib.sha256(
            stable_json(immutable).encode("utf-8")
        ).hexdigest()
        self.manifest = {
            **immutable,
            "fingerprint": self.fingerprint,
            "created_at": "test",
            "adapter": {},
            "environment": {},
        }
        identity = {
            kind: self._identity(row)
            for kind, row in zip(("real", "forged"), self.input_rows)
        }
        for kind in ("real", "forged"):
            self.ok_rows[kind] = {
                **identity[kind],
                **self.ok_rows[kind],
            }
        self.error_forged = {
            **identity["forged"],
            "status": "error",
            "valid_for_metrics": False,
            "error_type": "RuntimeError",
        }
        self.result_rows = [
            self.ok_rows["real"],
            self.error_forged,
            self.ok_rows["forged"],
        ]
        self.summary = {
            "schema_version": "opensource_summary_v1",
            "run_id": self.run_id,
            "condition": "test",
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": self.pins.CHECKPOINT_BUNDLE_SHA256,
            "input_manifest_sha256": self.inputs_sha256,
            "run_manifest_fingerprint": self.fingerprint,
            "coverage": {
                "expected_images": 2,
                "result_images": 2,
                "valid_images": 2,
                "error_images": 0,
                "missing_images": 0,
            },
            "task_scope": {
                "primary_task": "T1_detection_and_T2_localization",
                "primary_detection_score": "score",
                "primary_detection_semantics": (
                    "psccnet_image_level_manipulation_probability"
                ),
                "primary_localization_space": "native",
                "primary_localization_semantics": (
                    "psccnet_native_manipulation_probability"
                ),
                "classification_threshold": 0.5,
                "mask_threshold": 0.5,
                "threshold_operator": ">",
            },
        }

    def _identity(self, row: dict) -> dict:
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
            "checkpoint_sha256": self.pins.CHECKPOINT_BUNDLE_SHA256,
            "valid_for_t1": True,
            "valid_for_t2": True,
        }

    def _immutable_manifest(self) -> dict:
        checkpoints = self.pins.CHECKPOINTS
        return {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": self.run_id,
            "condition": "test",
            "input": {
                "dataset_id": "test-dataset",
                "dataset_manifest": "dataset.json",
                "dataset_manifest_sha256": self.dataset_sha256,
                "dataset_contract_sha256": "contract",
                "inputs_manifest": "inputs.jsonl",
                "inputs_sha256": self.inputs_sha256,
                "selection_sha256": hashlib.sha256(
                    stable_json(self.ordered_inputs).encode("utf-8")
                ).hexdigest(),
                "encoding": {},
            },
            "ordered_inputs": self.ordered_inputs,
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "repo_url": self.pins.MODEL_REPO_URL,
                "source_commit": self.pins.MODEL_SOURCE_COMMIT,
                "source_tracked_clean": True,
                "variant": "official_committed_pretrained_checkpoint",
                "training_manipulations": [],
                "source_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in self.pins.SOURCE_FILES.items()
                ],
                "license": {
                    "path": "LICENSE",
                    "sha256": self.pins.SOURCE_FILES["LICENSE"],
                    "spdx": "MIT",
                    "scope": "project_repository",
                },
                "initialization_weight": self.pins.INITIALIZATION_WEIGHT,
                "checkpoint": {
                    "provider": "official_author_git_repository",
                    "source_commit": self.pins.MODEL_SOURCE_COMMIT,
                    "bundle_sha256": self.pins.CHECKPOINT_BUNDLE_SHA256,
                    "components": [
                        {"role": role, **contract}
                        for role, contract in checkpoints.items()
                    ],
                    "strict_load": True,
                    "safe_weights_only_load": True,
                },
                "parameter_count": sum(
                    value["parameters"] for value in checkpoints.values()
                ),
                "buffer_elements": sum(
                    value["buffers"] for value in checkpoints.values()
                ),
                "class_names": ["authentic", "forged"],
                "positive_class_index": 1,
                "supports_image_level_t1": True,
                "image_score_source": "native_independent_classification_head",
                "supports_pixel_level_t2": True,
                "primary_localization_output": "progressive_mask1",
            },
            "inference": {
                "precision": "float32",
                "batch_size": 1,
                "seed": 42,
                "deterministic": True,
                "compatibility_shim": (
                    "numpy.int=builtin_int_for_hrnet_constructor"
                ),
                "input_source": "canonical_jpeg_original_bytes",
                "decoder": "imageio.v2.imread",
                "channel_order": "RGB",
                "input_resize": "none",
                "input_crop": None,
                "input_reencode": False,
                "normalization": "uint8_rgb_divide_255",
                "feature_extractor": "HRNet-W18-small-v2",
                "internal_crop_size": [256, 256],
                "progressive_output_shapes": [
                    [256, 256],
                    [128, 128],
                    [64, 64],
                    [32, 32],
                ],
                "primary_map": "progressive_mask1_sigmoid_probability",
                "primary_map_selection": (
                    "fixed_by_official_test_py_index_0"
                ),
                "native_restore": (
                    "bilinear_probability_align_corners_true_to_input_size"
                ),
                "classification_output": (
                    "softmax_two_class_logits_positive_index_1"
                ),
                "classification_threshold": 0.5,
                "classification_threshold_comparison": (
                    "strict_greater_than"
                ),
                "mask_threshold": 0.5,
                "mask_threshold_comparison": "strict_greater_than",
                "test_time_augmentation": False,
                "ensemble": False,
            },
            "metrics": {
                "task": "T1_image_detection_and_T2_pixel_localization",
                "positive_class": "forged_or_manipulated",
                "classification_threshold": 0.5,
                "mask_threshold": 0.5,
                "threshold_comparison": "strict_greater_than",
                "prediction_inversion": False,
                "model_space_gt_resize": "nearest_neighbor_to_256x256",
            },
            "expected_pairs": 1,
            "expected_images": 2,
            "artifact_dir": "artifacts",
            "adapter_contract": [
                {
                    "path": "adapter.py",
                    "sha256": sha256_file(self.adapter_path),
                }
            ],
        }

    def _result(self, kind: str, *, probability: float) -> dict:
        image_path = self.real_image if kind == "real" else self.forged_image
        target = (
            np.zeros((2, 2), dtype=bool)
            if kind == "real"
            else self.target
        )
        primary = np.full((256, 256), 0.1, dtype=np.float32)
        if kind == "forged":
            primary[:, :128] = np.float32(0.9)
        maps = [
            primary,
            np.full((128, 128), 0.2, dtype=np.float32),
            np.full((64, 64), 0.3, dtype=np.float32),
            np.full((32, 32), 0.4, dtype=np.float32),
        ]
        progressive = []
        for stage, value in enumerate(maps, start=1):
            path = (
                self.artifact_dir
                / f"progressive_mask{stage}"
                / f"{kind}.npy"
            )
            _save_npy(path, value)
            progressive.append(
                {
                    "stage": stage,
                    "path": str(path.relative_to(self.root)),
                    "sha256": sha256_file(path),
                    "shape": list(value.shape),
                    "primary": stage == 1,
                }
            )
        native = _bilinear_align_corners_true(primary, width=2, height=2)
        native_path = (
            self.artifact_dir / "score_maps_native" / f"{kind}.npy"
        )
        mask_path = self.artifact_dir / "masks_native" / f"{kind}.png"
        _save_npy(native_path, native)
        _save_l(
            mask_path,
            np.where(native > 0.5, 255, 0).astype(np.uint8),
        )
        evidence, _ = _preprocess_evidence(image_path)
        target_model = np.repeat(
            np.repeat(target, 128, axis=0),
            128,
            axis=1,
        )
        return {
            "status": "ok",
            "valid_for_metrics": True,
            **_t1_fields(probability),
            "progressive_maps": progressive,
            "primary_model_score_map_path": progressive[0]["path"],
            "primary_model_score_map_sha256": progressive[0]["sha256"],
            "primary_model_score_map_shape": [256, 256],
            "score_map_path": str(native_path.relative_to(self.root)),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": [2, 2],
            "mask_path": str(mask_path.relative_to(self.root)),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": [2, 2],
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "localization": {
                "model_256": _binary_pixel_metrics_strict(
                    primary,
                    target_model,
                    include_ap=kind == "forged",
                ),
                "native": _binary_pixel_metrics_strict(
                    native,
                    target,
                    include_ap=kind == "forged",
                ),
            },
            "preprocess": evidence,
        }

    def pairs(self) -> list[Pair]:
        return [
            Pair(
                task_id="task",
                domain="lodging",
                real=self.ok_rows["real"],
                forged=self.ok_rows["forged"],
                input_row=self.input_rows[1],
            )
        ]


class PSCCNetProvenanceTest(unittest.TestCase):
    def test_fingerprint_history_identity_hashes_and_coverage_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            with mock.patch(
                "eval.opensource.analyze_psccnet_run._load_runner_pins",
                return_value=fixture.pins,
            ):
                provenance = validate_provenance(
                    repo_root=fixture.root,
                    run_id=fixture.run_id,
                    input_path=fixture.input_path,
                    input_rows=fixture.input_rows,
                    result_rows=fixture.result_rows,
                    manifest=fixture.manifest,
                    summary=fixture.summary,
                )
            history = summarize_result_history(fixture.result_rows)

        self.assertEqual(provenance["physical_result_rows_validated"], 3)
        self.assertEqual(provenance["latest_result_rows_validated"], 2)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_error_to_ok"], 1)

    def test_manifest_fingerprint_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            fixture.manifest["expected_images"] = 99
            with mock.patch(
                "eval.opensource.analyze_psccnet_run._load_runner_pins",
                return_value=fixture.pins,
            ), self.assertRaisesRegex(ValueError, "run manifest fingerprint"):
                validate_provenance(
                    repo_root=fixture.root,
                    run_id=fixture.run_id,
                    input_path=fixture.input_path,
                    input_rows=fixture.input_rows,
                    result_rows=fixture.result_rows,
                    manifest=fixture.manifest,
                    summary=fixture.summary,
                )

    def test_every_historical_result_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            fixture.result_rows[1]["checkpoint_sha256"] = "0" * 64
            with mock.patch(
                "eval.opensource.analyze_psccnet_run._load_runner_pins",
                return_value=fixture.pins,
            ), self.assertRaisesRegex(
                ValueError,
                "result row 2 field checkpoint_sha256",
            ):
                validate_provenance(
                    repo_root=fixture.root,
                    run_id=fixture.run_id,
                    input_path=fixture.input_path,
                    input_rows=fixture.input_rows,
                    result_rows=fixture.result_rows,
                    manifest=fixture.manifest,
                    summary=fixture.summary,
                )

    def test_manifest_selection_uses_declared_subset_order(self):
        rows = [
            {"sample_id": "unused"},
            {"sample_id": "forged"},
            {"sample_id": "real"},
        ]
        manifest = {
            "ordered_inputs": [
                {"sample_id": "real"},
                {"sample_id": "forged"},
            ]
        }
        selected = _select_manifest_inputs(rows, manifest)
        self.assertEqual(
            [row["sample_id"] for row in selected],
            ["real", "forged"],
        )


class PSCCNetArtifactAuditTest(unittest.TestCase):
    def test_audit_recomputes_t1_t2_gt_box_hit_and_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            audit = audit_and_best_threshold(
                fixture.pairs(),
                repo_root=fixture.root,
                bins=256,
            )

        self.assertEqual(audit["artifact_integrity"]["status"], "ok")
        self.assertEqual(audit["artifact_integrity"]["checked_files"], 15)
        self.assertEqual(
            audit["box_hit_at_mask_threshold_0_5"]["hits"],
            1,
        )
        oracle = audit["localization_best_threshold"]
        self.assertFalse(oracle["eligible_for_primary_metrics"])
        self.assertFalse(oracle["test_set_threshold_selection"])
        self.assertEqual(
            oracle["single_global_oracle"]["comparison"],
            ">",
        )

    def test_native_float32_map_must_derive_from_primary_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            forged = fixture.ok_rows["forged"]
            native_path = fixture.root / forged["score_map_path"]
            native = np.load(native_path, allow_pickle=False)
            native[0, 0] = np.float32(0.7)
            _save_npy(native_path, native)
            forged["score_map_sha256"] = sha256_file(native_path)
            with self.assertRaisesRegex(
                ValueError,
                "bilinear align_corners=True",
            ):
                audit_and_best_threshold(
                    fixture.pairs(),
                    repo_root=fixture.root,
                    bins=256,
                )

    def test_t1_is_recomputed_from_logits_not_trusted_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            fixture.ok_rows["forged"]["score"] = 0.123
            with self.assertRaisesRegex(ValueError, "T1 score mismatch"):
                audit_and_best_threshold(
                    fixture.pairs(),
                    repo_root=fixture.root,
                    bins=256,
                )

    def test_t2_metrics_are_recomputed_from_native_map_and_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PSCCNetFixture(Path(temporary))
            fixture.ok_rows["forged"]["localization"]["native"]["tp"] = 0
            with self.assertRaisesRegex(
                ValueError,
                "native localization tp",
            ):
                audit_and_best_threshold(
                    fixture.pairs(),
                    repo_root=fixture.root,
                    bins=256,
                )

    def test_threshold_mask_uses_strict_greater_than(self):
        score = np.asarray([[0.5, 0.50001]], dtype=np.float32)
        metrics = _binary_pixel_metrics_strict(
            score,
            np.zeros_like(score, dtype=bool),
            include_ap=False,
        )
        self.assertEqual(metrics["predicted_positive_pixels"], 1)
        self.assertEqual(metrics["threshold_operator"], ">")

    def test_oracle_is_strict_and_explicitly_separate_from_fixed_threshold(self):
        scores = np.asarray([[0.1, 0.9], [0.1, 0.9]], dtype=np.float32)
        target = np.asarray([[False, True], [False, True]])
        best, _, _ = _oracle_histograms(scores, target, bins=256)
        self.assertEqual(best["comparison"], ">")
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(MASK_THRESHOLD, 0.5)


def _statistics_pair(
    name: str,
    *,
    real_score: float,
    forged_score: float,
    detects: bool,
    edit_pixels: int = 1,
    domain: str = "lodging",
) -> Pair:
    real_map = np.asarray([[0.5, 0.6], [0.1, 0.1]], dtype=np.float32)
    forged_map = (
        np.asarray([[0.9, 0.1], [0.1, 0.1]], dtype=np.float32)
        if detects
        else np.full((2, 2), 0.1, dtype=np.float32)
    )
    real_target = np.zeros((2, 2), dtype=bool)
    forged_target = np.zeros((2, 2), dtype=bool)
    forged_target.reshape(-1)[:edit_pixels] = True
    real_metrics = _binary_pixel_metrics_strict(
        real_map,
        real_target,
        include_ap=False,
    )
    forged_metrics = _binary_pixel_metrics_strict(
        forged_map,
        forged_target,
        include_ap=True,
    )
    real = {
        "id": f"{name}_real",
        **_t1_fields(real_score),
        "localization": {"native": real_metrics},
    }
    forged = {
        "id": f"{name}_forged",
        **_t1_fields(forged_score),
        "localization": {"native": forged_metrics},
    }
    return Pair(
        task_id=name,
        domain=domain,
        real=real,
        forged=forged,
        input_row={},
    )


class PSCCNetStatisticsTest(unittest.TestCase):
    def test_paired_bootstrap_is_fixed_seed_deterministic_and_has_sign_test(self):
        pairs = [
            _statistics_pair(
                "win",
                real_score=0.2,
                forged_score=0.8,
                detects=True,
            ),
            _statistics_pair(
                "loss",
                real_score=0.8,
                forged_score=0.6,
                detects=False,
                edit_pixels=2,
            ),
            _statistics_pair(
                "tie",
                real_score=0.4,
                forged_score=0.4,
                detects=True,
                edit_pixels=3,
                domain="retail",
            ),
        ]
        first = summarize_psccnet_pair_slice(
            pairs,
            iterations=1000,
            seed=20260724,
        )
        second = summarize_psccnet_pair_slice(
            pairs,
            iterations=1000,
            seed=20260724,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["paired_sign_test"]["wins"], 1)
        self.assertEqual(first["paired_sign_test"]["losses"], 1)
        self.assertEqual(first["paired_sign_test"]["ties"], 1)
        self.assertEqual(
            first["threshold_source"],
            "pre_registered_fixed_0_5_not_test_selected",
        )
        self.assertFalse(first["t1_prediction_inversion"])

    def test_auc_is_never_flipped_when_direction_is_bad(self):
        point = _detection_point(
            np.asarray([0.9, 0.8], dtype=np.float64),
            np.asarray([0.2, 0.1], dtype=np.float64),
        )
        self.assertEqual(point["auroc"], 0.0)

    def test_equal_logits_produce_half_and_authentic_decision(self):
        row = {
            "id": "equal",
            "classification_logits": [0.0, 0.0],
            "classification_probabilities": [0.5, 0.5],
            "score": CLASSIFICATION_THRESHOLD,
            "decision": "authentic",
        }
        self.assertEqual(
            _recomputed_t1_score(row, validate_recorded=True),
            0.5,
        )

    def test_edit_area_quintiles_are_stable_and_cover_every_pair(self):
        pairs = [
            _statistics_pair(
                str(index),
                real_score=0.2,
                forged_score=0.8,
                detects=True,
                edit_pixels=(index % 4) + 1,
            )
            for index in range(7)
        ]
        quintiles = _quintiles(pairs)
        self.assertEqual(len(quintiles), 5)
        self.assertTrue(quintiles[0][0].startswith("q1_smallest"))
        self.assertTrue(quintiles[-1][0].endswith("largest"))
        self.assertEqual(sum(len(chunk) for _, chunk in quintiles), 7)


if __name__ == "__main__":
    unittest.main()
