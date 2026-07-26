from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from eval.opensource import run_fsd as runner


OFFICIAL_WEIGHTS = Path(
    "/root/.cache/claimforge/checkpoints/fsd-v1.2.0"
)


def _selection_rows(pair_count: int = 2) -> list[dict]:
    rows: list[dict] = []
    for pair_rank in range(pair_count):
        for offset, kind in enumerate(("real", "forged")):
            rank = pair_rank * 2 + offset
            rows.append(
                {
                    "rank": rank,
                    "pair_rank": pair_rank,
                    "sample_id": f"{kind}-{pair_rank}",
                    "task_id": f"task-{pair_rank}",
                    "domain": "lodging",
                    "kind": kind,
                    "label": int(kind == "forged"),
                    "width": 30,
                    "height": 20,
                    "canonical_path": f"images/{kind}-{pair_rank}.png",
                    "canonical_sha256": f"{rank + 1:064x}",
                    "gt_mask_kind": (
                        "exact_diff" if kind == "forged" else "all_zero"
                    ),
                    "gt_mask_path": (
                        f"masks/{pair_rank}.png"
                        if kind == "forged"
                        else None
                    ),
                    "gt_mask_sha256": (
                        f"{rank + 101:064x}" if kind == "forged" else None
                    ),
                    "gt_positive_pixels": int(kind == "forged"),
                    "edit_region_xyxy": [10, 8, 20, 12],
                }
            )
    return rows


class _FakeGMM:
    def score_samples(self, value: torch.Tensor) -> torch.Tensor:
        return value.sum(dim=1) / 1000.0


class RunFsdTest(unittest.TestCase):
    def test_frozen_release_contract(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "50f2eae06efdac2e5a33f407ca9a27a2295133ac",
        )
        self.assertEqual(runner.RELEASE_TAG, "v1.2.0")
        self.assertEqual(
            runner.RELEASE_TAG_COMMIT,
            "5b317a00251988b5ec5a47317f4d82e5bdfd009d",
        )
        self.assertEqual(runner.LICENSE_SPDX, "CC-BY-NC-SA-4.0")
        self.assertEqual(runner.FSD_DIMENSION, 960)
        self.assertEqual(runner.FRE_KERNEL_SIZE, 15)
        self.assertEqual(runner.FRE_BORDER, 7)
        self.assertEqual(runner.FSD_NEIGHBORHOOD, 11)
        self.assertEqual(runner.FSD_SCALES, 3)
        self.assertEqual(runner.PROJECTION_COUNT, 20)
        self.assertEqual(runner.PROJECTION_PARAMETERS, 5_267_200)
        self.assertEqual(runner.RELEASED_Z_THRESHOLD, -2.0)
        self.assertEqual(runner.AI_SCORE_THRESHOLD, 2.0)
        self.assertEqual(runner.THRESHOLD_OPERATOR, ">")
        self.assertEqual(runner.RELEASED_THRESHOLD_OPERATOR, "<")
        self.assertEqual(
            {
                name: (value["bytes"], value["sha256"])
                for name, value in runner.WEIGHT_FILES.items()
            },
            {
                "config.json": (
                    634,
                    "7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4",
                ),
                "fre.pt": (
                    9_861,
                    "d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb",
                ),
                "gmm.pt": (
                    14_786_229,
                    "0f9fa030a3d5816266d0329fd0fb614b65e322d4bda6d083613c713bfe9bc829",
                ),
                "fsd_transforms.pt": (
                    42_177_409,
                    "1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699",
                ),
            },
        )
        self.assertTrue(
            all(
                len(value) == 64
                for value in runner.SOURCE_FILES.values()
            )
        )
        self.assertEqual(
            runner.SOURCE_TAG_DRIFT["commits_ahead_of_tag"],
            1,
        )
        self.assertIn(
            "does not claim paper-protocol parity",
            runner.PAPER_RELEASE_DRIFT["evaluation_claim"],
        )

    def test_repository_contract_pins_tracked_tree_and_tag_drift(self):
        changed = "\n".join(runner.SOURCE_TAG_DRIFT["changed_files"])
        with (
            mock.patch.object(
                runner,
                "_git_value",
                side_effect=[
                    runner.MODEL_SOURCE_COMMIT,
                    "",
                    runner.RELEASE_TAG_COMMIT,
                    "1",
                    changed,
                ],
            ) as git_value,
            mock.patch.object(runner, "_verify_runtime_file"),
        ):
            audit = runner._verify_repository_contract(
                runner.DEFAULT_SOURCE_ROOT
            )
        self.assertFalse(audit["tracked_dirty"])
        self.assertEqual(
            git_value.call_args_list[1],
            mock.call(
                runner.DEFAULT_SOURCE_ROOT,
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
        )

        with (
            mock.patch.object(
                runner,
                "_git_value",
                side_effect=[
                    runner.MODEL_SOURCE_COMMIT,
                    " M fsd/detector.py",
                ],
            ),
            mock.patch.object(runner, "_verify_runtime_file"),
        ):
            with self.assertRaisesRegex(ValueError, "tracked source tree is dirty"):
                runner._verify_repository_contract(runner.DEFAULT_SOURCE_ROOT)

    def test_official_source_and_four_assets_pass_static_contract(self):
        source, weights = runner._verify_static_contract(
            source_root=runner.DEFAULT_SOURCE_ROOT,
            weights_dir=OFFICIAL_WEIGHTS,
        )
        self.assertEqual(source["commit"], runner.MODEL_SOURCE_COMMIT)
        self.assertEqual(set(weights["files"]), set(runner.WEIGHT_FILES))
        self.assertEqual(weights["config"], runner.EXPECTED_CONFIG)
        self.assertTrue(runner._valid_sha256(weights["bundle_sha256"]))
        for filename in ("fre.pt", "gmm.pt", "fsd_transforms.pt"):
            safety = weights["files"][filename]["serialization_safety"]
            self.assertEqual(safety["unsafe_globals"], [])
            self.assertTrue(safety["weights_only"])

    def test_weight_preflight_rejects_any_unsafe_global_before_load(self):
        with mock.patch(
            "torch.serialization.get_unsafe_globals_in_checkpoint",
            return_value=["builtins.eval"],
        ):
            with self.assertRaisesRegex(ValueError, "contains unsafe globals"):
                runner._verify_weights_contract(OFFICIAL_WEIGHTS)

    def test_payload_schema_fails_closed_on_arbitrary_objects(self):
        class Arbitrary:
            pass

        with self.assertRaisesRegex(ValueError, "unexpected object"):
            runner._payload_schema({"unsafe": Arbitrary()})

    def test_select_inputs_preserves_pairs_and_supports_preflight(self):
        rows = _selection_rows(3)
        self.assertEqual(
            [
                (row["pair_rank"], row["kind"])
                for row in runner.select_inputs(rows, pair_limit=2)
            ],
            [
                (0, "real"),
                (0, "forged"),
                (1, "real"),
                (1, "forged"),
            ],
        )
        self.assertEqual(
            runner.select_inputs(
                rows,
                pair_limit=None,
                sample_id="forged-2",
            ),
            [rows[-1]],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.select_inputs(rows, pair_limit=1, sample_id="real-0")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            runner.select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.select_inputs(rows, pair_limit=None, sample_id="missing")
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            runner.select_inputs(rows[:-1], pair_limit=None)

    def test_geometry_is_exact_released_border_resize_crop_chain(self):
        geometry = runner.compute_preprocess_geometry(1800, 1350)
        self.assertEqual(geometry["native_size"], [1800, 1350])
        self.assertEqual(geometry["fre"]["border_each_side"], 7)
        self.assertEqual(geometry["fre"]["post_trim_size"], [1786, 1336])
        scale = 1024 / 1336
        self.assertEqual(geometry["resize"]["scale_factor_nominal"], scale)
        self.assertEqual(
            geometry["resize"]["destination_size"],
            [round(1786 * scale), 1024],
        )
        resized_width = round(1786 * scale)
        expected_start_x = (resized_width - 1024) // 2
        self.assertEqual(
            geometry["center_crop"],
            {
                "enabled": True,
                "source_size": [resized_width, 1024],
                "start_xy": [expected_start_x, 0],
                "size": [1024, 1024],
                "end_xy": [expected_start_x + 1024, 1024],
            },
        )
        self.assertEqual(
            geometry["scales"]["sizes"],
            [[1024, 1024], [512, 512], [256, 256]],
        )
        self.assertEqual(geometry["descriptor"]["dtype"], "float64")

    def test_determinism_contract_disables_tf32_and_uses_highest_precision(self):
        runner.configure_determinism(20260724)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        runtime = runner._runtime_contract("cpu")
        self.assertEqual(
            runtime["packages"]["torch"]["version"],
            str(torch.__version__),
        )
        self.assertEqual(
            runtime["packages"]["torch"]["full_version"],
            str(torch.__version__),
        )
        self.assertEqual(
            runtime["accelerator"]["torch_version"],
            str(torch.__version__),
        )
        self.assertEqual(
            runtime["numerical_flags"]["float32_matmul_precision"],
            "highest",
        )

    def test_pair_visibility_uses_exact_align_corners_false_pixel_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "masks").mkdir()
            mask = np.zeros((20, 30), dtype=np.uint8)
            # With residual width=16, height=6, the center point is visible
            # after the very wide resize and center crop; edge points are not.
            mask[10, 7] = 255
            mask[10, 15] = 255
            mask[10, 22] = 255
            mask_path = root / "masks" / "0.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            digest = runner.sha256_file(mask_path)
            rows = _selection_rows(1)
            rows[1]["gt_mask_path"] = "masks/0.png"
            rows[1]["gt_mask_sha256"] = digest
            rows[1]["gt_positive_pixels"] = 3

            visibility = runner.build_pair_visibility(rows, root)["task-0"]

        self.assertEqual(visibility["edit_visibility"], "partial")
        self.assertEqual(visibility["edit_visible_gt_fraction"], 1 / 3)
        evidence = visibility["edit_visibility_evidence"]["gt"]
        self.assertEqual(evidence["positive_pixels"], 3)
        self.assertEqual(evidence["visible_positive_pixel_centers"], 1)
        self.assertIn("align_corners_false", evidence["basis"])

    def test_mouse_visibility_distribution_is_frozen(self):
        repo_root = Path(__file__).resolve().parents[1]
        release, _inputs_path, rows = runner.load_release(
            repo_root,
            repo_root / runner.DEFAULT_DATASET_MANIFEST,
        )
        self.assertEqual(release["pairs"], 275)
        visibility = runner.build_pair_visibility(rows, repo_root)
        self.assertEqual(len(visibility), 275)
        self.assertEqual(
            Counter(
                value["edit_visibility"] for value in visibility.values()
            ),
            Counter({"none": 49, "partial": 34, "full": 192}),
        )

    def test_load_detector_always_passes_explicit_weights_dir(self):
        calls: list[dict] = []

        class FakeDetectorClass:
            @classmethod
            def load(cls, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    config=runner.EXPECTED_CONFIG,
                    threshold=-2.0,
                    train_mean=runner.TRAIN_MEAN,
                    train_std=runner.TRAIN_STD,
                    projections=[object()] * 20,
                    fre=SimpleNamespace(
                        conv=SimpleNamespace(w=torch.empty(8, 1, 15, 15))
                    ),
                    __class__=SimpleNamespace,
                )

        package = SimpleNamespace(FSDDetector=FakeDetectorClass)
        with (
            mock.patch.object(
                runner,
                "_verify_static_contract",
                return_value=({"commit": "source"}, {"bundle_sha256": "b" * 64}),
            ),
            mock.patch.object(
                runner,
                "_load_official_package",
                return_value=package,
            ),
        ):
            detector, device, audit = runner.load_detector(
                source_root=runner.DEFAULT_SOURCE_ROOT,
                weights_dir=OFFICIAL_WEIGHTS,
                device_name="cpu",
            )
        self.assertIsNotNone(detector)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(
            calls,
            [
                {
                    "weights_dir": OFFICIAL_WEIGHTS,
                    "device": "cpu",
                    "threshold": -2.0,
                    "attribution": False,
                }
            ],
        )
        self.assertTrue(audit["weights_dir_was_explicit"])
        self.assertFalse(audit["automatic_download_used"])

    def test_infer_one_runs_official_compute_once_and_exact_manual_replay(self):
        package = "_claimforge_test_fsd"
        detector_module_name = f"{package}.detector"
        projection_module_name = f"{package}.projection"
        detector_module = types.ModuleType(detector_module_name)
        projection_module = types.ModuleType(projection_module_name)
        calls = {"compute": 0}

        def compute_fsd(_image):
            calls["compute"] += 1
            return torch.arange(960, dtype=torch.float64)

        def apply_projections(value, projections):
            self.assertEqual(projections, [])
            return value

        detector_module.compute_fsd = compute_fsd
        projection_module.apply_projections = apply_projections

        class FakeDetector:
            def __init__(self):
                self.gmm = _FakeGMM()
                self.projections = []
                self.train_mean = runner.TRAIN_MEAN
                self.train_std = runner.TRAIN_STD

            def score(self, image):
                descriptor = detector_module.compute_fsd(image)
                projected = projection_module.apply_projections(
                    descriptor.unsqueeze(0),
                    self.projections,
                )
                raw = float(self.gmm.score_samples(projected).item())
                z_score = (raw - self.train_mean) / self.train_std
                return SimpleNamespace(
                    raw_score=raw,
                    z_score=z_score,
                    is_fake=z_score < -2.0,
                    threshold=-2.0,
                )

        FakeDetector.__module__ = detector_module_name
        sys.modules[detector_module_name] = detector_module
        sys.modules[projection_module_name] = projection_module
        try:
            processed, descriptor, peak, latency = runner.infer_one(
                FakeDetector(),
                torch.device("cpu"),
                Path("unused.png"),
            )
        finally:
            sys.modules.pop(detector_module_name, None)
            sys.modules.pop(projection_module_name, None)
        self.assertEqual(calls["compute"], 1)
        self.assertEqual(descriptor.dtype, np.float64)
        self.assertEqual(descriptor.shape, (960,))
        self.assertIsNone(peak)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(
            processed["ai_score"],
            -processed["released_z_score"],
        )
        self.assertTrue(
            processed["manual_replay"]["official_raw_exact_match"]
        )
        self.assertEqual(
            processed["classification_decision"],
            processed["released_is_fake"],
        )

    def test_infer_one_rejects_high_level_score_drift(self):
        package = "_claimforge_test_fsd_drift"
        detector_module_name = f"{package}.detector"
        projection_module_name = f"{package}.projection"
        detector_module = types.ModuleType(detector_module_name)
        projection_module = types.ModuleType(projection_module_name)
        detector_module.compute_fsd = lambda _image: torch.ones(
            960,
            dtype=torch.float64,
        )
        projection_module.apply_projections = lambda value, _projections: value

        class FakeDetector:
            projections = []
            train_mean = runner.TRAIN_MEAN
            train_std = runner.TRAIN_STD
            gmm = _FakeGMM()

            def score(self, image):
                descriptor = detector_module.compute_fsd(image)
                raw = float(
                    self.gmm.score_samples(descriptor.unsqueeze(0)).item()
                )
                z_score = (raw - self.train_mean) / self.train_std
                return SimpleNamespace(
                    raw_score=raw + 1e-12,
                    z_score=z_score,
                    is_fake=False,
                    threshold=-2.0,
                )

        FakeDetector.__module__ = detector_module_name
        sys.modules[detector_module_name] = detector_module
        sys.modules[projection_module_name] = projection_module
        try:
            with self.assertRaisesRegex(ValueError, "manual descriptor"):
                runner.infer_one(
                    FakeDetector(),
                    torch.device("cpu"),
                    Path("unused.png"),
                )
        finally:
            sys.modules.pop(detector_module_name, None)
            sys.modules.pop(projection_module_name, None)

    def test_resume_validates_every_physical_row_and_descriptor(self):
        expected = {
            "sample_id": "sample",
            "rank": 0,
            "pair_rank": 0,
            "task_id": "task",
            "domain": "lodging",
            "kind": "real",
            "label": 0,
            "width": 30,
            "height": 20,
            "canonical_path": "images/sample.png",
            "canonical_sha256": "1" * 64,
            "gt_mask_kind": "all_zero",
            "gt_mask_sha256": None,
            "edit_region_xyxy": [10, 8, 20, 12],
        }
        fingerprint = "2" * 64
        bundle = "3" * 64
        input_manifest = "4" * 64
        visibility = {
            "task": {
                "edit_visibility": "partial",
                "edit_visible_gt_fraction": 0.5,
                "edit_visibility_evidence": {
                    "gt": {"visible_positive_pixel_centers": 1},
                    "edit_box": {"visible_fraction": 0.5},
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor_path = root / "descriptor.npy"
            runner._atomic_save_npy(
                descriptor_path,
                np.arange(960, dtype=np.float64),
            )
            raw = 17.5
            z_score = (raw - runner.TRAIN_MEAN) / runner.TRAIN_STD
            identity = {
                "schema_version": "opensource_result_v1",
                "run_id": "run",
                "run_manifest_fingerprint": fingerprint,
                "input_manifest_sha256": input_manifest,
                "id": "sample",
                "rank": 0,
                "task_id": "task",
                "pair_rank": 0,
                "domain": "lodging",
                "kind": "real",
                "label": 0,
                "image_path": "images/sample.png",
                "image_sha256": "1" * 64,
                "image_size": [30, 20],
                "gt_mask_kind": "all_zero",
                "gt_mask_sha256": None,
                "edit_region_xyxy": [10, 8, 20, 12],
                **visibility["task"],
                "model": runner.MODEL_NAME,
                "model_slug": runner.MODEL_SLUG,
                "weights_bundle_sha256": bundle,
                "model_source_commit": runner.MODEL_SOURCE_COMMIT,
                "release_tag": runner.RELEASE_TAG,
                "valid_for_t1": True,
                "valid_for_t2": False,
                "t1_policy": "released_FSD_v1.2.0_whole_image_z_score",
                "t2_policy": "unsupported_whole_image_detector",
                "completed_at": "2026-07-24T00:00:00+00:00",
            }
            error = {
                **identity,
                "status": "error",
                "valid_for_metrics": False,
                "error_type": "RuntimeError",
                "error_message": "retry me",
                "traceback": "trace",
            }
            classification = {
                "score": -z_score,
                "raw_likelihood": raw,
                "released_z_score": z_score,
                "decision": -z_score > 2.0,
                "threshold": 2.0,
                "threshold_operator": ">",
                "semantics": "higher_is_more_AI_negative_released_z",
            }
            ok = {
                **identity,
                "status": "ok",
                "gt_mask_sha256": None,
                "score": -z_score,
                "score_semantics": "negative_released_FSD_z_score",
                "raw_descriptor_path": str(descriptor_path),
                "raw_descriptor_sha256": runner.sha256_file(descriptor_path),
                "raw_descriptor_shape": [960],
                "raw_descriptor_dtype": "float64",
                "raw_descriptor_semantics": (
                    "official_compute_fsd_before_released_transforms"
                ),
                "artifact_paths": {
                    "raw_descriptor_npy": str(descriptor_path)
                },
                "raw_likelihood": raw,
                "released_z_score": z_score,
                "ai_score": -z_score,
                "released_is_fake": z_score < -2.0,
                "released_threshold": -2.0,
                "released_threshold_operator": "<",
                "classification_decision": -z_score > 2.0,
                "classification_threshold": 2.0,
                "classification_threshold_operator": ">",
                "classification": classification,
                "t1": {
                    **classification,
                    "policy": (
                        "released_FSD_whole_image_score_sign_inverted"
                    ),
                },
                "manual_replay": {
                    "raw_likelihood": raw,
                    "released_z_score": z_score,
                    "ai_score": -z_score,
                    "released_is_fake": z_score < -2.0,
                    "classification_decision": -z_score > 2.0,
                    "official_raw_exact_match": True,
                    "official_z_exact_match": True,
                    "compute_fsd_calls": 1,
                },
                "valid_for_t1": True,
                "valid_for_t2": False,
                "valid_for_metrics": True,
                "preprocess": runner.compute_preprocess_geometry(30, 20),
                "latency_ms": 1.25,
                "peak_cuda_memory_bytes": None,
            }
            ok["t1"].pop("semantics")
            runner._validate_resume_rows(
                [error, ok],
                [expected],
                fingerprint,
                bundle,
                run_id="run",
                input_manifest_sha256=input_manifest,
                pair_visibility=visibility,
                repo_root=root,
            )
            wrong_run = dict(ok)
            wrong_run["run_id"] = "copied-run"
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                runner._validate_resume_rows(
                    [wrong_run],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            wrong_alias = dict(ok)
            wrong_alias["score"] = math.nextafter(
                float(ok["score"]),
                math.inf,
            )
            with self.assertRaisesRegex(ValueError, "decision contract"):
                runner._validate_resume_rows(
                    [wrong_alias],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            wrong_manual = dict(ok)
            wrong_manual["manual_replay"] = {
                **ok["manual_replay"],
                "official_raw_exact_match": False,
            }
            with self.assertRaisesRegex(ValueError, "decision contract"):
                runner._validate_resume_rows(
                    [wrong_manual],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            wrong_preprocess = dict(ok)
            wrong_preprocess["preprocess"] = {
                **ok["preprocess"],
                "decoder": "different",
            }
            with self.assertRaisesRegex(ValueError, "preprocess changed"):
                runner._validate_resume_rows(
                    [wrong_preprocess],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            wrong_error = dict(error)
            wrong_error["domain"] = "restaurant"
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                runner._validate_resume_rows(
                    [wrong_error],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            tampered = dict(ok)
            tampered["ai_score"] = math.nextafter(-z_score, math.inf)
            with self.assertRaisesRegex(ValueError, "calibration mismatch"):
                runner._validate_resume_rows(
                    [error, tampered],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )
            descriptor_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner._validate_resume_rows(
                    [ok],
                    [expected],
                    fingerprint,
                    bundle,
                    run_id="run",
                    input_manifest_sha256=input_manifest,
                    pair_visibility=visibility,
                    repo_root=root,
                )

    def test_completed_resume_preserves_safe_model_load_audit(self):
        audit = {
            "class_name": "FSDDetector",
            "class_module": "_claimforge_fsd_50f2eae.detector",
            "load_api": "FSDDetector.load",
            "weights_dir_was_explicit": True,
            "automatic_download_used": False,
            "attribution_loaded": False,
            "released_threshold": -2.0,
            "projection_count": 20,
            "source": {"commit": runner.MODEL_SOURCE_COMMIT},
            "weights_dir_argument": str(OFFICIAL_WEIGHTS),
            "weights": {
                "bundle_sha256": "b" * 64,
                "release_tag": runner.RELEASE_TAG,
                "weights_dir": str(OFFICIAL_WEIGHTS),
                "explicit_weights_dir_required": True,
                "automatic_download_used": False,
                "files": {
                    filename: {
                        "serialization_safety": {
                            "preflight": (
                                "torch.serialization."
                                "get_unsafe_globals_in_checkpoint"
                            ),
                            "unsafe_globals": [],
                            "required_unsafe_globals": [],
                            "weights_only": True,
                        }
                    }
                    for filename in (
                        "fre.pt",
                        "gmm.pt",
                        "fsd_transforms.pt",
                    )
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "run_manifest_fingerprint": "a" * 64,
                        "weights_bundle_sha256": "b" * 64,
                        "model_load_audit": audit,
                    }
                ),
                encoding="utf-8",
            )
            loaded = runner._load_completed_model_audit(
                path,
                run_id="run",
                manifest_fingerprint="a" * 64,
                weights_bundle_sha256="b" * 64,
            )
            self.assertEqual(loaded, audit)
            audit["weights"]["files"]["fre.pt"]["serialization_safety"][
                "unsafe_globals"
            ] = ["evil"]
            path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "run_manifest_fingerprint": "a" * 64,
                        "weights_bundle_sha256": "b" * 64,
                        "model_load_audit": audit,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safety audit"):
                runner._load_completed_model_audit(
                    path,
                    run_id="run",
                    manifest_fingerprint="a" * 64,
                    weights_bundle_sha256="b" * 64,
                )

    def test_cli_requires_explicit_weights_dir(self):
        with mock.patch.object(
            sys,
            "argv",
            ["run_fsd.py", "--run-id", "test"],
        ):
            with self.assertRaises(SystemExit):
                runner.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_fsd.py",
                "--run-id",
                "test",
                "--weights-dir",
                str(OFFICIAL_WEIGHTS),
            ],
        ):
            args = runner.parse_args()
        self.assertEqual(args.weights_dir, OFFICIAL_WEIGHTS)

    def test_run_rejects_unsafe_run_id_and_threshold_before_io(self):
        base = argparse.Namespace(
            run_id="../escape",
            classification_threshold=2.0,
            bootstrap_samples=10,
            weights_dir=OFFICIAL_WEIGHTS,
        )
        with self.assertRaisesRegex(ValueError, "unsafe characters"):
            runner.run(base)
        base.run_id = "safe"
        base.classification_threshold = math.nextafter(2.0, math.inf)
        with self.assertRaisesRegex(ValueError, "threshold must be 2.0"):
            runner.run(base)
        base.classification_threshold = 2.0
        base.weights_dir = None
        with self.assertRaisesRegex(ValueError, "mandatory"):
            runner.run(base)


if __name__ == "__main__":
    unittest.main()
