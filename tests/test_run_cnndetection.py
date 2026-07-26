from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional

from eval.opensource import run_cnndetection as runner


SOURCE_ROOT = runner.DEFAULT_SOURCE_ROOT
CHECKPOINT_PATH = runner.DEFAULT_CHECKPOINT
HAS_OFFICIAL_ASSETS = SOURCE_ROOT.is_dir() and CHECKPOINT_PATH.is_file()


class _TinyCNNDetection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(runner.FEATURE_DIMENSION, 1)
        self.forward_calls = 0
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        mean = image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        feature = mean.expand(-1, runner.FEATURE_DIMENSION).contiguous()
        return self.fc(feature)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunCNNDetectionTest(unittest.TestCase):
    def test_frozen_contract(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "ea0b5622365e3a9cd31d1b54b6b5971131a839ab",
        )
        self.assertEqual(
            runner.CHECKPOINT["sha256"],
            "a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea",
        )
        self.assertEqual(runner.CHECKPOINT["bytes"], 282_442_597)
        self.assertEqual(runner.CHECKPOINT["state_entries"], 320)
        self.assertEqual(runner.CHECKPOINT["state_elements"], 23_563_254)
        self.assertEqual(
            runner.CHECKPOINT["trainable_parameters"],
            23_510_081,
        )
        self.assertEqual(runner.CHECKPOINT["total_steps"], 270_048)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD, 0.5)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD_OPERATOR, ">")
        self.assertEqual(
            set(runner.PREPROCESS_PROFILES),
            {runner.PRIMARY_PROFILE, runner.PAPER_CROP_PROFILE},
        )
        self.assertEqual(
            runner.PREPROCESS_PROFILES[runner.PRIMARY_PROFILE]["role"],
            "primary",
        )
        self.assertEqual(
            runner.PREPROCESS_PROFILES[runner.PAPER_CROP_PROFILE]["role"],
            "preregistered_sensitivity_not_primary",
        )
        self.assertFalse(
            runner.LICENSE_RECORD["repository"]["commercial_use_permitted"]
        )

    def test_native_geometry_is_full_image(self):
        geometry = runner.compute_preprocess_geometry(
            400,
            300,
            runner.PRIMARY_PROFILE,
        )
        self.assertFalse(geometry["resize"]["enabled"])
        self.assertFalse(geometry["center_crop"]["enabled"])
        self.assertEqual(
            geometry["effective_native_xyxy"],
            [0, 0, 400, 300],
        )
        self.assertEqual(geometry["output_size"], [400, 300])

    def test_paper_crop_geometry_matches_torchvision(self):
        geometry = runner.compute_preprocess_geometry(
            400,
            300,
            runner.PAPER_CROP_PROFILE,
        )
        self.assertEqual(
            geometry["center_crop"]["start_xy_in_padded_image"],
            [88, 38],
        )
        self.assertEqual(
            geometry["effective_native_xyxy"],
            [88, 38, 312, 262],
        )
        small = runner.compute_preprocess_geometry(
            31,
            20,
            runner.PAPER_CROP_PROFILE,
        )
        self.assertEqual(
            small["center_crop"]["padding_ltrb"],
            [96, 102, 97, 102],
        )
        self.assertEqual(
            small["effective_native_xyxy"],
            [0, 0, 31, 20],
        )

    def test_preprocessing_matches_torchvision_exactly(self):
        rng = np.random.default_rng(11)
        rgb = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            for profile in (
                runner.PRIMARY_PROFILE,
                runner.PAPER_CROP_PROFILE,
            ):
                actual, audit = runner.preprocess_image(path, profile)
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    if profile == runner.PAPER_CROP_PROFILE:
                        image = vision_functional.center_crop(
                            image,
                            [224, 224],
                        )
                    expected = vision_functional.normalize(
                        vision_functional.to_tensor(image),
                        runner.IMAGE_MEAN,
                        runner.IMAGE_STD,
                    )
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(
                    audit["tensor_sha256"],
                    runner._tensor_sha256(expected),
                )
                self.assertFalse(audit["geometry"]["resize"]["enabled"])

    def test_no_blur_or_jpeg_exists_in_inference_profiles(self):
        for profile in runner.PREPROCESS_PROFILES.values():
            steps = " ".join(profile["steps"]).lower()
            self.assertNotIn("blur", steps)
            self.assertNotIn("jpeg", steps)
            self.assertIsNone(profile["resize"])
            self.assertEqual(profile["batch_size"], 1)

    def test_infer_one_captures_feature_and_uses_strict_threshold(self):
        model = _TinyCNNDetection()
        tensor = torch.zeros((3, 16, 16), dtype=torch.float32)
        scoring, feature = runner.infer_one(
            model=model,
            tensor=tensor,
            device=torch.device("cpu"),
        )
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(scoring["raw_logit"], 0.0)
        self.assertEqual(scoring["ai_score"], 0.5)
        self.assertFalse(scoring["classification_decision"])
        self.assertFalse(scoring["calibrated_probability"])
        self.assertEqual(feature.shape, (runner.FEATURE_DIMENSION,))
        self.assertEqual(feature.dtype, np.float32)
        self.assertTrue(
            scoring["manual_replay"]["official_logit_exact_match"]
        )

    def test_pair_visibility_is_profile_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((300, 400), dtype=np.uint8)
            mask[0:10, 0:10] = 255
            mask_path = root / "mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            common = {
                "task_id": "task-0",
                "width": 400,
                "height": 300,
            }
            rows = [
                {
                    **common,
                    "sample_id": "real-0",
                    "kind": "real",
                    "label": 0,
                    "gt_mask_kind": "all_zero",
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": 0,
                },
                {
                    **common,
                    "sample_id": "fake-0",
                    "kind": "forged",
                    "label": 1,
                    "gt_mask_kind": "exact_diff",
                    "gt_mask_path": str(mask_path),
                    "gt_mask_sha256": _sha256(mask_path),
                    "gt_positive_pixels": 100,
                },
            ]
            native = runner.build_pair_visibility(
                rows,
                root,
                runner.PRIMARY_PROFILE,
            )["task-0"]
            crop = runner.build_pair_visibility(
                rows,
                root,
                runner.PAPER_CROP_PROFILE,
            )["task-0"]
        self.assertEqual(native["edit_visibility"], "full")
        self.assertEqual(native["edit_visible_gt_fraction"], 1.0)
        self.assertEqual(crop["edit_visibility"], "none")
        self.assertEqual(crop["edit_visible_gt_fraction"], 0.0)

    def test_input_selection_is_pair_safe(self):
        rows = [
            {
                "sample_id": f"{kind}-{pair}",
                "pair_rank": pair,
                "kind": kind,
            }
            for pair in range(3)
            for kind in ("real", "forged")
        ]
        selected = runner.select_inputs(rows, pair_limit=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {row["pair_rank"] for row in selected},
            {0, 1},
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.select_inputs(rows, pair_limit=1, sample_id="real-0")

    def test_source_contract_rejects_tracked_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(
                    runner,
                    "_git_value",
                    side_effect=[
                        runner.MODEL_SOURCE_COMMIT,
                        " M networks/resnet.py",
                    ],
                ),
                mock.patch.object(runner, "_verify_runtime_file"),
            ):
                with self.assertRaisesRegex(ValueError, "tracked source tree"):
                    runner._verify_source_contract(Path(temporary))

    def test_preflight_rejects_cuda(self):
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            runner.run_preflight(
                source_root=SOURCE_ROOT,
                checkpoint_path=CHECKPOINT_PATH,
                device_text="cuda:0",
            )

    def test_bare_preflight_defaults_to_cpu(self):
        report = {
            "schema_version": "cnndetection_preflight_v1",
            "status": "passed",
        }
        with (
            mock.patch.object(
                runner,
                "run_preflight",
                return_value=report,
            ) as preflight,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runner.main(["--preflight"]), 0)
        self.assertEqual(preflight.call_args.kwargs["device_text"], "cpu")

    def test_runner_has_no_baseexception_handler(self):
        tree = ast.parse(inspect.getsource(runner))
        caught = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "BaseException"
        ]
        self.assertEqual(caught, [])

    def test_resume_validator_rejects_artifact_and_payload_tampering(self):
        model = _TinyCNNDetection()
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(
                16,
                16,
                3,
            )
            Image.fromarray(pixels, mode="RGB").save(image_path)
            tensor, preprocess = runner.preprocess_image(
                image_path,
                runner.PRIMARY_PROFILE,
            )
            scoring, feature = runner.infer_one(
                model=model,
                tensor=tensor,
                device=device,
            )
            feature_path = root / "feature.npy"
            np.save(feature_path, feature, allow_pickle=False)
            expected = {
                "input_path": str(image_path),
                "preprocess_profile": runner.PRIMARY_PROFILE,
                "config_fingerprint": "fingerprint",
            }
            row = {
                **expected,
                "status": "ok",
                "preprocess": preprocess,
                "preprocess_latency_ms": 1.0,
                "latency_ms": scoring["latency_ms"],
                "peak_cuda_memory_bytes": None,
                "cnndetection_feature_path": str(feature_path),
                "cnndetection_feature_sha256": _sha256(feature_path),
                "cnndetection_feature_shape": [runner.FEATURE_DIMENSION],
                "cnndetection_feature_dtype": "float32",
                "cnndetection_feature_semantics": (
                    "official_fc_input_after_adaptive_global_average_pool"
                ),
                **scoring,
            }

            runner._validate_resume_row(
                row,
                expected=expected,
                repo_root=root,
                config_fingerprint="fingerprint",
                model=model,
                device=device,
            )

            row_mutations = (
                (
                    "preprocess",
                    lambda value: value["preprocess"].update(
                        {"tensor_sha256": "0" * 64}
                    ),
                    "preprocessing audit",
                ),
                (
                    "nested classification",
                    lambda value: value["classification"].update(
                        {"decision": True}
                    ),
                    "nested classification",
                ),
                (
                    "nested T1",
                    lambda value: value["t1"].update({"policy": "changed"}),
                    "nested T1",
                ),
                (
                    "manual replay",
                    lambda value: value["manual_replay"].update(
                        {"official_score_exact_match": False}
                    ),
                    "manual replay",
                ),
                (
                    "feature shape metadata",
                    lambda value: value.update(
                        {"cnndetection_feature_shape": [1]}
                    ),
                    "shape metadata",
                ),
                (
                    "feature dtype metadata",
                    lambda value: value.update(
                        {"cnndetection_feature_dtype": "float64"}
                    ),
                    "dtype metadata",
                ),
                (
                    "feature semantics",
                    lambda value: value.update(
                        {"cnndetection_feature_semantics": "changed"}
                    ),
                    "feature semantics",
                ),
            )
            for name, mutate, message in row_mutations:
                with self.subTest(tamper=name):
                    changed = copy.deepcopy(row)
                    mutate(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        runner._validate_resume_row(
                            changed,
                            expected=expected,
                            repo_root=root,
                            config_fingerprint="fingerprint",
                            model=model,
                            device=device,
                        )

            artifact_cases = (
                (
                    "nonfinite",
                    np.full(
                        runner.FEATURE_DIMENSION,
                        np.nan,
                        dtype=np.float32,
                    ),
                    "non-finite",
                ),
                (
                    "shape",
                    np.zeros(
                        runner.FEATURE_DIMENSION - 1,
                        dtype=np.float32,
                    ),
                    "feature shape",
                ),
                (
                    "dtype",
                    np.zeros(
                        runner.FEATURE_DIMENSION,
                        dtype=np.float64,
                    ),
                    "feature dtype",
                ),
            )
            for name, artifact, message in artifact_cases:
                with self.subTest(artifact_tamper=name):
                    np.save(feature_path, artifact, allow_pickle=False)
                    changed = copy.deepcopy(row)
                    changed["cnndetection_feature_sha256"] = _sha256(
                        feature_path
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        runner._validate_resume_row(
                            changed,
                            expected=expected,
                            repo_root=root,
                            config_fingerprint="fingerprint",
                            model=model,
                            device=device,
                        )
            np.save(feature_path, feature, allow_pickle=False)

    @unittest.skipUnless(HAS_OFFICIAL_ASSETS, "official assets not cached")
    def test_official_cpu_preflight_and_checkpoint_schema(self):
        report = runner.run_preflight(
            source_root=SOURCE_ROOT,
            checkpoint_path=CHECKPOINT_PATH,
            device_text="cpu",
        )
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["cuda_used"])
        self.assertFalse(report["mouse_inference_run"])
        self.assertEqual(len(report["golden_cases"]), 4)
        self.assertEqual(
            report["asset"]["sha256"],
            runner.CHECKPOINT["sha256"],
        )
        safety = report["asset"]["serialization_safety"]
        self.assertTrue(safety["weights_only"])
        self.assertTrue(safety["weights_only_load_succeeded"])
        self.assertFalse(safety["unrestricted_pickle_used"])
        self.assertFalse(
            safety["static_unsafe_global_scan"]["supported"]
        )
        schema = report["asset"]["schema"]
        self.assertEqual(schema["outer_keys"], ["model", "optimizer", "total_steps"])
        self.assertEqual(schema["state_entries"], 320)
        self.assertEqual(schema["state_elements"], 23_563_254)
        self.assertEqual(schema["total_steps"], 270_048)
        decisions = {
            (case["profile"], case["filename"]): case[
                "classification_decision"
            ]
            for case in report["golden_cases"]
        }
        for profile in (
            runner.PRIMARY_PROFILE,
            runner.PAPER_CROP_PROFILE,
        ):
            self.assertFalse(decisions[(profile, "real.png")])
            self.assertTrue(decisions[(profile, "fake.png")])


if __name__ == "__main__":
    unittest.main()
