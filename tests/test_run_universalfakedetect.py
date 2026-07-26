from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from eval.opensource import run_universalfakedetect as runner


SOURCE_ROOT = Path(
    "/root/.cache/claimforge/third_party/"
    "UniversalFakeDetect-76a0e3e60a8a"
)
HEAD_PATH = Path(
    "/root/.cache/claimforge/checkpoints/"
    "universal-fake-detect-76a0e3e60a8a/fc_weights.pth"
)
BACKBONE_PATH = Path(
    "/root/.cache/claimforge/checkpoints/openai-clip/ViT-L-14.pt"
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


class _TinyOfficialModel(torch.nn.Module):
    def __init__(self, drift: float = 0.0) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(768, 1)
        self.forward_calls = 0
        self.drift = drift
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.linspace(-0.1, 0.1, 768).reshape(1, -1)
            )
            self.fc.bias.fill_(0.125)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        flattened = image.reshape(image.shape[0], -1)
        feature = flattened[:, :768].contiguous()
        return self.fc(feature) + self.drift


class RunUniversalFakeDetectTest(unittest.TestCase):
    def test_frozen_contract_and_two_profiles_only(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "76a0e3e60a8a06458707a625d269ba815a2e5919",
        )
        self.assertEqual(
            set(runner.PREPROCESS_PROFILES),
            {
                "current_head_native_center_crop224",
                "checkpoint_era_resize256_center_crop224",
            },
        )
        self.assertEqual(runner.FEATURE_DIMENSION, 768)
        self.assertEqual(runner.MODEL_INPUT_SIZE, 224)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD, 0.5)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD_OPERATOR, ">")
        self.assertEqual(runner.HEAD_CHECKPOINT["bytes"], 4_083)
        self.assertEqual(
            runner.HEAD_CHECKPOINT["sha256"],
            "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
        )
        self.assertEqual(runner.BACKBONE_CHECKPOINT["bytes"], 932_768_134)
        self.assertEqual(
            runner.BACKBONE_CHECKPOINT["sha256"],
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        )
        self.assertEqual(
            runner.PREPROCESS_PROFILES[runner.CHECKPOINT_ERA_PROFILE][
                "source_commit"
            ],
            runner.HEAD_INTRO_COMMIT,
        )
        self.assertEqual(runner.LICENSE_SPDX, "MIT")

    def test_official_source_and_assets_pass_static_contract(self):
        source, assets, state = runner._verify_asset_contract(
            source_root=SOURCE_ROOT,
            head_checkpoint=HEAD_PATH,
            backbone_checkpoint=BACKBONE_PATH,
        )
        self.assertEqual(source["commit"], runner.MODEL_SOURCE_COMMIT)
        self.assertFalse(source["tracked_dirty"])
        self.assertEqual(list(state), ["weight", "bias"])
        self.assertEqual(tuple(state["weight"].shape), (1, 768))
        self.assertEqual(state["weight"].dtype, torch.float32)
        self.assertEqual(
            assets["head"]["serialization_safety"]["unsafe_globals"],
            [],
        )
        self.assertTrue(
            assets["head"]["serialization_safety"]["weights_only"]
        )
        self.assertTrue(
            assets["backbone"]["archive_preflight_passed"]
        )
        self.assertTrue(runner._valid_sha256(assets["bundle_sha256"]))

    def test_repository_contract_rejects_tracked_changes(self):
        with (
            mock.patch.object(
                runner,
                "_git_value",
                side_effect=[
                    runner.MODEL_SOURCE_COMMIT,
                    " M models/clip_models.py",
                ],
            ),
            mock.patch.object(runner, "_verify_runtime_file"),
        ):
            with self.assertRaisesRegex(ValueError, "tracked source tree is dirty"):
                runner._verify_source_contract(SOURCE_ROOT)

    def test_head_preflight_rejects_unsafe_globals(self):
        with mock.patch(
            "torch.serialization.get_unsafe_globals_in_checkpoint",
            return_value=["builtins.eval"],
        ):
            with self.assertRaisesRegex(ValueError, "contains unsafe globals"):
                runner._verify_asset_contract(
                    source_root=SOURCE_ROOT,
                    head_checkpoint=HEAD_PATH,
                    backbone_checkpoint=BACKBONE_PATH,
                )

    def test_current_profile_geometry_and_pixels_match_torchvision(self):
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            actual, audit = runner.preprocess_image(
                path,
                runner.CURRENT_PROFILE,
            )
            with Image.open(path) as opened:
                reference_crop = transforms.CenterCrop(224)(
                    opened.convert("RGB")
                )
            expected = transforms.Normalize(
                runner.CLIP_MEAN,
                runner.CLIP_STD,
            )(transforms.ToTensor()(reference_crop)).numpy()
            crop_array = np.ascontiguousarray(
                np.asarray(reference_crop, dtype=np.uint8)
            )
        np.testing.assert_array_equal(actual, expected)
        geometry = audit["geometry"]
        self.assertFalse(geometry["resize"]["enabled"])
        self.assertEqual(
            geometry["center_crop"]["start_xy"],
            [88, 38],
        )
        self.assertEqual(
            geometry["effective_native_crop_xyxy"],
            [88.0, 38.0, 312.0, 262.0],
        )
        self.assertEqual(
            audit["crop_rgb_sha256"],
            runner._sha256_array(crop_array),
        )
        self.assertEqual(audit["tensor_sha256"], runner._sha256_array(actual))

    def test_checkpoint_era_geometry_and_pixels_match_torchvision(self):
        rng = np.random.default_rng(8)
        rgb = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            actual, audit = runner.preprocess_image(
                path,
                runner.CHECKPOINT_ERA_PROFILE,
            )
            with Image.open(path) as opened:
                reference_crop = transforms.Compose(
                    [
                        transforms.Resize(
                            256,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                        transforms.CenterCrop(224),
                    ]
                )(opened.convert("RGB"))
            expected = transforms.Normalize(
                runner.CLIP_MEAN,
                runner.CLIP_STD,
            )(transforms.ToTensor()(reference_crop)).numpy()
        np.testing.assert_array_equal(actual, expected)
        geometry = audit["geometry"]
        self.assertTrue(geometry["resize"]["enabled"])
        self.assertEqual(
            geometry["resize"]["destination_size"],
            [341, 256],
        )
        self.assertEqual(
            geometry["center_crop"]["start_xy"],
            [58, 16],
        )
        self.assertEqual(audit["tensor_shape"], [3, 224, 224])
        self.assertEqual(audit["tensor_dtype"], "float32")

    def test_center_crop_padding_geometry_matches_torchvision(self):
        geometry = runner.compute_preprocess_geometry(
            31,
            20,
            runner.CURRENT_PROFILE,
        )
        self.assertEqual(
            geometry["center_crop"]["padding_ltrb"],
            [96, 102, 97, 102],
        )
        self.assertEqual(geometry["center_crop"]["start_xy"], [0, 0])
        self.assertEqual(
            geometry["effective_native_crop_xyxy"],
            [0.0, 0.0, 31.0, 20.0],
        )

    def test_mouse_visibility_distributions_are_profile_specific(self):
        repo_root = Path(__file__).resolve().parents[1]
        release, _inputs, rows = runner.load_release(
            repo_root,
            repo_root / runner.DEFAULT_DATASET_MANIFEST,
        )
        self.assertEqual(release["pairs"], 275)
        expected = {
            runner.CURRENT_PROFILE: Counter(
                {"none": 247, "partial": 14, "full": 14}
            ),
            runner.CHECKPOINT_ERA_PROFILE: Counter(
                {"none": 80, "partial": 33, "full": 162}
            ),
        }
        for profile_id, expected_counts in expected.items():
            visibility = runner.build_pair_visibility(
                rows,
                repo_root,
                profile_id,
            )
            self.assertEqual(
                Counter(
                    item["edit_visibility"]
                    for item in visibility.values()
                ),
                expected_counts,
            )

    def test_official_cpu_loader_uses_pinned_download_without_network(self):
        model, device, audit = runner.load_model(
            source_root=SOURCE_ROOT,
            head_checkpoint=HEAD_PATH,
            backbone_checkpoint=BACKBONE_PATH,
            device_name="cpu",
        )
        try:
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(type(model).__name__, "CLIPModel")
            self.assertEqual((model.fc.in_features, model.fc.out_features), (768, 1))
            self.assertEqual(audit["visual_input_resolution"], 224)
            self.assertEqual(audit["urlopen_calls"], 0)
            self.assertTrue(audit["network_blocked"])
            self.assertTrue(audit["clip_torch_load_fallback_blocked"])
            self.assertEqual(len(audit["official_download_patch_calls"]), 1)
            self.assertEqual(
                audit["official_download_patch_calls"][0]["url"],
                runner.BACKBONE_CHECKPOINT["official_url"],
            )
        finally:
            del model
            gc.collect()

    def test_infer_one_calls_model_once_and_replays_feature_exactly(self):
        model = _TinyOfficialModel()
        image = np.linspace(
            -1.0,
            1.0,
            3 * 224 * 224,
            dtype=np.float32,
        ).reshape(3, 224, 224)
        processed, feature, peak, latency = runner.infer_one(
            model,
            torch.device("cpu"),
            np.ascontiguousarray(image),
        )
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(feature.shape, (768,))
        self.assertEqual(feature.dtype, np.float32)
        self.assertEqual(peak, 0)
        self.assertGreaterEqual(latency, 0.0)
        self.assertTrue(
            processed["manual_replay"]["official_logit_exact_match"]
        )
        self.assertEqual(
            processed["classification_decision"],
            processed["probability"] > 0.5,
        )

    def test_manual_replay_rejects_official_forward_drift(self):
        model = _TinyOfficialModel(drift=1e-4)
        image = np.zeros((3, 224, 224), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "manual F.linear"):
            runner.infer_one(
                model,
                torch.device("cpu"),
                image,
            )

    def test_strict_threshold_does_not_mark_exact_half_fake(self):
        classifier = torch.nn.Linear(768, 1)
        with torch.no_grad():
            classifier.weight.zero_()
            classifier.bias.zero_()
        result = runner.replay_classifier(
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros((1, 768), dtype=torch.float32),
            classifier,
        )
        self.assertEqual(result["raw_logit"], 0.0)
        self.assertEqual(result["probability"], 0.5)
        self.assertFalse(result["classification_decision"])
        self.assertFalse(
            result["manual_replay"]["classification_decision"]
        )

    def test_resume_accepts_one_ulp_sigmoid_drift_and_checks_logit_sign(self):
        raw_logit = -7.389964580535889
        cpu_probability = runner._float32_sigmoid(raw_logit)
        gpu_probability = float(
            np.nextafter(
                np.float32(cpu_probability),
                np.float32(np.inf),
                dtype=np.float32,
            )
        )
        self.assertEqual(gpu_probability, 0.0006170368869788945)
        self.assertFalse(
            runner._validated_resume_decision(
                raw_logit,
                gpu_probability,
                sample_id="one-ulp",
            )
        )

        tiny_positive_logit = float(
            np.nextafter(
                np.float32(0.0),
                np.float32(np.inf),
                dtype=np.float32,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "decision boundary mismatch",
        ):
            runner._validated_resume_decision(
                tiny_positive_logit,
                0.5,
                sample_id="boundary",
            )

    def test_select_inputs_preserves_pairs_and_supports_sample_preflight(self):
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

    def test_manifest_fingerprint_rejects_profile_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            first = {
                "fingerprint": runner._manifest_fingerprint(
                    {"profile": runner.CURRENT_PROFILE}
                )
            }
            runner._write_or_validate_run_manifest(path, first)
            runner._write_or_validate_run_manifest(path, dict(first))
            second = {
                "fingerprint": runner._manifest_fingerprint(
                    {"profile": runner.CHECKPOINT_ERA_PROFILE}
                )
            }
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                runner._write_or_validate_run_manifest(path, second)

    def test_resume_validates_retry_history_provenance_aliases_and_artifacts(self):
        fingerprint = "a" * 64
        bundle = "b" * 64
        inputs_hash = "c" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            image_path = root / "images" / "sample.png"
            rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(
                20,
                30,
                3,
            )
            Image.fromarray(rgb, mode="RGB").save(image_path)
            feature_path = root / "feature.npy"
            runner._atomic_save_npy(
                feature_path,
                np.arange(768, dtype=np.float32),
            )
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
                "canonical_sha256": runner.sha256_file(image_path),
                "gt_mask_kind": "all_zero",
                "gt_mask_sha256": None,
                "edit_region_xyxy": [10, 8, 20, 12],
            }
            visibility = {
                "task": {
                    "edit_visibility": "full",
                    "edit_visible_gt_fraction": 1.0,
                    "edit_visibility_evidence": {
                        "gt": {
                            "category": "full",
                            "visible_fraction": 1.0,
                        },
                        "edit_box": {
                            "category": "full",
                            "visible_fraction": 1.0,
                        },
                    },
                }
            }
            identity = {
                "schema_version": "opensource_result_v1",
                "run_id": "run",
                "run_manifest_fingerprint": fingerprint,
                "input_manifest_sha256": inputs_hash,
                "id": "sample",
                "rank": 0,
                "task_id": "task",
                "pair_rank": 0,
                "domain": "lodging",
                "kind": "real",
                "label": 0,
                "image_path": "images/sample.png",
                "image_sha256": expected["canonical_sha256"],
                "image_size": [30, 20],
                "gt_mask_kind": "all_zero",
                "gt_mask_sha256": None,
                "edit_region_xyxy": [10, 8, 20, 12],
                **visibility["task"],
                "model": runner.MODEL_NAME,
                "model_slug": runner.MODEL_SLUG,
                "model_arch": runner.MODEL_ARCH,
                "model_source_commit": runner.MODEL_SOURCE_COMMIT,
                "asset_bundle_sha256": bundle,
                "preprocess_profile": runner.CURRENT_PROFILE,
                "valid_for_t1": True,
                "valid_for_t2": False,
                "t1_policy": (
                    "official_UFD_CLIP_linear_probe_probability"
                ),
                "t2_policy": "unsupported_whole_image_detector",
                "completed_at": "2026-07-24T00:00:00+00:00",
            }
            error = {
                **identity,
                "status": "error",
                "valid_for_metrics": False,
                "error_type": "RuntimeError",
                "error_message": "retry",
                "traceback": "trace",
            }
            probability = 0.5
            classification = {
                "raw_logit": 0.0,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "decision": False,
                "threshold": 0.5,
                "threshold_operator": ">",
                "semantics": (
                    "official_sigmoid_probability_higher_is_fake"
                ),
            }
            t1 = {
                key: value
                for key, value in classification.items()
                if key != "semantics"
            }
            t1["policy"] = (
                "official_UFD_CLIP_linear_probe_probability"
            )
            _tensor, preprocess = runner.preprocess_image(
                image_path,
                runner.CURRENT_PROFILE,
            )
            ok = {
                **identity,
                "status": "ok",
                "valid_for_metrics": True,
                "raw_logit": 0.0,
                "probability": probability,
                "ai_score": probability,
                "score": probability,
                "score_semantics": (
                    "official_sigmoid_probability_higher_is_fake"
                ),
                "classification_decision": False,
                "classification_threshold": 0.5,
                "classification_threshold_operator": ">",
                "classification": classification,
                "t1": t1,
                "manual_replay": {
                    "raw_logit": 0.0,
                    "probability": probability,
                    "ai_score": probability,
                    "classification_decision": False,
                    "official_logit_exact_match": True,
                    "official_probability_exact_match": True,
                    "model_forward_calls": 1,
                    "fc_hook_calls": 1,
                },
                "clip_feature_path": str(feature_path),
                "clip_feature_sha256": runner.sha256_file(feature_path),
                "clip_feature_shape": [768],
                "clip_feature_dtype": "float32",
                "clip_feature_semantics": (
                    "official_CLIP_encode_image_output_before_linear_head"
                ),
                "artifact_paths": {
                    "clip_feature_npy": str(feature_path)
                },
                "preprocess": preprocess,
                "latency_ms": 1.0,
                "peak_cuda_memory_bytes": 0,
            }
            kwargs = {
                "run_id": "run",
                "input_manifest_sha256": inputs_hash,
                "profile_id": runner.CURRENT_PROFILE,
                "pair_visibility": visibility,
                "repo_root": root,
            }
            runner._validate_resume_rows(
                [error, ok],
                [expected],
                fingerprint,
                bundle,
                **kwargs,
            )
            raw_logit = -7.389964580535889
            cpu_probability = runner._float32_sigmoid(raw_logit)
            one_ulp_probability = float(
                np.nextafter(
                    np.float32(cpu_probability),
                    np.float32(np.inf),
                    dtype=np.float32,
                )
            )
            self.assertEqual(
                one_ulp_probability,
                0.0006170368869788945,
            )
            self.assertNotEqual(one_ulp_probability, cpu_probability)
            one_ulp_ok = {
                **ok,
                "raw_logit": raw_logit,
                "probability": one_ulp_probability,
                "ai_score": one_ulp_probability,
                "score": one_ulp_probability,
                "classification_decision": False,
                "classification": {
                    **classification,
                    "raw_logit": raw_logit,
                    "probability": one_ulp_probability,
                    "ai_score": one_ulp_probability,
                    "score": one_ulp_probability,
                },
                "t1": {
                    **t1,
                    "raw_logit": raw_logit,
                    "probability": one_ulp_probability,
                    "ai_score": one_ulp_probability,
                    "score": one_ulp_probability,
                },
                "manual_replay": {
                    **ok["manual_replay"],
                    "raw_logit": raw_logit,
                    "probability": one_ulp_probability,
                    "ai_score": one_ulp_probability,
                },
            }
            runner._validate_resume_rows(
                [one_ulp_ok],
                [expected],
                fingerprint,
                bundle,
                **kwargs,
            )
            wrong_run = {**ok, "run_id": "copied"}
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                runner._validate_resume_rows(
                    [wrong_run],
                    [expected],
                    fingerprint,
                    bundle,
                    **kwargs,
                )
            wrong_alias = {
                **ok,
                "ai_score": math.nextafter(0.5, math.inf),
            }
            with self.assertRaisesRegex(ValueError, "score aliases"):
                runner._validate_resume_rows(
                    [wrong_alias],
                    [expected],
                    fingerprint,
                    bundle,
                    **kwargs,
                )
            wrong_error = {**error, "ai_score": 0.5}
            with self.assertRaisesRegex(ValueError, "invalid resume error"):
                runner._validate_resume_rows(
                    [wrong_error],
                    [expected],
                    fingerprint,
                    bundle,
                    **kwargs,
                )
            feature_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner._validate_resume_rows(
                    [ok],
                    [expected],
                    fingerprint,
                    bundle,
                    **kwargs,
                )

    def test_completed_resume_preserves_audited_no_network_model_load(self):
        bundle = "d" * 64
        audit = {
            "class_module": (
                "_claimforge_ufd_76a0e3e.models.clip_models"
            ),
            "class_name": "CLIPModel",
            "construction_api": (
                "official models.get_model('CLIP:ViT-L/14')"
            ),
            "network_blocked": True,
            "urlopen_calls": 0,
            "clip_torch_load_fallback_blocked": True,
            "feature_dimension": 768,
            "visual_input_resolution": 224,
            "head_parameters": 769,
            "source": {"commit": runner.MODEL_SOURCE_COMMIT},
            "assets": {
                "bundle_sha256": bundle,
                "head": {
                    "serialization_safety": {
                        "unsafe_globals": [],
                        "required_unsafe_globals": [],
                        "weights_only": True,
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "run_manifest_fingerprint": "a" * 64,
                        "asset_bundle_sha256": bundle,
                        "preprocess_profile": runner.CURRENT_PROFILE,
                        "model_load_audit": audit,
                    }
                ),
                encoding="utf-8",
            )
            loaded = runner._load_completed_model_audit(
                path,
                run_id="run",
                manifest_fingerprint="a" * 64,
                asset_bundle_sha256=bundle,
                profile_id=runner.CURRENT_PROFILE,
            )
            self.assertEqual(loaded, audit)
            audit["network_blocked"] = False
            path.write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "run_manifest_fingerprint": "a" * 64,
                        "asset_bundle_sha256": bundle,
                        "preprocess_profile": runner.CURRENT_PROFILE,
                        "model_load_audit": audit,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model_load_audit"):
                runner._load_completed_model_audit(
                    path,
                    run_id="run",
                    manifest_fingerprint="a" * 64,
                    asset_bundle_sha256=bundle,
                    profile_id=runner.CURRENT_PROFILE,
                )

    def test_cli_requires_all_explicit_assets_and_profile(self):
        with mock.patch.object(
            sys,
            "argv",
            ["run_universalfakedetect.py", "--run-id", "test"],
        ):
            with self.assertRaises(SystemExit):
                runner.parse_args()
        argv = [
            "run_universalfakedetect.py",
            "--run-id",
            "test",
            "--source-root",
            str(SOURCE_ROOT),
            "--head-checkpoint",
            str(HEAD_PATH),
            "--backbone-checkpoint",
            str(BACKBONE_PATH),
            "--preprocess-profile",
            runner.CURRENT_PROFILE,
        ]
        with mock.patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.source_root, SOURCE_ROOT)
        self.assertEqual(args.head_checkpoint, HEAD_PATH)
        self.assertEqual(args.backbone_checkpoint, BACKBONE_PATH)
        self.assertEqual(args.preprocess_profile, runner.CURRENT_PROFILE)

    def test_run_rejects_bad_id_profile_threshold_and_missing_paths_before_io(self):
        args = argparse.Namespace(
            run_id="../escape",
            preprocess_profile=runner.CURRENT_PROFILE,
            classification_threshold=0.5,
            bootstrap_samples=10,
            source_root=SOURCE_ROOT,
            head_checkpoint=HEAD_PATH,
            backbone_checkpoint=BACKBONE_PATH,
        )
        with self.assertRaisesRegex(ValueError, "unsafe characters"):
            runner.run(args)
        args.run_id = "safe"
        args.preprocess_profile = "other"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            runner.run(args)
        args.preprocess_profile = runner.CURRENT_PROFILE
        args.classification_threshold = math.nextafter(0.5, math.inf)
        with self.assertRaisesRegex(ValueError, "threshold must be 0.5"):
            runner.run(args)
        args.classification_threshold = 0.5
        args.head_checkpoint = None
        with self.assertRaisesRegex(ValueError, "mandatory"):
            runner.run(args)


if __name__ == "__main__":
    unittest.main()
