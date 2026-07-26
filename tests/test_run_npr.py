from __future__ import annotations

import tempfile
import unittest
from collections import Counter, OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as torch_functional
from torchvision.transforms import functional as vision_functional

from eval.opensource import run_npr as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _selection_rows(pair_count: int = 3) -> list[dict]:
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
                    "kind": kind,
                    "label": int(kind == "forged"),
                }
            )
    return rows


def _preprocess_fixture() -> np.ndarray:
    return np.arange(45, dtype=np.uint8).reshape(3, 5, 3)


class _TinyNpr(torch.nn.Module):
    def __init__(self, *, bias: float, output_offset: float = 0.0) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(runner.FEATURE_DIMENSION, 1)
        self.register_buffer(
            "fixed_feature",
            torch.linspace(
                -1.0,
                1.0,
                runner.FEATURE_DIMENSION,
                dtype=torch.float32,
            ),
        )
        self.forward_calls = 0
        self.output_offset = output_offset
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.fill_(bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        feature = self.fixed_feature.unsqueeze(0).expand(image.shape[0], -1)
        return self.fc1(feature) + self.output_offset


class RunNprTests(unittest.TestCase):
    def _require_local_assets(self) -> tuple[Path, Path]:
        source_root = runner.DEFAULT_SOURCE_ROOT
        checkpoint = runner.DEFAULT_CHECKPOINT
        if not source_root.is_dir() or not checkpoint.is_file():
            self.skipTest("frozen NPR source/checkpoint are not present")
        return source_root, checkpoint

    def _require_mouse_release(self) -> Path:
        manifest = REPO_ROOT / runner.DEFAULT_DATASET_MANIFEST
        if not manifest.is_file():
            self.skipTest("canonical Mouse release is not present")
        return manifest

    def test_frozen_release_constants_source_and_checkpoint_contract(self):
        self.assertEqual(runner.MODEL_NAME, "NPR")
        self.assertEqual(
            runner.MODEL_SLUG,
            "npr_aigcdetect_progan4class",
        )
        self.assertEqual(
            runner.MODEL_ARCH,
            "NPR truncated ResNet-50 (stem + layer1 + layer2)",
        )
        self.assertEqual(
            runner.MODEL_REPO_URL,
            "https://github.com/chuangchuangtan/NPR-DeepfakeDetection",
        )
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a",
        )
        self.assertEqual(
            runner.HF_SPACE_COMMIT,
            "522a9f1020f7454d486f28a0d5c148ec37919b32",
        )
        self.assertEqual(
            runner.PREPROCESS_PROFILE,
            "official_aigcdetect_native_even_trim",
        )
        self.assertEqual(runner.MODEL_SEED, 100)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD, 0.5)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD_OPERATOR, ">")
        self.assertEqual(
            runner.CLASSIFICATION_THRESHOLD,
            runner.FIXED_THRESHOLD,
        )
        self.assertEqual(
            runner.CLASSIFICATION_THRESHOLD_OPERATOR,
            runner.THRESHOLD_OPERATOR,
        )
        self.assertEqual(runner.FEATURE_DIMENSION, 512)
        self.assertEqual(runner.IMAGE_MEAN, (0.485, 0.456, 0.406))
        self.assertEqual(runner.IMAGE_STD, (0.229, 0.224, 0.225))

        self.assertEqual(
            runner.CHECKPOINT,
            {
                "id": "NPR-AIGC-ProGAN4class@68338a",
                "filename": "model_epoch_last_3090.pth",
                "repo_relative_path": "model_epoch_last_3090.pth",
                "introduced_commit": (
                    "68338a07847e891534f3d0b0a0e25bb137b684f7"
                ),
                "bytes": 5_842_385,
                "sha256": (
                    "b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3feb"
                    "d2ae5618a80f7e8a"
                ),
                "state_entries": 146,
                "state_elements": 1_447_897,
                "trainable_parameters": 1_437_761,
                "format": "torch_ordered_state_dict_weights_only",
                "official_role": (
                    "AIGCDetectBenchmark ProGAN-4class checkpoint; also "
                    "downloaded by the HF demo"
                ),
                "pinned_url": (
                    "https://raw.githubusercontent.com/chuangchuangtan/"
                    "NPR-DeepfakeDetection/"
                    "68338a07847e891534f3d0b0a0e25bb137b684f7/"
                    "model_epoch_last_3090.pth"
                ),
            },
        )
        self.assertEqual(
            runner.SOURCE_FILES,
            {
                "README.md": (
                    "65d4d9806fea6ab49ecaa6d0bd20e32380348d4858c9a630"
                    "5d09387b64644871"
                ),
                "test.py": (
                    "fbe86617998638b325be8d12eac15e903817ebc2a54784e4"
                    "0266cf7e1f788e79"
                ),
                "validate.py": (
                    "dca8aa2aed02d6630ba4d58feab69e2f80e4f76e803d5365"
                    "bf35b2b9ca776be3"
                ),
                "data/datasets.py": (
                    "6b82e0bfc5251ba12e94b10bd9a9b8bcb9405472dbd3b107"
                    "73035dfd5907717f"
                ),
                "options/base_options.py": (
                    "e967e19807ab44ecefaeddf7e75022dfe0f470a8bd3326fc"
                    "806eca0099139e91"
                ),
                "options/test_options.py": (
                    "d0a6e520c9d4a9b034ba237f97d594a0284872879042ba5a"
                    "7325b601adced10e"
                ),
                "networks/resnet.py": (
                    "c7663a02a322dc2a68625535b367e8800df42bacca3d0526c"
                    "afef8c32168c67e"
                ),
            },
        )
        self.assertEqual(
            set(runner.EXCLUDED_RELEASE_ASSETS),
            {"NPR.pth", "NPR_GenImage_sdv4.pth"},
        )
        self.assertEqual(
            runner.EXCLUDED_RELEASE_ASSETS["NPR.pth"]["sha256"],
            "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53",
        )
        self.assertEqual(
            runner.EXCLUDED_RELEASE_ASSETS[
                "NPR_GenImage_sdv4.pth"
            ]["sha256"],
            "9bc961e7d643581aa0ea879cbd322dcc2e543877568a43d2f6cdb92906379015",
        )

    def test_adapter_contract_covers_exact_runtime_components(self):
        contract = runner.adapter_contract(REPO_ROOT)
        self.assertEqual(
            set(contract),
            {
                "eval/opensource/run_npr.py",
                "eval/opensource/npr_metrics.py",
                "eval/opensource/common.py",
                "eval/opensource/maskclip_metrics.py",
            },
        )
        for relative, evidence in contract.items():
            path = REPO_ROOT / relative
            self.assertEqual(evidence["path"], str(path.resolve()))
            self.assertEqual(evidence["bytes"], path.stat().st_size)
            self.assertEqual(
                evidence["sha256"],
                runner.sha256_file(path),
            )

    def test_selection_supports_pairs_and_one_image_preflight(self):
        rows = _selection_rows()
        self.assertEqual(
            runner.select_inputs(rows, pair_limit=2),
            rows[:4],
        )
        self.assertEqual(
            runner.select_inputs(rows, pair_limit=None),
            rows,
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
            runner.select_inputs(
                rows,
                pair_limit=1,
                sample_id="real-0",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.select_inputs(
                rows,
                pair_limit=None,
                sample_id="missing",
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            runner.select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            runner.select_inputs(rows[:-1], pair_limit=None)

    def test_preprocess_is_pixel_identical_to_official_torchvision_path(self):
        rgb = _preprocess_fixture()
        official = vision_functional.normalize(
            vision_functional.to_tensor(Image.fromarray(rgb)),
            runner.IMAGE_MEAN,
            runner.IMAGE_STD,
        )
        expected = official[:, :2, :4].contiguous()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.png"
            Image.fromarray(rgb).save(path)
            actual, audit = runner.preprocess_image(path)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(audit["decoded_size"], [5, 3])
        self.assertEqual(audit["effective_size"], [4, 2])
        self.assertEqual(audit["tensor_shape"], [3, 2, 4])
        self.assertEqual(audit["tensor_dtype"], "float32")
        self.assertEqual(
            audit["tensor_sha256"],
            "4b6aa1f5624a49f13ef57303277b986eb9779c3afdecb485524ee18c7f141936",
        )

    def test_odd_dimensions_trim_only_bottom_and_right(self):
        rgb = _preprocess_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.png"
            Image.fromarray(rgb).save(path)
            trimmed, audit = runner.preprocess_image(path)

        self.assertEqual(runner.effective_native_size(5, 3), (4, 2))
        self.assertEqual(audit["trim_bottom"], 1)
        self.assertEqual(audit["trim_right"], 1)
        expected_untrimmed = vision_functional.normalize(
            vision_functional.to_tensor(Image.fromarray(rgb)),
            runner.IMAGE_MEAN,
            runner.IMAGE_STD,
        )
        torch.testing.assert_close(
            trimmed,
            expected_untrimmed[:, :2, :4],
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaisesRegex(ValueError, "exceed one pixel"):
            runner.effective_native_size(1, 2)

    def test_npr_residual_formula_and_frozen_hash(self):
        rgb = _preprocess_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.png"
            Image.fromarray(rgb).save(path)
            tensor, audit = runner.preprocess_image(path)

        down = torch_functional.interpolate(
            tensor.unsqueeze(0),
            scale_factor=0.5,
            mode="nearest",
            recompute_scale_factor=True,
        )
        reconstructed = torch_functional.interpolate(
            down,
            scale_factor=2.0,
            mode="nearest",
            recompute_scale_factor=True,
        )
        residual = (
            tensor.unsqueeze(0) - reconstructed
        ).squeeze(0).contiguous()
        self.assertEqual(
            runner._tensor_sha256(residual),
            audit["npr_residual_sha256"],
        )
        self.assertEqual(
            audit["npr_residual_sha256"],
            "2035d15130a735b2a16219ecc26046cd2289d9a13b05a26348f14e48090620c2",
        )
        self.assertEqual(audit["npr_residual_shape"], [3, 2, 4])
        self.assertEqual(audit["npr_residual_dtype"], "float32")
        self.assertEqual(
            audit["npr_residual_stats"]["nonzero_elements"],
            int(torch.count_nonzero(residual).item()),
        )

    def test_first_mouse_image_has_frozen_tensor_and_residual_hashes(self):
        manifest = self._require_mouse_release()
        _release, _inputs_path, rows = runner.load_release(
            REPO_ROOT,
            manifest,
        )
        first = rows[0]
        self.assertEqual(
            first["sample_id"],
            "0cca3f606edf434904f7bea5",
        )
        tensor, audit = runner.preprocess_image(
            REPO_ROOT / first["canonical_path"]
        )
        self.assertEqual(list(tensor.shape), [3, 1350, 1800])
        self.assertEqual(
            audit["decoded_rgb_sha256"],
            "3a8d388733c61a4d56ad2c55b54ae65f055d0bf4a975ed36aacd2925e8a96a30",
        )
        self.assertEqual(
            audit["tensor_sha256"],
            "7e46ed1649a103f7e6044b25f47d05234f2c01cb8ea7f77d6b07fd45ddfab958",
        )
        self.assertEqual(
            audit["npr_residual_sha256"],
            "f10a311f00fb1b40e5dc9c69f0ec4211bb99f32ddf69d32eef93dffd75be0305",
        )

    def test_mouse_pair_parity_visibility_and_trim_census(self):
        manifest = self._require_mouse_release()
        release, _inputs_path, rows = runner.load_release(
            REPO_ROOT,
            manifest,
        )
        self.assertEqual(release["pairs"], 275)
        self.assertEqual(release["images"], 550)
        self.assertEqual(len(rows), 550)

        by_task: dict[str, list[dict]] = {}
        for row in rows:
            by_task.setdefault(str(row["task_id"]), []).append(row)
        self.assertEqual(len(by_task), 275)
        for pair_rows in by_task.values():
            self.assertEqual(
                {row["kind"] for row in pair_rows},
                {"real", "forged"},
            )
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(
                {
                    (int(row["width"]), int(row["height"]))
                    for row in pair_rows
                },
                {
                    (
                        int(pair_rows[0]["width"]),
                        int(pair_rows[0]["height"]),
                    )
                },
            )

        trimmed_images = sum(
            runner.effective_native_size(
                int(row["width"]),
                int(row["height"]),
            )
            != (int(row["width"]), int(row["height"]))
            for row in rows
        )
        self.assertEqual(trimmed_images, 134)

        visibility = runner.build_pair_visibility(rows, REPO_ROOT)
        self.assertEqual(len(visibility), 275)
        self.assertEqual(
            Counter(
                value["edit_visibility"]
                for value in visibility.values()
            ),
            Counter({"full": 275}),
        )
        self.assertTrue(
            all(
                value["edit_visible_gt_fraction"] == 1.0
                and value["edit_visible_gt_pixels"]
                == value["edit_total_gt_pixels"]
                for value in visibility.values()
            )
        )
        self.assertEqual(
            sum(
                bool(value["trim_bottom"] or value["trim_right"])
                for value in visibility.values()
            ),
            67,
        )

    def test_source_contract_rejects_tracked_dirty_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            with mock.patch.object(
                runner,
                "_git_value",
                side_effect=[
                    runner.MODEL_SOURCE_COMMIT,
                    " M networks/resnet.py",
                ],
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "tracked source tree is dirty",
                ):
                    runner._verify_source_contract(source_root)

    def test_pinned_hf_source_only_corroborates_checkpoint_and_preprocess(self):
        if not runner.DEFAULT_HF_SOURCE_ROOT.is_dir():
            self.skipTest("pinned NPR Hugging Face source is not present")
        audit = runner._verify_hf_source_contract(
            runner.DEFAULT_HF_SOURCE_ROOT
        )
        self.assertEqual(audit["commit"], runner.HF_SPACE_COMMIT)
        self.assertFalse(
            audit["deployment_mode_defect"]["calls_model_eval"]
        )
        self.assertIn(
            "checkpoint and native-size preprocessing only",
            audit["role"],
        )

    def test_checkpoint_loader_requires_weights_only_and_rejects_unsafe(self):
        source_root, checkpoint = self._require_local_assets()
        source_audit = {
            "commit": runner.MODEL_SOURCE_COMMIT,
        }

        with (
            mock.patch.object(
                runner,
                "_verify_source_contract",
                return_value=source_audit,
            ),
            mock.patch.object(
                torch.serialization,
                "get_unsafe_globals_in_checkpoint",
                return_value=["malicious.fixture.Payload"],
            ) as unsafe_globals,
        ):
            with self.assertRaisesRegex(ValueError, "unsafe globals"):
                runner.verify_assets(
                    source_root=source_root,
                    checkpoint_path=checkpoint,
                )
        unsafe_globals.assert_called_once_with(checkpoint.resolve())

        with (
            mock.patch.object(
                runner,
                "_verify_source_contract",
                return_value=source_audit,
            ),
            mock.patch.object(
                torch.serialization,
                "get_unsafe_globals_in_checkpoint",
                return_value=[],
            ),
            mock.patch.object(
                torch,
                "load",
                return_value={"weight": torch.zeros(1)},
            ) as load,
        ):
            with self.assertRaisesRegex(ValueError, "flat OrderedDict"):
                runner.verify_assets(
                    source_root=source_root,
                    checkpoint_path=checkpoint,
                )
        load.assert_called_once_with(
            checkpoint.resolve(),
            map_location="cpu",
            weights_only=True,
        )

    def test_real_checkpoint_is_safe_and_strictly_loads_official_model(self):
        source_root, checkpoint = self._require_local_assets()
        original_load = torch.load
        with mock.patch.object(
            torch,
            "load",
            wraps=original_load,
        ) as load:
            source_audit, asset_audit, state, module = runner.verify_assets(
                source_root=source_root,
                checkpoint_path=checkpoint,
            )

        load.assert_called_once_with(
            checkpoint.resolve(),
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(
            source_audit["commit"],
            runner.MODEL_SOURCE_COMMIT,
        )
        self.assertIsInstance(state, OrderedDict)
        self.assertEqual(
            asset_audit["checkpoint"]["serialization_safety"],
            {
                "unsafe_globals": [],
                "weights_only": True,
                "map_location": "cpu",
            },
        )
        self.assertEqual(
            asset_audit["checkpoint"]["schema"],
            {
                "container": "collections.OrderedDict",
                "entries": 146,
                "elements": 1_447_897,
                "items_sha256": (
                    "e60d79370c937aede4ff54ff57663207b6282f566c28caf56"
                    "f6afd924af530d6"
                ),
            },
        )
        self.assertEqual(
            asset_audit["bundle_sha256"],
            "9c19c48e4a3a42f4628b89445e2a39fe564802efbfb8c93854aeae55dfa81b66",
        )

        model = runner.load_model(
            module=module,
            state=state,
            device=torch.device("cpu"),
        )
        self.assertFalse(model.training)
        self.assertEqual(list(model.state_dict()), list(state))
        self.assertEqual(
            sum(int(parameter.numel()) for parameter in model.parameters()),
            runner.CHECKPOINT["trainable_parameters"],
        )

        incomplete = OrderedDict(state)
        incomplete.popitem()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            runner.load_model(
                module=module,
                state=incomplete,
                device=torch.device("cpu"),
            )

    def test_tiny_model_fc1_hook_manual_replay_and_strict_threshold(self):
        tensor = torch.zeros((3, 2, 4), dtype=torch.float32)
        cases = ((0.0, 0.5, False), (1.0, None, True))
        for bias, exact_probability, expected_decision in cases:
            with self.subTest(bias=bias):
                model = _TinyNpr(bias=bias)
                self.assertEqual(len(model.fc1._forward_pre_hooks), 0)
                scoring, feature = runner._infer_one(
                    model=model,
                    tensor=tensor,
                    device=torch.device("cpu"),
                )

                self.assertEqual(model.forward_calls, 1)
                self.assertEqual(len(model.fc1._forward_pre_hooks), 0)
                np.testing.assert_array_equal(
                    feature,
                    model.fixed_feature.numpy(),
                )
                self.assertEqual(feature.shape, (runner.FEATURE_DIMENSION,))
                self.assertEqual(feature.dtype, np.float32)
                if exact_probability is not None:
                    self.assertEqual(
                        scoring["probability"],
                        exact_probability,
                    )
                self.assertEqual(
                    scoring["probability"],
                    scoring["ai_score"],
                )
                self.assertEqual(
                    scoring["probability"],
                    scoring["score"],
                )
                self.assertEqual(
                    scoring["classification_decision"],
                    expected_decision,
                )
                self.assertEqual(
                    scoring["classification_threshold_operator"],
                    ">",
                )
                self.assertEqual(
                    scoring["classification"]["decision"],
                    expected_decision,
                )
                self.assertEqual(
                    scoring["t1"]["decision"],
                    expected_decision,
                )
                self.assertEqual(
                    scoring["manual_replay"]["classification_decision"],
                    expected_decision,
                )
                self.assertEqual(
                    scoring["manual_replay"]["model_forward_calls"],
                    1,
                )
                self.assertEqual(
                    scoring["manual_replay"]["fc_hook_calls"],
                    1,
                )
                self.assertTrue(
                    scoring["manual_replay"]["official_logit_exact_match"]
                )
                self.assertTrue(
                    scoring["manual_replay"][
                        "official_probability_exact_match"
                    ]
                )
                self.assertIsNone(scoring["peak_cuda_memory_bytes"])
                self.assertGreaterEqual(scoring["latency_ms"], 0.0)

        mismatched = _TinyNpr(bias=0.0, output_offset=1.0)
        with self.assertRaisesRegex(ValueError, "manual replay"):
            runner._infer_one(
                model=mismatched,
                tensor=tensor,
                device=torch.device("cpu"),
            )
        self.assertEqual(mismatched.forward_calls, 1)
        self.assertEqual(len(mismatched.fc1._forward_pre_hooks), 0)

    def test_resume_row_requires_complete_replay_and_finite_feature(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            image_path = run_dir / "fixture.png"
            Image.fromarray(_preprocess_fixture()).save(image_path)
            tensor, preprocess = runner.preprocess_image(image_path)
            scoring, feature = runner._infer_one(
                model=_TinyNpr(bias=0.0),
                tensor=tensor,
                device=torch.device("cpu"),
            )
            feature_path = run_dir / "feature.npy"
            np.save(feature_path, feature, allow_pickle=False)
            identity = {
                "schema_version": "npr_detection_result_v1",
                "id": "sample",
                "sample_id": "sample",
                "rank": 0,
                "pair_rank": 0,
                "task_id": "task",
                "kind": "forged",
                "label": 1,
                "domain": "lodging",
                "candidate": "mouse",
                "dataset_id": "fixture",
                "input_path": str(image_path),
                "input_sha256": "a" * 64,
                "input_width": 5,
                "input_height": 3,
                "preprocess_profile": runner.PREPROCESS_PROFILE,
                "checkpoint_id": runner.CHECKPOINT["id"],
                "config_fingerprint": "b" * 64,
                "edit_visibility": "full",
                "edit_visible_gt_fraction": 1.0,
                "edit_visibility_evidence": {
                    "edit_visibility": "full",
                },
                "task_scope": {
                    "valid_for_t1": True,
                    "valid_for_t2": False,
                    "native_dense_output": False,
                },
            }
            row = {
                **identity,
                "status": "ok",
                "preprocess": preprocess,
                "preprocess_latency_ms": 1.0,
                "npr_feature_path": str(feature_path),
                "npr_feature_sha256": runner.sha256_file(feature_path),
                "npr_feature_shape": [runner.FEATURE_DIMENSION],
                "npr_feature_dtype": "float32",
                "npr_feature_semantics": (
                    "official_fc1_input_after_adaptive_global_average_pool"
                ),
                **scoring,
            }
            runner._validate_resume_row(
                row,
                expected=identity,
                repo_root=REPO_ROOT,
                run_dir=run_dir,
                config_fingerprint="b" * 64,
            )

            tampered = dict(row)
            tampered["raw_logit"] = 1.0
            with self.assertRaisesRegex(ValueError, "classification aliases"):
                runner._validate_resume_row(
                    tampered,
                    expected=identity,
                    repo_root=REPO_ROOT,
                    run_dir=run_dir,
                    config_fingerprint="b" * 64,
                )

            bad_feature = feature.copy()
            bad_feature[0] = np.nan
            np.save(feature_path, bad_feature, allow_pickle=False)
            row["npr_feature_sha256"] = runner.sha256_file(feature_path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                runner._validate_resume_row(
                    row,
                    expected=identity,
                    repo_root=REPO_ROOT,
                    run_dir=run_dir,
                    config_fingerprint="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
