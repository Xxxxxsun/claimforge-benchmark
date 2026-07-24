import argparse
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from eval.opensource import run_mesorch as runner


def _rows(pair_count: int = 2) -> list[dict]:
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
                    "width": 3,
                    "height": 2,
                    "canonical_path": f"images/{kind}-{pair_rank}.jpg",
                    "canonical_sha256": f"{rank + 1:064x}",
                    "gt_mask_sha256": (
                        f"{rank + 101:064x}" if kind == "forged" else None
                    ),
                }
            )
    return rows


class _TinyMesorch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resize = torch.nn.Upsample(
            size=(runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        self.calls = 0
        self.seen_dummy_mask: torch.Tensor | None = None

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        self.calls += 1
        self.seen_dummy_mask = mask.detach().clone()
        logits = image[:, :1, ::4, ::4]
        resized_logits = self.resize(logits)
        probability = torch.sigmoid(resized_logits.float())
        return {
            "pred_mask": probability,
            "backward_loss": torch.nn.functional.binary_cross_entropy_with_logits(
                resized_logits,
                mask,
            ),
        }


class RunMesorchTest(unittest.TestCase):
    def test_select_inputs_keeps_complete_fixed_pairs(self):
        rows = _rows(3)
        selected = runner.select_inputs(rows, pair_limit=2)
        self.assertEqual(
            [(row["pair_rank"], row["kind"]) for row in selected],
            [
                (0, "real"),
                (0, "forged"),
                (1, "real"),
                (1, "forged"),
            ],
        )
        self.assertEqual(runner.select_inputs(rows, pair_limit=None), rows)
        for invalid in (0, -1):
            with self.subTest(pair_limit=invalid):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    runner.select_inputs(rows, pair_limit=invalid)
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            runner.select_inputs(rows[:-1], pair_limit=None)

    def test_preprocess_is_exact_rgb_cv2_linear_stretch_and_imagenet(self):
        import albumentations as albu
        from albumentations.pytorch import ToTensorV2

        rgb = np.asarray(
            [
                [[255, 0, 1], [0, 255, 2], [0, 0, 255]],
                [[12, 34, 56], [127, 128, 129], [253, 254, 255]],
            ],
            dtype=np.uint8,
        )
        official_transform = albu.Compose(
            [
                albu.Resize(512, 512),
                albu.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                albu.Crop(0, 0, 512, 512),
                ToTensorV2(transpose_mask=True),
            ]
        )
        expected = official_transform(
            image=rgb,
            mask=np.zeros(rgb.shape[:2], dtype=np.uint8),
        )["image"].numpy()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            tensor, native_size, metadata = runner.preprocess_image(path)

        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tensor.shape, (3, 512, 512))
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(metadata["decoder_format"], "PNG")
        self.assertEqual(metadata["channel_order"], "RGB")
        self.assertEqual(
            metadata["geometry"],
            "direct_stretch_without_aspect_ratio_preservation",
        )
        self.assertEqual(
            metadata["resize_interpolation"],
            "cv2.INTER_LINEAR_default",
        )
        self.assertEqual(metadata["tensor_sha256"], runner._sha256_array(tensor))

    def test_first_canonical_tensor_has_frozen_official_transform_hash(self):
        path = Path(
            "outputs/opensource/mouse_canonical_v1/images/"
            "0cca3f606edf434904f7bea5.jpg"
        )
        if not path.is_file():
            self.skipTest("canonical release is not present")
        tensor, native_size, metadata = runner.preprocess_image(path)
        self.assertEqual(native_size, (1800, 1350))
        self.assertEqual(
            runner._sha256_array(tensor),
            "9bd5b56c520796c8a75bc8016d4a373e4151e4006a4d231d57144a132089739d",
        )
        self.assertEqual(
            metadata["tensor_sha256"],
            "9bd5b56c520796c8a75bc8016d4a373e4151e4006a4d231d57144a132089739d",
        )

    def test_postprocess_preserves_official_probability_and_native_restore(self):
        logits = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]]]],
            dtype=torch.float32,
        )
        # The production contract is 128x128. Repeat each fixture cell so its
        # interpolation still distinguishes align_corners=True from False.
        logits = logits.repeat_interleave(64, dim=2).repeat_interleave(64, dim=3)
        official = torch.sigmoid(
            F.interpolate(
                logits,
                size=(runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
                mode="bilinear",
                align_corners=True,
            ).float()
        )
        expected_native = F.interpolate(
            official,
            size=(5, 9),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        processed = runner.postprocess_outputs(
            {"pred_mask": official},
            logits,
            native_width=9,
            native_height=5,
        )
        np.testing.assert_array_equal(
            processed["internal_logits_model_128"],
            logits[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_model_512"],
            official[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_native"],
            expected_native,
        )

    def test_postprocess_rejects_wrong_or_inconsistent_output_contract(self):
        logits = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        official = torch.full((1, 1, 512, 512), 0.5, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "not a mapping"):
            runner.postprocess_outputs(
                official,
                logits,
                native_width=2,
                native_height=2,
            )
        with self.assertRaisesRegex(ValueError, "pred_mask shape"):
            runner.postprocess_outputs(
                {"pred_mask": official[:, :, :-1]},
                logits,
                native_width=2,
                native_height=2,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner.postprocess_outputs(
                {"pred_mask": official + 0.1},
                logits,
                native_width=2,
                native_height=2,
            )

    def test_model_space_target_uses_nearest_binary_resize(self):
        target = np.asarray(
            [[True, False], [False, False]],
            dtype=bool,
        )
        resized = runner.model_space_target(target)
        self.assertEqual(resized.shape, (512, 512))
        self.assertEqual(resized.dtype, np.bool_)
        self.assertTrue(resized[:256, :256].all())
        self.assertFalse(resized[:256, 256:].any())
        self.assertFalse(resized[256:, :].any())

    def test_artifact_writers_are_lossless_and_typed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array_path = root / "map.npy"
            mask_path = root / "mask.png"
            array = np.asarray([[0.5, 0.50001]], dtype=np.float32)
            runner._atomic_save_npy(array_path, array)
            runner._atomic_save_mask(
                mask_path,
                array > runner.MASK_THRESHOLD,
            )
            loaded = np.load(array_path, allow_pickle=False)
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened, dtype=np.uint8)
        np.testing.assert_array_equal(loaded, array)
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(
            mask,
            np.asarray([[0, 255]], dtype=np.uint8),
        )

    def test_checkpoint_loader_safe_allowlists_namespace_and_loads_strictly(self):
        with tempfile.TemporaryDirectory() as temporary:
            good_path = Path(temporary) / "good.pth"
            bad_path = Path(temporary) / "bad.pth"
            source = torch.nn.Linear(2, 1)
            state = source.state_dict()
            payload = {
                "model": state,
                "optimizer": {},
                "epoch": 98,
                "scaler": None,
                "args": argparse.Namespace(model="MesorchFull"),
            }
            torch.save(payload, good_path)
            torch.save({key: value for key, value in payload.items() if key != "args"}, bad_path)
            contract = {
                "top_level_keys": [
                    "model",
                    "optimizer",
                    "epoch",
                    "scaler",
                    "args",
                ],
                "epoch": 98,
                "state_keys": len(state),
                "state_elements": sum(
                    int(value.numel()) for value in state.values()
                ),
                "tensor_bytes": sum(
                    int(value.numel()) * int(value.element_size())
                    for value in state.values()
                ),
                "state_dtypes": dict(
                    Counter(str(value.dtype) for value in state.values())
                ),
            }
            target = torch.nn.Linear(2, 1)
            loaded = runner._load_checkpoint_state(
                module=target,
                path=good_path,
                contract=contract,
                label="fixture",
            )
            for expected, actual in zip(
                source.parameters(),
                loaded.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
            with self.assertRaisesRegex(ValueError, "top-level schema"):
                runner._load_checkpoint_state(
                    module=torch.nn.Linear(2, 1),
                    path=bad_path,
                    contract=contract,
                    label="fixture",
                )

    def test_cached_model_module_from_wrong_path_is_rejected(self):
        fake = types.ModuleType("mesorch")
        fake.__file__ = "/tmp/not-the-pinned-repository/mesorch.py"
        with mock.patch.dict(sys.modules, {"mesorch": fake}):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runner._require_cached_module_origin(
                    "mesorch",
                    Path("/pinned/Mesorch/mesorch.py"),
                )

    def test_installed_dependency_from_shadow_path_is_rejected(self):
        fake = types.ModuleType("timm")
        fake.__file__ = "/tmp/shadow/timm.py"
        fake.__version__ = "1.0.15"
        with mock.patch.dict(sys.modules, {"timm": fake}):
            with self.assertRaisesRegex(
                ValueError,
                "not owned by an allowed installed distribution",
            ):
                runner._installed_module_contract("timm", ("timm",))

    def test_runtime_contract_contains_numeric_dependencies(self):
        contract = runner._runtime_contract("cpu")
        self.assertEqual(
            set(contract["packages"]),
            {
                "torch",
                "torchvision",
                "timm",
                "IMDLBenCo",
                "numpy",
                "Pillow",
                "albumentations",
                "scikit-learn",
                "cv2",
            },
        )
        self.assertIn("IMDLBenCo.registry", contract["critical_submodules"])
        for package in contract["packages"].values():
            self.assertTrue(Path(package["source"]).is_file())
            self.assertTrue(package["distributions"])
        self.assertEqual(
            contract["accelerator"]["requested_device"],
            "cpu",
        )

    def test_runtime_contract_is_inside_immutable_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_manifest = root / "dataset.json"
            inputs_path = root / "inputs.jsonl"
            dataset_manifest.write_text("{}", encoding="utf-8")
            inputs_path.write_text("", encoding="utf-8")
            release = {
                "dataset_id": "fixture",
                "contract_sha256": "a" * 64,
                "inputs_sha256": runner.sha256_file(inputs_path),
                "jpeg": {"quality": 95},
            }
            args = argparse.Namespace(
                run_id="runtime-fixture",
                condition="fixture",
                seed=42,
                device="cpu",
                mask_threshold=0.5,
                bootstrap_samples=10,
            )
            first_runtime = {
                "python": {"version": "3.12.3"},
                "packages": {"torch": {"module_version": "2.8.0"}},
            }
            second_runtime = {
                "python": {"version": "3.12.3"},
                "packages": {"torch": {"module_version": "2.9.0"}},
            }
            common = {
                "args": args,
                "repo_root": root,
                "dataset_manifest_path": dataset_manifest,
                "release": release,
                "inputs_path": inputs_path,
                "selected": [],
                "mesorch_root": root / "upstream",
                "checkpoint_path": root / "mesorch-98.pth",
                "artifact_dir": root / "artifacts",
            }
            with mock.patch.object(
                runner,
                "_runtime_contract",
                return_value=first_runtime,
            ):
                first = runner.build_run_manifest(**common)
            with mock.patch.object(
                runner,
                "_runtime_contract",
                return_value=second_runtime,
            ):
                second = runner.build_run_manifest(**common)
        self.assertEqual(first["runtime_contract"], first_runtime)
        self.assertEqual(second["runtime_contract"], second_runtime)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def _resume_fixture(
        self,
        root: Path,
    ) -> tuple[list[dict], Path, dict[str, dict]]:
        selected = [
            {
                "sample_id": "x",
                "canonical_sha256": "image-sha",
                "task_id": "task",
                "kind": "real",
                "width": 3,
                "height": 2,
            }
        ]
        artifact_dir = root / "artifacts"
        logits_path = artifact_dir / "raw_logits_model_128" / "x.npy"
        model_path = artifact_dir / "score_maps_model_512" / "x.npy"
        native_path = artifact_dir / "score_maps_native" / "x.npy"
        mask_path = artifact_dir / "masks_native" / "x.png"
        runner._atomic_save_npy(
            logits_path,
            np.zeros((128, 128), dtype=np.float32),
        )
        runner._atomic_save_npy(
            model_path,
            np.full((512, 512), 0.5, dtype=np.float32),
        )
        runner._atomic_save_npy(
            native_path,
            np.full((2, 3), 0.5, dtype=np.float32),
        )
        runner._atomic_save_mask(
            mask_path,
            np.zeros((2, 3), dtype=bool),
        )
        row = {
            "status": "ok",
            "run_manifest_fingerprint": "run-sha",
            "image_sha256": "image-sha",
            "task_id": "task",
            "kind": "real",
            "raw_logits_model_path": runner.repo_relative(
                logits_path,
                root,
            ),
            "raw_logits_model_sha256": runner.sha256_file(logits_path),
            "raw_logits_model_shape": [128, 128],
            "raw_logits_model_dtype": "float32",
            "score_map_model_path": runner.repo_relative(model_path, root),
            "score_map_model_sha256": runner.sha256_file(model_path),
            "score_map_model_shape": [512, 512],
            "score_map_model_dtype": "float32",
            "score_map_path": runner.repo_relative(native_path, root),
            "score_map_sha256": runner.sha256_file(native_path),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "mask_path": runner.repo_relative(mask_path, root),
            "mask_sha256": runner.sha256_file(mask_path),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
        }
        return selected, artifact_dir, {"x": row}

    def test_resume_requires_matching_fingerprint_identity_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, artifact_dir, valid = self._resume_fixture(root)
            runner._validate_resume_rows(
                valid,
                selected,
                "run-sha",
                repo_root=root,
                artifact_dir=artifact_dir,
            )
            broken = json.loads(json.dumps(valid))
            broken["x"]["run_manifest_fingerprint"] = "other"
            with self.assertRaisesRegex(
                ValueError,
                "incompatible run fingerprint",
            ):
                runner._validate_resume_rows(
                    broken,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )
            broken = json.loads(json.dumps(valid))
            broken["x"]["image_sha256"] = "other"
            with self.assertRaisesRegex(
                ValueError,
                "incompatible input identity",
            ):
                runner._validate_resume_rows(
                    broken,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )
            broken = json.loads(json.dumps(valid))
            broken["x"]["score_map_path"] = "artifacts/elsewhere/x.npy"
            with self.assertRaisesRegex(
                ValueError,
                "artifact path",
            ):
                runner._validate_resume_rows(
                    broken,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )

    def test_resume_rejects_missing_modified_and_wrong_schema_artifacts(self):
        for corruption in ("missing", "modified", "metadata"):
            with self.subTest(corruption=corruption):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, artifact_dir, latest = self._resume_fixture(root)
                    native_path = (
                        artifact_dir / "score_maps_native" / "x.npy"
                    )
                    if corruption == "missing":
                        native_path.unlink()
                        pattern = "missing native probability artifact"
                    elif corruption == "modified":
                        runner._atomic_save_npy(
                            native_path,
                            np.ones((2, 3), dtype=np.float32),
                        )
                        pattern = "modified native probability artifact"
                    else:
                        latest["x"]["score_map_dtype"] = "float64"
                        pattern = "dtype metadata"
                    with self.assertRaisesRegex(ValueError, pattern):
                        runner._validate_resume_rows(
                            latest,
                            selected,
                            "run-sha",
                            repo_root=root,
                            artifact_dir=artifact_dir,
                        )

    def test_non_ok_resume_row_does_not_require_artifacts(self):
        selected = [
            {
                "sample_id": "x",
                "canonical_sha256": "image-sha",
                "task_id": "task",
                "kind": "real",
                "width": 3,
                "height": 2,
            }
        ]
        failed = {
            "x": {
                "status": "error",
                "run_manifest_fingerprint": "run-sha",
                "image_sha256": "image-sha",
                "task_id": "task",
                "kind": "real",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner._validate_resume_rows(
                failed,
                selected,
                "run-sha",
                repo_root=root,
                artifact_dir=root / "artifacts",
            )

    def test_existing_manifest_must_have_same_immutable_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            runner._write_or_validate_run_manifest(
                path,
                {"fingerprint": "a", "created_at": "one"},
            )
            runner._write_or_validate_run_manifest(
                path,
                {"fingerprint": "a", "created_at": "two"},
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runner._write_or_validate_run_manifest(
                    path,
                    {"fingerprint": "b"},
                )

    def test_infer_one_uses_one_forward_zero_mask_and_one_hook_capture(self):
        model = _TinyMesorch()
        processed, peak, latency = runner.infer_one(
            model,
            torch.device("cpu"),
            np.zeros((3, 512, 512), dtype=np.float32),
            native_width=3,
            native_height=2,
        )
        self.assertEqual(model.calls, 1)
        self.assertIsNotNone(model.seen_dummy_mask)
        assert model.seen_dummy_mask is not None
        self.assertEqual(tuple(model.seen_dummy_mask.shape), (1, 1, 512, 512))
        self.assertEqual(model.seen_dummy_mask.dtype, torch.float32)
        self.assertEqual(int(torch.count_nonzero(model.seen_dummy_mask)), 0)
        self.assertEqual(
            processed["internal_logits_model_128"].shape,
            (128, 128),
        )
        self.assertEqual(processed["probability_model_512"].shape, (512, 512))
        self.assertEqual(processed["probability_native"].shape, (2, 3))
        self.assertEqual(peak, 0)
        self.assertGreaterEqual(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
