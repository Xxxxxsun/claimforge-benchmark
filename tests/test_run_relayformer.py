from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from collections import Counter, OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from eval.opensource import run_relayformer as runner


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


class _TinyRelayFormer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.assemble_calls = 0
        self.seen_origin_shape: torch.Tensor | None = None
        self.seen_clip_len: torch.Tensor | None = None

    def assemble_and_decode(self, image: torch.Tensor) -> torch.Tensor:
        self.assemble_calls += 1
        return image[:, :1] * torch.tensor(2.0, device=image.device)

    def forward(
        self,
        image: torch.Tensor,
        *,
        origin_shape: torch.Tensor,
        clip_len: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        self.seen_origin_shape = origin_shape.detach().clone()
        self.seen_clip_len = clip_len.detach().clone()
        return torch.sigmoid(self.assemble_and_decode(image))


class _TinyCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.reg_token = torch.nn.Parameter(
            torch.zeros((1, 2, 768), dtype=torch.float32)
        )
        self.projection = torch.nn.Linear(2, 1)


def _checkpoint_contract(state: MappingLike) -> dict:
    return {
        "top_level_keys": [
            "model",
            "optimizer",
            "epoch",
            "scaler",
            "args",
        ],
        "epoch": 164,
        "state_keys": len(state),
        "state_elements": sum(int(value.numel()) for value in state.values()),
        "tensor_bytes": sum(
            int(value.numel()) * int(value.element_size())
            for value in state.values()
        ),
        "state_dtypes": dict(
            Counter(str(value.dtype) for value in state.values())
        ),
    }


MappingLike = dict[str, torch.Tensor] | OrderedDict[str, torch.Tensor]


class RunRelayFormerTest(unittest.TestCase):
    def test_frozen_constants_match_official_release(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "3fc863c7691d93fb5b11ca8e12e3a214d771e384",
        )
        self.assertEqual(runner.CHECKPOINT["epoch"], 164)
        self.assertEqual(runner.CHECKPOINT["bytes"], 1_102_625_388)
        self.assertEqual(
            runner.CHECKPOINT["sha256"],
            "00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090",
        )
        self.assertEqual(runner.CHECKPOINT["state_keys"], 410)
        self.assertEqual(runner.CHECKPOINT["parameters"], 91_909_179)
        self.assertEqual(runner.CHECKPOINT["buffers"], 786_435)

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

    def test_preprocess_is_exact_thumbnail_top_left_pad_and_float32_imagenet(self):
        rgb = np.asarray(
            [
                [[255, 0, 1], [0, 255, 2], [0, 0, 255]],
                [[12, 34, 56], [127, 128, 129], [253, 254, 255]],
            ],
            dtype=np.uint8,
        )
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        canvas[:2, :3] = rgb
        expected = canvas.astype(np.float32) / np.float32(255.0)
        expected = (
            expected
            - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        expected = np.ascontiguousarray(expected.transpose(2, 0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            tensor, native_size, resized_size, metadata = (
                runner.preprocess_image(path)
            )

        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tensor.shape, (3, 1024, 1024))
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(resized_size, (3, 2))
        self.assertEqual(metadata["protocol"], runner.PREPROCESS_PROTOCOL)
        self.assertEqual(metadata["decoder_format"], "PNG")
        self.assertEqual(metadata["native_size_wh"], [3, 2])
        self.assertEqual(metadata["resized_size_wh"], [3, 2])
        self.assertEqual(metadata["padding_ltrb"], [0, 0, 1021, 1022])
        self.assertEqual(
            metadata["resize_interpolation"],
            "Pillow.Image.Resampling.BILINEAR_reducing_gap_None",
        )
        self.assertEqual(metadata["tensor_sha256"], runner._sha256_array(tensor))

    def test_preprocess_thumbnail_rounding_is_pillow_11_1_downscale_only(self):
        rgb = np.arange(2051 * 1003 * 3, dtype=np.uint32)
        rgb = np.asarray((rgb % 256).reshape(1003, 2051, 3), dtype=np.uint8)
        expected_image = Image.fromarray(rgb, mode="RGB")
        expected_image.thumbnail(
            (1024, 1024),
            resample=Image.Resampling.BILINEAR,
            reducing_gap=None,
        )
        expected_size = expected_image.size
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            _, native_size, resized_size, metadata = runner.preprocess_image(path)
        self.assertEqual(native_size, (2051, 1003))
        self.assertEqual(resized_size, expected_size)
        self.assertEqual(metadata["scale"], 1024 / 2051)
        self.assertEqual(
            metadata["resize_rounding"],
            "pillow_thumbnail_floor_ceil_min_aspect_error",
        )

    def test_first_canonical_tensor_has_frozen_protocol_hash(self):
        path = Path(
            "outputs/opensource/mouse_canonical_v1/images/"
            "0cca3f606edf434904f7bea5.jpg"
        )
        if not path.is_file():
            self.skipTest("canonical release is not present")
        tensor, native_size, resized_size, metadata = runner.preprocess_image(path)
        self.assertEqual(native_size, (1800, 1350))
        self.assertEqual(resized_size, (1024, 768))
        self.assertEqual(
            runner._sha256_array(tensor),
            "7a32d4419d732be17259e5249c91d4281a4821baa895b9f6209e28c26e7fd7e4",
        )
        self.assertEqual(
            metadata["tensor_sha256"],
            "7a32d4419d732be17259e5249c91d4281a4821baa895b9f6209e28c26e7fd7e4",
        )

    def test_strict_probability_threshold_treats_exact_half_as_negative(self):
        above = np.nextafter(
            np.float32(0.5),
            np.float32(1.0),
            dtype=np.float32,
        )
        scores = np.asarray([[0.5, above]], dtype=np.float32)
        target = np.asarray([[False, True]], dtype=bool)
        metrics = runner.binary_pixel_metrics_strict(
            scores,
            target,
            threshold=0.5,
            include_ap=True,
        )
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["predicted_positive_pixels"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)

    def test_postprocess_replays_sigmoid_and_restores_logits_and_probability_separately(self):
        base = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]]]],
            dtype=torch.float32,
        )
        valid_logits = F.interpolate(
            base,
            size=(320, 512),
            mode="bilinear",
            align_corners=False,
        )
        logits = torch.full((1, 1, 1024, 1024), -10.0)
        logits[:, :, :320, :512] = valid_logits
        probability = torch.sigmoid(logits)
        expected_native_logits = F.interpolate(
            valid_logits,
            size=(640, 1024),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        expected_native_probability = F.interpolate(
            probability[:, :, :320, :512],
            size=(640, 1024),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        processed = runner.postprocess_outputs(
            probability,
            logits,
            native_width=1024,
            native_height=640,
            resized_width=512,
            resized_height=320,
        )
        np.testing.assert_array_equal(
            processed["raw_logits_model_1024"],
            logits[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_model_1024"],
            probability[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["raw_logits_native"],
            expected_native_logits,
        )
        np.testing.assert_array_equal(
            processed["probability_native"],
            expected_native_probability,
        )
        self.assertFalse(
            np.array_equal(
                processed["probability_native"],
                torch.sigmoid(
                    torch.from_numpy(processed["raw_logits_native"])
                ).numpy(),
            )
        )

    def test_postprocess_rejects_wrong_shape_and_inconsistent_probability(self):
        logits = torch.zeros((1, 1, 1024, 1024), dtype=torch.float32)
        probability = torch.full_like(logits, 0.5)
        with self.assertRaisesRegex(ValueError, "probability shape"):
            runner.postprocess_outputs(
                probability[:, :, :-1],
                logits,
                native_width=2,
                native_height=2,
                resized_width=2,
                resized_height=2,
            )
        with self.assertRaisesRegex(ValueError, "does not match sigmoid"):
            runner.postprocess_outputs(
                probability + 0.1,
                logits,
                native_width=2,
                native_height=2,
                resized_width=2,
                resized_height=2,
            )

    def test_infer_one_wraps_plain_method_once_and_restores_it(self):
        model = _TinyRelayFormer()
        original_function = model.assemble_and_decode.__func__
        image = np.zeros((3, 1024, 1024), dtype=np.float32)
        image[0, :2, :3] = np.asarray(
            [[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]],
            dtype=np.float32,
        )
        processed, peak, latency = runner.infer_one(
            model,
            torch.device("cpu"),
            image,
            native_width=3,
            native_height=2,
            resized_width=3,
            resized_height=2,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.assemble_calls, 1)
        self.assertIs(model.assemble_and_decode.__func__, original_function)
        assert model.seen_origin_shape is not None
        assert model.seen_clip_len is not None
        torch.testing.assert_close(
            model.seen_origin_shape,
            torch.tensor([[2, 3]], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model.seen_clip_len,
            torch.tensor([1], dtype=torch.int64),
        )
        self.assertEqual(processed["raw_logits_model_1024"].shape, (1024, 1024))
        self.assertEqual(processed["raw_logits_native"].shape, (2, 3))
        self.assertEqual(processed["probability_native"].shape, (2, 3))
        self.assertEqual(peak, 0)
        self.assertGreaterEqual(latency, 0.0)

    def test_model_space_target_uses_pillow_nearest_without_padding(self):
        target = np.asarray(
            [[True, False], [False, False]],
            dtype=bool,
        )
        resized = runner.model_space_target(
            target,
            resized_width=5,
            resized_height=3,
        )
        expected = np.asarray(
            Image.fromarray(target.astype(np.uint8), mode="L").resize(
                (5, 3),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
        np.testing.assert_array_equal(resized, expected)
        self.assertEqual(resized.shape, (3, 5))
        self.assertEqual(resized.dtype, np.bool_)

    def test_artifact_writers_are_lossless_and_strictly_thresholded(self):
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
        np.testing.assert_array_equal(
            mask,
            np.asarray([[0, 255]], dtype=np.uint8),
        )

    def test_checkpoint_loader_safe_allowlists_namespace_and_validates_release_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            good_path = Path(temporary) / "good.pth"
            bad_args_path = Path(temporary) / "bad-args.pth"
            bad_reg_path = Path(temporary) / "bad-reg.pth"
            source = _TinyCheckpointModel()
            state = source.state_dict()
            args = argparse.Namespace(
                model="RelayFormer",
                image_size=1024,
                if_padding=True,
                if_resizing=False,
            )
            payload = {
                "model": state,
                "optimizer": {},
                "epoch": 164,
                "scaler": None,
                "args": args,
            }
            torch.save(payload, good_path)
            bad_args = dict(payload)
            bad_args["args"] = argparse.Namespace(
                model="RelayFormer",
                image_size=512,
                if_padding=True,
                if_resizing=False,
            )
            torch.save(bad_args, bad_args_path)
            bad_reg_state = OrderedDict(state)
            bad_reg_state["vit.reg_token"] = torch.zeros(
                (1, 1, 768),
                dtype=torch.float32,
            )
            bad_reg = dict(payload)
            bad_reg["model"] = bad_reg_state
            torch.save(bad_reg, bad_reg_path)

            contract = _checkpoint_contract(state)
            loaded_payload = runner._load_checkpoint_payload(
                path=good_path,
                contract=contract,
                label="fixture",
            )
            target = _TinyCheckpointModel()
            runner._strict_load_checkpoint_state(
                module=target,
                payload=loaded_payload,
                label="fixture",
            )
            for expected, actual in zip(
                source.parameters(),
                target.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
            with self.assertRaisesRegex(ValueError, "args.image_size"):
                runner._load_checkpoint_payload(
                    path=bad_args_path,
                    contract=contract,
                    label="fixture",
                )
            bad_reg_contract = _checkpoint_contract(bad_reg_state)
            with self.assertRaisesRegex(ValueError, "reg_token shape"):
                runner._load_checkpoint_payload(
                    path=bad_reg_path,
                    contract=bad_reg_contract,
                    label="fixture",
                )

    def test_cached_model_module_from_wrong_path_is_rejected(self):
        fake = types.ModuleType("models.RelayFormer")
        fake.__file__ = "/tmp/not-the-pinned-repository/RelayFormer.py"
        with mock.patch.dict(sys.modules, {"models.RelayFormer": fake}):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runner._require_cached_module_origin(
                    "models.RelayFormer",
                    Path("/pinned/RelayFormer/models/RelayFormer.py"),
                )

    def test_runtime_contract_contains_relayformer_dependencies(self):
        contract = runner._runtime_contract("cpu")
        self.assertEqual(
            set(contract["packages"]),
            {
                "torch",
                "torchvision",
                "timm",
                "IMDLBenCo",
                "rotary-embedding-torch",
                "numpy",
                "Pillow",
                "albumentations",
                "cv2",
                "scikit-learn",
            },
        )
        self.assertIn("IMDLBenCo.registry", contract["critical_submodules"])
        self.assertIn("timm.layers", contract["critical_submodules"])
        self.assertEqual(
            contract["packages"]["Pillow"]["distributions"][0]["version"],
            "11.1.0",
        )
        self.assertEqual(contract["accelerator"]["requested_device"], "cpu")

    def _manifest_fixture(
        self,
        root: Path,
        runtime: dict,
    ) -> dict:
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
        with mock.patch.object(
            runner,
            "_runtime_contract",
            return_value=runtime,
        ):
            return runner.build_run_manifest(
                args=args,
                repo_root=root,
                dataset_manifest_path=dataset_manifest,
                release=release,
                inputs_path=inputs_path,
                selected=[],
                relayformer_root=root / "upstream",
                checkpoint_path=root / "checkpoint-164.pth",
                artifact_dir=root / "artifacts",
            )

    def test_manifest_freezes_t2_only_contract_and_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._manifest_fixture(
                root,
                {
                    "python": {"version": "3.12.3"},
                    "packages": {"torch": {"module_version": "2.8.0"}},
                },
            )
            second = self._manifest_fixture(
                root,
                {
                    "python": {"version": "3.12.3"},
                    "packages": {"torch": {"module_version": "2.9.0"}},
                },
            )
        self.assertFalse(first["model"]["supports_image_level_t1"])
        self.assertIsNone(first["model"]["image_score_source"])
        self.assertTrue(first["model"]["supports_pixel_level_t2"])
        self.assertEqual(
            first["metrics"]["t1_policy"],
            "unsupported_no_derived_image_score",
        )
        self.assertEqual(
            first["artifacts"]["raw_logits_native"]["restore"],
            runner.NATIVE_RESTORE,
        )
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_existing_manifest_rejects_immutable_drift(self):
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
        paths = {
            "raw_logits_model": (
                artifact_dir / "raw_logits_model_1024" / "x.npy"
            ),
            "raw_logits_native": artifact_dir / "raw_logits_native" / "x.npy",
            "score_map_model": (
                artifact_dir / "score_maps_model_1024" / "x.npy"
            ),
            "score_map": artifact_dir / "score_maps_native" / "x.npy",
            "mask": artifact_dir / "masks_native" / "x.png",
        }
        runner._atomic_save_npy(
            paths["raw_logits_model"],
            np.zeros((1024, 1024), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["raw_logits_native"],
            np.zeros((2, 3), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["score_map_model"],
            np.full((1024, 1024), 0.5, dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["score_map"],
            np.full((2, 3), 0.5, dtype=np.float32),
        )
        runner._atomic_save_mask(
            paths["mask"],
            np.zeros((2, 3), dtype=bool),
        )
        row = {
            "status": "ok",
            "run_manifest_fingerprint": "run-sha",
            "image_sha256": "image-sha",
            "task_id": "task",
            "kind": "real",
            "valid_for_t1": False,
            "t1_policy": "unsupported_no_derived_image_score",
            "raw_logits_model_path": runner.repo_relative(
                paths["raw_logits_model"],
                root,
            ),
            "raw_logits_model_sha256": runner.sha256_file(
                paths["raw_logits_model"]
            ),
            "raw_logits_model_shape": [1024, 1024],
            "raw_logits_model_dtype": "float32",
            "raw_logits_native_path": runner.repo_relative(
                paths["raw_logits_native"],
                root,
            ),
            "raw_logits_native_sha256": runner.sha256_file(
                paths["raw_logits_native"]
            ),
            "raw_logits_native_shape": [2, 3],
            "raw_logits_native_dtype": "float32",
            "score_map_model_path": runner.repo_relative(
                paths["score_map_model"],
                root,
            ),
            "score_map_model_sha256": runner.sha256_file(
                paths["score_map_model"]
            ),
            "score_map_model_shape": [1024, 1024],
            "score_map_model_dtype": "float32",
            "score_map_path": runner.repo_relative(paths["score_map"], root),
            "score_map_sha256": runner.sha256_file(paths["score_map"]),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "mask_path": runner.repo_relative(paths["mask"], root),
            "mask_sha256": runner.sha256_file(paths["mask"]),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
        }
        return selected, artifact_dir, {"x": row}

    def test_resume_validates_artifacts_and_forbids_t1_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, artifact_dir, latest = self._resume_fixture(root)
            runner._validate_resume_rows(
                latest,
                selected,
                "run-sha",
                repo_root=root,
                artifact_dir=artifact_dir,
            )
            with_t1 = json.loads(json.dumps(latest))
            with_t1["x"]["score"] = None
            with self.assertRaisesRegex(ValueError, "forbidden T1 fields"):
                runner._validate_resume_rows(
                    with_t1,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )
            native_path = artifact_dir / "score_maps_native" / "x.npy"
            runner._atomic_save_npy(
                native_path,
                np.ones((2, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "modified native probability"):
                runner._validate_resume_rows(
                    latest,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )

    def test_non_ok_resume_row_needs_t2_identity_but_no_artifacts(self):
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
                "valid_for_t1": False,
                "t1_policy": "unsupported_no_derived_image_score",
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


if __name__ == "__main__":
    unittest.main()
