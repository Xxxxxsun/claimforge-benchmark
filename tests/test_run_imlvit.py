import argparse
import hashlib
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

from eval.opensource.common import sha256_file, stable_json
from eval.opensource import run_imlvit as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


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
                    "canonical_path": f"images/{kind}-{pair_rank}.jpg",
                    "canonical_sha256": f"{rank + 1:064x}",
                    "gt_mask_sha256": (
                        f"{rank + 101:064x}" if kind == "forged" else None
                    ),
                }
            )
    return rows


def _normalized_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (
        rgb.astype(np.float32) / np.float32(255.0) - mean
    ) / std


class _TinyIMLViT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer(
            "counter",
            torch.zeros((), dtype=torch.int64),
        )


class RunIMLViTTest(unittest.TestCase):
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

    def test_preprocess_large_image_downscales_long_side_and_raw_zero_pads(self):
        # The 4:1 shape makes the paper geometry exact: 2048x512 -> 1024x256.
        rgb = np.empty((512, 2048, 3), dtype=np.uint8)
        rgb[...] = np.asarray([255, 128, 1], dtype=np.uint8)
        expected_content = _normalized_rgb(
            np.asarray([255, 128, 1], dtype=np.uint8)
        )
        expected_raw_zero = _normalized_rgb(
            np.asarray([0, 0, 0], dtype=np.uint8)
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            tensor, native_size, resized_size, metadata = (
                runner.preprocess_image(path)
            )

        self.assertEqual(native_size, (2048, 512))
        self.assertEqual(resized_size, (1024, 256))
        self.assertEqual(
            tensor.shape,
            (3, runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
        )
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[:, 0, 0],
            expected_content,
            rtol=0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            tensor[:, 255, 1023],
            expected_content,
            rtol=0,
            atol=2e-7,
        )
        # If padding happened after normalization this region would be zero.
        np.testing.assert_allclose(
            tensor[:, 256, 0],
            expected_raw_zero,
            rtol=0,
            atol=2e-7,
        )
        self.assertGreater(float(np.max(np.abs(tensor[:, 256, 0]))), 1.0)

        self.assertEqual(metadata["decoder"], "Pillow.Image.open.convert_RGB")
        self.assertEqual(metadata["decoder_format"], "PNG")
        self.assertEqual(metadata["channel_order"], "RGB")
        self.assertEqual(metadata["native_size"], [2048, 512])
        self.assertEqual(metadata["resized_content_size"], [1024, 256])
        self.assertEqual(metadata["model_canvas_size"], [1024, 1024])
        self.assertEqual(
            metadata["resize_policy"],
            "albumentations_longest_max_size_downscale_only",
        )
        self.assertEqual(
            metadata["resize_interpolation"],
            "cv2.INTER_LINEAR_via_albumentations",
        )
        self.assertEqual(metadata["resize_scale_x"], 0.5)
        self.assertEqual(metadata["resize_scale_y"], 0.5)
        self.assertEqual(
            metadata["padding"],
            {
                "placement": "top_left",
                "right_pixels": 0,
                "bottom_pixels": 768,
                "raw_rgb_value": 0,
                "applied_before_normalization": True,
            },
        )
        self.assertIsNone(metadata["input_crop"])
        self.assertFalse(metadata["input_reencode"])
        self.assertEqual(metadata["tensor_shape"], [3, 1024, 1024])
        self.assertEqual(metadata["tensor_dtype"], "float32")
        self.assertEqual(
            metadata["tensor_sha256"],
            runner._sha256_array(tensor),
        )

    def test_preprocess_small_image_is_not_upscaled_and_is_top_left_padded(self):
        import albumentations as albu

        rgb = np.asarray(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[1, 2, 3], [127, 128, 129], [253, 254, 255]],
            ],
            dtype=np.uint8,
        )
        expected_content = _normalized_rgb(rgb).transpose(2, 0, 1)
        expected_raw_zero = _normalized_rgb(
            np.asarray([0, 0, 0], dtype=np.uint8)
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            with mock.patch.object(
                albu,
                "LongestMaxSize",
                side_effect=AssertionError("small images must not be resized"),
            ) as resize:
                tensor, native_size, resized_size, metadata = (
                    runner.preprocess_image(path)
                )

        resize.assert_not_called()
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(resized_size, (3, 2))
        np.testing.assert_allclose(
            tensor[:, :2, :3],
            expected_content,
            rtol=0,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            tensor[:, 0, 3],
            expected_raw_zero,
            rtol=0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            tensor[:, 2, 0],
            expected_raw_zero,
            rtol=0,
            atol=2e-7,
        )
        self.assertEqual(
            metadata["resize_policy"],
            "none_image_within_1024_limit",
        )
        self.assertEqual(metadata["resize_scale_x"], 1.0)
        self.assertEqual(metadata["resize_scale_y"], 1.0)
        self.assertEqual(
            metadata["padding"],
            {
                "placement": "top_left",
                "right_pixels": 1021,
                "bottom_pixels": 1022,
                "raw_rgb_value": 0,
                "applied_before_normalization": True,
            },
        )
        self.assertEqual(
            metadata["normalization"],
            {
                "scale": "uint8_divide_255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )

    def test_postprocess_upsamples_logits_then_sigmoids_once_and_restores(self):
        logits = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]]]],
            dtype=torch.float32,
        )
        resized_width, resized_height = 800, 600
        native_width, native_height = 9, 5
        expected_logits = F.interpolate(
            logits,
            size=(runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        expected_model_score = torch.sigmoid(expected_logits)
        expected_native = F.interpolate(
            expected_model_score[
                :, :, :resized_height, :resized_width
            ],
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        sigmoid_first = F.interpolate(
            torch.sigmoid(logits),
            size=(runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        wrong_order_native = F.interpolate(
            sigmoid_first[:, :, :resized_height, :resized_width],
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        wrong_padding_native = F.interpolate(
            expected_model_score,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        with (
            mock.patch.object(
                torch,
                "sigmoid",
                wraps=torch.sigmoid,
            ) as sigmoid,
            mock.patch.object(
                F,
                "interpolate",
                wraps=F.interpolate,
            ) as interpolate,
        ):
            model_logits, model_score, native_score = (
                runner.postprocess_head_logits(
                    logits,
                    native_width=native_width,
                    native_height=native_height,
                    resized_width=resized_width,
                    resized_height=resized_height,
                )
            )

        sigmoid.assert_called_once()
        self.assertEqual(interpolate.call_count, 2)
        self.assertEqual(
            interpolate.call_args_list[0].kwargs,
            {
                "size": (1024, 1024),
                "mode": "bilinear",
                "align_corners": False,
            },
        )
        self.assertEqual(
            interpolate.call_args_list[1].kwargs,
            {
                "size": (native_height, native_width),
                "mode": "bilinear",
                "align_corners": False,
            },
        )
        self.assertEqual(model_logits.shape, (1024, 1024))
        self.assertEqual(model_score.shape, (1024, 1024))
        self.assertEqual(native_score.shape, (native_height, native_width))
        for output in (model_logits, model_score, native_score):
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(output.flags.c_contiguous)
        np.testing.assert_allclose(
            model_logits,
            expected_logits[0, 0].numpy(),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            model_score,
            expected_model_score[0, 0].numpy(),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            native_score,
            expected_native,
            rtol=0,
            atol=1e-7,
        )
        self.assertGreater(
            float(np.max(np.abs(native_score - wrong_order_native))),
            0.05,
        )
        self.assertGreater(
            float(np.max(np.abs(native_score - wrong_padding_native))),
            0.05,
        )

    def test_postprocess_validates_shapes_sizes_and_finite_outputs(self):
        valid = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        malformed = (
            torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            torch.zeros((1, 1, 2), dtype=torch.float32),
            np.zeros((1, 1, 2, 2), dtype=np.float32),
        )
        for value in malformed:
            with self.subTest(shape=getattr(value, "shape", None)):
                with self.assertRaisesRegex(
                    ValueError,
                    "unexpected IML-ViT head-logit shape",
                ):
                    runner.postprocess_head_logits(
                        value,
                        native_width=4,
                        native_height=3,
                        resized_width=4,
                        resized_height=3,
                    )
        for keyword, value, message in (
            ("resized_width", 0, "invalid resized width"),
            ("resized_height", 1025, "invalid resized height"),
            ("native_width", 0, "invalid native size"),
        ):
            sizes = {
                "native_width": 4,
                "native_height": 3,
                "resized_width": 4,
                "resized_height": 3,
            }
            sizes[keyword] = value
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, message):
                    runner.postprocess_head_logits(valid, **sizes)
        with self.assertRaisesRegex(ValueError, "logits contain non-finite"):
            runner.postprocess_head_logits(
                torch.full((1, 1, 2, 2), float("nan")),
                native_width=4,
                native_height=3,
                resized_width=4,
                resized_height=3,
            )

    def test_model_space_target_uses_nearest_neighbor_without_softening(self):
        import cv2

        target = np.asarray([[0, 1], [1, 0]], dtype=bool)
        original = target.copy()
        expected = np.asarray(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=bool,
        )

        with mock.patch.object(
            cv2,
            "resize",
            wraps=cv2.resize,
        ) as resize:
            restored = runner.model_space_target(
                target,
                resized_width=4,
                resized_height=4,
            )

        resize.assert_called_once()
        self.assertEqual(resize.call_args.args[1], (4, 4))
        self.assertEqual(
            resize.call_args.kwargs["interpolation"],
            cv2.INTER_NEAREST,
        )
        np.testing.assert_array_equal(target, original)
        np.testing.assert_array_equal(restored, expected)
        self.assertEqual(restored.dtype, np.bool_)

    def _mock_checkpoint_contract(
        self,
        checkpoint_bytes: int,
    ) -> dict:
        return {
            **runner.CHECKPOINT,
            "bytes": checkpoint_bytes,
            "sha256": "f" * 64,
            "state_keys": 2,
            "tensor_values": 2,
            "state_elements": 3,
            "tensor_bytes": 16,
            "state_dtypes": {"torch.float32": 1, "torch.int64": 1},
            "parameters": 2,
            "buffers": 1,
        }

    def test_load_model_enforces_source_identity_and_strict_raw_state_dict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "IML-ViT"
            root.mkdir()
            checkpoint_path = Path(temporary) / "checkpoint.pth"
            checkpoint_path.write_bytes(b"fixture")
            contract = self._mock_checkpoint_contract(
                checkpoint_path.stat().st_size
            )
            model = _TinyIMLViT()
            state = OrderedDict(
                (
                    ("weight", torch.ones(2, dtype=torch.float32)),
                    ("counter", torch.ones((), dtype=torch.int64)),
                )
            )
            fake_module = types.ModuleType("iml_vit_model")
            factory = mock.Mock(return_value=model)
            fake_module.iml_vit_model = factory

            def git_value(_root: Path, *args: str) -> str:
                if args == ("rev-parse", "HEAD"):
                    return runner.MODEL_SOURCE_COMMIT
                if args == (
                    "status",
                    "--short",
                    "--untracked-files=no",
                ):
                    return ""
                raise AssertionError(f"unexpected git query: {args}")

            with (
                mock.patch.object(runner, "CHECKPOINT", contract),
                mock.patch.object(
                    runner,
                    "SOURCE_FILES",
                    {"model.py": "a" * 64, "LICENSE": "b" * 64},
                ),
                mock.patch.object(
                    runner,
                    "_git_value",
                    side_effect=git_value,
                ),
                mock.patch.object(
                    runner,
                    "_verify_runtime_file",
                ) as verify,
                mock.patch.object(
                    torch,
                    "load",
                    return_value=state,
                ) as torch_load,
                mock.patch.object(
                    model,
                    "load_state_dict",
                    wraps=model.load_state_dict,
                ) as strict_load,
                mock.patch.dict(
                    sys.modules,
                    {"iml_vit_model": fake_module},
                ),
            ):
                loaded, device = runner.load_model(
                    imlvit_root=root,
                    checkpoint_path=checkpoint_path,
                    device_name="cpu",
                )

        self.assertIs(loaded, model)
        self.assertEqual(device, torch.device("cpu"))
        self.assertFalse(model.training)
        factory.assert_called_once_with(vit_pretrain_path=None)
        torch_load.assert_called_once_with(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        strict_load.assert_called_once_with(state, strict=True)
        self.assertEqual(
            verify.call_args_list,
            [
                mock.call(root / "model.py", "a" * 64, "IML-ViT source file model.py"),
                mock.call(root / "LICENSE", "b" * 64, "IML-ViT source file LICENSE"),
                mock.call(
                    checkpoint_path,
                    "f" * 64,
                    "IML-ViT CAT-protocol checkpoint",
                ),
            ],
        )

    def test_load_model_rejects_source_and_checkpoint_schema_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "IML-ViT"
            root.mkdir()
            checkpoint_path = Path(temporary) / "checkpoint.pth"
            checkpoint_path.write_bytes(b"fixture")
            contract = self._mock_checkpoint_contract(
                checkpoint_path.stat().st_size
            )
            model = _TinyIMLViT()
            fake_module = types.ModuleType("iml_vit_model")
            fake_module.iml_vit_model = mock.Mock(return_value=model)
            valid_state = OrderedDict(
                (
                    ("weight", torch.ones(2, dtype=torch.float32)),
                    ("counter", torch.ones((), dtype=torch.int64)),
                )
            )

            with (
                mock.patch.object(runner, "CHECKPOINT", contract),
                mock.patch.object(runner, "_git_value", return_value="wrong"),
            ):
                with self.assertRaisesRegex(ValueError, "source commit mismatch"):
                    runner.load_model(
                        imlvit_root=root,
                        checkpoint_path=checkpoint_path,
                        device_name="cpu",
                    )

            def dirty_git(_root: Path, *args: str) -> str:
                return (
                    runner.MODEL_SOURCE_COMMIT
                    if args == ("rev-parse", "HEAD")
                    else " M iml_vit_model.py"
                )

            with (
                mock.patch.object(runner, "CHECKPOINT", contract),
                mock.patch.object(
                    runner,
                    "_git_value",
                    side_effect=dirty_git,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "local modifications"):
                    runner.load_model(
                        imlvit_root=root,
                        checkpoint_path=checkpoint_path,
                        device_name="cpu",
                    )

            def clean_git(_root: Path, *args: str) -> str:
                return (
                    runner.MODEL_SOURCE_COMMIT
                    if args == ("rev-parse", "HEAD")
                    else ""
                )

            for bad_state, message in (
                (dict(valid_state), "not the registered raw OrderedDict"),
                (
                    OrderedDict(
                        (
                            ("weight", torch.ones(2, dtype=torch.float64)),
                            ("counter", torch.ones((), dtype=torch.int64)),
                        )
                    ),
                    "tensor-byte count mismatch",
                ),
                (
                    OrderedDict(
                        (("weight", torch.ones(2)), ("counter", "not tensor"))
                    ),
                    "non-tensor state value",
                ),
            ):
                with self.subTest(message=message):
                    with (
                        mock.patch.object(runner, "CHECKPOINT", contract),
                        mock.patch.object(runner, "SOURCE_FILES", {}),
                        mock.patch.object(
                            runner,
                            "_git_value",
                            side_effect=clean_git,
                        ),
                        mock.patch.object(
                            runner,
                            "_verify_runtime_file",
                        ),
                        mock.patch.object(
                            torch,
                            "load",
                            return_value=bad_state,
                        ),
                        mock.patch.dict(
                            sys.modules,
                            {"iml_vit_model": fake_module},
                        ),
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            runner.load_model(
                                imlvit_root=root,
                                checkpoint_path=checkpoint_path,
                                device_name="cpu",
                            )

    def test_manifest_is_t2_only_and_fingerprint_covers_immutable_contract(self):
        selected = runner.select_inputs(_rows(), pair_limit=1)
        args = argparse.Namespace(
            run_id="imlvit_unit_test",
            condition="mouse_canonical_v1",
            seed=42,
            device="cpu",
            mask_threshold=runner.MASK_THRESHOLD,
            bootstrap_samples=1000,
        )
        release = {
            "dataset_id": "mouse_canonical_v1_fixture",
            "contract_sha256": "a" * 64,
            "inputs_sha256": "b" * 64,
            "jpeg": {"quality": 95, "subsampling": 0},
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_manifest = root / "manifest.json"
            dataset_manifest.write_text(
                '{"fixture":"canonical release"}\n',
                encoding="utf-8",
            )
            inputs_path = root / "inputs.jsonl"
            inputs_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in _rows()
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(runner, "_git_value", return_value=""),
                mock.patch.object(
                    runner,
                    "load_model",
                    side_effect=AssertionError(
                        "manifest construction must not load the model"
                    ),
                ) as load_model,
                mock.patch.object(
                    torch.cuda,
                    "get_device_name",
                    side_effect=AssertionError(
                        "CPU manifest must not query a CUDA device"
                    ),
                ) as get_device_name,
            ):
                manifest = runner.build_run_manifest(
                    args=args,
                    repo_root=REPO_ROOT,
                    dataset_manifest_path=dataset_manifest,
                    release=release,
                    inputs_path=inputs_path,
                    selected=selected,
                    imlvit_root=root / "IML-ViT",
                    checkpoint_path=root / "checkpoint.pth",
                    artifact_dir=root / "artifacts",
                )

        load_model.assert_not_called()
        get_device_name.assert_not_called()
        immutable = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "fingerprint",
                "created_at",
                "adapter",
                "environment",
            }
        }
        expected_fingerprint = hashlib.sha256(
            stable_json(immutable).encode("utf-8")
        ).hexdigest()
        self.assertEqual(manifest["fingerprint"], expected_fingerprint)
        self.assertEqual(
            manifest["fingerprint"],
            runner._manifest_fingerprint(immutable),
        )
        self.assertNotEqual(
            manifest["fingerprint"],
            runner._manifest_fingerprint(
                {**immutable, "condition": "changed"}
            ),
        )
        self.assertEqual(
            manifest["input"]["selection_sha256"],
            hashlib.sha256(
                stable_json(runner._selection_contract(selected)).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        self.assertEqual(manifest["expected_pairs"], 1)
        self.assertEqual(manifest["expected_images"], 2)

        model = manifest["model"]
        inference = manifest["inference"]
        metrics = manifest["metrics"]
        self.assertFalse(model["supports_image_level_t1"])
        self.assertIsNone(model["image_score_source"])
        self.assertTrue(model["supports_pixel_level_t2"])
        self.assertEqual(metrics["task"], "T2_pixel_localization_only")
        self.assertEqual(
            metrics["t1_policy"],
            "unsupported_no_derived_image_score",
        )
        self.assertNotIn("classification_threshold", inference)
        self.assertNotIn("classification_threshold", metrics)
        self.assertNotIn("image_decision", inference)
        self.assertNotIn("image_score_aggregation", inference)
        self.assertEqual(
            inference["model_probability"],
            "single_sigmoid_after_logit_restore",
        )
        self.assertEqual(
            inference["native_restore"],
            (
                "crop_right_bottom_padding_then_bilinear_probability_to_"
                "native_align_corners_false"
            ),
        )
        self.assertEqual(
            inference["mask_threshold_comparison"],
            "strict_greater_than",
        )
        self.assertEqual(metrics["localization_spaces"], ["model_1024", "native"])
        self.assertEqual(metrics["model_space_gt_resize"], "cv2_INTER_NEAREST")
        self.assertTrue(model["checkpoint"]["strict_load"])
        self.assertTrue(model["checkpoint"]["safe_weights_only_load"])
        self.assertFalse(model["checkpoint"]["schema_fallbacks"])
        self.assertFalse(model["checkpoint"]["prefix_rewrites"])

    def test_existing_manifest_rejects_fingerprint_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            first = {
                "run_id": "imlvit-test",
                "fingerprint": "a" * 64,
            }
            runner._write_or_validate_run_manifest(path, first)
            runner._write_or_validate_run_manifest(
                path,
                {**first, "created_at": "ignored-on-resume"},
            )
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                first,
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runner._write_or_validate_run_manifest(
                    path,
                    {**first, "fingerprint": "b" * 64},
                )

    @unittest.skipUnless(
        runner.DEFAULT_CHECKPOINT.is_file(),
        "official IML-ViT checkpoint is not installed",
    )
    def test_installed_official_checkpoint_matches_registered_safe_schema(self):
        checkpoint_path = runner.DEFAULT_CHECKPOINT
        contract = runner.CHECKPOINT
        self.assertEqual(checkpoint_path.stat().st_size, contract["bytes"])
        self.assertEqual(sha256_file(checkpoint_path), contract["sha256"])

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.assertIs(type(state), OrderedDict)
        self.assertEqual(len(state), contract["state_keys"])
        self.assertTrue(
            all(isinstance(value, torch.Tensor) for value in state.values())
        )
        self.assertEqual(
            sum(int(value.numel()) for value in state.values()),
            contract["state_elements"],
        )
        self.assertEqual(
            sum(
                int(value.numel()) * int(value.element_size())
                for value in state.values()
            ),
            contract["tensor_bytes"],
        )
        self.assertEqual(
            dict(Counter(str(value.dtype) for value in state.values())),
            contract["state_dtypes"],
        )
        self.assertEqual(
            tuple(state["encoder_net.patch_embed.proj.weight"].shape),
            (768, 3, 16, 16),
        )
        self.assertEqual(
            tuple(state["predict_head.linear_predict.weight"].shape),
            (1, 256, 1, 1),
        )
        self.assertEqual(
            state["predict_head.norm.num_batches_tracked"].dtype,
            torch.int64,
        )


if __name__ == "__main__":
    unittest.main()
