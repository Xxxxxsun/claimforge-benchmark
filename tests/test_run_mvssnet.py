import argparse
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from eval.opensource.common import read_jsonl
from eval.opensource.mvssnet_metrics import (
    binary_pixel_metrics_strict,
    image_detection_metrics_strict,
)
from eval.opensource.run_mvssnet import (
    CHECKPOINT_BYTES,
    CHECKPOINT_STATE_ELEMENTS,
    CHECKPOINT_STATE_KEYS,
    CLASSIFICATION_THRESHOLD,
    DEFAULT_CHECKPOINT,
    DEFAULT_MVSSNET_ROOT,
    MASK_THRESHOLD,
    MODEL_BUFFER_ELEMENTS,
    MODEL_INPUT_SIZE,
    MODEL_PARAMETER_COUNT,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    _write_or_validate_run_manifest,
    build_run_manifest,
    load_model,
    official_postprocess,
    preprocess_image,
    select_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = (
    REPO_ROOT / "outputs/opensource/mouse_canonical_v1/manifest.json"
)
CANONICAL_INPUTS = (
    REPO_ROOT / "outputs/opensource/mouse_canonical_v1/inputs.jsonl"
)


def _canonical_rows() -> list[dict]:
    return read_jsonl(CANONICAL_INPUTS)


class RunMVSSNetTest(unittest.TestCase):
    def test_preprocess_uses_cv2_bgr_512_stretch_and_bgr_normalization(self):
        import cv2

        rgb = np.asarray(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[11, 37, 83], [101, 149, 211], [7, 19, 251]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "colors.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            decoded_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            expected_resized = cv2.resize(
                decoded_bgr,
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            expected = (
                expected_resized.astype(np.float32) / np.float32(255.0)
                - NORMALIZE_MEAN
            ) / NORMALIZE_STD
            expected = np.ascontiguousarray(
                expected.transpose(2, 0, 1),
                dtype=np.float32,
            )

            with mock.patch.object(
                cv2,
                "resize",
                wraps=cv2.resize,
            ) as resize:
                tensor, native_size, metadata = preprocess_image(path)

        resize.assert_called_once()
        self.assertEqual(
            resize.call_args.args[1],
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        )
        self.assertEqual(
            resize.call_args.kwargs["interpolation"],
            cv2.INTER_LINEAR,
        )
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(tensor.shape, (3, 512, 512))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_array_equal(tensor, expected)

        # The top-left RGB red pixel must arrive at the model as BGR blue=0,
        # green=0, red=255, with the published three constants applied in
        # that BGR order.
        np.testing.assert_array_equal(
            tensor[:, 0, 0],
            (
                np.asarray([0, 0, 255], dtype=np.float32)
                / np.float32(255.0)
                - NORMALIZE_MEAN
            )
            / NORMALIZE_STD,
        )
        self.assertEqual(metadata["decoder"], "opencv_imread_color")
        self.assertEqual(metadata["channel_order"], "BGR")
        self.assertEqual(
            metadata["resize"],
            "opencv_inter_linear_stretch",
        )
        self.assertEqual(metadata["model_size"], [512, 512])
        self.assertEqual(
            metadata["normalization"]["mean_in_bgr_order"],
            NORMALIZE_MEAN.tolist(),
        )
        self.assertEqual(
            metadata["normalization"]["std_in_bgr_order"],
            NORMALIZE_STD.tolist(),
        )

    def test_official_postprocess_truncates_to_uint8_before_native_resize(self):
        import cv2

        rng = np.random.default_rng(7)
        score_map = rng.random(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            dtype=np.float32,
        )

        restored = official_postprocess(score_map, width=7, height=5)

        model_uint8 = (
            score_map * np.float32(255.0)
        ).astype(np.uint8)
        expected = cv2.resize(
            model_uint8,
            (7, 5),
            interpolation=cv2.INTER_LINEAR,
        )
        wrong_resize_then_quantize = (
            cv2.resize(
                score_map,
                (7, 5),
                interpolation=cv2.INTER_LINEAR,
            )
            * np.float32(255.0)
        ).astype(np.uint8)
        np.testing.assert_array_equal(restored, expected)
        self.assertGreater(
            int(np.count_nonzero(restored != wrong_resize_then_quantize)),
            0,
        )
        self.assertEqual(restored.dtype, np.uint8)
        self.assertTrue(restored.flags.c_contiguous)

        # A continuous GMP score can cross 0.5 while the official saved PNG
        # score remains below 0.5 because float*255 is truncated first.
        boundary_map = np.full(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            0.501,
            dtype=np.float32,
        )
        boundary_png = official_postprocess(
            boundary_map,
            width=1,
            height=1,
        )
        raw_score = float(np.max(boundary_map))
        png_score = float(np.max(boundary_png)) / 255.0
        self.assertGreater(raw_score, CLASSIFICATION_THRESHOLD)
        self.assertEqual(int(boundary_png[0, 0]), 127)
        self.assertLess(png_score, CLASSIFICATION_THRESHOLD)

    def test_t1_and_t2_thresholds_are_strict_greater_than(self):
        score_map = np.asarray(
            [[0.5, 0.5001], [0.1, 0.9]],
            dtype=np.float32,
        )
        target = np.asarray(
            [[1, 1], [0, 0]],
            dtype=bool,
        )

        localization = binary_pixel_metrics_strict(
            score_map,
            target,
            MASK_THRESHOLD,
        )
        detection = image_detection_metrics_strict(
            [
                {"status": "ok", "label": 0, "score": 0.5},
                {"status": "ok", "label": 1, "score": 0.5},
                {"status": "ok", "label": 1, "score": 0.5001},
            ],
            CLASSIFICATION_THRESHOLD,
        )

        self.assertEqual(localization["threshold_operator"], ">")
        self.assertEqual(localization["tp"], 1)
        self.assertEqual(localization["fn"], 1)
        self.assertEqual(localization["fp"], 1)
        self.assertEqual(detection["threshold_operator"], ">")
        self.assertEqual(detection["tp"], 1)
        self.assertEqual(detection["fn"], 1)
        self.assertEqual(detection["fp"], 0)
        self.assertEqual(detection["tn"], 1)

    def test_select_inputs_keeps_complete_fixed_pairs(self):
        rows = [
            {
                "rank": pair * 2 + offset,
                "pair_rank": pair,
                "kind": kind,
            }
            for pair in range(3)
            for offset, kind in enumerate(("real", "forged"))
        ]

        selected = select_inputs(rows, pair_limit=2)

        self.assertEqual(
            [(row["pair_rank"], row["kind"]) for row in selected],
            [
                (0, "real"),
                (0, "forged"),
                (1, "real"),
                (1, "forged"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            select_inputs(rows[:-1], pair_limit=None)

    @unittest.skipUnless(
        CANONICAL_MANIFEST.is_file() and CANONICAL_INPUTS.is_file(),
        "canonical release fixture is required",
    )
    def test_manifest_declares_map_derived_t1_without_separate_head(self):
        release = json.loads(
            CANONICAL_MANIFEST.read_text(encoding="utf-8")
        )
        selected = select_inputs(_canonical_rows(), pair_limit=1)
        args = argparse.Namespace(
            run_id="mvssnet_unit_test",
            condition="mouse_canonical_v1",
            seed=42,
            device="cpu",
            classification_threshold=CLASSIFICATION_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
        )

        manifest = build_run_manifest(
            args=args,
            repo_root=REPO_ROOT,
            dataset_manifest_path=CANONICAL_MANIFEST,
            release=release,
            inputs_path=CANONICAL_INPUTS,
            selected=selected,
            mvssnet_root=DEFAULT_MVSSNET_ROOT,
            checkpoint_path=DEFAULT_CHECKPOINT,
            artifact_dir=(
                REPO_ROOT
                / "outputs/opensource/mvssnet/mvssnet_unit_test"
            ),
        )

        model = manifest["model"]
        inference = manifest["inference"]
        metrics = manifest["metrics"]
        self.assertTrue(model["supports_image_level_t1"])
        self.assertTrue(model["supports_pixel_level_t2"])
        self.assertEqual(
            model["image_level_head"],
            "none_map_global_max_pooling",
        )
        self.assertEqual(
            inference["image_level_head"],
            "none_map_derived_gmp",
        )
        self.assertEqual(
            inference["primary_t1_score"],
            "continuous_global_max_of_model_512_sigmoid_probability",
        )
        self.assertEqual(
            inference["official_evaluate_t1_score"],
            "maximum_of_saved_native_uint8_map_divided_by_255",
        )
        self.assertEqual(
            inference["classification_threshold_comparison"],
            "strict_greater_than",
        )
        self.assertEqual(
            inference["mask_threshold_comparison"],
            "strict_greater_than",
        )
        self.assertEqual(metrics["primary_t1"], "continuous_model_space_gmp")
        self.assertEqual(metrics["secondary_t1"], "official_saved_png_gmp")
        self.assertEqual(
            model["checkpoint"]["safe_weights_only_load"],
            True,
        )
        self.assertEqual(model["checkpoint"]["strict_load"], True)
        self.assertEqual(
            model["license"]["commercial_or_redistribution_permission"],
            "not_established",
        )
        self.assertEqual(manifest["expected_pairs"], 1)
        self.assertEqual(manifest["expected_images"], 2)

    def test_existing_manifest_rejects_resume_fingerprint_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            first = {
                "run_id": "mvssnet-test",
                "fingerprint": "a" * 64,
            }
            _write_or_validate_run_manifest(path, first)
            _write_or_validate_run_manifest(
                path,
                {**first, "created_at": "ignored-for-resume"},
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _write_or_validate_run_manifest(
                    path,
                    {**first, "fingerprint": "b" * 64},
                )

    @unittest.skipUnless(
        DEFAULT_MVSSNET_ROOT.is_dir() and DEFAULT_CHECKPOINT.is_file(),
        "official MVSS-Net source and checkpoint are not installed",
    )
    def test_official_checkpoint_safe_strict_load_bypasses_downloads(self):
        import torch.utils.model_zoo as model_zoo

        self.assertEqual(DEFAULT_CHECKPOINT.stat().st_size, CHECKPOINT_BYTES)
        strict_loads: list[tuple[str, bool, int]] = []
        original_load_state_dict = torch.nn.Module.load_state_dict

        def record_load_state_dict(
            module,
            state_dict,
            strict=True,
            assign=False,
        ):
            strict_loads.append(
                (type(module).__name__, bool(strict), len(state_dict))
            )
            return original_load_state_dict(
                module,
                state_dict,
                strict=strict,
                assign=assign,
            )

        real_torch_load = torch.load
        with (
            mock.patch.object(
                model_zoo,
                "load_url",
                side_effect=AssertionError(
                    "MVSS-Net attempted an unpinned ImageNet download"
                ),
            ) as load_url,
            mock.patch.object(
                torch,
                "load",
                wraps=real_torch_load,
            ) as safe_load,
            mock.patch.object(
                torch.nn.Module,
                "load_state_dict",
                new=record_load_state_dict,
            ),
        ):
            model, device = load_model(
                mvssnet_root=DEFAULT_MVSSNET_ROOT,
                checkpoint_path=DEFAULT_CHECKPOINT,
                device_name="cpu",
            )

        load_url.assert_not_called()
        checkpoint_calls = [
            call
            for call in safe_load.call_args_list
            if Path(call.args[0]).resolve() == DEFAULT_CHECKPOINT.resolve()
        ]
        self.assertEqual(len(checkpoint_calls), 1)
        self.assertEqual(
            checkpoint_calls[0].kwargs,
            {"map_location": "cpu", "weights_only": True},
        )
        self.assertIn(
            ("MVSSNet", True, CHECKPOINT_STATE_KEYS),
            strict_loads,
        )
        self.assertEqual(str(device), "cpu")
        self.assertFalse(model.training)
        self.assertEqual(len(model.state_dict()), CHECKPOINT_STATE_KEYS)
        self.assertEqual(
            sum(int(value.numel()) for value in model.state_dict().values()),
            CHECKPOINT_STATE_ELEMENTS,
        )
        self.assertEqual(
            sum(int(value.numel()) for value in model.parameters()),
            MODEL_PARAMETER_COUNT,
        )
        self.assertEqual(
            sum(int(value.numel()) for value in model.buffers()),
            MODEL_BUFFER_ELEMENTS,
        )
        self.assertEqual(
            next(model.parameters()).device.type,
            "cpu",
        )
        del model
        gc.collect()


if __name__ == "__main__":
    unittest.main()
