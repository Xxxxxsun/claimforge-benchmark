import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.run_hifi_ifdl import (
    CENTER_RADIUS,
    CLASSIFICATION_THRESHOLD,
    FINE_CLASS_NAMES,
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    PAIRWISE_EPS,
    _atomic_save_mask,
    _atomic_save_npy,
    _load_center_radius,
    _load_checkpoint_state,
    _numpy_int_compatibility,
    _validate_resume_rows,
    _write_or_validate_run_manifest,
    infer_one,
    model_space_target,
    postprocess_outputs,
    preprocess_image,
    select_inputs,
)


def _synthetic_outputs():
    import torch

    embedding = torch.zeros(
        (1, 18, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        dtype=torch.float32,
    )
    auxiliary = torch.full(
        (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        0.25,
        dtype=torch.float32,
    )
    coarse = torch.asarray([[0.0, 1.0, 2.0]], dtype=torch.float32)
    five = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    seven = torch.arange(7, dtype=torch.float32).unsqueeze(0)
    # Authentic is the single largest fine class, while the sum of the
    # thirteen forged probabilities is greater than 0.5.
    fine = torch.asarray(
        [[1.0, *([0.9] * 13)]],
        dtype=torch.float32,
    )
    return embedding, auxiliary, coarse, five, seven, fine


def _fake_imageio_modules():
    imageio_package = types.ModuleType("imageio")
    imageio_v2 = types.ModuleType("imageio.v2")

    def imread(path):
        with Image.open(path) as opened:
            return np.asarray(opened)

    imageio_v2.imread = imread
    imageio_package.v2 = imageio_v2
    return {
        "imageio": imageio_package,
        "imageio.v2": imageio_v2,
    }


class HiFiIFDLRunnerTests(unittest.TestCase):
    def test_select_inputs_preserves_complete_pairs(self):
        rows = [
            {
                "pair_rank": pair,
                "kind": kind,
                "rank": pair * 2 + offset,
            }
            for pair in range(3)
            for offset, kind in enumerate(("real", "forged"))
        ]
        selected = select_inputs(rows, pair_limit=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["pair_rank"] for row in selected}, {0, 1})
        with self.assertRaisesRegex(ValueError, "positive"):
            select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            select_inputs(rows[:-1], pair_limit=None)

    def test_preprocess_is_rgb_float32_stretched_to_256(self):
        pixels = np.zeros((3, 5, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        pixels[1, 2] = [0, 128, 255]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            with mock.patch.dict(
                sys.modules,
                _fake_imageio_modules(),
            ):
                first, native_size, metadata = preprocess_image(path)
                second, _, second_metadata = preprocess_image(path)
        self.assertEqual(first.shape, (3, 256, 256))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)
        self.assertEqual(native_size, (5, 3))
        self.assertEqual(metadata["geometry"], (
            "direct_stretch_without_aspect_ratio_preservation"
        ))
        self.assertEqual(metadata["tensor_sha256"], second_metadata["tensor_sha256"])
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_preprocess_rejects_non_rgb_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            Image.fromarray(
                np.zeros((2, 2), dtype=np.uint8),
                mode="L",
            ).save(path)
            with mock.patch.dict(
                sys.modules,
                _fake_imageio_modules(),
            ):
                with self.assertRaisesRegex(ValueError, "decoded RGB"):
                    preprocess_image(path)

    def test_postprocess_keeps_official_and_benchmark_t1_separate(self):
        import torch

        result = postprocess_outputs(
            _synthetic_outputs(),
            torch.zeros(18, dtype=torch.float32),
            native_width=7,
            native_height=5,
        )
        self.assertEqual(result["embedding"].shape, (18, 256, 256))
        self.assertEqual(result["embedding"].dtype, np.float32)
        self.assertEqual(result["distance_model_256"].shape, (256, 256))
        self.assertEqual(result["distance_native"].shape, (5, 7))
        self.assertEqual(result["distance_native"].dtype, np.float32)
        self.assertTrue(
            np.allclose(
                result["distance_model_256"],
                np.sqrt(18) * PAIRWISE_EPS,
                rtol=0.0,
                atol=1e-9,
            )
        )
        self.assertGreater(result["score"], CLASSIFICATION_THRESHOLD)
        self.assertTrue(result["benchmark_binary_decision"])
        self.assertEqual(result["official_fine_class_index"], 0)
        self.assertEqual(result["official_fine_class_name"], "authentic")
        self.assertFalse(result["official_binary_decision"])
        self.assertEqual(
            set(result["hierarchy_logits"]),
            {
                "out0_coarse_3class",
                "out1_5class",
                "out2_7class",
                "out3_fine_14class",
            },
        )
        self.assertAlmostEqual(
            sum(result["fine_probabilities"]),
            1.0,
            places=5,
        )

    def test_postprocess_rejects_wrong_output_contract(self):
        import torch

        with self.assertRaisesRegex(ValueError, "expected 6"):
            postprocess_outputs(
                _synthetic_outputs()[:-1],
                torch.zeros(18, dtype=torch.float32),
                native_width=2,
                native_height=2,
            )
        broken = list(_synthetic_outputs())
        broken[0] = torch.zeros((1, 17, 256, 256))
        with self.assertRaisesRegex(ValueError, "embedding shape"):
            postprocess_outputs(
                broken,
                torch.zeros(18, dtype=torch.float32),
                native_width=2,
                native_height=2,
            )

    def test_model_space_target_is_exact_nearest_binary(self):
        target = np.asarray(
            [[True, False], [False, False]],
            dtype=bool,
        )
        resized = model_space_target(target)
        self.assertEqual(resized.shape, (256, 256))
        self.assertEqual(resized.dtype, np.bool_)
        self.assertTrue(resized[:128, :128].all())
        self.assertFalse(resized[:128, 128:].any())
        self.assertFalse(resized[128:, :].any())

    def test_artifact_writers_are_lossless_and_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array_path = root / "map.npy"
            mask_path = root / "mask.png"
            array = np.asarray([[0.0, 3.5]], dtype=np.float32)
            _atomic_save_npy(array_path, array)
            _atomic_save_mask(mask_path, array >= MASK_THRESHOLD)
            loaded = np.load(array_path, allow_pickle=False)
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened, dtype=np.uint8)
        np.testing.assert_array_equal(loaded, array)
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(
            mask,
            np.asarray([[0, 255]], dtype=np.uint8),
        )

    def test_checkpoint_loader_requires_exact_registered_schema(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            good_path = Path(directory) / "good.pth"
            bad_path = Path(directory) / "bad.pth"
            source = torch.nn.Linear(2, 1)
            state = torch.nn.DataParallel(source).state_dict()
            torch.save({"model": state, "optimizer": {}}, good_path)
            torch.save({"model": state}, bad_path)
            contract = {
                "top_level_keys": ["model", "optimizer"],
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
            loaded = _load_checkpoint_state(
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
                _load_checkpoint_state(
                    module=torch.nn.Linear(2, 1),
                    path=bad_path,
                    contract=contract,
                    label="fixture",
                )

    def test_center_radius_loader_requires_frozen_value_and_shape(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            good_path = Path(directory) / "center.pth"
            bad_path = Path(directory) / "bad.pth"
            torch.save(
                {
                    "center": torch.zeros(18, dtype=torch.float32),
                    "radius": torch.tensor(
                        CENTER_RADIUS["radius_value"],
                        dtype=torch.float32,
                    ),
                },
                good_path,
            )
            torch.save(
                {
                    "center": torch.zeros(17, dtype=torch.float32),
                    "radius": torch.tensor(
                        CENTER_RADIUS["radius_value"],
                        dtype=torch.float32,
                    ),
                },
                bad_path,
            )
            center, radius = _load_center_radius(good_path)
            self.assertEqual(tuple(center.shape), (18,))
            self.assertEqual(tuple(radius.shape), ())
            with self.assertRaisesRegex(ValueError, "center tensor schema"):
                _load_center_radius(bad_path)

    def test_numpy_compatibility_alias_is_scoped(self):
        original_present = "int" in np.__dict__
        original = np.__dict__.get("int")
        if original_present:
            delattr(np, "int")
        try:
            self.assertNotIn("int", np.__dict__)
            with _numpy_int_compatibility():
                self.assertIs(np.int, int)
            self.assertNotIn("int", np.__dict__)
        finally:
            if original_present:
                setattr(np, "int", original)

    def test_resume_requires_matching_fingerprint_and_identity(self):
        selected = [
            {
                "sample_id": "x",
                "canonical_sha256": "image-sha",
                "task_id": "task",
                "kind": "real",
            }
        ]
        valid = {
            "x": {
                "run_manifest_fingerprint": "run-sha",
                "image_sha256": "image-sha",
                "task_id": "task",
                "kind": "real",
            }
        }
        _validate_resume_rows(valid, selected, "run-sha")
        broken = json.loads(json.dumps(valid))
        broken["x"]["run_manifest_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "incompatible run fingerprint"):
            _validate_resume_rows(broken, selected, "run-sha")

    def test_existing_manifest_must_have_same_immutable_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = {"fingerprint": "a", "created_at": "one"}
            _write_or_validate_run_manifest(path, first)
            _write_or_validate_run_manifest(
                path,
                {"fingerprint": "a", "created_at": "two"},
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _write_or_validate_run_manifest(
                    path,
                    {"fingerprint": "b"},
                )

    def test_infer_one_uses_one_forward_for_each_model(self):
        import torch

        class Feature(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                return (image,)

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, features, image):
                self.calls += 1
                self.asserted = features[0] is image
                return _synthetic_outputs()

        feature = Feature()
        head = Head()
        result, peak, latency = infer_one(
            (feature, head),
            torch.zeros(18, dtype=torch.float32),
            torch.device("cpu"),
            np.zeros((3, 256, 256), dtype=np.float32),
            native_width=3,
            native_height=2,
        )
        self.assertEqual(feature.calls, 1)
        self.assertEqual(head.calls, 1)
        self.assertTrue(head.asserted)
        self.assertEqual(result["distance_native"].shape, (2, 3))
        self.assertEqual(peak, 0)
        self.assertGreaterEqual(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
