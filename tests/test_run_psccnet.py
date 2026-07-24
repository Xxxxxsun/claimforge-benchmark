import argparse
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from eval.opensource.common import stable_json
from eval.opensource import run_psccnet as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pil_imread(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened)


def _fake_imageio_modules(
    *,
    return_value: np.ndarray | None = None,
) -> tuple[dict[str, types.ModuleType], mock.Mock]:
    read = mock.Mock(
        side_effect=_pil_imread if return_value is None else None,
        return_value=return_value,
    )
    v2 = types.ModuleType("imageio.v2")
    v2.imread = read
    package = types.ModuleType("imageio")
    package.__path__ = []
    package.v2 = v2
    return {"imageio": package, "imageio.v2": v2}, read


def _rows(pair_count: int = 3) -> list[dict]:
    rows = []
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
                        f"{rank + 100:064x}" if kind == "forged" else None
                    ),
                }
            )
    return rows


def _valid_progressive_masks() -> list[torch.Tensor]:
    mask1 = torch.linspace(
        0.0,
        1.0,
        256 * 256,
        dtype=torch.float32,
    ).reshape(1, 1, 256, 256)
    return [
        mask1,
        torch.full((1, 1, 128, 128), 0.9, dtype=torch.float32),
        torch.full((1, 1, 64, 64), 0.2, dtype=torch.float32),
        torch.full((1, 1, 32, 32), 0.7, dtype=torch.float32),
    ]


class RunPSCCNetTest(unittest.TestCase):
    def test_select_inputs_keeps_only_complete_fixed_pairs(self):
        rows = _rows()

        selected = runner.select_inputs(rows, pair_limit=2)
        all_selected = runner.select_inputs(rows, pair_limit=None)

        self.assertEqual(
            [(row["pair_rank"], row["kind"]) for row in selected],
            [
                (0, "real"),
                (0, "forged"),
                (1, "real"),
                (1, "forged"),
            ],
        )
        self.assertEqual(all_selected, rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {
                pair_rank: {
                    row["kind"]
                    for row in selected
                    if row["pair_rank"] == pair_rank
                }
                for pair_rank in {row["pair_rank"] for row in selected}
            },
            {0: {"real", "forged"}, 1: {"real", "forged"}},
        )

    def test_select_inputs_rejects_invalid_limit_and_incomplete_pairs(self):
        rows = _rows()
        for invalid_limit in (0, -1, -100):
            with self.subTest(pair_limit=invalid_limit):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    runner.select_inputs(rows, pair_limit=invalid_limit)

        incomplete = rows[:-1]
        with self.assertRaisesRegex(
            ValueError,
            "canonical selection contains incomplete pairs",
        ):
            runner.select_inputs(incomplete, pair_limit=None)

    def test_preprocess_rgb_preserves_native_size_and_divides_by_255(self):
        rgb = np.asarray(
            [
                [[0, 1, 2], [127, 128, 129], [253, 254, 255]],
                [[255, 0, 64], [3, 9, 27], [81, 162, 243]],
            ],
            dtype=np.uint8,
        )
        expected = np.ascontiguousarray(
            rgb.astype(np.float32).transpose(2, 0, 1)
            / np.float32(255.0),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            modules, read = _fake_imageio_modules()
            with mock.patch.dict(sys.modules, modules):
                tensor, native_size, metadata = runner.preprocess_image(path)

        read.assert_called_once_with(path)
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(tensor.shape, (3, 2, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(metadata["decoder"], "imageio.v2.imread")
        self.assertEqual(metadata["channel_order"], "RGB")
        self.assertEqual(metadata["native_size"], [3, 2])
        self.assertEqual(metadata["input_resize"], "none")
        self.assertIsNone(metadata["input_crop"])
        self.assertFalse(metadata["input_reencode"])
        self.assertEqual(
            metadata["normalization"],
            "uint8_rgb_divide_255",
        )
        self.assertEqual(metadata["alpha_policy"], "not_applicable")
        self.assertEqual(metadata["tensor_shape"], [3, 2, 3])
        self.assertEqual(
            metadata["tensor_sha256"],
            runner._sha256_array(expected),
        )

    def test_preprocess_grayscale_repeats_channel_without_resize(self):
        grayscale = np.asarray(
            [[0, 64, 255], [1, 128, 254]],
            dtype=np.uint8,
        )
        expected_rgb = np.repeat(grayscale[..., None], 3, axis=2)
        expected = np.ascontiguousarray(
            expected_rgb.astype(np.float32).transpose(2, 0, 1)
            / np.float32(255.0),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grayscale.png"
            Image.fromarray(grayscale, mode="L").save(path)
            modules, _ = _fake_imageio_modules()
            with mock.patch.dict(sys.modules, modules):
                tensor, native_size, metadata = runner.preprocess_image(path)

        self.assertEqual(native_size, (3, 2))
        self.assertEqual(tensor.shape, (3, 2, 3))
        np.testing.assert_array_equal(tensor, expected)
        np.testing.assert_array_equal(tensor[0], tensor[1])
        np.testing.assert_array_equal(tensor[1], tensor[2])
        self.assertEqual(
            metadata["alpha_policy"],
            "grayscale_repeated_to_rgb",
        )
        self.assertEqual(metadata["input_resize"], "none")

    def test_preprocess_rgba_composites_on_official_white_background(self):
        rgba = np.asarray(
            [
                [
                    [255, 0, 0, 255],
                    [0, 255, 0, 0],
                    [0, 0, 255, 128],
                ],
                [
                    [10, 20, 30, 64],
                    [200, 100, 50, 192],
                    [7, 11, 13, 1],
                ],
            ],
            dtype=np.uint8,
        )
        rgba_float = rgba.astype(np.float32)
        alpha = rgba_float[..., 3:4] / np.float32(255.0)
        expected_rgb = (
            rgba_float[..., :3] * alpha
            + np.float32(255.0) * (np.float32(1.0) - alpha)
        ).astype(np.uint8)
        expected = np.ascontiguousarray(
            expected_rgb.astype(np.float32).transpose(2, 0, 1)
            / np.float32(255.0),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgba.png"
            Image.fromarray(rgba, mode="RGBA").save(path)
            modules, _ = _fake_imageio_modules()
            with mock.patch.dict(sys.modules, modules):
                tensor, native_size, metadata = runner.preprocess_image(path)

        self.assertEqual(native_size, (3, 2))
        self.assertEqual(tensor.shape, (3, 2, 3))
        np.testing.assert_array_equal(tensor, expected)
        np.testing.assert_array_equal(
            expected_rgb[0],
            np.asarray(
                [
                    [255, 0, 0],
                    [255, 255, 255],
                    [126, 126, 255],
                ],
                dtype=np.uint8,
            ),
        )
        self.assertEqual(
            metadata["alpha_policy"],
            "official_white_background_rgba_composite",
        )
        self.assertEqual(metadata["input_resize"], "none")

    def test_preprocess_rejects_non_uint8_or_unsupported_decodes(self):
        modules, _ = _fake_imageio_modules(
            return_value=np.zeros((2, 3, 3), dtype=np.float32),
        )
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                ValueError,
                "unexpected decoded image dtype",
            ):
                runner.preprocess_image(Path("unused.png"))

        modules, _ = _fake_imageio_modules(
            return_value=np.zeros((2, 3, 2), dtype=np.uint8),
        )
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                ValueError,
                "unexpected decoded image shape",
            ):
                runner.preprocess_image(Path("unused.png"))

    def test_postprocess_uses_four_stages_class1_and_mask1_only(self):
        masks = _valid_progressive_masks()
        logits = torch.tensor([[-1.25, 2.5]], dtype=torch.float32)
        expected_native = F.interpolate(
            masks[0],
            size=(3, 5),
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        wrong_align_corners = F.interpolate(
            masks[0],
            size=(3, 5),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        expected_probabilities = torch.softmax(logits, dim=1)[0].numpy()

        with mock.patch.object(
            F,
            "interpolate",
            wraps=F.interpolate,
        ) as interpolate:
            (
                model_masks,
                native_map,
                logits_array,
                probabilities,
            ) = runner.postprocess_outputs(
                masks,
                logits,
                native_width=5,
                native_height=3,
            )

        interpolate.assert_called_once()
        self.assertIs(interpolate.call_args.args[0], masks[0])
        self.assertEqual(interpolate.call_args.kwargs["size"], (3, 5))
        self.assertEqual(interpolate.call_args.kwargs["mode"], "bilinear")
        self.assertIs(
            interpolate.call_args.kwargs["align_corners"],
            True,
        )
        self.assertEqual(
            [array.shape for array in model_masks],
            [(256, 256), (128, 128), (64, 64), (32, 32)],
        )
        for source, result in zip(masks, model_masks, strict=True):
            np.testing.assert_array_equal(result, source[0, 0].numpy())
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(native_map.shape, (3, 5))
        self.assertEqual(native_map.dtype, np.float32)
        self.assertTrue(native_map.flags.c_contiguous)
        np.testing.assert_allclose(
            native_map,
            expected_native,
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            float(np.max(np.abs(native_map - wrong_align_corners))),
            1e-3,
        )
        np.testing.assert_array_equal(logits_array, logits[0].numpy())
        np.testing.assert_allclose(
            probabilities,
            expected_probabilities,
            rtol=0,
            atol=1e-7,
        )
        self.assertAlmostEqual(
            float(probabilities[1]),
            float(expected_probabilities[1]),
        )
        self.assertGreater(float(probabilities[1]), 0.5)

    def test_postprocess_fails_closed_on_sequence_and_stage_shapes(self):
        logits = torch.zeros((1, 2), dtype=torch.float32)
        valid = _valid_progressive_masks()

        with self.assertRaisesRegex(ValueError, "not a sequence"):
            runner.postprocess_outputs(
                valid[0],
                logits,
                native_width=5,
                native_height=3,
            )
        with self.assertRaisesRegex(ValueError, "returned 3 masks"):
            runner.postprocess_outputs(
                valid[:3],
                logits,
                native_width=5,
                native_height=3,
            )

        expected_shapes = ((256, 256), (128, 128), (64, 64), (32, 32))
        for index, (height, width) in enumerate(expected_shapes):
            malformed = list(valid)
            malformed[index] = torch.zeros(
                (1, 1, height, width + 1),
                dtype=torch.float32,
            )
            with self.subTest(stage=index + 1):
                with self.assertRaisesRegex(
                    ValueError,
                    f"unexpected PSCC-Net mask{index + 1} shape",
                ):
                    runner.postprocess_outputs(
                        malformed,
                        logits,
                        native_width=5,
                        native_height=3,
                    )

        for shape in ((2,), (1, 1), (2, 2), (1, 2, 1)):
            with self.subTest(logits_shape=shape):
                with self.assertRaisesRegex(
                    ValueError,
                    "unexpected PSCC-Net logits shape",
                ):
                    runner.postprocess_outputs(
                        valid,
                        torch.zeros(shape, dtype=torch.float32),
                        native_width=5,
                        native_height=3,
                    )

    def test_postprocess_fails_closed_on_nonfinite_and_out_of_range_values(self):
        logits = torch.zeros((1, 2), dtype=torch.float32)
        for value, message in (
            (float("nan"), "contains non-finite values"),
            (float("inf"), "contains non-finite values"),
            (-0.0001, r"falls outside \[0, 1\]"),
            (1.0001, r"falls outside \[0, 1\]"),
        ):
            malformed = _valid_progressive_masks()
            malformed[2] = malformed[2].clone()
            malformed[2][0, 0, 0, 0] = value
            with self.subTest(mask_value=value):
                with self.assertRaisesRegex(ValueError, message):
                    runner.postprocess_outputs(
                        malformed,
                        logits,
                        native_width=5,
                        native_height=3,
                    )

        for value in (float("nan"), float("inf"), -float("inf")):
            malformed_logits = logits.clone()
            malformed_logits[0, 1] = value
            with self.subTest(logit_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "logits contain non-finite values",
                ):
                    runner.postprocess_outputs(
                        _valid_progressive_masks(),
                        malformed_logits,
                        native_width=5,
                        native_height=3,
                    )

        with mock.patch.object(
            F,
            "interpolate",
            return_value=torch.full(
                (1, 1, 3, 5),
                1.01,
                dtype=torch.float32,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"native map falls outside \[0, 1\]",
            ):
                runner.postprocess_outputs(
                    _valid_progressive_masks(),
                    logits,
                    native_width=5,
                    native_height=3,
                )

        with mock.patch.object(
            torch,
            "softmax",
            return_value=torch.full(
                (1, 2),
                float("nan"),
                dtype=torch.float32,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "probabilities are invalid",
            ):
                runner.postprocess_outputs(
                    _valid_progressive_masks(),
                    logits,
                    native_width=5,
                    native_height=3,
                )

    def test_manifest_fingerprint_and_official_contract_without_model_load(self):
        selected = runner.select_inputs(_rows(), pair_limit=1)
        args = argparse.Namespace(
            run_id="psccnet_unit_test",
            condition="mouse_canonical_v1",
            seed=42,
            device="cpu",
            classification_threshold=runner.CLASSIFICATION_THRESHOLD,
            mask_threshold=runner.MASK_THRESHOLD,
        )

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
            release = {
                "dataset_id": "mouse_canonical_v1_fixture",
                "contract_sha256": "a" * 64,
                "inputs_sha256": "b" * 64,
                "jpeg": {"quality": 95, "subsampling": 0},
            }

            with (
                mock.patch.object(
                    runner,
                    "_git_value",
                    return_value="",
                ),
                mock.patch.object(
                    runner,
                    "load_model",
                    side_effect=AssertionError(
                        "manifest construction loaded the GPU model"
                    ),
                ) as load_model,
                mock.patch.object(
                    torch.cuda,
                    "get_device_name",
                    side_effect=AssertionError(
                        "CPU manifest queried a CUDA device"
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
                    psccnet_root=root / "PSCC-Net",
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
                    {**immutable, "run_id": "different-run"}
                ),
            )

            ordered_inputs = runner._selection_contract(selected)
            expected_selection_sha = hashlib.sha256(
                stable_json(ordered_inputs).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                manifest["input"]["selection_sha256"],
                expected_selection_sha,
            )
            self.assertEqual(manifest["ordered_inputs"], ordered_inputs)
            self.assertEqual(manifest["expected_pairs"], 1)
            self.assertEqual(manifest["expected_images"], 2)

            model = manifest["model"]
            self.assertEqual(model["name"], runner.MODEL_NAME)
            self.assertEqual(model["model_slug"], runner.MODEL_SLUG)
            self.assertEqual(model["repo_url"], runner.MODEL_REPO_URL)
            self.assertEqual(
                model["source_commit"],
                runner.MODEL_SOURCE_COMMIT,
            )
            self.assertTrue(model["source_tracked_clean"])
            self.assertEqual(
                model["variant"],
                "official_committed_pretrained_checkpoint",
            )
            self.assertEqual(
                model["source_files"],
                [
                    {"path": path, "sha256": sha256}
                    for path, sha256 in runner.SOURCE_FILES.items()
                ],
            )
            self.assertEqual(
                model["license"],
                {
                    "path": "LICENSE",
                    "sha256": runner.SOURCE_FILES["LICENSE"],
                    "spdx": "MIT",
                    "scope": "project_repository",
                },
            )
            self.assertEqual(
                model["initialization_weight"],
                runner.INITIALIZATION_WEIGHT,
            )
            checkpoint = model["checkpoint"]
            self.assertEqual(
                checkpoint["provider"],
                "official_author_git_repository",
            )
            self.assertEqual(
                checkpoint["source_commit"],
                runner.MODEL_SOURCE_COMMIT,
            )
            self.assertEqual(
                checkpoint["bundle_sha256"],
                runner.CHECKPOINT_BUNDLE_SHA256,
            )
            self.assertEqual(
                checkpoint["components"],
                [
                    {"role": role, **contract}
                    for role, contract in runner.CHECKPOINTS.items()
                ],
            )
            self.assertIs(checkpoint["strict_load"], True)
            self.assertIs(checkpoint["safe_weights_only_load"], True)
            self.assertEqual(
                model["parameter_count"],
                sum(
                    contract["parameters"]
                    for contract in runner.CHECKPOINTS.values()
                ),
            )
            self.assertEqual(model["class_names"], ["authentic", "forged"])
            self.assertEqual(model["positive_class_index"], 1)
            self.assertTrue(model["supports_image_level_t1"])
            self.assertTrue(model["supports_pixel_level_t2"])
            self.assertEqual(
                model["image_score_source"],
                "native_independent_classification_head",
            )
            self.assertEqual(
                model["primary_localization_output"],
                "progressive_mask1",
            )

            inference = manifest["inference"]
            self.assertEqual(inference["decoder"], "imageio.v2.imread")
            self.assertEqual(inference["channel_order"], "RGB")
            self.assertEqual(inference["input_resize"], "none")
            self.assertIsNone(inference["input_crop"])
            self.assertFalse(inference["input_reencode"])
            self.assertEqual(
                inference["normalization"],
                "uint8_rgb_divide_255",
            )
            self.assertEqual(inference["internal_crop_size"], [256, 256])
            self.assertEqual(
                inference["progressive_output_shapes"],
                [[256, 256], [128, 128], [64, 64], [32, 32]],
            )
            self.assertEqual(
                inference["primary_map"],
                "progressive_mask1_sigmoid_probability",
            )
            self.assertEqual(
                inference["primary_map_selection"],
                "fixed_by_official_test_py_index_0",
            )
            self.assertEqual(
                inference["native_restore"],
                "bilinear_probability_align_corners_true_to_input_size",
            )
            self.assertEqual(
                inference["classification_output"],
                "softmax_two_class_logits_positive_index_1",
            )
            self.assertEqual(
                inference["classification_threshold"],
                0.5,
            )
            self.assertEqual(
                inference["classification_threshold_comparison"],
                "strict_greater_than",
            )
            self.assertEqual(inference["mask_threshold"], 0.5)
            self.assertEqual(
                inference["mask_threshold_comparison"],
                "strict_greater_than",
            )

            metrics = manifest["metrics"]
            self.assertEqual(
                metrics["task"],
                "T1_image_detection_and_T2_pixel_localization",
            )
            self.assertEqual(metrics["classification_threshold"], 0.5)
            self.assertEqual(metrics["mask_threshold"], 0.5)
            self.assertEqual(
                metrics["threshold_comparison"],
                "strict_greater_than",
            )
            self.assertFalse(metrics["prediction_inversion"])
            self.assertEqual(
                metrics["model_space_gt_resize"],
                "nearest_neighbor_to_256x256",
            )

            adapter_paths = {
                item["path"] for item in manifest["adapter_contract"]
            }
            self.assertEqual(
                adapter_paths,
                {
                    "eval/opensource/run_psccnet.py",
                    "eval/opensource/psccnet_metrics.py",
                    "eval/opensource/common.py",
                },
            )
            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in manifest["adapter_contract"]
                )
            )

            run_manifest_path = root / "run_manifest.json"
            runner._write_or_validate_run_manifest(
                run_manifest_path,
                manifest,
            )
            runner._write_or_validate_run_manifest(
                run_manifest_path,
                {
                    **manifest,
                    "created_at": "ignored-for-compatible-resume",
                },
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runner._write_or_validate_run_manifest(
                    run_manifest_path,
                    {
                        **manifest,
                        "fingerprint": "f" * 64,
                    },
                )


if __name__ == "__main__":
    unittest.main()
