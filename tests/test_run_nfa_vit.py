from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import albumentations as albu
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from eval.opensource import run_nfa_vit as runner


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
                    "domain": "lodging",
                    "kind": kind,
                    "label": int(kind == "forged"),
                    "width": 3,
                    "height": 2,
                    "canonical_path": f"images/{kind}-{pair_rank}.jpg",
                    "canonical_sha256": f"{rank + 1:064x}",
                    "gt_mask_kind": (
                        "exact_diff" if kind == "forged" else "all_zero"
                    ),
                    "gt_mask_sha256": (
                        f"{rank + 101:064x}" if kind == "forged" else None
                    ),
                    "edit_region_xyxy": [0, 0, 1, 1],
                }
            )
    return rows


class _TinySegDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.avg_pool2d(image[:, :1], kernel_size=4, stride=4)


class _TinyClassificationDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return image[:, :1].mean(dim=(2, 3))


class _TinyNfaForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seg_decoder = _TinySegDecoder()
        self.cls_decoder = _TinyClassificationDecoder()
        self.seen_mask: torch.Tensor | None = None
        self.seen_label: torch.Tensor | None = None

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.seen_mask = mask.detach().clone()
        self.seen_label = label.detach().clone()
        logits = self.seg_decoder(image)
        pred_mask = torch.sigmoid(
            F.interpolate(
                logits,
                size=(runner.MODEL_INPUT_SIZE, runner.MODEL_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        )
        cls_logit = self.cls_decoder(image)
        return {
            "pred_mask": pred_mask,
            # The author code squeezes this to a scalar for batch size one.
            "pred_label": torch.sigmoid(cls_logit).squeeze(),
        }


class _TinyDnCNN(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 1, kernel_size=1))


class _UnexpectedCheckpointGlobal:
    pass


class _TinyConstructorModel(torch.nn.Module):
    def __init__(
        self,
        np_pretrain_weights: str,
        seg_b0_pretrain_weights: str,
        seg_b2_pretrain_weights: str,
    ) -> None:
        super().__init__()
        # Match the three unconditional torch.load calls in the author
        # constructor. The returned initialization payloads are deliberately
        # irrelevant because the released full state must replace everything.
        self.constructor_payloads = [
            torch.load(np_pretrain_weights),
            torch.load(seg_b0_pretrain_weights),
            torch.load(seg_b2_pretrain_weights),
        ]
        self.noise_extractor = torch.nn.Linear(1, 1)
        for parameter in self.noise_extractor.parameters():
            parameter.requires_grad = False
        self.projection = torch.nn.Linear(2, 1)
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


_TINY_PACKAGE = "_claimforge_test_nfa"
_TINY_MODEL_MODULE = f"{_TINY_PACKAGE}.nfa_vit"
_TinyConstructorModel.__module__ = _TINY_MODEL_MODULE


class RunNfaVitTest(unittest.TestCase):
    def test_repository_contract_ignores_generated_untracked_files(self):
        with (
            mock.patch.object(
                runner,
                "_git_value",
                side_effect=[
                    runner.MODEL_SOURCE_COMMIT,
                    "",
                ],
            ) as git_value,
            mock.patch.object(runner, "_verify_runtime_file"),
        ):
            runner._verify_repository_contract(
                runner.DEFAULT_BRGEN_ROOT,
                expected_commit=runner.MODEL_SOURCE_COMMIT,
                source_files={"model_zoo/nfa_vit/nfa_vit.py": "a" * 64},
                label="BR-Gen",
            )
        self.assertEqual(
            git_value.call_args_list[1],
            mock.call(
                runner.DEFAULT_BRGEN_ROOT,
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
        )

    def test_frozen_static_contract(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "4ced0e0966e96b9bd637cb34aa4ab8ab8eade782",
        )
        self.assertEqual(
            runner.IMDLBENCO_SOURCE_COMMIT,
            "4e55633c3e68ede63974f72ea9af1a803a7f5ae8",
        )
        self.assertEqual(runner.UPSTREAM_CLASS_NAME, "NFA_ViT_modify1")
        self.assertEqual(runner.UPSTREAM_BROKEN_EXPORT_NAME, "NFA_ViT")
        self.assertEqual(runner.MODEL_INPUT_SIZE, 512)
        self.assertEqual(runner.DECODER_LOGIT_SIZE, 128)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD, 0.5)
        self.assertEqual(runner.MASK_THRESHOLD, 0.5)
        self.assertEqual(runner.CLASSIFICATION_THRESHOLD_OPERATOR, ">")
        self.assertEqual(runner.MASK_THRESHOLD_OPERATOR, ">")
        self.assertEqual(
            runner.IMDLBENCO_SOURCE_FILES[
                "IMDLBenCo/training_scripts/utils/misc.py"
            ],
            "743545325d61b3f40ea2f2ffd30f77d6a941d4ebbe64532a961cf3fc98d422e5",
        )
        self.assertEqual(
            runner.CHECKPOINT_SAFE_GLOBALS,
            {"argparse.Namespace": argparse.Namespace},
        )
        self.assertEqual(runner.CHECKPOINT["released_filename"], "checkpoint-9999.pth")
        self.assertEqual(runner.CHECKPOINT["bytes"], 312_770_914)
        self.assertEqual(
            runner.CHECKPOINT["md5"],
            "b7f0b0e3ff2be6d49b31ea57c2d09cff",
        )
        self.assertIsNone(
            runner.CHECKPOINT["sha256"],
            "the official checkpoint is still access-blocked, so no SHA-256 may be invented",
        )
        self.assertFalse(
            runner._valid_sha256(runner.CHECKPOINT["sha256"]),
            "the closed checkpoint gate must remain non-executable",
        )

    def test_select_inputs_preserves_pairs_and_supports_one_sample_preflight(self):
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
        self.assertEqual(
            runner.select_inputs(
                rows,
                pair_limit=None,
                sample_id="forged-2",
            ),
            [rows[-1]],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.select_inputs(rows, pair_limit=1, sample_id="forged-0")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            runner.select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.select_inputs(rows, pair_limit=None, sample_id="missing")
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            runner.select_inputs(rows[:-1], pair_limit=None)

    def test_preprocess_matches_pinned_albumentations_1_3_pipeline_exactly(self):
        self.assertEqual(albu.__version__, "1.3.0")
        rgb = np.asarray(
            [
                [[255, 0, 1], [0, 255, 2], [0, 0, 255]],
                [[12, 34, 56], [127, 128, 129], [253, 254, 255]],
            ],
            dtype=np.uint8,
        )
        reference = albu.Compose(
            [
                albu.Resize(
                    512,
                    512,
                    interpolation=cv2.INTER_LINEAR,
                ),
                albu.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
            ]
        )(image=rgb)["image"]
        expected = np.ascontiguousarray(
            np.asarray(reference, dtype=np.float32).transpose(2, 0, 1)
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            tensor, native_size, metadata = runner.preprocess_image(path)

        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tensor.shape, (3, 512, 512))
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(metadata["channel_order"], "RGB")
        self.assertFalse(metadata["exif_transpose"])
        self.assertFalse(metadata["icc_conversion"])
        self.assertEqual(metadata["native_size"], [3, 2])
        self.assertEqual(metadata["model_size"], [512, 512])
        self.assertEqual(
            metadata["geometry"],
            "direct_stretch_without_aspect_ratio_preservation",
        )
        self.assertEqual(
            metadata["resize"],
            "albumentations_1_3_0.Resize_cv2_INTER_LINEAR",
        )
        self.assertEqual(
            metadata["noiseprint_input"],
            "same_ImageNet_normalized_RGB_tensor_no_separate_preprocessing",
        )
        self.assertEqual(metadata["tensor_sha256"], runner._sha256_array(tensor))

    def test_postprocess_replays_logits_and_uses_native_t1_head_strictly(self):
        base = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]]]],
            dtype=torch.float32,
        )
        logits_128 = F.interpolate(
            base,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        logits_512 = F.interpolate(
            logits_128,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = torch.sigmoid(logits_512)
        # The dense head is strongly positive in places, while the native
        # classification head is exactly on the threshold.
        cls_logit = torch.tensor([[0.0]], dtype=torch.float32)
        output = {
            "pred_mask": pred_mask,
            "pred_label": torch.sigmoid(cls_logit).squeeze(),
        }
        processed = runner.postprocess_outputs(
            output,
            logits_128,
            cls_logit,
            native_width=3,
            native_height=2,
        )

        np.testing.assert_array_equal(
            processed["decoder_logits_128"],
            logits_128[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["resized_logits_512"],
            logits_512[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_512"],
            pred_mask[0, 0].numpy(),
        )
        expected_native = F.interpolate(
            pred_mask,
            size=(2, 3),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        np.testing.assert_array_equal(
            processed["probability_native"],
            expected_native,
        )
        self.assertEqual(processed["classification_raw_logit"], 0.0)
        self.assertEqual(processed["classification_score"], 0.5)
        self.assertFalse(processed["classification_decision"])
        self.assertGreater(float(processed["probability_512"].max()), 0.99)

        above = np.nextafter(
            np.float32(0.5),
            np.float32(1.0),
            dtype=np.float32,
        )
        metrics = runner.binary_pixel_metrics_strict(
            np.asarray([[0.5, above]], dtype=np.float32),
            np.asarray([[False, True]], dtype=bool),
            threshold=0.5,
            include_ap=True,
        )
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["predicted_positive_pixels"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)

    def test_postprocess_rejects_non_replayable_outputs(self):
        logits = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        cls_logit = torch.zeros((1, 1), dtype=torch.float32)
        good = {
            "pred_mask": torch.full((1, 1, 512, 512), 0.5),
            "pred_label": torch.tensor(0.5),
        }
        with self.assertRaisesRegex(ValueError, "decoder logit shape"):
            runner.postprocess_outputs(
                good,
                logits[:, :, :-1],
                cls_logit,
                native_width=2,
                native_height=2,
            )
        wrong_mask = dict(good)
        wrong_mask["pred_mask"] = good["pred_mask"] + 0.1
        with self.assertRaisesRegex(ValueError, "does not replay"):
            runner.postprocess_outputs(
                wrong_mask,
                logits,
                cls_logit,
                native_width=2,
                native_height=2,
            )
        wrong_label = dict(good)
        wrong_label["pred_label"] = torch.tensor(0.6)
        with self.assertRaisesRegex(ValueError, "does not replay"):
            runner.postprocess_outputs(
                wrong_label,
                logits,
                cls_logit,
                native_width=2,
                native_height=2,
            )
        threshold_crossing_mask = dict(good)
        threshold_crossing_mask["pred_mask"] = torch.nextafter(
            good["pred_mask"],
            torch.ones_like(good["pred_mask"]),
        )
        self.assertTrue(
            torch.allclose(
                threshold_crossing_mask["pred_mask"],
                good["pred_mask"],
                atol=2e-6,
                rtol=0,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "segmentation strict >0.5 decision",
        ):
            runner.postprocess_outputs(
                threshold_crossing_mask,
                logits,
                cls_logit,
                native_width=2,
                native_height=2,
            )
        threshold_crossing_label = dict(good)
        threshold_crossing_label["pred_label"] = torch.nextafter(
            torch.tensor(0.5),
            torch.tensor(1.0),
        )
        self.assertTrue(
            torch.allclose(
                threshold_crossing_label["pred_label"],
                good["pred_label"],
                atol=2e-7,
                rtol=0,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "classifier strict >0.5 decision",
        ):
            runner.postprocess_outputs(
                threshold_crossing_label,
                logits,
                cls_logit,
                native_width=2,
                native_height=2,
            )

    def test_tiny_cpu_inference_captures_each_native_head_once(self):
        model = _TinyNfaForward()
        image = np.zeros((3, 512, 512), dtype=np.float32)
        image[0, :256, :256] = 2.0
        processed, peak_bytes, latency_ms = runner.infer_one(
            model,
            torch.device("cpu"),
            image,
            native_width=3,
            native_height=2,
        )

        self.assertEqual(model.seg_decoder.calls, 1)
        self.assertEqual(model.cls_decoder.calls, 1)
        self.assertEqual(len(model.seg_decoder._forward_hooks), 0)
        self.assertEqual(len(model.cls_decoder._forward_hooks), 0)
        assert model.seen_mask is not None
        assert model.seen_label is not None
        self.assertEqual(tuple(model.seen_mask.shape), (1, 1, 512, 512))
        self.assertEqual(tuple(model.seen_label.shape), (1,))
        self.assertEqual(int(torch.count_nonzero(model.seen_mask)), 0)
        self.assertEqual(int(torch.count_nonzero(model.seen_label)), 0)
        self.assertEqual(processed["decoder_logits_128"].shape, (128, 128))
        self.assertEqual(processed["resized_logits_512"].shape, (512, 512))
        self.assertEqual(processed["probability_native"].shape, (2, 3))
        self.assertIsNone(peak_bytes)
        self.assertGreaterEqual(latency_ms, 0.0)

    def test_safe_checkpoint_loader_rejects_non_tensor_model_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good_path = root / "good.pth"
            bad_path = root / "bad.pth"
            unexpected_path = root / "unexpected.pth"
            state = OrderedDict(
                [
                    ("weight", torch.ones((2, 2), dtype=torch.float32)),
                    ("counter", torch.tensor(3, dtype=torch.int64)),
                ]
            )
            torch.save(
                {
                    "model": state,
                    "epoch": 9_999,
                    "args": argparse.Namespace(epoch=9_999),
                },
                good_path,
            )
            torch.save(
                {
                    "model": {
                        **state,
                        "metadata": "not permitted inside model state",
                    }
                },
                bad_path,
            )
            torch.save(
                {
                    "model": state,
                    "unexpected": _UnexpectedCheckpointGlobal(),
                },
                unexpected_path,
            )

            safe_globals_before = list(torch.serialization.get_safe_globals())
            payload, loaded_state, audit = runner._safe_checkpoint_payload(
                good_path
            )
            self.assertEqual(
                torch.serialization.get_safe_globals(),
                safe_globals_before,
            )
            self.assertIn("model", payload)
            self.assertIsInstance(payload["args"], argparse.Namespace)
            self.assertEqual(list(loaded_state), ["weight", "counter"])
            self.assertEqual(audit["state_keys"], 2)
            self.assertEqual(audit["state_elements"], 5)
            self.assertEqual(
                audit["global_safety_audit"],
                {
                    "preflight": (
                        "torch.serialization.get_unsafe_globals_in_checkpoint"
                    ),
                    "unsafe_globals": ["argparse.Namespace"],
                    "allowlisted_globals": ["argparse.Namespace"],
                    "unexpected_globals": [],
                    "allowlist_scope": (
                        "torch.serialization.safe_globals_context"
                    ),
                    "weights_only": True,
                },
            )
            with self.assertRaisesRegex(ValueError, "non-tensor"):
                runner._safe_checkpoint_payload(bad_path)
            with self.assertRaisesRegex(
                ValueError,
                "unexpected unsafe globals",
            ):
                runner._safe_checkpoint_payload(unexpected_path)

    def test_selected_gt_contract_rehashes_shape_pixels_and_real_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_path = root / "mask.png"
            Image.fromarray(
                np.asarray([[0, 255, 0], [255, 255, 0]], dtype=np.uint8),
                mode="L",
            ).save(mask_path)
            selected = _rows(1)
            forged = selected[1]
            forged["gt_mask_path"] = str(mask_path)
            forged["gt_mask_sha256"] = runner.sha256_file(mask_path)

            runner._validate_selected_gt_contract(selected, root)
            np.testing.assert_array_equal(
                runner._load_target(forged, root, 3, 2),
                np.asarray(
                    [[False, True, False], [True, True, False]],
                    dtype=bool,
                ),
            )

            wrong_hash = copy.deepcopy(selected)
            wrong_hash[1]["gt_mask_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner._validate_selected_gt_contract(wrong_hash, root)

            Image.fromarray(
                np.zeros((2, 2), dtype=np.uint8),
                mode="L",
            ).save(mask_path)
            wrong_shape = copy.deepcopy(selected)
            wrong_shape[1]["gt_mask_sha256"] = runner.sha256_file(mask_path)
            with self.assertRaisesRegex(ValueError, "GT shape mismatch"):
                runner._validate_selected_gt_contract(wrong_shape, root)

            Image.fromarray(
                np.asarray([[0, 127, 0], [255, 255, 0]], dtype=np.uint8),
                mode="L",
            ).save(mask_path)
            wrong_pixels = copy.deepcopy(selected)
            wrong_pixels[1]["gt_mask_sha256"] = runner.sha256_file(mask_path)
            with self.assertRaisesRegex(ValueError, "outside 0/255"):
                runner._validate_selected_gt_contract(wrong_pixels, root)

            for field, value, message in (
                ("label", 1, "label 0"),
                ("gt_mask_kind", "exact_diff", "all_zero"),
                ("gt_mask_path", "mask.png", "null GT mask path"),
                ("gt_mask_sha256", "0" * 64, "null GT mask hash"),
            ):
                invalid_real = copy.deepcopy(selected)
                invalid_real[0][field] = value
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, message):
                        runner._validate_selected_gt_contract(
                            invalid_real,
                            root,
                        )

    def test_constructor_shims_three_loads_then_strictly_installs_full_state(self):
        package = types.ModuleType(_TINY_PACKAGE)
        package.__package__ = _TINY_PACKAGE
        model_module = types.ModuleType(_TINY_MODEL_MODULE)
        model_module.__package__ = _TINY_PACKAGE
        dncnn_module = types.ModuleType(f"{_TINY_PACKAGE}.DnCNN")
        dncnn_module.DnCNN = _TinyDnCNN
        modules = {
            _TINY_PACKAGE: package,
            _TINY_MODEL_MODULE: model_module,
            f"{_TINY_PACKAGE}.DnCNN": dncnn_module,
        }
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.object(torch, "load", return_value={}):
                source = _TinyConstructorModel("noise", "b0", "b2")
            state = OrderedDict(
                (key, torch.full_like(value, index + 1))
                for index, (key, value) in enumerate(source.state_dict().items())
            )
            model, audit = runner._construct_model_with_full_state(
                _TinyConstructorModel,
                state,
            )

            self.assertEqual(
                audit["constructor_torch_load_calls"],
                [
                    "__claimforge_noiseprint_init__",
                    "__claimforge_segformer_b0_init__",
                    "__claimforge_segformer_b2_init__",
                ],
            )
            self.assertTrue(audit["strict_full_state_load"])
            self.assertEqual(audit["missing_keys"], [])
            self.assertEqual(audit["unexpected_keys"], [])
            self.assertFalse(audit["noise_extractor_requires_grad"])
            self.assertEqual(list(model.state_dict()), list(state))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key])

            incomplete = OrderedDict(state)
            incomplete.pop(next(iter(incomplete)))
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                runner._construct_model_with_full_state(
                    _TinyConstructorModel,
                    incomplete,
                )

    def _manifest_fixture(self, root: Path, runtime: dict) -> dict:
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
            run_id="fixture",
            condition="mouse_canonical_v1",
            sample_id=None,
            device="cpu",
            seed=42,
            classification_threshold=0.5,
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
                brgen_root=root / "BR-Gen",
                imdlbenco_root=root / "IMDLBenCo",
                checkpoint_path=root / "checkpoint-9999.pth",
                artifact_dir=root / "artifacts",
            )

    def test_manifest_freezes_native_t1_t2_and_rejects_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._manifest_fixture(
                root,
                {"python": {"version": "3.12.3"}, "torch": "2.8"},
            )
            second = self._manifest_fixture(
                root,
                {"python": {"version": "3.12.3"}, "torch": "2.9"},
            )
            path = root / "run-manifest.json"
            runner._write_or_validate_run_manifest(path, first)
            same = json.loads(json.dumps(first))
            same["created_at"] = "a later non-immutable timestamp"
            runner._write_or_validate_run_manifest(path, same)
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                runner._write_or_validate_run_manifest(path, second)

        self.assertTrue(first["model"]["supports_image_level_t1"])
        self.assertTrue(first["model"]["supports_pixel_level_t2"])
        self.assertEqual(
            first["model"]["image_score_source"],
            "official_cls_decoder_sigmoid",
        )
        self.assertTrue(
            first["model"]["checkpoint"]["safe_weights_only_load"]
        )
        self.assertTrue(
            first["model"]["checkpoint"]["strict_full_state_load"]
        )
        self.assertEqual(
            first["inference"]["classification_threshold_operator"],
            ">",
        )
        self.assertEqual(first["inference"]["mask_threshold_operator"], ">")
        self.assertEqual(
            first["artifacts"]["mask_native"]["relation"],
            "probability_native > 0.5",
        )
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def _resume_fixture(
        self,
        root: Path,
    ) -> tuple[list[dict], dict[str, dict], dict[str, Path]]:
        selected = [
            {
                "sample_id": "x",
                "canonical_sha256": "input-sha",
                "width": 3,
                "height": 2,
            }
        ]
        artifact_root = root / "artifacts"
        paths = {
            "raw": artifact_root / "decoder_logits_128" / "x.npy",
            "resized": artifact_root / "resized_logits_512" / "x.npy",
            "model_probability": artifact_root / "probability_512" / "x.npy",
            "native_probability": artifact_root / "probability_native" / "x.npy",
            "mask": artifact_root / "mask_native" / "x.png",
        }
        runner._atomic_save_npy(
            paths["raw"],
            np.zeros((128, 128), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["resized"],
            np.zeros((512, 512), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["model_probability"],
            np.full((512, 512), 0.5, dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["native_probability"],
            np.full((2, 3), 0.5, dtype=np.float32),
        )
        runner._atomic_save_mask(
            paths["mask"],
            np.zeros((2, 3), dtype=bool),
        )
        row = {
            "id": "x",
            "status": "ok",
            "run_manifest_fingerprint": "manifest-sha",
            "image_sha256": "input-sha",
            "raw_logits_model_path": runner.repo_relative(paths["raw"], root),
            "raw_logits_model_sha256": runner.sha256_file(paths["raw"]),
            "raw_logits_model_shape": [128, 128],
            "raw_logits_model_dtype": "float32",
            "resized_logits_model_path": runner.repo_relative(
                paths["resized"],
                root,
            ),
            "resized_logits_model_sha256": runner.sha256_file(paths["resized"]),
            "resized_logits_model_shape": [512, 512],
            "resized_logits_model_dtype": "float32",
            "score_map_model_path": runner.repo_relative(
                paths["model_probability"],
                root,
            ),
            "score_map_model_sha256": runner.sha256_file(
                paths["model_probability"]
            ),
            "score_map_model_shape": [512, 512],
            "score_map_model_dtype": "float32",
            "score_map_path": runner.repo_relative(
                paths["native_probability"],
                root,
            ),
            "score_map_sha256": runner.sha256_file(paths["native_probability"]),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "mask_path": runner.repo_relative(paths["mask"], root),
            "mask_sha256": runner.sha256_file(paths["mask"]),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
        }
        return selected, {"x": row}, paths

    def test_resume_requires_every_artifact_hash_shape_dtype_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, existing, paths = self._resume_fixture(root)
            runner._validate_resume_rows(
                existing,
                selected,
                "manifest-sha",
                repo_root=root,
            )

            wrong_manifest = copy.deepcopy(existing)
            wrong_manifest["x"]["run_manifest_fingerprint"] = "other"
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                runner._validate_resume_rows(
                    wrong_manifest,
                    selected,
                    "manifest-sha",
                    repo_root=root,
                )

            wrong_declared_shape = copy.deepcopy(existing)
            wrong_declared_shape["x"]["raw_logits_model_shape"] = [1, 1]
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                runner._validate_resume_rows(
                    wrong_declared_shape,
                    selected,
                    "manifest-sha",
                    repo_root=root,
                )

            runner._atomic_save_npy(
                paths["native_probability"],
                np.full((2, 3), 0.5, dtype=np.float64),
            )
            wrong_dtype = copy.deepcopy(existing)
            wrong_dtype["x"]["score_map_sha256"] = runner.sha256_file(
                paths["native_probability"]
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                runner._validate_resume_rows(
                    wrong_dtype,
                    selected,
                    "manifest-sha",
                    repo_root=root,
                )

            runner._atomic_save_npy(
                paths["native_probability"],
                np.full((2, 3), 0.5, dtype=np.float32),
            )
            self.assertEqual(
                runner.sha256_file(paths["native_probability"]),
                existing["x"]["score_map_sha256"],
            )
            paths["mask"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing resume"):
                runner._validate_resume_rows(
                    existing,
                    selected,
                    "manifest-sha",
                    repo_root=root,
                )

    def test_non_ok_resume_row_retries_without_requiring_artifacts(self):
        selected = [
            {
                "sample_id": "x",
                "canonical_sha256": "input-sha",
                "width": 3,
                "height": 2,
            }
        ]
        existing = {
            "x": {
                "id": "x",
                "status": "error",
                "run_manifest_fingerprint": "manifest-sha",
                "image_sha256": "input-sha",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            runner._validate_resume_rows(
                existing,
                selected,
                "manifest-sha",
                repo_root=Path(temporary),
            )

    def test_completed_resume_requires_and_preserves_bound_model_load_audit(self):
        checkpoint_sha256 = "c" * 64
        audit = {
            "checkpoint": {
                "global_safety_audit": {
                    "preflight": (
                        "torch.serialization.get_unsafe_globals_in_checkpoint"
                    ),
                    "unsafe_globals": ["argparse.Namespace"],
                    "allowlisted_globals": ["argparse.Namespace"],
                    "unexpected_globals": [],
                    "allowlist_scope": (
                        "torch.serialization.safe_globals_context"
                    ),
                    "weights_only": True,
                },
            },
            "construction": {
                "strict_full_state_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
            },
            "class_module": "_claimforge_brgen_nfa_4ced0e0.nfa_vit",
            "class_name": "NFA_ViT_modify1",
        }
        summary = {
            "run_id": "complete",
            "run_manifest_fingerprint": "manifest-fingerprint",
            "checkpoint_sha256": checkpoint_sha256,
            "model_load_audit": audit,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "complete.summary.json"
            runner.atomic_write_json(summary_path, summary)
            self.assertEqual(
                runner._load_completed_resume_model_audit(
                    summary_path,
                    run_id="complete",
                    manifest_fingerprint="manifest-fingerprint",
                    checkpoint_sha256=checkpoint_sha256,
                ),
                audit,
            )

            for field, expected, message in (
                ("run_id", "other", "run_id mismatch"),
                (
                    "manifest_fingerprint",
                    "other",
                    "run_manifest_fingerprint mismatch",
                ),
                (
                    "checkpoint_sha256",
                    "d" * 64,
                    "checkpoint_sha256 mismatch",
                ),
            ):
                arguments = {
                    "run_id": "complete",
                    "manifest_fingerprint": "manifest-fingerprint",
                    "checkpoint_sha256": checkpoint_sha256,
                }
                arguments[field] = expected
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, message):
                        runner._load_completed_resume_model_audit(
                            summary_path,
                            **arguments,
                        )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "missing prior summary",
            ):
                runner._load_completed_resume_model_audit(
                    root / "missing.summary.json",
                    run_id="complete",
                    manifest_fingerprint="manifest-fingerprint",
                    checkpoint_sha256=checkpoint_sha256,
                )

            invalid_summary = copy.deepcopy(summary)
            invalid_summary["model_load_audit"] = None
            runner.atomic_write_json(summary_path, invalid_summary)
            with self.assertRaisesRegex(ValueError, "no valid model_load_audit"):
                runner._load_completed_resume_model_audit(
                    summary_path,
                    run_id="complete",
                    manifest_fingerprint="manifest-fingerprint",
                    checkpoint_sha256=checkpoint_sha256,
                )


if __name__ == "__main__":
    unittest.main()
