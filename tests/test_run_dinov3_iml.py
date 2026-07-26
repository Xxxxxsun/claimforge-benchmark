from __future__ import annotations

import argparse
import json
import os
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

from eval.opensource import run_dinov3_iml as runner


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
                    "gt_mask_kind": ("file" if kind == "forged" else "all_zero"),
                    "gt_mask_sha256": (
                        f"{rank + 101:064x}" if kind == "forged" else None
                    ),
                    "edit_region_xyxy": [0, 0, 1, 1],
                }
            )
    return rows


class _TinySegHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.avg_pool2d(image[:, :1], kernel_size=16, stride=16)


class _TinyDinoPredict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seg_head = _TinySegHead()
        self.predict_calls = 0

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.predict_calls += 1
        logits = self.seg_head(image)
        logits = F.interpolate(
            logits,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits)


class _TinyCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)
        self.register_buffer(
            "counter",
            torch.tensor(0, dtype=torch.int64),
        )


def _checkpoint_contract(
    state: OrderedDict[str, torch.Tensor],
) -> dict:
    return {
        "top_level_keys": [
            "model",
            "optimizer",
            "epoch",
            "scaler",
            "args",
        ],
        "epoch": 48,
        "state_keys": len(state),
        "state_elements": sum(int(value.numel()) for value in state.values()),
        "tensor_bytes": sum(
            int(value.numel()) * int(value.element_size()) for value in state.values()
        ),
        "state_dtypes": dict(Counter(str(value.dtype) for value in state.values())),
        "expected_args": {
            "model": "DINOv3ForensicsLoRA",
            "image_size": 512,
            "lr": 0.0003,
        },
        "critical_state_shapes": {
            "projection.weight": [1, 2],
            "counter": [],
        },
    }


class RunDinoV3IMLTest(unittest.TestCase):
    def test_frozen_release_constants(self):
        self.assertEqual(
            runner.MODEL_SOURCE_COMMIT,
            "ba45b0a203c698b36fe2b0e658bb49ebbb1163cc",
        )
        self.assertEqual(
            runner.DINOV3_SOURCE_COMMIT,
            "31703e4cbf1ccb7c4a72daa1350405f86754b6d1",
        )
        self.assertEqual(runner.CHECKPOINT["epoch"], 48)
        self.assertEqual(runner.CHECKPOINT["bytes"], 1_321_705_819)
        self.assertEqual(
            runner.CHECKPOINT["sha256"],
            "01f23401e048f706ea0e63fb0429ddef80db3197ac0f5707bd584a8b056177fa",
        )
        self.assertEqual(runner.CHECKPOINT["state_keys"], 432)
        self.assertEqual(
            runner.CHECKPOINT["state_elements"],
            312_275_987,
        )
        self.assertEqual(
            runner.CHECKPOINT["trainable_parameters"],
            9_046_529,
        )
        self.assertEqual(runner.MODEL_INPUT_SIZE, 512)
        self.assertEqual(runner.INTERNAL_LOGIT_SIZE, 32)
        self.assertEqual(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            ":4096:8",
        )

    def test_select_inputs_supports_complete_pairs_and_explicit_preflight(self):
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
            runner.select_inputs(
                rows,
                pair_limit=1,
                sample_id="forged-0",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.select_inputs(
                rows,
                pair_limit=None,
                sample_id="missing",
            )
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            runner.select_inputs(rows[:-1], pair_limit=None)

    def test_preprocess_matches_author_pillow_float32_path(self):
        rgb = np.asarray(
            [
                [[255, 0, 1], [0, 255, 2], [0, 0, 255]],
                [[12, 34, 56], [127, 128, 129], [253, 254, 255]],
            ],
            dtype=np.uint8,
        )
        resized = Image.fromarray(rgb, mode="RGB").resize(
            (512, 512),
            resample=Image.Resampling.BILINEAR,
        )
        expected = np.array(resized, dtype=np.float32) / 255.0
        expected = (
            expected - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected = np.ascontiguousarray(expected.astype(np.float32).transpose(2, 0, 1))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            tensor, native_size, metadata = runner.preprocess_image(path)

        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(native_size, (3, 2))
        self.assertEqual(metadata["protocol"], runner.PREPROCESS_PROTOCOL)
        self.assertEqual(metadata["decoder_format"], "PNG")
        self.assertEqual(metadata["native_size_wh"], [3, 2])
        self.assertEqual(metadata["model_size_wh"], [512, 512])
        self.assertEqual(
            metadata["resize_interpolation"],
            "Pillow.Image.Resampling.BILINEAR",
        )
        self.assertEqual(
            metadata["tensor_sha256"],
            runner._sha256_array(tensor),
        )

    def test_first_canonical_tensor_has_frozen_hash(self):
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
            "0eaabc875a4abe662f520ae854609027b4f1c9fc54aa2df8a0bb3da4b56cd20a",
        )
        self.assertEqual(
            metadata["tensor_sha256"],
            "0eaabc875a4abe662f520ae854609027b4f1c9fc54aa2df8a0bb3da4b56cd20a",
        )

    def test_strict_probability_threshold_treats_half_as_negative(self):
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

    def test_postprocess_replays_32_to_512_sigmoid_then_probability_native(self):
        base = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]]]],
            dtype=torch.float32,
        )
        logits_32 = F.interpolate(
            base,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        logits_512 = F.interpolate(
            logits_32,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        probability = torch.sigmoid(logits_512)
        expected_native = F.interpolate(
            probability,
            size=(2, 3),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        processed = runner.postprocess_outputs(
            probability,
            logits_32,
            native_width=3,
            native_height=2,
        )

        np.testing.assert_array_equal(
            processed["raw_logits_model_32"],
            logits_32[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["raw_logits_model_512"],
            logits_512[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_model_512"],
            probability[0, 0].numpy(),
        )
        np.testing.assert_array_equal(
            processed["probability_native"],
            expected_native,
        )
        wrong_native = torch.sigmoid(
            F.interpolate(
                logits_512,
                size=(2, 3),
                mode="bilinear",
                align_corners=False,
            )
        )[0, 0].numpy()
        self.assertFalse(np.array_equal(processed["probability_native"], wrong_native))

    def test_postprocess_rejects_shape_and_probability_chain_drift(self):
        logits = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        probability = torch.full(
            (1, 1, 512, 512),
            0.5,
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "logit shape"):
            runner.postprocess_outputs(
                probability,
                logits[:, :, :-1],
                native_width=2,
                native_height=2,
            )
        with self.assertRaisesRegex(ValueError, "probability shape"):
            runner.postprocess_outputs(
                probability[:, :, :-1],
                logits,
                native_width=2,
                native_height=2,
            )
        with self.assertRaisesRegex(ValueError, "does not match sigmoid"):
            runner.postprocess_outputs(
                probability + 0.1,
                logits,
                native_width=2,
                native_height=2,
            )

    def test_infer_one_hooks_seg_head_once_and_removes_hook(self):
        model = _TinyDinoPredict()
        image = np.zeros((3, 512, 512), dtype=np.float32)
        image[0, :256, :256] = 2.0
        self.assertEqual(len(model.seg_head._forward_hooks), 0)
        processed, peak, latency = runner.infer_one(
            model,
            torch.device("cpu"),
            image,
            native_width=3,
            native_height=2,
        )

        self.assertEqual(model.predict_calls, 1)
        self.assertEqual(model.seg_head.calls, 1)
        self.assertEqual(len(model.seg_head._forward_hooks), 0)
        self.assertEqual(processed["raw_logits_model_32"].shape, (32, 32))
        self.assertEqual(
            processed["probability_model_512"].shape,
            (512, 512),
        )
        self.assertEqual(processed["probability_native"].shape, (2, 3))
        self.assertEqual(peak, 0)
        self.assertGreaterEqual(latency, 0.0)

    def test_model_space_target_uses_pillow_nearest(self):
        target = np.asarray(
            [[True, False], [False, False]],
            dtype=bool,
        )
        resized = runner.model_space_target(target)
        expected = (
            np.asarray(
                Image.fromarray(target.astype(np.uint8), mode="L").resize(
                    (512, 512),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            > 0
        )
        np.testing.assert_array_equal(resized, expected)
        self.assertEqual(resized.dtype, np.bool_)

    def test_artifact_writers_are_lossless_and_strict(self):
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
                self.assertEqual(opened.format, "PNG")
                mask = np.asarray(opened, dtype=np.uint8)
        np.testing.assert_array_equal(loaded, array)
        np.testing.assert_array_equal(
            mask,
            np.asarray([[0, 255]], dtype=np.uint8),
        )

    def test_checkpoint_loader_safe_schema_and_strict_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good_path = root / "good.pth"
            bad_args_path = root / "bad-args.pth"
            source = _TinyCheckpointModel()
            state = source.state_dict()
            payload = {
                "model": state,
                "optimizer": {},
                "epoch": 48,
                "scaler": None,
                "args": argparse.Namespace(
                    model="DINOv3ForensicsLoRA",
                    image_size=512,
                    lr=0.0003,
                ),
            }
            torch.save(payload, good_path)
            bad_args = dict(payload)
            bad_args["args"] = argparse.Namespace(
                model="DINOv3ForensicsLoRA",
                image_size=256,
                lr=0.0003,
            )
            torch.save(bad_args, bad_args_path)
            contract = _checkpoint_contract(state)

            loaded = runner._load_checkpoint_payload(
                path=good_path,
                contract=contract,
                label="fixture",
            )
            target = _TinyCheckpointModel()
            runner._strict_load_checkpoint_state(
                module=target,
                payload=loaded,
                label="fixture",
            )
            for expected, actual in zip(
                source.state_dict().values(),
                target.state_dict().values(),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
            with self.assertRaisesRegex(ValueError, "args.image_size"):
                runner._load_checkpoint_payload(
                    path=bad_args_path,
                    contract=contract,
                    label="fixture",
                )

    def test_constructor_patch_builds_architecture_without_weights_once(self):
        calls: list[bool] = []
        backbone = object()

        def factory(*, pretrained):
            calls.append(pretrained)
            return backbone

        class FakeAuthorModel:
            def __init__(
                self,
                dinov3_repo_path,
                dinov3_weights_path,
                dinov3_model_type,
                image_size,
                edge_lambda,
                lora_rank,
                lora_alpha,
            ):
                self.arguments = {
                    "repo": dinov3_repo_path,
                    "weights": dinov3_weights_path,
                    "model": dinov3_model_type,
                    "image_size": image_size,
                    "edge_lambda": edge_lambda,
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                }
                self.backbone = torch.hub.load(
                    dinov3_repo_path,
                    dinov3_model_type,
                    source="local",
                    weights=dinov3_weights_path,
                )

        original = torch.hub.load
        with tempfile.TemporaryDirectory() as temporary:
            model = runner._construct_author_model(
                model_class=FakeAuthorModel,
                dinov3_root=Path(temporary),
                backbone_factory=factory,
            )
        self.assertIs(torch.hub.load, original)
        self.assertIs(model.backbone, backbone)
        self.assertEqual(calls, [False])
        self.assertEqual(
            model.arguments["weights"],
            runner.ARCHITECTURE_ONLY_WEIGHTS_SENTINEL,
        )
        self.assertEqual(model.arguments["model"], "dinov3_vitl16")
        self.assertEqual(model.arguments["rank"], 32)
        self.assertEqual(model.arguments["alpha"], 64.0)

    def test_constructor_patch_rejects_signature_drift_and_download(self):
        class WrongWeights:
            def __init__(self, **kwargs):
                torch.hub.load(
                    kwargs["dinov3_repo_path"],
                    kwargs["dinov3_model_type"],
                    source="local",
                    weights="unregistered",
                )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "signature changed"):
                runner._construct_author_model(
                    model_class=WrongWeights,
                    dinov3_root=Path(temporary),
                    backbone_factory=lambda **_kwargs: object(),
                )

        class CorrectModel:
            def __init__(self, **kwargs):
                self.backbone = torch.hub.load(
                    kwargs["dinov3_repo_path"],
                    kwargs["dinov3_model_type"],
                    source="local",
                    weights=kwargs["dinov3_weights_path"],
                )

        def downloading_factory(*, pretrained):
            self.assertFalse(pretrained)
            return torch.hub.load_state_dict_from_url("https://invalid.test/x")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "weight download"):
                runner._construct_author_model(
                    model_class=CorrectModel,
                    dinov3_root=Path(temporary),
                    backbone_factory=downloading_factory,
                )

    def test_cached_modules_outside_pinned_sources_are_rejected(self):
        fake = types.ModuleType(runner.AUTHOR_MODULE_ALIAS)
        fake.__file__ = "/tmp/not-pinned/model.py"
        with mock.patch.dict(
            sys.modules,
            {runner.AUTHOR_MODULE_ALIAS: fake},
        ):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runner._require_cached_module_origin(
                    runner.AUTHOR_MODULE_ALIAS,
                    Path("/pinned/model.py"),
                )

        fake_dino = types.ModuleType("dinov3.layers.fake")
        fake_dino.__file__ = "/tmp/not-pinned/fake.py"
        with mock.patch.dict(
            sys.modules,
            {"dinov3.layers.fake": fake_dino},
        ):
            with self.assertRaisesRegex(ValueError, "escapes pinned"):
                runner._require_module_tree_origin(
                    "dinov3",
                    Path("/pinned/dinov3"),
                )

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
            sample_id=None,
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
                dinov3_iml_root=root / "author",
                dinov3_root=root / "meta",
                checkpoint_path=root / "checkpoint-48.pth",
                artifact_dir=root / "artifacts",
            )

    def test_manifest_freezes_full_state_t2_and_probability_chain(self):
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
        architecture = first["model"]["dinov3_architecture_source"]
        self.assertFalse(architecture["pretrained"])
        self.assertFalse(architecture["separate_backbone_weights_loaded"])
        self.assertEqual(architecture["role"], "architecture_only")
        checkpoint = first["model"]["checkpoint"]
        self.assertTrue(checkpoint["full_state_includes_backbone_lora_and_seg_head"])
        self.assertFalse(checkpoint["separate_backbone_weights_required"])
        self.assertFalse(first["model"]["constructor"]["lora_merged"])
        self.assertEqual(
            first["inference"]["native_compatibility_adapter"]["source"],
            "official_model_512_probability_not_logits",
        )
        self.assertEqual(
            first["artifacts"]["score_maps_native"]["restore"],
            runner.NATIVE_RESTORE,
        )
        self.assertEqual(
            first["metrics"]["t1_policy"],
            "unsupported_no_derived_image_score",
        )
        self.assertTrue(
            any(
                item.get("role") == "shared_512_native_t2_numerical_kernel"
                for item in first["adapter_contract"]
            )
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
            "raw": artifact_dir / "raw_logits_model_32" / "x.npy",
            "resized": artifact_dir / "raw_logits_model_512" / "x.npy",
            "model": artifact_dir / "score_maps_model_512" / "x.npy",
            "native": artifact_dir / "score_maps_native" / "x.npy",
            "mask": artifact_dir / "masks_native" / "x.png",
        }
        runner._atomic_save_npy(
            paths["raw"],
            np.zeros((32, 32), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["resized"],
            np.zeros((512, 512), dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["model"],
            np.full((512, 512), 0.5, dtype=np.float32),
        )
        runner._atomic_save_npy(
            paths["native"],
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
            "checkpoint_sha256": runner.CHECKPOINT["sha256"],
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">",
            "raw_logits_capture": runner.LOGIT_CAPTURE,
            "resized_logits_derivation": runner.MODEL_LOGIT_RESIZE,
            "score_map_native_restore": runner.NATIVE_RESTORE,
            "score_map_native_source": "official_model_512_probability",
            "raw_logits_model_path": runner.repo_relative(
                paths["raw"],
                root,
            ),
            "raw_logits_model_sha256": runner.sha256_file(paths["raw"]),
            "raw_logits_model_shape": [32, 32],
            "raw_logits_model_dtype": "float32",
            "resized_logits_model_path": runner.repo_relative(
                paths["resized"],
                root,
            ),
            "resized_logits_model_sha256": runner.sha256_file(paths["resized"]),
            "resized_logits_model_shape": [512, 512],
            "resized_logits_model_dtype": "float32",
            "score_map_model_path": runner.repo_relative(
                paths["model"],
                root,
            ),
            "score_map_model_sha256": runner.sha256_file(paths["model"]),
            "score_map_model_shape": [512, 512],
            "score_map_model_dtype": "float32",
            "score_map_path": runner.repo_relative(paths["native"], root),
            "score_map_sha256": runner.sha256_file(paths["native"]),
            "score_map_shape": [2, 3],
            "score_map_dtype": "float32",
            "mask_path": runner.repo_relative(paths["mask"], root),
            "mask_sha256": runner.sha256_file(paths["mask"]),
            "mask_shape": [2, 3],
            "mask_dtype": "uint8",
        }
        return selected, artifact_dir, {"x": row}

    def test_resume_validates_all_artifacts_and_recursively_forbids_t1(self):
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
            with_t1["x"]["audit"] = {"nested": {"classification": None}}
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
            with self.assertRaisesRegex(
                ValueError,
                "modified native probability",
            ):
                runner._validate_resume_rows(
                    latest,
                    selected,
                    "run-sha",
                    repo_root=root,
                    artifact_dir=artifact_dir,
                )

    def test_non_ok_resume_row_needs_identity_but_not_artifacts(self):
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
                "checkpoint_sha256": runner.CHECKPOINT["sha256"],
                "valid_for_t1": False,
                "valid_for_t2": True,
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
