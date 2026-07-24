import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from eval.opensource.analyze_hifi_ifdl_run import (
    CLASSIFICATION_LEVEL_NAMES,
    CLASSIFICATION_LEVEL_SIZES,
    CLASSIFICATION_THRESHOLD,
    EMBEDDING_CHANNELS,
    MASK_THRESHOLD,
    MODEL_INPUT_SIZE,
    Pair,
    _bilinear_align_corners_false,
    _binary_distance_metrics_strict,
    _nearest_resize_mask,
    _pairwise_distance_float32,
    _preprocess_evidence,
    _quintiles,
    _recomputed_t1,
    _verify_upstream_runtime,
    audit_artifacts,
)
from eval.opensource.common import sha256_file


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _save_l(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)


class ArtifactFixture:
    def __init__(self, root: Path):
        self.root = root
        self.center = np.zeros(EMBEDDING_CHANNELS, dtype=np.float32)
        self.radius = 1.2404824495315552
        self.image_dir = root / "images"
        self.artifact_dir = root / "artifacts"
        self.real_image = self.image_dir / "real.png"
        self.forged_image = self.image_dir / "forged.png"
        self.gt_path = self.image_dir / "forged_gt.png"
        real_pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        real_pixels[..., 1] = 80
        forged_pixels = real_pixels.copy()
        forged_pixels[:, 0, 0] = 220
        _save_rgb(self.real_image, real_pixels)
        _save_rgb(self.forged_image, forged_pixels)
        self.target = np.asarray(
            [[True, False], [True, False]],
            dtype=bool,
        )
        _save_l(
            self.gt_path,
            np.where(self.target, 255, 0).astype(np.uint8),
        )
        self.real = self._result("real", self.real_image)
        self.forged = self._result("forged", self.forged_image)
        self.input_row = {
            "sample_id": "forged",
            "task_id": "task",
            "kind": "forged",
            "gt_mask_kind": "exact_diff",
            "gt_mask_path": str(self.gt_path),
            "gt_mask_sha256": sha256_file(self.gt_path),
            "edit_region_xyxy": [0, 0, 1, 2],
        }
        self.pair = Pair(
            task_id="task",
            domain="lodging",
            real=self.real,
            forged=self.forged,
            input_row=self.input_row,
        )

    def _result(self, kind: str, image_path: Path) -> dict:
        embedding = np.zeros(
            (EMBEDDING_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            dtype=np.float32,
        )
        if kind == "forged":
            embedding[0, :, : MODEL_INPUT_SIZE // 2] = np.float32(3.0)
        model_distance = _pairwise_distance_float32(
            embedding,
            self.center,
        )
        native_distance = _bilinear_align_corners_false(
            model_distance,
            width=2,
            height=2,
        )
        target = (
            self.target
            if kind == "forged"
            else np.zeros((2, 2), dtype=bool)
        )
        model_target = _nearest_resize_mask(
            target,
            width=MODEL_INPUT_SIZE,
            height=MODEL_INPUT_SIZE,
        )
        embedding_path = self.artifact_dir / kind / "embedding.npy"
        model_path = self.artifact_dir / kind / "model.npy"
        native_path = self.artifact_dir / kind / "native.npy"
        mask_path = self.artifact_dir / kind / "mask.png"
        _save_npy(embedding_path, embedding)
        _save_npy(model_path, model_distance)
        _save_npy(native_path, native_distance)
        _save_l(
            mask_path,
            np.where(
                native_distance >= MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            ),
        )
        evidence, tensor, native_size = _preprocess_evidence(image_path)
        self.asserted_input = (tensor, native_size)
        hierarchy = {
            name: [0.0] * size
            for name, size in zip(
                CLASSIFICATION_LEVEL_NAMES,
                CLASSIFICATION_LEVEL_SIZES,
                strict=True,
            )
        }
        fine = hierarchy[CLASSIFICATION_LEVEL_NAMES[-1]]
        fine[0 if kind == "real" else 1] = 4.0
        partial = {
            "id": kind,
            "classification_hierarchy_logits": hierarchy,
        }
        replay = _recomputed_t1(partial)
        official_binary = bool(replay["official_binary_decision"])
        benchmark_binary = bool(replay["benchmark_binary_decision"])
        return {
            "id": kind,
            "task_id": "task",
            "pair_rank": 0,
            "domain": "lodging",
            "kind": kind,
            "label": int(kind == "forged"),
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_size": [2, 2],
            "preprocess": evidence,
            "score": float(replay["score"]),
            "score_source": "native_out3_fine_14class_head",
            "score_semantics": (
                "one_minus_softmax_probability_fine_class_0_authentic"
            ),
            "classification_threshold": 0.5,
            "classification_threshold_operator": ">",
            "decision": "forged" if benchmark_binary else "authentic",
            "benchmark_binary_decision": benchmark_binary,
            "official_fine_class_index": int(replay["fine_class_index"]),
            "official_fine_class_name": (
                "authentic" if kind == "real" else "splice"
            ),
            "official_binary_decision": official_binary,
            "official_decision": (
                "forged" if official_binary else "authentic"
            ),
            "official_decision_rule": (
                "argmax_fine_14class_index_not_equal_to_0"
            ),
            "classification_probabilities": [
                float(value) for value in replay["probabilities"]
            ],
            "classification_hierarchy_logits": hierarchy,
            "auxiliary_learned_mask": {
                "shape": [256, 256],
                "dtype": "float32",
                "minimum": 0.1,
                "maximum": 0.9,
                "mean": 0.5,
                "primary_output": False,
                "reason": (
                    "the official public localize API ignores this sigmoid "
                    "mask and thresholds hypersphere distance instead"
                ),
            },
            "mask_feature_model_path": str(embedding_path),
            "mask_feature_model_sha256": sha256_file(embedding_path),
            "mask_feature_model_shape": list(embedding.shape),
            "mask_feature_model_dtype": "float32",
            "distance_map_model_path": str(model_path),
            "distance_map_model_sha256": sha256_file(model_path),
            "distance_map_model_shape": list(model_distance.shape),
            "distance_map_model_dtype": "float32",
            "score_map_path": str(native_path),
            "score_map_sha256": sha256_file(native_path),
            "score_map_shape": list(native_distance.shape),
            "score_map_dtype": "float32",
            "score_map_semantics": (
                "raw_hifi_hypersphere_euclidean_distance"
            ),
            "score_map_native_restore": (
                "bilinear_align_corners_false_from_256_raw_distance_"
                "compatibility_adapter"
            ),
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": list(native_distance.shape),
            "mask_threshold": MASK_THRESHOLD,
            "mask_threshold_operator": ">=",
            "pairwise_distance": {"p": 2.0, "eps": 1e-6},
            "localization": {
                "model_256": _binary_distance_metrics_strict(
                    model_distance,
                    model_target,
                    include_ap=kind == "forged",
                ),
                "native": _binary_distance_metrics_strict(
                    native_distance,
                    target,
                    include_ap=kind == "forged",
                ),
            },
        }


class AnalyzeHiFiIFDLMathTests(unittest.TestCase):
    def test_preprocess_is_imageio_pillow_bicubic_float32_chw(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            pixels = np.asarray(
                [
                    [[0, 64, 255], [255, 128, 0]],
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(pixels, mode="RGB").save(path)
            evidence, tensor, native_size = _preprocess_evidence(path)

        expected = np.ascontiguousarray(
            (
                np.asarray(
                    Image.fromarray(pixels).resize(
                        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                        resample=Image.Resampling.BICUBIC,
                    ),
                    dtype=np.uint8,
                ).astype(np.float32)
                / np.float32(255.0)
            ).transpose(2, 0, 1)
        )
        self.assertEqual(native_size, (2, 3))
        self.assertEqual(tensor.shape, (3, 256, 256))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(evidence["decoder"], "imageio.v2.imread")
        self.assertEqual(
            evidence["resize_interpolation"],
            "Pillow.Image.Resampling.BICUBIC",
        )
        self.assertEqual(len(evidence["tensor_sha256"]), 64)

    def test_pairwise_distance_replays_torch_epsilon_semantics(self):
        embedding = np.zeros(
            (EMBEDDING_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            dtype=np.float32,
        )
        center = np.zeros((EMBEDDING_CHANNELS,), dtype=np.float32)
        distance = _pairwise_distance_float32(embedding, center)
        expected = np.float32(
            np.sqrt(np.float32(EMBEDDING_CHANNELS) * np.float32(1e-12))
        )
        self.assertEqual(distance.shape, (256, 256))
        self.assertEqual(distance.dtype, np.float32)
        np.testing.assert_allclose(distance, expected, rtol=0.0, atol=1e-12)

    def test_pairwise_distance_enforces_18_by_256_squared_embedding(self):
        with self.assertRaisesRegex(ValueError, "embedding shape mismatch"):
            _pairwise_distance_float32(
                np.zeros((18, 2, 2), dtype=np.float32),
                np.zeros(18, dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "epsilon"):
            _pairwise_distance_float32(
                np.zeros((18, 256, 256), dtype=np.float32),
                np.zeros(18, dtype=np.float32),
                eps=0.0,
            )

    def test_native_restore_uses_half_pixel_bilinear_not_binary_resize(self):
        source = np.asarray(
            [[0.0, 4.0], [8.0, 12.0]],
            dtype=np.float32,
        )
        restored = _bilinear_align_corners_false(
            source,
            width=4,
            height=4,
        )
        self.assertEqual(restored.dtype, np.float32)
        self.assertGreater(float(restored[1, 1]), 0.0)
        self.assertLess(float(restored[1, 1]), 12.0)
        np.testing.assert_allclose(
            restored[[0, -1], :][:, [0, -1]],
            source,
            rtol=0.0,
            atol=1e-6,
        )

    def test_model_gt_resize_matches_pillow_nearest_at_noninteger_ratio(self):
        source = np.zeros((1350, 1800), dtype=bool)
        source[311:947, 477:1291] = True
        source[702, 901] = False
        expected = np.asarray(
            Image.fromarray(
                np.where(source, 255, 0).astype(np.uint8),
                mode="L",
            ).resize(
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
        actual = _nearest_resize_mask(
            source,
            width=MODEL_INPUT_SIZE,
            height=MODEL_INPUT_SIZE,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            int(np.count_nonzero(actual)),
            int(np.count_nonzero(expected)),
        )

    def test_mask_threshold_is_inclusive_at_exactly_2_3(self):
        distance = np.asarray([[2.299999, 2.3, 9.0]], dtype=np.float32)
        target = np.asarray([[False, True, True]], dtype=bool)
        metrics = _binary_distance_metrics_strict(
            distance,
            target,
            include_ap=True,
        )
        self.assertEqual(metrics["threshold"], MASK_THRESHOLD)
        self.assertEqual(metrics["threshold_operator"], ">=")
        self.assertEqual(metrics["predicted_positive_pixels"], 2)
        self.assertEqual(metrics["tp"], 2)

    def test_all_positive_forged_mask_has_defined_average_precision(self):
        metrics = _binary_distance_metrics_strict(
            np.asarray([[0.1, 3.0]], dtype=np.float32),
            np.asarray([[True, True]], dtype=bool),
            include_ap=True,
        )
        self.assertEqual(metrics["pixel_ap"], 1.0)

    def test_four_level_logits_recompute_both_decision_rules(self):
        levels = {
            name: [0.0] * size
            for name, size in zip(
                CLASSIFICATION_LEVEL_NAMES,
                CLASSIFICATION_LEVEL_SIZES,
                strict=True,
            )
        }
        levels[CLASSIFICATION_LEVEL_NAMES[-1]][0] = 1.0
        result = {
            "id": "sample",
            "classification_hierarchy_logits": levels,
        }
        replay = _recomputed_t1(result)
        self.assertEqual(replay["fine_class_index"], 0)
        self.assertFalse(replay["official_binary_decision"])
        self.assertGreater(replay["score"], CLASSIFICATION_THRESHOLD)
        self.assertTrue(replay["benchmark_binary_decision"])

    def test_all_four_logit_dimensions_are_enforced(self):
        levels = {
            name: [0.0] * size
            for name, size in zip(
                CLASSIFICATION_LEVEL_NAMES,
                CLASSIFICATION_LEVEL_SIZES,
                strict=True,
            )
        }
        levels[CLASSIFICATION_LEVEL_NAMES[1]].append(0.0)
        with self.assertRaisesRegex(ValueError, "out1_5class"):
            _recomputed_t1(
                {
                    "id": "sample",
                    "classification_hierarchy_logits": levels,
                }
            )

    def test_quintiles_sort_by_edit_fraction(self):
        pairs = []
        for index, positives in enumerate((5, 1, 4, 2, 3)):
            localization = {
                "native": {
                    "pixels": 10,
                    "target_positive_pixels": positives,
                }
            }
            pairs.append(
                Pair(
                    task_id=f"task-{index}",
                    domain="domain",
                    real={},
                    forged={"localization": localization},
                    input_row={},
                )
            )
        chunks = _quintiles(pairs)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0][1][0].edit_fraction, 0.1)
        self.assertEqual(chunks[-1][1][0].edit_fraction, 0.5)


class AnalyzeHiFiIFDLArtifactTests(unittest.TestCase):
    def test_artifact_audit_replays_full_t1_t2_chain_and_box_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            result = audit_artifacts(
                [fixture.pair],
                repo_root=Path(temporary),
                center=fixture.center,
                radius=fixture.radius,
            )
        self.assertEqual(result["artifact_integrity"]["status"], "ok")
        self.assertIn(
            "18x256x256 embedding replays PairwiseDistance p=2 eps=1e-6",
            result["artifact_integrity"]["checks"],
        )
        box = result["box_hit_at_native_mask_threshold_2_3"]
        self.assertEqual(box["threshold_operator"], ">=")
        self.assertEqual(box["any_overlap"]["hits"], 1)

    def test_self_consistent_hash_change_cannot_hide_wrong_native_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            wrong = np.full((2, 2), 2.5, dtype=np.float32)
            native_path = Path(fixture.forged["score_map_path"])
            _save_npy(native_path, wrong)
            fixture.forged["score_map_sha256"] = sha256_file(native_path)
            with self.assertRaisesRegex(
                ValueError,
                "bilinear align_corners=False restore",
            ):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=Path(temporary),
                    center=fixture.center,
                    radius=fixture.radius,
                )

    def test_self_consistent_hash_change_cannot_hide_wrong_pairwise_distance(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            model_path = Path(fixture.forged["distance_map_model_path"])
            with model_path.open("rb") as handle:
                wrong = np.load(handle, allow_pickle=False) + np.float32(0.1)
            _save_npy(model_path, wrong)
            fixture.forged["distance_map_model_sha256"] = sha256_file(
                model_path
            )
            with self.assertRaisesRegex(ValueError, "PairwiseDistance"):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=Path(temporary),
                    center=fixture.center,
                    radius=fixture.radius,
                )

    def test_softmax_tamper_is_rejected_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            fixture.forged["classification_probabilities"][0] += 0.01
            with self.assertRaisesRegex(ValueError, "softmax"):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=Path(temporary),
                    center=fixture.center,
                    radius=fixture.radius,
                )

    def test_official_and_benchmark_decisions_are_both_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary))
            fixture.forged["official_binary_decision"] = False
            with self.assertRaisesRegex(ValueError, "official binary decision"):
                audit_artifacts(
                    [fixture.pair],
                    repo_root=Path(temporary),
                    center=fixture.center,
                    radius=fixture.radius,
                )


class AnalyzeHiFiIFDLPinTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
    ) -> tuple[Path, SimpleNamespace, dict[str, Path]]:
        import torch

        upstream = root / "upstream"
        upstream.mkdir()
        (upstream / "LICENSE").write_text("MIT test\n", encoding="utf-8")
        (upstream / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        initialization = upstream / "init.pth"
        initialization.write_bytes(b"initialization")
        center_path = upstream / "center" / "radius_center.pth"
        center_path.parent.mkdir()
        torch.save(
            {
                "center": torch.zeros(18, dtype=torch.float32),
                "radius": torch.tensor(1.25, dtype=torch.float32),
            },
            center_path,
        )
        subprocess.run(
            ["git", "init", "-q", str(upstream)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "config", "user.email", "test@example"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "add", "."],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "commit", "-qm", "fixture"],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        checkpoints = {
            "feature_extractor": root / "hrnet.pth",
            "hierarchical_localizer_classifier": root / "nlc.pth",
        }
        checkpoints["feature_extractor"].write_bytes(b"hrnet")
        checkpoints["hierarchical_localizer_classifier"].write_bytes(b"nlc")
        checkpoint_pins = {
            role: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in checkpoints.items()
        }
        pins = SimpleNamespace(
            MODEL_SOURCE_COMMIT=commit,
            SOURCE_FILES={
                "LICENSE": sha256_file(upstream / "LICENSE"),
                "source.py": sha256_file(upstream / "source.py"),
            },
            INITIALIZATION_WEIGHT={
                "path": "init.pth",
                "bytes": initialization.stat().st_size,
                "sha256": sha256_file(initialization),
            },
            CHECKPOINTS=checkpoint_pins,
            CENTER={
                "path": "center/radius_center.pth",
                "bytes": center_path.stat().st_size,
                "sha256": sha256_file(center_path),
                "center_shape": [18],
                "radius_value": 1.25,
            },
        )
        return upstream, pins, checkpoints

    def test_source_checkpoint_and_center_pins_are_physically_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, pins, checkpoints = self._fixture(Path(temporary))
            center, radius, result = _verify_upstream_runtime(
                hifi_ifdl_root=upstream,
                pins=pins,
                checkpoint_paths=checkpoints,
            )
        np.testing.assert_array_equal(center, np.zeros(18, dtype=np.float32))
        self.assertEqual(radius, 1.25)
        self.assertEqual(result["source_files_checked"], 2)
        self.assertEqual(result["checkpoint_files_checked"], 2)
        self.assertTrue(result["initialization_weight_checked"])

    def test_checkpoint_bytes_cannot_change_behind_manifest_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, pins, checkpoints = self._fixture(Path(temporary))
            checkpoints["feature_extractor"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _verify_upstream_runtime(
                    hifi_ifdl_root=upstream,
                    pins=pins,
                    checkpoint_paths=checkpoints,
                )


if __name__ == "__main__":
    unittest.main()
