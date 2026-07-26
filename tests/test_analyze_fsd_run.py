import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource import analyze_fsd_run as analyzer
from eval.opensource.analyze_fsd_run import (
    DESCRIPTOR_DIMENSION,
    DetectionPair,
    ReplayRuntime,
    _audit_score_fields,
    _diagnostic_slices,
    _kernel_replay_runtime,
    _latest_by_id,
    _load_descriptor,
    _manifest_fingerprint,
    _official_preprocess_geometry,
    _preprocess_and_visibility_audit,
    _reject_localization_contract,
    _visible_gt_evidence,
    audit_artifacts,
    audit_prefix_reproducibility,
    recompute_summary,
    summarize_result_history,
)
from eval.opensource.common import atomic_write_jsonl, sha256_file
from eval.opensource.whole_image_metrics import summarize_whole_image_results


OFFICIAL_CONFIG = {
    "fre": {
        "in_channels": 1,
        "out_channels": 8,
        "kernel_size": 15,
        "weights_file": "fre.pt",
    },
    "fsd": {
        "kernel_size": 11,
        "num_scales": 3,
        "max_size": 1024,
        "resize_mode": "resize_and_crop",
    },
    "gmm": {
        "n_components": 5,
        "covariance_type": "tied",
        "weights_file": "gmm.pt",
    },
    "transforms": {"weights_file": "fsd_transforms.pt"},
    "scoring": {
        "train_mean": -42.25325127289017,
        "train_std": 706.0556010649537,
        "default_threshold": -2.0,
    },
    "attribution": {
        "weights_file": "attribution_transforms.pt",
        "source_gmms_file": "source_gmms.pt",
    },
}


def _save_rgb(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 1] = 80
    Image.fromarray(pixels, mode="RGB").save(path)


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)


def _save_descriptor(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _score_row_fields(raw: float) -> dict:
    mean = OFFICIAL_CONFIG["scoring"]["train_mean"]
    std = OFFICIAL_CONFIG["scoring"]["train_std"]
    z = (raw - mean) / std
    score = -z
    decision = score > 2.0
    return {
        "score": score,
        "score_semantics": "negative_released_FSD_z_score",
        "raw_likelihood": raw,
        "released_z_score": z,
        "ai_score": score,
        "released_is_fake": z < -2.0,
        "released_threshold": -2.0,
        "released_threshold_operator": "<",
        "classification_decision": decision,
        "classification_threshold": 2.0,
        "classification_threshold_operator": ">",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "classification": {
            "score": score,
            "raw_likelihood": raw,
            "released_z_score": z,
            "decision": decision,
            "threshold": 2.0,
            "threshold_operator": ">",
            "semantics": "higher_is_more_AI_negative_released_z",
        },
        "t1": {
            "score": score,
            "raw_likelihood": raw,
            "released_z_score": z,
            "threshold": 2.0,
            "threshold_operator": ">",
            "decision": decision,
            "policy": "released_FSD_whole_image_score_sign_inverted",
        },
        "manual_replay": {
            "raw_likelihood": raw,
            "released_z_score": z,
            "ai_score": score,
            "released_is_fake": z < -2.0,
            "classification_decision": decision,
            "official_raw_exact_match": True,
            "official_z_exact_match": True,
            "compute_fsd_calls": 1,
        },
    }


def _result_row(
    row_id: str,
    *,
    task_id: str,
    kind: str,
    score: float,
    visibility: str = "full",
    fraction: float = 1.0,
) -> dict:
    return {
        "id": row_id,
        "status": "ok",
        "task_id": task_id,
        "pair_rank": 0,
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": "lodging",
        "ai_score": score,
        "edit_visibility": visibility,
        "edit_visible_gt_fraction": fraction,
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 0,
    }


class GeometryAndVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _forged(self, positive_xy: list[tuple[int, int]]) -> tuple[dict, dict]:
        width, height = 2000, 1000
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in positive_xy:
            mask[y, x] = 255
        path = self.root / "gt.png"
        _save_mask(path, mask)
        forged = {
            "sample_id": "forged",
            "gt_mask_path": str(path),
            "gt_mask_sha256": sha256_file(path),
            "edit_region_xyxy": [850, 450, 1050, 550],
        }
        geometry = _official_preprocess_geometry(
            width=width,
            height=height,
            config=OFFICIAL_CONFIG,
        )
        return forged, geometry

    def test_official_geometry_replays_border_round_resize_and_center_crop(self):
        geometry = _official_preprocess_geometry(
            width=2000,
            height=1000,
            config=OFFICIAL_CONFIG,
        )
        self.assertEqual(geometry["fre"]["border_each_side"], 7)
        self.assertEqual(geometry["fre"]["post_trim_size"], [1986, 986])
        self.assertEqual(
            geometry["resize"]["destination_size"],
            [round(1986 * 1024 / 986), 1024],
        )
        self.assertEqual(geometry["center_crop"]["size"], [1024, 1024])
        self.assertEqual(geometry["scales"]["sizes"], [[1024, 1024], [512, 512], [256, 256]])
        crop = geometry["effective_native_crop_xyxy"]
        self.assertGreater(crop[0], 400)
        self.assertLess(crop[2], 1600)

    def test_exact_gt_pixel_center_visibility_none_partial_full(self):
        cases = (
            ([(100, 500)], "none", 0.0),
            ([(100, 500), (1000, 500)], "partial", 0.5),
            ([(900, 500), (1000, 500)], "full", 1.0),
        )
        for positives, category, fraction in cases:
            with self.subTest(category=category):
                forged, geometry = self._forged(positives)
                result = _visible_gt_evidence(
                    forged_input=forged,
                    geometry=geometry,
                    repo_root=self.root,
                )
                self.assertEqual(result["category"], category)
                self.assertEqual(result["visible_fraction"], fraction)
                self.assertIn("align_corners_false", result["basis"])

    def test_visibility_rejects_nonbinary_or_wrong_shape_gt(self):
        geometry = _official_preprocess_geometry(
            width=2000,
            height=1000,
            config=OFFICIAL_CONFIG,
        )
        wrong = np.zeros((10, 10), dtype=np.uint8)
        wrong[0, 0] = 1
        path = self.root / "wrong.png"
        _save_mask(path, wrong)
        forged = {
            "gt_mask_path": str(path),
            "gt_mask_sha256": sha256_file(path),
        }
        with self.assertRaisesRegex(ValueError, "shape"):
            _visible_gt_evidence(
                forged_input=forged,
                geometry=geometry,
                repo_root=self.root,
            )

    def test_preprocess_and_pair_level_visibility_are_exact(self):
        forged, geometry = self._forged([(900, 500), (1000, 500)])
        row = {
            "id": "real",
            "preprocess": geometry,
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "gt": _visible_gt_evidence(
                    forged_input=forged,
                    geometry=geometry,
                    repo_root=self.root,
                ),
                "edit_box": analyzer._edit_box_visibility(
                    forged["edit_region_xyxy"],
                    geometry["effective_native_crop_xyxy"],
                ),
            },
        }
        canonical = {"width": 2000, "height": 1000}
        result = _preprocess_and_visibility_audit(
            row=row,
            canonical=canonical,
            forged_input=forged,
            config=OFFICIAL_CONFIG,
            repo_root=self.root,
        )
        self.assertEqual(result["category"], "full")

        broken = copy.deepcopy(row)
        broken["edit_visible_gt_fraction"] = np.nextafter(1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "visible GT fraction mismatch"):
            _preprocess_and_visibility_audit(
                row=broken,
                canonical=canonical,
                forged_input=forged,
                config=OFFICIAL_CONFIG,
                repo_root=self.root,
            )


class DescriptorAndScoreReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_descriptor_requires_exact_shape_dtype_and_file_hash(self):
        path = self.root / "descriptor.npy"
        value = np.arange(DESCRIPTOR_DIMENSION, dtype=np.float64)
        _save_descriptor(path, value)
        row = {
            "id": "sample",
            "raw_descriptor_path": str(path),
            "raw_descriptor_sha256": sha256_file(path),
            "raw_descriptor_shape": [DESCRIPTOR_DIMENSION],
            "raw_descriptor_dtype": "float64",
            "raw_descriptor_array_sha256": hashlib.sha256(
                value.tobytes(order="C")
            ).hexdigest(),
        }
        loaded = _load_descriptor(row, self.root)
        np.testing.assert_array_equal(loaded, value)

        wrong_dtype = self.root / "float32.npy"
        _save_descriptor(
            wrong_dtype,
            np.zeros(DESCRIPTOR_DIMENSION, dtype=np.float32),
        )
        broken = {
            **row,
            "raw_descriptor_path": str(wrong_dtype),
            "raw_descriptor_sha256": sha256_file(wrong_dtype),
        }
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            _load_descriptor(broken, self.root)

        broken = dict(row)
        broken["raw_descriptor_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            _load_descriptor(broken, self.root)

    def test_released_boundary_uses_strict_greater_than(self):
        mean = OFFICIAL_CONFIG["scoring"]["train_mean"]
        std = OFFICIAL_CONFIG["scoring"]["train_std"]
        raw = mean - 2.0 * std
        row = _score_row_fields(raw)
        result = _audit_score_fields(
            row=row,
            raw_likelihood=raw,
            train_mean=mean,
            train_std=std,
            z_threshold=-2.0,
            ai_threshold=2.0,
        )
        self.assertEqual(result["ai_score"], 2.0)
        self.assertFalse(result["classification_decision"])
        self.assertFalse(result["released_is_fake"])

        broken = copy.deepcopy(row)
        broken["classification_decision"] = True
        with self.assertRaisesRegex(ValueError, "classification decision"):
            _audit_score_fields(
                row=broken,
                raw_likelihood=raw,
                train_mean=mean,
                train_std=std,
                z_threshold=-2.0,
                ai_threshold=2.0,
            )

    def test_nested_replay_aliases_cannot_disagree(self):
        raw = -100.0
        row = _score_row_fields(raw)
        row["manual_replay"]["ai_score"] += 1e-6
        with self.assertRaisesRegex(ValueError, "replay ai_score mismatch"):
            _audit_score_fields(
                row=row,
                raw_likelihood=raw,
                train_mean=OFFICIAL_CONFIG["scoring"]["train_mean"],
                train_std=OFFICIAL_CONFIG["scoring"]["train_std"],
                z_threshold=-2.0,
                ai_threshold=2.0,
            )

    def test_exact_runner_runtime_contract_replays_and_version_drift_closes(self):
        from eval.opensource import run_fsd

        run_fsd.configure_determinism(17)
        manifest = {
            "protocol": {"seed": 17},
            "runtime_contract": run_fsd._runtime_contract("cpu"),
        }
        replay = _kernel_replay_runtime(manifest)
        self.assertEqual(replay.recorded_device, "cpu")

        broken = copy.deepcopy(manifest)
        broken["runtime_contract"]["packages"]["torch"]["version"] = "0.0"
        with self.assertRaisesRegex(RuntimeError, "exact torch version differs"):
            _kernel_replay_runtime(broken)


class ScopeAndHistoryTests(unittest.TestCase):
    def test_localization_and_s_joint_must_be_absent_or_explicit_na(self):
        _reject_localization_contract(
            {
                "valid_for_t2": False,
                "t2": "N/A",
                "s_joint": {"supported": False, "status": "not_applicable"},
                "preprocess": {"center_crop": {"size": [1024, 1024]}},
            },
            label="record",
        )
        for value, message in (
            ({"localization": {}}, "fabricates localization"),
            ({"t2": {"iou": 0.5}}, "fabricates localization"),
            ({"s_joint": 0.2}, "fabricates S_joint"),
            ({"nested": {"mask_path": "mask.png"}}, "fabricates localization"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    _reject_localization_contract(value, label="record")

    def test_physical_history_preserves_recovery_evidence(self):
        rows = [
            {"id": "a", "status": "error"},
            {"id": "b", "status": "ok"},
            {"id": "a", "status": "ok"},
        ]
        summary = summarize_result_history(rows)
        self.assertEqual(summary["physical_rows"], 3)
        self.assertEqual(summary["unique_ids"], 2)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["recovered_ids"], ["a"])
        self.assertEqual(_latest_by_id(rows)["a"]["status"], "ok")

    def test_manifest_fingerprint_excludes_only_declared_dynamic_fields(self):
        immutable = {"schema_version": "v", "run_id": "run", "model": {"x": 1}}
        fingerprint = hashlib.sha256(
            json.dumps(
                immutable,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            **immutable,
            "fingerprint": fingerprint,
            "created_at": "now",
            "adapter": {"path": "adapter.py"},
            "environment": {"torch": "copy"},
        }
        self.assertEqual(_manifest_fingerprint(manifest), fingerprint)
        changed = copy.deepcopy(manifest)
        changed["model"]["x"] = 2
        self.assertNotEqual(_manifest_fingerprint(changed), fingerprint)


class SummaryAndPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _metrics_rows(self) -> tuple[list[dict], list[dict]]:
        inputs = [
            {
                "sample_id": "real",
                "task_id": "task",
                "kind": "real",
                "label": 0,
                "domain": "lodging",
            },
            {
                "sample_id": "forged",
                "task_id": "task",
                "kind": "forged",
                "label": 1,
                "domain": "lodging",
            },
        ]
        rows = [
            _result_row(
                "real",
                task_id="task",
                kind="real",
                score=0.0,
                visibility="partial",
                fraction=0.5,
            ),
            _result_row(
                "forged",
                task_id="task",
                kind="forged",
                score=3.0,
                visibility="partial",
                fraction=0.5,
            ),
        ]
        return inputs, rows

    def test_whole_image_summary_is_recomputed_and_mismatch_rejected(self):
        inputs, rows = self._metrics_rows()
        expected = summarize_whole_image_results(
            rows,
            inputs,
            threshold=2.0,
            bootstrap_samples=10,
            seed=7,
        )
        recomputed, pairs = recompute_summary(
            latest=_latest_by_id(rows),
            input_rows=inputs,
            manifest={
                "protocol": {
                    "classification": {"ai_score_threshold": 2.0},
                    "bootstrap_samples": 10,
                    "seed": 7,
                }
            },
            recorded_summary=expected,
        )
        self.assertEqual(recomputed, expected)
        self.assertEqual(len(pairs), 1)

        retry_error = {
            key: value
            for key, value in rows[0].items()
            if key not in {"ai_score", "latency_ms", "peak_cuda_memory_bytes"}
        }
        retry_error["status"] = "error"
        history = [retry_error, *rows]
        history_summary = summarize_whole_image_results(
            history,
            inputs,
            threshold=2.0,
            bootstrap_samples=10,
            seed=7,
        )
        replayed_history, _ = recompute_summary(
            latest=_latest_by_id(history),
            result_rows=history,
            input_rows=inputs,
            manifest={
                "protocol": {
                    "classification": {"ai_score_threshold": 2.0},
                    "bootstrap_samples": 10,
                    "seed": 7,
                }
            },
            recorded_summary=history_summary,
        )
        self.assertEqual(
            replayed_history["coverage"]["physical_result_rows"],
            3,
        )

        broken = copy.deepcopy(expected)
        broken["detection"]["tp"] += 1
        with self.assertRaisesRegex(ValueError, r"summary.detection.tp mismatch"):
            recompute_summary(
                latest=_latest_by_id(rows),
                input_rows=inputs,
                manifest={
                    "protocol": {
                        "classification": {"ai_score_threshold": 2.0},
                        "bootstrap_samples": 10,
                        "seed": 7,
                    }
                },
                recorded_summary=broken,
            )

    def test_diagnostics_always_create_none_partial_full_slices(self):
        _, rows = self._metrics_rows()
        pair = DetectionPair(
            task_id="task",
            domain="lodging",
            real=rows[0],
            forged=rows[1],
            forged_input={},
        )
        slices = _diagnostic_slices([pair], iterations=5, seed=2)
        self.assertEqual(set(slices), {"none", "partial", "full"})
        self.assertEqual(slices["partial"]["pairs"], 1)
        self.assertEqual(slices["none"]["status"], "empty_slice")

    def test_prefix_requires_independent_identity_and_descriptor_paths(self):
        descriptor = np.arange(DESCRIPTOR_DIMENSION, dtype=np.float64)
        full_path = self.root / "full" / "a.npy"
        prefix_path = self.root / "prefix" / "a.npy"
        _save_descriptor(full_path, descriptor)
        _save_descriptor(prefix_path, descriptor)
        digest = sha256_file(full_path)
        ordered = [{"sample_id": "a"}]
        common_model = {
            "name": "FSD",
            "slug": "fsd_v1_2_0_official",
            "repo_url": "repo",
            "source_commit": "c" * 40,
            "release_tag": "v1.2.0",
            "weights_bundle_sha256": "b" * 64,
        }

        def manifest(run_id: str) -> dict:
            immutable = {
                "schema_version": "opensource_run_manifest_v1",
                "run_id": run_id,
                "selection": {"rows": ordered},
                "model": common_model,
                "runtime_contract": {"same": True},
                "protocol": {"same": True},
            }
            return {
                **immutable,
                "fingerprint": hashlib.sha256(
                    json.dumps(
                        immutable,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "created_at": "now",
            }

        full_manifest = manifest("full")
        prefix_manifest = manifest("prefix")
        shared = {
            "id": "a",
            "status": "ok",
            "raw_descriptor_sha256": digest,
            "raw_descriptor_shape": [DESCRIPTOR_DIMENSION],
            "raw_descriptor_dtype": "float64",
            "raw_likelihood": 1.0,
            "released_z_score": 2.0,
            "ai_score": -2.0,
            "released_is_fake": False,
            "classification_decision": False,
            "classification_threshold": 2.0,
            "classification_threshold_operator": ">",
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "preprocess": {"same": True},
        }
        full_row = {
            **shared,
            "run_id": "full",
            "run_manifest_fingerprint": full_manifest["fingerprint"],
            "raw_descriptor_path": str(full_path),
        }
        prefix_row = {
            **shared,
            "run_id": "prefix",
            "run_manifest_fingerprint": prefix_manifest["fingerprint"],
            "raw_descriptor_path": str(prefix_path),
        }
        result = audit_prefix_reproducibility(
            repo_root=self.root,
            full_run_id="full",
            full_manifest=full_manifest,
            full_rows=[full_row],
            prefix_run_id="prefix",
            prefix_manifest=prefix_manifest,
            prefix_rows=[prefix_row],
        )
        self.assertTrue(result["copied_full_rows_rejected"])

        copied = dict(full_row)
        copied["raw_descriptor_path"] = str(prefix_path)
        with self.assertRaisesRegex(ValueError, "own run ID"):
            audit_prefix_reproducibility(
                repo_root=self.root,
                full_run_id="full",
                full_manifest=full_manifest,
                full_rows=[full_row],
                prefix_run_id="prefix",
                prefix_manifest=prefix_manifest,
                prefix_rows=[copied],
            )

        reused = dict(prefix_row)
        reused["raw_descriptor_path"] = str(full_path)
        with self.assertRaisesRegex(ValueError, "reuses full-run descriptor"):
            audit_prefix_reproducibility(
                repo_root=self.root,
                full_run_id="full",
                full_manifest=full_manifest,
                full_rows=[full_row],
                prefix_run_id="prefix",
                prefix_manifest=prefix_manifest,
                prefix_rows=[reused],
            )


class EndToEndArtifactReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_real = self.root / "real.png"
        self.image_forged = self.root / "forged.png"
        _save_rgb(self.image_real, 2000, 1000)
        _save_rgb(self.image_forged, 2000, 1000)
        mask = np.zeros((1000, 2000), dtype=np.uint8)
        mask[500, 900] = 255
        self.gt = self.root / "gt.png"
        _save_mask(self.gt, mask)

    def tearDown(self):
        self.temporary.cleanup()

    def test_twenty_transforms_gmm_normalization_and_visibility_replay(self):
        import torch

        geometry = _official_preprocess_geometry(
            width=2000,
            height=1000,
            config=OFFICIAL_CONFIG,
        )
        descriptor = np.zeros(DESCRIPTOR_DIMENSION, dtype=np.float64)
        latest = {}
        input_rows = []
        raw = 20.0 * DESCRIPTOR_DIMENSION / 1000.0
        forged_contract = {
            "sample_id": "forged",
            "gt_mask_path": str(self.gt),
            "gt_mask_sha256": sha256_file(self.gt),
            "edit_region_xyxy": [850, 450, 1050, 550],
        }
        visibility_evidence = {
            "gt": _visible_gt_evidence(
                forged_input=forged_contract,
                geometry=geometry,
                repo_root=self.root,
            ),
            "edit_box": analyzer._edit_box_visibility(
                forged_contract["edit_region_xyxy"],
                geometry["effective_native_crop_xyxy"],
            ),
        }
        for kind, image in (("real", self.image_real), ("forged", self.image_forged)):
            sample_id = kind
            path = self.root / "descriptors" / f"{kind}.npy"
            _save_descriptor(path, descriptor)
            canonical = {
                "sample_id": sample_id,
                "task_id": "task",
                "pair_rank": 0,
                "kind": kind,
                "label": int(kind == "forged"),
                "domain": "lodging",
                "width": 2000,
                "height": 1000,
                "canonical_path": str(image),
                "canonical_sha256": sha256_file(image),
                "gt_mask_path": str(self.gt) if kind == "forged" else None,
                "gt_mask_sha256": sha256_file(self.gt) if kind == "forged" else None,
                "edit_region_xyxy": [850, 450, 1050, 550],
            }
            input_rows.append(canonical)
            latest[sample_id] = {
                "id": sample_id,
                "status": "ok",
                "raw_descriptor_path": str(path),
                "raw_descriptor_sha256": sha256_file(path),
                "raw_descriptor_shape": [DESCRIPTOR_DIMENSION],
                "raw_descriptor_dtype": "float64",
                "preprocess": geometry,
                "edit_visibility": "full",
                "edit_visible_gt_fraction": 1.0,
                "edit_visibility_evidence": visibility_evidence,
                **_score_row_fields(raw),
            }
        canonical_path = self.root / "inputs.jsonl"
        atomic_write_jsonl(canonical_path, input_rows)

        class Projection:
            @staticmethod
            def load_transforms(path, device):
                return list(range(20))

            @staticmethod
            def apply_projections(tensor, transforms):
                result = tensor
                for _ in transforms:
                    result = result + 1.0
                return result

        class FakeGMM:
            @staticmethod
            def score_samples(tensor):
                return tensor.sum(dim=1) / 1000.0

        class GMM:
            @staticmethod
            def load_gmm(path, device):
                return FakeGMM()

        pins = SimpleNamespace(
            RELEASED_Z_THRESHOLD=-2.0,
            AI_SCORE_THRESHOLD=2.0,
        )
        runtime = ReplayRuntime(
            torch=torch,
            device=torch.device("cpu"),
            recorded_device="cpu",
            evidence={"torch": str(torch.__version__), "device": "cpu"},
        )
        with (
            mock.patch.object(analyzer, "_load_runner_pins", return_value=pins),
            mock.patch.object(
                analyzer,
                "_validate_weight_files",
                return_value=(self.root, {}, "b" * 64, OFFICIAL_CONFIG),
            ),
        ):
            report = audit_artifacts(
                repo_root=self.root,
                fsd_root=self.root,
                manifest={
                    "model": {},
                    "dataset": {"inputs_path": str(canonical_path)},
                },
                input_rows=input_rows,
                latest=latest,
                runtime=runtime,
                projection_module=Projection,
                gmm_module=GMM,
            )
        self.assertEqual(report["images_replayed"], 2)
        self.assertEqual(report["transforms_replayed"], 20)
        self.assertEqual(report["edit_visibility_tasks"], {"full": 1})
        self.assertEqual(report["complete_pair_edit_visibility"], {"full": 1})
        self.assertEqual(report["incomplete_selection_tasks"], [])
        self.assertEqual(
            report["maximum_raw_likelihood_absolute_difference"],
            0.0,
        )
        forged_input = [row for row in input_rows if row["kind"] == "forged"]
        with (
            mock.patch.object(analyzer, "_load_runner_pins", return_value=pins),
            mock.patch.object(
                analyzer,
                "_validate_weight_files",
                return_value=(self.root, {}, "b" * 64, OFFICIAL_CONFIG),
            ),
        ):
            preflight = audit_artifacts(
                repo_root=self.root,
                fsd_root=self.root,
                manifest={
                    "model": {},
                    "dataset": {"inputs_path": str(canonical_path)},
                },
                input_rows=forged_input,
                latest={"forged": latest["forged"]},
                runtime=runtime,
                projection_module=Projection,
                gmm_module=GMM,
            )
        self.assertEqual(preflight["edit_visibility_tasks"], {"full": 1})
        self.assertEqual(preflight["complete_pair_edit_visibility"], {})
        self.assertEqual(preflight["incomplete_selection_tasks"], ["task"])


if __name__ == "__main__":
    unittest.main()
