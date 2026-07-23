import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource.analyze_mvssnet_run import (
    CLASSIFICATION_THRESHOLD,
    MODEL_NAME,
    MODEL_SLUG,
    OFFICIAL_PNG_SCORE_SEMANTICS,
    PRIMARY_SCORE_SEMANTICS,
    T2_MAP_SEMANTICS,
    THRESHOLD_COMPARISON,
    THRESHOLD_OPERATOR,
    Pair,
    _best_uint8_threshold,
    _binary_pixel_metrics_strict,
    _official_native_uint8,
    _preprocess_evidence,
    _select_manifest_inputs,
    _sigmoid,
    audit_and_best_threshold,
    summarize_mvssnet_pair_slice,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
)


class _FakeCV2:
    IMREAD_COLOR = 1
    INTER_LINEAR = 1
    INTER_NEAREST = 0

    @staticmethod
    def imread(path, _mode):
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1])

    @staticmethod
    def resize(array, size, interpolation):
        if interpolation not in (
            _FakeCV2.INTER_LINEAR,
            _FakeCV2.INTER_NEAREST,
        ):
            raise AssertionError("unexpected interpolation")
        width, height = size
        image = Image.fromarray(np.asarray(array))
        resized = image.resize(
            (width, height),
            resample=(
                Image.Resampling.BILINEAR
                if interpolation == _FakeCV2.INTER_LINEAR
                else Image.Resampling.NEAREST
            ),
        )
        return np.asarray(resized, dtype=np.asarray(array).dtype)


def _fake_pins():
    return SimpleNamespace(
        MODEL_SOURCE_COMMIT="1" * 40,
        MODEL_NETWORK_SHA256="2" * 64,
        MODEL_INFERENCE_SHA256="3" * 64,
        MODEL_EVALUATE_SHA256="4" * 64,
        MODEL_TOOLS_SHA256="5" * 64,
        MODEL_TRANSFORMS_SHA256="6" * 64,
        CHECKPOINT_SHA256="7" * 64,
        CHECKPOINT_BYTES=123,
        CHECKPOINT_STATE_KEYS=456,
    )


def _selection_contract(rows):
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


class MVSSNetProvenanceTest(unittest.TestCase):
    def _fixture(self, root: Path):
        pins = _fake_pins()
        run_id = "mvssnet_test"
        input_rows = [
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "rank": index,
                "pair_rank": 0,
                "sample_id": kind,
                "task_id": "task",
                "domain": "lodging",
                "kind": kind,
                "label": index,
                "canonical_path": f"images/{kind}.jpg",
                "canonical_sha256": f"{index + 8:064x}",
                "gt_mask_kind": (
                    "all_zero" if kind == "real" else "exact_diff"
                ),
                "gt_mask_sha256": None if kind == "real" else "a" * 64,
                "edit_region_xyxy": [0, 0, 1, 1],
                "width": 2,
                "height": 2,
            }
            for index, kind in enumerate(("real", "forged"))
        ]
        input_path = root / "inputs.jsonl"
        atomic_write_jsonl(input_path, input_rows)
        inputs_sha256 = sha256_file(input_path)
        atomic_write_json(
            root / "dataset.json",
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "contract_sha256": "contract",
                "inputs_path": "inputs.jsonl",
                "inputs_sha256": inputs_sha256,
            },
        )
        adapter = root / "adapter.py"
        adapter.write_text("adapter\n", encoding="utf-8")

        ordered_inputs = _selection_contract(input_rows)
        immutable = {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": run_id,
            "condition": "test",
            "input": {
                "dataset_id": "test-dataset",
                "dataset_manifest": "dataset.json",
                "dataset_contract_sha256": "contract",
                "inputs_manifest": "inputs.jsonl",
                "inputs_sha256": inputs_sha256,
                "selection_sha256": hashlib.sha256(
                    stable_json(ordered_inputs).encode("utf-8")
                ).hexdigest(),
            },
            "ordered_inputs": ordered_inputs,
            "model": {
                "name": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "source_commit": pins.MODEL_SOURCE_COMMIT,
                "source_tracked_clean": True,
                "supports_image_level_t1": True,
                "supports_pixel_level_t2": True,
                "image_level_head": "none_map_global_max_pooling",
                "implementation": {
                    "network_sha256": pins.MODEL_NETWORK_SHA256,
                    "inference_sha256": pins.MODEL_INFERENCE_SHA256,
                    "evaluation_sha256": pins.MODEL_EVALUATE_SHA256,
                    "tools_sha256": pins.MODEL_TOOLS_SHA256,
                    "transforms_sha256": pins.MODEL_TRANSFORMS_SHA256,
                },
                "license": {
                    "project_wide_status": "no_project_license_found",
                    "classification": "source_available_research_release",
                },
                "checkpoint": {
                    "sha256": pins.CHECKPOINT_SHA256,
                    "bytes": pins.CHECKPOINT_BYTES,
                    "state_keys": pins.CHECKPOINT_STATE_KEYS,
                    "strict_load": True,
                    "safe_weights_only_load": True,
                },
            },
            "inference": {
                "precision": "float32",
                "batch_size": 1,
                "model_input_size": [512, 512],
                "channel_order": "BGR",
                "primary_t1_score": PRIMARY_SCORE_SEMANTICS,
                "official_evaluate_t1_score": (
                    OFFICIAL_PNG_SCORE_SEMANTICS
                ),
                "classification_threshold": 0.5,
                "classification_threshold_comparison": (
                    THRESHOLD_COMPARISON
                ),
                "mask_threshold": 0.5,
                "mask_threshold_comparison": THRESHOLD_COMPARISON,
            },
            "expected_pairs": 1,
            "expected_images": 2,
            "adapter_contract": [
                {
                    "path": "adapter.py",
                    "sha256": sha256_file(adapter),
                }
            ],
        }
        fingerprint = hashlib.sha256(
            stable_json(immutable).encode("utf-8")
        ).hexdigest()
        manifest = {
            **immutable,
            "fingerprint": fingerprint,
            "created_at": "test",
            "adapter": {},
            "environment": {},
        }

        identity = {
            row["sample_id"]: {
                "schema_version": "opensource_result_v1",
                "run_id": run_id,
                "run_manifest_fingerprint": fingerprint,
                "input_manifest_sha256": inputs_sha256,
                "id": row["sample_id"],
                "rank": row["rank"],
                "task_id": row["task_id"],
                "pair_rank": row["pair_rank"],
                "domain": row["domain"],
                "kind": row["kind"],
                "label": row["label"],
                "image_path": row["canonical_path"],
                "image_sha256": row["canonical_sha256"],
                "image_size": [row["width"], row["height"]],
                "gt_mask_kind": row["gt_mask_kind"],
                "gt_mask_sha256": row["gt_mask_sha256"],
                "edit_region_xyxy": row["edit_region_xyxy"],
                "model": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "checkpoint_sha256": pins.CHECKPOINT_SHA256,
                "valid_for_t1": True,
                "valid_for_t2": True,
            }
            for row in input_rows
        }
        ok_fields = {
            "status": "ok",
            "raw_score_semantics": PRIMARY_SCORE_SEMANTICS,
            "official_png_score_semantics": OFFICIAL_PNG_SCORE_SEMANTICS,
            "classification_threshold": 0.5,
            "mask_threshold": 0.5,
        }
        result_rows = [
            {**identity["real"], **ok_fields},
            {
                **identity["forged"],
                "status": "error",
                "error_type": "RuntimeError",
            },
            {**identity["forged"], **ok_fields},
        ]
        summary = {
            "schema_version": "opensource_summary_v1",
            "run_id": run_id,
            "condition": "test",
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": pins.CHECKPOINT_SHA256,
            "input_manifest_sha256": inputs_sha256,
            "run_manifest_fingerprint": fingerprint,
            "coverage": {
                "expected_images": 2,
                "result_images": 2,
                "valid_images": 2,
                "error_images": 0,
                "missing_images": 0,
            },
            "task_scope": {
                "primary_detection_score": "score",
                "primary_detection_semantics": "raw_GMP_model_512",
                "secondary_detection_score": "official_png_score",
                "primary_localization_space": "native",
                "primary_localization_semantics": "official_uint8_div_255",
                "threshold_operator": THRESHOLD_OPERATOR,
            },
        }
        return (
            pins,
            run_id,
            input_path,
            input_rows,
            result_rows,
            manifest,
            summary,
        )

    def test_error_then_ok_history_and_full_provenance_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            pins, run_id, inputs_path, inputs, results, manifest, summary = (
                fixture
            )
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._load_runner_pins",
                return_value=pins,
            ):
                provenance = validate_provenance(
                    repo_root=root,
                    run_id=run_id,
                    input_path=inputs_path,
                    input_rows=inputs,
                    result_rows=results,
                    manifest=manifest,
                    summary=summary,
                )
            history = summarize_result_history(results)

        self.assertEqual(provenance["physical_result_rows_validated"], 3)
        self.assertEqual(provenance["adapter_contract_files_validated"], 1)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_error_to_ok"], 1)

    def test_every_historical_row_checkpoint_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            pins, run_id, inputs_path, inputs, results, manifest, summary = (
                fixture
            )
            results[1]["checkpoint_sha256"] = "wrong"
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._load_runner_pins",
                return_value=pins,
            ), self.assertRaisesRegex(
                ValueError,
                "result row 2 field checkpoint_sha256",
            ):
                validate_provenance(
                    repo_root=root,
                    run_id=run_id,
                    input_path=inputs_path,
                    input_rows=inputs,
                    result_rows=results,
                    manifest=manifest,
                    summary=summary,
                )

    def test_manifest_fingerprint_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            pins, run_id, inputs_path, inputs, results, manifest, summary = (
                fixture
            )
            manifest["expected_images"] = 99
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._load_runner_pins",
                return_value=pins,
            ), self.assertRaisesRegex(ValueError, "run manifest fingerprint"):
                validate_provenance(
                    repo_root=root,
                    run_id=run_id,
                    input_path=inputs_path,
                    input_rows=results,
                    result_rows=results,
                    manifest=manifest,
                    summary=summary,
                )

    def test_manifest_selection_supports_smoke_subset(self):
        rows = [
            {"sample_id": "a"},
            {"sample_id": "b"},
            {"sample_id": "c"},
        ]
        manifest = {"ordered_inputs": [{"sample_id": "b"}]}
        self.assertEqual(
            _select_manifest_inputs(rows, manifest),
            [rows[1]],
        )


def _metric_row(scores, target, *, include_ap):
    return _binary_pixel_metrics_strict(
        np.asarray(scores, dtype=np.float32),
        np.asarray(target, dtype=bool),
        include_ap=include_ap,
    )


def _pair(
    name,
    *,
    real_score,
    forged_score,
    real_png_score,
    forged_png_score,
    forged_detects,
):
    real_metrics = _metric_row(
        [[0.5, 0.6], [0.1, 0.1]],
        [[0, 0], [0, 0]],
        include_ap=False,
    )
    forged_metrics = _metric_row(
        (
            [[0.9, 0.1], [0.1, 0.1]]
            if forged_detects
            else [[0.1, 0.1], [0.1, 0.1]]
        ),
        [[1, 0], [0, 0]],
        include_ap=True,
    )
    return Pair(
        task_id=name,
        domain="lodging",
        real={
            "score": real_score,
            "official_png_score": real_png_score,
            "localization": {"native": real_metrics},
        },
        forged={
            "score": forged_score,
            "official_png_score": forged_png_score,
            "localization": {"native": forged_metrics},
        },
        input_row={},
    )


class MVSSNetStatisticsTest(unittest.TestCase):
    def test_strict_threshold_keeps_exact_half_negative(self):
        metrics = _metric_row(
            [[0.5, 0.5001]],
            [[0, 0]],
            include_ap=False,
        )
        self.assertEqual(metrics["predicted_positive_pixels"], 1)
        self.assertEqual(metrics["threshold_operator"], ">")

    def test_pair_bootstrap_covers_continuous_and_official_png_t1_and_t2(self):
        pairs = [
            _pair(
                "a",
                real_score=0.2,
                forged_score=0.8,
                real_png_score=0.1,
                forged_png_score=0.7,
                forged_detects=True,
            ),
            _pair(
                "b",
                real_score=0.8,
                forged_score=0.6,
                real_png_score=0.9,
                forged_png_score=0.6,
                forged_detects=False,
            ),
        ]
        summary = summarize_mvssnet_pair_slice(
            pairs,
            iterations=50,
            seed=9,
        )
        repeated = summarize_mvssnet_pair_slice(
            pairs,
            iterations=50,
            seed=9,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(
            summary["paired_ranking_accuracy"]["estimate"],
            0.5,
        )
        self.assertEqual(
            summary["official_png_paired_ranking_accuracy"]["estimate"],
            0.5,
        )
        self.assertEqual(
            summary["pixel_f1_macro_at_0_5"]["estimate"],
            0.5,
        )
        self.assertEqual(
            summary[
                "real_false_positive_area_fraction_macro_at_0_5"
            ]["estimate"],
            0.25,
        )
        self.assertEqual(summary["threshold_operator"], ">")

    def test_exact_uint8_oracle_uses_strict_greater_than(self):
        all_hist = np.zeros(256, dtype=np.int64)
        positive_hist = np.zeros(256, dtype=np.int64)
        all_hist[[0, 100, 200, 255]] = 1
        positive_hist[[200, 255]] = 1

        best = _best_uint8_threshold(all_hist, positive_hist)

        self.assertEqual(best["threshold_byte"], 100)
        self.assertEqual(best["comparison"], ">")
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(best["iou"], 1.0)


class MVSSNetArtifactAuditTest(unittest.TestCase):
    def _save_result(
        self,
        *,
        root: Path,
        sample_id: str,
        kind: str,
        logits: np.ndarray,
        image_path: Path,
        target: np.ndarray,
    ):
        artifact_dir = root / "artifacts" / sample_id
        artifact_dir.mkdir(parents=True)
        logits_path = artifact_dir / "logits.npy"
        model_map_path = artifact_dir / "model.npy"
        native_path = artifact_dir / "native.png"
        mask_path = artifact_dir / "mask.png"

        model_map = _sigmoid(logits)
        native = _official_native_uint8(model_map, width=4, height=3)
        score_map = native.astype(np.float32) / 255.0
        mask = np.where(score_map > 0.5, 255, 0).astype(np.uint8)
        np.save(logits_path, logits, allow_pickle=False)
        np.save(model_map_path, model_map, allow_pickle=False)
        Image.fromarray(native, mode="L").save(native_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        evidence, _ = _preprocess_evidence(image_path)
        metrics = _binary_pixel_metrics_strict(
            score_map,
            target,
            include_ap=kind == "forged",
        )
        target_model = _FakeCV2.resize(
            np.asarray(target, dtype=np.uint8),
            (512, 512),
            _FakeCV2.INTER_NEAREST,
        ) > 0
        model_metrics = _binary_pixel_metrics_strict(
            model_map,
            target_model,
            include_ap=kind == "forged",
        )
        return {
            "id": sample_id,
            "kind": kind,
            "image_path": str(image_path.relative_to(root)),
            "image_sha256": sha256_file(image_path),
            "image_size": [4, 3],
            "raw_logits_model_path": str(logits_path.relative_to(root)),
            "raw_logits_model_sha256": sha256_file(logits_path),
            "raw_logits_model_shape": [512, 512],
            "score_map_model_path": str(model_map_path.relative_to(root)),
            "score_map_model_sha256": sha256_file(model_map_path),
            "score_map_model_shape": [512, 512],
            "score_map_native_path": str(native_path.relative_to(root)),
            "score_map_native_sha256": sha256_file(native_path),
            "score_map_native_shape": [3, 4],
            "mask_path": str(mask_path.relative_to(root)),
            "mask_sha256": sha256_file(mask_path),
            "mask_shape": [3, 4],
            "score": float(np.max(model_map)),
            "official_png_score": float(np.max(native)) / 255.0,
            "decision": bool(np.max(model_map) > 0.5),
            "official_png_decision": bool(np.max(native) / 255.0 > 0.5),
            "preprocess": evidence,
            "localization": {
                "model_512": model_metrics,
                "native": metrics,
            },
        }

    def _fixture(self, root: Path):
        image_dir = root / "images"
        image_dir.mkdir()
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        pixels[..., 0] = 30
        pixels[..., 1] = 90
        pixels[..., 2] = 170
        real_image = image_dir / "real.png"
        forged_image = image_dir / "forged.png"
        Image.fromarray(pixels, mode="RGB").save(real_image)
        Image.fromarray(pixels + 1, mode="RGB").save(forged_image)

        real_target = np.zeros((3, 4), dtype=bool)
        forged_target = np.zeros((3, 4), dtype=bool)
        forged_target[:, :2] = True
        gt_path = root / "gt.png"
        Image.fromarray(
            np.where(forged_target, 255, 0).astype(np.uint8),
            mode="L",
        ).save(gt_path)

        real_logits = np.zeros((512, 512), dtype=np.float32)
        forged_logits = np.full((512, 512), -4.0, dtype=np.float32)
        forged_logits[:, :256] = 4.0
        real = self._save_result(
            root=root,
            sample_id="real",
            kind="real",
            logits=real_logits,
            image_path=real_image,
            target=real_target,
        )
        forged = self._save_result(
            root=root,
            sample_id="forged",
            kind="forged",
            logits=forged_logits,
            image_path=forged_image,
            target=forged_target,
        )
        forged["task_id"] = real["task_id"] = "task"
        forged["pair_rank"] = real["pair_rank"] = 0
        forged["domain"] = real["domain"] = "lodging"
        input_row = {
            "gt_mask_path": str(gt_path.relative_to(root)),
            "gt_mask_sha256": sha256_file(gt_path),
            "edit_region_xyxy": [0, 0, 2, 3],
        }
        return Pair(
            task_id="task",
            domain="lodging",
            real=real,
            forged=forged,
            input_row=input_row,
        )

    def test_audit_recomputes_preprocess_postprocess_scores_masks_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._cv2",
                return_value=_FakeCV2,
            ):
                pair = self._fixture(root)
                audit = audit_and_best_threshold([pair], repo_root=root)

        self.assertEqual(audit["artifact_integrity"]["status"], "ok")
        self.assertEqual(audit["artifact_integrity"]["checked_files"], 11)
        self.assertEqual(
            audit["box_hit_at_mask_threshold_0_5"]["hits"],
            1,
        )
        self.assertEqual(
            audit["localization_best_threshold"]["single_global_oracle"][
                "comparison"
            ],
            ">",
        )

    def test_audit_rejects_native_map_not_derived_by_official_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._cv2",
                return_value=_FakeCV2,
            ):
                pair = self._fixture(root)
                native_path = root / pair.forged["score_map_native_path"]
                with Image.open(native_path) as opened:
                    tampered = np.asarray(opened, dtype=np.uint8).copy()
                tampered[0, 0] ^= np.uint8(1)
                Image.fromarray(tampered, mode="L").save(native_path)
                pair.forged["score_map_native_sha256"] = sha256_file(
                    native_path
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "official postprocess mismatch",
                ):
                    audit_and_best_threshold([pair], repo_root=root)

    def test_audit_rejects_tampered_primary_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "eval.opensource.analyze_mvssnet_run._cv2",
                return_value=_FakeCV2,
            ):
                pair = self._fixture(root)
                pair.forged["score"] = 0.123
                with self.assertRaisesRegex(ValueError, "score mismatch"):
                    audit_and_best_threshold([pair], repo_root=root)


if __name__ == "__main__":
    unittest.main()
