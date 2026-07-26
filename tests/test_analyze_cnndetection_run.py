from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from eval.opensource import analyze_cnndetection_run as analyzer
from eval.opensource import run_cnndetection as runner
from eval.opensource.common import stable_json


HAS_OFFICIAL_ASSETS = (
    runner.DEFAULT_SOURCE_ROOT.is_dir()
    and runner.DEFAULT_CHECKPOINT.is_file()
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnalyzeCNNDetectionRunTest(unittest.TestCase):
    def test_independent_preprocessing_matches_runner(self):
        rng = np.random.default_rng(19)
        pixels = rng.integers(0, 256, (257, 301, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            for profile in (
                runner.PRIMARY_PROFILE,
                runner.PAPER_CROP_PROFILE,
            ):
                expected_tensor, expected_audit = runner.preprocess_image(
                    path,
                    profile,
                )
                actual_tensor, actual_audit = (
                    analyzer.independent_preprocess_image(path, profile)
                )
                self.assertTrue(expected_tensor.equal(actual_tensor))
                self.assertEqual(expected_audit, actual_audit)

    def test_localization_claims_are_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, "localization key"):
            analyzer._reject_localization_claims(
                {"nested": [{"t2": {"score": 1.0}}]},
                "payload",
            )
        analyzer._reject_localization_claims(
            {"task_scope": {"valid_for_t2": False}},
            "payload",
        )

    def test_score_alias_audit_accepts_strict_official_contract(self):
        row = {
            "raw_logit": 0.0,
            "ai_score": 0.5,
            "probability": 0.5,
            "score": 0.5,
            "score_semantics": (
                "official_float32_sigmoid_uncalibrated_fake_score"
            ),
            "calibrated_probability": False,
            "classification_threshold": 0.5,
            "classification_threshold_operator": ">",
            "classification_decision": False,
            "classification": {
                "raw_logit": 0.0,
                "probability": 0.5,
                "ai_score": 0.5,
                "score": 0.5,
                "threshold": 0.5,
                "threshold_operator": ">",
                "decision": False,
                "semantics": (
                    "official_float32_sigmoid_uncalibrated_fake_score"
                ),
            },
            "t1": {
                "raw_logit": 0.0,
                "probability": 0.5,
                "ai_score": 0.5,
                "score": 0.5,
                "threshold": 0.5,
                "threshold_operator": ">",
                "decision": False,
                "semantics": (
                    "official_float32_sigmoid_uncalibrated_fake_score"
                ),
                "policy": (
                    "official_CNNDetection_float32_sigmoid_strict_gt_0_5"
                ),
            },
            "manual_replay": {
                "raw_logit": 0.0,
                "probability": 0.5,
                "ai_score": 0.5,
                "classification_decision": False,
                "model_forward_calls": 1,
                "fc_hook_calls": 1,
                "official_logit_exact_match": True,
                "official_score_exact_match": True,
            },
        }
        analyzer._compare_score_fields(
            row,
            replay_logit=0.0,
            replay_score=0.5,
            replay_decision=False,
        )

    def test_score_alias_audit_rejects_calibrated_claim(self):
        row = {
            "raw_logit": 0.0,
            "ai_score": 0.5,
            "probability": 0.5,
            "score": 0.5,
            "score_semantics": (
                "official_float32_sigmoid_uncalibrated_fake_score"
            ),
            "calibrated_probability": True,
        }
        with self.assertRaisesRegex(ValueError, "marked calibrated"):
            analyzer._compare_score_fields(
                row,
                replay_logit=0.0,
                replay_score=0.5,
                replay_decision=False,
            )

    def test_stored_summary_must_recompute(self):
        analyzer._compare_summary(
            stored={
                "value": 1,
                "run_id": "x",
                "generated_at": "time",
            },
            recomputed={"value": 1},
        )
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            analyzer._compare_summary(
                stored={"value": 2},
                recomputed={"value": 1},
            )

    def test_replay_audit_requires_complete_successful_coverage(self):
        manifest = {"status": "complete"}
        coverage = {
            "expected_images": 2,
            "result_images": 2,
            "valid_images": 2,
            "error_images": 0,
            "missing_images": 0,
            "coverage_fraction": 1.0,
            "valid_fraction": 1.0,
            "is_complete": True,
        }
        analyzer._require_complete_replay_target(
            manifest=manifest,
            summary={"coverage": coverage},
            expected_images=2,
        )
        cases = (
            (
                "manifest",
                {"status": "incomplete"},
                coverage,
                "manifest status complete",
            ),
            (
                "missing",
                manifest,
                {**coverage, "missing_images": 1, "is_complete": False},
                "complete successful coverage",
            ),
            (
                "error",
                manifest,
                {**coverage, "error_images": 1, "is_complete": False},
                "complete successful coverage",
            ),
            (
                "valid",
                manifest,
                {**coverage, "valid_images": 1, "is_complete": False},
                "complete successful coverage",
            ),
        )
        for name, changed_manifest, changed_coverage, message in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    analyzer._require_complete_replay_target(
                        manifest=changed_manifest,
                        summary={"coverage": changed_coverage},
                        expected_images=2,
                    )

    def test_default_output_is_named_independent_audit(self):
        report = {
            "schema_version": "cnndetection_replay_audit_v1",
            "status": "replay_audit_passed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    analyzer,
                    "analyze",
                    return_value=report,
                ) as analyze_call,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    analyzer.main(
                        [
                            "--repo-root",
                            str(root),
                            "--results-dir",
                            "results",
                            "--run-id",
                            "run",
                        ]
                    ),
                    0,
                )
        self.assertEqual(
            analyze_call.call_args.kwargs["output_path"].name,
            "independent_audit.json",
        )

    @unittest.skipUnless(HAS_OFFICIAL_ASSETS, "official assets not cached")
    def test_tiny_cpu_run_resume_replay_audit_and_tamper_gate(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_examples = runner.DEFAULT_SOURCE_ROOT / "examples"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_path = source_examples / "real.png"
            fake_path = source_examples / "fake.png"
            with Image.open(real_path) as real_image:
                real = np.asarray(real_image.convert("RGB"), dtype=np.int16)
            with Image.open(fake_path) as fake_image:
                fake = np.asarray(fake_image.convert("RGB"), dtype=np.int16)
            mask = (
                np.max(np.abs(real - fake), axis=2) > 0
            ).astype(np.uint8) * 255
            mask_path = root / "mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            positive = int(np.count_nonzero(mask == 255))
            dataset_id = "cnndetection-official-golden-pair-v1"
            rows = [
                {
                    "schema_version": "claimforge_mouse_canonical_v1",
                    "sample_id": "golden-real",
                    "task_id": "golden-pair",
                    "rank": 0,
                    "pair_rank": 0,
                    "kind": "real",
                    "label": 0,
                    "domain": "golden",
                    "candidate": "official_example",
                    "dataset_id": dataset_id,
                    "canonical_path": str(real_path),
                    "canonical_sha256": _sha256(real_path),
                    "width": 256,
                    "height": 256,
                    "gt_mask_kind": "all_zero",
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": 0,
                },
                {
                    "schema_version": "claimforge_mouse_canonical_v1",
                    "sample_id": "golden-fake",
                    "task_id": "golden-pair",
                    "rank": 1,
                    "pair_rank": 0,
                    "kind": "forged",
                    "label": 1,
                    "domain": "golden",
                    "candidate": "official_example",
                    "dataset_id": dataset_id,
                    "canonical_path": str(fake_path),
                    "canonical_sha256": _sha256(fake_path),
                    "width": 256,
                    "height": 256,
                    "gt_mask_kind": "exact_diff",
                    "gt_mask_path": str(mask_path),
                    "gt_mask_sha256": _sha256(mask_path),
                    "gt_positive_pixels": positive,
                },
            ]
            inputs_path = root / "inputs.jsonl"
            inputs_path.write_text(
                "".join(f"{stable_json(row)}\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "claimforge_mouse_canonical_v1",
                        "dataset_id": dataset_id,
                        "inputs_path": str(inputs_path),
                        "inputs_sha256": _sha256(inputs_path),
                        "images": 2,
                        "pairs": 1,
                    }
                ),
                encoding="utf-8",
            )
            results_dir = root / "results"
            run_arguments = [
                "--source-root",
                str(runner.DEFAULT_SOURCE_ROOT),
                "--checkpoint",
                str(runner.DEFAULT_CHECKPOINT),
                "--dataset-manifest",
                str(manifest_path),
                "--results-dir",
                str(results_dir),
                "--run-id",
                "tiny",
                "--device",
                "cpu",
                "--bootstrap-samples",
                "5",
                "--bootstrap-seed",
                "17",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    run_arguments
                )
            self.assertEqual(exit_code, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                resume_exit_code = runner.main(
                    [*run_arguments, "--resume"]
                )
            self.assertEqual(resume_exit_code, 0)
            completed_manifest = json.loads(
                (results_dir / "tiny" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                completed_manifest["execution"]["resume_skips"],
                2,
            )
            self.assertEqual(
                completed_manifest["execution"]["new_successes"],
                0,
            )
            report = analyzer.analyze(
                repo_root=repo_root,
                results_dir=results_dir,
                run_id="tiny",
                source_root=runner.DEFAULT_SOURCE_ROOT,
                checkpoint_path=runner.DEFAULT_CHECKPOINT,
                device_text="cpu",
                output_path=None,
            )
            results_path = results_dir / "tiny" / "results.jsonl"
            physical_rows = [
                json.loads(line)
                for line in results_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            physical_rows[0]["manual_replay"][
                "official_score_exact_match"
            ] = False
            results_path.write_text(
                "".join(
                    f"{stable_json(row)}\n" for row in physical_rows
                ),
                encoding="utf-8",
            )
            with (
                self.assertRaisesRegex(ValueError, "manual replay"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.main([*run_arguments, "--resume"])
        self.assertEqual(report["status"], "replay_audit_passed")
        self.assertEqual(
            report["schema_version"],
            "cnndetection_replay_audit_v1",
        )
        self.assertEqual(report["expected_images"], 2)
        self.assertEqual(report["successful_images_replayed"], 2)
        self.assertTrue(
            report["summary_recomputed_with_shared_metrics"]
        )
        self.assertFalse(
            report["audit_scope"][
                "fully_independent_statistical_implementation"
            ]
        )
        self.assertTrue(report["localization_claims_rejected"])


if __name__ == "__main__":
    unittest.main()
