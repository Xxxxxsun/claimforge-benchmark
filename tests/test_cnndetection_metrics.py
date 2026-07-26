from __future__ import annotations

import unittest

from eval.opensource import cnndetection_metrics as metrics


def _expected() -> list[dict]:
    return [
        {
            "sample_id": "real-0",
            "task_id": "task-0",
            "kind": "real",
            "label": 0,
            "domain": "test",
        },
        {
            "sample_id": "fake-0",
            "task_id": "task-0",
            "kind": "forged",
            "label": 1,
            "domain": "test",
        },
    ]


def _row(
    sample_id: str,
    kind: str,
    score: float,
    logit: float,
    *,
    status: str = "ok",
) -> dict:
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_id": "task-0",
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": "test",
        "status": status,
        "ai_score": score,
        "raw_logit": logit,
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": None,
    }


class CNNDetectionMetricsTest(unittest.TestCase):
    def test_frozen_operating_point(self):
        self.assertEqual(metrics.FIXED_THRESHOLD, 0.5)
        self.assertEqual(metrics.THRESHOLD_OPERATOR, ">")
        rows = [
            _row("real-0", "real", 0.5, 0.0),
            _row("fake-0", "forged", 0.5, 0.0),
        ]
        result = metrics.cnndetection_detection_metrics_strict(rows)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(
            result["score_semantics"],
            "official_float32_sigmoid_uncalibrated_fake_score",
        )

    def test_wrong_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed threshold"):
            metrics.cnndetection_detection_metrics_strict([], threshold=0.4)

    def test_nonfinite_and_out_of_range_scores_are_rejected(self):
        for bad in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(score=bad):
                with self.assertRaises(ValueError):
                    metrics.cnndetection_detection_metrics_strict(
                        [_row("real-0", "real", bad, 0.0)]
                    )

    def test_complete_pair_summary(self):
        rows = [
            _row("real-0", "real", 0.1, -2.1972246),
            _row("fake-0", "forged", 0.9, 2.1972246),
        ]
        summary = metrics.summarize_cnndetection_results(
            rows,
            _expected(),
            bootstrap_samples=20,
            seed=7,
        )
        self.assertEqual(
            summary["schema_version"],
            "cnndetection_detection_summary_v1",
        )
        self.assertTrue(summary["coverage"]["is_complete"])
        self.assertEqual(summary["paired_coverage"]["complete_valid_pairs"], 1)
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertEqual(summary["paired_ranking_accuracy"], 1.0)
        self.assertFalse(
            summary["task_scope"]["calibrated_probability"]
        )

    def test_last_physical_retry_wins(self):
        rows = [
            _row("real-0", "real", 0.9, 2.0),
            _row("fake-0", "forged", 0.1, -2.0),
            _row("real-0", "real", 0.1, -2.0),
            _row("fake-0", "forged", 0.9, 2.0),
        ]
        summary = metrics.summarize_cnndetection_results(
            rows,
            _expected(),
            bootstrap_samples=5,
            seed=3,
        )
        self.assertEqual(summary["coverage"]["physical_result_rows"], 4)
        self.assertEqual(summary["paired_ranking_accuracy"], 1.0)

    def test_raw_logit_diagnostic_preserves_saturated_order(self):
        rows = [
            _row("real-0", "real", 0.0, -120.0),
            _row("fake-0", "forged", 0.0, -100.0),
        ]
        result = metrics.summarize_cnndetection_raw_logits(
            rows,
            _expected(),
        )
        self.assertEqual(
            result["schema_version"],
            "cnndetection_raw_logit_diagnostic_v1",
        )
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["paired_ranking_accuracy"], 1.0)
        self.assertEqual(result["paired_logit_delta"]["mean"], 20.0)

    def test_raw_logit_must_be_finite(self):
        rows = [
            _row("real-0", "real", 0.1, float("nan")),
            _row("fake-0", "forged", 0.9, 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "not finite"):
            metrics.summarize_cnndetection_raw_logits(rows, _expected())

    def test_unexpected_result_id_is_rejected(self):
        rows = [_row("other", "real", 0.1, -2.0)]
        with self.assertRaisesRegex(ValueError, "unexpected result id"):
            metrics.summarize_cnndetection_raw_logits(rows, _expected())

    def test_error_row_can_be_retried_successfully(self):
        error = _row(
            "real-0",
            "real",
            0.0,
            -1.0,
            status="error",
        )
        success = _row("real-0", "real", 0.1, -2.0)
        rows = [
            error,
            success,
            _row("fake-0", "forged", 0.9, 2.0),
        ]
        summary = metrics.summarize_cnndetection_results(
            rows,
            _expected(),
            bootstrap_samples=5,
            seed=1,
        )
        self.assertTrue(summary["coverage"]["is_complete"])
        self.assertEqual(summary["coverage"]["error_images"], 0)

    def test_boolean_is_not_a_numeric_score(self):
        with self.assertRaises(ValueError):
            metrics.cnndetection_detection_metrics_strict(
                [_row("real-0", "real", True, 0.0)]
            )


if __name__ == "__main__":
    unittest.main()
