import unittest

import numpy as np

from eval.opensource.psccnet_metrics import (
    binary_pixel_metrics_strict,
    image_detection_metrics_strict,
    summarize_psccnet_pair_slice,
    summarize_psccnet_results,
)


def _metrics(scores, target):
    return binary_pixel_metrics_strict(
        np.asarray(scores, dtype=np.float64),
        np.asarray(target),
    )


def _row(
    row_id,
    *,
    task_id,
    kind,
    score,
    native,
    latency_ms=2.0,
    peak_cuda_memory_bytes=100,
):
    return {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "status": "ok",
        "score": score,
        "localization": {"native": native},
        "latency_ms": latency_ms,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


class PSCCNetMetricsTest(unittest.TestCase):
    def test_t2_uses_strict_greater_than_at_exact_boundary(self):
        metrics = _metrics(
            [[0.5, 0.5001], [0.1, 0.9]],
            [[1, 1], [0, 0]],
        )

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["predicted_positive_pixels"], 2)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)

    def test_t2_rejects_invalid_arrays_and_non_official_threshold(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            binary_pixel_metrics_strict(
                np.zeros((2, 2)),
                np.zeros((1, 2)),
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            binary_pixel_metrics_strict(
                np.asarray([]),
                np.asarray([]),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            binary_pixel_metrics_strict(
                np.asarray([[np.nan]]),
                np.asarray([[0]]),
            )
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            binary_pixel_metrics_strict(
                np.asarray([[1.1]]),
                np.asarray([[0]]),
            )
        with self.assertRaisesRegex(ValueError, "other than 0 and 1"):
            binary_pixel_metrics_strict(
                np.asarray([[0.1]]),
                np.asarray([[2]]),
            )
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            binary_pixel_metrics_strict(
                np.asarray([[0.1]]),
                np.asarray([[0]]),
                0.4,
            )

    def test_t1_uses_strict_decisions_and_reports_standard_metrics(self):
        rows = [
            {"status": "ok", "label": 0, "score": 0.1},
            {"status": "ok", "label": 0, "score": 0.5},
            {"status": "ok", "label": 1, "score": 0.5},
            {"status": "ok", "label": 1, "score": 0.9},
        ]

        metrics = image_detection_metrics_strict(rows)

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["auroc"], 0.875)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertIn("tpr_at_fpr_5_percent", metrics)

        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            image_detection_metrics_strict(rows, threshold=0.5001)
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            image_detection_metrics_strict(
                [{"status": "ok", "label": 1, "score": 1.1}]
            )

    def test_summary_uses_latest_rows_and_reports_t1_t2_and_real_fp_area(self):
        real_a = _metrics(
            [[0.5, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged_a = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real_b = _metrics(
            [[0.6, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged_b = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        expected = [
            {"sample_id": "real-a"},
            {"sample_id": "forged-a"},
            {"sample_id": "real-b"},
            {"sample_id": "forged-b"},
            {"sample_id": "missing"},
            {"sample_id": "error"},
        ]
        rows = [
            _row(
                "real-a",
                task_id="a",
                kind="real",
                score=0.9,
                native=real_a,
            ),
            _row(
                "real-a",
                task_id="a",
                kind="real",
                score=0.2,
                native=real_a,
            ),
            _row(
                "forged-a",
                task_id="a",
                kind="forged",
                score=0.9,
                native=forged_a,
            ),
            _row(
                "real-b",
                task_id="b",
                kind="real",
                score=0.3,
                native=real_b,
            ),
            _row(
                "forged-b",
                task_id="b",
                kind="forged",
                score=0.8,
                native=forged_b,
            ),
            {"id": "error", "status": "error"},
            _row(
                "unexpected",
                task_id="x",
                kind="real",
                score=1.0,
                native=real_a,
            ),
        ]

        summary = summarize_psccnet_results(rows, expected)

        self.assertEqual(
            summary["coverage"],
            {
                "expected_images": 6,
                "result_images": 5,
                "valid_images": 4,
                "error_images": 1,
                "missing_images": 1,
            },
        )
        self.assertEqual(summary["task_scope"]["threshold_operator"], ">")
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertAlmostEqual(summary["paired_score_delta"]["mean"], 0.6)
        self.assertEqual(summary["paired_ranking_accuracy"], 1.0)
        forged = summary["localization_forged"]["native"]
        real = summary["localization_real"]["native"]
        self.assertEqual(forged["images"], 2)
        self.assertEqual(real["images"], 2)
        self.assertEqual(forged["macro_at_threshold"]["f1"], 0.5)
        self.assertAlmostEqual(
            forged["micro_at_threshold"]["f1"],
            2 / 3,
        )
        self.assertEqual(
            real["macro_at_threshold"]["false_positive_area_fraction"],
            0.125,
        )
        self.assertEqual(
            real["micro_at_threshold"]["false_positive_area_fraction"],
            1 / 8,
        )
        self.assertFalse(
            _contains_key(
                summary,
                {
                    "official_png_score",
                    "official_png_detection",
                    "official_png_paired_score_delta",
                },
            )
        )

    def test_summary_rejects_threshold_or_operator_drift(self):
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_psccnet_results(
                [],
                [],
                classification_threshold=0.4,
            )
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_psccnet_results([], [], mask_threshold=0.6)

        metrics = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        metrics["threshold_operator"] = ">="
        with self.assertRaisesRegex(ValueError, "expected '>'"):
            summarize_psccnet_results(
                [
                    _row(
                        "forged",
                        task_id="task",
                        kind="forged",
                        score=0.9,
                        native=metrics,
                    )
                ],
                [{"sample_id": "forged"}],
            )

    def test_pair_bootstrap_is_paired_deterministic_and_has_real_fp_area(self):
        pairs = []
        for index in range(4):
            real = _metrics(
                [[0.6, 0.1], [0.1, 0.1]],
                [[0, 0], [0, 0]],
            )
            forged = _metrics(
                [[0.9, 0.1], [0.1, 0.1]],
                [[1, 0], [0, 0]],
            )
            pairs.append(
                {
                    "real": _row(
                        f"real-{index}",
                        task_id=str(index),
                        kind="real",
                        score=0.1 + index * 0.01,
                        native=real,
                    ),
                    "forged": _row(
                        f"forged-{index}",
                        task_id=str(index),
                        kind="forged",
                        score=0.8 + index * 0.01,
                        native=forged,
                    ),
                }
            )

        summary = summarize_psccnet_pair_slice(
            pairs,
            iterations=50,
            seed=7,
        )
        repeated = summarize_psccnet_pair_slice(
            pairs,
            iterations=50,
            seed=7,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["pairs"], 4)
        self.assertEqual(summary["images"], 8)
        self.assertEqual(summary["threshold_operator"], ">")
        self.assertEqual(summary["auroc"]["estimate"], 1.0)
        self.assertEqual(summary["paired_ranking_accuracy"]["estimate"], 1.0)
        self.assertEqual(summary["paired_sign_test"]["wins"], 4)
        self.assertEqual(
            summary["real_false_positive_area_fraction_macro_at_0_5"][
                "estimate"
            ],
            0.25,
        )
        self.assertEqual(
            summary["real_false_positive_area_fraction_micro_at_0_5"][
                "estimate"
            ],
            0.25,
        )

    def test_pair_bootstrap_validates_arguments_and_pair_schema(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_psccnet_pair_slice([], iterations=10, seed=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            summarize_psccnet_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=0,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            summarize_psccnet_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
                localization_space="wrong",
            )
        with self.assertRaisesRegex(ValueError, "not status 'ok'"):
            summarize_psccnet_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
            )

        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        forged["threshold_operator"] = ">="
        with self.assertRaisesRegex(ValueError, "expected '>'"):
            summarize_psccnet_pair_slice(
                [
                    {
                        "real": _row(
                            "real",
                            task_id="task",
                            kind="real",
                            score=0.1,
                            native=real,
                        ),
                        "forged": _row(
                            "forged",
                            task_id="task",
                            kind="forged",
                            score=0.9,
                            native=forged,
                        ),
                    }
                ],
                iterations=1,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
