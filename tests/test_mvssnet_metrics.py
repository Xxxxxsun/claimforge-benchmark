import unittest

import numpy as np

from eval.opensource.mvssnet_metrics import (
    binary_pixel_metrics_strict,
    image_detection_metrics_strict,
    summarize_mvssnet_pair_slice,
    summarize_mvssnet_results,
)


def _metrics(scores, target):
    return binary_pixel_metrics_strict(
        np.asarray(scores, dtype=np.float32),
        np.asarray(target, dtype=bool),
        0.5,
    )


def _row(
    row_id,
    *,
    task_id,
    kind,
    score,
    official_png_score,
    native,
    model_512=None,
):
    return {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "status": "ok",
        "score": score,
        "official_png_score": official_png_score,
        "localization": {
            "native": native,
            "model_512": model_512 if model_512 is not None else native,
        },
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 100,
    }


class MVSSNetMetricsTest(unittest.TestCase):
    def test_binary_pixel_metrics_uses_strict_greater_than(self):
        metrics = _metrics(
            [[0.5, 0.5001], [0.1, 0.9]],
            [[1, 1], [0, 0]],
        )

        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["predicted_positive_pixels"], 2)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)

    def test_binary_pixel_metrics_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            binary_pixel_metrics_strict(
                np.zeros((2, 2)),
                np.zeros((1, 2)),
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

    def test_detection_reports_standard_metrics_and_strict_decisions(self):
        rows = [
            {"status": "ok", "label": 0, "score": 0.1},
            {"status": "ok", "label": 0, "score": 0.5},
            {"status": "ok", "label": 1, "score": 0.5},
            {"status": "ok", "label": 1, "score": 0.9},
        ]

        metrics = image_detection_metrics_strict(rows)

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

    def test_summary_covers_raw_and_official_t1_and_both_t2_spaces(self):
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
        ]
        rows = [
            {"id": "real-a", "status": "error"},
            _row(
                "real-a",
                task_id="a",
                kind="real",
                score=0.2,
                official_png_score=0.2,
                native=real_a,
            ),
            _row(
                "forged-a",
                task_id="a",
                kind="forged",
                score=0.9,
                official_png_score=0.8,
                native=forged_a,
            ),
            _row(
                "real-b",
                task_id="b",
                kind="real",
                score=0.3,
                official_png_score=0.4,
                native=real_b,
            ),
            _row(
                "forged-b",
                task_id="b",
                kind="forged",
                score=0.8,
                official_png_score=0.6,
                native=forged_b,
            ),
            _row(
                "unexpected",
                task_id="x",
                kind="real",
                score=1.0,
                official_png_score=1.0,
                native=real_a,
            ),
        ]

        summary = summarize_mvssnet_results(rows, expected)

        self.assertEqual(summary["coverage"]["valid_images"], 4)
        self.assertEqual(summary["coverage"]["missing_images"], 0)
        self.assertEqual(summary["task_scope"]["threshold_operator"], ">")
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertEqual(summary["official_png_detection"]["auroc"], 1.0)
        self.assertAlmostEqual(
            summary["paired_score_delta"]["mean"],
            0.6,
        )
        self.assertAlmostEqual(
            summary["official_png_paired_score_delta"]["mean"],
            0.4,
        )
        self.assertEqual(summary["paired_ranking_accuracy"], 1.0)
        self.assertEqual(
            summary["localization_forged"]["native"]["images"],
            2,
        )
        self.assertEqual(
            summary["localization_forged"]["model_512"]["images"],
            2,
        )
        self.assertEqual(
            summary["localization_forged"]["native"][
                "macro_at_threshold"
            ]["f1"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["localization_forged"]["native"][
                "micro_at_threshold"
            ]["f1"],
            2 / 3,
        )
        self.assertEqual(
            summary["localization_real"]["native"][
                "macro_at_threshold"
            ]["false_positive_area_fraction"],
            0.125,
        )
        self.assertEqual(
            summary["localization_real"]["native"][
                "micro_at_threshold"
            ]["false_positive_area_fraction"],
            1 / 8,
        )

    def test_summary_rejects_non_official_thresholds(self):
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_mvssnet_results(
                [],
                [],
                classification_threshold=0.4,
            )
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_mvssnet_results([], [], mask_threshold=0.6)

    def test_pair_bootstrap_preserves_pairs_and_is_deterministic(self):
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
                        official_png_score=0.1,
                        native=real,
                    ),
                    "forged": _row(
                        f"forged-{index}",
                        task_id=str(index),
                        kind="forged",
                        score=0.8 + index * 0.01,
                        official_png_score=0.8,
                        native=forged,
                    ),
                }
            )

        summary = summarize_mvssnet_pair_slice(
            pairs,
            iterations=50,
            seed=7,
        )
        repeated = summarize_mvssnet_pair_slice(
            pairs,
            iterations=50,
            seed=7,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["pairs"], 4)
        self.assertEqual(summary["images"], 8)
        self.assertEqual(summary["threshold_operator"], ">")
        self.assertEqual(summary["auroc"]["estimate"], 1.0)
        self.assertEqual(
            summary["official_png_auroc"]["estimate"],
            1.0,
        )
        self.assertEqual(
            summary["paired_ranking_accuracy"]["estimate"],
            1.0,
        )
        self.assertEqual(summary["paired_sign_test"]["wins"], 4)
        self.assertEqual(
            summary["real_false_positive_area_fraction_micro_at_0_5"][
                "estimate"
            ],
            0.25,
        )

    def test_pair_bootstrap_validates_arguments_and_schema(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_mvssnet_pair_slice([], iterations=10, seed=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            summarize_mvssnet_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=0,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            summarize_mvssnet_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
                localization_space="wrong",
            )


if __name__ == "__main__":
    unittest.main()
