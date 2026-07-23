import unittest

import numpy as np

from eval.opensource.catnet_metrics import (
    binary_pixel_metrics,
    summarize_catnet_pair_slice,
    summarize_catnet_results,
)


def _metrics(scores, target):
    return binary_pixel_metrics(
        np.asarray(scores, dtype=np.float32),
        np.asarray(target, dtype=bool),
        0.5,
    )


def _row(
    row_id,
    *,
    task_id,
    kind,
    metrics,
    latency_ms=1,
    peak_cuda_memory_bytes=10,
):
    return {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "status": "ok",
        "localization": {"native": metrics},
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


class CATNetMetricsTest(unittest.TestCase):
    def test_binary_pixel_metrics_handles_a_real_image(self):
        metrics = _metrics(
            [[0.1, 0.6], [0.2, 0.3]],
            [[0, 0], [0, 0]],
        )
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 3)
        self.assertEqual(metrics["predicted_positive_fraction"], 0.25)
        self.assertIsNone(metrics["pixel_ap"])

    def test_summary_is_localization_only_with_forged_and_real_aggregates(self):
        forged_a = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        forged_b = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real_a = _metrics(
            [[0.6, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        real_b = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        expected = [
            {"sample_id": "real-a"},
            {"sample_id": "forged-a"},
            {"sample_id": "real-b"},
            {"sample_id": "forged-b"},
        ]
        rows = [
            _row(
                "real-a",
                task_id="a",
                kind="real",
                metrics=real_a,
            ),
            _row(
                "forged-a",
                task_id="a",
                kind="forged",
                metrics=forged_a,
            ),
            _row(
                "real-b",
                task_id="b",
                kind="real",
                metrics=real_b,
            ),
            _row(
                "forged-b",
                task_id="b",
                kind="forged",
                metrics=forged_b,
            ),
        ]

        summary = summarize_catnet_results(rows, expected)

        self.assertEqual(summary["coverage"]["valid_images"], 4)
        forged = summary["localization"]["native"]
        real = summary["real_localization"]["native"]
        self.assertEqual(forged["images"], 2)
        self.assertEqual(real["images"], 2)
        self.assertEqual(forged["f1"]["mean"], 0.5)
        self.assertEqual(forged["macro_at_threshold"]["f1"], 0.5)
        self.assertAlmostEqual(
            forged["micro_at_threshold"]["f1"],
            2 / 3,
        )
        self.assertEqual(
            real["micro_at_threshold"]["predicted_positive_fraction"],
            1 / 8,
        )
        self.assertEqual(
            summary["task_scope"],
            {
                "primary_task": "T2_localization",
                "mask_threshold": 0.5,
            },
        )
        self.assertFalse(
            _contains_key(
                summary,
                {
                    "score",
                    "decision",
                    "detection",
                    "auroc",
                    "average_precision",
                    "paired_score_delta",
                    "paired_ranking_accuracy",
                },
            )
        )

    def test_summary_uses_latest_physical_row_for_each_expected_id(self):
        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        expected = [{"sample_id": "real"}, {"sample_id": "forged"}]
        rows = [
            {"id": "real", "status": "error"},
            _row(
                "real",
                task_id="task",
                kind="real",
                metrics=real,
            ),
            _row(
                "forged",
                task_id="task",
                kind="forged",
                metrics=forged,
            ),
            _row(
                "unexpected",
                task_id="other",
                kind="real",
                metrics=real,
            ),
        ]

        summary = summarize_catnet_results(rows, expected)

        self.assertEqual(summary["coverage"]["result_images"], 2)
        self.assertEqual(summary["coverage"]["valid_images"], 2)
        self.assertEqual(summary["coverage"]["error_images"], 0)
        self.assertEqual(summary["coverage"]["missing_images"], 0)

    def test_summary_rejects_non_primary_mask_threshold(self):
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_catnet_results([], [], mask_threshold=0.4)

    def test_pair_bootstrap_preserves_pairs_and_reports_only_t2(self):
        real_a = _metrics(
            [[0.6, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged_a = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real_b = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged_b = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        pairs = [
            {
                "real": _row(
                    "real-a",
                    task_id="a",
                    kind="real",
                    metrics=real_a,
                ),
                "forged": _row(
                    "forged-a",
                    task_id="a",
                    kind="forged",
                    metrics=forged_a,
                ),
            },
            {
                "real": _row(
                    "real-b",
                    task_id="b",
                    kind="real",
                    metrics=real_b,
                ),
                "forged": _row(
                    "forged-b",
                    task_id="b",
                    kind="forged",
                    metrics=forged_b,
                ),
            },
        ]

        summary = summarize_catnet_pair_slice(
            pairs,
            iterations=100,
            seed=7,
        )
        repeated = summarize_catnet_pair_slice(
            pairs,
            iterations=100,
            seed=7,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["images"], 4)
        self.assertEqual(summary["pixel_f1_macro_at_0_5"]["estimate"], 0.5)
        self.assertAlmostEqual(
            summary["pixel_f1_micro_at_0_5"]["estimate"],
            2 / 3,
        )
        self.assertEqual(
            summary[
                "real_predicted_positive_fraction_micro_at_0_5"
            ]["estimate"],
            1 / 8,
        )
        self.assertFalse(
            _contains_key(
                summary,
                {
                    "score",
                    "decision",
                    "detection",
                    "auroc",
                    "average_precision",
                },
            )
        )

    def test_pair_bootstrap_rejects_empty_input_and_bad_iterations(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_catnet_pair_slice([], iterations=10, seed=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            summarize_catnet_pair_slice(
                [
                    {
                        "real": {},
                        "forged": {},
                    }
                ],
                iterations=0,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
