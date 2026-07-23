import unittest

import numpy as np

from eval.opensource.maskclip_metrics import (
    binary_pixel_metrics,
    image_detection_metrics,
    summarize_results,
)


class MaskCLIPMetricsTest(unittest.TestCase):
    def test_binary_metrics_are_exact_for_toy_map(self):
        scores = np.asarray([[0.9, 0.8], [0.7, 0.1]], dtype=np.float32)
        target = np.asarray([[1, 0], [1, 0]], dtype=bool)
        metrics = binary_pixel_metrics(scores, target, 0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["iou"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["pixel_ap"], 5 / 6)

    def test_detection_metrics_use_higher_as_more_forged(self):
        rows = [
            {"status": "ok", "label": 0, "score": 0.1},
            {"status": "ok", "label": 0, "score": 0.2},
            {"status": "ok", "label": 1, "score": 0.8},
            {"status": "ok", "label": 1, "score": 0.9},
        ]
        metrics = image_detection_metrics(rows, 0.5)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["paired"] if "paired" in metrics else None, None)

    def test_summary_uses_latest_result_for_each_id(self):
        expected = [
            {"sample_id": "a"},
            {"sample_id": "b"},
        ]
        base_localization = {
            "model_512": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 3,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 1.0,
                "mcc": 1.0,
                "pixel_ap": 1.0,
                "predicted_positive_fraction": 0.25,
                "score_mean": 0.25,
                "score_max": 1.0,
            },
            "native": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 3,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 1.0,
                "mcc": 1.0,
                "pixel_ap": 1.0,
                "predicted_positive_fraction": 0.25,
                "score_mean": 0.25,
                "score_max": 1.0,
            },
        }
        rows = [
            {"id": "a", "status": "error"},
            {
                "id": "a",
                "task_id": "t",
                "kind": "real",
                "label": 0,
                "status": "ok",
                "score": 0.1,
                "latency_ms": 1,
                "localization": {
                    "model_512": {
                        **base_localization["model_512"],
                        "target_positive_pixels": 0,
                    }
                },
            },
            {
                "id": "b",
                "task_id": "t",
                "kind": "forged",
                "label": 1,
                "status": "ok",
                "score": 0.9,
                "latency_ms": 2,
                "localization": base_localization,
            },
        ]
        summary = summarize_results(
            rows,
            expected,
            classification_threshold=0.5,
            mask_threshold=0.5,
        )
        self.assertEqual(summary["coverage"]["valid_images"], 2)
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertEqual(summary["paired_score_delta"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
