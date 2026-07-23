import unittest

from eval.opensource.trufor_metrics import summarize_trufor_results


class TruForMetricsTest(unittest.TestCase):
    def test_summary_uses_paired_forged_direction_and_native_metrics(self):
        expected = [{"sample_id": "real"}, {"sample_id": "forged"}]
        native_real = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 4,
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
            "mcc": None,
            "pixel_ap": None,
            "predicted_positive_fraction": 0.0,
            "score_mean": 0.1,
            "score_max": 0.2,
        }
        native_forged = {
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
            "score_mean": 0.3,
            "score_max": 0.9,
        }
        rows = [
            {
                "id": "real",
                "task_id": "task",
                "kind": "real",
                "label": 0,
                "status": "ok",
                "score": 0.1,
                "latency_ms": 1,
                "peak_cuda_memory_bytes": 10,
                "localization": {"native": native_real},
                "reliability": {
                    "min": 0.1,
                    "mean": 0.5,
                    "median": 0.5,
                    "p05": 0.2,
                    "p95": 0.8,
                    "max": 0.9,
                },
            },
            {
                "id": "forged",
                "task_id": "task",
                "kind": "forged",
                "label": 1,
                "status": "ok",
                "score": 0.9,
                "latency_ms": 2,
                "peak_cuda_memory_bytes": 20,
                "localization": {"native": native_forged},
                "reliability": {
                    "min": 0.1,
                    "mean": 0.6,
                    "median": 0.6,
                    "p05": 0.2,
                    "p95": 0.9,
                    "max": 1.0,
                },
            },
        ]
        summary = summarize_trufor_results(
            rows,
            expected,
            classification_threshold=0.5,
            mask_threshold=0.5,
        )
        self.assertEqual(summary["coverage"]["valid_images"], 2)
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertEqual(summary["paired_score_delta"]["mean"], 0.8)
        self.assertEqual(
            summary["localization_forged"]["native"]["micro_at_threshold"]["iou"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
