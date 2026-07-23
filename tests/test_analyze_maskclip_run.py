import unittest

import numpy as np

from eval.opensource.analyze_maskclip_run import (
    Pair,
    histogram_best_metrics,
    summarize_pair_slice,
)


def result(kind, score, *, pixel_ap=0.5, tp=1, fp=0, fn=0):
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "kind": kind,
        "score": score,
        "localization": {
            "native": {
                "pixel_ap": pixel_ap,
                "f1": 2 * tp / denominator if denominator else 0.0,
                "iou": tp / union if union else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "target_positive_pixels": tp + fn,
                "pixels": 100,
            }
        },
    }


class AnalyzeMaskCLIPRunTest(unittest.TestCase):
    def test_pair_bootstrap_preserves_pairing_and_direction(self):
        pairs = [
            Pair(
                task_id=f"t{index}",
                domain="lodging",
                real=result("real", 0.1 + index * 0.01),
                forged=result("forged", 0.8 + index * 0.01),
                input_row={},
            )
            for index in range(4)
        ]
        summary = summarize_pair_slice(pairs, iterations=50, seed=7)
        self.assertEqual(summary["pairs"], 4)
        self.assertEqual(summary["auroc"]["estimate"], 1.0)
        self.assertEqual(summary["paired_ranking_accuracy"]["estimate"], 1.0)
        self.assertEqual(summary["paired_sign_test"]["wins"], 4)
        self.assertEqual(summary["paired_sign_test"]["losses"], 0)

    def test_histogram_best_metrics_finds_separating_threshold(self):
        scores = np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        target = np.asarray([[1, 1], [0, 0]], dtype=bool)
        best, all_hist, positive_hist = histogram_best_metrics(
            scores,
            target,
            bins=256,
        )
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(best["iou"], 1.0)
        self.assertEqual(int(all_hist.sum()), 4)
        self.assertEqual(int(positive_hist.sum()), 2)


if __name__ == "__main__":
    unittest.main()
