import unittest

import numpy as np

from eval.opensource.imlvit_metrics import (
    binary_pixel_metrics,
    binary_pixel_metrics_strict,
    summarize_imlvit_pair_slice,
    summarize_imlvit_results,
)


def _metrics(scores, target):
    return binary_pixel_metrics_strict(
        np.asarray(scores),
        np.asarray(target),
    )


def _row(
    row_id,
    *,
    task_id,
    kind,
    metrics,
    model_1024=None,
    latency_ms=2.0,
    peak_cuda_memory_bytes=100,
):
    return {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "status": "ok",
        # This intentionally has no defined semantics for IML-ViT.  Summary
        # code must ignore it instead of deriving an unofficial T1 result.
        "score": object(),
        "localization": {
            "model_1024": (
                metrics if model_1024 is None else model_1024
            ),
            "native": metrics,
        },
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


class _RejectT1Access(dict):
    def get(self, key, default=None):
        if key in {"score", "decision", "detection"}:
            raise AssertionError(f"T1 field {key!r} was read")
        return super().get(key, default)


class IMLViTMetricsTest(unittest.TestCase):
    def test_binary_metrics_use_float32_and_strict_greater_than(self):
        next_float32 = float(
            np.nextafter(np.float32(0.5), np.float32(1.0))
        )
        metrics = binary_pixel_metrics(
            np.asarray(
                [[0.5 + 1e-9, next_float32], [0.1, 0.9]],
                dtype=np.float64,
            ),
            np.asarray([[1, 1], [0, 0]]),
        )

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["probability_dtype"], "float32")
        # 0.5 + 1e-9 rounds to exactly 0.5 in float32 and is negative.
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["predicted_positive_pixels"], 2)
        self.assertEqual(metrics["predicted_positive_fraction"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["mcc"], 0.0)

    def test_binary_metrics_report_forged_ap_and_null_real_ap(self):
        forged = _metrics(
            [[0.9, 0.8], [0.7, 0.1]],
            [[1, 0], [1, 0]],
        )
        real = _metrics(
            [[0.9, 0.8], [0.7, 0.1]],
            [[0, 0], [0, 0]],
        )

        self.assertAlmostEqual(forged["pixel_ap"], 5 / 6)
        self.assertIsNone(real["pixel_ap"])
        self.assertEqual(real["fp"], 3)
        self.assertEqual(real["predicted_positive_fraction"], 0.75)

    def test_binary_metrics_reject_invalid_inputs_and_threshold_drift(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            binary_pixel_metrics_strict(
                np.zeros((2, 2)),
                np.zeros((1, 2)),
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            binary_pixel_metrics_strict(np.asarray([]), np.asarray([]))
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
                threshold=0.4,
            )

    def test_summary_is_t2_only_with_two_spaces_macro_micro_and_bootstrap(self):
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
            {"sample_id": "error"},
            {"sample_id": "missing"},
        ]
        rows = [
            {"id": "real-a", "status": "error"},
            _RejectT1Access(
                _row(
                    "real-a",
                    task_id="a",
                    kind="real",
                    metrics=real_a,
                )
            ),
            _RejectT1Access(
                _row(
                    "forged-a",
                    task_id="a",
                    kind="forged",
                    metrics=forged_a,
                )
            ),
            _RejectT1Access(
                _row(
                    "real-b",
                    task_id="b",
                    kind="real",
                    metrics=real_b,
                )
            ),
            _RejectT1Access(
                _row(
                    "forged-b",
                    task_id="b",
                    kind="forged",
                    metrics=forged_b,
                )
            ),
            {"id": "error", "status": "error"},
            _row(
                "unexpected",
                task_id="other",
                kind="real",
                metrics=real_a,
            ),
        ]

        summary = summarize_imlvit_results(
            rows,
            expected,
            bootstrap_samples=100,
            seed=7,
        )
        repeated = summarize_imlvit_results(
            rows,
            expected,
            bootstrap_samples=100,
            seed=7,
        )

        self.assertEqual(summary, repeated)
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
        self.assertEqual(
            summary["paired_coverage"],
            {
                "complete_pairs": 2,
                "paired_images": 4,
                "unpaired_valid_images": 0,
            },
        )
        self.assertFalse(summary["task_scope"]["valid_for_t1"])
        self.assertTrue(summary["task_scope"]["valid_for_t2"])
        self.assertEqual(summary["task_scope"]["threshold_operator"], ">")
        for space in ("model_1024", "native"):
            forged = summary["localization_forged"][space]
            real = summary["localization_real"][space]
            bootstrap = summary["pair_bootstrap"][space]
            self.assertEqual(forged["images"], 2)
            self.assertEqual(real["images"], 2)
            self.assertEqual(forged["pixel_ap"]["count"], 2)
            self.assertIsNone(real["pixel_ap"])
            self.assertEqual(forged["macro_at_threshold"]["f1"], 0.5)
            self.assertAlmostEqual(
                forged["micro_at_threshold"]["f1"],
                2 / 3,
            )
            self.assertEqual(
                real["macro_at_threshold"][
                    "false_positive_area_fraction"
                ],
                0.125,
            )
            self.assertEqual(
                real["micro_at_threshold"][
                    "false_positive_area_fraction"
                ],
                1 / 8,
            )
            self.assertEqual(
                bootstrap["pixel_f1_macro_at_0_5"]["estimate"],
                0.5,
            )
            self.assertAlmostEqual(
                bootstrap["pixel_f1_micro_at_0_5"]["estimate"],
                2 / 3,
            )
            self.assertEqual(
                bootstrap[
                    "real_false_positive_area_fraction_micro_at_0_5"
                ]["estimate"],
                1 / 8,
            )
            self.assertEqual(
                bootstrap["forged_micro_at_threshold"]["tp"],
                1,
            )
            self.assertEqual(
                bootstrap["forged_micro_at_threshold"]["fn"],
                1,
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

    def test_summary_supports_an_unpaired_preflight_without_fake_t1(self):
        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        summary = summarize_imlvit_results(
            [_row("real", task_id="task", kind="real", metrics=real)],
            bootstrap_samples=10,
            seed=1,
        )

        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 0)
        self.assertEqual(summary["paired_coverage"]["unpaired_valid_images"], 1)
        self.assertIsNone(summary["pair_bootstrap"]["model_1024"])
        self.assertIsNone(summary["pair_bootstrap"]["native"])

    def test_summary_rejects_contract_drift_and_real_ap_leakage(self):
        forged = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_imlvit_results(
                [],
                mask_threshold=0.4,
                bootstrap_samples=10,
            )

        bad_operator = dict(forged)
        bad_operator["threshold_operator"] = ">="
        with self.assertRaisesRegex(ValueError, "expected '>'"):
            summarize_imlvit_results(
                [
                    _row(
                        "forged",
                        task_id="task",
                        kind="forged",
                        metrics=bad_operator,
                    )
                ],
                bootstrap_samples=10,
            )

        leaked_ap = dict(real)
        leaked_ap["pixel_ap"] = 0.0
        with self.assertRaisesRegex(ValueError, "pixel AP must be null"):
            summarize_imlvit_results(
                [
                    _row(
                        "real",
                        task_id="task",
                        kind="real",
                        metrics=leaked_ap,
                    )
                ],
                bootstrap_samples=10,
            )

        missing_model_space = _row(
            "forged",
            task_id="task",
            kind="forged",
            metrics=forged,
        )
        del missing_model_space["localization"]["model_1024"]
        with self.assertRaisesRegex(ValueError, "model_1024"):
            summarize_imlvit_results(
                [missing_model_space],
                bootstrap_samples=10,
            )

    def test_pair_bootstrap_validates_arguments_and_schema(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_imlvit_pair_slice([], iterations=10, seed=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            summarize_imlvit_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=0,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            summarize_imlvit_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
                localization_space="wrong",
            )
        with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
            summarize_imlvit_results([], bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
