from __future__ import annotations

import copy
import unittest

import numpy as np

from eval.opensource.dinov3_iml_metrics import (
    binary_pixel_metrics,
    binary_pixel_metrics_strict,
    summarize_dinov3_iml_pair_slice,
    summarize_dinov3_iml_results,
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
    model_512=None,
    domain="lodging",
    latency_ms=2.0,
    peak_cuda_memory_bytes=100,
):
    return {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": domain,
        "status": "ok",
        "score": object(),
        "localization": {
            "model_512": metrics if model_512 is None else model_512,
            "native": metrics,
        },
        "latency_ms": latency_ms,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }


def _expected(row):
    return {
        "sample_id": row["id"],
        "task_id": row["task_id"],
        "kind": row["kind"],
        "label": row["label"],
        "domain": row["domain"],
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
        if key in {
            "score",
            "decision",
            "detection",
            "classification",
            "image_score",
            "image_decision",
        }:
            raise AssertionError(f"T1 field {key!r} was read")
        return super().get(key, default)


class DinoV3IMLMetricsTest(unittest.TestCase):
    def test_binary_metrics_use_float32_and_strict_greater_than(self):
        above = float(np.nextafter(np.float32(0.5), np.float32(1.0)))
        metrics = binary_pixel_metrics(
            np.asarray(
                [[0.5 + 1e-9, above], [0.1, 0.9]],
                dtype=np.float64,
            ),
            np.asarray([[1, 1], [0, 0]]),
        )

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["threshold_operator"], ">")
        self.assertEqual(metrics["probability_dtype"], "float32")
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)

    def test_ap_is_forged_only_and_real_fp_area_is_retained(self):
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
        self.assertEqual(real["predicted_positive_fraction"], 0.75)

    def test_binary_metrics_reject_invalid_maps_and_threshold_drift(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            binary_pixel_metrics_strict(
                np.zeros((2, 2)),
                np.zeros((1, 2)),
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            binary_pixel_metrics_strict(
                np.zeros((1, 1, 1)),
                np.zeros((1, 1, 1)),
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
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            binary_pixel_metrics_strict(
                np.asarray([[0.1]]),
                np.asarray([[0]]),
                threshold=0.4,
            )
        with self.assertRaisesRegex(ValueError, "DINOv3-IML localization"):
            binary_pixel_metrics_strict(
                np.asarray([[0.1]]),
                np.asarray([[0]]),
                threshold=0.4,
            )

    def test_summary_is_t2_only_in_model_and_native_spaces(self):
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
        valid = [
            _RejectT1Access(_row("real-a", task_id="a", kind="real", metrics=real_a)),
            _RejectT1Access(
                _row(
                    "forged-a",
                    task_id="a",
                    kind="forged",
                    metrics=forged_a,
                )
            ),
            _RejectT1Access(_row("real-b", task_id="b", kind="real", metrics=real_b)),
            _RejectT1Access(
                _row(
                    "forged-b",
                    task_id="b",
                    kind="forged",
                    metrics=forged_b,
                )
            ),
        ]
        expected = [
            *[_expected(row) for row in valid],
            {
                "sample_id": "error",
                "task_id": "error-task",
                "kind": "real",
                "label": 0,
                "domain": "lodging",
            },
            {
                "sample_id": "missing",
                "task_id": "missing-task",
                "kind": "forged",
                "label": 1,
                "domain": "lodging",
            },
        ]
        rows = [
            {"id": "real-a", "status": "error"},
            *valid,
            {
                "id": "error",
                "task_id": "error-task",
                "kind": "real",
                "label": 0,
                "domain": "lodging",
                "status": "error",
            },
        ]

        summary = summarize_dinov3_iml_results(
            rows,
            expected,
            bootstrap_samples=100,
            seed=7,
        )
        repeated = summarize_dinov3_iml_results(
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
            summary["task_scope"],
            {
                "primary_task": "T2_localization",
                "valid_for_t1": False,
                "valid_for_t2": True,
                "primary_localization_space": "native",
                "auxiliary_localization_space": "model_512",
                "localization_semantics": (
                    "dinov3_iml_sigmoid_manipulation_probability_float32"
                ),
                "probability_dtype": "float32",
                "mask_threshold": 0.5,
                "threshold_operator": ">",
                "model_space_probability_source": (
                    "sigmoid_bilinear_align_corners_false_" "seg_head_logits_32_to_512"
                ),
                "native_probability_source": (
                    "bilinear_align_corners_false_resize_of_" "model_512_probability"
                ),
            },
        )
        for space in ("model_512", "native"):
            forged = summary["localization_forged"][space]
            real = summary["localization_real"][space]
            bootstrap = summary["pair_bootstrap"][space]
            self.assertEqual(forged["images"], 2)
            self.assertEqual(real["images"], 2)
            self.assertEqual(forged["macro_at_threshold"]["f1"], 0.5)
            self.assertAlmostEqual(
                forged["micro_at_threshold"]["f1"],
                2 / 3,
            )
            self.assertEqual(
                real["micro_at_threshold"]["false_positive_area_fraction"],
                1 / 8,
            )
            self.assertEqual(
                bootstrap["pixel_f1_macro_at_0_5"]["estimate"],
                0.5,
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

    def test_model_and_native_spaces_are_aggregated_independently(self):
        real_native = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        real_model = _metrics(
            [[0.9, 0.9], [0.9, 0.9]],
            [[0, 0], [0, 0]],
        )
        forged_native = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        forged_model = _metrics(
            [[0.1, 0.9], [0.9, 0.9]],
            [[1, 0], [0, 0]],
        )
        rows = [
            _row(
                "real",
                task_id="task",
                kind="real",
                metrics=real_native,
                model_512=real_model,
            ),
            _row(
                "forged",
                task_id="task",
                kind="forged",
                metrics=forged_native,
                model_512=forged_model,
            ),
        ]
        summary = summarize_dinov3_iml_results(
            rows,
            bootstrap_samples=20,
            seed=3,
        )

        self.assertEqual(
            summary["localization_forged"]["native"]["micro_at_threshold"]["f1"],
            1.0,
        )
        self.assertEqual(
            summary["localization_forged"]["model_512"]["micro_at_threshold"]["f1"],
            0.0,
        )
        self.assertEqual(
            summary["localization_real"]["native"]["micro_at_threshold"][
                "false_positive_area_fraction"
            ],
            0.0,
        )
        self.assertEqual(
            summary["localization_real"]["model_512"]["micro_at_threshold"][
                "false_positive_area_fraction"
            ],
            1.0,
        )
        self.assertNotEqual(
            summary["pair_bootstrap"]["native"]["pixel_f1_micro_at_0_5"]["estimate"],
            summary["pair_bootstrap"]["model_512"]["pixel_f1_micro_at_0_5"]["estimate"],
        )

    def test_summary_supports_unpaired_preflight_and_retry_history(self):
        metrics = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        valid = _row(
            "forged",
            task_id="task",
            kind="forged",
            metrics=metrics,
        )
        summary = summarize_dinov3_iml_results(
            [{"id": "forged", "status": "error"}, valid],
            [_expected(valid)],
            bootstrap_samples=10,
            seed=1,
        )

        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 0)
        self.assertEqual(summary["paired_coverage"]["unpaired_valid_images"], 1)
        self.assertIsNone(summary["pair_bootstrap"]["model_512"])
        self.assertIsNone(summary["pair_bootstrap"]["native"])

    def test_summary_rejects_metric_and_identity_drift(self):
        forged = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        with self.assertRaisesRegex(
            ValueError,
            "DINOv3-IML localization uses the fixed 0.5 threshold",
        ):
            summarize_dinov3_iml_results(
                [],
                mask_threshold=0.4,
                bootstrap_samples=10,
            )
        bad = copy.deepcopy(forged)
        bad["threshold_operator"] = ">="
        with self.assertRaisesRegex(ValueError, "expected '>'"):
            summarize_dinov3_iml_results(
                [
                    _row(
                        "forged",
                        task_id="task",
                        kind="forged",
                        metrics=bad,
                    )
                ],
                bootstrap_samples=10,
            )

        leaked_ap = copy.deepcopy(real)
        leaked_ap["pixel_ap"] = 0.0
        with self.assertRaisesRegex(ValueError, "pixel AP must be null"):
            summarize_dinov3_iml_results(
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

        real_row = _row(
            "real",
            task_id="task",
            kind="real",
            metrics=real,
        )
        bad_label = copy.deepcopy(real_row)
        bad_label["label"] = 1
        with self.assertRaisesRegex(ValueError, "kind/label mismatch"):
            summarize_dinov3_iml_results(
                [bad_label],
                bootstrap_samples=10,
            )

    def test_pair_bootstrap_validates_arguments_and_pair_schema(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_dinov3_iml_pair_slice([], iterations=10, seed=1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            summarize_dinov3_iml_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
                localization_space="wrong",
            )
        with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
            summarize_dinov3_iml_results([], bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
