import copy
import unittest

import numpy as np

from eval.opensource.relayformer_metrics import (
    binary_pixel_metrics,
    binary_pixel_metrics_strict,
    summarize_relayformer_pair_slice,
    summarize_relayformer_results,
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
        # RelayFormer has no official T1 semantics.  Tests below ensure summary
        # code does not inspect this deliberately poisonous field.
        "score": object(),
        "localization": {
            "model_1024": metrics if model_1024 is None else model_1024,
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
        if key in {"score", "decision", "detection"}:
            raise AssertionError(f"T1 field {key!r} was read")
        return super().get(key, default)


class RelayFormerMetricsTest(unittest.TestCase):
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
        # The first value rounds to exactly 0.5 in float32 and stays negative.
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
        all_positive = _metrics(
            [[0.2, 0.1], [0.4, 0.3]],
            [[1, 1], [1, 1]],
        )

        self.assertAlmostEqual(forged["pixel_ap"], 5 / 6)
        self.assertIsNone(real["pixel_ap"])
        self.assertEqual(real["fp"], 3)
        self.assertEqual(real["predicted_positive_fraction"], 0.75)
        self.assertEqual(all_positive["pixel_ap"], 1.0)

    def test_binary_metrics_reject_invalid_inputs_and_threshold_drift(self):
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
        with self.assertRaisesRegex(ValueError, "empty"):
            binary_pixel_metrics_strict(
                np.zeros((0, 0)),
                np.zeros((0, 0)),
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
        valid_rows = [
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
        ]
        expected = [
            *[_expected(row) for row in valid_rows],
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
            *valid_rows,
            {
                "id": "error",
                "task_id": "error-task",
                "kind": "real",
                "label": 0,
                "domain": "lodging",
                "status": "error",
            },
            _row(
                "unexpected",
                task_id="other",
                kind="real",
                metrics=real_a,
            ),
        ]

        summary = summarize_relayformer_results(
            rows,
            expected,
            bootstrap_samples=100,
            seed=7,
        )
        repeated = summarize_relayformer_results(
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
        scope = summary["task_scope"]
        self.assertFalse(scope["valid_for_t1"])
        self.assertTrue(scope["valid_for_t2"])
        self.assertEqual(scope["primary_localization_space"], "native")
        self.assertEqual(scope["auxiliary_localization_space"], "model_1024")
        self.assertEqual(
            scope["auxiliary_localization_extent"],
            "resized_valid_content_excluding_right_bottom_padding",
        )
        self.assertEqual(scope["threshold_operator"], ">")
        for space in ("model_1024", "native"):
            forged = summary["localization_forged"][space]
            real = summary["localization_real"][space]
            bootstrap = summary["pair_bootstrap"][space]
            self.assertEqual(forged["images"], 2)
            self.assertEqual(real["images"], 2)
            self.assertEqual(forged["pixel_ap"]["count"], 2)
            self.assertIsNone(real["pixel_ap"])
            self.assertTrue(
                {"precision", "recall", "f1", "iou"}.issubset(
                    forged["macro_at_threshold"]
                )
            )
            self.assertTrue(
                {"precision", "recall", "f1", "iou"}.issubset(
                    forged["micro_at_threshold"]
                )
            )
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

    def test_summary_supports_unpaired_preflight_and_retry_history(self):
        real_metrics = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        valid = _row(
            "real",
            task_id="task",
            kind="real",
            metrics=real_metrics,
        )
        summary = summarize_relayformer_results(
            [
                {"id": "real", "status": "error"},
                valid,
            ],
            [_expected(valid)],
            bootstrap_samples=10,
            seed=1,
        )

        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 0)
        self.assertEqual(summary["paired_coverage"]["unpaired_valid_images"], 1)
        self.assertIsNone(summary["pair_bootstrap"]["model_1024"])
        self.assertIsNone(summary["pair_bootstrap"]["native"])

    def test_summary_rejects_metric_contract_drift(self):
        forged = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        with self.assertRaisesRegex(ValueError, "fixed 0.5 threshold"):
            summarize_relayformer_results(
                [],
                mask_threshold=0.4,
                bootstrap_samples=10,
            )

        changes = (
            ("threshold_operator", ">=", "expected '>'"),
            ("probability_dtype", "float64", "probability dtype"),
            ("tp", 2, "counts do not sum"),
            ("f1", 0.25, "f1 is inconsistent"),
            ("score_mean", np.nan, "probability summary"),
            ("score_max", 1.1, "probability summary"),
        )
        for key, value, error in changes:
            with self.subTest(key=key):
                bad = dict(forged)
                bad[key] = value
                with self.assertRaisesRegex(ValueError, error):
                    summarize_relayformer_results(
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

        leaked_ap = dict(real)
        leaked_ap["pixel_ap"] = 0.0
        with self.assertRaisesRegex(ValueError, "pixel AP must be null"):
            summarize_relayformer_results(
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
            summarize_relayformer_results(
                [missing_model_space],
                bootstrap_samples=10,
            )

    def test_summary_rejects_identity_label_duplicates_and_nonfinite_runtime(self):
        metrics = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        real_a = _row(
            "real-a",
            task_id="task",
            kind="real",
            metrics=metrics,
        )
        real_b = _row(
            "real-b",
            task_id="task",
            kind="real",
            metrics=metrics,
        )

        bad_label = copy.deepcopy(real_a)
        bad_label["label"] = 1
        with self.assertRaisesRegex(ValueError, "kind/label mismatch"):
            summarize_relayformer_results([bad_label], bootstrap_samples=10)

        with self.assertRaisesRegex(ValueError, "duplicate real row"):
            summarize_relayformer_results(
                [real_a, real_b],
                bootstrap_samples=10,
            )

        bad_runtime = copy.deepcopy(real_a)
        bad_runtime["latency_ms"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            summarize_relayformer_results(
                [bad_runtime],
                bootstrap_samples=10,
            )

        mismatched_expected = _expected(real_a)
        mismatched_expected["task_id"] = "different"
        with self.assertRaisesRegex(ValueError, "does not match expected"):
            summarize_relayformer_results(
                [real_a],
                [mismatched_expected],
                bootstrap_samples=10,
            )

        expected = _expected(real_a)
        with self.assertRaisesRegex(ValueError, "duplicate expected"):
            summarize_relayformer_results(
                [real_a],
                [expected, dict(expected)],
                bootstrap_samples=10,
            )

    def test_pair_bootstrap_validates_arguments_and_pair_schema(self):
        with self.assertRaisesRegex(ValueError, "pair slice is empty"):
            summarize_relayformer_pair_slice([], iterations=10, seed=1)
        for iterations in (0, 1.5, True):
            with self.subTest(iterations=iterations):
                with self.assertRaisesRegex(ValueError, "iterations"):
                    summarize_relayformer_pair_slice(
                        [{"real": {}, "forged": {}}],
                        iterations=iterations,
                        seed=1,
                    )
        with self.assertRaisesRegex(ValueError, "bootstrap seed"):
            summarize_relayformer_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1.5,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            summarize_relayformer_pair_slice(
                [{"real": {}, "forged": {}}],
                iterations=1,
                seed=1,
                localization_space="wrong",
            )
        with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
            summarize_relayformer_results([], bootstrap_samples=0)

        real_metrics = _metrics(
            [[0.1, 0.1], [0.1, 0.1]],
            [[0, 0], [0, 0]],
        )
        forged_metrics = _metrics(
            [[0.9, 0.1], [0.1, 0.1]],
            [[1, 0], [0, 0]],
        )
        real = _row(
            "real",
            task_id="task",
            kind="real",
            metrics=real_metrics,
        )
        forged = _row(
            "forged",
            task_id="task",
            kind="forged",
            metrics=forged_metrics,
        )
        with self.assertRaisesRegex(ValueError, "no 'forged'"):
            summarize_relayformer_pair_slice(
                [{"real": real}],
                iterations=1,
                seed=1,
            )

        wrong_task = copy.deepcopy(forged)
        wrong_task["task_id"] = "other"
        with self.assertRaisesRegex(ValueError, "mismatched task IDs"):
            summarize_relayformer_pair_slice(
                [{"real": real, "forged": wrong_task}],
                iterations=1,
                seed=1,
            )

        wrong_kind = copy.deepcopy(forged)
        wrong_kind["kind"] = "real"
        wrong_kind["label"] = 0
        with self.assertRaisesRegex(ValueError, "wrong kind"):
            summarize_relayformer_pair_slice(
                [{"real": real, "forged": wrong_kind}],
                iterations=1,
                seed=1,
            )

        duplicate_pair = [
            {"real": real, "forged": forged},
            {
                "real": {**real, "id": "real-two"},
                "forged": {**forged, "id": "forged-two"},
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate task pair"):
            summarize_relayformer_pair_slice(
                duplicate_pair,
                iterations=1,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
