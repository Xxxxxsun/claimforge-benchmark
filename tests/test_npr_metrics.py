import copy
import json
import math
import unittest

from eval.opensource.npr_metrics import (
    FIXED_THRESHOLD,
    summarize_npr_pair_slice,
    summarize_npr_raw_logit_diagnostic,
    summarize_npr_results,
    npr_detection_metrics_strict,
)


def _expected(
    sample_id: str,
    *,
    task_id: str,
    kind: str,
    domain: str = "lodging",
) -> dict:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": domain,
    }


def _row(
    row_id: str,
    *,
    task_id: str,
    kind: str,
    score: float = 0.5,
    domain: str = "lodging",
    visibility: str = "partial",
    visible_fraction: float = 0.25,
    status: str = "ok",
    peak_cuda_memory_bytes: int | None = 17,
) -> dict:
    row = {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": domain,
        "edit_visibility": visibility,
        "edit_visible_gt_fraction": visible_fraction,
        "status": status,
        "latency_ms": 3.0,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    if status == "ok":
        row["ai_score"] = score
    return row


def _pair(
    task_id: str,
    real_score: float,
    forged_score: float,
    *,
    domain: str = "lodging",
    visibility: str = "partial",
    visible_fraction: float = 0.25,
) -> dict:
    return {
        "real": _row(
            f"{task_id}-real",
            task_id=task_id,
            kind="real",
            score=real_score,
            domain=domain,
            visibility=visibility,
            visible_fraction=visible_fraction,
        ),
        "forged": _row(
            f"{task_id}-forged",
            task_id=task_id,
            kind="forged",
            score=forged_score,
            domain=domain,
            visibility=visibility,
            visible_fraction=visible_fraction,
        ),
    }


class NprDetectionMetricsTests(unittest.TestCase):
    def test_probability_boundaries_and_strict_threshold_equality(self):
        result = npr_detection_metrics_strict(
            [
                {"status": "ok", "label": 0, "ai_score": 0.0},
                {"status": "ok", "label": 0, "ai_score": 0.5},
                {"status": "ok", "label": 1, "ai_score": 0.5},
                {"status": "ok", "label": 1, "ai_score": 1.0},
            ]
        )

        self.assertEqual(result["threshold"], FIXED_THRESHOLD)
        self.assertEqual(result["threshold_operator"], ">")
        self.assertEqual(
            (result["tp"], result["fp"], result["fn"], result["tn"]),
            (1, 0, 1, 2),
        )
        self.assertEqual(result["accuracy_at_0_5"], 0.75)
        self.assertEqual(result["balanced_accuracy_at_0_5"], 0.75)
        self.assertEqual(result["precision_at_0_5"], 1.0)
        self.assertEqual(result["recall_at_0_5"], 0.5)
        self.assertAlmostEqual(result["f1_at_0_5"], 2 / 3)
        self.assertEqual(result["specificity_at_0_5"], 1.0)
        self.assertEqual(result["auroc"], 0.875)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertNotIn("at_2", json.dumps(result, sort_keys=True))

    def test_scores_must_be_finite_probabilities(self):
        invalid = (
            (math.nan, "not finite"),
            (math.inf, "not finite"),
            (-1e-12, r"outside \[0, 1\]"),
            (1.000000000001, r"outside \[0, 1\]"),
            ("0.5", "real number"),
            (True, "real number"),
            (None, "real number"),
        )
        for score, message in invalid:
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, message):
                    npr_detection_metrics_strict(
                        [
                            {
                                "status": "ok",
                                "label": 1,
                                "ai_score": score,
                            }
                        ]
                    )

        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            npr_detection_metrics_strict(
                [{"status": "error", "ai_score": 1.1}]
            )

    def test_status_label_and_threshold_are_strict(self):
        with self.assertRaisesRegex(ValueError, "invalid status"):
            npr_detection_metrics_strict(
                [{"status": "banana", "label": 0, "ai_score": 0.0}]
            )
        with self.assertRaisesRegex(ValueError, "integer 0/1"):
            npr_detection_metrics_strict(
                [{"status": "ok", "label": True, "ai_score": 0.1}]
            )
        for threshold in (0.0, 0.500000000001, 1.0):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError,
                    "fixed threshold 0.5",
                ):
                    npr_detection_metrics_strict([], threshold=threshold)

    def test_real_only_higher_quantile_uses_strict_comparison(self):
        real_scores = [index / 20 for index in range(21)]
        forged_scores = [0.95, 0.975, 1.0]
        rows = [
            {"status": "ok", "label": 0, "ai_score": score}
            for score in real_scores
        ] + [
            {"status": "ok", "label": 1, "ai_score": score}
            for score in forged_scores
        ]

        result = npr_detection_metrics_strict(rows)

        self.assertEqual(result["tpr_at_fpr_5_percent_threshold"], 0.95)
        self.assertEqual(result["tpr_at_fpr_5_percent"], 2 / 3)
        self.assertEqual(
            result["tpr_at_fpr_5_percent_actual_fpr"],
            1 / 21,
        )
        self.assertEqual(
            result["tpr_at_fpr_5_percent_threshold_operator"],
            ">",
        )

    def test_one_class_preflight_has_null_ranking_metrics(self):
        result = npr_detection_metrics_strict(
            [{"status": "ok", "label": 1, "ai_score": 0.5}]
        )
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["recall_at_0_5"], 0.0)
        self.assertIsNone(result["specificity_at_0_5"])
        self.assertIsNone(result["balanced_accuracy_at_0_5"])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["tpr_at_fpr_5_percent"])


class NprPairMetricsTests(unittest.TestCase):
    def test_pair_ranking_delta_and_exact_sign_test(self):
        pairs = [
            _pair("a", 0.1, 0.2),
            _pair("b", 0.2, 0.3),
            _pair("c", 0.3, 0.4),
            _pair("d", 0.8, 0.7),
            _pair("e", 0.5, 0.5),
        ]

        result = summarize_npr_pair_slice(
            pairs,
            iterations=20,
            seed=8,
        )

        self.assertEqual(result["paired_ranking"]["wins"], 3)
        self.assertEqual(result["paired_ranking"]["losses"], 1)
        self.assertEqual(result["paired_ranking"]["ties"], 1)
        self.assertEqual(result["paired_ranking"]["strict_accuracy"], 3 / 5)
        self.assertEqual(result["paired_ranking_accuracy"]["estimate"], 3 / 5)
        self.assertEqual(result["paired_sign_test"]["non_ties"], 4)
        self.assertEqual(result["paired_sign_test"]["two_sided_exact_p"], 0.625)
        self.assertAlmostEqual(result["paired_score_delta"]["mean"], 0.04)
        self.assertNotIn("at_2", json.dumps(result, sort_keys=True))

    def test_bootstrap_is_deterministic_and_reselects_fpr_threshold(self):
        pairs = [
            _pair("a", 0.0, 0.5),
            _pair("b", 0.01, 0.5),
            _pair("c", 0.02, 0.5),
            _pair("d", 1.0, 0.5),
        ]
        first = summarize_npr_pair_slice(
            pairs,
            iterations=200,
            seed=99,
        )
        second = summarize_npr_pair_slice(
            pairs,
            iterations=200,
            seed=99,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["tpr_at_fpr_5_percent"]["estimate"],
            0.0,
        )
        self.assertEqual(
            first["tpr_at_fpr_5_percent"]["ci95_percentile"],
            [0.0, 1.0],
        )

    def test_pair_identity_score_and_status_are_strict(self):
        base = _pair("a", 0.1, 0.9)
        changes = (
            ("task IDs", "task_id", "other"),
            ("domains", "domain", "restaurant"),
            ("wrong kind", "kind", "real"),
            (r"outside \[0, 1\]", "ai_score", 1.1),
        )
        for message, key, value in changes:
            broken = copy.deepcopy(base)
            broken["forged"][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_npr_pair_slice(
                        [broken],
                        iterations=2,
                    )

        broken = copy.deepcopy(base)
        broken["forged"]["status"] = "error"
        with self.assertRaisesRegex(ValueError, "not status ok"):
            summarize_npr_pair_slice([broken], iterations=2)

        with self.assertRaisesRegex(ValueError, "duplicate task pair"):
            summarize_npr_pair_slice(
                [copy.deepcopy(base), copy.deepcopy(base)],
                iterations=2,
            )

    def test_pair_visibility_contract_and_arguments_are_strict(self):
        pair = _pair("a", 0.1, 0.9)
        pair["real"]["edit_visibility"] = "tiny"
        with self.assertRaisesRegex(ValueError, "invalid category"):
            summarize_npr_pair_slice([pair], iterations=2)

        pair = _pair("a", 0.1, 0.9)
        pair["real"]["edit_visible_gt_fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "category fraction mismatch"):
            summarize_npr_pair_slice([pair], iterations=2)

        pair = _pair("a", 0.1, 0.9)
        pair["forged"]["edit_visible_gt_fraction"] = 0.3
        with self.assertRaisesRegex(
            ValueError,
            "mismatched edit_visible_gt_fraction",
        ):
            summarize_npr_pair_slice([pair], iterations=2)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            summarize_npr_pair_slice(
                [_pair("a", 0.1, 0.9)],
                iterations=0,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            summarize_npr_pair_slice(
                [_pair("a", 0.1, 0.9)],
                iterations=True,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            summarize_npr_pair_slice(
                [_pair("a", 0.1, 0.9)],
                iterations=2,
                seed=True,
            )


class NprSummaryTests(unittest.TestCase):
    def _complete_fixture(self):
        expected = [
            _expected("a-real", task_id="a", kind="real"),
            _expected("a-forged", task_id="a", kind="forged"),
            _expected(
                "b-real",
                task_id="b",
                kind="real",
                domain="restaurant",
            ),
            _expected(
                "b-forged",
                task_id="b",
                kind="forged",
                domain="restaurant",
            ),
        ]
        rows = [
            _row(
                "a-real",
                task_id="a",
                kind="real",
                score=0.1,
                visibility="partial",
                visible_fraction=0.05,
            ),
            _row(
                "a-forged",
                task_id="a",
                kind="forged",
                score=0.9,
                visibility="partial",
                visible_fraction=0.05,
            ),
            _row(
                "b-real",
                task_id="b",
                kind="real",
                score=0.8,
                domain="restaurant",
                visibility="full",
                visible_fraction=1.0,
                peak_cuda_memory_bytes=None,
            ),
            _row(
                "b-forged",
                task_id="b",
                kind="forged",
                score=0.5,
                domain="restaurant",
                visibility="full",
                visible_fraction=1.0,
                peak_cuda_memory_bytes=None,
            ),
        ]
        return rows, expected

    def test_latest_retry_coverage_slices_and_cpu_memory(self):
        rows, expected = self._complete_fixture()
        history = [
            _row(
                "a-real",
                task_id="a",
                kind="real",
                visibility="partial",
                visible_fraction=0.05,
                status="error",
                peak_cuda_memory_bytes=None,
            ),
            *rows,
        ]
        first = summarize_npr_results(
            history,
            expected,
            bootstrap_samples=30,
            seed=11,
        )
        second = summarize_npr_results(
            history,
            expected,
            bootstrap_samples=30,
            seed=11,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["schema_version"],
            "npr_detection_summary_v1",
        )
        self.assertEqual(first["coverage"]["physical_result_rows"], 5)
        self.assertEqual(first["coverage"]["result_images"], 4)
        self.assertEqual(first["coverage"]["valid_images"], 4)
        self.assertTrue(first["coverage"]["is_complete"])
        self.assertEqual(first["paired_coverage"]["complete_valid_pairs"], 2)
        self.assertEqual(first["paired_ranking"]["wins"], 1)
        self.assertEqual(first["paired_ranking"]["losses"], 1)
        self.assertEqual(
            set(first["by_domain"]),
            {"lodging", "restaurant"},
        )
        self.assertEqual(
            set(first["by_edit_visibility"]),
            {"partial", "full"},
        )
        self.assertEqual(first["pair_bootstrap"]["bootstrap_samples"], 30)
        self.assertEqual(first["peak_cuda_memory_bytes"]["count"], 2)
        self.assertEqual(
            first["peak_cuda_memory_bytes"]["unmeasured_cpu_rows"],
            2,
        )
        self.assertNotIn("at_2", json.dumps(first, sort_keys=True))

    def test_physical_file_order_makes_last_retry_authoritative(self):
        rows, expected = self._complete_fixture()
        latest_error = [
            *rows,
            _row(
                "a-forged",
                task_id="a",
                kind="forged",
                visibility="partial",
                visible_fraction=0.05,
                status="error",
            ),
        ]
        summary = summarize_npr_results(
            latest_error,
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(summary["coverage"]["valid_images"], 3)
        self.assertEqual(summary["coverage"]["error_images"], 1)
        self.assertFalse(summary["coverage"]["is_complete"])
        self.assertEqual(
            summary["paired_coverage"]["complete_valid_pairs"],
            1,
        )

        error_then_success = [latest_error[-1], *rows]
        summary = summarize_npr_results(
            error_then_success,
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(summary["coverage"]["valid_images"], 4)
        self.assertTrue(summary["coverage"]["is_complete"])

    def test_every_physical_retry_is_validated_not_only_latest(self):
        rows, expected = self._complete_fixture()
        invalid_old_score = [
            {**copy.deepcopy(rows[0]), "ai_score": -0.1},
            *rows,
        ]
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            summarize_npr_results(
                invalid_old_score,
                expected,
                bootstrap_samples=2,
            )

        invalid_old_domain = [
            {**copy.deepcopy(rows[0]), "domain": "restaurant"},
            *rows,
        ]
        with self.assertRaisesRegex(
            ValueError,
            "domain.*expected identity",
        ):
            summarize_npr_results(
                invalid_old_domain,
                expected,
                bootstrap_samples=2,
            )

        invalid_old_status = [
            {**copy.deepcopy(rows[0]), "status": "retrying"},
            *rows,
        ]
        with self.assertRaisesRegex(ValueError, "invalid status"):
            summarize_npr_results(
                invalid_old_status,
                expected,
                bootstrap_samples=2,
            )

    def test_physical_retry_visibility_must_remain_pair_consistent(self):
        rows, expected = self._complete_fixture()
        mismatched = [
            _row(
                "a-real",
                task_id="a",
                kind="real",
                score=0.2,
                visibility="none",
                visible_fraction=0.0,
            ),
            *rows,
        ]
        with self.assertRaisesRegex(
            ValueError,
            "physical result rows.*mismatched edit_visibility",
        ):
            summarize_npr_results(
                mismatched,
                expected,
                bootstrap_samples=2,
            )

    def test_ok_performance_fields_and_cpu_none_are_strict(self):
        rows, expected = self._complete_fixture()
        for field, value, message in (
            ("latency_ms", -1.0, "latency_ms is negative"),
            ("latency_ms", math.inf, "latency_ms is not finite"),
            (
                "peak_cuda_memory_bytes",
                -1,
                "peak_cuda_memory_bytes",
            ),
            (
                "peak_cuda_memory_bytes",
                1.5,
                "peak_cuda_memory_bytes",
            ),
        ):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(rows)
                invalid[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    summarize_npr_results(
                        invalid,
                        expected,
                        bootstrap_samples=2,
                    )

        missing_peak = copy.deepcopy(rows)
        missing_peak[0].pop("peak_cuda_memory_bytes")
        with self.assertRaisesRegex(ValueError, "is missing"):
            summarize_npr_results(
                missing_peak,
                expected,
                bootstrap_samples=2,
            )

        all_cpu = copy.deepcopy(rows)
        for row in all_cpu:
            row["peak_cuda_memory_bytes"] = None
        summary = summarize_npr_results(
            all_cpu,
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(summary["peak_cuda_memory_bytes"]["count"], 0)
        self.assertEqual(
            summary["peak_cuda_memory_bytes"]["unmeasured_cpu_rows"],
            4,
        )

    def test_expected_identity_result_identity_and_visibility_are_strict(self):
        rows, expected = self._complete_fixture()

        duplicate_id = [expected[0], copy.deepcopy(expected[0])]
        with self.assertRaisesRegex(ValueError, "duplicate expected sample_id"):
            summarize_npr_results(
                [],
                duplicate_id,
                bootstrap_samples=2,
            )

        bad_label = copy.deepcopy(expected)
        bad_label[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "kind/label mismatch"):
            summarize_npr_results(
                rows,
                bad_label,
                bootstrap_samples=2,
            )

        unexpected = copy.deepcopy(rows)
        unexpected.append(_row("other", task_id="other", kind="real"))
        with self.assertRaisesRegex(ValueError, "unexpected result id"):
            summarize_npr_results(
                unexpected,
                expected,
                bootstrap_samples=2,
            )

        invalid_visibility = copy.deepcopy(rows)
        invalid_visibility[0]["edit_visibility"] = "tiny"
        with self.assertRaisesRegex(ValueError, "invalid category"):
            summarize_npr_results(
                invalid_visibility,
                expected,
                bootstrap_samples=2,
            )

        invalid_fraction = copy.deepcopy(rows)
        invalid_fraction[0]["edit_visible_gt_fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "category fraction mismatch"):
            summarize_npr_results(
                invalid_fraction,
                expected,
                bootstrap_samples=2,
            )

    def test_incomplete_and_empty_preflight_are_explicit(self):
        expected = [
            _expected("only-forged", task_id="pilot", kind="forged")
        ]
        rows = [
            _row(
                "only-forged",
                task_id="pilot",
                kind="forged",
                score=0.5,
            )
        ]
        summary = summarize_npr_results(
            rows,
            expected,
            bootstrap_samples=2,
        )
        self.assertTrue(summary["coverage"]["is_complete"])
        self.assertEqual(
            summary["paired_coverage"]["expected_complete_pairs"],
            0,
        )
        self.assertEqual(
            summary["paired_coverage"]["expected_incomplete_tasks"],
            1,
        )
        self.assertIsNone(summary["pair_bootstrap"])
        self.assertIsNone(summary["paired_sign_test"])
        self.assertIsNone(summary["detection"]["auroc"])
        self.assertEqual(summary["detection"]["fn"], 1)

        empty = summarize_npr_results(
            [],
            [],
            bootstrap_samples=2,
        )
        self.assertEqual(empty["coverage"]["coverage_fraction"], 1.0)
        self.assertTrue(empty["coverage"]["is_complete"])
        self.assertEqual(empty["detection"]["valid_images"], 0)
        self.assertIsNone(empty["pair_bootstrap"])

    def test_missing_rows_and_wrong_summary_threshold_are_rejected_or_counted(self):
        rows, expected = self._complete_fixture()
        summary = summarize_npr_results(
            rows[:2],
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(summary["coverage"]["missing_images"], 2)
        self.assertEqual(summary["coverage"]["coverage_fraction"], 0.5)
        self.assertFalse(summary["coverage"]["is_complete"])

        with self.assertRaisesRegex(ValueError, "fixed threshold 0.5"):
            summarize_npr_results(
                rows,
                expected,
                threshold=0.5001,
                bootstrap_samples=2,
            )


class NprRawLogitDiagnosticTests(unittest.TestCase):
    def _fixture(self):
        specifications = (
            ("a", 0.0, 0.0, -170.0, -169.0),
            ("b", 0.0, 0.0, -100.0, -101.0),
            ("c", 1e-10, 1e-9, -23.0, -20.0),
        )
        rows = []
        expected = []
        for task_id, real_probability, forged_probability, real_raw, forged_raw in specifications:
            for kind, probability, raw_logit in (
                ("real", real_probability, real_raw),
                ("forged", forged_probability, forged_raw),
            ):
                row_id = f"{task_id}-{kind}"
                row = _row(
                    row_id,
                    task_id=task_id,
                    kind=kind,
                    score=probability,
                )
                row["raw_logit"] = raw_logit
                rows.append(row)
                expected.append(
                    _expected(row_id, task_id=task_id, kind=kind)
                )
        return rows, expected

    def test_underflow_ties_are_disclosed_but_raw_order_is_preserved(self):
        rows, expected = self._fixture()
        result = summarize_npr_raw_logit_diagnostic(
            rows,
            expected,
            bootstrap_samples=20,
            seed=9,
        )
        self.assertEqual(
            result["schema_version"],
            "npr_raw_logit_numerical_diagnostic_v1",
        )
        self.assertEqual(
            result["probability_saturation"]["exact_zero_images"],
            4,
        )
        self.assertEqual(
            result["probability_saturation"][
                "paired_probability_ties_with_distinct_raw_logits"
            ],
            2,
        )
        self.assertEqual(result["paired_ranking"]["wins"], 2)
        self.assertEqual(result["paired_ranking"]["losses"], 1)
        self.assertEqual(result["paired_ranking"]["ties"], 0)
        self.assertEqual(
            result["pair_bootstrap"]["paired_ranking_accuracy"]["estimate"],
            2 / 3,
        )
        self.assertTrue(
            result["released_decision_equivalence"][
                "all_valid_images_equal"
            ]
        )
        self.assertEqual(result["detection"]["threshold"], 0.0)
        self.assertEqual(result["detection"]["score_key"], "raw_logit")
        self.assertEqual(result["by_domain"]["lodging"]["pairs"], 3)
        self.assertEqual(
            result["by_domain"]["lodging"]["pair_bootstrap"][
                "paired_ranking_accuracy"
            ]["estimate"],
            2 / 3,
        )
        self.assertEqual(
            result["by_edit_visibility"]["partial"]["pairs"],
            3,
        )

    def test_raw_logits_must_be_finite(self):
        rows, expected = self._fixture()
        rows[0]["raw_logit"] = math.nan
        with self.assertRaisesRegex(ValueError, "not finite"):
            summarize_npr_raw_logit_diagnostic(
                rows,
                expected,
                bootstrap_samples=2,
            )

    def test_probability_and_logit_threshold_rounding_is_disclosed(self):
        rows, expected = self._fixture()
        rows[0]["raw_logit"] = 1.0
        result = summarize_npr_raw_logit_diagnostic(
            rows,
            expected,
            bootstrap_samples=2,
        )
        equivalence = result["released_decision_equivalence"]
        self.assertFalse(equivalence["all_valid_images_equal"])
        self.assertEqual(equivalence["disagreement_images"], 1)
        self.assertEqual(equivalence["disagreement_ids"], ["a-real"])


if __name__ == "__main__":
    unittest.main()
