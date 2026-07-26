import copy
import math
import unittest

import numpy as np

from eval.opensource.whole_image_metrics import (
    FIXED_THRESHOLD,
    image_detection_metrics_strict,
    summarize_whole_image_pair_slice,
    summarize_whole_image_results,
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
    score: float = 0.0,
    domain: str = "lodging",
    visibility: str = "partial",
    visible_fraction: float = 0.25,
    status: str = "ok",
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
        "peak_cuda_memory_bytes": 17,
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


class WholeImageDetectionMetricsTests(unittest.TestCase):
    def test_arbitrary_real_scores_and_strict_released_threshold(self):
        result = image_detection_metrics_strict(
            [
                {"status": "ok", "label": 0, "ai_score": -100.0},
                {"status": "ok", "label": 0, "ai_score": 2.0},
                {"status": "ok", "label": 1, "ai_score": 2.0},
                {"status": "ok", "label": 1, "ai_score": 100.0},
            ]
        )

        self.assertEqual(result["threshold"], FIXED_THRESHOLD)
        self.assertEqual(result["threshold_operator"], ">")
        self.assertEqual(
            (result["tp"], result["fp"], result["fn"], result["tn"]),
            (1, 0, 1, 2),
        )
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["specificity"], 1.0)
        self.assertEqual(result["auroc"], 0.875)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_direct_metric_rejects_unknown_status(self):
        with self.assertRaisesRegex(ValueError, "invalid status"):
            image_detection_metrics_strict(
                [{"status": "banana", "label": 0, "ai_score": 0.0}]
            )

    def test_real_only_higher_quantile_and_strict_fpr_operator(self):
        real_scores = list(range(21))
        forged_scores = [19.0, 20.0, 21.0]
        rows = [
            {"status": "ok", "label": 0, "ai_score": score}
            for score in real_scores
        ] + [
            {"status": "ok", "label": 1, "ai_score": score}
            for score in forged_scores
        ]

        result = image_detection_metrics_strict(rows)

        self.assertEqual(result["tpr_at_fpr_5_percent_threshold"], 19.0)
        self.assertEqual(result["tpr_at_fpr_5_percent"], 2 / 3)
        self.assertEqual(
            result["tpr_at_fpr_5_percent_actual_fpr"],
            1 / 21,
        )
        self.assertLessEqual(
            result["tpr_at_fpr_5_percent_actual_fpr"],
            0.05,
        )
        self.assertEqual(
            result["tpr_at_fpr_5_percent_threshold_operator"],
            ">",
        )

    def test_detection_never_silently_drops_bad_successful_rows(self):
        invalid = (
            ({"status": "ok", "label": 1, "ai_score": math.nan}, "not finite"),
            ({"status": "ok", "label": 1, "ai_score": math.inf}, "not finite"),
            ({"status": "ok", "label": 1, "ai_score": "3"}, "real number"),
            ({"status": "ok", "label": 1}, "real number"),
            ({"status": "ok", "label": True, "ai_score": 3}, "integer 0/1"),
            ({"status": "ok", "label": 2, "ai_score": 3}, "0/1 label"),
        )
        for row, message in invalid:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    image_detection_metrics_strict([row])

    def test_one_class_preflight_keeps_fixed_metrics_and_null_rank_metrics(self):
        result = image_detection_metrics_strict(
            [{"status": "ok", "label": 1, "ai_score": 3.0}]
        )
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["recall"], 1.0)
        self.assertIsNone(result["specificity"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["tpr_at_fpr_5_percent"])

    def test_wrong_released_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed threshold 2.0"):
            image_detection_metrics_strict([], threshold=0.5)
        with self.assertRaisesRegex(ValueError, "fixed threshold 2.0"):
            summarize_whole_image_results([], [], threshold=2.0001)


class WholeImagePairMetricsTests(unittest.TestCase):
    def test_ranking_delta_and_exact_two_sided_sign_test(self):
        pairs = [
            _pair("a", 0, 1),
            _pair("b", 1, 2),
            _pair("c", 2, 3),
            _pair("d", 4, 3),
            _pair("e", 5, 5),
        ]

        result = summarize_whole_image_pair_slice(
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
        self.assertAlmostEqual(
            result["paired_score_delta"]["mean"],
            0.4,
        )

    def test_bootstrap_is_deterministic_and_reselects_fpr_threshold(self):
        pairs = [
            _pair("a", 0, 50),
            _pair("b", 1, 50),
            _pair("c", 2, 50),
            _pair("d", 100, 50),
        ]
        first = summarize_whole_image_pair_slice(
            pairs,
            iterations=200,
            seed=99,
        )
        second = summarize_whole_image_pair_slice(
            pairs,
            iterations=200,
            seed=99,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["tpr_at_fpr_5_percent"]["estimate"],
            0.0,
        )
        # Replicates which omit the real-score outlier choose a lower
        # real-only threshold and therefore have TPR 1.  A point-fixed
        # threshold bootstrap would have an upper endpoint of zero.
        self.assertEqual(
            first["tpr_at_fpr_5_percent"]["ci95_percentile"],
            [0.0, 1.0],
        )

    def test_pair_validation_rejects_identity_and_metadata_mismatches(self):
        base = _pair("a", 0, 1)
        changes = (
            ("task IDs", "task_id", "other"),
            ("domains", "domain", "restaurant"),
            ("edit_visibility", "edit_visibility", "full"),
            (
                "edit_visible_gt_fraction",
                "edit_visible_gt_fraction",
                0.3,
            ),
        )
        for message, key, value in changes:
            broken = copy.deepcopy(base)
            broken["forged"][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_whole_image_pair_slice(
                        [broken],
                        iterations=2,
                    )

        broken = copy.deepcopy(base)
        broken["real"]["kind"] = "forged"
        with self.assertRaisesRegex(ValueError, "wrong kind"):
            summarize_whole_image_pair_slice([broken], iterations=2)

        duplicate = [copy.deepcopy(base), copy.deepcopy(base)]
        with self.assertRaisesRegex(ValueError, "duplicate task pair"):
            summarize_whole_image_pair_slice(duplicate, iterations=2)

    def test_pair_fraction_range_and_bootstrap_arguments_are_strict(self):
        pair = _pair("a", 0, 1)
        pair["real"]["edit_visible_gt_fraction"] = 1.1
        pair["forged"]["edit_visible_gt_fraction"] = 1.1
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            summarize_whole_image_pair_slice([pair], iterations=2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            summarize_whole_image_pair_slice(
                [_pair("a", 0, 1)],
                iterations=0,
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            summarize_whole_image_pair_slice(
                [_pair("a", 0, 1)],
                iterations=True,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            summarize_whole_image_pair_slice(
                [_pair("a", 0, 1)],
                iterations=2,
                seed=True,
            )


class WholeImageSummaryTests(unittest.TestCase):
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
                score=-3,
                visibility="partial",
                visible_fraction=0.05,
            ),
            _row(
                "a-forged",
                task_id="a",
                kind="forged",
                score=4,
                visibility="partial",
                visible_fraction=0.05,
            ),
            _row(
                "b-real",
                task_id="b",
                kind="real",
                score=3,
                domain="restaurant",
                visibility="full",
                visible_fraction=1.0,
            ),
            _row(
                "b-forged",
                task_id="b",
                kind="forged",
                score=2,
                domain="restaurant",
                visibility="full",
                visible_fraction=1.0,
            ),
        ]
        return rows, expected

    def test_summary_latest_rows_coverage_and_slices(self):
        rows, expected = self._complete_fixture()
        history = [
            _row(
                "a-real",
                task_id="a",
                kind="real",
                visibility="partial",
                visible_fraction=0.05,
                status="error",
            ),
            *rows,
        ]

        first = summarize_whole_image_results(
            history,
            expected,
            bootstrap_samples=30,
            seed=11,
        )
        second = summarize_whole_image_results(
            history,
            expected,
            bootstrap_samples=30,
            seed=11,
        )

        self.assertEqual(first, second)
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
        self.assertEqual(
            first["pair_bootstrap"]["bootstrap_samples"],
            30,
        )
        self.assertEqual(
            first["edit_visible_gt_fraction"]["count"],
            2,
        )

    def test_latest_error_overrides_success_and_reduces_pair_coverage(self):
        rows, expected = self._complete_fixture()
        history = [
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

        summary = summarize_whole_image_results(
            history,
            expected,
            bootstrap_samples=3,
        )

        self.assertEqual(summary["coverage"]["valid_images"], 3)
        self.assertEqual(summary["coverage"]["error_images"], 1)
        self.assertFalse(summary["coverage"]["is_complete"])
        self.assertEqual(
            summary["paired_coverage"]["complete_valid_pairs"],
            1,
        )
        self.assertEqual(
            summary["paired_coverage"]["unpaired_valid_images"],
            1,
        )

    def test_incomplete_preflight_is_supported_and_explicit(self):
        expected = [
            _expected("only-forged", task_id="pilot", kind="forged")
        ]
        rows = [
            _row(
                "only-forged",
                task_id="pilot",
                kind="forged",
                score=3,
            )
        ]

        summary = summarize_whole_image_results(
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
        self.assertTrue(
            summary["paired_coverage"][
                "preflight_expected_incomplete_pairs"
            ]
        )
        self.assertEqual(
            summary["paired_coverage"]["complete_valid_pairs"],
            0,
        )
        self.assertIsNone(summary["pair_bootstrap"])
        self.assertIsNone(summary["paired_sign_test"])
        self.assertIsNone(summary["detection"]["auroc"])
        self.assertEqual(summary["by_domain"], {})
        self.assertEqual(summary["by_edit_visibility"], {})

    def test_empty_preflight_has_unambiguous_coverage(self):
        summary = summarize_whole_image_results(
            [],
            [],
            bootstrap_samples=2,
        )
        self.assertEqual(summary["coverage"]["expected_images"], 0)
        self.assertEqual(summary["coverage"]["coverage_fraction"], 1.0)
        self.assertTrue(summary["coverage"]["is_complete"])
        self.assertEqual(summary["detection"]["valid_images"], 0)
        self.assertIsNone(summary["pair_bootstrap"])

    def test_every_physical_row_is_identity_and_score_validated(self):
        rows, expected = self._complete_fixture()

        unexpected = copy.deepcopy(rows)
        unexpected.append(
            _row("other", task_id="other", kind="real")
        )
        with self.assertRaisesRegex(ValueError, "unexpected result id"):
            summarize_whole_image_results(
                unexpected,
                expected,
                bootstrap_samples=2,
            )

        wrong_domain = copy.deepcopy(rows)
        wrong_domain[0]["domain"] = "restaurant"
        with self.assertRaisesRegex(ValueError, "domain.*expected identity"):
            summarize_whole_image_results(
                wrong_domain,
                expected,
                bootstrap_samples=2,
            )

        invalid_old_retry = copy.deepcopy(rows)
        invalid_old_retry.insert(
            0,
            {
                **copy.deepcopy(rows[0]),
                "ai_score": math.nan,
            },
        )
        with self.assertRaisesRegex(ValueError, "not finite"):
            summarize_whole_image_results(
                invalid_old_retry,
                expected,
                bootstrap_samples=2,
            )

        missing_visibility = copy.deepcopy(rows)
        missing_visibility[0].pop("edit_visibility")
        with self.assertRaisesRegex(ValueError, "edit_visibility"):
            summarize_whole_image_results(
                missing_visibility,
                expected,
                bootstrap_samples=2,
            )

        error_metadata_mismatch = [
            *copy.deepcopy(rows),
            _row(
                "a-real",
                task_id="a",
                kind="real",
                visibility="none",
                visible_fraction=0.05,
                status="error",
            ),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "edit_visibility/category fraction mismatch",
        ):
            summarize_whole_image_results(
                error_metadata_mismatch,
                expected,
                bootstrap_samples=2,
            )

    def test_visibility_enum_and_fraction_contract_are_strict(self):
        rows, expected = self._complete_fixture()

        invalid_category = copy.deepcopy(rows)
        invalid_category[0]["edit_visibility"] = "tiny"
        with self.assertRaisesRegex(
            ValueError,
            "edit_visibility has invalid category",
        ):
            summarize_whole_image_results(
                invalid_category,
                expected,
                bootstrap_samples=2,
            )

        for visibility, fraction in (
            ("none", 0.5),
            ("full", 0.0),
            ("partial", 0.0),
            ("partial", 1.0),
        ):
            with self.subTest(visibility=visibility, fraction=fraction):
                inconsistent = copy.deepcopy(rows)
                inconsistent[0]["edit_visibility"] = visibility
                inconsistent[0]["edit_visible_gt_fraction"] = fraction
                with self.assertRaisesRegex(
                    ValueError,
                    "edit_visibility/category fraction mismatch",
                ):
                    summarize_whole_image_results(
                        inconsistent,
                        expected,
                        bootstrap_samples=2,
                    )

    def test_nonfinite_derived_pair_delta_fails_closed(self):
        pair = _pair("extreme", -1e308, 1e308)
        with self.assertRaisesRegex(ValueError, "paired score delta"):
            summarize_whole_image_pair_slice(
                [pair],
                iterations=2,
            )

    def test_ok_row_performance_fields_are_strict(self):
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
                    summarize_whole_image_results(
                        invalid,
                        expected,
                        bootstrap_samples=2,
                    )

        missing_peak = copy.deepcopy(rows)
        missing_peak[0].pop("peak_cuda_memory_bytes")
        with self.assertRaisesRegex(
            ValueError,
            "peak_cuda_memory_bytes is missing",
        ):
            summarize_whole_image_results(
                missing_peak,
                expected,
                bootstrap_samples=2,
            )

        cpu_rows = copy.deepcopy(rows)
        cpu_rows[0]["peak_cuda_memory_bytes"] = None
        summary = summarize_whole_image_results(
            cpu_rows,
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(
            summary["peak_cuda_memory_bytes"]["count"],
            len(cpu_rows) - 1,
        )

    def test_expected_identity_and_coverage_schema_are_strict(self):
        rows, expected = self._complete_fixture()

        duplicate_id = [expected[0], copy.deepcopy(expected[0])]
        with self.assertRaisesRegex(ValueError, "duplicate expected sample_id"):
            summarize_whole_image_results(
                [],
                duplicate_id,
                bootstrap_samples=2,
            )

        duplicate_kind = [
            expected[0],
            {
                **copy.deepcopy(expected[0]),
                "sample_id": "different",
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate expected real"):
            summarize_whole_image_results(
                [],
                duplicate_kind,
                bootstrap_samples=2,
            )

        bad_label = copy.deepcopy(expected)
        bad_label[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "kind/label mismatch"):
            summarize_whole_image_results(
                rows,
                bad_label,
                bootstrap_samples=2,
            )

        mismatched_expected_domain = copy.deepcopy(expected)
        mismatched_expected_domain[1]["domain"] = "restaurant"
        with self.assertRaisesRegex(ValueError, "mismatched domains"):
            summarize_whole_image_results(
                rows,
                mismatched_expected_domain,
                bootstrap_samples=2,
            )

    def test_missing_rows_are_counted_instead_of_silently_ignored(self):
        rows, expected = self._complete_fixture()
        summary = summarize_whole_image_results(
            rows[:2],
            expected,
            bootstrap_samples=2,
        )
        self.assertEqual(summary["coverage"]["result_images"], 2)
        self.assertEqual(summary["coverage"]["missing_images"], 2)
        self.assertEqual(summary["coverage"]["coverage_fraction"], 0.5)
        self.assertFalse(summary["coverage"]["is_complete"])
        self.assertEqual(
            summary["paired_coverage"]["complete_valid_pairs"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
