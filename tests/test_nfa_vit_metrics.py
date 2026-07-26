from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from eval.opensource.nfa_vit_metrics import (
    FIXED_CLASSIFICATION_THRESHOLD,
    FIXED_MASK_THRESHOLD,
    binary_pixel_metrics_strict,
    image_detection_metrics_strict,
    summarize_nfa_vit_pair_slice,
    summarize_nfa_vit_results,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _classification(probability: float) -> dict[str, object]:
    return {
        "raw_logit": _logit(probability),
        "probability": probability,
        "score": probability,
        "decision": probability > 0.5,
        "threshold": 0.5,
        "threshold_operator": ">",
    }


def _metric(
    *,
    forged: bool,
    probability: np.ndarray | None = None,
) -> dict[str, object]:
    if forged:
        target = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        scores = (
            np.asarray([[0.9, 0.1], [0.1, 0.1]], dtype=np.float32)
            if probability is None
            else probability
        )
    else:
        target = np.zeros((2, 2), dtype=np.uint8)
        scores = (
            np.asarray([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
            if probability is None
            else probability
        )
    return binary_pixel_metrics_strict(
        scores,
        target,
        include_ap=forged,
    )


def _row(
    task: str,
    kind: str,
    score: float,
    *,
    probability: np.ndarray | None = None,
) -> dict[str, object]:
    forged = kind == "forged"
    metric = _metric(forged=forged, probability=probability)
    return {
        "id": f"{task}-{kind}",
        "task_id": task,
        "kind": kind,
        "label": int(forged),
        "domain": "restaurant",
        "status": "ok",
        "score": score,
        "classification": _classification(score),
        "decision": "forged" if score > 0.5 else "authentic",
        "valid_for_t1": True,
        "valid_for_t2": True,
        "localization": {
            "model_512": copy.deepcopy(metric),
            "native": copy.deepcopy(metric),
        },
        "latency_ms": 10.0,
        "peak_cuda_memory_bytes": 1024,
    }


def _pairs() -> list[dict[str, dict[str, object]]]:
    return [
        {
            "real": _row("a", "real", 0.1),
            "forged": _row("a", "forged", 0.9),
        },
        {
            "real": _row("b", "real", 0.2),
            "forged": _row("b", "forged", 0.8),
        },
    ]


class NFAViTMetricsTests(unittest.TestCase):
    def test_pixel_threshold_is_strict(self) -> None:
        scores = np.asarray([[0.5, 0.50001], [0.0, 1.0]], dtype=np.float32)
        target = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        result = binary_pixel_metrics_strict(scores, target)
        self.assertEqual(result["threshold"], FIXED_MASK_THRESHOLD)
        self.assertEqual(result["threshold_operator"], ">")
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)

    def test_pixel_metric_rebrands_shared_validation_errors(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "NFA-ViT localization uses the fixed 0.5 threshold",
        ):
            binary_pixel_metrics_strict(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.uint8),
                threshold=0.4,
            )
        with self.assertRaisesRegex(ValueError, "falls outside"):
            binary_pixel_metrics_strict(
                np.asarray([[1.1]], dtype=np.float32),
                np.asarray([[1]], dtype=np.uint8),
            )

    def test_authentic_pixel_ap_is_null(self) -> None:
        result = binary_pixel_metrics_strict(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
            include_ap=False,
        )
        self.assertIsNone(result["pixel_ap"])
        self.assertEqual(result["predicted_positive_pixels"], 0)

    def test_native_image_head_detection_is_perfect(self) -> None:
        rows = [
            _row("a", "real", 0.1),
            _row("a", "forged", 0.9),
            _row("b", "real", 0.2),
            _row("b", "forged", 0.8),
        ]
        result = image_detection_metrics_strict(rows)
        self.assertEqual(result["threshold"], FIXED_CLASSIFICATION_THRESHOLD)
        self.assertEqual(result["threshold_operator"], ">")
        self.assertEqual(result["score_semantics"], (
            "native_sigmoid_cls_decoder_manipulation_probability"
        ))
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["tn"], 2)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["average_precision"], 1.0)

    def test_score_equal_to_half_is_authentic(self) -> None:
        row = _row("a", "real", 0.5)
        row["classification"] = {
            "raw_logit": 0.0,
            "probability": 0.5,
            "score": 0.5,
            "decision": False,
            "threshold": 0.5,
            "threshold_operator": ">",
        }
        result = image_detection_metrics_strict([row])
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 0)

    def test_score_tolerance_cannot_flip_float32_sigmoid_decision(self) -> None:
        score = float(
            np.nextafter(
                np.float32(0.5),
                np.float32(np.inf),
            )
        )
        row = _row("a", "forged", score)
        row["classification"] = {
            "raw_logit": 0.0,
            "probability": score,
            "score": score,
            "decision": True,
            "threshold": 0.5,
            "threshold_operator": ">",
        }
        row["decision"] = "forged"
        with self.assertRaisesRegex(
            ValueError,
            "float32 sigmoid classifier logit",
        ):
            image_detection_metrics_strict([row])

    def test_top_level_classification_aliases_are_accepted(self) -> None:
        row = _row("a", "forged", 0.8)
        row.pop("classification")
        row.update(
            {
                "classification_logit": _logit(0.8),
                "classification_probability": 0.8,
                "classification_decision_strict_gt_0_5": True,
                "classification_threshold": 0.5,
                "classification_threshold_operator": ">",
            }
        )
        result = image_detection_metrics_strict([row])
        self.assertEqual(result["tp"], 1)

    def test_classifier_logit_probability_and_decision_are_audited(self) -> None:
        row = _row("a", "forged", 0.8)
        row["classification"]["raw_logit"] = 0.0
        with self.assertRaisesRegex(ValueError, "sigmoid classifier logit"):
            image_detection_metrics_strict([row])

        row = _row("a", "forged", 0.8)
        row["classification"]["decision"] = False
        with self.assertRaisesRegex(ValueError, "decision/score mismatch"):
            image_detection_metrics_strict([row])

        row = _row("a", "forged", 0.8)
        row["classification_decision"] = False
        with self.assertRaisesRegex(ValueError, "decision/score mismatch"):
            image_detection_metrics_strict([row])

        row = _row("a", "forged", 0.8)
        row["decision"] = "authentic"
        with self.assertRaisesRegex(ValueError, "wrong decision label"):
            image_detection_metrics_strict([row])

        row = _row("a", "forged", 0.8)
        row["classification"]["threshold_operator"] = ">="
        with self.assertRaisesRegex(ValueError, "strict '>'"):
            image_detection_metrics_strict([row])

    def test_no_dense_map_fallback_for_missing_native_score(self) -> None:
        row = _row("a", "forged", 0.8)
        row.pop("score")
        with self.assertRaisesRegex(ValueError, "native T1 score"):
            image_detection_metrics_strict([row])

    def test_pair_slice_has_t1_t2_and_joint_bootstrap(self) -> None:
        result = summarize_nfa_vit_pair_slice(
            _pairs(),
            iterations=25,
            seed=7,
            localization_space="native",
        )
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["images"], 4)
        self.assertEqual(result["auroc"]["estimate"], 1.0)
        self.assertEqual(result["image_f1_at_0_5"]["estimate"], 1.0)
        self.assertEqual(result["pixel_f1_macro_at_0_5"]["estimate"], 1.0)
        self.assertEqual(result["pixel_mcc_micro_at_0_5"]["estimate"], 1.0)
        self.assertEqual(result["s_joint_macro_at_0_5"]["estimate"], 1.0)
        self.assertEqual(result["s_joint_micro_at_0_5"]["estimate"], 1.0)
        self.assertEqual(
            result["paired_sign_test"],
            {
                "wins": 2,
                "losses": 0,
                "ties": 0,
                "two_sided_exact_p": 0.5,
            },
        )

    def test_pair_bootstrap_is_deterministic(self) -> None:
        first = summarize_nfa_vit_pair_slice(
            _pairs(),
            iterations=30,
            seed=19,
        )
        second = summarize_nfa_vit_pair_slice(
            _pairs(),
            iterations=30,
            seed=19,
        )
        self.assertEqual(first, second)

    def test_pair_slice_rejects_invalid_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_nfa_vit_pair_slice([], iterations=2, seed=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            summarize_nfa_vit_pair_slice(_pairs(), iterations=0, seed=1)
        with self.assertRaisesRegex(ValueError, "unsupported localization"):
            summarize_nfa_vit_pair_slice(
                _pairs(),
                iterations=2,
                seed=1,
                localization_space="raw_128",
            )

    def test_full_summary_matches_frozen_schema(self) -> None:
        rows = [
            _row("a", "real", 0.1),
            _row("a", "forged", 0.9),
            _row("b", "real", 0.2),
            _row("b", "forged", 0.8),
        ]
        expected = [
            {
                "sample_id": row["id"],
                "task_id": row["task_id"],
                "kind": row["kind"],
                "label": row["label"],
                "domain": row["domain"],
            }
            for row in rows
        ]
        summary = summarize_nfa_vit_results(
            rows,
            expected,
            bootstrap_samples=20,
            seed=11,
        )
        self.assertEqual(summary["schema_version"], "opensource_summary_v1")
        self.assertEqual(summary["coverage"]["expected_images"], 4)
        self.assertEqual(summary["coverage"]["valid_images"], 4)
        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 2)
        self.assertTrue(summary["task_scope"]["valid_for_t1"])
        self.assertTrue(summary["task_scope"]["valid_for_t2"])
        self.assertTrue(
            summary["task_scope"]["image_score_independent_of_dense_map"]
        )
        self.assertEqual(summary["detection"]["auroc"], 1.0)
        self.assertEqual(summary["paired_ranking_accuracy"], 1.0)
        self.assertFalse(
            summary["native_t1_saturation_diagnostic"][
                "all_images_at_least_0_999"
            ]
        )
        self.assertFalse(
            summary["native_t1_saturation_diagnostic"][
                "public_release_evaluator_uses_native_t1_for_checkpoint_selection"
            ]
        )
        self.assertIsNone(
            summary["native_t1_saturation_diagnostic"][
                "exact_checkpoint_selection_used_native_t1"
            ]
        )
        self.assertEqual(
            summary["joint_diagnostics"]["status"],
            "claimforge_diagnostic_not_official_nfa_vit_metric",
        )
        self.assertFalse(
            summary["raw_logit_secondary_diagnostic"][
                "raw_logit_replaces_probability_for_primary_metrics"
            ]
        )
        self.assertEqual(
            summary["joint_diagnostics"]["spaces"]["native"]["macro"][
                "estimate"
            ],
            1.0,
        )
        self.assertEqual(
            summary["localization_forged"]["native"]["pixel_ap"]["mean"],
            1.0,
        )
        self.assertEqual(
            summary["localization_real"]["native"]["macro_at_threshold"][
                "false_positive_area_fraction"
            ],
            0.0,
        )

    def test_latest_physical_row_is_used(self) -> None:
        stale = _row("a", "real", 0.9)
        stale["classification"] = _classification(0.9)
        stale["decision"] = "forged"
        latest = _row("a", "real", 0.1)
        forged = _row("a", "forged", 0.9)
        summary = summarize_nfa_vit_results(
            [stale, latest, forged],
            bootstrap_samples=5,
            seed=1,
        )
        self.assertEqual(summary["coverage"]["result_images"], 2)
        self.assertEqual(summary["detection"]["accuracy"], 1.0)

    def test_summary_flags_native_head_near_one_saturation(self) -> None:
        rows = [
            _row("a", "real", 0.9995),
            _row("a", "forged", 0.9995),
        ]
        summary = summarize_nfa_vit_results(
            rows,
            bootstrap_samples=5,
            seed=1,
        )
        diagnostic = summary["native_t1_saturation_diagnostic"]
        self.assertTrue(diagnostic["all_images_at_least_0_999"])
        self.assertTrue(diagnostic["saturation_flag"])
        self.assertEqual(diagnostic["paired_equal_within_1e_7"], 1)
        self.assertTrue(diagnostic["dense_map_fallback_forbidden"])

    def test_raw_logit_is_secondary_when_float32_probability_saturates(
        self,
    ) -> None:
        rows = [
            _row("a", "real", 0.9995),
            _row("a", "forged", 0.9995),
        ]
        for row, raw_logit in zip(rows, (19.0, 20.0), strict=True):
            row["score"] = 1.0
            row["decision"] = "forged"
            row["classification"] = {
                "raw_logit": raw_logit,
                "probability": 1.0,
                "score": 1.0,
                "decision": True,
                "threshold": 0.5,
                "threshold_operator": ">",
            }
        summary = summarize_nfa_vit_results(
            rows,
            bootstrap_samples=5,
            seed=1,
        )
        self.assertEqual(summary["detection"]["auroc"], 0.5)
        secondary = summary["raw_logit_secondary_diagnostic"]
        self.assertEqual(secondary["ranking_metrics"]["auroc"], 1.0)
        self.assertEqual(secondary["paired_logit_ranking_accuracy"], 1.0)
        self.assertFalse(
            secondary["raw_logit_replaces_probability_for_primary_metrics"]
        )

    def test_summary_rejects_identity_mismatch(self) -> None:
        rows = [_row("a", "real", 0.1), _row("a", "forged", 0.9)]
        expected = [
            {
                "sample_id": rows[0]["id"],
                "task_id": "wrong",
                "kind": "real",
                "label": 0,
            }
        ]
        with self.assertRaisesRegex(ValueError, "does not match expected"):
            summarize_nfa_vit_results(
                rows,
                expected,
                bootstrap_samples=2,
            )

    def test_summary_rejects_non_frozen_thresholds(self) -> None:
        rows = [_row("a", "real", 0.1), _row("a", "forged", 0.9)]
        with self.assertRaisesRegex(ValueError, "fixed threshold"):
            summarize_nfa_vit_results(
                rows,
                classification_threshold=0.4,
                bootstrap_samples=2,
            )
        with self.assertRaisesRegex(ValueError, "fixed threshold"):
            summarize_nfa_vit_results(
                rows,
                mask_threshold=0.6,
                bootstrap_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
