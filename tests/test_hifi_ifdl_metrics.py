import copy
import unittest

import numpy as np

from eval.opensource.hifi_ifdl_metrics import (
    FINE_CLASS_NAMES,
    FIXED_CLASSIFICATION_THRESHOLD,
    FIXED_MASK_THRESHOLD,
    binary_distance_metrics_strict,
    image_detection_metrics_strict,
    official_argmax_detection_metrics,
    summarize_hifi_ifdl_pair_slice,
    summarize_hifi_ifdl_results,
)


def _t1_fields(score: float, *, official: bool) -> dict:
    authentic = 1.0 - score
    if official:
        forged_peak = min(score * 0.9, max(authentic + 0.05, score / 13))
        if forged_peak >= score:
            raise ValueError("fixture cannot realize requested official decision")
        remainder = (score - forged_peak) / 12
        probabilities = np.asarray(
            [authentic, forged_peak, *([remainder] * 12)],
            dtype=np.float64,
        )
    else:
        probabilities = np.asarray(
            [authentic, *([score / 13] * 13)],
            dtype=np.float64,
        )
    logits = np.log(probabilities)
    fine_index = int(np.argmax(logits))
    assert (fine_index != 0) == official
    return {
        "score": score,
        "score_source": "native_out3_fine_14class_head",
        "score_semantics": (
            "one_minus_softmax_probability_fine_class_0_authentic"
        ),
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "decision": "forged" if score > 0.5 else "authentic",
        "benchmark_binary_decision": score > 0.5,
        "official_fine_class_index": fine_index,
        "official_fine_class_name": FINE_CLASS_NAMES[fine_index],
        "official_binary_decision": official,
        "classification_probabilities": probabilities.tolist(),
        "classification_hierarchy_logits": {
            "out0_coarse_3class": [0.0, 0.0, 0.0],
            "out1_5class": [0.0] * 5,
            "out2_7class": [0.0] * 7,
            "out3_fine_14class": logits.tolist(),
        },
    }


def _localization(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    forged: bool,
) -> dict:
    return {
        space: binary_distance_metrics_strict(
            scores,
            target,
            include_ap=forged,
        )
        for space in ("model_256", "native")
    }


def _row(
    *,
    row_id: str,
    task_id: str,
    kind: str,
    score: float,
    official: bool,
    scores: np.ndarray,
    target: np.ndarray,
    status: str = "ok",
) -> dict:
    value = {
        "id": row_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "status": status,
    }
    if status == "ok":
        value.update(
            {
                **_t1_fields(score, official=official),
                "localization": _localization(
                    scores,
                    target,
                    forged=kind == "forged",
                ),
                "latency_ms": 1.0,
                "peak_cuda_memory_bytes": 2,
            }
        )
    return value


def _pairs() -> tuple[list[dict], list[dict]]:
    truth = np.asarray([[True, False], [True, False]], dtype=bool)
    empty = np.zeros_like(truth)
    rows = [
        _row(
            row_id="real-a",
            task_id="a",
            kind="real",
            score=0.2,
            official=False,
            scores=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            target=empty,
        ),
        _row(
            row_id="forged-a",
            task_id="a",
            kind="forged",
            score=0.8,
            official=True,
            scores=np.asarray([[3.0, 0.2], [2.3, 0.4]], dtype=np.float32),
            target=truth,
        ),
        _row(
            row_id="real-b",
            task_id="b",
            kind="real",
            score=0.7,
            official=False,
            scores=np.asarray([[2.3, 0.2], [0.3, 0.4]], dtype=np.float32),
            target=empty,
        ),
        _row(
            row_id="forged-b",
            task_id="b",
            kind="forged",
            score=0.6,
            official=True,
            scores=np.asarray([[2.4, 2.5], [0.3, 0.4]], dtype=np.float32),
            target=truth,
        ),
    ]
    pairs = [
        {"real": rows[0], "forged": rows[1]},
        {"real": rows[2], "forged": rows[3]},
    ]
    return rows, pairs


class HiFiIFDLMetricsTests(unittest.TestCase):
    def test_distance_threshold_is_inclusive_at_exactly_2_3(self):
        scores = np.asarray([[2.299999, 2.3, 4.7]], dtype=np.float32)
        target = np.asarray([[False, True, True]], dtype=bool)
        metrics = binary_distance_metrics_strict(scores, target)
        self.assertEqual(metrics["threshold"], FIXED_MASK_THRESHOLD)
        self.assertEqual(metrics["threshold_operator"], ">=")
        self.assertEqual(metrics["predicted_positive_pixels"], 2)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)

    def test_distance_is_unbounded_but_must_be_nonnegative_and_finite(self):
        scores = np.asarray([[0.0, 10.0]], dtype=np.float32)
        target = np.asarray([[False, True]])
        metrics = binary_distance_metrics_strict(scores, target)
        self.assertEqual(metrics["score_max"], 10.0)
        with self.assertRaisesRegex(ValueError, "negative"):
            binary_distance_metrics_strict(
                np.asarray([[-0.1]], dtype=np.float32),
                np.asarray([[False]]),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            binary_distance_metrics_strict(
                np.asarray([[np.nan]], dtype=np.float32),
                np.asarray([[False]]),
            )

    def test_real_pixel_ap_is_null_when_not_requested(self):
        metrics = binary_distance_metrics_strict(
            np.asarray([[0.1, 0.2]], dtype=np.float32),
            np.asarray([[False, False]]),
            include_ap=False,
        )
        self.assertIsNone(metrics["pixel_ap"])

    def test_all_positive_forged_mask_has_defined_pixel_ap(self):
        metrics = binary_distance_metrics_strict(
            np.asarray([[0.1, 3.0]], dtype=np.float32),
            np.asarray([[True, True]]),
            include_ap=True,
        )
        self.assertEqual(metrics["pixel_ap"], 1.0)

    def test_distance_requires_raw_float32_two_dimensional_map(self):
        with self.assertRaisesRegex(ValueError, "dtype must be float32"):
            binary_distance_metrics_strict(
                np.asarray([[2.3]], dtype=np.float64),
                np.asarray([[True]]),
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            binary_distance_metrics_strict(
                np.asarray([[[2.3]]], dtype=np.float32),
                np.asarray([[[True]]]),
            )

    def test_wrong_localization_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed threshold"):
            binary_distance_metrics_strict(
                np.asarray([[2.3]], dtype=np.float32),
                np.asarray([[True]]),
                threshold=2.29,
            )

    def test_benchmark_detection_uses_strict_greater_than_half(self):
        rows = [
            {"status": "ok", "label": 0, "score": 0.5},
            {"status": "ok", "label": 1, "score": 0.5001},
        ]
        result = image_detection_metrics_strict(rows)
        self.assertEqual(result["threshold"], FIXED_CLASSIFICATION_THRESHOLD)
        self.assertEqual(result["threshold_operator"], ">")
        self.assertEqual(
            (result["tp"], result["fp"], result["fn"], result["tn"]),
            (1, 0, 0, 1),
        )

    def test_official_argmax_decision_is_separate_from_score_threshold(self):
        rows = [
            {
                "status": "ok",
                "label": 1,
                "score": 0.7,
                "official_binary_decision": False,
            }
        ]
        benchmark = image_detection_metrics_strict(rows)
        official = official_argmax_detection_metrics(rows)
        self.assertEqual(benchmark["tp"], 1)
        self.assertEqual(official["fn"], 1)
        self.assertFalse(official["eligible_for_threshold_free_metrics"])

    def test_pair_bootstrap_is_deterministic_and_preserves_pairs(self):
        _, pairs = _pairs()
        first = summarize_hifi_ifdl_pair_slice(
            pairs,
            iterations=50,
            seed=9,
            localization_space="native",
        )
        second = summarize_hifi_ifdl_pair_slice(
            pairs,
            iterations=50,
            seed=9,
            localization_space="native",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["pairs"], 2)
        self.assertEqual(first["images"], 4)
        self.assertEqual(first["mask_threshold_operator"], ">=")
        self.assertEqual(first["paired_sign_test"]["wins"], 1)
        self.assertEqual(first["paired_sign_test"]["losses"], 1)

    def test_pair_bootstrap_rejects_mismatched_task_ids(self):
        _, pairs = _pairs()
        broken = copy.deepcopy(pairs)
        broken[0]["forged"]["task_id"] = "other"
        with self.assertRaisesRegex(ValueError, "mismatched task IDs"):
            summarize_hifi_ifdl_pair_slice(
                broken,
                iterations=2,
                seed=1,
            )

    def test_pair_bootstrap_rejects_missing_or_duplicate_identity(self):
        _, pairs = _pairs()
        missing = copy.deepcopy(pairs)
        missing[0]["real"].pop("kind")
        with self.assertRaisesRegex(ValueError, "wrong kind"):
            summarize_hifi_ifdl_pair_slice(
                missing,
                iterations=2,
                seed=1,
            )
        duplicate = [copy.deepcopy(pairs[0]), copy.deepcopy(pairs[0])]
        with self.assertRaisesRegex(ValueError, "duplicate task pair"):
            summarize_hifi_ifdl_pair_slice(
                duplicate,
                iterations=2,
                seed=1,
            )

    def test_summary_uses_latest_physical_row(self):
        rows, _ = _pairs()
        history = [
            {
                "id": "forged-a",
                "task_id": "a",
                "kind": "forged",
                "label": 1,
                "status": "error",
            },
            *rows,
        ]
        expected = [
            {"sample_id": row["id"]}
            for row in rows
        ]
        summary = summarize_hifi_ifdl_results(
            history,
            expected,
            bootstrap_samples=10,
            seed=3,
        )
        self.assertEqual(summary["coverage"]["result_images"], 4)
        self.assertEqual(summary["coverage"]["valid_images"], 4)
        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 2)
        self.assertEqual(
            summary["task_scope"]["mask_threshold_operator"],
            ">=",
        )
        self.assertEqual(summary["pair_bootstrap"]["bootstrap_samples"], 10)

    def test_latest_error_overrides_earlier_success(self):
        rows, _ = _pairs()
        history = [
            *rows,
            {
                "id": "forged-a",
                "task_id": "a",
                "kind": "forged",
                "label": 1,
                "status": "error",
            },
        ]
        summary = summarize_hifi_ifdl_results(
            history,
            [{"sample_id": row["id"]} for row in rows],
            bootstrap_samples=2,
            seed=3,
        )
        self.assertEqual(summary["coverage"]["result_images"], 4)
        self.assertEqual(summary["coverage"]["valid_images"], 3)
        self.assertEqual(summary["coverage"]["error_images"], 1)
        self.assertEqual(summary["paired_coverage"]["complete_pairs"], 1)

    def test_summary_rejects_tampered_metric_operator(self):
        rows, _ = _pairs()
        rows[1]["localization"]["native"]["threshold_operator"] = ">"
        with self.assertRaisesRegex(ValueError, "threshold operator"):
            summarize_hifi_ifdl_results(
                rows,
                [{"sample_id": row["id"]} for row in rows],
                bootstrap_samples=2,
            )

    def test_detection_score_outside_probability_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            image_detection_metrics_strict(
                [{"status": "ok", "label": 1, "score": 1.1}]
            )

    def test_valid_detection_row_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "no finite score"):
            image_detection_metrics_strict(
                [{"status": "ok", "label": 1}]
            )
        with self.assertRaisesRegex(ValueError, "invalid label"):
            image_detection_metrics_strict(
                [{"status": "ok", "label": 2, "score": 0.7}]
            )

    def test_summary_rejects_kind_label_mismatch(self):
        rows, _ = _pairs()
        rows[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "kind/label mismatch"):
            summarize_hifi_ifdl_results(
                rows,
                [{"sample_id": row["id"]} for row in rows],
                bootstrap_samples=2,
            )

    def test_summary_rejects_tampered_t1_derivation(self):
        rows, _ = _pairs()
        rows[0]["score"] = 0.3
        with self.assertRaisesRegex(ValueError, "one minus P"):
            summarize_hifi_ifdl_results(
                rows,
                [{"sample_id": row["id"]} for row in rows],
                bootstrap_samples=2,
            )

    def test_summary_requires_t1_provenance_fields(self):
        rows, _ = _pairs()
        rows[0].pop("classification_hierarchy_logits")
        with self.assertRaisesRegex(ValueError, "hierarchy logits"):
            summarize_hifi_ifdl_results(
                rows,
                [{"sample_id": row["id"]} for row in rows],
                bootstrap_samples=2,
            )

    def test_summary_rejects_metric_inconsistent_with_counts(self):
        rows, _ = _pairs()
        rows[1]["localization"]["native"]["f1"] = 0.25
        with self.assertRaisesRegex(ValueError, "inconsistent f1"):
            summarize_hifi_ifdl_results(
                rows,
                [{"sample_id": row["id"]} for row in rows],
                bootstrap_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
