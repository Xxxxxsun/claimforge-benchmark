"""Tests for fail-closed Balanced250 T1 aggregation."""

from __future__ import annotations

import copy
import hashlib
import unittest
from collections import Counter

from eval.opensource.balanced250_metrics import (
    FORGED_CONDITIONS,
    FULLFRAME_CONDITIONS,
    LOCAL_CONDITIONS,
    summarize_balanced250_t1,
)
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    CapabilityBinding,
    LedgerBinding,
    RunDatasetContract,
    ScoreSpec,
    SelectionBinding,
    selected_ids_sha256,
)
from eval.opensource.common import stable_json


_RUN_ID = "balanced250-metrics-fixture"
_RUN_FINGERPRINT = "f" * 64


_CONDITION_FIELDS = {
    "real": ("real", "authentic", "real", 0),
    **{
        condition: ("local_splice", "local_insertion", "forged", 1)
        for condition in LOCAL_CONDITIONS
    },
    **{
        condition: (
            "full_frame_conditional_edit",
            "conditional_full_frame_edit",
            "forged",
            1,
        )
        for condition in FULLFRAME_CONDITIONS
    },
}


def _tiny_release():
    """Build two source clusters across all seven conditions.

    Every real row shares its task ID with six forged rows.  A task-keyed
    implementation would therefore overwrite or conflate observations.
    """

    inputs = []
    panel = []
    source_pairs = []
    for condition, (
        family,
        scope,
        kind,
        label,
    ) in _CONDITION_FIELDS.items():
        for index, domain in enumerate(("lodging", "restaurant")):
            sample_id = f"{condition}-{index}"
            row = {
                "schema_version": "balanced250_fixture_v1",
                "dataset_id": "balanced250-fixture",
                "sample_id": sample_id,
                "id": sample_id,
                "condition": condition,
                "condition_family": family,
                "manipulation_scope": scope,
                "kind": kind,
                "label": label,
                "domain": domain,
                "source_content_cluster": f"source-{index}",
                "normalized_task_id": f"normalized-shared-{index}",
                "task_id": f"task-shared-{index}",
                "gt_mask_kind": "all_zero" if kind == "real" else "binary",
                "rank": index,
                "canonical_path": f"fixture/images/{sample_id}.jpg",
                "canonical_sha256": hashlib.sha256(
                    sample_id.encode()
                ).hexdigest(),
                "width": 640,
                "height": 480,
            }
            if condition in LOCAL_CONDITIONS:
                row["gt_mask_kind"] = "exact_diff"
            elif condition in FULLFRAME_CONDITIONS:
                row["gt_mask_kind"] = "not_applicable"
            inputs.append(row)
            panel.append({**row, "input_rank": index, "panel_rank": index})

    by_id = {row["sample_id"]: row for row in inputs}
    for condition in FORGED_CONDITIONS:
        for index in range(2):
            real_id = f"real-{index}"
            forged_id = f"{condition}-{index}"
            source_pairs.append(
                {
                    "schema_version": "balanced250_fixture_v1",
                    "dataset_id": "balanced250-fixture",
                    "pair_id": f"{condition}-pair-{index}",
                    "condition": condition,
                    "real_sample_id": real_id,
                    "forged_sample_id": forged_id,
                    "source_content_cluster": by_id[real_id][
                        "source_content_cluster"
                    ],
                    "normalized_task_id": by_id[real_id][
                        "normalized_task_id"
                    ],
                    "domain": by_id[real_id]["domain"],
                    "condition_pair_rank": index,
                }
            )

    scores = {"real-0": 0.1, "real-1": 0.2}
    for condition_index, condition in enumerate(FORGED_CONDITIONS):
        scores[f"{condition}-0"] = 0.8 + condition_index * 0.01
        scores[f"{condition}-1"] = 0.9 + condition_index * 0.01
    results = [
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": _RUN_ID,
            "run_manifest_fingerprint": _RUN_FINGERPRINT,
            "sample_id": row["sample_id"],
            "id": row["sample_id"],
            "status": "ok",
            "valid_for_metrics": True,
            "condition": row["condition"],
            "condition_family": row["condition_family"],
            "manipulation_scope": row["manipulation_scope"],
            "kind": row["kind"],
            "label": row["label"],
            "domain": row["domain"],
            "dataset_id": row["dataset_id"],
            "normalized_task_id": row["normalized_task_id"],
            "task_id": row["task_id"],
            "gt_mask_kind": row["gt_mask_kind"],
            "rank": row["rank"],
            "input_path": row["canonical_path"],
            "input_sha256": row["canonical_sha256"],
            "input_width": row["width"],
            "input_height": row["height"],
            "ai_score": scores[row["sample_id"]],
        }
        for row in reversed(inputs)
    ]
    return inputs, panel, source_pairs, results


def _rows_sha256(rows) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_contract(
    inputs,
    panel,
    source_pairs,
    *,
    score_spec: ScoreSpec,
) -> RunDatasetContract:
    counts = Counter(row["condition"] for row in inputs)
    bindings = {
        name: LedgerBinding(
            name=name,
            path=f"fixture/{name}.jsonl",
            sha256=_rows_sha256(rows),
            rows=len(rows),
        )
        for name, rows in (
            ("inputs", inputs),
            ("panel", panel),
            ("source_pairs", source_pairs),
        )
    }
    capability = CapabilityBinding(
        name="whole_image_t1",
        conditions=("real", *FORGED_CONDITIONS),
        valid_for_t1=True,
        valid_for_t2=False,
    )
    selection = SelectionBinding(
        capability=capability.name,
        conditions=None,
        per_condition_limit=None,
        sample_id=None,
        pair_limit=None,
        selected_images=len(inputs),
        selected_ids_sha256=selected_ids_sha256(
            [row["sample_id"] for row in inputs]
        ),
        counts_by_condition=tuple(
            (condition, counts[condition])
            for condition in ("real", *FORGED_CONDITIONS)
        ),
    )
    return RunDatasetContract(
        release_schema_version="balanced250_fixture_v1",
        release_kind="balanced250",
        dataset_id="balanced250-fixture",
        dataset_manifest="fixture/manifest.json",
        dataset_manifest_sha256="a" * 64,
        dataset_contract_sha256="b" * 64,
        inputs_ledger=bindings["inputs"],
        panel_ledger=bindings["panel"],
        source_pairs_ledger=bindings["source_pairs"],
        capability=capability,
        selection=selection,
        score_spec=score_spec,
    )


def _summarize(
    inputs,
    panel,
    source_pairs,
    results,
    **kwargs,
):
    score_key = kwargs.pop("score_key", "ai_score")
    direction = kwargs.pop("direction", "higher_means_fake")
    fixed_threshold = kwargs.pop("fixed_threshold", 0.5)
    fixed_operator = kwargs.pop(
        "fixed_threshold_operator",
        ">" if direction == "higher_means_fake" else "<",
    )
    score_spec = ScoreSpec(
        score_key,
        direction,
        fixed_threshold,
        fixed_operator,
    )
    return summarize_balanced250_t1(
        inputs,
        panel,
        source_pairs,
        results,
        run_id=_RUN_ID,
        run_manifest_fingerprint=_RUN_FINGERPRINT,
        run_dataset_contract=_run_contract(
            inputs,
            panel,
            source_pairs,
            score_spec=score_spec,
        ),
        **kwargs,
    )


class Balanced250MetricsTests(unittest.TestCase):
    def test_tiny_seven_condition_release_uses_explicit_ids(self):
        inputs, panel, source_pairs, results = _tiny_release()

        summary = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            seed=17,
            iterations=12,
        )
        repeated = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            seed=17,
            iterations=12,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["coverage"]["inputs"], 14)
        self.assertEqual(summary["coverage"]["panel"], 14)
        self.assertEqual(summary["coverage"]["source_pairs"], 12)
        self.assertEqual(summary["coverage"]["results"], 14)
        self.assertTrue(summary["coverage"]["is_complete"])
        self.assertFalse(summary["coverage"]["task_id_pair_inference"])
        self.assertEqual(summary["primary"]["join_key"], "sample_id")
        self.assertEqual(
            summary["secondary"]["join_keys"],
            ["real_sample_id", "forged_sample_id"],
        )

        for condition in FORGED_CONDITIONS:
            condition_summary = summary["primary"]["by_condition"][condition]
            self.assertEqual(
                condition_summary["overall"]["auroc"]["estimate"],
                1.0,
            )
            self.assertEqual(
                condition_summary["overall"]["average_precision"]["estimate"],
                1.0,
            )
            self.assertEqual(
                set(condition_summary["by_domain"]),
                {"lodging", "restaurant"},
            )
            matched = summary["secondary"]["by_condition"][condition]
            self.assertEqual(matched["pairs"], 2)
            self.assertEqual(matched["source_content_clusters"], 2)
            self.assertGreater(matched["mean_score_delta"]["estimate"], 0.0)

        for family in ("local", "fullframe"):
            macro = summary["primary"]["family_macro"][family]["overall"]
            self.assertEqual(macro["auroc"]["estimate"], 1.0)
            self.assertEqual(macro["average_precision"]["estimate"], 1.0)
        self.assertEqual(
            summary["primary"]["all_conditions_macro"]["overall"]["auroc"][
                "estimate"
            ],
            1.0,
        )
        self.assertEqual(
            summary["secondary"]["all_pairs"]["source_content_clusters"],
            2,
        )
        self.assertEqual(
            summary["primary"]["source_cluster_overlap_with_real"],
            {condition: 2 for condition in FORGED_CONDITIONS},
        )
        self.assertEqual(
            summary["secondary"]["all_conditions_macro"]["bootstrap_unit"],
            "condition_macro_with_shared_source_content_cluster_"
            "poisson_bootstrap",
        )

    def test_custom_score_key_and_lower_is_forged_direction(self):
        inputs, panel, source_pairs, results = _tiny_release()
        for result in results:
            result["detector_score"] = 1.0 - result.pop("ai_score")

        summary = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            score_key="detector_score",
            direction="lower_means_fake",
            fixed_threshold=0.5,
            seed=23,
            iterations=5,
        )

        self.assertEqual(
            summary["score_contract"]["direction"],
            "lower_is_forged",
        )
        self.assertEqual(summary["score_contract"]["fixed_threshold_operator"], "<")
        first = summary["primary"]["by_condition"]["local_mouse"]["overall"]
        self.assertEqual(first["auroc"]["estimate"], 1.0)
        self.assertEqual(
            first["fixed_threshold_confusion"],
            {"tp": 2, "fp": 0, "fn": 0, "tn": 2},
        )
        self.assertGreater(
            summary["secondary"]["by_condition"]["local_mouse"][
                "mean_score_delta"
            ]["estimate"],
            0.0,
        )

    def test_primary_macro_bootstrap_shares_source_cluster_weights(self):
        inputs, panel, source_pairs, results = _tiny_release()
        for result in results:
            if result["condition"] == "real":
                result["ai_score"] = (
                    0.1 if result["sample_id"] == "real-0" else 0.9
                )
            else:
                result["ai_score"] = 0.5

        summary = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            seed=101,
            iterations=101,
        )
        condition_cis = [
            summary["primary"]["by_condition"][condition]["overall"][
                "auroc"
            ]["ci95_percentile"]
            for condition in FORGED_CONDITIONS
        ]
        self.assertTrue(
            all(value == condition_cis[0] for value in condition_cis)
        )
        macro = summary["primary"]["all_conditions_macro"]["overall"]
        self.assertEqual(
            macro["bootstrap_unit"],
            "condition_macro_with_shared_source_content_cluster_"
            "poisson_bootstrap",
        )
        self.assertEqual(
            macro["auroc"]["ci95_percentile"],
            condition_cis[0],
        )

    def test_fixed_threshold_operator_preserves_inclusive_ties(self):
        inputs, panel, source_pairs, results = _tiny_release()
        for result in results:
            if result["kind"] == "forged":
                result["ai_score"] = (
                    0.5 if result["sample_id"].endswith("-0") else 0.9
                )
        strict = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            fixed_threshold_operator=">",
            iterations=3,
        )
        inclusive = _summarize(
            inputs,
            panel,
            source_pairs,
            results,
            fixed_threshold_operator=">=",
            iterations=3,
        )
        strict_slice = strict["primary"]["by_condition"]["local_mouse"][
            "overall"
        ]
        inclusive_slice = inclusive["primary"]["by_condition"]["local_mouse"][
            "overall"
        ]
        self.assertEqual(
            strict_slice["fixed_threshold_confusion"]["tp"],
            1,
        )
        self.assertEqual(
            inclusive_slice["fixed_threshold_confusion"]["tp"],
            2,
        )
        self.assertEqual(
            inclusive["score_contract"]["fixed_threshold_operator"],
            ">=",
        )
        self.assertEqual(
            inclusive["score_contract"][
                "tpr_at_fpr_5_percent_threshold_operator"
            ],
            ">",
        )

    def test_rejects_unknown_explicit_pair_reference(self):
        inputs, panel, source_pairs, results = _tiny_release()
        source_pairs[0]["forged_sample_id"] = "missing-forged-sample"

        with self.assertRaisesRegex(ValueError, "unknown forged_sample_id"):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

    def test_rejects_duplicate_ids_and_pair_endpoints(self):
        for case in (
            "input_sample_id",
            "panel_sample_id",
            "result_sample_id",
            "pair_id",
            "forged_endpoint",
        ):
            with self.subTest(case=case):
                inputs, panel, source_pairs, results = _tiny_release()
                if case == "input_sample_id":
                    inputs.append(copy.deepcopy(inputs[0]))
                    message = "duplicates"
                elif case == "panel_sample_id":
                    panel.append(copy.deepcopy(panel[0]))
                    message = "duplicate sample_id"
                elif case == "result_sample_id":
                    results.append(copy.deepcopy(results[0]))
                    message = "duplicate sample_id"
                elif case == "pair_id":
                    source_pairs[1]["pair_id"] = source_pairs[0]["pair_id"]
                    message = "duplicate pair_id"
                else:
                    source_pairs[1]["forged_sample_id"] = source_pairs[0][
                        "forged_sample_id"
                    ]
                    message = "repeats forged_sample_id"

                with self.assertRaisesRegex(ValueError, message):
                    _summarize(
                        inputs,
                        panel,
                        source_pairs,
                        results,
                        iterations=1,
                    )

    def test_result_coverage_and_success_are_fail_closed(self):
        inputs, panel, source_pairs, results = _tiny_release()
        with self.assertRaisesRegex(ValueError, "result coverage mismatch"):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results[:-1],
                iterations=1,
            )

        inputs, panel, source_pairs, results = _tiny_release()
        results[0]["status"] = "error"
        with self.assertRaisesRegex(ValueError, "is not status ok"):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

    def test_result_identity_fields_are_required_and_must_match(self):
        inputs, panel, source_pairs, results = _tiny_release()
        missing_sample_id = results[0]["sample_id"]
        results[0].pop("normalized_task_id")
        with self.assertRaisesRegex(
            ValueError,
            f"result {missing_sample_id} lacks identity field "
            "normalized_task_id",
        ):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

        inputs, panel, source_pairs, results = _tiny_release()
        result_id = results[0]["sample_id"]
        results[0]["run_manifest_fingerprint"] = "e" * 64
        with self.assertRaisesRegex(
            ValueError,
            f"result {result_id} run_manifest_fingerprint mismatch",
        ):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

        inputs, panel, source_pairs, results = _tiny_release()
        result_id = results[0]["sample_id"]
        results[0]["pair_rank"] = 0
        with self.assertRaisesRegex(
            ValueError,
            f"result {result_id} must not contain pair_rank",
        ):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

        inputs, panel, source_pairs, results = _tiny_release()
        wrong_sample_id = results[0]["sample_id"]
        results[0]["gt_mask_kind"] = "wrong-mask-kind"
        with self.assertRaisesRegex(
            ValueError,
            f"result {wrong_sample_id} gt_mask_kind mismatch",
        ):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )

    def test_panel_coverage_is_fail_closed(self):
        inputs, panel, source_pairs, results = _tiny_release()
        panel = [
            row
            for row in panel
            if row["sample_id"] != "local_mouse-1"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "same non-zero row count",
        ):
            _summarize(
                inputs,
                panel,
                source_pairs,
                results,
                iterations=1,
            )


if __name__ == "__main__":
    unittest.main()
